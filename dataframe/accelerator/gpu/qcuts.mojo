"""Isolated query-integration QCUT kernels with CPU-proven separate rounding.

Input is an already observed immutable Float64 resident field. The caller
supplies ordered dictionary codes for the ORIGINAL custom labels. Only an
N-byte non-NaN validity mask is copied; source value words remain shared.
"""
from max.gpu import global_idx
from std.memory import Pointer
from dataframe.accelerator.gpu import floating as f
from dataframe.accelerator.gpu.quantiles import (
    select as select_rank,
    reduce_selected,
)
from dataframe.accelerator.rows import ROW_QUANTILE_LINEAR

comptime NO_FAILURE = UInt32(0)
comptime INVALID_EDGES = UInt32(1)
comptime INVALID_PROBABILITY = UInt32(2)
comptime UNSUPPORTED_NAN_PROBABILITY = UInt32(3)
comptime RAW_EMPTY = UInt32(0)
comptime RAW_VALID = UInt32(1)
comptime RAW_OUT_OF_RANGE = UInt32(2)
comptime RAW_NAN_PROBABILITY = UInt32(3)


def failure_message(code: UInt32) -> String:
    if code == INVALID_EDGES:
        return "breaks must be unique and strictly increasing"
    if code == INVALID_PROBABILITY:
        return "qcut quantiles must be between 0 and 1"
    if code == UNSUPPORTED_NAN_PROBABILITY:
        return "Metal unsupported [qcut]: NaN probability conversion requires CPU proof"
    return String()


def empty_controls(probabilities: List[UInt64]) -> Tuple[Int, UInt32]:
    """Empty samples produce no edges, but still validate q in original order."""
    for value in probabilities:
        if f.is_nan(value):
            return (0, UNSUPPORTED_NAN_PROBABILITY)
        if f.less(value, 0) or f.less(0x3FF0000000000000, value):
            return (0, INVALID_PROBABILITY)
    return (0, NO_FAILURE)


def sample_mask(
    words: Pointer[UInt64, ImmutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    mask: Pointer[UInt8, MutAnyOrigin],
    base: UInt64,
    rows: UInt32,
):
    var row = global_idx.x
    if row >= Int(rows):
        return
    var at = Int(base) + row
    mask[unsafe_offset=row] = UInt8(
        validity[unsafe_offset=at] != 0
        and not f.is_nan(words[unsafe_offset=at])
    )


def linear_edge(
    words: Pointer[UInt64, ImmutAnyOrigin],
    mask: Pointer[UInt8, ImmutAnyOrigin],
    rows: Int,
    probability: UInt64,
) -> Tuple[UInt64, UInt8]:
    """CPU linear interpolation association; mask excludes all input NaNs.

    Actual M1 dev2026100905 CPU diagnostics prove separate rounding for four
    independent one/two-ULP witnesses. The zero correction phase separately
    preserves quickselect's representative permutation.
    """
    var count = UInt32(0)
    for row in range(rows):
        count += UInt32(mask[unsafe_offset=row] != 0)
    if count == 0:
        return (0, 0)
    var position = f.mul(probability, f.from_unsigned(UInt64(count - 1)))
    var lower = UInt32(f.to_integer(position, 32, True)[0])
    var fraction = f.sub(position, f.from_unsigned(UInt64(lower)))
    var upper = min(lower + UInt32(f.absolute(fraction) != 0), count - 1)
    var index = words.unsafe_bitcast[UInt32]()
    var lo = select_rank[False](words, mask, index, 0, 0, rows, lower)
    if lower == upper or f.absolute(fraction) == 0:
        return (lo, 1)
    var hi = select_rank[False](words, mask, index, 0, 0, rows, upper)
    return (f.add(lo, f.mul(f.sub(hi, lo), fraction)), 1)


def raw_edges(
    words: Pointer[UInt64, ImmutAnyOrigin],
    mask: Pointer[UInt8, ImmutAnyOrigin],
    probabilities: Pointer[UInt64, ImmutAnyOrigin],
    edges: Pointer[UInt64, MutAnyOrigin],
    status: Pointer[UInt32, MutAnyOrigin],
    base: UInt64,
    rows: UInt32,
    count: UInt32,
):
    var q = global_idx.x
    if q >= Int(count):
        return
    var probability = probabilities[unsafe_offset=q]
    edges[unsafe_offset=q] = 0
    if f.is_nan(probability):
        status[unsafe_offset=q] = RAW_NAN_PROBABILITY
        return
    if f.less(probability, 0) or f.less(0x3FF0000000000000, probability):
        status[unsafe_offset=q] = RAW_OUT_OF_RANGE
        return
    # The pointer shift lets the existing order statistic consume the compact
    # N-byte mask at base0 while reading the original immutable value plane.
    var shifted = words.unsafe_offset(Int(base))
    var result = linear_edge(shifted, mask, Int(rows), probability)
    edges[unsafe_offset=q] = result[0]
    status[unsafe_offset=q] = UInt32(result[1])


def correct_zero_edges(
    words: Pointer[UInt64, ImmutAnyOrigin],
    mask: Pointer[UInt8, ImmutAnyOrigin],
    probabilities: Pointer[UInt64, ImmutAnyOrigin],
    edges: Pointer[UInt64, MutAnyOrigin],
    status: Pointer[UInt32, ImmutAnyOrigin],
    scratch: Pointer[UInt32, MutAnyOrigin],
    base: UInt64,
    rows: UInt32,
    count: UInt32,
):
    # One source-index plane is reused serially by zero-valued controls.
    if global_idx.x != 0:
        return
    var shifted = words.unsafe_offset(Int(base))
    for q in range(Int(count)):
        if (
            status[unsafe_offset=q] != RAW_VALID
            or f.absolute(edges[unsafe_offset=q]) != 0
        ):
            continue
        var result = reduce_selected[False](
            shifted,
            mask,
            shifted.unsafe_bitcast[UInt32](),
            scratch,
            0,
            0,
            Int(rows),
            0,
            probabilities[unsafe_offset=q],
            ROW_QUANTILE_LINEAR,
        )
        edges[unsafe_offset=q] = result[0]


def compress_edges(
    raw: Pointer[UInt64, ImmutAnyOrigin],
    status: Pointer[UInt32, ImmutAnyOrigin],
    edges: Pointer[UInt64, MutAnyOrigin],
    label_indices: Pointer[UInt32, MutAnyOrigin],
    control: Pointer[UInt32, MutAnyOrigin],
    count: UInt32,
    allow_duplicates: UInt32,
):
    """Single-thread CPU q-loop order, including first-failure precedence."""
    if global_idx.x != 0:
        return
    for index in range(Int(count)):
        edges[unsafe_offset=index] = 0
    var used = 0
    var failure = NO_FAILURE
    label_indices[unsafe_offset=0] = 0
    for q in range(Int(count)):
        var flag = status[unsafe_offset=q]
        if flag == RAW_OUT_OF_RANGE:
            failure = INVALID_PROBABILITY
            break
        if flag == RAW_NAN_PROBABILITY:
            failure = UNSUPPORTED_NAN_PROBABILITY
            break
        if flag == RAW_EMPTY:
            continue
        var value = raw[unsafe_offset=q]
        if used > 0 and not f.less(edges[unsafe_offset=used - 1], value):
            if allow_duplicates == 0 or not f.equal(
                value, edges[unsafe_offset=used - 1]
            ):
                failure = INVALID_EDGES
                break
            label_indices[unsafe_offset=used] = UInt32(q + 1)
        else:
            edges[unsafe_offset=used] = value
            used += 1
            label_indices[unsafe_offset=used] = UInt32(q + 1)
    control[unsafe_offset=0] = UInt32(used)
    control[unsafe_offset=1] = failure


def classify(
    words: Pointer[UInt64, ImmutAnyOrigin],
    errors: Pointer[UInt32, ImmutAnyOrigin],
    mask: Pointer[UInt8, ImmutAnyOrigin],
    edges: Pointer[UInt64, ImmutAnyOrigin],
    label_indices: Pointer[UInt32, ImmutAnyOrigin],
    label_codes: Pointer[UInt32, ImmutAnyOrigin],
    output_words: Pointer[UInt64, MutAnyOrigin],
    output_validity: Pointer[UInt8, MutAnyOrigin],
    output_errors: Pointer[UInt32, MutAnyOrigin],
    base: UInt64,
    rows: UInt32,
    edge_count: UInt32,
    left_closed: UInt32,
    include_breaks: UInt32,
    capacity: UInt32,
    slots: UInt32,
    category_slot: UInt32,
    breakpoint_slot: UInt32,
):
    """Run only after the host observed control failure==0 (eight bytes)."""
    var row = global_idx.x
    if row >= Int(rows):
        return
    var source = Int(base) + row
    var present = mask[unsafe_offset=row]
    var value = words[unsafe_offset=source]
    var bin = 0
    if present:
        while bin < Int(edge_count):
            var edge = edges[unsafe_offset=bin]
            var cross = f.less(edge, value)
            if left_closed:
                cross = cross or f.equal(edge, value)
            if not cross:
                break
            bin += 1
    var category = Int(capacity) * Int(category_slot) + row
    output_words[unsafe_offset=category] = UInt64(
        label_codes[unsafe_offset=Int(label_indices[unsafe_offset=bin])]
    ) if present else UInt64(0)
    output_words[unsafe_offset=Int(capacity) * Int(slots) + category] = 0
    output_validity[unsafe_offset=category] = present
    output_errors[unsafe_offset=category] = errors[unsafe_offset=source]
    if include_breaks:
        var upper = edges[unsafe_offset=bin] if bin < Int(edge_count) else f.INF
        var breakpoint = Int(capacity) * Int(breakpoint_slot) + row
        output_words[unsafe_offset=breakpoint] = upper if present else UInt64(0)
        output_words[unsafe_offset=Int(capacity) * Int(slots) + breakpoint] = 0
        output_validity[unsafe_offset=breakpoint] = present
        output_errors[unsafe_offset=breakpoint] = errors[unsafe_offset=source]


def default_label_indices(labels: Pointer[UInt32, MutAnyOrigin], count: UInt32):
    var index = global_idx.x
    if index <= Int(count):
        labels[unsafe_offset=index] = UInt32(index)
