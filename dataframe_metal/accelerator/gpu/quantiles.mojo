"""Fixed-pass binary64 order statistics expressed entirely in Mojo.

Eight radix refinements select one rank without allocating a sorted row copy.
Indexed inputs let grouped kernels share the same selection and interpolation.
"""
from max.gpu import global_idx
from std.memory import Pointer, unsafe_stack_allocation
from dataframe_metal.accelerator.rows import (
    ROW_QUANTILE_LINEAR,
    ROW_QUANTILE_LOWER,
    ROW_QUANTILE_HIGHER,
    ROW_QUANTILE_NEAREST,
    ROW_QUANTILE_MIDPOINT,
)
from . import floating as f
from . import quantile_selection as permutation

comptime STATE_WORDS = 8
comptime DESCRIPTOR_WORDS = 5


def supported(op: Int64) -> Bool:
    return op >= ROW_QUANTILE_LINEAR and op <= ROW_QUANTILE_MIDPOINT


def key(word: UInt64) -> UInt64:
    if f.is_nan(word):
        return UInt64.MAX
    if f.absolute(word) == 0:
        return f.SIGN
    return ~word if word >> 63 else word ^ f.SIGN


def select[
    Indexed: Bool
](
    words: Pointer[UInt64, ImmutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    index: Pointer[UInt32, ImmutAnyOrigin],
    base: Int,
    begin: Int,
    end: Int,
    rank: UInt32,
) -> UInt64:
    var prefix = UInt64(0)
    var mask = UInt64(0)
    var wanted = rank
    var bins = unsafe_stack_allocation[256, DType.uint32]()
    for pass_number in range(8):
        for bin in range(256):
            bins[unsafe_offset=bin] = 0
        var shift = UInt64(56 - 8 * pass_number)
        for at in range(begin, end):
            var row = at
            comptime if Indexed:
                row = Int(index[unsafe_offset=at])
            var source = base + row
            if validity[unsafe_offset=source] == 0:
                continue
            var value = key(words[unsafe_offset=source])
            if value & mask == prefix:
                bins[unsafe_offset=Int((value >> shift) & 255)] += 1
        var chosen = 0
        while chosen < 255 and wanted >= bins[unsafe_offset=chosen]:
            wanted -= bins[unsafe_offset=chosen]
            chosen += 1
        prefix |= UInt64(chosen) << shift
        mask |= UInt64(255) << shift
    if prefix == UInt64.MAX:
        return 0x7FF8000000000000
    return prefix ^ f.SIGN if prefix >> 63 else ~prefix


def reduce[
    Indexed: Bool
](
    words: Pointer[UInt64, ImmutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    index: Pointer[UInt32, ImmutAnyOrigin],
    base: Int,
    begin: Int,
    end: Int,
    probability: UInt64,
    op: Int64,
) -> Tuple[UInt64, UInt8]:
    var count = UInt32(0)
    for at in range(begin, end):
        var row = at
        comptime if Indexed:
            row = Int(index[unsafe_offset=at])
        count += UInt32(validity[unsafe_offset=base + row] != 0)
    if count == 0:
        return (0, 0)
    var position = f.mul(probability, f.from_unsigned(UInt64(count - 1)))
    var lower = UInt32(f.to_integer(position, 32, True)[0])
    var fraction = f.sub(position, f.from_unsigned(UInt64(lower)))
    var upper = min(lower + UInt32(f.absolute(fraction) != 0), count - 1)
    var wanted = lower
    if op == ROW_QUANTILE_HIGHER:
        wanted = upper
    elif op == ROW_QUANTILE_NEAREST:
        wanted = min(
            UInt32(
                f.to_integer(f.add(position, 0x3FE0000000000000), 32, True)[0]
            ),
            count - 1,
        )
    var lo = select[Indexed](words, validity, index, base, begin, end, wanted)
    if (
        op == ROW_QUANTILE_LOWER
        or op == ROW_QUANTILE_HIGHER
        or op == ROW_QUANTILE_NEAREST
        or lower == upper
        or f.absolute(fraction) == 0
    ):
        return (lo, 1)
    var hi = select[Indexed](words, validity, index, base, begin, end, upper)
    if op == ROW_QUANTILE_MIDPOINT:
        return (f.div(f.add(lo, hi), 0x4000000000000000), 1)
    return (f.add(lo, f.mul(f.sub(hi, lo), fraction)), 1)


def empty_result(op: Int64, minimum: Int64) -> Tuple[UInt64, UInt8, UInt32]:
    """Every order statistic of an input with no valid values is null."""
    return (0, 0, 0)


def reduce_partials(
    words: Pointer[UInt64, ImmutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    errors: Pointer[UInt32, ImmutAnyOrigin],
    descriptors: Pointer[Int64, ImmutAnyOrigin],
    partials: Pointer[UInt64, MutAnyOrigin],
    outputs: UInt32,
    groups: UInt32,
    rows: UInt32,
    stride: UInt32,
):
    var work = global_idx.x
    if work >= Int(outputs) * Int(groups):
        return
    var column = work // Int(groups)
    if work % Int(groups) != 0:
        return
    var descriptor = column * DESCRIPTOR_WORDS
    var op = descriptors[unsafe_offset=descriptor + 3]
    if not supported(op):
        return
    var base = Int(descriptors[unsafe_offset=descriptor]) * Int(stride)
    var probability = descriptors[unsafe_offset=descriptor + 4].cast[
        DType.uint64
    ]()
    # The unindexed specialization does not read its index argument.
    var result = reduce[False](
        words,
        validity,
        words.unsafe_bitcast[UInt32](),
        base,
        0,
        Int(rows),
        probability,
        op,
    )
    var error = UInt32(0)
    for row in range(Int(rows)):
        error |= errors[unsafe_offset=base + row]
    var at = work * STATE_WORDS
    for field in range(STATE_WORDS):
        partials[unsafe_offset=at + field] = 0
    partials[unsafe_offset=at] = result[0]
    partials[unsafe_offset=at + 1] = UInt64(result[1])
    partials[unsafe_offset=at + 6] = UInt64(error)


def reduce_finish(
    partials: Pointer[UInt64, ImmutAnyOrigin],
    descriptors: Pointer[Int64, ImmutAnyOrigin],
    words: Pointer[UInt64, MutAnyOrigin],
    validity: Pointer[UInt8, MutAnyOrigin],
    errors: Pointer[UInt32, MutAnyOrigin],
    outputs: UInt32,
    groups: UInt32,
    rows: UInt32,
):
    var column = global_idx.x
    if column >= Int(outputs):
        return
    if not supported(descriptors[unsafe_offset=column * DESCRIPTOR_WORDS + 3]):
        return
    var at = column * Int(groups) * STATE_WORDS
    words[unsafe_offset=column] = partials[unsafe_offset=at]
    validity[unsafe_offset=column] = UInt8(partials[unsafe_offset=at + 1])
    errors[unsafe_offset=column] = UInt32(partials[unsafe_offset=at + 6])


def reduce_selected[
    Indexed: Bool
](
    words: Pointer[UInt64, ImmutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    index: Pointer[UInt32, ImmutAnyOrigin],
    scratch: Pointer[UInt32, MutAnyOrigin],
    base: Int,
    begin: Int,
    end: Int,
    scratch_start: Int,
    probability: UInt64,
    op: Int64,
) -> Tuple[UInt64, UInt8]:
    """Same arithmetic as reduce, preserving CPU zero representative swaps."""
    var counts = permutation.initialize[Indexed](
        words, validity, index, scratch, base, begin, end, scratch_start
    )
    var count = UInt32(counts[1])
    if count == 0:
        return (0, 0)
    var position = f.mul(probability, f.from_unsigned(UInt64(count - 1)))
    var lower = UInt32(f.to_integer(position, 32, True)[0])
    var fraction = f.sub(position, f.from_unsigned(UInt64(lower)))
    var upper = min(lower + UInt32(f.absolute(fraction) != 0), count - 1)
    var wanted = lower
    if op == ROW_QUANTILE_HIGHER:
        wanted = upper
    elif op == ROW_QUANTILE_NEAREST:
        wanted = min(
            UInt32(
                f.to_integer(f.add(position, 0x3FE0000000000000), 32, True)[0]
            ),
            count - 1,
        )
    var pair = permutation.selected_pair(
        words, base, scratch, scratch_start, counts[0], Int(wanted), Int(upper)
    )
    if (
        op == ROW_QUANTILE_LOWER
        or op == ROW_QUANTILE_HIGHER
        or op == ROW_QUANTILE_NEAREST
        or lower == upper
        or f.absolute(fraction) == 0
    ):
        return (pair[0], 1)
    if op == ROW_QUANTILE_MIDPOINT:
        return (f.div(f.add(pair[0], pair[1]), 0x4000000000000000), 1)
    return (f.add(pair[0], f.mul(f.sub(pair[1], pair[0]), fraction)), 1)


def correct_zero_partials(
    words: Pointer[UInt64, ImmutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    descriptors: Pointer[Int64, ImmutAnyOrigin],
    partials: Pointer[UInt64, MutAnyOrigin],
    scratch: Pointer[UInt32, MutAnyOrigin],
    outputs: UInt32,
    groups: UInt32,
    rows: UInt32,
    stride: UInt32,
):
    # One thread serializes output descriptors sharing one N-index plane.
    if global_idx.x != 0:
        return
    for column in range(Int(outputs)):
        var descriptor = column * DESCRIPTOR_WORDS
        var op = descriptors[unsafe_offset=descriptor + 3]
        var at = column * Int(groups) * STATE_WORDS
        if (
            not supported(op)
            or partials[unsafe_offset=at + 1] == 0
            or f.absolute(partials[unsafe_offset=at]) != 0
        ):
            continue
        var result = reduce_selected[False](
            words,
            validity,
            words.unsafe_bitcast[UInt32](),
            scratch,
            Int(descriptors[unsafe_offset=descriptor]) * Int(stride),
            0,
            Int(rows),
            0,
            descriptors[unsafe_offset=descriptor + 4].cast[DType.uint64](),
            op,
        )
        partials[unsafe_offset=at] = result[0]
        # Existing count validity and expression errors remain untouched.


def correct_zero_groups(
    words: Pointer[UInt64, ImmutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    output_words: Pointer[UInt64, MutAnyOrigin],
    output_validity: Pointer[UInt8, ImmutAnyOrigin],
    index: Pointer[UInt32, ImmutAnyOrigin],
    starts: Pointer[UInt32, ImmutAnyOrigin],
    code: Pointer[Int64, ImmutAnyOrigin],
    literals: Pointer[UInt64, ImmutAnyOrigin],
    scratch: Pointer[UInt32, MutAnyOrigin],
    start: UInt32,
    nodes: UInt32,
    groups: UInt32,
    rows: UInt32,
    stride: UInt32,
):
    var group = global_idx.x
    if group >= Int(groups):
        return
    var header = Int(start) * 4
    var keys = Int(code[unsafe_offset=header + 1])
    var begin = Int(starts[unsafe_offset=group])
    var end = Int(starts[unsafe_offset=group + 1]) if group + 1 < Int(
        groups
    ) else Int(rows)
    # Different groups own disjoint [begin,end) scratch spans; all descriptors
    # for one group run serially, after group_reduce's independent work.
    for column in range(keys + 1, Int(nodes)):
        var node = Int(start) + column
        var op = code[unsafe_offset=node * 4 + 3]
        if not supported(op):
            continue
        var target = Int(code[unsafe_offset=node * 4 + 2]) * Int(stride) + group
        if (
            output_validity[unsafe_offset=target] == 0
            or f.absolute(output_words[unsafe_offset=target]) != 0
        ):
            continue
        var result = reduce_selected[True](
            words,
            validity,
            index,
            scratch,
            Int(code[unsafe_offset=node * 4 + 1]) * Int(stride),
            begin,
            end,
            begin,
            literals[unsafe_offset=node],
            op,
        )
        output_words[unsafe_offset=target] = result[0]
