"""Stable typed GPU joins translated from native/dfmetal/dfmetal_join.inc.

Columns describe combined input buffers with [word base, row count, dtype].
Keys are [left column, right column] pairs; outputs describe [left column,
right column, prefer right, output slot], with -1 for an absent source.

Sort the build-side indices, match probe rows, scan counts, and synchronize
only the five-word metadata before allocating output. For full joins, scan
coverage differences, mark unmatched build rows in source order, and append
those rows. Gather launches only the limited output cardinality, so a bounded
cross join never materializes its complete fanout. Metadata words are:
[limited output rows, probe output rows, unmatched build rows, full rows, 0].

How codes preserve shared lowering: 0 inner, 1 left, 2 right, 3 full,
4 semi, 5 anti, 6 cross, 7 null-equal left. For a right join probe_side=1
(build left, probe right); all other styles use probe_side=0. Explicit
nulls_equal also works for all keyed styles. All scan launches use 256 lanes.
"""
from max.gpu import global_idx, thread_idx, block_idx
from max.gpu.primitives.block import prefix_sum
from std.atomic import Atomic
from std.memory import Pointer
from .grouping import key_equal
from .ordering import _word_less, _nan
from . import decimal

comptime COLUMN_WORDS = 3
comptime OUTPUT_WORDS = 4
comptime METADATA_WORDS = 5
comptime MISSING = UInt32.MAX


def workspace_bytes(probe: Int, build: Int, how: Int) -> Int:
    """Device scratch bytes, including nonempty allocations for empty arrays.

    Input words/validity/errors and descriptors are charged separately.
    Host metadata readback requires another METADATA_WORDS * 8 bytes.
    """
    var ordered = how != 6
    var full = how == 3
    return (
        8 * max(1, build if ordered else 0)
        + 16 * max(1, probe)
        + 8 * max(1, (probe + 255) // 256)
        + 4 * max(1, build + 1 if full else 0)
        + 8 * max(1, build if full else 0)
        + 8 * max(1, (build + 255) // 256 if full else 0)
        + 8 * max(1, build if full else 0)
        + METADATA_WORDS * 8
    )


def result_bytes(rows: Int, slots: Int) -> Int:
    """GPU result word/validity/error storage, before host and typed results."""
    return 21 * max(1, rows) * slots


@always_inline
def tuple_compare(
    a: UInt32,
    b: UInt32,
    cross: Bool,
    words: Pointer[UInt64, ImmutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    columns: Pointer[Int64, ImmutAnyOrigin],
    keys: Pointer[Int64, ImmutAnyOrigin],
    key_count: UInt32,
    probe_side: UInt32,
    cells: UInt64 = 0,
) -> Int32:
    """Compare build a with build b, or with probe b when cross is true."""
    for key in range(Int(key_count)):
        var build = keys[unsafe_offset=2 * key + (0 if probe_side else 1)]
        var other = keys[
            unsafe_offset=2 * key + (1 if probe_side else 0)
        ] if cross else build
        var ai = Int(columns[unsafe_offset=Int(build) * COLUMN_WORDS]) + Int(a)
        var bi = Int(columns[unsafe_offset=Int(other) * COLUMN_WORDS]) + Int(b)
        var av = validity[unsafe_offset=ai] != 0
        var bv = validity[unsafe_offset=bi] != 0
        if av != bv:
            return 1 if av else -1
        if not av:
            continue
        var aw = words[unsafe_offset=ai]
        var bw = words[unsafe_offset=bi]
        var dtype = columns[unsafe_offset=Int(build) * COLUMN_WORDS + 2]
        if dtype == 12:
            var ah = words[unsafe_offset=Int(cells) + ai]
            var bh = words[unsafe_offset=Int(cells) + bi]
            if aw == bw and ah == bh:
                continue
            return -1 if decimal.signed_less(
                decimal.Wide(aw, ah), decimal.Wide(bw, bh)
            ) else 1
        if key_equal(aw, bw, dtype):
            continue
        var an = _nan(aw, dtype)
        var bn = _nan(bw, dtype)
        if an != bn:
            return 1 if an else -1
        return -1 if _word_less(aw, bw, dtype) else 1
    return 0


@always_inline
def _less(
    a: UInt32,
    b: UInt32,
    words: Pointer[UInt64, ImmutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    columns: Pointer[Int64, ImmutAnyOrigin],
    keys: Pointer[Int64, ImmutAnyOrigin],
    key_count: UInt32,
    probe_side: UInt32,
    cells: UInt64 = 0,
) -> Bool:
    var comparison = tuple_compare(
        a,
        b,
        False,
        words,
        validity,
        columns,
        keys,
        key_count,
        probe_side,
        cells,
    )
    return comparison < 0 or (comparison == 0 and a < b)


def join_init(index: Pointer[UInt32, MutAnyOrigin], rows: UInt32):
    var row = global_idx.x
    if row < Int(rows):
        index[unsafe_offset=row] = UInt32(row)


def join_merge(
    words: Pointer[UInt64, ImmutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    index: Pointer[UInt32, ImmutAnyOrigin],
    target: Pointer[UInt32, MutAnyOrigin],
    columns: Pointer[Int64, ImmutAnyOrigin],
    keys: Pointer[Int64, ImmutAnyOrigin],
    key_count: UInt32,
    probe_side: UInt32,
    build_rows: UInt32,
    run_width: UInt32,
    cells: UInt64 = 0,
):
    var row = global_idx.x
    if row >= Int(build_rows):
        return
    var width = Int(run_width)
    var base = row // (2 * width) * (2 * width)
    var middle = min(base + width, Int(build_rows))
    var end = min(base + 2 * width, Int(build_rows))
    var first = row < middle
    var start = middle if first else base
    var lo = start
    var hi = end if first else middle
    var own = index[unsafe_offset=row]
    while lo < hi:
        var mid = lo + (hi - lo) // 2
        if _less(
            index[unsafe_offset=mid],
            own,
            words,
            validity,
            columns,
            keys,
            key_count,
            probe_side,
            cells,
        ):
            lo = mid + 1
        else:
            hi = mid
    target[
        unsafe_offset=base + row - (base if first else middle) + lo - start
    ] = own


def join_zero(differences: Pointer[Int32, MutAnyOrigin], rows: UInt32):
    var row = global_idx.x
    if row <= Int(rows):
        differences[unsafe_offset=row] = 0


def join_match(
    words: Pointer[UInt64, ImmutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    index: Pointer[UInt32, ImmutAnyOrigin],
    columns: Pointer[Int64, ImmutAnyOrigin],
    keys: Pointer[Int64, ImmutAnyOrigin],
    ranges: Pointer[UInt64, MutAnyOrigin],
    counts: Pointer[UInt64, MutAnyOrigin],
    differences: Pointer[Int32, MutAnyOrigin],
    key_count: UInt32,
    probe_side: UInt32,
    probe_rows: UInt32,
    build_rows: UInt32,
    how: UInt32,
    nulls_equal: UInt32,
    cells: UInt64 = 0,
):
    var row = global_idx.x
    if row >= Int(probe_rows):
        return
    var low = UInt32(0)
    var high = UInt32(0)
    var valid = True
    var match_nulls = nulls_equal != 0 or how == 7
    if how == 6:
        high = build_rows
    else:
        if not match_nulls:
            for key in range(Int(key_count)):
                var column = keys[unsafe_offset=2 * key + Int(probe_side)]
                var at = (
                    Int(columns[unsafe_offset=Int(column) * COLUMN_WORDS]) + row
                )
                valid &= validity[unsafe_offset=at] != 0
        if valid:
            var lo = UInt32(0)
            var hi = build_rows
            while lo < hi:
                var mid = lo + (hi - lo) // 2
                if (
                    tuple_compare(
                        index[unsafe_offset=Int(mid)],
                        UInt32(row),
                        True,
                        words,
                        validity,
                        columns,
                        keys,
                        key_count,
                        probe_side,
                        cells,
                    )
                    < 0
                ):
                    lo = mid + 1
                else:
                    hi = mid
            low = lo
            hi = build_rows
            while lo < hi:
                var mid = lo + (hi - lo) // 2
                if (
                    tuple_compare(
                        index[unsafe_offset=Int(mid)],
                        UInt32(row),
                        True,
                        words,
                        validity,
                        columns,
                        keys,
                        key_count,
                        probe_side,
                        cells,
                    )
                    <= 0
                ):
                    lo = mid + 1
                else:
                    hi = mid
            high = lo
    ranges[unsafe_offset=row] = UInt64(low) | (UInt64(high) << 32)
    var count = UInt64(high - low)
    if how == 4:
        count = UInt64(count != 0)
    elif how == 5:
        count = UInt64(count == 0)
    elif how == 1 or how == 2 or how == 3 or how == 7:
        count = max(count, UInt64(1))
    counts[unsafe_offset=row] = count
    if how == 3 and high > low:
        _ = Atomic[Int32].fetch_add(
            differences.unsafe_offset(Int(low)), Int32(1)
        )
        _ = Atomic[Int32].fetch_add(
            differences.unsafe_offset(Int(high)), Int32(-1)
        )


def join_scan(
    values: Pointer[UInt64, MutAnyOrigin],
    blocks: Pointer[UInt64, MutAnyOrigin],
    differences: Pointer[Int32, ImmutAnyOrigin],
    flags: Pointer[UInt32, ImmutAnyOrigin],
    rows: UInt32,
    mode: UInt32,
):
    """Modes: 0 probe counts, 1 coverage differences, 2 unmatched flags."""
    var row = global_idx.x
    var value = UInt64(0)
    if row < Int(rows):
        if mode == 1:
            value = Int64(differences[unsafe_offset=row]).cast[DType.uint64]()
        elif mode == 2:
            value = UInt64(flags[unsafe_offset=row])
        else:
            value = values[unsafe_offset=row]
    var total = prefix_sum[block_size=256](value)
    if row < Int(rows):
        values[unsafe_offset=row] = total - value
    if thread_idx.x == 255:
        blocks[unsafe_offset=block_idx.x] = total


def join_offsets(
    blocks: Pointer[UInt64, MutAnyOrigin],
    metadata: Pointer[UInt64, MutAnyOrigin],
    count: UInt32,
    mode: UInt32,
    limit: UInt64,
):
    """Launch one thread after join_scan, including for empty inputs."""
    var total = UInt64(0)
    for block in range(Int(count)):
        var value = blocks[unsafe_offset=block]
        blocks[unsafe_offset=block] = total
        total += value
    if mode == 0:
        metadata[unsafe_offset=0] = min(total, limit)
        metadata[unsafe_offset=1] = total
        metadata[unsafe_offset=2] = 0
        metadata[unsafe_offset=3] = total
        metadata[unsafe_offset=4] = 0
    elif mode == 2:
        metadata[unsafe_offset=2] = total
        total += metadata[unsafe_offset=1]
        metadata[unsafe_offset=0] = min(total, limit)
        metadata[unsafe_offset=3] = total


def join_coverage(
    prefix: Pointer[UInt64, ImmutAnyOrigin],
    blocks: Pointer[UInt64, ImmutAnyOrigin],
    differences: Pointer[Int32, ImmutAnyOrigin],
    index: Pointer[UInt32, ImmutAnyOrigin],
    flags: Pointer[UInt32, MutAnyOrigin],
    build_rows: UInt32,
):
    var row = global_idx.x
    if row >= Int(build_rows):
        return
    var covered = (
        prefix[unsafe_offset=row]
        + blocks[unsafe_offset=row // 256]
        + Int64(differences[unsafe_offset=row]).cast[DType.uint64]()
    )
    flags[unsafe_offset=Int(index[unsafe_offset=row])] = UInt32(covered == 0)


def join_unmatched(
    prefix: Pointer[UInt64, ImmutAnyOrigin],
    blocks: Pointer[UInt64, ImmutAnyOrigin],
    flags: Pointer[UInt32, ImmutAnyOrigin],
    index: Pointer[UInt32, MutAnyOrigin],
    build_rows: UInt32,
):
    var row = global_idx.x
    if row < Int(build_rows) and flags[unsafe_offset=row] != 0:
        var target = (
            prefix[unsafe_offset=row] + blocks[unsafe_offset=row // 256]
        )
        index[unsafe_offset=Int(target)] = UInt32(row)


@always_inline
def output_pair(
    row: UInt64,
    sorted_index: Pointer[UInt32, ImmutAnyOrigin],
    ranges: Pointer[UInt64, ImmutAnyOrigin],
    prefix: Pointer[UInt64, ImmutAnyOrigin],
    blocks: Pointer[UInt64, ImmutAnyOrigin],
    unmatched: Pointer[UInt32, ImmutAnyOrigin],
    metadata: Pointer[UInt64, ImmutAnyOrigin],
    probe_rows: UInt32,
    probe_side: UInt32,
    how: UInt32,
) -> Tuple[UInt32, UInt32]:
    if row >= metadata[unsafe_offset=1]:
        return (
            MISSING,
            unmatched[unsafe_offset=Int(row - metadata[unsafe_offset=1])],
        )
    var lo = UInt32(0)
    var hi = probe_rows
    while lo < hi:
        var mid = lo + (hi - lo) // 2
        var start = (
            prefix[unsafe_offset=Int(mid)]
            + blocks[unsafe_offset=Int(mid) // 256]
        )
        if start <= row:
            lo = mid + 1
        else:
            hi = mid
    var probe = lo - 1
    var start = (
        prefix[unsafe_offset=Int(probe)]
        + blocks[unsafe_offset=Int(probe) // 256]
    )
    var pair = ranges[unsafe_offset=Int(probe)]
    var low = UInt32(pair)
    var high = UInt32(pair >> 32)
    var build = MISSING
    if how != 4 and how != 5 and high > low:
        var slot = low + UInt32(row - start)
        build = slot if how == 6 else sorted_index[unsafe_offset=Int(slot)]
    return (build, probe) if probe_side else (probe, build)


def join_pairs(
    sorted_index: Pointer[UInt32, ImmutAnyOrigin],
    ranges: Pointer[UInt64, ImmutAnyOrigin],
    prefix: Pointer[UInt64, ImmutAnyOrigin],
    blocks: Pointer[UInt64, ImmutAnyOrigin],
    unmatched: Pointer[UInt32, ImmutAnyOrigin],
    metadata: Pointer[UInt64, ImmutAnyOrigin],
    left: Pointer[UInt32, MutAnyOrigin],
    right: Pointer[UInt32, MutAnyOrigin],
    probe_rows: UInt32,
    probe_side: UInt32,
    how: UInt32,
):
    """Optional source-pair materialization; MISSING denotes an absent row."""
    var row = global_idx.x
    if UInt64(row) < metadata[unsafe_offset=0]:
        var pair = output_pair(
            UInt64(row),
            sorted_index,
            ranges,
            prefix,
            blocks,
            unmatched,
            metadata,
            probe_rows,
            probe_side,
            how,
        )
        left[unsafe_offset=row] = pair[0]
        right[unsafe_offset=row] = pair[1]


def join_gather(
    words: Pointer[UInt64, ImmutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    errors: Pointer[UInt32, ImmutAnyOrigin],
    columns: Pointer[Int64, ImmutAnyOrigin],
    outputs: Pointer[Int64, ImmutAnyOrigin],
    sorted_index: Pointer[UInt32, ImmutAnyOrigin],
    ranges: Pointer[UInt64, ImmutAnyOrigin],
    prefix: Pointer[UInt64, ImmutAnyOrigin],
    blocks: Pointer[UInt64, ImmutAnyOrigin],
    unmatched: Pointer[UInt32, ImmutAnyOrigin],
    metadata: Pointer[UInt64, ImmutAnyOrigin],
    output_words: Pointer[UInt64, MutAnyOrigin],
    output_validity: Pointer[UInt8, MutAnyOrigin],
    output_errors: Pointer[UInt32, MutAnyOrigin],
    probe_rows: UInt32,
    probe_side: UInt32,
    how: UInt32,
    output_count: UInt32,
    cells: UInt64 = 0,
):
    var row = global_idx.x
    var rows = metadata[unsafe_offset=0]
    if UInt64(row) >= rows:
        return
    var pair = output_pair(
        UInt64(row),
        sorted_index,
        ranges,
        prefix,
        blocks,
        unmatched,
        metadata,
        probe_rows,
        probe_side,
        how,
    )
    for column in range(Int(output_count)):
        var at = column * OUTPUT_WORDS
        var left = outputs[unsafe_offset=at]
        var right = outputs[unsafe_offset=at + 1]
        var prefer_right = outputs[unsafe_offset=at + 2] != 0
        var slot = outputs[unsafe_offset=at + 3]
        var source = Int64(-1)
        var selected = MISSING
        if prefer_right and pair[1] != MISSING:
            source = right
            selected = pair[1]
        elif left >= 0 and pair[0] != MISSING:
            source = left
            selected = pair[0]
        elif right >= 0 and pair[1] != MISSING:
            source = right
            selected = pair[1]
        var target = Int(slot) * Int(max(rows, UInt64(1))) + row
        var word = UInt64(0)
        var high = UInt64(0)
        var valid = UInt8(0)
        var error = UInt32(0)
        if source >= 0:
            var index = Int(
                columns[unsafe_offset=Int(source) * COLUMN_WORDS]
            ) + Int(selected)
            valid = validity[unsafe_offset=index]
            error = errors[unsafe_offset=index]
            if valid:
                word = words[unsafe_offset=index]
                if cells:
                    high = words[unsafe_offset=Int(cells) + index]
        output_words[unsafe_offset=target] = word
        if cells:
            output_words[
                unsafe_offset=Int(output_count) * Int(max(rows, UInt64(1)))
                + target
            ] = high
        output_validity[unsafe_offset=target] = valid
        output_errors[unsafe_offset=target] = error
