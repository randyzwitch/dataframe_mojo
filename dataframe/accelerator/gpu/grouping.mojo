"""Stable fixed-width GPU grouping over shared row-plan descriptors.

Sort row indices by the header's key descriptors with ordering.order_merge,
then mark, scan, materialize starts, and reduce each contiguous group. Key
representatives preserve original encodings; NaNs, signed zeros and nulls
have the same grouping equivalence as the dataframe CPU path.
"""

from max.gpu import global_idx, thread_idx, block_idx
from max.gpu.primitives.block import prefix_sum
from std.memory import Pointer
from dataframe.accelerator.rows import (
    ROW_GROUP_HEADER_LAST,
    ROW_GROUP_HEADER_WINDOW,
)
from dataframe.expr import ARG_MIN, ARG_MAX, COV, CORR, SUM, MEAN
from . import floating as f
from .decimal_descriptors import DECIMAL_SUM, DECIMAL_MEAN
from .reductions import State as BasicState
from .statistics import State as StatisticalState, supported as statistical
from . import quantiles


@always_inline
def key_equal(a: UInt64, b: UInt64, dtype: Int64) -> Bool:
    if dtype == 11:
        return f.equal(a, b) or (f.is_nan(a) and f.is_nan(b))
    if dtype == 1:
        var x = UInt32(a)
        var y = UInt32(b)
        var ax = x & UInt32(0x7FFFFFFF)
        var ay = y & UInt32(0x7FFFFFFF)
        return (
            x == y
            or (ax == 0 and ay == 0)
            or (ax > UInt32(0x7F800000) and ay > UInt32(0x7F800000))
        )
    return a == b


@always_inline
def group_equal(
    a: UInt32,
    b: UInt32,
    words: Pointer[UInt64, ImmutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    code: Pointer[Int64, ImmutAnyOrigin],
    types: Pointer[Int64, ImmutAnyOrigin],
    start: UInt32,
    keys: UInt32,
    stride: UInt32,
    slots: UInt32 = 0,
) -> Bool:
    for key in range(Int(keys)):
        var node = Int(start) + 1 + key
        var base = Int(code[unsafe_offset=node * 4 + 1]) * Int(stride)
        var ai = base + Int(a)
        var bi = base + Int(b)
        var av = validity[unsafe_offset=ai] != 0
        var bv = validity[unsafe_offset=bi] != 0
        if av != bv:
            return False
        if av and types[unsafe_offset=node] == 12:
            var high_base = Int(stride) * Int(slots)
            if (
                words[unsafe_offset=high_base + ai]
                != words[unsafe_offset=high_base + bi]
            ):
                return False
        if av and not key_equal(
            words[unsafe_offset=ai],
            words[unsafe_offset=bi],
            types[unsafe_offset=node],
        ):
            return False
    return True


def group_mark(
    words: Pointer[UInt64, ImmutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    errors: Pointer[UInt32, ImmutAnyOrigin],
    index: Pointer[UInt32, ImmutAnyOrigin],
    positions: Pointer[UInt32, MutAnyOrigin],
    blocks: Pointer[UInt32, MutAnyOrigin],
    code: Pointer[Int64, ImmutAnyOrigin],
    types: Pointer[Int64, ImmutAnyOrigin],
    start: UInt32,
    keys: UInt32,
    rows: UInt32,
    stride: UInt32,
    slots: UInt32 = 0,
):
    # Padding participates in both scans. Launch block_dim=256.
    var row = global_idx.x
    var head = UInt32(0)
    var error = UInt32(0)
    if row < Int(rows):
        var source = index[unsafe_offset=row]
        head = UInt32(
            row == 0
            or not group_equal(
                index[unsafe_offset=row - 1],
                source,
                words,
                validity,
                code,
                types,
                start,
                keys,
                stride,
                slots,
            )
        )
        for key in range(Int(keys)):
            var node = Int(start) + 1 + key
            var slot = Int(code[unsafe_offset=node * 4 + 1])
            error |= errors[unsafe_offset=slot * Int(stride) + Int(source)]
    var rank = prefix_sum[block_size=256](head)
    var failed = prefix_sum[block_size=256](UInt32(error != 0))
    if row < Int(rows):
        positions[unsafe_offset=row] = rank - head
    if thread_idx.x == 255:
        var count = (Int(rows) + 255) // 256
        blocks[unsafe_offset=block_idx.x] = rank
        blocks[unsafe_offset=count + block_idx.x] = failed


def group_offsets(
    blocks: Pointer[UInt32, MutAnyOrigin],
    metadata: Pointer[UInt32, MutAnyOrigin],
    count: UInt32,
):
    """Launch one thread; metadata is [groups, key error, minsize, maxsize]."""
    var groups = UInt32(0)
    var failed = UInt32(0)
    for block in range(Int(count)):
        var heads = blocks[unsafe_offset=block]
        blocks[unsafe_offset=block] = groups
        groups += heads
        failed |= blocks[unsafe_offset=Int(count) + block]
    metadata[unsafe_offset=0] = groups
    metadata[unsafe_offset=1] = failed
    metadata[unsafe_offset=2] = 0
    metadata[unsafe_offset=3] = 0


def group_starts(
    words: Pointer[UInt64, ImmutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    index: Pointer[UInt32, ImmutAnyOrigin],
    positions: Pointer[UInt32, ImmutAnyOrigin],
    blocks: Pointer[UInt32, ImmutAnyOrigin],
    starts: Pointer[UInt32, MutAnyOrigin],
    code: Pointer[Int64, ImmutAnyOrigin],
    types: Pointer[Int64, ImmutAnyOrigin],
    start: UInt32,
    keys: UInt32,
    rows: UInt32,
    stride: UInt32,
    slots: UInt32 = 0,
):
    var row = global_idx.x
    if row >= Int(rows):
        return
    if row == 0 or not group_equal(
        index[unsafe_offset=row - 1],
        index[unsafe_offset=row],
        words,
        validity,
        code,
        types,
        start,
        keys,
        stride,
        slots,
    ):
        var group = (
            positions[unsafe_offset=row] + blocks[unsafe_offset=row // 256]
        )
        starts[unsafe_offset=Int(group)] = UInt32(row)


def _workers(
    parameter: UInt64, rows: UInt32, groups: UInt32, window: Bool
) -> Int:
    var configured = Int(parameter >> 47)
    var workers = max(
        1, min(configured, Int(rows) // (65536 if window else 16384))
    )
    return max(
        1, min(workers, Int(rows) // max(1, 4 * Int(groups)))
    ) if window else workers


def group_window_bounds(
    starts: Pointer[UInt32, ImmutAnyOrigin],
    metadata: Pointer[UInt32, MutAnyOrigin],
    rows: UInt32,
):
    """Launch one thread after group_starts; scan each group exactly once."""
    var groups = Int(metadata[unsafe_offset=0])
    var smallest = rows
    var largest = UInt32(0)
    for group in range(groups):
        var begin = starts[unsafe_offset=group]
        var end = (
            starts[unsafe_offset=group + 1] if group + 1 < groups else rows
        )
        smallest = min(smallest, end - begin)
        largest = max(largest, end - begin)
    metadata[unsafe_offset=2] = smallest
    metadata[unsafe_offset=3] = largest


def _local_window_workers(
    starts: Pointer[UInt32, ImmutAnyOrigin],
    bounds: Pointer[UInt32, ImmutAnyOrigin],
    metadata: UInt64,
    group: Int,
    groups: UInt32,
    rows: UInt32,
) -> Int:
    """Return a proved CPU contiguous-run worker count, or zero.

    Bit32 records that the original OVER inner had no nested OVER. CPU
    layouts may number keys in first-occurrence or hash-bucket order. The
    two cases below establish identical group-local association under any
    such ordering; unproved cases retain the existing source-row protocol.
    """
    var configured = Int(metadata & 0xFFFFFFFF)
    var lanes = min(configured, Int(rows) // 2048)
    if (metadata & (UInt64(1) << 32)) == 0 or lanes <= 1 or groups <= 1:
        return 0
    var target = max(2048, (Int(rows) + 4 * lanes - 1) // (4 * lanes))
    var smallest = Int(bounds[unsafe_offset=2])
    var largest = Int(bounds[unsafe_offset=3])
    if smallest >= target:
        # Every group is a whole CPU run, independently of group ordering.
        var begin = Int(starts[unsafe_offset=group])
        var end = Int(starts[unsafe_offset=group + 1]) if group + 1 < Int(
            groups
        ) else Int(rows)
        return max(1, min(configured, (end - begin) // 65536))
    if Int(rows) - largest >= target and target - 1 + largest < 65536:
        # There is more than one run, and even the largest possible run
        # (below target before adding its final group) stays single-worker.
        return 1
    return 0


def group_reduce(
    words: Pointer[UInt64, ImmutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    errors: Pointer[UInt32, ImmutAnyOrigin],
    output_words: Pointer[UInt64, MutAnyOrigin],
    output_validity: Pointer[UInt8, MutAnyOrigin],
    output_errors: Pointer[UInt32, MutAnyOrigin],
    index: Pointer[UInt32, ImmutAnyOrigin],
    starts: Pointer[UInt32, ImmutAnyOrigin],
    group_metadata: Pointer[UInt32, ImmutAnyOrigin],
    code: Pointer[Int64, ImmutAnyOrigin],
    literals: Pointer[UInt64, ImmutAnyOrigin],
    slot_types: Pointer[Int64, ImmutAnyOrigin],
    start: UInt32,
    nodes: UInt32,
    groups: UInt32,
    rows: UInt32,
    stride: UInt32,
    slots: UInt32 = 0,
):
    var work = global_idx.x
    if work >= Int(groups) * Int(nodes):
        return
    var group = work // Int(nodes)
    var column = work % Int(nodes)
    var header = Int(start) * 4
    var keys = Int(code[unsafe_offset=header + 1])
    var begin = Int(starts[unsafe_offset=group])
    var end = Int(starts[unsafe_offset=group + 1]) if group + 1 < Int(
        groups
    ) else Int(rows)
    var last = code[unsafe_offset=header] == ROW_GROUP_HEADER_LAST
    var representative = Int(index[unsafe_offset=end - 1 if last else begin])
    if column == 0:
        var target = Int(code[unsafe_offset=header + 3]) * Int(stride) + group
        output_words[unsafe_offset=target] = UInt64(representative)
        output_validity[unsafe_offset=target] = 1
        output_errors[unsafe_offset=target] = 0
        return
    var node = Int(start) + column
    var source_slot = Int(code[unsafe_offset=node * 4 + 1])
    if column <= keys:
        var source = source_slot * Int(stride) + representative
        var target = source_slot * Int(stride) + group
        output_words[unsafe_offset=target] = words[unsafe_offset=source]
        if slots:
            output_words[
                unsafe_offset=Int(stride) * Int(slots) + target
            ] = words[unsafe_offset=Int(stride) * Int(slots) + source]
        output_validity[unsafe_offset=target] = validity[unsafe_offset=source]
        output_errors[unsafe_offset=target] = errors[unsafe_offset=source]
        return
    var target_slot = Int(code[unsafe_offset=node * 4 + 2])
    var op = code[unsafe_offset=node * 4 + 3]
    var dtype = slot_types[unsafe_offset=source_slot]
    var target_dtype = slot_types[unsafe_offset=target_slot]
    var parameter = literals[unsafe_offset=node]
    var result: Tuple[UInt64, UInt8, UInt32]
    var result_high = UInt64(0)
    var local_workers = 0
    var window_metadata = literals[unsafe_offset=Int(start)]
    if code[unsafe_offset=header] == ROW_GROUP_HEADER_WINDOW and (
        statistical(op)
        or op == DECIMAL_SUM
        or op == DECIMAL_MEAN
        or ((op == SUM or op == MEAN) and (dtype == 1 or dtype == 11))
    ):
        local_workers = _local_window_workers(
            starts, group_metadata, window_metadata, group, groups, rows
        )
    if quantiles.supported(op):
        var quantile = quantiles.reduce[True](
            words,
            validity,
            index,
            source_slot * Int(stride),
            begin,
            end,
            parameter,
            op,
        )
        var error = UInt32(0)
        for at in range(begin, end):
            error |= errors[
                unsafe_offset=source_slot * Int(stride)
                + Int(index[unsafe_offset=at])
            ]
        result = (quantile[0], quantile[1], error)
    elif (
        code[unsafe_offset=header] == ROW_GROUP_HEADER_WINDOW
        and (op == SUM or op == MEAN)
        and (dtype == 1 or dtype == 11)
    ):
        # CPU OVER sums preserve serial source order within each contiguous
        # worker, then merge every worker in order (including empty partials).
        var workers = max(
            1,
            min(
                Int(window_metadata & 0xFFFFFFFF),
                min(Int(rows) // 65536, Int(rows) // max(1, 4 * Int(groups))),
            ),
        )
        workers = local_workers if local_workers else workers
        var span = end - begin if local_workers else Int(rows)
        var step = ((span + workers - 1) // workers + 8191) // 8192 * 8192
        var state = BasicState()
        var at = begin
        for worker in range(workers):
            var partial = BasicState()
            var stop = min(span, (worker + 1) * step)
            while at < end:
                var row = Int(index[unsafe_offset=at])
                if (at - begin if local_workers else row) >= stop:
                    break
                at += 1
                var source = source_slot * Int(stride) + row
                partial.add(
                    words[unsafe_offset=source],
                    validity[unsafe_offset=source] != 0,
                    errors[unsafe_offset=source],
                    UInt64(row),
                    op,
                    dtype,
                )
            state.word = partial.word if worker == 0 else f.add(
                state.word, partial.word
            )
            state.count += partial.count
            state.error |= partial.error
        result = state.finish(
            op,
            dtype,
            target_dtype,
            parameter.cast[DType.int64](),
            UInt64(end - begin),
        )
    elif statistical(op):
        var state = StatisticalState()
        var window = code[unsafe_offset=header] == ROW_GROUP_HEADER_WINDOW
        var workers = _workers(parameter, rows, groups, window)
        workers = local_workers if local_workers else workers
        var span = end - begin if local_workers else Int(rows)
        var alignment = 8192 if window else 64
        var step = (
            ((span + workers - 1) // workers + alignment - 1)
            // alignment
            * alignment
        )
        var at = begin
        for worker in range(workers):
            var partial = StatisticalState()
            var stop = min(span, (worker + 1) * step)
            while at < end:
                var row = Int(index[unsafe_offset=at])
                if (at - begin if local_workers else row) >= stop:
                    break
                at += 1
                var source = source_slot * Int(stride) + row
                var valid = validity[unsafe_offset=source] != 0
                var error = errors[unsafe_offset=source]
                var y = UInt64(0)
                if op == COV or op == CORR:
                    var right = (
                        Int((parameter >> 31) & 65535) * Int(stride) + row
                    )
                    valid &= validity[unsafe_offset=right] != 0
                    error |= errors[unsafe_offset=right]
                    y = words[unsafe_offset=right]
                partial.add(words[unsafe_offset=source], y, valid, error, op)
            state.merge(partial, op)
        result = state.finish(op, parameter.cast[DType.int64]())
    else:
        var state = BasicState()
        var decimal_sum = op == DECIMAL_SUM or op == DECIMAL_MEAN
        var workers = max(
            1,
            min(
                Int((parameter >> 44) & 65535),
                min(Int(rows) // 65536, Int(rows) // max(1, 4 * Int(groups))),
            ),
        ) if decimal_sum else 1
        workers = local_workers if local_workers else workers
        var span = end - begin if local_workers else Int(rows)
        var step = ((span + workers - 1) // workers + 8191) // 8192 * 8192
        var at = begin
        for worker in range(workers):
            var partial = BasicState()
            var stop = min(span, (worker + 1) * step)
            while at < end:
                var row = Int(index[unsafe_offset=at])
                if (at - begin if local_workers else row) >= stop:
                    break
                var position = (
                    at - begin if op == ARG_MIN or op == ARG_MAX else row
                )
                at += 1
                var source = source_slot * Int(stride) + row
                partial.add(
                    words[unsafe_offset=source],
                    validity[unsafe_offset=source] != 0,
                    errors[unsafe_offset=source],
                    UInt64(position),
                    op,
                    dtype,
                    words[
                        unsafe_offset=Int(stride) * Int(slots) + source
                    ] if dtype
                    == 12 else UInt64(0),
                    parameter.cast[DType.int64](),
                )
            state.merge(partial, op, dtype, parameter.cast[DType.int64]())
        result = state.finish(
            op,
            dtype,
            target_dtype,
            parameter.cast[DType.int64](),
            UInt64(end - begin),
        )
        result_high = state.result_high(op, target_dtype, result[1])
    var target = target_slot * Int(stride) + group
    output_words[unsafe_offset=target] = result[0]
    if slots:
        output_words[
            unsafe_offset=Int(stride) * Int(slots) + target
        ] = result_high
    output_validity[unsafe_offset=target] = result[1]
    output_errors[unsafe_offset=target] = result[2]
