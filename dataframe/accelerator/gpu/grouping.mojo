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
from dataframe.partition import _mix, _combine
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
        if av and (
            types[unsafe_offset=node] == 12
            or code[unsafe_offset=node * 4 + 3] == 2
        ):
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
    """Launch one thread; metadata is [groups, key error, local mode, runs]."""
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


@always_inline
def _partition_key(
    words: Pointer[UInt64, ImmutAnyOrigin],
    key_values: Pointer[UInt64, ImmutAnyOrigin],
    source: Int,
    dtype: Int64,
    descriptor: UInt64,
    cells: Int,
) -> UInt64:
    """Return the CPU raw key, before its per-column hash mixing.

    Bit63 denotes dictionary lookup, bit62 an original categorical UInt32
    code rather than a string hash. The remaining bits locate that key's
    staged-code-to-raw-key table. Null handling belongs to the caller.
    """
    var word = words[unsafe_offset=source]
    if descriptor & (UInt64(1) << 63):
        if descriptor & (UInt64(1) << 61):
            return words[unsafe_offset=cells + source] - 1
        return key_values[
            unsafe_offset=Int(descriptor & ((UInt64(1) << 62) - 1)) + Int(word)
        ]
    if dtype == 12:
        return word ^ words[unsafe_offset=cells + source]
    if dtype == 4:
        return UInt64(1) if word else UInt64(2)
    if dtype == 1 or dtype == 11:
        word = f.from_float32(UInt32(word)) if dtype == 1 else word
        if f.is_nan(word):
            return UInt64(0x7FF8000000000000)
        return UInt64(0) if (word & UInt64(0x7FFFFFFFFFFFFFFF)) == 0 else word
    return word


def group_window_prepare(
    words: Pointer[UInt64, MutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    index: Pointer[UInt32, ImmutAnyOrigin],
    starts: Pointer[UInt32, ImmutAnyOrigin],
    order: Pointer[UInt32, MutAnyOrigin],
    code: Pointer[Int64, ImmutAnyOrigin],
    literals: Pointer[UInt64, ImmutAnyOrigin],
    types: Pointer[Int64, ImmutAnyOrigin],
    key_values: Pointer[UInt64, ImmutAnyOrigin],
    start: UInt32,
    groups: UInt32,
    rows: UInt32,
    stride: UInt32,
    slots: UInt32,
):
    """One thread, linear in groups*keys; build exact CPU group-order keys.

    Only private source-matrix slots are modified. The ordinary lexical
    row index and group starts remain intact for grouping and scattering.
    """
    if global_idx.x != 0:
        return
    var header = Int(start) * 4
    var keys = Int(code[unsafe_offset=header + 1])
    var private_base = (Int(code[unsafe_offset=header + 2]) - 1) * Int(stride)
    var configured = Int(literals[unsafe_offset=Int(start)] & 0xFFFFFFFF)
    var lanes = min(configured, Int(rows) // 2048)
    var buckets = 1
    var shift = 64
    while buckets < 2 * lanes and buckets < 256:
        buckets *= 2
        shift -= 1
    var dense = False
    if keys == 1:
        var node = Int(start) + 1
        var dtype = types[unsafe_offset=node]
        var descriptor = literals[unsafe_offset=node]
        var lookup = (descriptor & (UInt64(1) << 63)) != 0
        dense = (descriptor & (UInt64(1) << 62)) != 0 if lookup else (
            dtype == 2 or dtype == 3 or (dtype >= 5 and dtype <= 10)
        )
        if dense:
            var base = Int(code[unsafe_offset=node * 4 + 1]) * Int(stride)
            var signed = not lookup and (
                dtype == 2 or dtype == 3 or dtype == 5 or dtype == 6
            )
            var first = True
            var low = UInt64(0)
            var high = UInt64(0)
            for group in range(Int(groups)):
                var source = base + Int(
                    index[unsafe_offset=Int(starts[unsafe_offset=group])]
                )
                if validity[unsafe_offset=source] == 0:
                    continue
                var value = _partition_key(
                    words,
                    key_values,
                    source,
                    dtype,
                    descriptor,
                    Int(stride) * Int(slots),
                )
                if first:
                    low = value
                    high = value
                    first = False
                else:
                    if (
                        value.cast[DType.int64]() < low.cast[DType.int64]()
                    ) if signed else (value < low):
                        low = value
                    if (
                        value.cast[DType.int64]() > high.cast[DType.int64]()
                    ) if signed else (value > high):
                        high = value
            var small = not lookup and (
                dtype == 5 or dtype == 6 or dtype == 7 or dtype == 8
            )
            var limit = min(
                max(65536 if small else 4096, min(Int(rows), 65536)),
                max(4096, 4 * Int(rows)),
            )
            dense = high - low < UInt64(limit)
    for group in range(Int(groups)):
        var representative = Int(
            index[unsafe_offset=Int(starts[unsafe_offset=group])]
        )
        var hashed = UInt64(0)
        if not dense:
            for key in range(keys):
                var node = Int(start) + 1 + key
                var source = (
                    Int(code[unsafe_offset=node * 4 + 1]) * Int(stride)
                    + representative
                )
                var raw = UInt64(0x9E3779B97F4A7C15)
                if validity[unsafe_offset=source] != 0:
                    raw = _partition_key(
                        words,
                        key_values,
                        source,
                        types[unsafe_offset=node],
                        literals[unsafe_offset=node],
                        Int(stride) * Int(slots),
                    )
                hashed = _mix(raw) if key == 0 else _combine(hashed, raw)
        words[unsafe_offset=private_base + group] = UInt64(
            0
        ) if dense else hashed >> UInt64(shift)
        order[unsafe_offset=group] = UInt32(group)


@always_inline
def _window_group_less(
    a: UInt32,
    b: UInt32,
    words: Pointer[UInt64, ImmutAnyOrigin],
    index: Pointer[UInt32, ImmutAnyOrigin],
    starts: Pointer[UInt32, ImmutAnyOrigin],
    base: Int,
) -> Bool:
    var left = words[unsafe_offset=base + Int(a)]
    var right = words[unsafe_offset=base + Int(b)]
    return (
        left
        < right if left
        != right else (
            index[unsafe_offset=Int(starts[unsafe_offset=Int(a)])]
            < index[unsafe_offset=Int(starts[unsafe_offset=Int(b)])]
        )
    )


def group_window_merge(
    words: Pointer[UInt64, ImmutAnyOrigin],
    index: Pointer[UInt32, ImmutAnyOrigin],
    starts: Pointer[UInt32, ImmutAnyOrigin],
    order: Pointer[UInt32, ImmutAnyOrigin],
    target: Pointer[UInt32, MutAnyOrigin],
    start_slot: UInt32,
    groups: UInt32,
    stride: UInt32,
    run_width: UInt32,
):
    """Stable parallel merge of CPU group IDs in the existing index scratch."""
    var row = global_idx.x
    if row >= Int(groups):
        return
    var width = Int(run_width)
    var base = row // (2 * width) * (2 * width)
    var middle = min(base + width, Int(groups))
    var end = min(base + 2 * width, Int(groups))
    var left = row < middle
    var first = middle if left else base
    var lo = first
    var hi = end if left else middle
    var own = order[unsafe_offset=row]
    while lo < hi:
        var mid = lo + (hi - lo) // 2
        if _window_group_less(
            order[unsafe_offset=mid],
            own,
            words,
            index,
            starts,
            (Int(start_slot) - 1) * Int(stride),
        ):
            lo = mid + 1
        else:
            hi = mid
    target[
        unsafe_offset=base + row - (base if left else middle) + lo - first
    ] = own


def group_window_schedule(
    words: Pointer[UInt64, MutAnyOrigin],
    starts: Pointer[UInt32, ImmutAnyOrigin],
    order: Pointer[UInt32, ImmutAnyOrigin],
    metadata: Pointer[UInt32, MutAnyOrigin],
    start_slot: UInt32,
    groups: UInt32,
    rows: UInt32,
    stride: UInt32,
    configured: UInt32,
):
    """One thread; reproduce CPU whole-group greedy runs and worker bounds.

    metadata[2] enables run-local positions only for more than one run:
    a single greedy run makes CPU _over_contiguous return None. metadata[3]
    retains the run count. No source values or result slots are modified.
    """
    if global_idx.x != 0:
        return
    var lanes = min(Int(configured), Int(rows) // 2048)
    if lanes <= 1 or groups <= 1:
        metadata[unsafe_offset=2] = 0
        metadata[unsafe_offset=3] = 0
        return
    var target = max(2048, (Int(rows) + 4 * lanes - 1) // (4 * lanes))
    var begin = 0
    var runs = UInt32(0)
    while begin < Int(groups):
        var end = begin
        var count = 0
        while end < Int(groups):
            if count >= target:
                break
            var group = Int(order[unsafe_offset=end])
            var first = Int(starts[unsafe_offset=group])
            var last = Int(starts[unsafe_offset=group + 1]) if group + 1 < Int(
                groups
            ) else Int(rows)
            count += last - first
            end += 1
        var workers = max(
            1,
            min(
                Int(configured),
                min(count // 65536, count // max(1, 4 * (end - begin))),
            ),
        )
        var step = ((count + workers - 1) // workers + 8191) // 8192 * 8192
        var offset = 0
        for at in range(begin, end):
            var group = Int(order[unsafe_offset=at])
            words[
                unsafe_offset=(Int(start_slot) - 1) * Int(stride) + group
            ] = UInt64(workers) | (UInt64(step) << 32)
            words[unsafe_offset=Int(start_slot) * Int(stride) + group] = UInt64(
                offset
            )
            var first = Int(starts[unsafe_offset=group])
            var last = Int(starts[unsafe_offset=group + 1]) if group + 1 < Int(
                groups
            ) else Int(rows)
            offset += last - first
        runs += 1
        begin = end
    metadata[unsafe_offset=2] = UInt32(runs > 1)
    metadata[unsafe_offset=3] = runs


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
    unordered: UInt32 = 0,
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
    if unordered and column <= keys:
        for at in range(begin + 1, end):
            var source = Int(index[unsafe_offset=at])
            representative = max(representative, source) if last else min(
                representative, source
            )
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
        if (window_metadata & (UInt64(1) << 32)) != 0 and group_metadata[
            unsafe_offset=2
        ] != 0:
            var start_slot = Int(code[unsafe_offset=header + 2])
            local_workers = Int(
                words[unsafe_offset=(start_slot - 1) * Int(stride) + group]
                & 0xFFFFFFFF
            )
    var local_step = 0
    var local_offset = 0
    if local_workers:
        var start_slot = Int(code[unsafe_offset=header + 2])
        local_step = Int(
            words[unsafe_offset=(start_slot - 1) * Int(stride) + group] >> 32
        )
        local_offset = Int(
            words[unsafe_offset=start_slot * Int(stride) + group]
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
        var span = Int(rows)
        var step = ((span + workers - 1) // workers + 8191) // 8192 * 8192
        var state = BasicState()
        var at = begin
        for worker in range(workers):
            var partial = BasicState()
            var stop = (worker + 1) * local_step if local_workers else min(
                span, (worker + 1) * step
            )
            while at < end:
                var row = Int(index[unsafe_offset=at])
                if (
                    local_offset + at - begin if local_workers else row
                ) >= stop:
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
        var span = Int(rows)
        var alignment = 8192 if window else 64
        var step = (
            ((span + workers - 1) // workers + alignment - 1)
            // alignment
            * alignment
        )
        var at = begin
        for worker in range(workers):
            var partial = StatisticalState()
            var stop = (worker + 1) * local_step if local_workers else min(
                span, (worker + 1) * step
            )
            while at < end:
                var row = Int(index[unsafe_offset=at])
                if (
                    local_offset + at - begin if local_workers else row
                ) >= stop:
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
        var span = Int(rows)
        var step = ((span + workers - 1) // workers + 8191) // 8192 * 8192
        var at = begin
        for worker in range(workers):
            var partial = BasicState()
            var stop = (worker + 1) * local_step if local_workers else min(
                span, (worker + 1) * step
            )
            while at < end:
                var row = Int(index[unsafe_offset=at])
                if (
                    local_offset + at - begin if local_workers else row
                ) >= stop:
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
