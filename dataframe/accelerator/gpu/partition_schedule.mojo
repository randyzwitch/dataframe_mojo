"""Exact CPU OVER run and fallback partition ordering metadata.

This is independent of GroupBy's heavy/light phase scheduler. All source keys
remain on the device. The same typed/raw-domain grouping key helpers determine
both keyed_layout bucket order and fallback _partition_ids dispatch.
"""
from max.gpu import global_idx
from std.memory import Pointer
from dataframe.partition import _mix, _combine
from .grouping import _partition_key


def _dense(
    words: Pointer[UInt64, ImmutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    index: Pointer[UInt32, ImmutAnyOrigin],
    starts: Pointer[UInt32, ImmutAnyOrigin],
    code: Pointer[Int64, ImmutAnyOrigin],
    literals: Pointer[UInt64, ImmutAnyOrigin],
    types: Pointer[Int64, ImmutAnyOrigin],
    key_values: Pointer[UInt64, ImmutAnyOrigin],
    start: UInt32,
    groups: UInt32,
    rows: UInt32,
    stride: UInt32,
    slots: UInt32,
) -> Bool:
    var keys = Int(code[unsafe_offset=Int(start) * 4 + 1])
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
    return dense


def dense_kind(
    words: Pointer[UInt64, ImmutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    index: Pointer[UInt32, ImmutAnyOrigin],
    starts: Pointer[UInt32, ImmutAnyOrigin],
    code: Pointer[Int64, ImmutAnyOrigin],
    literals: Pointer[UInt64, ImmutAnyOrigin],
    types: Pointer[Int64, ImmutAnyOrigin],
    key_values: Pointer[UInt64, ImmutAnyOrigin],
    control: Pointer[UInt32, MutAnyOrigin],
    start: UInt32,
    groups: UInt32,
    rows: UInt32,
    stride: UInt32,
    slots: UInt32,
):
    if global_idx.x == 0:
        control[unsafe_offset=0] = UInt32(
            _dense(
                words,
                validity,
                index,
                starts,
                code,
                literals,
                types,
                key_values,
                start,
                groups,
                rows,
                stride,
                slots,
            )
        )


def group_hashes(
    words: Pointer[UInt64, ImmutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    index: Pointer[UInt32, ImmutAnyOrigin],
    starts: Pointer[UInt32, ImmutAnyOrigin],
    code: Pointer[Int64, ImmutAnyOrigin],
    literals: Pointer[UInt64, ImmutAnyOrigin],
    types: Pointer[Int64, ImmutAnyOrigin],
    key_values: Pointer[UInt64, ImmutAnyOrigin],
    hashes: Pointer[UInt64, MutAnyOrigin],
    contiguous_order: Pointer[UInt32, MutAnyOrigin],
    fallback_order: Pointer[UInt32, MutAnyOrigin],
    start: UInt32,
    groups: UInt32,
    stride: UInt32,
    slots: UInt32,
):
    var group = global_idx.x
    if group >= Int(groups):
        return
    var representative = Int(
        index[unsafe_offset=Int(starts[unsafe_offset=group])]
    )
    var keys = Int(code[unsafe_offset=Int(start) * 4 + 1])
    var hash = UInt64(0)
    for key in range(keys):
        var node = Int(start) + 1 + key
        var at = (
            Int(code[unsafe_offset=node * 4 + 1]) * Int(stride) + representative
        )
        var raw = UInt64(0x9E3779B97F4A7C15)
        if validity[unsafe_offset=at]:
            raw = _partition_key(
                words,
                key_values,
                at,
                types[unsafe_offset=node],
                literals[unsafe_offset=node],
                Int(stride) * Int(slots),
            )
        hash = _mix(raw) if key == 0 else _combine(hash, raw)
    hashes[unsafe_offset=group] = hash
    contiguous_order[unsafe_offset=group] = UInt32(group)
    fallback_order[unsafe_offset=group] = UInt32(group)


def row_groups(
    index: Pointer[UInt32, ImmutAnyOrigin],
    starts: Pointer[UInt32, MutAnyOrigin],
    group_by_row: Pointer[UInt32, MutAnyOrigin],
    groups: UInt32,
    rows: UInt32,
):
    var at = global_idx.x
    if at >= Int(rows):
        return
    # starts[G] is initialized by a separate one-thread sentinel launch.
    var low = 0
    var high = Int(groups)
    while low + 1 < high:
        var mid = low + (high - low) // 2
        if Int(starts[unsafe_offset=mid]) <= at:
            low = mid
        else:
            high = mid
    group_by_row[unsafe_offset=Int(index[unsafe_offset=at])] = UInt32(low)


def end_sentinel(
    starts: Pointer[UInt32, MutAnyOrigin], groups: UInt32, rows: UInt32
):
    if global_idx.x == 0:
        starts[unsafe_offset=Int(groups)] = rows


def fallback_sample(
    hashes: Pointer[UInt64, ImmutAnyOrigin],
    group_by_row: Pointer[UInt32, ImmutAnyOrigin],
    distinct_hashes: Pointer[UInt64, MutAnyOrigin],
    distinct_seen: Pointer[UInt8, MutAnyOrigin],
    counts: Pointer[UInt32, MutAnyOrigin],
    control: Pointer[UInt32, MutAnyOrigin],
    rows: UInt32,
    configured: UInt32,
):
    """Exact low_cardinality sample through row->group->typed hash.

    Dense keys/worker_count<=1 need no sample scratch and return immediately.
    Otherwise scratch has8192UInt64,8192UInt8 and256UInt32 entries. Every
    equal-key row shares its representative's canonical typed/raw-byte hash.
    """
    if global_idx.x != 0:
        return
    var workers = max(1, min(Int(configured), Int(rows) // 65536))
    if control[unsafe_offset=0] or workers <= 1 or rows == 0:
        control[unsafe_offset=1] = 1
        return
    for slot in range(8192):
        distinct_seen[unsafe_offset=slot] = 0
    for slot in range(256):
        counts[unsafe_offset=slot] = 0
    var sample = min(Int(rows), 4096)
    var stride = max(1, Int(rows) // sample)
    var row = 0
    var taken = 0
    var occupied = 0
    var largest = 0
    var distinct = 0
    while row < Int(rows) and taken < sample:
        var hash = hashes[unsafe_offset=Int(group_by_row[unsafe_offset=row])]
        var slot = Int(hash >> UInt64(56))
        if not counts[unsafe_offset=slot]:
            occupied += 1
        counts[unsafe_offset=slot] += 1
        largest = max(largest, Int(counts[unsafe_offset=slot]))
        var at = Int(hash & UInt64(8191))
        while (
            distinct_seen[unsafe_offset=at]
            and distinct_hashes[unsafe_offset=at] != hash
        ):
            at = (at + 1) & 8191
        if not distinct_seen[unsafe_offset=at]:
            distinct += 1
            distinct_seen[unsafe_offset=at] = 1
            distinct_hashes[unsafe_offset=at] = hash
        taken += 1
        row = taken * stride + (taken * 7919) % stride
    var whole = 2 * occupied < min(taken, 256)
    if not whole and taken >= 128 and 4 * largest >= taken:
        whole = 3 * distinct <= taken
    control[unsafe_offset=1] = UInt32(whole)


@always_inline
def _bucket(hash: UInt64, workers: Int) -> UInt64:
    var buckets = 1
    var shift = 64
    while buckets < 2 * workers and buckets < 256:
        buckets *= 2
        shift -= 1
    return hash >> UInt64(shift) if buckets > 1 else UInt64(0)


@always_inline
def _less(
    a: UInt32,
    b: UInt32,
    hashes: Pointer[UInt64, ImmutAnyOrigin],
    index: Pointer[UInt32, ImmutAnyOrigin],
    starts: Pointer[UInt32, ImmutAnyOrigin],
    control: Pointer[UInt32, ImmutAnyOrigin],
    rows: UInt32,
    configured: UInt32,
    fallback: UInt32,
) -> Bool:
    var first = control[unsafe_offset=0] != 0 or (
        fallback != 0 and control[unsafe_offset=1] != 0
    )
    var workers = max(
        1, min(Int(configured), Int(rows) // (65536 if fallback else 2048))
    )
    var left = UInt64(0) if first else _bucket(
        hashes[unsafe_offset=Int(a)], workers
    )
    var right = UInt64(0) if first else _bucket(
        hashes[unsafe_offset=Int(b)], workers
    )
    return (
        left
        < right if left
        != right else index[unsafe_offset=Int(starts[unsafe_offset=Int(a)])]
        < index[unsafe_offset=Int(starts[unsafe_offset=Int(b)])]
    )


def merge_groups(
    hashes: Pointer[UInt64, ImmutAnyOrigin],
    index: Pointer[UInt32, ImmutAnyOrigin],
    starts: Pointer[UInt32, ImmutAnyOrigin],
    control: Pointer[UInt32, ImmutAnyOrigin],
    source: Pointer[UInt32, ImmutAnyOrigin],
    target: Pointer[UInt32, MutAnyOrigin],
    groups: UInt32,
    rows: UInt32,
    configured: UInt32,
    fallback: UInt32,
    run_width: UInt32,
):
    var at = global_idx.x
    if at >= Int(groups):
        return
    var width = Int(run_width)
    var base = at // (2 * width) * (2 * width)
    var middle = min(base + width, Int(groups))
    var end = min(base + 2 * width, Int(groups))
    var left = at < middle
    var first = middle if left else base
    var low = first
    var high = end if left else middle
    var own = source[unsafe_offset=at]
    while low < high:
        var mid = low + (high - low) // 2
        if _less(
            source[unsafe_offset=mid],
            own,
            hashes,
            index,
            starts,
            control,
            rows,
            configured,
            fallback,
        ):
            low = mid + 1
        else:
            high = mid
    target[
        unsafe_offset=base + at - (base if left else middle) + low - first
    ] = own


def choose_runs(
    starts: Pointer[UInt32, ImmutAnyOrigin],
    contiguous_order: Pointer[UInt32, ImmutAnyOrigin],
    fallback_order: Pointer[UInt32, ImmutAnyOrigin],
    control: Pointer[UInt32, MutAnyOrigin],
    phase_by_group: Pointer[UInt32, MutAnyOrigin],
    rank_by_group: Pointer[UInt32, MutAnyOrigin],
    offset_by_group: Pointer[UInt32, MutAnyOrigin],
    groups: UInt32,
    rows: UInt32,
    configured: UInt32,
    allow_contiguous: UInt32,
    cached_layout: UInt32,
):
    """Greedy whole-group cuts and CPU cache-sensitive fallback ids.

    cached_layout certifies that this scope's shared WindowLayouts contains
    the key set. It affects state-group ordering, while fallback operands
    still evaluate in original source-row order.
    """
    if global_idx.x != 0:
        return
    var lanes = min(Int(configured), Int(rows) // 2048)
    var runs = UInt32(0)
    if lanes > 1 and allow_contiguous:
        var target = max(2048, (Int(rows) + 4 * lanes - 1) // (4 * lanes))
        var begin = 0
        while begin < Int(groups):
            var end = begin
            var size = 0
            while end < Int(groups) and size < target:
                var group = Int(contiguous_order[unsafe_offset=end])
                size += Int(
                    starts[unsafe_offset=group + 1]
                    - starts[unsafe_offset=group]
                )
                end += 1
            var offset = UInt32(0)
            for at in range(begin, end):
                var group = Int(contiguous_order[unsafe_offset=at])
                phase_by_group[unsafe_offset=group] = runs
                rank_by_group[unsafe_offset=group] = UInt32(at - begin)
                offset_by_group[unsafe_offset=group] = offset
                offset += (
                    starts[unsafe_offset=group + 1]
                    - starts[unsafe_offset=group]
                )
            runs += 1
            begin = end
    control[unsafe_offset=2] = max(UInt32(1), runs)
    control[unsafe_offset=3] = UInt32(runs > 1)
    if runs <= 1:
        for at in range(Int(groups)):
            var group = Int(
                contiguous_order[
                    unsafe_offset=at
                ] if cached_layout else fallback_order[unsafe_offset=at]
            )
            phase_by_group[unsafe_offset=group] = 0
            rank_by_group[unsafe_offset=group] = UInt32(at)
            offset_by_group[unsafe_offset=group] = 0


def row_ordinals(
    index: Pointer[UInt32, ImmutAnyOrigin],
    starts: Pointer[UInt32, ImmutAnyOrigin],
    group_by_row: Pointer[UInt32, ImmutAnyOrigin],
    phase_by_group: Pointer[UInt32, ImmutAnyOrigin],
    offset_by_group: Pointer[UInt32, ImmutAnyOrigin],
    control: Pointer[UInt32, ImmutAnyOrigin],
    phase_by_row: Pointer[UInt32, MutAnyOrigin],
    ordinal_by_row: Pointer[UInt32, MutAnyOrigin],
    rows: UInt32,
):
    var at = global_idx.x
    if at >= Int(rows):
        return
    var row = Int(index[unsafe_offset=at])
    var group = Int(group_by_row[unsafe_offset=row])
    phase_by_row[unsafe_offset=row] = phase_by_group[unsafe_offset=group]
    ordinal_by_row[unsafe_offset=row] = (
        offset_by_group[unsafe_offset=group]
        + UInt32(at)
        - starts[unsafe_offset=group] if control[unsafe_offset=3] else UInt32(
            row
        )
    )
