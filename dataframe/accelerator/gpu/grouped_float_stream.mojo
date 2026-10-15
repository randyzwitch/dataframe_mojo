"""Single-worker CPU pipeline float table epochs, without host value reads.

Records use their first source row as an index. At most N records exist.
A group's linked records run current table then reverse flushed tables.
Admission requires the original single-job pipeline certificate. NaN update
priority follows the independent CPU20 witnesses for this association.
"""
from max.gpu import global_idx
from std.memory import Pointer
from dataframe.expr import SUM, MEAN, MIN, MAX
from dataframe.accelerator.memory import checked_add, checked_mul
from . import floating as f

comptime SEEN = UInt32(1) << 30
comptime NAN_SEEN = UInt32(1) << 31


@fieldwise_init
struct Resources(Copyable):
    var retained_bytes: Int
    var build_peak_bytes: Int


def resources(rows: Int, groups: Int) raises -> Resources:
    if rows <= 0 or rows >= (1 << 30) or groups <= 0 or groups > rows:
        raise Error("Float stream epoch geometry exceeds supported bounds")
    var retained = checked_add(checked_mul(rows, 20), checked_mul(groups, 4))
    return Resources(retained, checked_add(retained, checked_mul(groups, 4)))


def _mix(value: UInt64) -> UInt64:
    var z = value
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def int32_part(word: UInt64, valid: Bool) -> UInt32:
    """pipeline._word_hash([rank, signedInt32]), including null rank 2."""
    var h = _mix(UInt64(0x9E3779B97F4A7C15) ^ UInt64(0 if valid else 2))
    h = _mix(h ^ (word if valid else UInt64(0)))
    return UInt32(h >> 60)


def build_epochs(
    key_words: Pointer[UInt64, ImmutAnyOrigin],
    key_validity: Pointer[UInt8, ImmutAnyOrigin],
    key_errors: Pointer[UInt32, ImmutAnyOrigin],
    row_groups: Pointer[UInt32, ImmutAnyOrigin],
    ranges: Pointer[UInt32, ImmutAnyOrigin],
    row_record: Pointer[UInt32, MutAnyOrigin],
    previous: Pointer[UInt32, MutAnyOrigin],
    heads: Pointer[UInt32, MutAnyOrigin],
    group_epoch: Pointer[UInt32, MutAnyOrigin],
    control: Pointer[UInt32, MutAnyOrigin],
    key_base: UInt32,
    rows: UInt32,
    groups: UInt32,
    morsels: UInt32,
    flush_groups: UInt32,
):
    """GPU keys and IDs; only metadata ranges/flush threshold come from host.

    The CPU tests flush strictly after an entire morsel and only when a
    partition's current distinct count exceeds the threshold. Repartition
    at >=min(4096,threshold) preserves existing states rather than resetting.
    """
    if global_idx.x != 0:
        return
    control[unsafe_offset=0] = 0
    if flush_groups == 0:
        control[unsafe_offset=0] = 1
        return
    for group in range(Int(groups)):
        heads[unsafe_offset=group] = UInt32.MAX
        group_epoch[unsafe_offset=group] = 0
    # Match lazy._sampled_groups before choosing the insert-table route.
    # Sampling uses canonical group IDs, so null/key equality is exact and
    # the existing build scratch suffices; no key value leaves the device.
    var sample = min(rows, UInt32(4096))
    var sampled_groups = UInt32(0)
    for at in range(Int(sample)):
        var row = UInt32(UInt64(at) * UInt64(rows) // UInt64(sample))
        var group = row_groups[unsafe_offset=Int(row)]
        if group >= groups:
            control[unsafe_offset=0] = 1
            return
        if not group_epoch[unsafe_offset=Int(group)]:
            group_epoch[unsafe_offset=Int(group)] = 1
            sampled_groups += 1
    if 4 * sampled_groups > sample:
        control[unsafe_offset=0] = 3
        return
    for group in range(Int(groups)):
        group_epoch[unsafe_offset=group] = 0
    var epoch = SIMD[DType.uint32, 16](0)
    var counts = SIMD[DType.uint32, 16](0)
    var split = False
    var consumed = UInt32(0)
    for morsel in range(Int(morsels)):
        var start = ranges[unsafe_offset=2 * morsel]
        var length = ranges[unsafe_offset=2 * morsel + 1]
        if start != consumed or length > rows - consumed:
            control[unsafe_offset=0] = 1
            return
        for row in range(Int(start), Int(start + length)):
            if key_errors[unsafe_offset=Int(key_base) + row]:
                control[unsafe_offset=0] = 2
                return
            var group = row_groups[unsafe_offset=row]
            if group >= groups:
                control[unsafe_offset=0] = 1
                return
            var part = int32_part(
                key_words[unsafe_offset=Int(key_base) + row],
                key_validity[unsafe_offset=Int(key_base) + row] != 0,
            ) if split else UInt32(0)
            var head = heads[unsafe_offset=Int(group)]
            if (
                head == UInt32.MAX
                or group_epoch[unsafe_offset=Int(group)] != epoch[Int(part)]
            ):
                previous[unsafe_offset=row] = head
                heads[unsafe_offset=Int(group)] = UInt32(row)
                group_epoch[unsafe_offset=Int(group)] = epoch[Int(part)]
                head = UInt32(row)
                counts[Int(part)] += 1
            row_record[unsafe_offset=row] = head
        consumed += length
        if not split and counts[0] >= min(UInt32(4096), flush_groups):
            split = True
            counts = SIMD[DType.uint32, 16](0)
            for group in range(Int(groups)):
                var head = heads[unsafe_offset=group]
                if head == UInt32.MAX:
                    continue
                var at = Int(key_base) + Int(head)
                var part = int32_part(
                    key_words[unsafe_offset=at],
                    key_validity[unsafe_offset=at] != 0,
                )
                counts[Int(part)] += 1
        if split:
            for part in range(16):
                if counts[part] > flush_groups:
                    counts[part] = 0
                    epoch[part] += 1
    if consumed != rows:
        control[unsafe_offset=0] = 1


def sample_insert(
    words: Pointer[UInt64, ImmutAnyOrigin],
    valid: Pointer[UInt8, ImmutAnyOrigin],
    representatives: Pointer[UInt32, MutAnyOrigin],
    control: Pointer[UInt32, MutAnyOrigin],
    key_base: UInt32,
    rows: UInt32,
):
    """CPU sample rule for one physical Int32 key, including null equality.

    Only representatives up to the insertion threshold are needed. The
    selected route is the sole readback; no sampled key leaves the device.
    """
    if global_idx.x != 0:
        return
    control[unsafe_offset=0] = 0
    var sample = min(rows, UInt32(4096))
    var distinct = UInt32(0)
    for at in range(Int(sample)):
        var row = UInt32(UInt64(at) * UInt64(rows) // UInt64(sample))
        var index = Int(key_base) + Int(row)
        var present = valid[unsafe_offset=index] != 0
        var seen = False
        for group in range(Int(distinct)):
            var previous = Int(key_base) + Int(
                representatives[unsafe_offset=group]
            )
            var other_present = valid[unsafe_offset=previous] != 0
            if present == other_present and (
                not present
                or words[unsafe_offset=index] == words[unsafe_offset=previous]
            ):
                seen = True
                break
        if not seen:
            distinct += 1
            if 4 * distinct > sample:
                return
            representatives[unsafe_offset=Int(distinct - 1)] = row
    control[unsafe_offset=0] = 1


def add_right_nan(left: UInt64, right: UInt64) -> UInt64:
    """Pinned M1 merge witnesses choose the right quiet NaN payload.

    This policy is local to the captured CPU pipeline association. Ordinary
    row arithmetic retains its separate established floating.add contract.
    """
    if f.is_nan(right):
        return f.quiet(right)
    return f.add(left, right)


def update(
    words: Pointer[UInt64, ImmutAnyOrigin],
    valid: Pointer[UInt8, ImmutAnyOrigin],
    errors: Pointer[UInt32, ImmutAnyOrigin],
    records: Pointer[UInt32, ImmutAnyOrigin],
    totals: Pointer[UInt64, MutAnyOrigin],
    counts: Pointer[UInt32, MutAnyOrigin],
    control: Pointer[UInt32, MutAnyOrigin],
    value_base: UInt32,
    rows: UInt32,
    dtype: Int64,
    op: Int64,
):
    if global_idx.x != 0 or control[unsafe_offset=0]:
        return
    if (dtype != 1 and dtype != 11) or (
        op != SUM and op != MEAN and op != MIN and op != MAX
    ):
        control[unsafe_offset=0] = 1
        return
    # Input errors belong to the already observed plain source, independently
    # of whether that row's float payload is null.
    for row in range(Int(rows)):
        if errors[unsafe_offset=Int(value_base) + row]:
            control[unsafe_offset=0] = 2
            return
        totals[unsafe_offset=row] = 0
        counts[unsafe_offset=row] = 0
    for row in range(Int(rows)):
        var at = Int(value_base) + row
        if not valid[unsafe_offset=at]:
            continue
        var record = records[unsafe_offset=row]
        if record > UInt32(row):
            control[unsafe_offset=0] = 1
            return
        var value = f.word(words[unsafe_offset=at], dtype)
        var count = counts[unsafe_offset=Int(record)]
        var total = totals[unsafe_offset=Int(record)]
        if op == SUM or op == MEAN:
            total = f.add(total, value) if dtype == 1 else add_right_nan(
                total, value
            )
            count += 1
        elif f.is_nan(value):
            count |= NAN_SEEN
        elif not (count & SEEN) or (
            f.less(total, value) if op == MAX else f.less(value, total)
        ):
            total = value
            count |= SEEN
        totals[unsafe_offset=Int(record)] = total
        counts[unsafe_offset=Int(record)] = count


def finish(
    previous: Pointer[UInt32, ImmutAnyOrigin],
    heads: Pointer[UInt32, ImmutAnyOrigin],
    totals: Pointer[UInt64, ImmutAnyOrigin],
    counts: Pointer[UInt32, ImmutAnyOrigin],
    control: Pointer[UInt32, ImmutAnyOrigin],
    output: Pointer[UInt64, MutAnyOrigin],
    valid: Pointer[UInt8, MutAnyOrigin],
    errors: Pointer[UInt32, MutAnyOrigin],
    rows: UInt32,
    groups: UInt32,
    dtype: Int64,
    op: Int64,
    minimum: UInt32,
    canonical_nan: UInt64,
):
    var group = global_idx.x
    if group >= Int(groups) or control[unsafe_offset=0]:
        return
    var record = heads[unsafe_offset=group]
    var total = UInt64(0)
    var count = UInt64(0)
    var flags = UInt32(0)
    var traversed = UInt32(0)
    while record != UInt32.MAX and record < rows and traversed < rows:
        var value = totals[unsafe_offset=Int(record)]
        var n = counts[unsafe_offset=Int(record)]
        if op == SUM or op == MEAN:
            total = add_right_nan(total, value)
            count += UInt64(n)
        else:
            if (n & SEEN) and (
                not (flags & SEEN)
                or (f.less(total, value) if op == MAX else f.less(value, total))
            ):
                total = value
            flags |= n
        record = previous[unsafe_offset=Int(record)]
        traversed += 1
    if record != UInt32.MAX:
        errors[unsafe_offset=group] = 1
        valid[unsafe_offset=group] = 0
        return
    var present = count >= UInt64(minimum) if op == SUM else count != 0
    if op == MEAN and present:
        total = f.div(total, f.from_unsigned(count))
    elif op == MIN or op == MAX:
        present = flags != 0
        if (flags & NAN_SEEN) and (op == MAX or not (flags & SEEN)):
            total = canonical_nan
    if dtype == 1 and op != MEAN:
        total = UInt64(f.to_float32(total))
    output[unsafe_offset=group] = total
    output[unsafe_offset=Int(groups) + group] = 0
    valid[unsafe_offset=group] = UInt8(present)
    errors[unsafe_offset=group] = 0
