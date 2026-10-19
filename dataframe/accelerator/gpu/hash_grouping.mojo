"""Bounded equality hash partitioning for order-independent grouped states.

Produces the same index/starts interface as sorted grouping. Hash slots hold
immutable source-row representatives; collisions always compare complete keys.
Probe exhaustion explicitly requests the existing sorted fallback.
"""
from max.gpu import global_idx, thread_idx, block_idx
from max.gpu.primitives.block import prefix_sum
from std.atomic import Atomic
from std.memory import Pointer
from dataframe.dtype import DataType
from dataframe.expr import (
    SUM,
    MEAN,
    MIN,
    MAX,
    COUNT,
    LEN,
    NULL_COUNT,
    FIRST,
    LAST,
    ANY,
    ALL,
)
from dataframe.partition import _mix, _combine
from dataframe.accelerator.rows import RowPlan, RowStep, ROW_GROUP_HEADER
from .grouping import group_equal
from . import floating


def capacity(rows: Int) -> Int:
    var result = 1024
    while result < rows * 2 and result < 1 << 20:
        result *= 2
    return result


def eligible(plan: RowPlan, step: RowStep) -> Bool:
    if not step.grouped or plan.code[4 * step.start] != ROW_GROUP_HEADER:
        return False
    var keys = Int(plan.code[4 * step.start + 1])
    if keys == 0:
        return False
    for node in range(step.start + 1 + keys, step.start + step.nodes):
        var op = plan.code[4 * node + 3]
        if (
            op == COUNT
            or op == LEN
            or op == NULL_COUNT
            or op == FIRST
            or op == LAST
            or op == ANY
            or op == ALL
        ):
            continue
        if op != SUM and op != MEAN and op != MIN and op != MAX:
            return False
        var dtype = plan.slot_dtypes[Int(plan.code[4 * node + 1])]
        # Floating SUM/MEAN have ordered feeds; MIN/MAX retain first zero/NaN
        # encodings. Decimal SUM has checked batch/update semantics as well.
        if (
            dtype.is_decimal()
            or dtype.physical() == DataType.FLOAT32
            or dtype.physical() == DataType.FLOAT64
        ):
            return False
    return True


def initialize(
    representatives: Pointer[UInt32, MutAnyOrigin],
    counts: Pointer[UInt32, MutAnyOrigin],
    metadata: Pointer[UInt32, MutAnyOrigin],
    slots: UInt32,
):
    var at = global_idx.x
    if at < Int(slots):
        representatives[unsafe_offset=at] = UInt32.MAX
        counts[unsafe_offset=at] = 0
    if at < 4:
        metadata[unsafe_offset=at] = 0


@always_inline
def _canonical(word: UInt64, dtype: Int64) -> UInt64:
    if dtype == 11:
        if floating.is_nan(word):
            return 0x7FF8000000000000
        if word & 0x7FFFFFFFFFFFFFFF == 0:
            return 0
    elif dtype == 1:
        var absolute = UInt32(word) & UInt32(0x7FFFFFFF)
        if absolute > UInt32(0x7F800000):
            return 0x7FC00000
        if absolute == 0:
            return 0
        return UInt64(UInt32(word))
    return word


def build(
    words: Pointer[UInt64, ImmutAnyOrigin],
    valid: Pointer[UInt8, ImmutAnyOrigin],
    errors: Pointer[UInt32, ImmutAnyOrigin],
    code: Pointer[Int64, ImmutAnyOrigin],
    types: Pointer[Int64, ImmutAnyOrigin],
    representatives: Pointer[UInt32, MutAnyOrigin],
    counts: Pointer[UInt32, MutAnyOrigin],
    buckets: Pointer[UInt32, MutAnyOrigin],
    metadata: Pointer[UInt32, MutAnyOrigin],
    start: UInt32,
    keys: UInt32,
    rows: UInt32,
    stride: UInt32,
    slots: UInt32,
    capacity: UInt32,
):
    var row = global_idx.x
    if row >= Int(rows):
        return
    var hashed = UInt64(0)
    var failed = UInt32(0)
    for key in range(Int(keys)):
        var node = Int(start) + 1 + key
        var source = Int(code[unsafe_offset=4 * node + 1]) * Int(stride) + row
        failed |= errors[unsafe_offset=source]
        var present = valid[unsafe_offset=source] != 0
        hashed = _combine(hashed, UInt64(present))
        if present:
            hashed = _combine(
                hashed,
                _canonical(
                    words[unsafe_offset=source], types[unsafe_offset=node]
                ),
            )
            if (
                types[unsafe_offset=node] == 12
                or code[unsafe_offset=4 * node + 3] == 2
            ):
                hashed = _combine(
                    hashed,
                    words[unsafe_offset=Int(stride) * Int(slots) + source],
                )
    if failed:
        Atomic[UInt32].max(metadata.unsafe_offset(1), 1)
    var bucket = UInt32(_mix(hashed)) & (capacity - 1)
    for probe in range(128):
        var expected = UInt32.MAX
        var inserted = Atomic[UInt32].compare_exchange[weak=True](
            representatives.unsafe_offset(Int(bucket)), expected, UInt32(row)
        )
        # Metal exposes weak CAS. A spurious empty-slot failure retries the
        # same bucket; moving on could split one key across multiple slots.
        if not inserted and expected == UInt32.MAX:
            continue
        if inserted or group_equal(
            UInt32(row),
            expected,
            words,
            valid,
            code,
            types,
            start,
            keys,
            stride,
            slots,
        ):
            buckets[unsafe_offset=row] = bucket
            _ = Atomic[UInt32].fetch_add(counts.unsafe_offset(Int(bucket)), 1)
            return
        bucket = (bucket + 1) & (capacity - 1)
    buckets[unsafe_offset=row] = UInt32.MAX
    Atomic[UInt32].max(metadata.unsafe_offset(2), 1)


def offsets_local(
    counts: Pointer[UInt32, MutAnyOrigin],
    group_ranks: Pointer[UInt32, MutAnyOrigin],
    blocks: Pointer[UInt32, MutAnyOrigin],
    metadata: Pointer[UInt32, ImmutAnyOrigin],
    capacity: UInt32,
):
    if metadata[unsafe_offset=2]:
        return
    var bucket = global_idx.x
    var count = counts[unsafe_offset=bucket] if bucket < Int(
        capacity
    ) else UInt32(0)
    var present = UInt32(count != 0)
    var rows_before = prefix_sum[block_size=256](count) - count
    var groups_before = prefix_sum[block_size=256](present) - present
    if bucket < Int(capacity):
        counts[unsafe_offset=bucket] = rows_before
        group_ranks[unsafe_offset=bucket] = groups_before | (present << 31)
    if thread_idx.x == 255:
        var length = (Int(capacity) + 255) // 256
        blocks[unsafe_offset=block_idx.x] = rows_before + count
        blocks[unsafe_offset=length + block_idx.x] = groups_before + present


def offsets_blocks(
    blocks: Pointer[UInt32, MutAnyOrigin],
    metadata: Pointer[UInt32, MutAnyOrigin],
    length: UInt32,
):
    # Only block totals are scanned serially (at most 4096 blocks).
    if metadata[unsafe_offset=2]:
        return
    var groups = UInt32(0)
    var offset = UInt32(0)
    for block in range(Int(length)):
        var count = blocks[unsafe_offset=block]
        var group_count = blocks[unsafe_offset=Int(length) + block]
        blocks[unsafe_offset=block] = offset
        blocks[unsafe_offset=Int(length) + block] = groups
        groups += group_count
        offset += count
    metadata[unsafe_offset=0] = groups


def offsets_finish(
    counts: Pointer[UInt32, MutAnyOrigin],
    group_ranks: Pointer[UInt32, ImmutAnyOrigin],
    blocks: Pointer[UInt32, ImmutAnyOrigin],
    starts: Pointer[UInt32, MutAnyOrigin],
    metadata: Pointer[UInt32, ImmutAnyOrigin],
    capacity: UInt32,
):
    var bucket = global_idx.x
    if bucket >= Int(capacity) or metadata[unsafe_offset=2]:
        return
    var length = (Int(capacity) + 255) // 256
    var offset = (
        counts[unsafe_offset=bucket] + blocks[unsafe_offset=block_idx.x]
    )
    counts[unsafe_offset=bucket] = offset
    var rank = group_ranks[unsafe_offset=bucket]
    if rank >> 31:
        var group = (rank & 0x7FFFFFFF) + blocks[
            unsafe_offset=length + block_idx.x
        ]
        starts[unsafe_offset=Int(group)] = offset


def scatter(
    buckets: Pointer[UInt32, ImmutAnyOrigin],
    counts: Pointer[UInt32, MutAnyOrigin],
    index: Pointer[UInt32, MutAnyOrigin],
    metadata: Pointer[UInt32, ImmutAnyOrigin],
    rows: UInt32,
):
    var row = global_idx.x
    if row >= Int(rows) or metadata[unsafe_offset=2]:
        return
    var bucket = buckets[unsafe_offset=row]
    var at = Atomic[UInt32].fetch_add(counts.unsafe_offset(Int(bucket)), 1)
    index[unsafe_offset=Int(at)] = UInt32(row)
