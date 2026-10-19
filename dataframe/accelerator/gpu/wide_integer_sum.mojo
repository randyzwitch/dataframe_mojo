"""Bounded exact two-word integer SUM state over native 64-bit inputs.

Each row operation retains its original checked 64-bit type. Only reduction
state widens to two words. The supported row bound makes every mathematical
partial fit signed 128 bits, including unsigned inputs, so merging may safely
reassociate without introducing a narrow intermediate overflow.
"""
from max.gpu import global_idx, thread_idx, block_idx
from max.gpu.primitives.block import prefix_sum
from std.memory import Pointer, bitcast
from .integers import _multiply


@always_inline
def _add_pair(
    low: UInt64, high: UInt64, other_low: UInt64, other_high: UInt64
) -> Tuple[UInt64, UInt64]:
    var result = low + other_low
    return (result, high + other_high + UInt64(result < low))


@always_inline
def _block_sum(low: UInt64, high: UInt64) -> Tuple[UInt64, UInt64]:
    # Summing 256 unsigned 32-bit limbs in UInt64 cannot overflow. Normalize
    # carries only after the four block scans. The last lane owns the total.
    var a = prefix_sum[block_size=256](low & 0xFFFFFFFF)
    var b = prefix_sum[block_size=256](low >> 32)
    var c = prefix_sum[block_size=256](high & 0xFFFFFFFF)
    var d = prefix_sum[block_size=256](high >> 32)
    b += a >> 32
    c += b >> 32
    d += c >> 32
    return (
        (a & 0xFFFFFFFF) | ((b & 0xFFFFFFFF) << 32),
        (c & 0xFFFFFFFF) | ((d & 0xFFFFFFFF) << 32),
    )


def partials[
    unsigned: Bool, nullable: Bool, add: Bool, multiply: Bool
](
    values: Pointer[UInt64 if unsigned else Int64, ImmutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    validity_offset: UInt32,
    rows: UInt32,
    stride: UInt32,
    addend: UInt64,
    factor: UInt64,
    lows: Pointer[Int64, MutAnyOrigin],
    highs: Pointer[Int64, MutAnyOrigin],
    counts: Pointer[UInt32, MutAnyOrigin],
    faults: Pointer[UInt32, MutAnyOrigin],
):
    var low = UInt64(0)
    var high = UInt64(0)
    var count = UInt32(0)
    var failures = UInt32(0)
    var row = global_idx.x
    while row < Int(rows):
        var present = True
        comptime if nullable:
            var bit = Int(validity_offset) + row
            present = (
                validity[unsafe_offset=bit // 8] >> UInt8(bit % 8)
            ) & 1 != 0
        if present:
            var value = UInt64(0)
            comptime if unsigned:
                value = rebind[UInt64](values[unsafe_offset=row])
            else:
                value = bitcast[DType.uint64](
                    rebind[Int64](values[unsafe_offset=row])
                )
            var failed = False
            comptime if add:
                var added = value + addend
                comptime if unsigned:
                    failed = added < value
                else:
                    failed = ((value ^ added) & (addend ^ added)) >> 63 != 0
                value = added
            comptime if multiply:
                if not failed:
                    var product = _multiply(value, factor, 64, unsigned)
                    value = product[0]
                    failed = product[1] != 0
            if failed:
                failures += 1
            else:
                var extension = UInt64(0)
                comptime if not unsigned:
                    extension = UInt64.MAX if value >> 63 else UInt64(0)
                var next = _add_pair(low, high, value, extension)
                low = next[0]
                high = next[1]
                count += 1
        row += Int(stride)
    var total = _block_sum(low, high)
    var block_count = prefix_sum[block_size=256](count)
    var block_failures = prefix_sum[block_size=256](failures)
    if thread_idx.x == 255:
        lows[unsafe_offset=block_idx.x] = bitcast[DType.int64](total[0])
        highs[unsafe_offset=block_idx.x] = bitcast[DType.int64](total[1])
        counts[unsafe_offset=block_idx.x] = block_count
        faults[unsafe_offset=block_idx.x] = block_failures


def finish[
    unsigned: Bool
](
    lows: Pointer[Int64, ImmutAnyOrigin],
    highs: Pointer[Int64, ImmutAnyOrigin],
    counts: Pointer[UInt32, ImmutAnyOrigin],
    faults: Pointer[UInt32, ImmutAnyOrigin],
    partial_count: UInt32,
    output_index: UInt32,
    min_count: Int64,
    result: Pointer[Int64, MutAnyOrigin],
):
    var low = UInt64(0)
    var high = UInt64(0)
    var count = UInt32(0)
    var failures = UInt32(0)
    var at = thread_idx.x
    while at < Int(partial_count):
        var next = _add_pair(
            low,
            high,
            bitcast[DType.uint64](lows[unsafe_offset=at]),
            bitcast[DType.uint64](highs[unsafe_offset=at]),
        )
        low = next[0]
        high = next[1]
        count += counts[unsafe_offset=at]
        failures += faults[unsafe_offset=at]
        at += 256
    var total = _block_sum(low, high)
    var block_count = prefix_sum[block_size=256](count)
    var block_failures = prefix_sum[block_size=256](failures)
    if thread_idx.x == 255:
        var expected_high = UInt64(0)
        comptime if not unsigned:
            expected_high = UInt64.MAX if total[0] >> 63 else UInt64(0)
        block_failures += UInt32(
            Int64(block_count) >= min_count and total[1] != expected_high
        )
        var base = Int(output_index) * 3
        result[unsafe_offset=base] = bitcast[DType.int64](total[0])
        result[unsafe_offset=base + 1] = Int64(block_count)
        result[unsafe_offset=base + 2] = Int64(block_failures)
