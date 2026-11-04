"""Typed, fused range-filter and sum kernels with independent input columns."""
from max.gpu import global_idx, thread_idx, block_idx
from max.gpu.primitives import block
from std.memory import Pointer
from .filter_sum import BLOCK_SIZE, _valid, split_total


def cross_partials[
    value_dtype: DType, filter_dtype: DType
](
    values: Pointer[Scalar[value_dtype], UntrackedOrigin[mut=False]],
    value_offset: UInt32,
    value_bits: Pointer[UInt8, UntrackedOrigin[mut=False]],
    value_bit_offset: UInt32,
    predicate: Pointer[Scalar[filter_dtype], UntrackedOrigin[mut=False]],
    predicate_offset: UInt32,
    predicate_bits: Pointer[UInt8, UntrackedOrigin[mut=False]],
    predicate_bit_offset: UInt32,
    validity_flags: UInt32,
    rows: UInt32,
    stride: UInt32,
    low_bound: Int64,
    high_bound: Int64,
    low_float: Float32,
    high_float: Float32,
    inclusive: UInt32,
    high: Pointer[Int64, MutUntrackedOrigin],
    low: Pointer[UInt64, MutUntrackedOrigin],
    sums: Pointer[Float32, MutUntrackedOrigin],
    compensations: Pointer[Float32, MutUntrackedOrigin],
    counts: Pointer[UInt32, MutUntrackedOrigin],
    partial_offset: UInt32,
):
    var high_sum = Int64(0)
    var low_sum = UInt64(0)
    var total = Float32(0)
    var compensation = Float32(0)
    var count = UInt32(0)
    var row = global_idx.x
    while row < Int(rows):
        var value = values[unsafe_offset=Int(value_offset) + row]
        var condition = predicate[unsafe_offset=Int(predicate_offset) + row]
        var keep = True
        if validity_flags & 1:
            keep = _valid(value_bits, value_bit_offset, row)
        if validity_flags & 2:
            keep = keep & _valid(predicate_bits, predicate_bit_offset, row)
        comptime if filter_dtype == DType.float32:
            var f = condition.cast[DType.float32]()
            var above = f >= low_float if inclusive & 1 else f > low_float
            var below = f <= high_float if inclusive & 2 else f < high_float
            keep = keep & above & below
        else:
            var n = condition.cast[DType.int64]()
            keep = keep & (n >= low_bound) & (n <= high_bound)
        if keep:
            comptime if value_dtype == DType.float32:
                var adjusted = value.cast[DType.float32]() - compensation
                var next = total + adjusted
                compensation = (next - total) - adjusted
                total = next
            else:
                var n = value.cast[DType.int64]()
                comptime if split_total[value_dtype]():
                    high_sum += n >> 32
                    low_sum += UInt64(n & 0xFFFF_FFFF)
                else:
                    high_sum += n
            count += 1
        row += Int(stride)
    var block_high = Int64(0)
    var block_low = UInt64(0)
    var block_total = Float32(0)
    var block_compensation = Float32(0)
    comptime if value_dtype == DType.float32:
        block_total = block.sum[block_size=BLOCK_SIZE, broadcast=False](total)
        block_compensation = block.sum[block_size=BLOCK_SIZE, broadcast=False](
            compensation
        )
    else:
        block_high = block.sum[block_size=BLOCK_SIZE, broadcast=False](high_sum)
        block_low = block.sum[block_size=BLOCK_SIZE, broadcast=False](low_sum)
    var block_count = block.sum[block_size=BLOCK_SIZE, broadcast=False](count)
    if thread_idx.x == 0:
        var output = Int(partial_offset) + block_idx.x
        high[unsafe_offset=output] = block_high
        low[unsafe_offset=output] = block_low
        sums[unsafe_offset=output] = block_total
        compensations[unsafe_offset=output] = block_compensation
        counts[unsafe_offset=output] = block_count
