"""Filtered column sums as per-block partials.

Each thread walks rows with a grid stride, keeps the values whose row is
valid and inside the filter's range, and adds them locally; MAX's
`block.sum` combines a block, and one partial per block goes to device
memory. The host adds the partials. Metal has no atomic add wider than 32
bits, so nothing here relies on one.

Integers of up to 32 bits are added as Int64, exact for fewer than 2**31
rows. 64-bit integers are added as two halves, the arithmetic high 32 bits
as Int64 and the low 32 bits as UInt64; for fewer than 2**31 rows neither
half can wrap, so the host rebuilds the exact total in 128 bits
(`split_total`) however the rows are split, and checks it against the
result type as the CPU does.

Float32 values are added with Kahan compensation in each thread; the block
sums and compensations are kept separately and finished in Float64 on the
host. The data stays Float32: Apple GPUs have no Float64.
"""

from max.gpu import global_idx, thread_idx, block_idx
from max.gpu.primitives import block
from std.memory import Pointer
from std.sys import bit_width_of

comptime BLOCK_SIZE = 256


def split_total[dtype: DType]() -> Bool:
    """Whether `integer_partials` adds `dtype` as two 32-bit halves; else
    the high partial is the whole sum and the low partial is zero."""
    return bit_width_of[dtype]() > 32


@always_inline
def _valid(
    validity: Pointer[UInt8, UntrackedOrigin[mut=False]],
    validity_offset: UInt32,
    row: Int,
) -> Bool:
    var bit = Int(validity_offset) + row
    return (validity[unsafe_offset=bit // 8] >> UInt8(bit % 8)) & 1 != 0


def integer_partials[
    dtype: DType, nullable: Bool, filtered: Bool
](
    values: Pointer[Scalar[dtype], UntrackedOrigin[mut=False]],
    validity: Pointer[UInt8, UntrackedOrigin[mut=False]],
    validity_offset: UInt32,
    rows: UInt32,
    stride: UInt32,
    low_bound: Int64,
    high_bound: Int64,
    high: Pointer[Int64, MutUntrackedOrigin],
    low: Pointer[UInt64, MutUntrackedOrigin],
    counts: Pointer[UInt32, MutUntrackedOrigin],
):
    """Sum of the kept values of one block, as high and low partials (see
    `split_total`), and how many values it kept."""
    var high_sum = Int64(0)
    var low_sum = UInt64(0)
    var count = UInt32(0)
    var row = global_idx.x
    while row < Int(rows):
        var keep = True
        comptime if nullable:
            keep = _valid(validity, validity_offset, row)
        if keep:
            var value = values[unsafe_offset=row].cast[DType.int64]()
            comptime if filtered:
                keep = value >= low_bound and value <= high_bound
            if keep:
                comptime if split_total[dtype]():
                    high_sum += value >> 32
                    low_sum += UInt64(value & 0xFFFF_FFFF)
                else:
                    high_sum += value
                count += 1
        row += Int(stride)
    var block_high = block.sum[block_size=BLOCK_SIZE, broadcast=False](high_sum)
    var block_low = block.sum[block_size=BLOCK_SIZE, broadcast=False](low_sum)
    var block_count = block.sum[block_size=BLOCK_SIZE, broadcast=False](count)
    if thread_idx.x == 0:
        high[unsafe_offset=block_idx.x] = block_high
        low[unsafe_offset=block_idx.x] = block_low
        counts[unsafe_offset=block_idx.x] = block_count


def float32_partials[
    nullable: Bool, filtered: Bool
](
    values: Pointer[Float32, UntrackedOrigin[mut=False]],
    validity: Pointer[UInt8, UntrackedOrigin[mut=False]],
    validity_offset: UInt32,
    rows: UInt32,
    stride: UInt32,
    low_bound: Float32,
    high_bound: Float32,
    inclusive: UInt32,
    sums: Pointer[Float32, MutUntrackedOrigin],
    compensations: Pointer[Float32, MutUntrackedOrigin],
    counts: Pointer[UInt32, MutUntrackedOrigin],
):
    """Kahan-compensated sum of the kept values of one block, its
    compensation, and how many values it kept. Bit 0 of `inclusive` makes
    the low bound inclusive, bit 1 the high bound."""
    var total = Float32(0)
    var compensation = Float32(0)
    var count = UInt32(0)
    var row = global_idx.x
    while row < Int(rows):
        var keep = True
        comptime if nullable:
            keep = _valid(validity, validity_offset, row)
        if keep:
            var value = values[unsafe_offset=row]
            comptime if filtered:
                var above = (
                    value >= low_bound if inclusive & 1 else value > low_bound
                )
                var below = (
                    value <= high_bound if inclusive & 2 else value < high_bound
                )
                keep = above and below
            if keep:
                var adjusted = value - compensation
                var next = total + adjusted
                compensation = (next - total) - adjusted
                total = next
                count += 1
        row += Int(stride)
    var block_total = block.sum[block_size=BLOCK_SIZE, broadcast=False](total)
    var block_compensation = block.sum[block_size=BLOCK_SIZE, broadcast=False](
        compensation
    )
    var block_count = block.sum[block_size=BLOCK_SIZE, broadcast=False](count)
    if thread_idx.x == 0:
        sums[unsafe_offset=block_idx.x] = block_total
        compensations[unsafe_offset=block_idx.x] = block_compensation
        counts[unsafe_offset=block_idx.x] = block_count
