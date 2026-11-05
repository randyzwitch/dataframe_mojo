"""Native integer extrema and bitwise Float32 extrema without GPU Float64."""
from std.memory import Pointer, bitcast
from max.gpu import global_idx, block_idx, thread_idx
from max.gpu.primitives import block
from .multi_reduce import Bytes, pick, read_integer
from .filter_sum import BLOCK_SIZE, _valid
from .float_predicate import float_in_range, _ordered

comptime EXTREMA_WORDS = 28


def extrema_partials[
    arity: Int
](
    v0: Bytes,
    v1: Bytes,
    v2: Bytes,
    v3: Bytes,
    b0: Bytes,
    b1: Bytes,
    b2: Bytes,
    b3: Bytes,
    predicate: Bytes,
    predicate_bits: Bytes,
    metadata: Pointer[UInt64, UntrackedOrigin[mut=False]],
    metadata_offset: UInt32,
    minima: Pointer[Int64, MutUntrackedOrigin],
    maxima: Pointer[Int64, MutUntrackedOrigin],
    counts: Pointer[UInt32, MutUntrackedOrigin],
    specials: Pointer[UInt32, MutUntrackedOrigin],
):
    comptime assert 1 <= arity and arity <= 4
    comptime assert BLOCK_SIZE < 1024
    var meta = metadata.unsafe_offset(Int(metadata_offset))
    var rows = Int(meta[unsafe_offset=0])
    var stride = Int(meta[unsafe_offset=1])
    var offset = Int(meta[unsafe_offset=2])
    var filtered = Bool(meta[unsafe_offset=3])
    var predicate_tag = UInt32(meta[unsafe_offset=4])
    var predicate_offset = UInt32(meta[unsafe_offset=5])
    var predicate_nullable = Bool(meta[unsafe_offset=6])
    var inclusive = UInt32(meta[unsafe_offset=7])
    var low = bitcast[DType.int64](meta[unsafe_offset=8])
    var high = bitcast[DType.int64](meta[unsafe_offset=9])
    var low_float = bitcast[DType.float32](UInt32(meta[unsafe_offset=10]))
    var high_float = bitcast[DType.float32](UInt32(meta[unsafe_offset=11]))
    var lo = SIMD[DType.int64, 4](Int64.MAX)
    var hi = SIMD[DType.int64, 4](Int64.MIN)
    var count = SIMD[DType.uint32, 4](0)
    var special = SIMD[DType.uint32, 4](0)
    var row = global_idx.x
    while row < rows:
        var kept = True
        if filtered:
            if predicate_nullable:
                kept = _valid(predicate_bits, predicate_offset, row)
            if predicate_tag == 2:
                kept = kept & float_in_range(
                    predicate.unsafe_bitcast[Float32]()[unsafe_offset=row],
                    low_float,
                    high_float,
                    inclusive,
                )
            else:
                var value = read_integer(predicate, predicate_tag, row)
                kept = kept & (value >= low) & (value <= high)
        comptime for field in range(arity):
            comptime base = 16 + field * 3
            var valid = kept
            if meta[unsafe_offset=base + 2]:
                valid = valid & _valid(
                    pick[field](b0, b1, b2, b3),
                    UInt32(meta[unsafe_offset=base + 1]),
                    row,
                )
            if valid:
                var tag = UInt32(meta[unsafe_offset=base])
                var value = Int64(0)
                var numeric = True
                if tag == 2:
                    # Integer loads preserve subnormal payloads, NaNs and zero
                    # signs even on a device that flushes float arithmetic.
                    var raw = pick[field](v0, v1, v2, v3).unsafe_bitcast[
                        UInt32
                    ]()[unsafe_offset=row]
                    var magnitude = raw & 0x7FFF_FFFF
                    numeric = magnitude <= 0x7F80_0000
                    if not numeric:
                        special[field] |= 1
                    elif magnitude == 0:
                        special[field] |= UInt32(1) << (
                            UInt32(20) if raw >> 31 else UInt32(10)
                        )
                    value = Int64(_ordered(raw))
                else:
                    value = read_integer(pick[field](v0, v1, v2, v3), tag, row)
                if numeric:
                    lo[field] = min(lo[field], value)
                    hi[field] = max(hi[field], value)
                    count[field] += 1
        row += stride
    comptime for field in range(arity):
        var minimum = block.min[block_size=BLOCK_SIZE](lo[field])
        var maximum = block.max[block_size=BLOCK_SIZE](hi[field])
        var n = block.sum[block_size=BLOCK_SIZE](count[field])
        # Three flags in disjoint 10-bit lanes; at most 256 contributions.
        var flags = block.sum[block_size=BLOCK_SIZE](special[field])
        if thread_idx.x == 0:
            var at = (offset + block_idx.x) * arity + field
            minima[unsafe_offset=at] = minimum
            maxima[unsafe_offset=at] = maximum
            counts[unsafe_offset=at] = n
            specials[unsafe_offset=at] = (
                UInt32((flags & 1023) != 0)
                | (UInt32(((flags >> 10) & 1023) != 0) << 1)
                | (UInt32((flags >> 20) != 0) << 2)
            )
