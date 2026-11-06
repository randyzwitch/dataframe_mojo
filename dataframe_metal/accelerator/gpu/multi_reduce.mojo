"""One native scan for a shared predicate and up to four value columns.

The field tags are uniform launch metadata, not per-row tagged cells.
Float32 means use signed integer exponent bins: finite Float32 inputs are
exact multiples of 2**-149, with at most 277 significant bit positions.
Splitting each significand across two 32-bit bins keeps every bin's signed
sum in Int64 under the handoff's fewer-than-2**31-row limit. The host rounds
the combined bins to Float64 once. No GPU Float64 arithmetic is needed.
"""
from max.gpu import global_idx, thread_idx, block_idx
from max.gpu.primitives import block
from .float_predicate import float_in_range
from std.memory import Pointer, bitcast
from std.bit import count_leading_zeros, count_trailing_zeros
from .filter_sum import BLOCK_SIZE, _valid

comptime Bytes = Pointer[UInt8, UntrackedOrigin[mut=False]]


@always_inline
def pick[field: Int](a: Bytes, b: Bytes, c: Bytes, d: Bytes) -> Bytes:
    comptime if field == 0:
        return a
    elif field == 1:
        return b
    elif field == 2:
        return c
    else:
        return d


@always_inline
def read_integer(values: Bytes, tag: UInt32, row: Int) -> Int64:
    if tag == 0:
        return values.unsafe_bitcast[Int32]()[unsafe_offset=row].cast[
            DType.int64
        ]()
    return values.unsafe_bitcast[Int64]()[unsafe_offset=row]


def multi_partials[
    precise: Bool, wide: Bool, arity: Int
](
    v0: Bytes,
    v1: Bytes,
    v2: Bytes,
    v3: Bytes,
    b0: Bytes,
    b1: Bytes,
    b2: Bytes,
    b3: Bytes,
    value_tags: UInt32,
    offsets01: UInt64,
    offsets23: UInt64,
    flags: UInt32,
    predicate: Bytes,
    predicate_bits: Bytes,
    predicate_meta: UInt32,
    predicate_bit_offset: UInt32,
    low_bound: Int64,
    high_bound: Int64,
    low_float: Float32,
    high_float: Float32,
    rows: UInt32,
    stride: UInt32,
    partial_offset: UInt32,
    high: Pointer[Int64, MutUntrackedOrigin],
    low: Pointer[UInt64, MutUntrackedOrigin],
    sums: Pointer[Float32, MutUntrackedOrigin],
    compensations: Pointer[Float32, MutUntrackedOrigin],
    counts: Pointer[UInt32, MutUntrackedOrigin],
    bins: Pointer[Int64, MutUntrackedOrigin],
    specials: Pointer[UInt32, MutUntrackedOrigin],
):
    comptime FIRST_BIN = 0 if wide else 3
    comptime BIN_COUNT = 9 if wide else 3
    comptime BIN_STRIDE = 16 if wide else 4
    comptime BIN_VECTOR = (4 if arity == 1 else (8 if arity == 2 else 16)) * (
        4 if wide else 1
    )
    var tags = SIMD[DType.uint32, 4](0)
    var bit_offsets = SIMD[DType.uint32, 4](
        UInt32(offsets01),
        UInt32(offsets01 >> 32),
        UInt32(offsets23),
        UInt32(offsets23 >> 32),
    )
    comptime for field in range(arity):
        tags[field] = (value_tags >> UInt32(field * 2)) & 3
    var validity_flags = flags & 15
    var sum_flags = (flags >> 4) & 15
    var fields = UInt32(arity)
    var predicate_nullable = (flags >> 12) & 1
    var predicate_tag = predicate_meta & 3
    var filtered = (predicate_meta >> 2) & 1
    var predicate_field = Int32(predicate_meta >> 3) - 1
    var inclusive = (flags >> 16) & 3
    var hi = SIMD[DType.int64, 4](0)
    var lo = SIMD[DType.uint64, 4](0)
    var fs = SIMD[DType.float32, 4](0)
    var fc = SIMD[DType.float32, 4](0)
    var nc = SIMD[DType.uint32, 4](0)
    var exact = SIMD[DType.int64, BIN_VECTOR](0)
    var outside = SIMD[DType.uint32, 4](0)
    var nan = SIMD[DType.uint32, 4](0)
    var positive_inf = SIMD[DType.uint32, 4](0)
    var negative_inf = SIMD[DType.uint32, 4](0)
    var least_bit = SIMD[DType.uint32, 4](512)
    var greatest_bit = SIMD[DType.uint32, 4](0)
    var row = global_idx.x
    while row < Int(rows):
        var kept = True
        var pf = Float32(0)
        var pi = Int64(0)
        if filtered:
            if predicate_nullable:
                kept = _valid(predicate_bits, predicate_bit_offset, row)
            if predicate_tag == 2:
                pf = predicate.unsafe_bitcast[Float32]()[unsafe_offset=row]
                kept = kept & float_in_range(
                    pf, low_float, high_float, inclusive
                )
            else:
                pi = read_integer(predicate, predicate_tag, row)
                kept = kept & (pi >= low_bound) & (pi <= high_bound)
        comptime for field in range(arity):
            if field < Int(fields):
                var valid = kept
                if validity_flags & (UInt32(1) << UInt32(field)):
                    valid = valid & _valid(
                        pick[field](b0, b1, b2, b3), bit_offsets[field], row
                    )
                if valid:
                    nc[field] += 1
                    if sum_flags & (UInt32(1) << UInt32(field)):
                        if tags[field] == 2:
                            var value = (
                                pf if filtered
                                and predicate_field
                                == Int32(field) else pick[field](
                                    v0, v1, v2, v3
                                ).unsafe_bitcast[Float32]()[unsafe_offset=row]
                            )
                            comptime if precise:
                                var raw = bitcast[DType.uint32](value)
                                var exponent = (raw >> 23) & 255
                                if exponent == 255:
                                    if raw & 0x7FFFFF:
                                        nan[field] = 1
                                    elif raw >> 31:
                                        negative_inf[field] = 1
                                    else:
                                        positive_inf[field] = 1
                                else:
                                    var significand = raw & 0x7FFFFF
                                    if exponent:
                                        significand |= 0x800000
                                    var shift = (
                                        exponent - 1
                                    ) if exponent else UInt32(0)
                                    if significand:
                                        least_bit[field] = min(
                                            least_bit[field],
                                            shift
                                            + UInt32(
                                                count_trailing_zeros(
                                                    significand
                                                )
                                            ),
                                        )
                                        greatest_bit[field] = max(
                                            greatest_bit[field],
                                            shift
                                            + 31
                                            - UInt32(
                                                count_leading_zeros(significand)
                                            ),
                                        )
                                    var index = shift >> 5
                                    var shifted = UInt64(significand) << (
                                        UInt64(shift & 31)
                                    )
                                    var sign = Int64(
                                        -1
                                    ) if raw >> 31 else Int64(1)
                                    var lower = (
                                        Int64(shifted & 0xFFFF_FFFF) * sign
                                    )
                                    var upper = Int64(shifted >> 32) * sign
                                    if (
                                        lower != 0
                                        and (
                                            index < UInt32(FIRST_BIN)
                                            or index
                                            >= UInt32(FIRST_BIN + BIN_COUNT)
                                        )
                                    ) or (
                                        upper != 0
                                        and (
                                            index + 1 < UInt32(FIRST_BIN)
                                            or index + 1
                                            >= UInt32(FIRST_BIN + BIN_COUNT)
                                        )
                                    ):
                                        outside[field] = 1
                                    comptime for bin in range(BIN_COUNT):
                                        if index == UInt32(bin + FIRST_BIN):
                                            exact[
                                                field * BIN_STRIDE + bin
                                            ] += lower
                                        if index + 1 == UInt32(bin + FIRST_BIN):
                                            exact[
                                                field * BIN_STRIDE + bin
                                            ] += upper
                            else:
                                var adjusted = value - fc[field]
                                var next = fs[field] + adjusted
                                fc[field] = (next - fs[field]) - adjusted
                                fs[field] = next
                        else:
                            var value = (
                                pi if filtered
                                and predicate_field
                                == Int32(field) else read_integer(
                                    pick[field](v0, v1, v2, v3),
                                    tags[field],
                                    row,
                                )
                            )
                            if tags[field] == 1:
                                hi[field] += value >> 32
                                lo[field] += UInt64(value & 0xFFFF_FFFF)
                            else:
                                hi[field] += value
        row += Int(stride)
    # Every lane executes every collective. Keep runtime dtype/arity tests
    # out of barrier-containing reductions.
    comptime for field in range(arity):
        var h = block.sum[block_size=BLOCK_SIZE, broadcast=False](hi[field])
        var l = block.sum[block_size=BLOCK_SIZE, broadcast=False](lo[field])
        var s = block.sum[block_size=BLOCK_SIZE, broadcast=False](fs[field])
        var c = block.sum[block_size=BLOCK_SIZE, broadcast=False](fc[field])
        var n = block.sum[block_size=BLOCK_SIZE, broadcast=False](nc[field])
        var output = (Int(partial_offset) + block_idx.x) * Int(fields) + field
        if thread_idx.x == 0 and field < Int(fields):
            high[unsafe_offset=output] = h
            low[unsafe_offset=output] = l
            sums[unsafe_offset=output] = s
            compensations[unsafe_offset=output] = c
            counts[unsafe_offset=output] = n
        comptime if precise:
            comptime for bin in range(BIN_COUNT):
                var b = block.sum[block_size=BLOCK_SIZE, broadcast=False](
                    exact[field * BIN_STRIDE + bin]
                )
                if thread_idx.x == 0 and field < Int(fields):
                    bins[unsafe_offset=output * BIN_COUNT + bin] = b
            var nn = block.sum[block_size=BLOCK_SIZE, broadcast=False](
                nan[field]
            )
            var pp = block.sum[block_size=BLOCK_SIZE, broadcast=False](
                positive_inf[field]
            )
            var mm = block.sum[block_size=BLOCK_SIZE, broadcast=False](
                negative_inf[field]
            )
            var least = block.min[block_size=BLOCK_SIZE, broadcast=False](
                least_bit[field]
            )
            var greatest = block.max[block_size=BLOCK_SIZE, broadcast=False](
                greatest_bit[field]
            )
            var window_fault = block.max[
                block_size=BLOCK_SIZE, broadcast=False
            ](outside[field])
            if thread_idx.x == 0 and field < Int(fields):
                specials[unsafe_offset=output * 6 + 5] = window_fault
                specials[unsafe_offset=output * 6] = nn
                specials[unsafe_offset=output * 6 + 1] = pp
                specials[unsafe_offset=output * 6 + 2] = mm
                specials[unsafe_offset=output * 6 + 3] = least
                specials[unsafe_offset=output * 6 + 4] = greatest
