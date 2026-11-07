"""Exact Float32 squares using integer significands, including subnormals."""
from std.memory import Pointer
from max.gpu import thread_idx
from max.gpu.primitives import block
from .filter_sum import BLOCK_SIZE

comptime SQUARE_BIN_COUNT = 18


@always_inline
def float32_square_words(
    raw: UInt32,
) -> Tuple[UInt32, UInt32, UInt32, UInt32, Bool]:
    """Index and three radix-2**32 words in units of 2**-298.

    A 24-bit significand squares exactly in UInt64. Splitting its shifted
    product before the shift can exceed 64 bits preserves every low bit.
    No device floating arithmetic or 64-bit atomic operation is required.
    """
    var exponent = (raw >> 23) & 255
    if exponent == 255:
        return (UInt32(0), UInt32(0), UInt32(0), UInt32(0), True)
    var significand = raw & 0x7FFFFF
    if exponent:
        significand |= 0x800000
    var square = UInt64(significand) * UInt64(significand)
    var shift = (exponent - 1) * 2 if exponent else UInt32(0)
    var offset = shift & 31
    return (
        shift >> 5,
        UInt32((square << UInt64(offset)) & 0xFFFF_FFFF),
        UInt32((square >> UInt64(32 - offset)) & 0xFFFF_FFFF),
        UInt32(square >> UInt64(64 - offset)) if offset else UInt32(0),
        False,
    )


struct SquareBins[slots: Int](Copyable, Movable):
    comptime STRIDE = 32
    comptime WIDTH = 32 if Self.slots == 1 else (64 if Self.slots == 2 else 128)
    var words: SIMD[DType.int64, Self.WIDTH]
    var non_finite: SIMD[DType.uint32, 4]

    @always_inline
    def __init__(out self):
        comptime assert 1 <= Self.slots and Self.slots <= 4
        self.words = SIMD[DType.int64, Self.WIDTH](0)
        self.non_finite = SIMD[DType.uint32, 4](0)

    @always_inline
    def add[field: Int](mut self, raw: UInt32):
        comptime assert 0 <= field and field < Self.slots
        var value = float32_square_words(raw)
        self.non_finite[field] |= UInt32(value[4])
        comptime for bin in range(SQUARE_BIN_COUNT):
            var contribution = UInt64(0)
            if value[0] == UInt32(bin):
                contribution = UInt64(value[1])
            if value[0] + 1 == UInt32(bin):
                contribution = UInt64(value[2])
            if value[0] + 2 == UInt32(bin):
                contribution = UInt64(value[3])
            self.words[field * Self.STRIDE + bin] += Int64(contribution)

    @always_inline
    def finish[
        field: Int
    ](
        self,
        output: Int,
        squares: Pointer[Int64, MutUntrackedOrigin],
        non_finite: Pointer[UInt32, MutUntrackedOrigin],
    ):
        comptime assert 0 <= field and field < Self.slots
        comptime for bin in range(SQUARE_BIN_COUNT):
            var word = block.sum[block_size=BLOCK_SIZE, broadcast=False](
                self.words[field * Self.STRIDE + bin]
            )
            if thread_idx.x == 0:
                squares[unsafe_offset=output * SQUARE_BIN_COUNT + bin] = word
        var special = block.max[block_size=BLOCK_SIZE, broadcast=False](
            self.non_finite[field]
        )
        if thread_idx.x == 0:
            non_finite[unsafe_offset=output] = special


@always_inline
def integer_moment_magnitude(value: Int64) -> Tuple[UInt64, Bool]:
    """Round an integer exactly as the CPU's Float64 variance input cast.

    This only rounds integer bits; it does not implement floating arithmetic
    on the device. Large-offset CPU accumulation still needs adjudication.
    """
    from std.memory import bitcast
    from std.bit import count_leading_zeros

    var negative = value < 0
    var raw = bitcast[DType.uint64](value)
    var magnitude = UInt64(0) - raw if negative else raw
    var width = 64 - Int(count_leading_zeros(magnitude))
    var shift = UInt64(max(0, width - 53))
    if shift:
        var significand = magnitude >> shift
        var discarded = magnitude & ((UInt64(1) << shift) - 1)
        var halfway = UInt64(1) << (shift - 1)
        if discarded > halfway or (
            discarded == halfway and Bool(significand & 1)
        ):
            significand += 1
        magnitude = significand << shift
    return (magnitude, negative)


@always_inline
def integer_square_words(
    magnitude: UInt64,
) -> Tuple[UInt32, UInt32, UInt32, UInt32]:
    """Exact UInt64 square in four words using only 64-bit intermediates."""
    var low = magnitude & 0xFFFF_FFFF
    var high = magnitude >> 32
    var first = low * low
    var cross = low * high
    var middle = (first >> 32) + ((cross & 0xFFFF_FFFF) << 1)
    var upper = high * high + ((cross >> 32) << 1) + (middle >> 32)
    return (
        UInt32(first & 0xFFFF_FFFF),
        UInt32(middle & 0xFFFF_FFFF),
        UInt32(upper & 0xFFFF_FFFF),
        UInt32(upper >> 32),
    )


struct IntegerMoments[slots: Int](Copyable, Movable):
    """Native integer moment partials; no floating arithmetic or atomics.

    A block must process at most Int32.MAX rows so its signed 64-bit word
    partials cannot overflow. Existing bounded reduction grids satisfy this
    for every input that fits the UInt32 row counter.
    """

    comptime WIDTH = 8 if Self.slots == 1 else (16 if Self.slots == 2 else 32)
    var words: SIMD[DType.int64, Self.WIDTH]
    var counts: SIMD[DType.uint32, 4]
    var largest: SIMD[DType.uint64, 4]

    @always_inline
    def __init__(out self):
        comptime assert 1 <= Self.slots and Self.slots <= 4
        self.words = SIMD[DType.int64, Self.WIDTH](0)
        self.counts = SIMD[DType.uint32, 4](0)
        self.largest = SIMD[DType.uint64, 4](0)

    @always_inline
    def add[field: Int](mut self, value: Int64):
        comptime assert 0 <= field and field < Self.slots
        var rounded = integer_moment_magnitude(value)
        var magnitude = rounded[0]
        var sign = Int64(-1) if rounded[1] else Int64(1)
        self.words[field * 8] += Int64(magnitude & 0xFFFF_FFFF) * sign
        self.words[field * 8 + 1] += Int64(magnitude >> 32) * sign
        var square = integer_square_words(magnitude)
        comptime for word in range(4):
            self.words[field * 8 + 2 + word] += Int64(square[word])
        self.counts[field] += 1
        self.largest[field] = max(self.largest[field], magnitude)

    @always_inline
    def finish[
        field: Int
    ](self, output: Int, partials: Pointer[Int64, MutUntrackedOrigin]):
        from std.memory import bitcast

        comptime assert 0 <= field and field < Self.slots
        comptime for word in range(6):
            var total = block.sum[block_size=BLOCK_SIZE, broadcast=False](
                self.words[field * 8 + word]
            )
            if thread_idx.x == 0:
                partials[unsafe_offset=output * 8 + word] = total
        var count = block.sum[block_size=BLOCK_SIZE, broadcast=False](
            self.counts[field]
        )
        var largest = block.max[block_size=BLOCK_SIZE, broadcast=False](
            self.largest[field]
        )
        if thread_idx.x == 0:
            partials[unsafe_offset=output * 8 + 6] = Int64(count)
            partials[unsafe_offset=output * 8 + 7] = bitcast[DType.int64](
                largest
            )
