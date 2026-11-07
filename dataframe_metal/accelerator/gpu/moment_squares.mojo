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
