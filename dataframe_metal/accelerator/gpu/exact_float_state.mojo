"""Exact Float32 reduction state shared by fused native expression kernels."""
from std.memory import Pointer, bitcast
from std.bit import count_leading_zeros, count_trailing_zeros
from max.gpu import thread_idx
from max.gpu.primitives import block
from .filter_sum import BLOCK_SIZE


struct ExactFloatState[wide: Bool, slots: Int](Copyable, Movable):
    comptime FIRST_BIN = 0 if Self.wide else 3
    comptime BIN_COUNT = 9 if Self.wide else 3
    comptime BIN_STRIDE = 16 if Self.wide else 4
    comptime BIN_VECTOR = (
        4 if Self.slots <= 1 else (8 if Self.slots == 2 else 16)
    ) * (4 if Self.wide else 1)
    var exact: SIMD[DType.int64, Self.BIN_VECTOR]
    var outside: SIMD[DType.uint32, 4]
    var nan: SIMD[DType.uint32, 4]
    var positive_inf: SIMD[DType.uint32, 4]
    var negative_inf: SIMD[DType.uint32, 4]
    var least_bit: SIMD[DType.uint32, 4]
    var greatest_bit: SIMD[DType.uint32, 4]

    @always_inline
    def __init__(out self):
        self.exact = SIMD[DType.int64, Self.BIN_VECTOR](0)
        self.outside = SIMD[DType.uint32, 4](0)
        self.nan = SIMD[DType.uint32, 4](0)
        self.positive_inf = SIMD[DType.uint32, 4](0)
        self.negative_inf = SIMD[DType.uint32, 4](0)
        self.least_bit = SIMD[DType.uint32, 4](512)
        self.greatest_bit = SIMD[DType.uint32, 4](0)

    @always_inline
    def add[field: Int](mut self, value: Float32):
        comptime assert 0 <= field and field < Self.slots
        var raw = bitcast[DType.uint32](value)
        var exponent = (raw >> 23) & 255
        if exponent == 255:
            if raw & 0x7FFFFF:
                self.nan[field] = 1
            elif raw >> 31:
                self.negative_inf[field] = 1
            else:
                self.positive_inf[field] = 1
        else:
            var significand = raw & 0x7FFFFF
            if exponent:
                significand |= 0x800000
            var shift = (exponent - 1) if exponent else UInt32(0)
            if significand:
                self.least_bit[field] = min(
                    self.least_bit[field],
                    shift + UInt32(count_trailing_zeros(significand)),
                )
                self.greatest_bit[field] = max(
                    self.greatest_bit[field],
                    shift + 31 - UInt32(count_leading_zeros(significand)),
                )
            var index = shift >> 5
            var shifted = UInt64(significand) << (UInt64(shift & 31))
            var sign = Int64(-1) if raw >> 31 else Int64(1)
            var lower = Int64(shifted & 0xFFFF_FFFF) * sign
            var upper = Int64(shifted >> 32) * sign
            if (
                lower != 0
                and (
                    index < UInt32(Self.FIRST_BIN)
                    or index >= UInt32(Self.FIRST_BIN + Self.BIN_COUNT)
                )
            ) or (
                upper != 0
                and (
                    index + 1 < UInt32(Self.FIRST_BIN)
                    or index + 1 >= UInt32(Self.FIRST_BIN + Self.BIN_COUNT)
                )
            ):
                self.outside[field] = 1
            comptime for bin in range(Self.BIN_COUNT):
                if index == UInt32(bin + Self.FIRST_BIN):
                    self.exact[field * Self.BIN_STRIDE + bin] += lower
                if index + 1 == UInt32(bin + Self.FIRST_BIN):
                    self.exact[field * Self.BIN_STRIDE + bin] += upper

    @always_inline
    def finish[
        field: Int
    ](
        self,
        output: Int,
        bins: Pointer[Int64, MutUntrackedOrigin],
        specials: Pointer[UInt32, MutUntrackedOrigin],
    ):
        comptime assert 0 <= field and field < Self.slots
        comptime for bin in range(Self.BIN_COUNT):
            var b = block.sum[block_size=BLOCK_SIZE, broadcast=False](
                self.exact[field * Self.BIN_STRIDE + bin]
            )
            if thread_idx.x == 0:
                bins[unsafe_offset=output * Self.BIN_COUNT + bin] = b
        var nn = block.sum[block_size=BLOCK_SIZE, broadcast=False](
            self.nan[field]
        )
        var pp = block.sum[block_size=BLOCK_SIZE, broadcast=False](
            self.positive_inf[field]
        )
        var mm = block.sum[block_size=BLOCK_SIZE, broadcast=False](
            self.negative_inf[field]
        )
        var least = block.min[block_size=BLOCK_SIZE, broadcast=False](
            self.least_bit[field]
        )
        var greatest = block.max[block_size=BLOCK_SIZE, broadcast=False](
            self.greatest_bit[field]
        )
        var window_fault = block.max[block_size=BLOCK_SIZE, broadcast=False](
            self.outside[field]
        )
        if thread_idx.x == 0:
            specials[unsafe_offset=output * 6 + 5] = window_fault
            specials[unsafe_offset=output * 6] = nn
            specials[unsafe_offset=output * 6 + 1] = pp
            specials[unsafe_offset=output * 6 + 2] = mm
            specials[unsafe_offset=output * 6 + 3] = least
            specials[unsafe_offset=output * 6 + 4] = greatest
