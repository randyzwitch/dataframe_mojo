"""Eight-lane validity bitmap decoding shared by computational kernels."""
from std.memory import Pointer


@always_inline
def _valid_lanes(
    bits: Pointer[UInt8, MutAnyOrigin], bit: Int
) -> SIMD[DType.bool, 8]:
    """Validity of the 8 rows starting at bitmap position `bit`."""
    var mask = UInt16(bits.unsafe_load(bit // 8)) >> UInt16(bit % 8)
    if bit % 8 > 0:
        mask |= UInt16(bits.unsafe_load(bit // 8 + 1)) << UInt16(8 - bit % 8)
    return (
        SIMD[DType.uint16, 8](mask)
        & SIMD[DType.uint16, 8](1, 2, 4, 8, 16, 32, 64, 128)
    ).ne(SIMD[DType.uint16, 8](0))
