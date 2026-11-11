"""Bounded LSB-first bitmap loads shared by storage and execution kernels."""
from std.sys import size_of
from std.memory import bitcast


@always_inline
def _bitmap_word[
    D: DType
](bits: Span[UInt8, _], bit: Int, count: Int, empty: Scalar[D] = 0) -> Scalar[
    D
]:
    """Read up to one word; missing physical bytes and unused tail bits are zero.

    `empty` supplies the caller's policy for an absent bitmap. The complete
    word path uses one unaligned load and, when needed, one spill byte.
    """
    comptime assert D == DType.uint8 or D == DType.uint32 or D == DType.uint64
    comptime BYTES = size_of[Scalar[D]]()
    comptime WIDTH = BYTES * 8
    assert bit >= 0 and count >= 0 and count <= WIDTH
    if count == 0:
        return 0
    if len(bits) == 0:
        return empty if count == WIDTH else empty & (
            (Scalar[D](1) << Scalar[D](count)) - 1
        )
    var byte = bit >> 3
    var shift = Scalar[D](bit & 7)
    if byte + BYTES + 1 <= len(bits):
        var at = bits.unsafe_ptr().unsafe_offset(byte)
        var loaded = bitcast[D, 1](at.unsafe_load[width=BYTES]())
        var spill = Scalar[D](at[unsafe_offset=BYTES])
        var fast = loaded >> shift
        if shift != 0:
            fast |= spill << (Scalar[D](WIDTH) - shift)
        if count < WIDTH:
            fast &= (Scalar[D](1) << Scalar[D](count)) - 1
        return fast
    var word = Scalar[D](0)
    comptime for k in range(BYTES):
        if byte + k < len(bits):
            word |= Scalar[D](bits[byte + k]) << Scalar[D](8 * k)
    word >>= shift
    if shift != 0 and byte + BYTES < len(bits):
        word |= Scalar[D](bits[byte + BYTES]) << (Scalar[D](WIDTH) - shift)
    if count < WIDTH:
        word &= (Scalar[D](1) << Scalar[D](count)) - 1
    return word


@always_inline
def _bitmap_bytes[
    W: Int
](bits: Span[UInt8, _], offset: Int, scalar: Bool, empty: UInt8) -> SIMD[
    DType.uint8, W
]:
    """Read W packed bytes for 8*W rows within one physical window."""
    comptime assert W > 0
    assert offset >= 0
    if len(bits) == 0:
        return SIMD[DType.uint8, W](empty)
    var byte = offset // 8
    var shift = UInt8(offset % 8)
    assert byte < len(bits)
    if scalar:
        var present = (bits[byte] >> shift) & 1 != 0
        return SIMD[DType.uint8, W](UInt8(255) if present else UInt8(0))
    assert W <= len(bits) - byte
    var low = bits.unsafe_ptr().unsafe_load[width=W](byte)
    if shift == 0:
        return low
    # Nonzero bit alignment needs one extra physical byte. Both overlapping
    # W-byte loads stay inside the bitmap; no padded allocation is assumed.
    assert W < len(bits) - byte
    var high = bits.unsafe_ptr().unsafe_load[width=W](byte + 1)
    return (low >> shift) | (high << (UInt8(8) - shift))
