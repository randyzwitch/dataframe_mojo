"""Native numeric casts with exact integer bit operations and CPU null rules."""
from std.bit import count_leading_zeros
from std.memory import bitcast


@always_inline
def integer_float32_bits(value: Int64) -> UInt32:
    var negative = value < 0
    var magnitude = UInt64(0) - UInt64(value) if negative else UInt64(value)
    if magnitude == 0:
        return 0
    var width = 64 - Int(count_leading_zeros(magnitude))
    var shift = max(0, width - 24)
    var significand = magnitude >> UInt64(shift)
    if shift:
        var discarded = magnitude & ((UInt64(1) << UInt64(shift)) - 1)
        var halfway = UInt64(1) << UInt64(shift - 1)
        if discarded > halfway or (
            discarded == halfway and Bool(significand & 1)
        ):
            significand += 1
        if significand == (UInt64(1) << 24):
            significand >>= 1
            width += 1
    return (
        (UInt32(negative) << 31)
        | (UInt32(width + 126) << 23)
        | UInt32((significand << UInt64(max(0, 24 - width))) & 0x7FFFFF)
    )


@always_inline
def float32_integer_bits(raw: UInt32, narrow: Bool) -> Tuple[UInt64, Bool]:
    var encoded = (raw >> 23) & 255
    if encoded == 255:
        return (UInt64(0), True)
    var exponent = Int(encoded) - 127
    if exponent < 0:
        return (UInt64(0), False)
    if exponent > 63:
        return (UInt64(0), True)
    var significand = UInt64((raw & 0x7FFFFF) | 0x800000)
    var magnitude = significand << UInt64(
        exponent - 23
    ) if exponent >= 23 else significand >> UInt64(23 - exponent)
    var negative = Bool(raw >> 31)
    var limit = (UInt64(1) << UInt64(31 if narrow else 63)) - UInt64(
        not negative
    )
    if magnitude > limit:
        return (UInt64(0), True)
    return (UInt64(0) - magnitude if negative else magnitude, False)


@always_inline
def native_cast_bits(
    bits: UInt64, source: UInt32, target: UInt32
) -> Tuple[UInt64, Bool]:
    if source == target:
        return (bits, False)
    if target == 2:
        return (UInt64(integer_float32_bits(bitcast[DType.int64](bits))), False)
    if source == 2:
        return float32_integer_bits(UInt32(bits), target == 0)
    if target == 0:
        var value = bitcast[DType.int64](bits)
        if value < Int64(Int32.MIN) or value > Int64(Int32.MAX):
            return (UInt64(0), True)
    return (bits, False)
