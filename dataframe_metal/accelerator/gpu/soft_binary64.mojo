"""Round-to-nearest binary64 arithmetic for Metal devices without Float64."""

from std.bit import count_leading_zeros


comptime _SIGN = UInt64(1) << 63
comptime _HIDDEN = UInt64(1) << 52
comptime _FRACTION = _HIDDEN - 1
comptime _INFINITY = UInt64(0x7FF0000000000000)
comptime _QUIET = UInt64(1) << 51


@always_inline
def widen_binary32(bits: UInt32) -> UInt64:
    """Promote IEEE binary32 to binary64, including subnormals and signed zero."""
    var sign = UInt64(bits >> 31) << 63
    var exponent = Int((bits >> 23) & 255)
    var fraction = UInt64(bits & 0x7FFFFF)
    if exponent == 255:
        if fraction:
            return sign | _INFINITY | _QUIET | (fraction << 29)
        return sign | _INFINITY
    if exponent:
        return sign | (UInt64(exponent + 896) << 52) | (fraction << 29)
    if fraction == 0:
        return sign
    var top = 63 - Int(count_leading_zeros(fraction))
    return (
        sign
        | (UInt64(top + 874) << 52)
        | ((fraction << UInt64(52 - top)) & _FRACTION)
    )


@always_inline
def _shift_right_sticky(value: UInt64, shift: Int) -> UInt64:
    """Shift right and fold any discarded nonzero bits into bit zero."""
    if shift == 0:
        return value
    if shift >= 64:
        return UInt64(value != 0)
    return (value >> UInt64(shift)) | UInt64((value << UInt64(64 - shift)) != 0)


@always_inline
def add_binary64(a: UInt64, b: UInt64) -> UInt64:
    """Add two binary64 bit patterns with one round to nearest, ties to even."""
    var a_mag = a & ~_SIGN
    var b_mag = b & ~_SIGN
    if a_mag > _INFINITY:
        return a | _QUIET
    if b_mag > _INFINITY:
        return b | _QUIET
    if a_mag == _INFINITY and b_mag == _INFINITY and a != b:
        return _INFINITY | _QUIET
    # The larger magnitude determines the sign unless the operands cancel.
    var large = a if a_mag >= b_mag else b
    var small = b if a_mag >= b_mag else a
    var sign = large & _SIGN
    if (large & ~_SIGN) == _INFINITY:
        return large
    if (small & ~_SIGN) == 0:
        if (large & ~_SIGN) == 0:
            return a & b & _SIGN
        return large
    var large_exp = Int((large >> 52) & 2047)
    var small_exp = Int((small >> 52) & 2047)
    var left = large & _FRACTION
    var right = small & _FRACTION
    if large_exp:
        left |= _HIDDEN
    if small_exp:
        right |= _HIDDEN
    # Subnormal and minimum-normal values share the same effective exponent.
    var exponent = max(1, large_exp)
    left <<= 3
    right = _shift_right_sticky(right << 3, exponent - max(1, small_exp))
    var result: UInt64
    if (small & _SIGN) == sign:
        result = left + right
        if result & (_HIDDEN << 4):
            result = _shift_right_sticky(result, 1)
            exponent += 1
    else:
        result = left - right
        if result == 0:
            return 0
        var shift = min(exponent - 1, Int(count_leading_zeros(result)) - 8)
        result <<= UInt64(shift)
        exponent -= shift
    var significand = result >> 3
    var discarded = result & 7
    if discarded > 4 or (discarded == 4 and Bool(significand & 1)):
        significand += 1
        if significand == (_HIDDEN << 1):
            significand >>= 1
            exponent += 1
    if exponent >= 2047:
        return sign | _INFINITY
    if significand < _HIDDEN:
        return sign | significand
    return sign | (UInt64(exponent) << 52) | (significand & _FRACTION)
