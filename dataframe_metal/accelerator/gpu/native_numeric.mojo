"""Native numeric value operations without device Float64 arithmetic."""
from std.memory import bitcast
from std.bit import count_trailing_zeros
from dataframe.expr import (
    NEG,
    ABS,
    SIGN,
    FLOOR,
    CEIL,
    TRUNCATE,
    ROUND,
    CLIP_LOW,
    CLIP_HIGH,
    DIV,
    FLOORDIV,
    MOD,
)


@always_inline
def numeric_unary(op: Int) -> Bool:
    return (
        op == NEG
        or op == ABS
        or op == SIGN
        or op == FLOOR
        or op == CEIL
        or op == TRUNCATE
        or op == ROUND
    )


@always_inline
def float32_unary_bits(raw: UInt32, op: Int) -> UInt32:
    """Exact bit operations, including subnormals and signed zero.

    ROUND is the CPU's zero-decimal half-away operation. In particular,
    ROUND(-0.0) produces +0.0, while rounding a small negative value can
    produce -0.0. No floating arithmetic may flush a subnormal operand.
    """
    var sign = raw & 0x8000_0000
    var magnitude = raw & 0x7FFF_FFFF
    if op == NEG:
        return raw ^ 0x8000_0000
    if op == ABS:
        return magnitude
    if op == SIGN:
        return (
            raw if magnitude == 0
            or magnitude > 0x7F80_0000 else sign | 0x3F80_0000
        )
    if magnitude >= 0x7F80_0000:
        return raw
    if op == ROUND and magnitude == 0:
        return 0
    var exponent = Int(magnitude >> 23) - 127
    if exponent >= 23:
        return raw
    if exponent < 0:
        if magnitude == 0:
            return raw
        var unit = (op == FLOOR and sign != 0) or (op == CEIL and sign == 0)
        unit = unit or (op == ROUND and exponent == -1)
        return sign | (UInt32(0x3F80_0000) if unit else UInt32(0))
    var unit = UInt32(1) << UInt32(23 - exponent)
    var fraction = magnitude & (unit - 1)
    var whole = magnitude - fraction
    if fraction != 0 and (
        (op == FLOOR and sign != 0)
        or (op == CEIL and sign == 0)
        or (op == ROUND and fraction >= unit >> 1)
    ):
        whole += unit
    return sign | whole


@always_inline
def integer_unary_bits(
    bits: UInt64, tag: UInt32, op: Int
) -> Tuple[UInt64, Bool]:
    var value = bitcast[DType.int64](bits)
    if op == NEG or op == ABS:
        if value == (Int64(Int32.MIN) if tag == 0 else Int64.MIN):
            return (UInt64(0), True)
        if op == NEG or value < 0:
            return (UInt64(0) - bits, False)
    elif op == SIGN:
        return (
            bitcast[DType.uint64](
                Int64(-1 if value < 0 else 1 if value > 0 else 0)
            ),
            False,
        )
    return (bits, False)


@always_inline
def float32_clip_bits(a: UInt32, b: UInt32, op: Int) -> UInt32:
    var am = a & 0x7FFF_FFFF
    var bm = b & 0x7FFF_FFFF
    if am > 0x7F80_0000 or bm > 0x7F80_0000 or (am == 0 and bm == 0):
        return a
    var ak = ~a if a >> 31 else a ^ 0x8000_0000
    var bk = ~b if b >> 31 else b ^ 0x8000_0000
    return b if (ak < bk if op == CLIP_LOW else ak > bk) else a


@always_inline
def integer_divmod_bits(
    a: Int64, b: Int64, tag: UInt32, op: Int
) -> Tuple[UInt64, Bool, Bool]:
    """Floor quotient/remainder: result bits, validity, checked fault."""
    if b == 0:
        return (UInt64(0), False, False)
    if a == Int64.MIN and b == -1:
        return (UInt64(0), True, op == FLOORDIV)
    var x = UInt64(0) - UInt64(a) if a < 0 else UInt64(a)
    var y = UInt64(0) - UInt64(b) if b < 0 else UInt64(b)
    var quotient = UInt64(0)
    var remainder = UInt64(0)
    if (y & (y - 1)) == 0:
        quotient = x >> count_trailing_zeros(y)
        remainder = x & (y - 1)
    elif tag == 0:
        # Int32 magnitudes fit UInt32, including abs(Int32.MIN).
        var narrow = UInt32(x) // UInt32(y)
        quotient = UInt64(narrow)
        remainder = UInt64(UInt32(x) - narrow * UInt32(y))
    else:
        quotient = x // y
        remainder = x - quotient * y
    var opposite = (a < 0) != (b < 0)
    var result = UInt64(0)
    if op == FLOORDIV:
        result = (
            UInt64(0)
            - quotient
            - UInt64(remainder != 0) if opposite else quotient
        )
    else:
        result = y - remainder if opposite and remainder != 0 else remainder
        if b < 0:
            result = UInt64(0) - result
    var value = bitcast[DType.int64](result)
    var fault = tag == 0 and (
        value < Int64(Int32.MIN) or value > Int64(Int32.MAX)
    )
    return (result, True, fault)


@always_inline
def float32_divide_bits(a: UInt32, b: UInt32) -> Tuple[UInt32, Bool]:
    """Round a finite binary32 ratio once, ties-to-even, using integer words.

    The flag requests CPU adjudication for non-finite operands or a zero
    divisor, preserving the CPU's IEEE payload/error behavior. Subnormal
    operands and results are handled directly, without flushing or Float64.
    """
    var sign = (a ^ b) & 0x8000_0000
    var am = a & 0x7FFF_FFFF
    var bm = b & 0x7FFF_FFFF
    if am >= 0x7F80_0000 or bm >= 0x7F80_0000 or bm == 0:
        return (UInt32(0), True)
    if am == 0:
        return (sign, False)
    var ae = Int(am >> 23)
    var be = Int(bm >> 23)
    var av = am & 0x7FFFFF
    var bv = bm & 0x7FFFFF
    if ae:
        av |= 0x800000
    else:
        ae = 1
        while av < 0x800000:
            av <<= 1
            ae -= 1
    if be:
        bv |= 0x800000
    else:
        be = 1
        while bv < 0x800000:
            bv <<= 1
            be -= 1
    var exponent = ae - be + 127
    if av < bv:
        av <<= 1
        exponent -= 1
    if exponent >= 255:
        return (sign | 0x7F80_0000, False)
    if exponent < -23:
        return (sign, False)
    var shift = 23 if exponent > 0 else exponent + 22
    var numerator = UInt64(av) << UInt64(max(shift, 0))
    var denominator = UInt64(bv) << UInt64(max(-shift, 0))
    # All estimate operands/results are normal binary32 values. Correct the
    # estimate using exact integer products before deciding the final rounding;
    # the hardware division's approximation cannot change the result bits.
    var scale = bitcast[DType.float32](UInt32(127 + shift) << 23)
    var estimate = (Float32(av) / Float32(bv)) * scale
    var quotient = UInt64(UInt32(estimate))
    var product = quotient * denominator
    while product > numerator:
        quotient -= 1
        product -= denominator
    var remainder = numerator - product
    while remainder >= denominator:
        quotient += 1
        remainder -= denominator
    if remainder * 2 > denominator or (
        remainder * 2 == denominator and Bool(quotient & 1)
    ):
        quotient += 1
    if exponent <= 0:
        return (sign | UInt32(quotient), False)
    if quotient == UInt64(1) << 24:
        quotient >>= 1
        exponent += 1
    if exponent >= 255:
        return (sign | 0x7F80_0000, False)
    return (
        sign | (UInt32(exponent) << 23) | (UInt32(quotient) & 0x7FFFFF),
        False,
    )


@always_inline
def native_numeric_binary(
    a: UInt64, b: UInt64, tag: UInt32, op: Int
) -> Tuple[UInt64, Bool, UInt32]:
    """Advanced binary value bits, validity, and existing data-fault flags."""
    if tag != 2:
        var left = bitcast[DType.int64](a)
        var right = bitcast[DType.int64](b)
        if op == CLIP_LOW or op == CLIP_HIGH:
            return (
                b if (left < right if op == CLIP_LOW else left > right) else a,
                True,
                UInt32(0),
            )
        var divided = integer_divmod_bits(left, right, tag, op)
        return (divided[0], divided[1], UInt32(divided[2]))
    var left = UInt32(a)
    var right = UInt32(b)
    if op == CLIP_LOW or op == CLIP_HIGH:
        return (UInt64(float32_clip_bits(left, right, op)), True, UInt32(0))
    var divided = float32_divide_bits(left, right)
    if divided[1]:
        return (UInt64(0), True, UInt32(2))
    if op == DIV:
        return (UInt64(divided[0]), True, UInt32(0))
    if op == FLOORDIV:
        return (UInt64(float32_unary_bits(divided[0], FLOOR)), True, UInt32(0))
    # Zero modulo a finite nonzero divisor is positive zero on CPU. No
    # arithmetic or gradual-underflow handling is needed, even for tiny divisors.
    if left & 0x7FFF_FFFF == 0:
        return (UInt64(0), True, UInt32(0))
    # The CPU SIMD remainder truncates the rounded quotient, uses a fused
    # multiply-subtract, then adds the divisor for opposite-sign operands.
    # It is not an exact mathematical remainder. Reproduce that recurrence.
    # Small exponent windows can require gradual underflow in FMA/add; keep
    # those cases on CPU rather than letting Metal flush their residual.
    var le = (left >> 23) & 255
    var re = (right >> 23) & 255
    if le <= 24 or re <= 24 or divided[0] & 0x7FFF_FFFF >= 0x7F80_0000:
        return (UInt64(0), True, UInt32(2))
    var quotient = float32_unary_bits(divided[0], TRUNCATE)
    var residual = bitcast[DType.float32](quotient ^ 0x8000_0000).fma(
        bitcast[DType.float32](right), bitcast[DType.float32](left)
    )
    if (left >> 31) != (right >> 31) and residual != 0:
        residual = residual + bitcast[DType.float32](right)
    var result = bitcast[DType.uint32](residual)
    return (UInt64(0 if result & 0x7FFF_FFFF == 0 else result), True, UInt32(0))
