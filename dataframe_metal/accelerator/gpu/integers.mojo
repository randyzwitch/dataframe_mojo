"""Checked integer division and powers without device Int128 arithmetic."""
from dataframe.expr import FLOORDIV, MOD, POW


@always_inline
def _limit(width: UInt32, unsigned: Bool, negative: Bool) -> UInt64:
    if unsigned:
        return UInt64.MAX if width == 64 else (UInt64(1) << UInt64(width)) - 1
    return (UInt64(1) << UInt64(width - 1)) - UInt64(not negative)


@always_inline
def _multiply(
    a: UInt64, b: UInt64, width: UInt32, unsigned: Bool
) -> Tuple[UInt64, UInt32]:
    var negative = not unsigned and (a ^ b) >> 63 != 0
    var left = UInt64(0) - a if not unsigned and a >> 63 else a
    var right = UInt64(0) - b if not unsigned and b >> 63 else b
    if right != 0 and left > _limit(width, unsigned, negative) // right:
        return (UInt64(0), UInt32(1))
    var value = left * right
    return (UInt64(0) - value if negative else value, UInt32(0))


@always_inline
def evaluate(
    op: Int64, a: UInt64, b: UInt64, width: UInt32, unsigned: Bool
) -> Tuple[UInt64, Bool, UInt32]:
    """Return raw word, validity, and checked error; zero divisors are null."""
    if op == POW:
        if not unsigned and b >> 63:
            return (UInt64(0), True, UInt32(1))
        var value = UInt64(1)
        var factor = a
        var remaining = b
        while remaining:
            if remaining & 1:
                var product = _multiply(value, factor, width, unsigned)
                if product[1]:
                    return (UInt64(0), True, product[1])
                value = product[0]
            remaining >>= 1
            if remaining:
                var product = _multiply(factor, factor, width, unsigned)
                if product[1]:
                    return (UInt64(0), True, product[1])
                factor = product[0]
        return (value, True, UInt32(0))
    if b == 0:
        return (UInt64(0), False, UInt32(0))
    var a_negative = not unsigned and a >> 63 != 0
    var b_negative = not unsigned and b >> 63 != 0
    var left = UInt64(0) - a if a_negative else a
    var right = UInt64(0) - b if b_negative else b
    var quotient = left // right
    var remainder = left % right
    var negative = a_negative != b_negative
    if negative and remainder:
        quotient += 1
        remainder = right - remainder
    if op == MOD:
        return (
            UInt64(0) - remainder if b_negative else remainder,
            True,
            UInt32(0),
        )
    if quotient > _limit(width, unsigned, negative):
        return (UInt64(0), True, UInt32(1))
    return (
        UInt64(0) - quotient if negative else quotient,
        True,
        UInt32(0),
    )
