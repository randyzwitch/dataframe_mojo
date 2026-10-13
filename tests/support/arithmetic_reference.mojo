"""Exact scalar integer arithmetic with explicit floor and range semantics.

No dataframe imports or kernels. Magnitude checks precede multiplication so
UInt64 extrema cannot overflow the Int128 reference's intermediate products.
"""
from std.ffi import external_call
from std.math import (
    floor as reference_floor,
    ceil as reference_ceil,
    isnan,
    isinf,
)
from tests.support.numeric_model import integer_bounds, integer_fits

comptime ADD = 0
comptime SUB = 1
comptime MUL = 2
comptime FLOOR_DIV = 3
comptime MOD = 4
comptime POW = 5
comptime TRUE_DIV = 6


def floor_quotient(left: Int128, right: Int128) raises -> Int128:
    if right == 0:
        raise Error("Zero divisor has no integer quotient")
    var a = -left if left < 0 else left
    var b = -right if right < 0 else right
    var q = a // b
    if (left < 0) != (right < 0):
        return -q - Int128(1 if a - q * b != 0 else 0)
    return q


def checked_product[D: DType](left: Int128, right: Int128) raises -> Int128:
    var bounds = integer_bounds[D]()
    var negative = (left < 0) != (right < 0)
    var a = -left if left < 0 else left
    var b = -right if right < 0 else right
    var limit = -bounds[0] if negative else bounds[1]
    if b != 0 and a > limit // b:
        raise Error("Reference multiplication outside dtype range")
    var magnitude = a * b
    var result = -magnitude if negative else magnitude
    if not integer_fits[D](result):
        raise Error("Reference multiplication outside dtype range")
    return result


def integer_result[
    D: DType
](op: Int, left: Int128, right: Int128) raises -> Optional[Int128]:
    var result: Int128
    if op == ADD:
        result = left + right
    elif op == SUB:
        result = left - right
    elif op == MUL:
        result = checked_product[D](left, right)
    elif op == FLOOR_DIV or op == MOD:
        if right == 0:
            return None
        var quotient = floor_quotient(left, right)
        result = quotient if op == FLOOR_DIV else left - quotient * right
    elif op == POW:
        if right < 0:
            raise Error("Reference integer exponent must be nonnegative")
        result = 1
        # Fixtures bound exponents explicitly; repeated multiplication gives a
        # transparent independent oracle rather than expression evaluation.
        for _ in range(Int(right)):
            result = checked_product[D](result, left)
    else:
        raise Error("Unknown integer reference operation")
    if not integer_fits[D](result):
        raise Error("Reference arithmetic outside dtype range")
    return result


def float_result(left: Float64, right: Float64, op: Int) raises -> Float64:
    if op == ADD:
        return left + right
    if op == SUB:
        return left - right
    if op == MUL:
        return left * right
    if op == TRUE_DIV:
        return left / right
    if op == FLOOR_DIV:
        return reference_floor(left / right)
    if op == MOD:
        if right == 0 or isinf(left) or isnan(left) or isnan(right):
            return Float64(0) / Float64(0)
        if isinf(right):
            return infinite_divisor_remainder(left, right)
        var result = left - right * reference_floor(left / right)
        return (
            result if result
            != 0 else Float64(-0.0) if right
            < 0 else Float64(0)
        )
    # Platform libm supplies IEEE exceptional-value semantics; the toolchain
    # approximation maps several NaN powers to infinity. Literal public
    # anchors independently validate these classifications and boundaries.
    return external_call["pow", Float64](left, right)


def infinite_divisor_remainder(left: Float64, right: Float64) -> Float64:
    """Python divisor-sign remainder for finite left and infinite right."""
    if left == 0:
        return Float64(-0.0) if right < 0 else Float64(0.0)
    if (left < 0) == (right < 0):
        return left
    return right
