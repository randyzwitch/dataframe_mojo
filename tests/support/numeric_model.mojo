"""Primitive range and rounding rules, with no dataframe conversion helpers."""
from std.math import isinf, isnan, trunc
from std.sys import size_of

comptime INTEGER_DTYPES = (
    DType.int8,
    DType.int16,
    DType.int32,
    DType.int64,
    DType.uint8,
    DType.uint16,
    DType.uint32,
    DType.uint64,
)
comptime FLOAT_DTYPES = (DType.float32, DType.float64)


def integer_bounds[D: DType]() -> Tuple[Int128, Int128]:
    comptime bits = size_of[Scalar[D]]() * 8
    comptime if D.is_signed():
        var magnitude = Int128(1) << Int128(bits - 1)
        return (-magnitude, magnitude - 1)
    else:
        return (Int128(0), (Int128(1) << Int128(bits)) - 1)


def integer_fits[D: DType](value: Int128) -> Bool:
    var bounds = integer_bounds[D]()
    return value >= bounds[0] and value <= bounds[1]


def truncated_integer[D: DType](value: Float64) raises -> Int128:
    if isnan(value) or isinf(value):
        raise Error("Nonfinite integer conversion")
    # This envelope is far wider than every public integer dtype. Check before
    # invoking a primitive float-to-Int128 conversion, then use exact bounds.
    if value <= -Float64(Int128(1) << 65) or value >= Float64(Int128(1) << 65):
        raise Error("Integer conversion outside the reference envelope")
    var whole = Int128(trunc(value))
    if not integer_fits[D](whole):
        raise Error("Integer conversion outside target range")
    return whole
