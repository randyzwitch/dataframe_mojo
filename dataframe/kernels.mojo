"""Scalar CPU reference kernels. Null payloads are never evaluated."""
from std.collections import Optional
from .column import Column


def checked_add(a: Int64, b: Int64) raises -> Int64:
    """Raise before signed 64-bit overflow; never silently promote to float."""
    comptime high = Int64(9223372036854775807)
    comptime low = Int64(-9223372036854775807) - 1
    if (b > 0 and a > high - b) or (b < 0 and a < low - b):
        raise Error("Int64 sum overflow")
    return a + b


def sum_int64(column: Column[Int64]) raises -> Optional[Int64]:
    var total = Int64(0)
    var count = 0
    for i in range(len(column)):
        if not column.is_null(i):
            total = checked_add(total, column.value(i))
            count += 1
    if count == 0:
        return Optional[Int64]()
    return Optional[Int64](total)


def sum_float64(column: Column[Float64]) raises -> Optional[Float64]:
    var total = Float64(0)
    var count = 0
    for i in range(len(column)):
        if not column.is_null(i):
            total += column.value(i)
            count += 1
    if count == 0:
        return Optional[Float64]()
    return Optional[Float64](total)


def greater_than[
    dtype: DType
](column: Column[Scalar[dtype]], threshold: Scalar[dtype]) raises -> Column[
    Bool
]:
    var values = List[Bool](capacity=len(column))
    var valid = List[Bool](capacity=len(column))
    for i in range(len(column)):
        var present = not column.is_null(i)
        valid.append(present)
        values.append(present and column.value(i) > threshold)
    return Column[Bool](values^, valid)


def multiply(
    column: Column[Float64], scalar: Float64
) raises -> Column[Float64]:
    var values = List[Float64](capacity=len(column))
    var valid = List[Bool](capacity=len(column))
    for i in range(len(column)):
        var present = not column.is_null(i)
        valid.append(present)
        var value = Float64(0)
        if present:
            value = column.value(i) * scalar
        values.append(value)
    return Column[Float64](values^, valid)
