"""Scalar row and integer range semantics; independent of dataframe kernels."""
from std.sys import size_of


def row_valid(row: Int, mode: Int) -> Bool:
    return True if mode == 0 else row % 5 != 2 if mode == 1 else False


def row_selected(row: Int, n: Int, density: Int) -> Bool:
    if density == 0:
        return False
    if density == 1:
        return True
    if density == 2:
        return row % 3 == 1
    if density == 3:
        return row % 19 < 13
    return row == n // 2


def expected_selection(n: Int, density: Int, validity: Int) -> List[Int]:
    var rows = List[Int]()
    for row in range(n):
        if row_selected(row, n, density) and row_valid(row, validity):
            rows.append(row)
    return rows^


def row_payload[D: DType](row: Int, valid: Bool) -> Scalar[D]:
    if valid:
        return Scalar[D]((row + 256) % 11 + 1)
    comptime if D.is_floating_point():
        return Scalar[D](0) / Scalar[D](0)
    else:
        return Scalar[D].MAX


def integer_bounds[D: DType]() -> Tuple[Int128, Int128]:
    var width = Int128(size_of[Scalar[D]]() * 8)
    comptime if D.is_signed():
        var magnitude = Int128(1) << (width - 1)
        return (-magnitude, magnitude - 1)
    else:
        return (Int128(0), (Int128(1) << width) - 1)


def checked_row_integer[
    D: DType
](operation: Int, a: Int128, b: Int128) raises -> Int128:
    var limits = integer_bounds[D]()
    var value: Int128
    if operation == 0:
        value = a + b
    elif operation == 1:
        value = a - b
    elif operation == 2:
        # Nonnegative UInt64 products can exceed Int128. Classify them
        # by division before multiplying, without wrapping the reference.
        if a >= 0 and b > 0 and a > limits[1] // b:
            raise Error("reference integer overflow")
        value = a * b
    elif operation == 3:
        if b == 0:
            raise Error("reference divide by zero")
        value = a // b
    elif operation == 4:
        if b == 0:
            raise Error("reference divide by zero")
        value = a % b
    else:
        if b < 0:
            raise Error("reference negative exponent")
        value = 1
        var exponent = b
        while exponent > 0:
            value = checked_row_integer[D](2, value, a)
            exponent -= 1
    if value < limits[0] or value > limits[1]:
        raise Error("reference integer overflow")
    return value


def decimal_power(scale: Int) -> Int128:
    var value = Int128(1)
    for _ in range(scale):
        value *= 10
    return value


def half_even_ratio(numerator: Int128, denominator: Int128) -> Int128:
    var negative = (numerator < 0) != (denominator < 0)
    var a = -numerator if numerator < 0 else numerator
    var b = -denominator if denominator < 0 else denominator
    var quotient = a // b
    var remainder = a % b
    if remainder > b - remainder or (
        remainder == b - remainder and quotient % 2 != 0
    ):
        quotient += 1
    return -quotient if negative else quotient


def decimal_row_result(
    operation: Int, a: Int128, b: Int128, left_scale: Int, right_scale: Int
) -> Int128:
    var result_scale = max(left_scale, right_scale)
    if operation == 0 or operation == 1:
        var x = a * decimal_power(result_scale - left_scale)
        var y = b * decimal_power(result_scale - right_scale)
        return x + y if operation == 0 else x - y
    if operation == 2:
        # Exact, at both scales (DuckDB's product type).
        return a * b
    return half_even_ratio(
        a * decimal_power(result_scale + right_scale - left_scale), b
    )
