"""Focused behavior checks for the fast-float2 decimal.rs port."""
from std.testing import TestSuite, assert_equal, assert_true
from dataframe.csv_decimal import CsvDecimal, parse_csv_decimal


def digits(d: CsvDecimal) -> List[UInt8]:
    var result = List[UInt8]()
    for i in range(d.num_digits):
        result.append(d.digits[i])
    return result^


def test_parse_decimal_point_and_trailing_zero_rules() raises:
    var integer = parse_csv_decimal("0012300")
    assert_equal(digits(integer), [UInt8(1), 2, 3])
    assert_equal(integer.decimal_point, 5)
    assert_equal(integer.round(), UInt64(12300))

    var fractional = parse_csv_decimal("000.0012300")
    assert_equal(digits(fractional), [UInt8(1), 2, 3])
    assert_equal(fractional.decimal_point, -2)
    assert_equal(fractional.round(), UInt64(0))

    var exponent = parse_csv_decimal("100e-2")
    assert_equal(digits(exponent), [UInt8(1)])
    assert_equal(exponent.decimal_point, 1)
    assert_equal(exponent.round(), UInt64(1))


def test_batched_binary_shifts_and_half_even_rounding() raises:
    var left = parse_csv_decimal("123")
    left.left_shift(10)
    assert_equal(digits(left), [UInt8(1), 2, 5, 9, 5, 2])
    assert_equal(left.decimal_point, 6)
    assert_equal(left.round(), UInt64(125952))

    var right = parse_csv_decimal("1")
    right.right_shift(1)
    assert_equal(digits(right), [UInt8(5)])
    assert_equal(right.decimal_point, 0)

    assert_equal(parse_csv_decimal("2.5").round(), UInt64(2))
    assert_equal(parse_csv_decimal("3.5").round(), UInt64(4))
    var truncated = CsvDecimal()
    truncated.num_digits = 2
    truncated.decimal_point = 1
    truncated.digits[0] = 2
    truncated.digits[1] = 5
    truncated.truncated = True
    assert_equal(truncated.round(), UInt64(3))
    assert_true(left.num_digits <= 768)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
