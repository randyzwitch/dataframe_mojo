from std.testing import TestSuite, assert_equal
from dataframe.csv_integer import (
    parse_csv_int64,
    parse_csv_uint64,
    parse_csv_integer,
)


def _fails_i64(text: String) raises:
    var failed = False
    try:
        _ = parse_csv_int64(StringSlice(text))
    except:
        failed = True
    assert_equal(failed, True, msg=text)


def _fails_u64(text: String) raises:
    var failed = False
    try:
        _ = parse_csv_uint64(StringSlice(text))
    except:
        failed = True
    assert_equal(failed, True, msg=text)


def test_i64_u64_atoi_simd_contract() raises:
    assert_equal(parse_csv_int64("+000000000000000000001234"), Int64(1234))
    assert_equal(parse_csv_int64("-000000000000000000001234"), Int64(-1234))
    assert_equal(parse_csv_int64("-000000000000000000000000"), Int64(0))
    assert_equal(parse_csv_int64("-9223372036854775808"), Int64.MIN)
    assert_equal(parse_csv_int64("9223372036854775807"), Int64.MAX)
    assert_equal(parse_csv_uint64("18446744073709551615"), UInt64.MAX)
    assert_equal(parse_csv_int64("1234567890123456"), Int64(1234567890123456))
    assert_equal(
        parse_csv_uint64("+0000000000000000000000000000000000"), UInt64(0)
    )
    for text in [
        "",
        "+",
        "-",
        "12x",
        "9223372036854775808",
        "0009223372036854775808",
    ]:
        _fails_i64(text)
    for text in [
        "-0",
        "-1",
        "18446744073709551616",
        "00018446744073709551616",
        "12x",
    ]:
        _fails_u64(text)


def test_all_native_widths() raises:
    assert_equal(parse_csv_integer[DType.int8]("-128"), Int8(-128))
    assert_equal(parse_csv_integer[DType.int16]("-32768"), Int16(-32768))
    assert_equal(parse_csv_integer[DType.int16]("32767"), Int16(32767))
    assert_equal(
        parse_csv_integer[DType.int32]("-2147483648"), Int32(-2147483648)
    )
    assert_equal(
        parse_csv_integer[DType.int32]("2147483647"), Int32(2147483647)
    )
    assert_equal(parse_csv_integer[DType.uint8]("255"), UInt8(255))
    assert_equal(parse_csv_integer[DType.uint16]("65535"), UInt16(65535))
    assert_equal(
        parse_csv_integer[DType.uint32]("4294967295"), UInt32(4294967295)
    )
    # SKIP_ZEROES starts only at the source size boundary, then retries.
    assert_equal(
        parse_csv_integer[DType.uint8]("0000000000000000127"), UInt8(127)
    )
    assert_equal(parse_csv_integer[DType.uint8]("0000000000000000"), UInt8(0))
    for text in ["128", "-129", "01x", "00000000000000000000000128"]:
        var failed = False
        try:
            _ = parse_csv_integer[DType.int8](text)
        except:
            failed = True
        assert_equal(failed, True, msg=text)


def test_lengths_signs_and_invalid_byte_positions() raises:
    # 1..19 fit signed Int64; 20 remains within UInt64. These force every
    # source route: short, partial SSE load, full 16-byte SSE, and 16+tail.
    assert_equal(parse_csv_int64("1"), Int64(1))
    assert_equal(parse_csv_int64("12"), Int64(12))
    assert_equal(parse_csv_int64("123"), Int64(123))
    assert_equal(parse_csv_int64("1234"), Int64(1234))
    assert_equal(parse_csv_int64("12345"), Int64(12345))
    assert_equal(parse_csv_int64("123456"), Int64(123456))
    assert_equal(parse_csv_int64("1234567"), Int64(1234567))
    assert_equal(parse_csv_int64("12345678"), Int64(12345678))
    assert_equal(parse_csv_int64("123456789"), Int64(123456789))
    assert_equal(parse_csv_int64("1234567890"), Int64(1234567890))
    assert_equal(parse_csv_int64("12345678901"), Int64(12345678901))
    assert_equal(parse_csv_int64("123456789012"), Int64(123456789012))
    assert_equal(parse_csv_int64("1234567890123"), Int64(1234567890123))
    assert_equal(parse_csv_int64("12345678901234"), Int64(12345678901234))
    assert_equal(parse_csv_int64("123456789012345"), Int64(123456789012345))
    assert_equal(parse_csv_int64("1234567890123456"), Int64(1234567890123456))
    assert_equal(parse_csv_int64("12345678901234567"), Int64(12345678901234567))
    assert_equal(
        parse_csv_int64("123456789012345678"), Int64(123456789012345678)
    )
    assert_equal(
        parse_csv_int64("1234567890123456789"), Int64(1234567890123456789)
    )
    assert_equal(
        parse_csv_uint64("12345678901234567890"), UInt64(12345678901234567890)
    )
    assert_equal(
        parse_csv_uint64("+12345678901234567890"), UInt64(12345678901234567890)
    )
    assert_equal(parse_csv_int64("-1"), Int64(-1))
    assert_equal(parse_csv_int64("-12"), Int64(-12))
    assert_equal(parse_csv_int64("-1234"), Int64(-1234))
    assert_equal(parse_csv_int64("-1234567890123456"), Int64(-1234567890123456))
    # One non-digit in every byte position of a 20-byte field must reject.
    for text in [
        "x2345678901234567890",
        "1x345678901234567890",
        "12x45678901234567890",
        "123x5678901234567890",
        "1234x678901234567890",
        "12345x78901234567890",
        "123456x8901234567890",
        "1234567x901234567890",
        "12345678x01234567890",
        "123456789x1234567890",
        "1234567890x234567890",
        "12345678901x34567890",
        "123456789012x4567890",
        "1234567890123x567890",
        "12345678901234x67890",
        "123456789012345x7890",
        "1234567890123456x890",
        "12345678901234567x90",
        "123456789012345678x0",
        "1234567890123456789x",
    ]:
        _fails_u64(text)


# Tests from test_csv_decimal.mojo.
# Focused behavior checks for the fast-float2 decimal.rs port.
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


# Tests from test_parse_integer.mojo.
# Integer parser boundaries for every CSV/cast storage dtype.
from std.testing import TestSuite, assert_equal, assert_raises

from dataframe.parse import parse_int64, parse_integer


def accepts[D: DType](text: String, expected: Scalar[D]) raises:
    assert_equal(parse_integer[D](StringSlice(text)), expected, msg=text)


def rejects[D: DType](text: String) raises:
    with assert_raises():
        _ = parse_integer[D](StringSlice(text))


def test_int64_exact_boundaries_and_syntax() raises:
    assert_equal(parse_int64("9223372036854775807"), Int64.MAX)
    assert_equal(
        parse_int64("-9223372036854775808"), Int64(-9223372036854775807) - 1
    )
    for text in [
        "9223372036854775808",
        "-9223372036854775809",
        "",
        "+",
        "-",
        "1.0",
        " 1",
        "1 ",
    ]:
        with assert_raises():
            _ = parse_int64(text)


def test_short_integer_path_preserves_signs_and_bad_byte_rejection() raises:
    assert_equal(parse_int64("+42"), Int64(42))
    assert_equal(parse_int64("-7654321"), Int64(-7654321))
    assert_equal(parse_int64("00000123"), Int64(123))
    for text in ["12x", "1_2", "12 ", " 12"]:
        with assert_raises(contains="non-decimal integer byte"):
            _ = parse_int64(text)


def test_signed_width_boundaries() raises:
    accepts[DType.int8]("-128", Int8.MIN)
    accepts[DType.int8]("127", Int8.MAX)
    rejects[DType.int8]("-129")
    rejects[DType.int8]("128")

    accepts[DType.int16]("-32768", Int16.MIN)
    accepts[DType.int16]("32767", Int16.MAX)
    rejects[DType.int16]("-32769")
    rejects[DType.int16]("32768")

    accepts[DType.int32]("-2147483648", Int32.MIN)
    accepts[DType.int32]("2147483647", Int32.MAX)
    rejects[DType.int32]("-2147483649")
    rejects[DType.int32]("2147483648")

    accepts[DType.int64]("-9223372036854775808", Int64.MIN)
    accepts[DType.int64]("9223372036854775807", Int64.MAX)
    rejects[DType.int64]("-9223372036854775809")
    rejects[DType.int64]("9223372036854775808")


def test_unsigned_width_boundaries_and_negative_zero() raises:
    accepts[DType.uint8]("0", UInt8(0))
    accepts[DType.uint8]("255", UInt8.MAX)
    accepts[DType.uint8]("-0", UInt8(0))
    rejects[DType.uint8]("256")
    rejects[DType.uint8]("-1")

    accepts[DType.uint16]("65535", UInt16.MAX)
    rejects[DType.uint16]("65536")

    accepts[DType.uint32]("4294967295", UInt32.MAX)
    rejects[DType.uint32]("4294967296")

    accepts[DType.uint64]("18446744073709551615", UInt64.MAX)
    rejects[DType.uint64]("18446744073709551616")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
