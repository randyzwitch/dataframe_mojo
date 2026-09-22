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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
