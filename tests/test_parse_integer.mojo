"""Integer parser boundaries for every CSV/cast storage dtype."""
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
