"""Exact Float64 rounding anchors for bounded JSON numeric source spans."""
# contract-group: kernel.jsonpath
from std.memory import bitcast
from std.testing import assert_equal, assert_raises
from dataframe.json.number import parse_json_float


def test_json_float_rounding_ties_subnormals_range_and_signed_zero() raises:
    # Bit patterns are literal IEEE-754 results. The long decimals are exact
    # halfway values at 1, plus a residual that must not be discarded.
    var cases: List[Tuple[String, UInt64]] = [
        ("41900288798.355793", UInt64(4765796760783992363)),
        ("-41900288798.355793", UInt64(13989168797638768171)),
        (
            "1.00000000000000011102230246251565404236316680908203125",
            UInt64(0x3FF0000000000000),
        ),
        (
            "1.0000000000000001110223024625156540423631668090820312501",
            UInt64(0x3FF0000000000001),
        ),
        (
            "1.00000000000000033306690738754696212708950042724609375",
            UInt64(0x3FF0000000000002),
        ),
        ("0", UInt64(0)),
        ("-0", UInt64(0x8000000000000000)),
        ("-0.0", UInt64(0x8000000000000000)),
        ("0e10000", UInt64(0)),
        ("-0e10000", UInt64(0x8000000000000000)),
        ("1e-10000", UInt64(0)),
        ("-1e-10000", UInt64(0x8000000000000000)),
        ("2.4703282292062327e-324", UInt64(0)),
        ("2.4703282292062328e-324", UInt64(1)),
        ("4.9406564584124654e-324", UInt64(1)),
        ("2.2250738585072014e-308", UInt64(0x0010000000000000)),
        ("1.7976931348623157e308", UInt64(0x7FEFFFFFFFFFFFFF)),
        ("18446744073709551616", UInt64(0x43F0000000000000)),
    ]
    for example in cases:
        for offset in [0, 1, 7, 15]:
            var owner = String()
            for _ in range(offset):
                owner += "x"
            owner += example[0] + "987e42"
            var field = owner[byte = offset : offset + example[0].byte_length()]
            assert_equal(
                bitcast[DType.uint64](parse_json_float(field)), example[1]
            )


def test_json_float_rejects_overflow_and_unconsumed_suffixes() raises:
    for text in [
        String("1e309"),
        String("-1e309"),
        String("1.7976931348623159e308"),
        String("1e10000"),
        String("1.25x"),
        String("1.25\x00"),
    ]:
        with assert_raises():
            _ = parse_json_float(text)
