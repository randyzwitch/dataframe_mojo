"""Bounded borrowed decimal parsers use independent grammar and rounding models."""
# contract-group: kernel.parse
from std.math import isnan
from std.memory import bitcast
from std.testing import assert_equal, assert_true
from dataframe.parse import parse_int64, parse_integer, parse_float64
from tests.support.numeric_model import INTEGER_DTYPES, integer_bounds
from tests.support.parser_reference import (
    integer_reference,
    float_grammar,
    float_reference,
)


def _integer[D: DType](field: StringSlice) raises:
    var expected = integer_reference[D](field)
    var actual = Scalar[D](0)
    var error = String()
    try:
        actual = parse_integer[D](field)
    except caught:
        error = String(caught)
    if expected[1]:
        assert_true(error.find(expected[1]) >= 0)
    else:
        assert_equal(error, "")
        assert_equal(Int128(actual), expected[0])
    comptime if D == DType.int64:
        error = ""
        try:
            actual = Scalar[D](parse_int64(field))
        except caught:
            error = String(caught)
        if expected[1]:
            assert_true(error.find(expected[1]) >= 0)
        else:
            assert_equal(error, "")
            assert_equal(Int128(actual), expected[0])


def _integer_field(text: String) raises:
    for offset in range(8):
        var prefix = String()
        for _ in range(offset):
            prefix += "9"
        var padded = prefix + text + "999e999"
        var field = StringSlice(
            unsafe_from_utf8=padded.as_bytes()[
                offset : offset + text.byte_length()
            ]
        )
        comptime for index in range(len(INTEGER_DTYPES)):
            _integer[INTEGER_DTYPES[index]](field)
        _ = padded^


def _float(field: StringSlice, text: String) raises:
    var expected = Float64(0)
    var refused = False
    try:
        expected = float_reference(text)
    except:
        refused = True
    var actual = Float64(0)
    var error = String()
    try:
        actual = parse_float64(field)
    except caught:
        error = String(caught)
    if refused != Bool(error):
        raise Error(
            "Borrowed float grammar/range differs from independent reference: "
            + text
        )
    if refused:
        return
    if isnan(expected):
        assert_true(isnan(actual))
    elif bitcast[DType.uint64](actual) != bitcast[DType.uint64](expected):
        raise Error(
            "Borrowed float bits differ from independent strtod: " + text
        )


def _float_field(text: String) raises:
    for offset in [0, 1, 3, 7, 15]:
        var prefix = String()
        for _ in range(offset):
            prefix += "9"
        var padded = prefix + text + "987e42"
        _float(padded[byte = offset : offset + text.byte_length()], text)
        _ = padded^


def test_parser_references_have_literal_limits_rounding_and_grammar_anchors() raises:
    assert_equal(integer_reference[DType.int8]("127")[0], Int128(127))
    assert_equal(integer_reference[DType.int8]("128")[1], "overflow")
    assert_equal(integer_reference[DType.int8]("128x")[1], "overflow")
    assert_equal(
        integer_reference[DType.int8]("12x8")[1], "non-decimal integer byte"
    )
    assert_equal(integer_reference[DType.uint64]("-000")[0], Int128(0))
    assert_equal(
        integer_reference[DType.uint64]("18446744073709551615")[0],
        (Int128(1) << 64) - 1,
    )
    assert_equal(bitcast[DType.uint64](float_reference("-0")), UInt64(1) << 63)
    assert_equal(
        bitcast[DType.uint64](float_reference("0.1")),
        UInt64(0x3FB999999999999A),
    )
    assert_equal(bitcast[DType.uint64](float_reference("5e-324")), UInt64(1))
    assert_equal(float_reference("9007199254740993"), 9007199254740992.0)
    for text in ["1.25", "-.5", "5.", "1.e+2", "nan", "NaN", "-Infinity"]:
        assert_true(float_grammar(text))
    for text in [
        "",
        "+",
        ".",
        "1e+",
        " 1",
        "1 ",
        "-nan",
        "INF",
        "0x1p2",
        "1_0",
        "1\x00",
    ]:
        assert_true(not float_grammar(text))


def test_borrowed_integer_parser_ranges_signs_and_first_error_use_checked_int128() raises:
    for text in [
        "",
        "+",
        "-",
        "+-1",
        "--1",
        " 1",
        "1 ",
        "1_2",
        "1.0",
        "-0",
        "-00",
        "-1",
        "0",
        "+1",
        "1",
        "00000000000000000000000000001",
        "9999999999999999999999999999999999999999",
        "-9999999999999999999999999999999999999999",
        "128x",
        "12x8",
        "18446744073709551616x",
    ]:
        _integer_field(text)
    comptime for index in range(len(INTEGER_DTYPES)):
        var bounds = integer_bounds[INTEGER_DTYPES[index]]()
        for value in [
            bounds[0] - 1,
            bounds[0],
            bounds[0] + 1,
            bounds[1] - 1,
            bounds[1],
            bounds[1] + 1,
        ]:
            _integer_field(String(value))
    for length in [1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 18, 19, 20]:
        var digits = String()
        for _ in range(length):
            digits += "1"
        for sign in ["", "+", "-"]:
            _integer_field(sign + digits)


def test_packed_integer_blocks_validate_every_byte_at_each_borrowed_alignment() raises:
    # Every byte at every position of 4/8/16-digit blocks and their tails,
    # signed and unsigned, and every 0..7 slice offset. Safe-width targets
    # reach packed conversion;
    # narrow targets independently retain first-overflow/error precedence.
    for length in [4, 8, 9, 16, 18, 19]:
        for position in range(length):
            for byte in range(256):
                var bytes = List[UInt8](length=length, fill=49)
                bytes[position] = UInt8(byte)
                var text = String(unsafe_from_utf8=bytes^)
                _integer_field(text)
                _integer_field("+" + text)


def test_borrowed_float_parser_strict_grammar_ieee_ranges_and_slice_ends() raises:
    for text in [
        "0",
        "-0",
        "+0.0",
        "-0.0",
        "1",
        "+1",
        "-1",
        "1.25",
        "-2.5",
        "-.5",
        "+.5",
        ".5",
        "5.",
        "0.1",
        "1e0",
        "1.e2",
        "1E-23",
        "0e99999",
        "-0e-99999",
        "1e-99999",
        "1e-324",
        "5e-324",
        "2.2250738585072014e-308",
        "1.7976931348623157e308",
        "1.7976931348623159e308",
        "1e309",
        "-1e309",
        "9007199254740991",
        "9007199254740992",
        "9007199254740993",
        "900719925474099.2",
        "0.9007199254740992",
        "99999999999999999999.9",
        "1.234567890123456789012345678901234567890123456789e-200",
        "1e00000000000000000000001",
        "-0e-0000000000000000000342",
        "4.9406564584124654e-324",
        "2.225073858507201e-308",
        "000000000000000000000000000001.25",
        "nan",
        "NaN",
        "inf",
        "+inf",
        "-inf",
        "Infinity",
        "+Infinity",
        "-Infinity",
        "",
        "+",
        "-",
        "--1",
        "+-1",
        "-+1",
        ".",
        "e5",
        "1e",
        "1e+",
        "1e-",
        "1e2x",
        "1e1e1",
        "1e1.0",
        "1.5.5",
        "1-2",
        "2024-02-28",
        "-nan",
        "INF",
        " 1",
        "1 ",
        "1\x00",
        "0x1p2",
    ]:
        _float_field(text)


def test_borrowed_float_rounding_4441829661224440_750_matches_strtod() raises:
    for sign in ["", "+", "-"]:
        _float_field(sign + "4441829661224440.750")


def test_borrowed_float_rounding_9651576341293971_matches_strtod() raises:
    for sign in ["", "+", "-"]:
        _float_field(sign + "9651576341293971.0")


def test_borrowed_float_rounding_41900288798_355793_matches_strtod() raises:
    for sign in ["", "+", "-"]:
        _float_field(sign + "41900288798.355793")


def test_borrowed_float_generated_mantissas_scales_and_exponents_match_strtod() raises:
    var state = UInt64(8303)
    for _ in range(32):
        state = state * 6364136223846793005 + 1442695040888963407
        var coefficient = String(
            UInt64(1000000000000000000) + state % UInt64(9000000000000000000)
        )
        for width in [17, 19]:
            var digits = String(coefficient[byte=0:width])
            for position in range(1, width):
                var decimal = (
                    String(digits[byte=0:position])
                    + "."
                    + String(digits[byte=position:])
                )
                for exponent in [
                    -343,
                    -342,
                    -324,
                    -308,
                    -23,
                    -22,
                    -1,
                    0,
                    1,
                    22,
                    23,
                    308,
                    309,
                    999,
                ]:
                    var text = decimal + "e" + String(exponent)
                    _float_field(text)
                    _float_field("-" + text)
