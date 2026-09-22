"""The float fast path must agree with the strict parser, bit for bit.

The strict parser is the reference: it checks the grammar, then hands the text
to the standard library. The fast path exists only to be faster, so every text
it accepts must come out identical and every text it refuses must be refused
by the reference too. These tests compare the two directly rather than
comparing either against expectations, so a disagreement is a real defect.
"""
from std.memory import bitcast
from std.testing import TestSuite, assert_equal

from dataframe.parse import parse_float64, _parse_float64_strict


def assert_agrees(text: String) raises:
    """Both parsers accept or refuse the same text, with the same bits."""
    var fast = Float64(0)
    var fast_refused = False
    try:
        fast = parse_float64(StringSlice(text))
    except:
        fast_refused = True
    var slow = Float64(0)
    var slow_refused = False
    try:
        slow = _parse_float64_strict(text)
    except:
        slow_refused = True
    assert_equal(
        fast_refused, slow_refused, msg="acceptance differs for '" + text + "'"
    )
    if fast_refused:
        return
    # Bits, not values: float equality would call -0.0 equal to 0.0 and every
    # NaN unequal to itself, hiding exactly the mistakes a sign can cause.
    assert_equal(
        bitcast[DType.int64](fast),
        bitcast[DType.int64](slow),
        msg="value differs for '" + text + "'",
    )


def test_signed_decimals_agree_with_the_reference() raises:
    var texts: List[String] = [
        "-0.0",
        "+0.0",
        "0.0",
        "-0",
        "-1",
        "+1",
        "-2.5",
        "+2.5",
        "-.5",
        "+.5",
        "-5.",
        "-49.87",
        "-0.7700000000000031",
        "-1000000",
        "-0.000001",
    ]
    for text in texts:
        assert_agrees(text)


def test_signed_text_the_fast_path_must_not_take() raises:
    """Anything the one-pass loop cannot finish belongs to the reference,
    which is what decides whether it is a number at all."""
    var texts: List[String] = [
        "-",
        "+",
        "--1",
        "+-1",
        "-+1",
        "-.",
        "- 1",
        "-1 ",
        "-nan",
        "-NaN",
        "-inf",
        "+inf",
        "-Infinity",
        "-1.5e3",
        "-1E10",
        "-12345678901234567890",
        "-99999999999999999.5",
        "-1.234567890123456789012345",
        "-1-2",
        "-1.5.5",
        "",
    ]
    for text in texts:
        assert_agrees(text)


def test_wide_mantissas_exponents_and_extremes_agree() raises:
    """The fallback covers values the one-pass decimal path cannot round.

    These include the ordinary long mantissas seen in CSV output, boundaries
    around the exact-integer range, and the exponent/subnormal limits where
    an apparently harmless shortcut can change a bit.
    """
    var texts: List[String] = [
        "49.970000000000006",
        "9007199254740991",
        "9007199254740992",
        "9007199254740993",
        "1.234567890123456789",
        "-1.234567890123456789",
        "1e0",
        "1e-324",
        "5e-324",
        "2.2250738585072014e-308",
        "1.7976931348623157e308",
        "1e309",
        "-1e309",
    ]
    for text in texts:
        assert_agrees(text)


def test_every_sign_and_magnitude_round_trips() raises:
    """A spread of signed values, formatted and read back, must give the
    reference's bits. A loop catches what a hand-written list does not."""
    var seed = UInt64(20260921)
    for _ in range(4000):
        # A small xorshift: the values only have to spread, not be random.
        seed ^= seed << 13
        seed ^= seed >> 7
        seed ^= seed << 17
        var magnitude = Float64(Int(seed % 100000000)) / 1000.0
        var value = -magnitude if (seed & 1) == 0 else magnitude
        assert_agrees(String(value))


def test_decimal_rounding_across_scales() raises:
    var seed = UInt64(152)
    for _ in range(1000):
        seed = seed * 6364136223846793005 + 1442695040888963407
        var digits = String(seed % UInt64(10000000000000000000))
        for point in range(1, digits.byte_length()):
            var text = (
                String(digits[byte=0:point]) + "." + String(digits[byte=point:])
            )
            assert_agrees(text)
            assert_agrees("-" + text)
    for text in [
        "9999999999999999999",
        "18446744073709551616",
        "99999999999999999999.9",
        "9007199254740993.0",
        "9007199254740995.0",
    ]:
        assert_agrees(text)


def test_seventeen_digit_decimal_rounding() raises:
    var seed = UInt64(152)
    for _ in range(2000):
        seed = seed * 6364136223846793005 + 1442695040888963407
        var digits = String(
            UInt64(10000000000000000) + seed % UInt64(90000000000000000)
        )
        for point in range(1, 17):
            var text = (
                String(digits[byte=0:point]) + "." + String(digits[byte=point:])
            )
            assert_agrees(text)
            assert_agrees("-" + text)


def test_borrowed_conversion_boundaries() raises:
    # Include historical rounding differences from mathematically rounded
    # conversion: this optimization must preserve the installed reference.
    var texts: List[String] = [
        "4441829661224440.750",
        "9651576341293971.0",
        "41900288798.355793",
        "9007199254740992",
        "900719925474099.2",
        "0.9007199254740992",
        "9999999999999999999",
        "00000000000000000000000000000000000001.25",
        "1.234567890123456789012345678901234567890123456789e-200",
        "0e99999",
        "1e-99999",
        "1e99999",
        "2.225073858507201e-308",
        "4.9406564584124654e-324",
        "1.7976931348623159e308",
        "nan",
        "NaN",
        "inf",
        "Infinity",
        "INF",
        "2024-02-28",
        "1-2",
        "1.5.5",
        "1e",
        ".",
        "+",
        "e5",
        "1e2x",
    ]
    for text in texts:
        assert_agrees(text)
        assert_agrees("-" + text)
        assert_agrees("+" + text)
        # The borrowed converter must respect the slice's end, even with
        # more numeric-looking bytes in the underlying String.
        var padded = "123" + text + "987e42"
        var field = padded[byte = 3 : 3 + text.byte_length()]
        var actual: String
        var expected: String
        try:
            actual = String(bitcast[DType.uint64](parse_float64(field)))
        except e:
            actual = String(e)
        try:
            expected = String(
                bitcast[DType.uint64](_parse_float64_strict(field))
            )
        except e:
            expected = String(e)
        assert_equal(actual, expected, msg="borrowed field: " + text)


def test_exponent_conversion_matches_reference() raises:
    for mantissa in [
        "0",
        "-0",
        "+0.0",
        "000.000",
        "1.",
        "9007199254740992",
        "9999999999999999999",
    ]:
        for exponent in [
            -999,
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
            assert_agrees(mantissa + "e" + String(exponent))
    for text in [
        "1e+",
        "1e-",
        "1.e",
        ".e1",
        "1e1.0",
        "1e1e1",
        "1e00000000000000000000001",
        "-0e-0000000000000000000342",
    ]:
        assert_agrees(text)
    var seed = UInt64(110)
    for _ in range(1000):
        seed = seed * 6364136223846793005 + 1442695040888963407
        var digits = String(seed % UInt64(10000000000000000000))
        var point = Int(seed % UInt64(digits.byte_length()))
        var value = (
            String(digits[byte=0:point]) + "." + String(digits[byte=point:])
        )
        for exponent in [
            -343,
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
        ]:
            var text = value + "e" + String(exponent)
            assert_agrees(text)
            assert_agrees("-" + text)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
