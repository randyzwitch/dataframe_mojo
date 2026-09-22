"""Differential coverage for the packed strict-integer parser.

The references below are the scalar #149 implementation frozen in the test.
They make the SIMD path prove both its value and its error priority without
depending on whichever parser happens to be imported at test time.
"""
from std.testing import TestSuite, assert_equal

from dataframe.parse import parse_int64, parse_integer


def _reference_int64(text: StringSlice) raises -> Int64:
    var bytes = text.as_bytes()
    if len(bytes) == 0:
        raise Error("empty integer")
    var index = 0
    var negative = False
    if bytes[0] == 45 or bytes[0] == 43:
        negative = bytes[0] == 45
        index = 1
    if index == len(bytes):
        raise Error("integer sign without digits")
    var digits = len(bytes) - index
    if digits <= 18:
        var magnitude = UInt64(0)
        while index < len(bytes):
            var byte = bytes[index]
            if byte < 48 or byte > 57:
                raise Error("non-decimal integer byte")
            magnitude = magnitude * 10 + UInt64(byte - 48)
            index += 1
        if negative:
            return -Int64(magnitude)
        return Int64(magnitude)
    var limit = UInt64(9223372036854775807) + UInt64(negative)
    var magnitude = UInt64(0)
    while index < len(bytes):
        var byte = bytes[index]
        if byte < 48 or byte > 57:
            raise Error("non-decimal integer byte")
        var digit = UInt64(byte - 48)
        if magnitude > (limit - digit) // 10:
            raise Error("Int64 overflow")
        magnitude = magnitude * 10 + digit
        index += 1
    if negative:
        if magnitude == UInt64(9223372036854775808):
            return Int64(-9223372036854775807) - 1
        return -Int64(magnitude)
    return Int64(magnitude)


def _reference_integer[D: DType](text: StringSlice) raises -> Scalar[D]:
    comptime assert D.is_integral(), "integer dtype required"
    comptime if D == DType.int64:
        return rebind[Scalar[D]](_reference_int64(text))
    var bytes = text.as_bytes()
    if len(bytes) == 0:
        raise Error("empty integer")
    var index = 0
    var negative = False
    if bytes[0] == 45 or bytes[0] == 43:
        negative = bytes[0] == 45
        index = 1
    if index == len(bytes):
        raise Error("integer sign without digits")
    var high = Scalar[D].MAX.cast[DType.uint64]()
    var limit = high
    comptime if D.is_signed():
        limit = high + UInt64(negative)
    else:
        if negative:
            limit = 0
    var magnitude = UInt64(0)
    while index < len(bytes):
        var byte = bytes[index]
        if byte < 48 or byte > 57:
            raise Error("non-decimal integer byte")
        var digit = UInt64(byte - 48)
        if magnitude > (limit - min(digit, limit)) // 10 or (
            digit > limit - magnitude * 10
        ):
            raise Error(String(D) + " overflow")
        magnitude = magnitude * 10 + digit
        index += 1
    comptime if D.is_signed():
        if negative:
            return (-magnitude.cast[DType.int64]()).cast[D]()
    return magnitude.cast[D]()


def _assert_int64(text: StringSlice, label: String) raises:
    var actual = Int64(0)
    var actual_error = ""
    try:
        actual = parse_int64(text)
    except e:
        actual_error = String(e)
    var expected = Int64(0)
    var expected_error = ""
    try:
        expected = _reference_int64(text)
    except e:
        expected_error = String(e)
    assert_equal(actual_error, expected_error, msg="Int64 error " + label)
    if not actual_error:
        assert_equal(actual, expected, msg="Int64 value " + label)


def _assert_width[D: DType](text: StringSlice, label: String) raises:
    var actual = Scalar[D](0)
    var actual_error = ""
    try:
        actual = parse_integer[D](text)
    except e:
        actual_error = String(e)
    var expected = Scalar[D](0)
    var expected_error = ""
    try:
        expected = _reference_integer[D](text)
    except e:
        expected_error = String(e)
    assert_equal(
        actual_error, expected_error, msg=String(D) + " error " + label
    )
    if not actual_error:
        assert_equal(actual, expected, msg=String(D) + " value " + label)


def _assert_all_widths(text: StringSlice, label: String) raises:
    _assert_int64(text, label)
    _assert_width[DType.int8](text, label)
    _assert_width[DType.uint8](text, label)
    _assert_width[DType.int16](text, label)
    _assert_width[DType.uint16](text, label)
    _assert_width[DType.int32](text, label)
    _assert_width[DType.uint32](text, label)
    _assert_width[DType.int64](text, label)
    _assert_width[DType.uint64](text, label)


def _digits(length: Int, seed: Int) -> String:
    var bytes = List[UInt8](capacity=length)
    for i in range(length):
        bytes.append(UInt8(48 + (i * 7 + seed) % 10))
    return String(unsafe_from_utf8=bytes^)


def _replace(text: String, index: Int, byte: UInt8) -> String:
    var bytes = List[UInt8](capacity=text.byte_length())
    bytes.extend(text.as_bytes())
    bytes[index] = byte
    return String(unsafe_from_utf8=bytes^)


def _assert_at_slice_offsets(text: String, label: String) raises:
    """The SIMD loads must respect the slice length at every alignment."""
    for prefix in range(8):
        var bytes = List[UInt8](capacity=prefix + text.byte_length() + 3)
        for _ in range(prefix):
            bytes.append(126)
        bytes.extend(text.as_bytes())
        bytes.extend([UInt8(126), 126, 126])
        var padded = String(unsafe_from_utf8=bytes^)
        var slice = StringSlice(
            unsafe_from_utf8=padded.as_bytes()[
                prefix : prefix + text.byte_length()
            ]
        )
        _assert_all_widths(slice, label + " offset " + String(prefix))


def test_packed_values_signs_and_slice_boundaries() raises:
    for width in [1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 18]:
        var digits = _digits(width, width)
        _assert_at_slice_offsets(digits, "digits " + String(width))
        _assert_at_slice_offsets("+" + digits, "plus " + String(width))
        _assert_at_slice_offsets("-" + digits, "minus " + String(width))
    for text in [
        "000000000000000000",
        "+000000000000000000",
        "-000000000000000000",
        "9223372036854775807",
        "9223372036854775808",
        "-9223372036854775808",
        "-9223372036854775809",
        "18446744073709551615",
        "18446744073709551616",
    ]:
        _assert_at_slice_offsets(text, "boundary")


def test_each_short_digit_position_preserves_error_order() raises:
    for width in [4, 8, 16]:
        var digits = _digits(width, 3)
        for position in range(width):
            for byte in [UInt8(0), 47, 58, 65, 128, 255]:
                _assert_at_slice_offsets(
                    _replace(digits, position, byte),
                    "bad byte " + String(width) + ":" + String(position),
                )
                _assert_at_slice_offsets(
                    _replace("+" + digits, position + 1, byte),
                    "signed bad byte " + String(width) + ":" + String(position),
                )


def test_syntax_and_long_overflow_fallbacks_match_scalar() raises:
    for text in [
        "",
        "+",
        "-",
        "++1",
        "--1",
        "+-1",
        "-+1",
        " 1",
        "1 ",
        "1_2",
        "1.0",
        "-0",
        "-00",
        "-000",
        "-1",
        "-12",
        "-123",
        "-12x",
        "-999x",
        "9999999999999999999999999999999999999999",
        "-9999999999999999999999999999999999999999",
    ]:
        _assert_at_slice_offsets(text, "syntax or long fallback")


def test_every_byte_in_packed_blocks() raises:
    # Carries and borrows in packed ASCII validation must not hide a bad
    # byte. Include every byte value, not only printable invalid text.
    for width in [4, 8, 9, 16, 18, 19]:
        var digits = _digits(width, 3)
        for position in range(width):
            for byte in range(256):
                var text = _replace(digits, position, UInt8(byte))
                var label = "byte " + String(byte) + " at " + String(position)
                _assert_all_widths(StringSlice(text), label)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
