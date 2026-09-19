"""Strict text parsing shared by read_csv and cast, so both agree."""
from std.utils.numerics import isinf


def parse_int64(text: String) raises -> Int64:
    """Parse strict decimal Int64 without a floating-point round trip."""
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


def edge_ascii_whitespace(text: String) -> Bool:
    var bytes = text.as_bytes()
    if len(bytes) == 0:
        return False
    var first = bytes[0]
    var last = bytes[len(bytes) - 1]
    return (
        first == 32
        or first == 9
        or first == 10
        or first == 13
        or last == 32
        or last == 9
        or last == 10
        or last == 13
    )


def parse_float64(text: String) raises -> Float64:
    """Mojo's float grammar without surrounding whitespace; only explicit
    infinity spellings may produce infinity."""
    if edge_ascii_whitespace(text):
        raise Error("Float64 fields cannot have surrounding whitespace")
    var value: Float64
    try:
        value = Float64(text)
    except:
        raise Error("invalid Float64 value '" + text + "'")
    if isinf(value) and (
        text != "inf"
        and text != "+inf"
        and text != "-inf"
        and text != "Infinity"
        and text != "+Infinity"
        and text != "-Infinity"
    ):
        raise Error("Float64 overflow for '" + text + "'")
    return value


def parse_bool(text: String) raises -> Bool:
    if text == "true":
        return True
    if text == "false":
        return False
    raise Error("Boolean must be exactly 'true' or 'false'")
