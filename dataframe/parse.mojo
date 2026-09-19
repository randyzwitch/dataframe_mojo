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


def parse_integer[D: DType](text: String) raises -> Scalar[D]:
    """Parse a strict decimal integer, range-checked for Scalar[D] on the
    digits themselves (no floating-point round trip). "-0" is 0 for
    unsigned types."""
    comptime assert D.is_integral(), "parse_integer needs an integer dtype"
    comptime if D == DType.int64:
        return rebind[Scalar[D]](parse_int64(text))
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


def _is_decimal(text: String) -> Bool:
    """[+-]? (digits [. digits?] | . digits) ([eE] [+-]? digits)?"""
    var b = text.as_bytes()
    var i = 0
    var n = len(b)
    if i < n and (b[i] == 43 or b[i] == 45):
        i += 1
    var digits = 0
    while i < n and b[i] >= 48 and b[i] <= 57:
        i += 1
        digits += 1
    if i < n and b[i] == 46:
        i += 1
        while i < n and b[i] >= 48 and b[i] <= 57:
            i += 1
            digits += 1
    if digits == 0:
        return False
    if i < n and (b[i] == 101 or b[i] == 69):
        i += 1
        if i < n and (b[i] == 43 or b[i] == 45):
            i += 1
        var exponent = 0
        while i < n and b[i] >= 48 and b[i] <= 57:
            i += 1
            exponent += 1
        if exponent == 0:
            return False
    return i == n


def _is_special_float(text: String) -> Bool:
    var body = text
    if text.startswith("+") or text.startswith("-"):
        body = String(text[byte=1:])
    return (
        body == "inf"
        or body == "Infinity"
        or ((body == "nan" or body == "NaN") and body == text)
    )


def parse_float64(text: String) raises -> Float64:
    """Decimal text with an optional exponent, "nan"/"NaN", or an explicit
    infinity spelling. Mojo's own parser is more lenient (it reads
    "2024-02-28" as a number), so the grammar is checked first."""
    if edge_ascii_whitespace(text):
        raise Error("Float64 fields cannot have surrounding whitespace")
    if not _is_decimal(text) and not _is_special_float(text):
        raise Error("invalid Float64 value '" + text + "'")
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
