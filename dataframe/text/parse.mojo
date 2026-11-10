"""Decimal grammar and range checks around standard numeric parsers."""
from std.utils.numerics import isinf
from std.sys import size_of


def parse_int64(text: StringSlice) raises -> Int64:
    """Parse decimal Int64 with the standard parser after lexical validation."""
    var bytes = text.as_bytes()
    var start = 0
    if len(bytes) and (bytes[0] == 43 or bytes[0] == 45):
        start = 1
    if start == len(bytes):
        raise Error("integer requires digits")
    for i in range(start, len(bytes)):
        if bytes[i] < 48 or bytes[i] > 57:
            raise Error("non-decimal integer byte")
    try:
        return Int64(Int(text))
    except:
        # Grammar has already been checked; preserve the public range error.
        raise Error("Int64 overflow")


def parse_integer[D: DType](text: StringSlice) raises -> Scalar[D]:
    """Parse a strict decimal integer, range-checked for Scalar[D] on the
    digits themselves (no floating-point round trip). "-0" is 0 for
    unsigned types."""
    comptime assert D.is_integral(), "parse_integer needs an integer dtype"
    comptime if size_of[Scalar[D]]() < 8 or D == DType.int64:
        var value = parse_int64(text)
        if value < Int64(Scalar[D].MIN) or value > Int64(Scalar[D].MAX):
            raise Error(String(D) + " overflow")
        return value.cast[D]()
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


def edge_ascii_whitespace(text: StringSlice) -> Bool:
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


def _is_decimal(text: StringSlice) -> Bool:
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


def _is_special_float(text: StringSlice) -> Bool:
    """Recognize the explicitly supported spellings without slicing text.

    A StringSlice is the representation CSV already has, so a byte-level
    check avoids allocating an owned suffix for signed infinity values.
    NaN deliberately has no sign, matching the historical contract.
    """
    var b = text.as_bytes()
    var start = 0
    if len(b) > 0 and (b[0] == 43 or b[0] == 45):
        start = 1
    var remaining = len(b) - start
    if remaining == 3:
        if b[start] == 105 and b[start + 1] == 110 and b[start + 2] == 102:
            return True
        # Unlike infinity, NaN may not have a sign.
        return start == 0 and (
            (b[0] == 110 and b[1] == 97 and b[2] == 110)
            or (b[0] == 78 and b[1] == 97 and b[2] == 78)
        )
    if remaining == 8:
        return (
            b[start] == 73
            and b[start + 1] == 110
            and b[start + 2] == 102
            and b[start + 3] == 105
            and b[start + 4] == 110
            and b[start + 5] == 105
            and b[start + 6] == 116
            and b[start + 7] == 121
        )
    return False


def parse_float64(text: StringSlice) raises -> Float64:
    """Validate the field grammar, then use the standard Float64 parser."""
    return _checked_float64(text)


def parse_float64(text: String) raises -> Float64:
    """Decimal text with an optional exponent, "nan"/"NaN", or an explicit
    infinity spelling. Mojo's own parser is more lenient (it reads
    "2024-02-28" as a number), so the grammar is checked first."""
    return parse_float64(StringSlice(text))


def _checked_float64(text: StringSlice) raises -> Float64:
    """Validate field grammar and overflow around standard conversion."""
    if edge_ascii_whitespace(text):
        raise Error("Float64 fields cannot have surrounding whitespace")
    if not _is_decimal(text) and not _is_special_float(text):
        raise Error("invalid Float64 value '" + String(text) + "'")
    var value: Float64
    try:
        value = Float64(text)
    except:
        raise Error("invalid Float64 value '" + String(text) + "'")
    if isinf(value) and not _is_special_float(text):
        raise Error("Float64 overflow for '" + String(text) + "'")
    return value


def parse_bool(text: StringSlice) raises -> Bool:
    if text == "true":
        return True
    if text == "false":
        return False
    raise Error("Boolean must be exactly 'true' or 'false'")
