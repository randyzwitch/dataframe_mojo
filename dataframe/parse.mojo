"""Strict text parsing shared by read_csv and cast, so both agree."""
from std.utils.numerics import isinf


def parse_int64(text: StringSlice) raises -> Int64:
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


def parse_integer[D: DType](text: StringSlice) raises -> Scalar[D]:
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


# Powers of ten that a Float64 holds exactly, so mantissa / 10**k is
# correctly rounded whenever the mantissa is exact too (the classic fast
# path: both operands exact means a single correctly-rounded division).
def _pow10(k: Int) -> Float64:
    """10**k for k in [0, 22]; every value here is an exact Float64."""
    if k == 0:
        return 1e0
    if k == 1:
        return 1e1
    if k == 2:
        return 1e2
    if k == 3:
        return 1e3
    if k == 4:
        return 1e4
    if k == 5:
        return 1e5
    if k == 6:
        return 1e6
    if k == 7:
        return 1e7
    if k == 8:
        return 1e8
    if k == 9:
        return 1e9
    if k == 10:
        return 1e10
    if k == 11:
        return 1e11
    if k == 12:
        return 1e12
    if k == 13:
        return 1e13
    if k == 14:
        return 1e14
    if k == 15:
        return 1e15
    if k == 16:
        return 1e16
    if k == 17:
        return 1e17
    if k == 18:
        return 1e18
    if k == 19:
        return 1e19
    if k == 20:
        return 1e20
    if k == 21:
        return 1e21
    if k == 22:
        return 1e22
    return 1.0


# 2**53: above this a Float64 cannot hold every integer, so the fast path
# hands such mantissas to the strict parser instead of rounding twice.
comptime _EXACT_LIMIT = UInt64(9007199254740992)


def parse_float64(text: StringSlice) raises -> Float64:
    """Strict decimal Float64. See the String overload for the contract.

    Plain decimals -- an optional sign, no exponent, at most 22 fraction
    digits and a mantissa a Float64 holds exactly -- are validated and
    computed in one pass. That is the overwhelming majority of real CSV
    data, and the strict parser below costs about 140 ns a field because it
    walks the text to check the grammar and then walks it again to convert.
    Anything the fast path does not fully consume, or cannot represent
    exactly, falls through to that parser, so the accepted grammar and every
    result are unchanged.
    """
    var b = text.as_bytes()
    var n = len(b)
    var mantissa = UInt64(0)
    var digits = 0
    var fraction = -1
    var i = 0
    # The strict grammar takes one leading sign, so reading it here leaves
    # this path a subset of it. Signed data is not an edge case: half the
    # float fields of a column centred on zero are negative, and sending
    # every one of them to the strict parser -- which allocates a String to
    # do it -- was 80 ms of a 730 ms single-threaded read of 1M rows.
    var negative = False
    if n > 0 and (b[0] == 43 or b[0] == 45):
        negative = b[0] == 45
        i = 1
    while i < n:
        var c = b[i]
        if c >= 48 and c <= 57:
            if digits == 19:
                return _parse_float64_strict(text)
            mantissa = mantissa * 10 + UInt64(c - 48)
            digits += 1
            if fraction >= 0:
                fraction += 1
        elif c == 46 and fraction < 0:
            fraction = 0
        else:
            # A sign, an exponent, "nan", "inf", or invalid text.
            return _parse_float64_strict(text)
        i += 1
    if (
        digits == 0
        or fraction == 0  # a trailing "." the strict grammar may reject
        or fraction > 22
        or mantissa >= _EXACT_LIMIT
    ):
        return _parse_float64_strict(text)
    var value = Float64(mantissa)
    if fraction > 0:
        value = value / _pow10(fraction)
    # The sign applies to the magnitude, so "-0.0" stays negative and no
    # other value's rounding changes.
    return -value if negative else value


def parse_float64(text: String) raises -> Float64:
    """Decimal text with an optional exponent, "nan"/"NaN", or an explicit
    infinity spelling. Mojo's own parser is more lenient (it reads
    "2024-02-28" as a number), so the grammar is checked first."""
    return parse_float64(StringSlice(text))


def _parse_float64_strict(text: StringSlice) raises -> Float64:
    """The reference implementation: check the grammar, then convert.

    Grammar validation reads the borrowed CSV field directly.  Mojo's current
    Float64 conversion materializes an owned string for a StringSlice, so
    this fallback is correct but not allocation-free for long fields.
    """
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
