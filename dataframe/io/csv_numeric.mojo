"""CSV numeric syntax around EmberJSON Float32 and standard Float64 conversion.

EmberJSON owns Float32 rounding, including halfway and subnormal values. This
adapter preserves CSV signs, leading zeroes, decimal-point forms, special
values, and overflow to infinity without a local decimal conversion algorithm.
"""
from emberjson import Parser, DerErrorKind, DeserializationError
from std.memory import bitcast


@always_inline
def _digit(c: UInt8) -> Bool:
    return c >= 48 and c <= 57


def _scan(text: StringSlice) raises -> Tuple[Int, Int, Int]:
    """Validate the whole CSV field; return sign, integer and exponent ends."""
    var b = text.as_bytes()
    var n = len(b)
    var i = 0
    if n and (b[0] == 43 or b[0] == 45):
        i = 1
    var start = i
    while i < n and _digit(b[i]):
        i += 1
    var integer_end = i
    var digits = i - start
    if i < n and b[i] == 46:
        i += 1
        var fraction_start = i
        while i < n and _digit(b[i]):
            i += 1
        digits += i - fraction_start
    var exponent_start = i
    if i < n and (b[i] == 101 or b[i] == 69):
        i += 1
        if i < n and (b[i] == 43 or b[i] == 45):
            i += 1
        var exponent_digits = i
        while i < n and _digit(b[i]):
            i += 1
        if i == exponent_digits:
            raise Error("invalid floating-point value")
    if digits == 0 or i != n:
        raise Error("invalid floating-point value")
    return start, integer_end, exponent_start


def _special(text: StringSlice) -> Tuple[Bool, Float32]:
    var b = text.as_bytes()
    var start = 0
    var negative = False
    if len(b) and (b[0] == 43 or b[0] == 45):
        negative = b[0] == 45
        start = 1
    var n = len(b) - start
    var bits = UInt32(0)
    if (
        n == 3
        and b[start] | 32 == 110
        and b[start + 1] | 32 == 97
        and b[start + 2] | 32 == 110
    ):
        bits = 0x7FC00000
    elif (
        (n == 3 or n == 8)
        and b[start] | 32 == 105
        and b[start + 1] | 32 == 110
        and b[start + 2] | 32 == 102
    ):
        if n == 8 and not (
            b[start + 3] | 32 == 105
            and b[start + 4] | 32 == 110
            and b[start + 5] | 32 == 105
            and b[start + 6] | 32 == 116
            and b[start + 7] | 32 == 121
        ):
            return False, 0
        bits = 0x7F800000
    else:
        return False, 0
    if negative:
        bits |= 0x80000000
    return True, bitcast[DType.float32](bits)


def _overflow(error: DeserializationError) -> Bool:
    # EmberJSON 0.5.0 has no distinct range-error kind. Match only its two
    # overflow diagnostics; changing this dependency requires range tests.
    return error.kind == DerErrorKind.InvalidValue and (
        error.message == "Infinite float" or error.message == "float overflow"
    )


def _parse_json(text: StringSlice) raises -> Float32:
    var parser = Parser(text)
    var result = Float32(0)
    try:
        result = parser.expect_float[DType.float32]()
    except err:
        # The caller already validated the complete CSV field.
        if _overflow(err):
            return bitcast[DType.float32](UInt32(0x7F800000))
        raise Error(String(err))
    if parser.has_more():
        raise Error("invalid floating-point value")
    return result


def _adapt_csv_float32(text: StringSlice) raises -> Float32:
    var special = _special(text)
    if special[0]:
        return special[1]
    var scanned = _scan(text)
    var start = scanned[0]
    var integer_end = scanned[1]
    var exponent_start = scanned[2]
    var bytes = text.as_bytes()
    var negative = start == 1 and bytes[0] == 45
    while start + 1 < integer_end and bytes[start] == 48:
        start += 1
    var value = Float32(0)
    if integer_end == scanned[0]:
        var normalized = String("0") + text[byte=start:]
        value = _parse_json(normalized)
    elif (
        integer_end < len(bytes)
        and bytes[integer_end] == 46
        and exponent_start == integer_end + 1
    ):
        var normalized = (
            String(text[byte=start:integer_end]) + text[byte=exponent_start:]
        )
        value = _parse_json(normalized)
    else:
        value = _parse_json(text[byte=start:])
    return -value if negative else value


def parse_csv_float32(text: StringSlice) raises -> Float32:
    """Parse a bounded CSV field at Float32 precision, preserving IEEE bits."""
    var bytes = text.as_bytes()
    var start = 0
    var end = len(bytes)
    var negative = False
    if end and (bytes[0] == 43 or bytes[0] == 45):
        negative = bytes[0] == 45
        start = 1
    if start < end and _digit(bytes[start]):
        # These CSV spellings can become JSON by borrowing a smaller span.
        # Neither a temporary String nor an expected parser failure is needed.
        while (
            start + 1 < end and bytes[start] == 48 and _digit(bytes[start + 1])
        ):
            start += 1
        if bytes[end - 1] == 46:
            for byte in bytes[start : end - 1]:
                if not _digit(byte):
                    return _adapt_csv_float32(text)
            end -= 1
        var parser = Parser(text[byte=start:end])
        try:
            var value = parser.expect_float[DType.float32]()
            if not parser.has_more():
                return -value if negative else value
        except err:
            if _overflow(err):
                # An overflowing prefix must not hide malformed trailing data.
                _ = _scan(text)
                return bitcast[DType.float32](
                    UInt32(0xFF800000) if negative else UInt32(0x7F800000)
                )
    return _adapt_csv_float32(text)


def parse_csv_float64(text: StringSlice) raises -> Float64:
    """Validate CSV syntax and retain the standard Float64 conversion policy."""
    if not _special(text)[0]:
        _ = _scan(text)
    return Float64(text)
