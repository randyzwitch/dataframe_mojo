"""Fixed-point decimal128 parsing, formatting, and arithmetic helpers."""
from .dtype import DataType


def pow10(scale: Int) -> Int128:
    var value = Int128(1)
    for _ in range(scale):
        value *= 10
    return value


def precision_limit(precision: Int) -> Int128:
    return pow10(precision)


def check_precision(value: Int128, dtype: DataType) raises -> Int128:
    var limit = precision_limit(dtype.precision())
    if value <= -limit or value >= limit:
        raise Error(
            "value exceeds decimal precision " + String(dtype.precision())
        )
    return value


def parse_decimal(text: StringSlice, dtype: DataType) raises -> Int128:
    """Parse plain decimal text exactly at dtype.scale(); excess digits fail."""
    var bytes = text.as_bytes()
    if len(bytes) == 0:
        raise Error("empty decimal value")
    var i = 0
    var negative = False
    if bytes[0] == 45 or bytes[0] == 43:
        negative = bytes[0] == 45
        i = 1
    var value = Int128(0)
    var digits = 0
    var fraction = -1
    while i < len(bytes):
        var byte = bytes[i]
        if byte == 46 and fraction < 0:
            fraction = 0
            i += 1
            continue
        if byte < 48 or byte > 57:
            raise Error("invalid decimal value '" + String(text) + "'")
        if fraction >= 0:
            fraction += 1
            if fraction > dtype.scale():
                raise Error(
                    "decimal has more than "
                    + String(dtype.scale())
                    + " fractional digits"
                )
        value = value * 10 + Int128(byte - 48)
        digits += 1
        i += 1
    if digits == 0:
        raise Error("invalid decimal value '" + String(text) + "'")
    var seen_scale = max(fraction, 0)
    value *= pow10(dtype.scale() - seen_scale)
    if negative:
        value = -value
    return check_precision(value, dtype)


def format_decimal(value: Int128, scale: Int) -> String:
    var negative = value < 0
    var magnitude = -value if negative else value
    var digits = String(magnitude)
    if scale == 0:
        return ("-" if negative else "") + digits
    while digits.byte_length() <= scale:
        digits = "0" + digits
    var split = digits.byte_length() - scale
    return (
        ("-" if negative else "")
        + String(digits[byte=0:split])
        + "."
        + String(digits[byte = split : digits.byte_length()])
    )
