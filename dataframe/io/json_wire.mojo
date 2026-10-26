"""Lossless JSON wire forms for durations and opaque byte columns."""
from ..dtype import DataType
from ..parse import parse_integer


def duration_text(value: Int64, dtype: DataType) -> String:
    """ISO 8601 seconds, using unsigned magnitude even for Int64.MIN."""
    var negative = value < 0
    var magnitude = UInt64(-(value + 1)) + 1 if negative else UInt64(value)
    var per_second = UInt64(dtype.per_second())
    var out = String("-PT" if negative else "PT")
    out += String(magnitude // per_second)
    var fraction = magnitude % per_second
    if fraction:
        var width = 9 if per_second == 1000000000 else (
            6 if per_second == 1000000 else 3
        )
        var digits = String(fraction)
        digits = String("0") * (width - digits.byte_length()) + digits
        var end = digits.byte_length()
        while end > 0 and digits.as_bytes()[end - 1] == 48:
            end -= 1
        out += "." + String(digits[byte=0:end])
    return out + "S"


def parse_duration(text: String, dtype: DataType) raises -> Int64:
    """Fixed ISO days/hours/minutes/seconds; calendar months are not ticks."""
    var bytes = text.as_bytes()
    var at = 0
    var negative = len(bytes) > 0 and bytes[0] == 45
    if negative or (len(bytes) > 0 and bytes[0] == 43):
        at += 1
    if at >= len(bytes) or bytes[at] != 80:
        raise Error("Expected ISO 8601 duration")
    at += 1
    var time = False
    var seen = False
    var time_value = False
    var previous = 0
    var nanos = Int128(0)
    while at < len(bytes):
        if bytes[at] == 84:
            if time:
                raise Error("Duplicate duration time separator")
            time = True
            at += 1
            continue
        var start = at
        while at < len(bytes) and bytes[at] >= 48 and bytes[at] <= 57:
            at += 1
        if at == start:
            raise Error("Expected duration digits")
        var whole = parse_integer[DType.uint64](text[byte=start:at])
        var fraction = Int128(0)
        var digits = 0
        if at < len(bytes) and bytes[at] == 46:
            at += 1
            while at < len(bytes) and bytes[at] >= 48 and bytes[at] <= 57:
                if digits >= 9:
                    raise Error("Duration precision exceeds nanoseconds")
                fraction = fraction * 10 + Int128(bytes[at] - 48)
                digits += 1
                at += 1
            if digits == 0:
                raise Error("Expected duration fraction")
        if at >= len(bytes):
            raise Error("Expected duration unit")
        var unit = bytes[at]
        at += 1
        var order: Int
        var seconds: Int128
        if not time and unit == 68:
            order = 1
            seconds = 86400
        elif time and unit == 72:
            order = 2
            seconds = 3600
        elif time and unit == 77:
            order = 3
            seconds = 60
        elif time and unit == 83:
            order = 4
            seconds = 1
        else:
            raise Error("Unsupported duration unit")
        if order <= previous or (digits > 0 and order != 4):
            raise Error("Invalid duration component order or fraction")
        previous = order
        while digits < 9:
            fraction *= 10
            digits += 1
        nanos += Int128(whole) * seconds * 1000000000 + fraction
        seen = True
        time_value = time_value or time
    if not seen or (time and not time_value):
        raise Error("Empty ISO duration")
    var divisor = Int128(1000000000 // dtype.per_second())
    if nanos % divisor != 0:
        raise Error("Duration cannot be represented in the requested unit")
    var ticks = nanos // divisor
    if negative:
        ticks = -ticks
    if ticks < -(Int128(1) << 63) or ticks >= (Int128(1) << 63):
        raise Error("Duration is outside Int64")
    return ticks.cast[DType.int64]()
