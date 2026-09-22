from dataframe.parse import parse_float64


comptime F32_EXACT_MANTISSA = UInt64(16777216)


@always_inline
def _pow10_f32(k: Int) -> Float32:
    if k == 0:
        return 1.0
    if k == 1:
        return 10.0
    if k == 2:
        return 100.0
    if k == 3:
        return 1000.0
    if k == 4:
        return 10000.0
    if k == 5:
        return 100000.0
    if k == 6:
        return 1000000.0
    return 10000000.0


def parse_float32_plain(text: StringSlice) raises -> Float32:
    var b = text.as_bytes()
    var n = len(b)
    var mantissa = UInt64(0)
    var digits = 0
    var fraction = -1
    var i = 0
    var negative = False
    if n > 0 and (b[0] == 43 or b[0] == 45):
        negative = b[0] == 45
        i = 1
    while i < n:
        var c = b[i]
        if c >= 48 and c <= 57:
            if digits == 19:
                return Float32(parse_float64(text))
            mantissa = mantissa * 10 + UInt64(c - 48)
            digits += 1
            if fraction >= 0:
                fraction += 1
        elif c == 46 and fraction < 0:
            fraction = 0
        else:
            return Float32(parse_float64(text))
        i += 1
    if (
        digits == 0
        or fraction == 0
        or fraction > 7
        or mantissa > F32_EXACT_MANTISSA
    ):
        return Float32(parse_float64(text))
    var value = Float32(mantissa)
    if fraction > 0:
        value = value / _pow10_f32(fraction)
    return -value if negative else value
