"""Checked Int128 decimal grammar and a separate libc floating consumer."""
from std.math import isinf
from tests.support.numeric_model import integer_bounds
from tests.support.csv_reference import csv_float


def integer_reference[D: DType](text: StringSlice) -> Tuple[Int128, String]:
    var bytes = text.as_bytes()
    if len(bytes) == 0:
        return (Int128(0), "empty integer")
    var start = 0
    var negative = False
    if bytes[0] == 43 or bytes[0] == 45:
        negative = bytes[0] == 45
        start = 1
    if start == len(bytes):
        return (Int128(0), "sign without digits")
    var bounds = integer_bounds[D]()
    var limit = -bounds[0] if negative else bounds[1]
    var magnitude = Int128(0)
    for index in range(start, len(bytes)):
        var byte = bytes[index]
        if byte < 48 or byte > 57:
            return (Int128(0), "non-decimal integer byte")
        # The previous magnitude is <=UInt64.MAX, hence this temporary
        # fits Int128 even for arbitrarily long input rejected at this digit.
        var next = 10 * magnitude + Int128(byte - 48)
        if next > limit:
            return (Int128(0), "overflow")
        magnitude = next
    return (-magnitude if negative else magnitude, String())


def float_grammar(text: String) -> Bool:
    if text == "nan" or text == "NaN":
        return True
    var b = text.as_bytes()
    var position = 0
    if len(b) and (b[0] == 43 or b[0] == 45):
        position = 1
    var suffix = String(text[byte=position:])
    if suffix == "inf" or suffix == "Infinity":
        return True
    var digits = 0
    var exponent_digits = 0
    var dot = False
    var exponent = False
    while position < len(b):
        var byte = b[position]
        if byte >= 48 and byte <= 57:
            if exponent:
                exponent_digits += 1
            else:
                digits += 1
        elif byte == 46 and not dot and not exponent:
            dot = True
        elif (byte == 101 or byte == 69) and not exponent and digits > 0:
            exponent = True
            if position + 1 < len(b) and (
                b[position + 1] == 43 or b[position + 1] == 45
            ):
                position += 1
        else:
            return False
        position += 1
    return digits > 0 and (not exponent or exponent_digits > 0)


def float_reference(text: String) raises -> Float64:
    if not float_grammar(text):
        raise Error("Reference rejects strict float grammar")
    # csv_float copies just this field to a terminated byte vector, retains
    # it throughout strtod, and verifies endptr consumed every field byte.
    var result = csv_float(text)
    if isinf(result) and not (
        text.endswith("inf") or text.endswith("Infinity")
    ):
        raise Error("Reference float range overflow")
    return result
