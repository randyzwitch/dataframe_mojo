"""JSON numeric validation and distinct decode/JSONPath text formatting."""
from std.memory import bitcast
from dataframe.json.reader import JsonReader
from dataframe.text.parse import parse_float64, parse_integer
from dataframe.json.path_number import parse_jsonpath_float


@always_inline
def is_json_number(kind: Int) -> Bool:
    return kind == 45 or (kind >= 48 and kind <= 57)


def is_json_fraction(text: StringSlice) -> Bool:
    for byte in text.as_bytes():
        if byte == 46 or byte == 101 or byte == 69:
            return True
    return False


def _wide_integer(text: StringSlice) -> Bool:
    var data = text.as_bytes()
    var negative = data[0] == 45
    var at = Int(negative)
    var limit = StringSlice(
        "170141183460469231731687303715884105728"
    ) if negative else StringSlice("340282366920938463463374607431768211455")
    var digits = len(data) - at
    if digits != limit.byte_length():
        return digits < limit.byte_length()
    for i in range(digits):
        if data[at + i] != limit.as_bytes()[i]:
            return data[at + i] < limit.as_bytes()[i]
    return True


def validate_json_numbers(reader: JsonReader, wide: Bool = True) raises:
    for i in range(len(reader.nodes)):
        if not is_json_number(reader.nodes[i].kind):
            continue
        var raw = reader.raw(i)
        var fraction = is_json_fraction(raw)
        if wide and not fraction and not _wide_integer(raw):
            raise Error("Invalid JSON integer outside 128-bit range")
        if fraction or (not wide and raw.byte_length() > 20):
            try:
                var value = parse_float64(
                    raw
                ) if wide else parse_jsonpath_float(raw)
                if value != value or abs(value) == Float64("inf"):
                    raise Error("nonfinite value")
            except e:
                raise Error("Invalid JSON number: " + String(e))


def float_text(value: Float64, json_style: Bool) -> String:
    var negative = (bitcast[DType.uint64](value) >> 63) != 0
    if value == 0:
        return ("-" if negative else "") + ("0.0" if json_style else "0")
    var raw = String(abs(value))
    var exponent = 0
    var exp_at = raw.find("e")
    if exp_at < 0:
        exp_at = raw.find("E")
    var end = raw.byte_length() if exp_at < 0 else exp_at
    if exp_at >= 0:
        var at = exp_at + 1
        var sign = 1
        if raw.as_bytes()[at] == 43 or raw.as_bytes()[at] == 45:
            sign = -1 if raw.as_bytes()[at] == 45 else 1
            at += 1
        while at < raw.byte_length():
            exponent = exponent * 10 + Int(raw.as_bytes()[at]) - 48
            at += 1
        exponent *= sign
    var digits = List[UInt8](capacity=end)
    var point = 0
    var saw_dot = False
    for at in range(end):
        var byte = raw.as_bytes()[at]
        if byte == 46:
            saw_dot = True
        else:
            digits.append(byte)
            if not saw_dot:
                point += 1
    point += exponent
    var start = 0
    while start + 1 < len(digits) and digits[start] == 48:
        start += 1
        point -= 1
    var last = len(digits)
    while last > start + 1 and digits[last - 1] == 48:
        last -= 1
    var power = point - 1
    var out = String("-" if negative else "")
    if json_style and (power >= 16 or power <= -6):
        out += chr(Int(digits[start]))
        if last > start + 1:
            out += "."
            for i in range(start + 1, last):
                out += chr(Int(digits[i]))
        out += "e" + ("+" if power >= 0 else "") + String(power)
    elif point <= 0:
        out += "0."
        for _ in range(-point):
            out += "0"
        for i in range(start, last):
            out += chr(Int(digits[i]))
    else:
        for i in range(max(point, last - start)):
            if i == point:
                out += "."
            out += chr(Int(digits[start + i])) if start + i < last else String(
                "0"
            )
        if point >= last - start and json_style:
            out += ".0"
    return out^


def json_number_text(
    reader: JsonReader, value: Int, wide: Bool = True
) raises -> String:
    var text = reader.raw(value)
    if not is_json_fraction(text):
        if wide:
            if not _wide_integer(text):
                raise Error("Invalid JSON integer outside 128-bit range")
            return String("0") if text == "-0" else String(text)
        if text != "-0":
            try:
                if text.as_bytes()[0] == 45:
                    _ = parse_integer[DType.int64](text)
                else:
                    _ = parse_integer[DType.uint64](text)
                return String(text)
            except:
                pass
    return float_text(
        parse_float64(text) if wide else parse_jsonpath_float(text), not wide
    )
