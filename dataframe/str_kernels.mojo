"""String kernels over code points. Nulls propagate; payloads of null rows
are never read."""
from .column import Column
from .expr import (
    Node,
    STR_LEN_CHARS,
    STR_LEN_BYTES,
    STR_UPPER,
    STR_LOWER,
    STR_STRIP,
    STR_STARTS_WITH,
    STR_ENDS_WITH,
    STR_CONTAINS,
    STR_REPLACE,
    STR_SLICE,
    STR_REVERSE,
    STR_PAD,
)
from .series import Series


def _codepoints(text: String) -> List[String]:
    var parts = List[String]()
    for codepoint in text.codepoints():
        parts.append(String(codepoint))
    return parts^


def _join(parts: List[String], start: Int, end: Int) -> String:
    var out = String()
    for i in range(start, end):
        out += parts[i]
    return out^


def _is_space(part: String) -> Bool:
    return (
        part == " "
        or part == "\t"
        or part == "\n"
        or part == "\r"
        or part == "\x0b"
        or part == "\x0c"
    )


def _strip(text: String, characters: String, mode: Int64) -> String:
    var parts = _codepoints(text)
    var remove = _codepoints(characters)
    var start = 0
    var end = len(parts)

    def strippable(part: String) capturing -> Bool:
        if len(remove) == 0:
            return _is_space(part)
        for r in remove:
            if r == part:
                return True
        return False

    if mode != 2:
        while start < end and strippable(parts[start]):
            start += 1
    if mode != 1:
        while end > start and strippable(parts[end - 1]):
            end -= 1
    return _join(parts, start, end)


def _replace(
    text: String, pattern: String, value: String, every: Bool
) -> String:
    if every:
        if pattern.byte_length() == 0:
            return text
        return text.replace(pattern, value)
    var at = text.find(pattern)
    if at < 0:
        return text
    return (
        String(text[byte=0:at])
        + value
        + String(text[byte = at + pattern.byte_length() :])
    )


def _slice(text: String, offset: Int, length: Int) -> String:
    var parts = _codepoints(text)
    var n = len(parts)
    var start = offset if offset >= 0 else max(n + offset, 0)
    start = min(start, n)
    var end = n if length < 0 else min(start + length, n)
    return _join(parts, start, end)


def _pad(text: String, fill: String, mode: Int64, width: Int) -> String:
    var parts = _codepoints(text)
    var missing = width - len(parts)
    if missing <= 0:
        return text
    var padding = fill * missing
    if mode == 1:
        return text + padding
    if mode == 2 and len(parts) > 0 and (parts[0] == "-" or parts[0] == "+"):
        return parts[0] + padding + _join(parts, 1, len(parts))
    return padding + text


def string_op(node: Node, input: Series) raises -> Series:
    ref column = input._data[Column[String]]
    var n = len(column)
    var valid = List[Bool](capacity=n)
    for i in range(n):
        valid.append(column._valid(i))
    var op = node.op
    if op == STR_LEN_CHARS or op == STR_LEN_BYTES:
        var lengths = List[Int64](length=n, fill=0)
        for i in range(n):
            if valid[i]:
                ref text = column._get(i)
                lengths[i] = Int64(
                    len(text.codepoints()) if op
                    == STR_LEN_CHARS else text.byte_length()
                )
        return Series("", Column[Int64](lengths^, valid))
    if op == STR_STARTS_WITH or op == STR_ENDS_WITH or op == STR_CONTAINS:
        var flags = List[Bool](length=n, fill=False)
        for i in range(n):
            if valid[i]:
                ref text = column._get(i)
                if op == STR_STARTS_WITH:
                    flags[i] = text.startswith(node.text)
                elif op == STR_ENDS_WITH:
                    flags[i] = text.endswith(node.text)
                else:
                    flags[i] = node.text in text
        return Series("", Column[Bool](flags^, valid))
    var values = List[String](length=n, fill="")
    for i in range(n):
        if not valid[i]:
            continue
        ref text = column._get(i)
        if op == STR_UPPER:
            values[i] = text.upper()
        elif op == STR_LOWER:
            values[i] = text.lower()
        elif op == STR_STRIP:
            values[i] = _strip(text, node.text, node.integer)
        elif op == STR_REPLACE:
            values[i] = _replace(text, node.text, node.text2, node.integer == 1)
        elif op == STR_SLICE:
            values[i] = _slice(text, Int(node.integer), node.min_count)
        elif op == STR_REVERSE:
            var parts = _codepoints(text)
            var out = String()
            for j in range(len(parts) - 1, -1, -1):
                out += parts[j]
            values[i] = out^
        elif op == STR_PAD:
            values[i] = _pad(text, node.text, node.integer, node.min_count)
        else:
            raise Error("Unsupported string operation")
    return Series("", Column[String](values^, valid))


def concat_strings(
    left: Series, right: Series, separator: String
) raises -> Series:
    ref a = left._data[Column[String]]
    ref b = right._data[Column[String]]
    if len(a) != len(b) and len(a) != 1 and len(b) != 1:
        raise Error("Incompatible expression lengths")
    var n = 0 if len(a) == 0 or len(b) == 0 else max(len(a), len(b))
    var values = List[String](length=n, fill="")
    var valid = List[Bool](length=n, fill=False)
    for i in range(n):
        var x = 0 if len(a) == 1 else i
        var y = 0 if len(b) == 1 else i
        valid[i] = a._valid(x) and b._valid(y)
        if valid[i]:
            values[i] = a._get(x) + separator + b._get(y)
    return Series("", Column[String](values^, valid))
