"""String kernels over code points. Nulls propagate; payloads of null rows
are never read."""
from .bool_column import BoolColumn
from .column import Column
from .regex import Regex, escape
from .string_column import StringColumn, StringBuilder
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
    STR_EXTRACT,
    STR_ESCAPE_REGEX,
    STR_IS_IN,
    STR_LIKE,
    STR_REPLACE,
    STR_SLICE,
    STR_REVERSE,
    STR_PAD,
)
from .regex_kernels import (
    compile_pattern,
    is_literal_pattern,
    regex_contains,
    regex_extract,
    regex_replace,
)
from .series import Series
from .string_predicates import (
    decode_values,
    is_in_literals,
    slice_strings,
    string_contains,
    string_ends_with,
    string_like,
    string_starts_with,
)


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

    def strippable(part: String) {imm remove} -> Bool:
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


def string_op(
    node: Node, input: Series, compiled: Optional[Regex] = None
) raises -> Series:
    if input.is_chunked():
        return string_op(node, input.rechunk(), compiled)
    ref column = input._data[StringColumn]
    if node.op == STR_IS_IN:
        return is_in_literals(column, decode_values(node.text))
    # Predicates and slices over the raw bytes, on every worker (#374).
    if node.op == STR_STARTS_WITH:
        return string_starts_with(column, node.text)
    if node.op == STR_ENDS_WITH:
        return string_ends_with(column, node.text)
    if node.op == STR_CONTAINS:
        if node.integer == 0:
            return string_contains(column, node.text)
        var regex: Regex
        try:
            regex = compiled.value().copy() if compiled else compile_pattern(
                node.text, False
            )
        except e:
            if node.integer == 1:
                raise e^
            # strict=False: an invalid pattern gives null.
            var n = len(column)
            return Series(
                "",
                BoolColumn(
                    List[Bool](length=n, fill=False),
                    List[Bool](length=n, fill=False),
                ),
            )
        return regex_contains(column, regex)
    if node.op == STR_EXTRACT:
        return regex_extract(
            column,
            compiled.value().copy() if compiled else compile_pattern(
                node.text, False
            ),
            Int(node.integer),
        )
    if node.op == STR_REPLACE:
        # Polars replaces a pattern without ASCII punctuation literally, and
        # then does not expand `$` references in the value.
        var literal = node.min_count == 1 or is_literal_pattern(node.text)
        var limit = Int(node.integer)
        if not literal and limit == 0:
            # Polars' regex path ignores n=0 and replaces the first match.
            limit = 1
        var regex = compiled.value().copy() if compiled else compile_pattern(
            node.text, literal
        )
        return regex_replace(
            column, regex, regex.template(node.text2, not literal), limit
        )
    if node.op == STR_LIKE:
        return string_like(column, node.text)
    if node.op == STR_SLICE and (
        column._is_view_storage() or len(column._bytes[]) <= 4_294_967_295
    ):
        return Series(
            "", slice_strings(column, Int(node.integer), node.min_count)
        )
    var n = len(column)
    var valid = List[Bool](capacity=n)
    for i in range(n):
        valid.append(column._valid(i))
    var op = node.op
    if op == STR_LEN_CHARS or op == STR_LEN_BYTES:
        var lengths = List[Int64](length=n, fill=0)
        for i in range(n):
            if valid[i]:
                var text = column._get(i)
                lengths[i] = Int64(
                    len(text.codepoints()) if op
                    == STR_LEN_CHARS else text.byte_length()
                )
        return Series("", Column[Int64](lengths^, valid))
    if op == STR_STARTS_WITH or op == STR_ENDS_WITH or op == STR_CONTAINS:
        var flags = List[Bool](length=n, fill=False)
        for i in range(n):
            if valid[i]:
                var text = column._get(i)
                if op == STR_STARTS_WITH:
                    flags[i] = text.startswith(node.text)
                elif op == STR_ENDS_WITH:
                    flags[i] = text.endswith(node.text)
                else:
                    flags[i] = node.text in text
        return Series("", BoolColumn(flags, valid))
    var out = StringBuilder(n, column._value_bytes())
    for i in range(n):
        if not valid[i]:
            out.append_null()
            continue
        var text = String(column._get(i))
        if op == STR_ESCAPE_REGEX:
            out.append(escape(text))
        elif op == STR_UPPER:
            out.append(text.upper())
        elif op == STR_LOWER:
            out.append(text.lower())
        elif op == STR_STRIP:
            out.append(_strip(text, node.text, node.integer))
        elif op == STR_SLICE:
            out.append(_slice(text, Int(node.integer), node.min_count))
        elif op == STR_REVERSE:
            var parts = _codepoints(text)
            var reversed = String()
            for j in range(len(parts) - 1, -1, -1):
                reversed += parts[j]
            out.append(reversed)
        elif op == STR_PAD:
            out.append(_pad(text, node.text, node.integer, node.min_count))
        else:
            raise Error("Unsupported string operation")
    return Series("", out^.finish())


def concat_strings(
    left: Series, right: Series, separator: String, skip_nulls: Bool = False
) raises -> Series:
    """Rows joined with the separator; a null side makes the row null, or
    with skip_nulls is left out (null only when both sides are)."""
    if left.is_chunked() or right.is_chunked():
        return concat_strings(
            left.rechunk(), right.rechunk(), separator, skip_nulls
        )
    ref a = left._data[StringColumn]
    ref b = right._data[StringColumn]
    if len(a) != len(b) and len(a) != 1 and len(b) != 1:
        raise Error("Incompatible expression lengths")
    var n = 0 if len(a) == 0 or len(b) == 0 else max(len(a), len(b))
    var out = StringBuilder(n)
    for i in range(n):
        var x = 0 if len(a) == 1 else i
        var y = 0 if len(b) == 1 else i
        if a._valid(x) and b._valid(y):
            out.append(String(a._get(x)) + separator + String(b._get(y)))
        elif skip_nulls and a._valid(x):
            out.append(a._get(x))
        elif skip_nulls and b._valid(y):
            out.append(b._get(y))
        else:
            out.append_null()
    return Series("", out^.finish())
