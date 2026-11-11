"""Trim, reverse and pad UTF-8 rows without strings for each code point."""
from dataframe.expr import Node, STR_STRIP, STR_REVERSE, STR_PAD
from dataframe.series import Series
from dataframe.text.bytes import _Bytes
from unicode_mojo.utf8 import count_codepoints
from unicode_mojo.transform import CodepointTrimSet, reverse_codepoints_into
from dataframe.storage.string import StringColumn, StringBuilder

comptime _Row = Span[UInt8, ImmutAnyOrigin]


@always_inline
def _part(row: _Row, first: Int, last: Int) -> _Row:
    return _Row(
        unsafe_ptr=row.unsafe_ptr().unsafe_offset(first), length=last - first
    )


@always_inline
def _end(mut out: StringBuilder):
    out._offsets.append(Int64(len(out._bytes)))
    out._push_bit(True)


def transform_strings[op: Int](node: Node, column: StringColumn) -> Series:
    var rows = _Bytes(column)
    var out = StringBuilder(len(column), column._value_bytes())
    # The removal set belongs to the expression, not to each input row.
    # Empty strip characters keep dataframe's established ASCII-whitespace
    # default. The shared trim set itself has no implicit whitespace policy.
    var characters = node.text if op == STR_STRIP else ""
    if not characters.byte_length():
        characters = " \t\n\r\v\f"
    var trim = CodepointTrimSet(characters)
    for i in range(len(column)):
        if not column._valid(i):
            out.append_null()
            continue
        var row = rows.get(i)
        var text = StringSpan(unsafe_from_utf8=row)
        comptime if op == STR_STRIP:
            var first, last = trim.bounds(
                text, leading=node.integer != 2, trailing=node.integer != 1
            )
            out._bytes.extend(row[first:last])
        elif op == STR_REVERSE:
            reverse_codepoints_into(text, out._bytes)
        elif op == STR_PAD:
            var chars = count_codepoints(text)
            var missing = max(node.min_count - chars, 0)
            var first = 0
            if node.integer == 1:
                out._bytes.extend(row)
            elif (
                node.integer == 2
                and len(row) > 0
                and (row[0] == 45 or row[0] == 43)
            ):
                out._bytes.append(row[0])
                first = 1
            for _ in range(missing):
                out._bytes.extend(node.text.as_bytes())
            if node.integer != 1:
                out._bytes.extend(_part(row, first, len(row)))
        _end(out)
    return Series("", out^.finish())
