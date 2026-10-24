"""Trim, reverse and pad UTF-8 rows without strings for each code point."""
from .expr import Node, STR_STRIP, STR_REVERSE, STR_PAD
from .series import Series
from .string_bytes import _Bytes, _same_bytes
from .string_column import StringColumn, StringBuilder

comptime _Row = Span[UInt8, ImmutAnyOrigin]


@always_inline
def _part(row: _Row, first: Int, last: Int) -> _Row:
    return _Row(
        unsafe_ptr=row.unsafe_ptr().unsafe_offset(first), length=last - first
    )


@always_inline
def _next(row: _Row, first: Int) -> Int:
    var end = first + 1
    while (
        end < len(row) and (row.unsafe_ptr().unsafe_offset(end)[] & 192) == 128
    ):
        end += 1
    return end


@always_inline
def _previous(row: _Row, last: Int) -> Int:
    var first = last - 1
    while first > 0 and (row.unsafe_ptr().unsafe_offset(first)[] & 192) == 128:
        first -= 1
    return first


struct _TrimSet(Movable):
    var ascii: List[Bool]
    var unicode: List[String]

    def __init__(out self, characters: String):
        self.ascii = List[Bool](length=128, fill=False)
        self.unicode = List[String]()
        if characters.byte_length() == 0:
            for c in List[Int]([32, 9, 10, 13, 11, 12]):
                self.ascii[c] = True
        else:
            for cp in characters.codepoints():
                var s = String(cp)
                if s.byte_length() == 1:
                    self.ascii[Int(s.as_bytes()[0])] = True
                else:
                    self.unicode.append(s^)

    @always_inline
    def contains(self, row: _Row, first: Int, last: Int) -> Bool:
        if last - first == 1:
            return self.ascii[Int(row.unsafe_ptr().unsafe_offset(first)[])]
        var part = _part(row, first, last)
        for text in self.unicode:
            var bytes = _Row(
                unsafe_ptr=text.unsafe_ptr()
                .unsafe_mut_cast[False]()
                .unsafe_origin_cast[ImmutAnyOrigin](),
                length=text.byte_length(),
            )
            if _same_bytes(part, bytes):
                return True
        return False


@always_inline
def _end(mut out: StringBuilder):
    out._offsets.append(Int64(len(out._bytes)))
    out._push_bit(True)


def transform_strings[op: Int](node: Node, column: StringColumn) -> Series:
    var rows = _Bytes(column)
    var out = StringBuilder(len(column), column._value_bytes())
    # The removal set belongs to the expression, not to each input row.
    var trim = _TrimSet(node.text if op == STR_STRIP else "")
    for i in range(len(column)):
        if not column._valid(i):
            out.append_null()
            continue
        var row = rows.get(i)
        comptime if op == STR_STRIP:
            var first = 0
            var last = len(row)
            if node.integer != 2:
                while first < last:
                    var end = _next(row, first)
                    if not trim.contains(row, first, end):
                        break
                    first = end
            if node.integer != 1:
                while last > first:
                    var start = _previous(row, last)
                    if not trim.contains(row, start, last):
                        break
                    last = start
            out._bytes.extend(_part(row, first, last))
        elif op == STR_REVERSE:
            var last = len(row)
            while last > 0:
                var first = _previous(row, last)
                out._bytes.extend(_part(row, first, last))
                last = first
        elif op == STR_PAD:
            var chars = 0
            for c in row:
                chars += Int((c & 192) != 128)
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
