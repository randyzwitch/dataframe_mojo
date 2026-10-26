"""String functions that need no regular expressions (#798):
to_titlecase, strip_prefix/strip_suffix, find, count_matches,
contains_any, replace_many, encode/decode, to_integer and split_exact/splitn.
Nulls propagate; payloads of null rows are never read.

Rows are read through `_Bytes`, which takes the storage's pointers once, for
both offsets and view storage. Results that are part of a row (a stripped
affix, a split field) are views into the source's buffers, so no row's bytes
are copied, as `slice_strings` does; other string results are appended to one
`StringBuilder` without a `String` per row (only `to_titlecase` allocates, for
each code point past ASCII). `contains_any` and `replace_many` match every
pattern in one pass with an Aho-Corasick automaton over byte classes.
"""
from ._buffer import _Buffer
from std.memory import ArcPointer, Pointer
from .bool_column import BoolColumn
from .column import Column, _copy_validity
from .expr import (
    Node,
    STR_TITLECASE,
    STR_STRIP_PREFIX,
    STR_STRIP_SUFFIX,
    STR_FIND,
    STR_COUNT_MATCHES,
    STR_CONTAINS_ANY,
    STR_REPLACE_MANY,
    STR_ENCODE,
    STR_DECODE,
    STR_TO_INTEGER,
    STR_SPLIT_FIELDS,
    STR_NORMALIZE,
    STR_JSON_DECODE,
    STR_JSON_PATH,
)
from .dtype import DataType
from .nested_column import StructColumn
from .regex import Regex
from .regex_kernels import compile_pattern, regex_count, regex_find
from .series import Series
from .string_bytes import _Bytes, _find_bytes, _same_bytes
from .string_column import StringColumn, StringBuilder
from .string_predicates import decode_values
from .string_view import StringView, StringViewStorage
from .json_decode import decode_json
from .json_strings import json_path_match
from .unicode_normalization import normalize_series
from .unicode_alphabetic import alphabetic_ranges, is_alphabetic

comptime _Row = Span[UInt8, ImmutAnyOrigin]


def text_function(
    node: Node, input: Series, compiled: Optional[Regex] = None
) raises -> Series:
    """Evaluate one text_functions operation on a String series."""
    if node.op == STR_JSON_PATH:
        return json_path_match(input, node.text)
    if node.op == STR_JSON_DECODE:
        var dtype = node.dtypes[0].value()
        # Like CAST, a root categorical is carried as strings inside the
        # evaluator and encoded once after the batches have been assembled.
        return decode_json(
            input, DataType.STRING if dtype.is_categorical() else dtype
        )
    if node.op == STR_NORMALIZE:
        return normalize_series(input, node.text)
    if input.is_chunked():
        return text_function(node, input.rechunk(), compiled)
    ref column = input._data[StringColumn]
    var op = node.op
    if op == STR_STRIP_PREFIX or op == STR_STRIP_SUFFIX:
        return Series(
            "", _strip_affix(column, node.text, op == STR_STRIP_SUFFIX)
        )
    if op == STR_SPLIT_FIELDS:
        return _split_fields(
            column,
            node.text,
            node.min_count,
            (node.integer & 1) != 0,
            (node.integer & 2) != 0,
        )
    if op == STR_FIND or op == STR_COUNT_MATCHES:
        var pattern = node.text
        if node.integer == 0:
            # A regular expression (#219), unless it only matches one
            # literal string.
            var regex = (
                compiled.value().copy() if compiled else compile_pattern(
                    node.text, False
                )
            )
            var literal = regex.literal()
            if not literal:
                if op == STR_FIND:
                    return regex_find(column, regex)
                return regex_count(column, regex)
            pattern = literal.value()
        return _find_or_count(column, pattern, op == STR_COUNT_MATCHES)
    if op == STR_CONTAINS_ANY or op == STR_REPLACE_MANY:
        var automaton = _Automaton(
            decode_values(node.text),
            (node.integer & 1) != 0,
            (node.integer & 2) != 0,
        )
        if op == STR_CONTAINS_ANY:
            return _contains_any(column, automaton)
        return _replace_many(column, automaton, decode_values(node.text2))
    if op == STR_ENCODE:
        return _encode(column, node.text == "hex")
    if op == STR_DECODE:
        return _decode(column, node.text == "hex", node.integer == 1)
    if op == STR_TO_INTEGER:
        return _to_integer(column, node.min_count, node.integer == 1)
    if op == STR_TITLECASE:
        return _titlecase(column)
    raise Error("Unsupported string operation")


@always_inline
def _literal(text: String) -> _Row:
    return _Row(
        unsafe_ptr=text.unsafe_ptr()
        .unsafe_mut_cast[False]()
        .unsafe_origin_cast[ImmutAnyOrigin](),
        length=text.byte_length(),
    )


@always_inline
def _clear_bit(mut bits: List[UInt8], row: Int, n: Int):
    """Mark row null in a bitmap that may still be empty (all valid)."""
    if len(bits) == 0:
        bits = List[UInt8](length=(n + 7) // 8, fill=255)
    bits[row // 8] &= ~(UInt8(1) << UInt8(row % 8))


@always_inline
def _end_row(mut out: StringBuilder):
    """Close the row whose bytes were appended to `out._bytes`."""
    out._offsets.append(Int64(len(out._bytes)))
    out._push_bit(True)


# Parts of rows ------------------------------------------------------------


trait _Parts(Copyable, Movable):
    """Rows that are each a byte range of the source row of the same index,
    written in row order."""

    def put(mut self, row: _Row, b0: Int, b1: Int, buffer: UInt64, base: Int):
        ...

    def put_null(mut self):
        ...

    def finish(deinit self) -> StringColumn:
        ...


struct _ViewParts(_Parts):
    """Parts as Arrow views: up to 12 bytes inline, a longer part pointing
    into the source buffer that holds its row (`buffer`, row start `base`)."""

    var views: List[StringView]
    var buffers: List[_Buffer[UInt8]]
    var bits: List[UInt8]
    var n: Int
    var k: Int
    var total: Int

    def __init__(out self, column: StringColumn, var bits: List[UInt8]):
        self.n = len(column)
        self.views = List[StringView](unsafe_uninit_length=self.n)
        self.buffers = List[_Buffer[UInt8]]()
        if column._is_view_storage():
            for buffer in column._view_storage_unchecked()._buffers[]:
                self.buffers.append(buffer.copy())
        else:
            self.buffers.append(column._bytes.copy())
        self.bits = bits^
        self.k = 0
        self.total = 0

    @always_inline
    def put(mut self, row: _Row, b0: Int, b1: Int, buffer: UInt64, base: Int):
        var words = self.views.unsafe_ptr().unsafe_bitcast[UInt64]()
        var part = b1 - b0
        var low = UInt64(part)
        var high = UInt64(0)
        var p = row.unsafe_ptr().unsafe_offset(b0)
        if part <= 12:
            for b in range(min(part, 4)):
                low |= UInt64(p.unsafe_offset(b)[]) << UInt64(32 + 8 * b)
            for b in range(4, part):
                high |= UInt64(p.unsafe_offset(b)[]) << UInt64(8 * (b - 4))
        else:
            # Longer than 12 bytes, so the source row was too: it lives in
            # a buffer, at `base`.
            for b in range(4):
                low |= UInt64(p.unsafe_offset(b)[]) << UInt64(32 + 8 * b)
            high = buffer | (UInt64(base + b0) << 32)
            self.total += part
        words.unsafe_offset(2 * self.k)[] = low
        words.unsafe_offset(2 * self.k + 1)[] = high
        self.k += 1

    @always_inline
    def put_null(mut self):
        var words = self.views.unsafe_ptr().unsafe_bitcast[UInt64]()
        words.unsafe_offset(2 * self.k)[] = 0
        words.unsafe_offset(2 * self.k + 1)[] = 0
        _clear_bit(self.bits, self.k, self.n)
        self.k += 1

    def finish(deinit self) -> StringColumn:
        return StringColumn(
            StringViewStorage(
                self.views^, self.buffers^, self.bits^, self.n, self.total
            )
        )


struct _CopiedParts(_Parts):
    """Parts copied into new offsets storage, for a source whose single
    byte buffer is past what a view's 32-bit offset reaches."""

    var out: StringBuilder

    def __init__(out self, column: StringColumn):
        self.out = StringBuilder(len(column), column._value_bytes())

    @always_inline
    def put(mut self, row: _Row, b0: Int, b1: Int, buffer: UInt64, base: Int):
        self.out._bytes.extend(
            _Row(unsafe_ptr=row.unsafe_ptr().unsafe_offset(b0), length=b1 - b0)
        )
        _end_row(self.out)

    @always_inline
    def put_null(mut self):
        self.out.append_null()

    def finish(deinit self) -> StringColumn:
        return self.out^.finish()


@always_inline
def _views_fit(column: StringColumn) -> Bool:
    return column._is_view_storage() or len(column._bytes[]) <= 4_294_967_295


@always_inline
def _origin(rows: _Bytes, i: Int) -> Tuple[UInt64, Int]:
    """The buffer holding row i and the row's first byte in it."""
    if rows.views != 0:
        ref view = Pointer[StringView, ImmutAnyOrigin](
            unsafe_from_address=rows.views
        ).unsafe_offset(i)[]
        return (UInt64(view.buffer_index), Int(view.offset))
    var offsets = Pointer[Int64, ImmutAnyOrigin](
        unsafe_from_address=rows.offsets
    )
    return (UInt64(0), Int(offsets.unsafe_offset(i)[]))


def _strip_affix(
    column: StringColumn, affix: String, suffix: Bool
) raises -> StringColumn:
    if _views_fit(column):
        var parts = _ViewParts(
            column, _copy_validity(column._bits[], column._offset, len(column))
        )
        _strip_into(column, affix, suffix, parts)
        return parts^.finish()
    var parts = _CopiedParts(column)
    _strip_into(column, affix, suffix, parts)
    return parts^.finish()


def _strip_into[
    P: _Parts
](column: StringColumn, affix: String, suffix: Bool, mut parts: P):
    var rows = _Bytes(column)
    var nulls = column.null_count() > 0
    var wanted = _literal(affix)
    var m = len(wanted)
    for i in range(len(column)):
        if nulls and not column._valid(i):
            parts.put_null()
            continue
        var row = rows.get(i)
        var b0 = 0
        var b1 = len(row)
        if b1 >= m:
            var at = b1 - m if suffix else 0
            if _same_bytes(
                _Row(unsafe_ptr=row.unsafe_ptr().unsafe_offset(at), length=m),
                wanted,
            ):
                if suffix:
                    b1 = at
                else:
                    b0 = m
        var origin = _origin(rows, i)
        parts.put(row, b0, b1, origin[0], origin[1])


@always_inline
def _codepoint_value[
    origin: Origin
](data: Pointer[UInt8, origin], at: Int, width: Int) -> UInt32:
    """The code point whose UTF-8 encoding of `width` bytes starts at `at`."""
    var lead = UInt32(data[unsafe_offset=at])
    if width == 1:
        return lead
    var value = lead & (UInt32(0x7F) >> UInt32(width))
    for b in range(1, width):
        value = (value << 6) | (UInt32(data[unsafe_offset=at + b]) & 0x3F)
    return value


@always_inline
def _codepoint_width(lead: UInt8) -> Int:
    if lead < 0x80:
        return 1
    if lead < 0xE0:
        return 2
    if lead < 0xF0:
        return 3
    return 4


def _split_fields(
    column: StringColumn,
    by: String,
    count: Int,
    inclusive: Bool,
    remainder: Bool,
) raises -> Series:
    """`split_exact` (count = n + 1 fields) or `splitn` (`remainder`: the
    last field keeps the rest of the row), as a struct of String fields."""
    var fields = List[Series](capacity=count)
    if _views_fit(column):
        var parts = List[_ViewParts](capacity=count)
        var valid = _copy_validity(column._bits[], column._offset, len(column))
        for _ in range(count):
            parts.append(_ViewParts(column, valid.copy()))
        _split_into(column, by, inclusive, remainder, parts)
        for k in range(count):
            var part = parts.pop(0)
            fields.append(Series("field_" + String(k), part^.finish()))
    else:
        var parts = List[_CopiedParts](capacity=count)
        for _ in range(count):
            parts.append(_CopiedParts(column))
        _split_into(column, by, inclusive, remainder, parts)
        for k in range(count):
            var part = parts.pop(0)
            fields.append(Series("field_" + String(k), part^.finish()))
    return Series("", StructColumn(fields^))


def _split_into[
    P: _Parts
](
    column: StringColumn,
    by: String,
    inclusive: Bool,
    remainder: Bool,
    mut parts: List[P],
):
    """Each row's parts into the fields in order; an empty separator takes
    one code point per field. Fields past the row's
    parts are null."""
    var rows = _Bytes(column)
    var nulls = column.null_count() > 0
    var separator = _literal(by)
    var m = len(separator)
    var count = len(parts)
    for i in range(len(column)):
        if nulls and not column._valid(i):
            for k in range(count):
                parts[k].put_null()
            continue
        var row = rows.get(i)
        var size = len(row)
        var origin = _origin(rows, i)
        var k = 0
        var at = 0
        if m == 0:
            var singles = count - 1 if remainder else count
            while k < singles and at < size:
                var width = _codepoint_width(row.unsafe_ptr()[unsafe_offset=at])
                parts[k].put(row, at, at + width, origin[0], origin[1])
                at += width
                k += 1
            if remainder and k == count - 1 and at < size:
                parts[k].put(row, at, size, origin[0], origin[1])
                k += 1
        else:
            while k < count:
                if remainder and k == count - 1:
                    parts[k].put(row, at, size, origin[0], origin[1])
                    k += 1
                    break
                var found = _find_bytes(row, separator, at)
                if found < 0:
                    # split_inclusive leaves out an empty tail.
                    if not inclusive or at < size:
                        parts[k].put(row, at, size, origin[0], origin[1])
                        k += 1
                    break
                var end = found + m if inclusive else found
                parts[k].put(row, at, end, origin[0], origin[1])
                k += 1
                at = found + m
        while k < count:
            parts[k].put_null()
            k += 1


# Searches -----------------------------------------------------------------


def _find_or_count(
    column: StringColumn, pattern: String, count: Bool
) raises -> Series:
    """`find`: the byte offset of the first occurrence, null when there is
    none. `count_matches`: non-overlapping occurrences; an empty pattern
    matches at every code point boundary, as a regex engine does."""
    var n = len(column)
    var rows = _Bytes(column)
    var nulls = column.null_count() > 0
    var needle = _literal(pattern)
    var values = List[UInt32](length=n, fill=0)
    var bits = _copy_validity(column._bits[], column._offset, n)
    for i in range(n):
        if nulls and not column._valid(i):
            continue
        var row = rows.get(i)
        if not count:
            var found = _find_bytes(row, needle, 0)
            if found < 0:
                _clear_bit(bits, i, n)
            else:
                values[i] = UInt32(found)
            continue
        var matches = 0
        if len(needle) == 0:
            matches = 1
            for b in range(len(row)):
                if (row.unsafe_ptr()[unsafe_offset=b] & 0xC0) != 0x80:
                    matches += 1
        else:
            var at = _find_bytes(row, needle, 0)
            while at >= 0:
                matches += 1
                at = _find_bytes(row, needle, at + len(needle))
        values[i] = UInt32(matches)
    return Series("", Column[UInt32](values=values^, bits=bits^))


struct _Match(Copyable):
    var start: Int
    var end: Int
    var pattern: Int

    def __init__(out self, start: Int, end: Int, pattern: Int):
        self.start = start
        self.end = end
        self.pattern = pattern


struct _Automaton(Movable):
    """Aho-Corasick over byte classes: a trie of the patterns whose missing
    transitions follow the failure links, completed into a table of
    `states x classes` next states. Bytes that occur in no pattern share one
    class, so the table stays small; with ASCII case folding an upper-case
    byte takes its lower-case byte's class.

    Standard semantics: scanning from a position, the first state that
    ends a pattern reports the longest pattern ending there. Leftmost-first
    (`leftmost`): the match that starts first, the
    earlier pattern among those starting at one position."""

    var patterns: List[String]
    var classes: List[Int32]
    var width: Int
    var next: List[Int32]
    # The pattern reported in each state (the longest pattern that is a
    # suffix of the state's text), or -1.
    var out: List[Int32]
    var fold: Bool
    var leftmost: Bool
    var longest: Int

    def __init__(
        out self, patterns: List[String], fold: Bool, leftmost: Bool
    ) raises:
        self.fold = fold
        self.leftmost = leftmost
        self.patterns = List[String](capacity=len(patterns))
        self.longest = 0
        for pattern in patterns:
            self.patterns.append(_fold_ascii(pattern) if fold else pattern)
            self.longest = max(self.longest, pattern.byte_length())
        # Byte classes: 0 for bytes in no pattern.
        self.classes = List[Int32](length=256, fill=0)
        var width = 1
        for pattern in self.patterns:
            for byte in pattern.as_bytes():
                if self.classes[Int(byte)] == 0:
                    self.classes[Int(byte)] = Int32(width)
                    width += 1
        if fold:
            for byte in range(65, 91):
                self.classes[byte] = self.classes[byte + 32]
        self.width = width
        # The trie; -1 marks a missing edge until the failure pass.
        self.next = List[Int32](length=width, fill=-1)
        self.out = List[Int32](length=1, fill=-1)
        for p in range(len(self.patterns)):
            var state = 0
            for byte in self.patterns[p].as_bytes():
                var slot = state * width + Int(self.classes[Int(byte)])
                if self.next[slot] < 0:
                    self.next[slot] = Int32(len(self.out))
                    self.out.append(-1)
                    self.next.resize(len(self.next) + width, -1)
                state = Int(self.next[slot])
            # A repeated pattern reports its first occurrence.
            if self.out[state] < 0:
                self.out[state] = Int32(p)
        # Failure links in breadth-first order; each state's missing edges
        # become its failure state's (already complete) transitions.
        var states = len(self.out)
        var failure = List[Int32](length=states, fill=0)
        var queue = List[Int32](capacity=states)
        for c in range(width):
            var child = self.next[c]
            if child < 0:
                self.next[c] = 0
            else:
                failure[Int(child)] = 0
                if self.out[Int(child)] < 0:
                    self.out[Int(child)] = self.out[0]
                queue.append(child)
        var head = 0
        while head < len(queue):
            var state = Int(queue[head])
            head += 1
            var fail = Int(failure[state])
            for c in range(width):
                var slot = state * width + c
                var child = self.next[slot]
                var through = self.next[fail * width + c]
                if child < 0:
                    self.next[slot] = through
                else:
                    failure[Int(child)] = through
                    if self.out[Int(child)] < 0:
                        self.out[Int(child)] = self.out[Int(through)]
                    queue.append(child)

    @always_inline
    def _class(self, byte: UInt8) -> Int:
        return Int(self.classes[Int(byte)])

    def _standard(self, row: _Row, at: Int) -> _Match:
        """The first match ending at or after `at` (start -1 if none)."""
        if self.out[0] >= 0:
            return _Match(at, at, Int(self.out[0]))
        var state = 0
        var data = row.unsafe_ptr()
        for i in range(at, len(row)):
            state = Int(
                self.next[
                    state * self.width + self._class(data[unsafe_offset=i])
                ]
            )
            var hit = self.out[state]
            if hit >= 0:
                var end = i + 1
                return _Match(
                    end - self.patterns[Int(hit)].byte_length(), end, Int(hit)
                )
        return _Match(-1, -1, -1)

    @always_inline
    def _matches_at(self, row: _Row, at: Int, p: Int) -> Bool:
        var pattern = self.patterns[p].as_bytes()
        var m = len(pattern)
        if m > len(row) - at:
            return False
        var data = row.unsafe_ptr()
        for b in range(m):
            var byte = data[unsafe_offset=at + b]
            if self.fold and byte >= 65 and byte <= 90:
                byte |= 32
            if byte != pattern[b]:
                return False
        return True

    def find(self, row: _Row, at: Int) -> _Match:
        """The next match at or after `at`, by the automaton's semantics."""
        var first = self._standard(row, at)
        if not self.leftmost or first.start < 0:
            return first^
        # A match starting before the first one to end must end after it,
        # so it starts within `longest` bytes of that end; the first
        # position with any match, and there the earliest pattern, wins.
        for start in range(max(at, first.end - self.longest), first.start + 1):
            for p in range(len(self.patterns)):
                if self._matches_at(row, start, p):
                    return _Match(
                        start, start + self.patterns[p].byte_length(), p
                    )
        return first^


def _fold_ascii(text: String) -> String:
    var bytes = List[UInt8](capacity=text.byte_length())
    for byte in text.as_bytes():
        bytes.append(byte | 32 if byte >= 65 and byte <= 90 else byte)
    return String(StringSlice(unsafe_from_utf8=Span(bytes)))


def _contains_any(column: StringColumn, automaton: _Automaton) raises -> Series:
    var n = len(column)
    var rows = _Bytes(column)
    var nulls = column.null_count() > 0
    var values = List[UInt8](length=(n + 7) // 8, fill=0)
    for i in range(n):
        if nulls and not column._valid(i):
            continue
        if automaton.find(rows.get(i), 0).start >= 0:
            values[i // 8] |= UInt8(1) << UInt8(i % 8)
    var valid = _copy_validity(column._bits[], column._offset, n)
    return Series("", BoolColumn(values=values^, bits=valid^, length=n))


def _replace_many(
    column: StringColumn, automaton: _Automaton, replacements: List[String]
) raises -> Series:
    """Every match is replaced, scanning left to right without overlaps.
    An empty match is not taken where the previous match ended, and after
    one the scan moves on by a code point."""
    var n = len(column)
    var rows = _Bytes(column)
    var nulls = column.null_count() > 0
    var out = StringBuilder(n, column._value_bytes())
    for i in range(n):
        if nulls and not column._valid(i):
            out.append_null()
            continue
        var row = rows.get(i)
        var size = len(row)
        var data = row.unsafe_ptr()
        var copied = 0
        var at = 0
        var last_end = -1
        while at <= size:
            var found = automaton.find(row, at)
            if found.start < 0:
                break
            var empty = found.end == found.start
            if not (empty and found.start == last_end):
                out._bytes.extend(
                    _Row(
                        unsafe_ptr=data.unsafe_offset(copied),
                        length=found.start - copied,
                    )
                )
                var replacement = replacements[
                    0 if len(replacements) == 1 else found.pattern
                ].as_bytes()
                out._bytes.extend(replacement)
                copied = found.end
                last_end = found.end
            if not empty:
                at = found.end
            elif found.start >= size:
                break
            else:
                at = found.start + _codepoint_width(
                    data[unsafe_offset=found.start]
                )
        out._bytes.extend(
            _Row(unsafe_ptr=data.unsafe_offset(copied), length=size - copied)
        )
        _end_row(out)
    return Series("", out^.finish())


# Encodings ----------------------------------------------------------------

comptime _HEX = "0123456789abcdef"
comptime _BASE64 = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
)


def _encode(column: StringColumn, hex: Bool) raises -> Series:
    """UTF-8 bytes as lowercase hex, or as standard padded base64."""
    var n = len(column)
    var rows = _Bytes(column)
    var nulls = column.null_count() > 0
    var size = column._value_bytes()
    var out = StringBuilder(n, 2 * size if hex else (size + 2) // 3 * 4)
    var hex_digits = StringSlice(_HEX).as_bytes()
    var alphabet = StringSlice(_BASE64).as_bytes()
    for i in range(n):
        if nulls and not column._valid(i):
            out.append_null()
            continue
        var row = rows.get(i)
        var data = row.unsafe_ptr()
        if hex:
            for b in range(len(row)):
                var byte = Int(data[unsafe_offset=b])
                out._bytes.append(hex_digits[byte >> 4])
                out._bytes.append(hex_digits[byte & 15])
        else:
            var b = 0
            while b < len(row):
                var left = len(row) - b
                var word = Int(data[unsafe_offset=b]) << 16
                if left > 1:
                    word |= Int(data[unsafe_offset=b + 1]) << 8
                if left > 2:
                    word |= Int(data[unsafe_offset=b + 2])
                out._bytes.append(alphabet[(word >> 18) & 63])
                out._bytes.append(alphabet[(word >> 12) & 63])
                out._bytes.append(
                    alphabet[(word >> 6) & 63] if left > 1 else 61
                )
                out._bytes.append(alphabet[word & 63] if left > 2 else 61)
                b += 3
        _end_row(out)
    return Series("", out^.finish())


@always_inline
def _digit(byte: UInt8) -> Int:
    """A byte's value as a digit in bases up to 36, or 99."""
    if byte >= 48 and byte <= 57:
        return Int(byte) - 48
    var lower = byte | 32
    if lower >= 97 and lower <= 122:
        return Int(lower) - 87
    return 99


@always_inline
def _sextet(byte: UInt8) -> Int:
    """A base64 character's value, or -1."""
    if byte >= 65 and byte <= 90:
        return Int(byte) - 65
    if byte >= 97 and byte <= 122:
        return Int(byte) - 71
    if byte >= 48 and byte <= 57:
        return Int(byte) + 4
    if byte == 43:
        return 62
    if byte == 47:
        return 63
    return -1


def _decode_row(row: _Row, hex: Bool, mut out: List[UInt8]) -> Bool:
    """Append the bytes a row encodes; False (with `out` partly written) if
    the row is not valid hex (an even number of digits) or canonical padded
    base64 (Rust's `base64` STANDARD engine: padding required, unused bits
    zero)."""
    var size = len(row)
    var data = row.unsafe_ptr()
    if hex:
        if size % 2 != 0:
            return False
        for b in range(0, size, 2):
            var high = _digit(data[unsafe_offset=b])
            var low = _digit(data[unsafe_offset=b + 1])
            if high > 15 or low > 15:
                return False
            out.append(UInt8(high * 16 + low))
        return True
    if size % 4 != 0:
        return False
    for b in range(0, size, 4):
        var last = b + 4 == size
        var values = SIMD[DType.int32, 4](0)
        var padding = 0
        for j in range(4):
            var byte = data[unsafe_offset=b + j]
            if byte == 61 and last and j >= 2:
                padding += 1
                continue
            if padding > 0:
                return False
            var value = _sextet(byte)
            if value < 0:
                return False
            values[j] = Int32(value)
        var word = Int(
            (values[0] << 18) | (values[1] << 12) | (values[2] << 6) | values[3]
        )
        out.append(UInt8((word >> 16) & 255))
        if padding == 2:
            if (word & 0xFFFF) != 0:
                return False
        elif padding == 1:
            if (word & 0xFF) != 0:
                return False
            out.append(UInt8((word >> 8) & 255))
        else:
            out.append(UInt8((word >> 8) & 255))
            out.append(UInt8(word & 255))
    return True


def _decode(column: StringColumn, hex: Bool, strict: Bool) raises -> Series:
    var n = len(column)
    var rows = _Bytes(column)
    var nulls = column.null_count() > 0
    var out = StringBuilder(n, column._value_bytes())
    for i in range(n):
        if nulls and not column._valid(i):
            out.append_null()
            continue
        var start = len(out._bytes)
        if _decode_row(rows.get(i), hex, out._bytes):
            _end_row(out)
            continue
        if strict:
            raise Error(
                "invalid `"
                + ("hex" if hex else "base64")
                + "` encoding found; try setting `strict=False` to ignore"
            )
        out._bytes.resize(start, 0)
        out.append_null()
    return Series("", out^.finish()).with_dtype(DataType.BINARY)


def _to_integer(column: StringColumn, base: Int, strict: Bool) raises -> Series:
    """Rust's `i64::from_str_radix`: an optional sign, then one or more
    digits of the base; anything else, or a value out of range, fails."""
    var n = len(column)
    var rows = _Bytes(column)
    var nulls = column.null_count() > 0
    var values = List[Int64](length=n, fill=0)
    var bits = _copy_validity(column._bits[], column._offset, n)
    var failures = 0
    var first_failure = String()
    for i in range(n):
        if nulls and not column._valid(i):
            continue
        var row = rows.get(i)
        var size = len(row)
        var data = row.unsafe_ptr()
        var at = 0
        var negative = False
        if size > 0 and (data[] == 43 or data[] == 45):
            negative = data[] == 45
            at = 1
        # The magnitude, up to 2^63 for a negative value.
        var limit = UInt64(1) << 63 if negative else (UInt64(1) << 63) - 1
        var magnitude = UInt64(0)
        var ok = at < size
        while ok and at < size:
            var digit = _digit(data[unsafe_offset=at])
            if digit >= base or magnitude > (limit - UInt64(digit)) // UInt64(
                base
            ):
                ok = False
                break
            magnitude = magnitude * UInt64(base) + UInt64(digit)
            at += 1
        if not ok:
            if failures == 0:
                first_failure = String(StringSlice(unsafe_from_utf8=row))
            failures += 1
            _clear_bit(bits, i, n)
            continue
        values[i] = Int64(0) - Int64(
            magnitude - 1
        ) - 1 if negative and magnitude > 0 else Int64(magnitude)
    if strict and failures > 0:
        raise Error(
            "strict integer parsing failed for "
            + String(failures)
            + " value(s), the first: '"
            + first_failure
            + "' (consider strict=False)"
        )
    return Series("", Column[Int64](values=values^, bits=bits^))


# Case ---------------------------------------------------------------------


@always_inline
def _ascii_row(row: _Row) -> Bool:
    var data = row.unsafe_ptr()
    var size = len(row)
    var b = 0
    while b + 8 <= size:
        if (
            data.unsafe_offset(b).unsafe_bitcast[UInt64]().unsafe_load()
            & 0x8080808080808080
        ) != 0:
            return False
        b += 8
    while b < size:
        if data[unsafe_offset=b] >= 0x80:
            return False
        b += 1
    return True


def _ends_word(row: _Row, at: Int, width: Int, ranges: List[UInt32]) -> Bool:
    """Whether the code point at `at` follows an alphabetic one and is not
    followed by one (Rust's final-sigma rule, without its skipping of
    case-ignorable characters)."""
    var data = row.unsafe_ptr()
    if at == 0:
        return False
    var before = at - 1
    while before > 0 and (data[unsafe_offset=before] & 0xC0) == 0x80:
        before -= 1
    if not is_alphabetic(
        ranges,
        _codepoint_value(
            data, before, _codepoint_width(data[unsafe_offset=before])
        ),
    ):
        return False
    var after = at + width
    if after >= len(row):
        return True
    return not is_alphabetic(
        ranges,
        _codepoint_value(
            data, after, _codepoint_width(data[unsafe_offset=after])
        ),
    )


def _titlecase(column: StringColumn) raises -> Series:
    """The row is lowercased, then each character that follows a
    non-alphabetic one (or starts the row) is uppercased. ASCII is
    done byte by byte; other code points go through the Unicode case maps
    of `String.lower` and `String.upper` (full mappings: "ß" uppercases to
    "SS")."""
    var n = len(column)
    var rows = _Bytes(column)
    var nulls = column.null_count() > 0
    var out = StringBuilder(n, column._value_bytes())
    var ranges = alphabetic_ranges()
    for i in range(n):
        if nulls and not column._valid(i):
            out.append_null()
            continue
        var row = rows.get(i)
        if _ascii_row(row):
            var upper = True
            for b in range(len(row)):
                var byte = row.unsafe_ptr()[unsafe_offset=b]
                var lower = byte | 32 if byte >= 65 and byte <= 90 else byte
                var letter = lower >= 97 and lower <= 122
                out._bytes.append(lower - 32 if upper and letter else lower)
                upper = not letter
            _end_row(out)
            continue
        var data = row.unsafe_ptr()
        var size = len(row)
        var upper = True
        var b = 0
        while b < size:
            var width = _codepoint_width(data[unsafe_offset=b])
            var value = _codepoint_value(data, b, width)
            if value < 128:
                var lower = UInt8(
                    value | 32
                ) if value >= 65 and value <= 90 else UInt8(value)
                var letter = lower >= 97 and lower <= 122
                out._bytes.append(lower - 32 if upper and letter else lower)
                upper = not letter
                b += width
                continue
            # Lowercase the code point (Rust's final sigma: a capital
            # sigma ending a word becomes ς), then uppercase what follows
            # a character that is not alphabetic.
            var lowered: String
            if value == 0x3A3:
                lowered = "ς" if _ends_word(row, b, width, ranges) else "σ"
            else:
                lowered = String(
                    StringSlice(
                        unsafe_from_utf8=_Row(
                            unsafe_ptr=data.unsafe_offset(b), length=width
                        )
                    )
                ).lower()
            var low = lowered.unsafe_ptr()
            var at = 0
            while at < lowered.byte_length():
                var part = _codepoint_width(low[unsafe_offset=at])
                var point = _codepoint_value(low, at, part)
                var text = _Row(
                    unsafe_ptr=low.unsafe_offset(at)
                    .unsafe_mut_cast[False]()
                    .unsafe_origin_cast[ImmutAnyOrigin](),
                    length=part,
                )
                if upper:
                    out._bytes.extend(
                        String(StringSlice(unsafe_from_utf8=text))
                        .upper()
                        .as_bytes()
                    )
                else:
                    out._bytes.extend(text)
                upper = not is_alphabetic(ranges, point)
                at += part
            # Read only through `low` above; keep it alive until here.
            _ = lowered^
            b += width
        _end_row(out)
    return Series("", out^.finish())
