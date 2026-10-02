"""String columns compared with literal values, and `is_in` (#373).

Comparing a string column with a literal used to build a `StringSlice` per
row, append to a `List[Bool]` of results and another of validity, and pack
both afterwards, on one thread: about 60 ns a row. `is_in` was a chain of
`==`, `fill_null` and `|` per value. Polars compares each row's 16-byte view
against the scalar and writes a bitmap (`tot_eq_kernel_broadcast` in
polars-compute's comparisons/view.rs), and builds a hash set once for
`is_in`. Here each worker takes a range of rows (a multiple of 8, so whole
bitmap bytes), reads row bytes from the buffers directly (`_Bytes`), and
writes the result bitmap; validity is the input's, copied once. Equality
checks the length before any byte; `is_in` with more than eight values
probes a small open-addressing table of their hashes.
"""
from .bool_column import BoolColumn
from .column import _copy_validity
from .expr import EQ, GE, GT, LE, LT, NE
from .parallel import Job, run_jobs, worker_count
from .partition import _hash_bytes
from .series import Series
from .string_bytes import (
    _Bytes,
    _find_bytes,
    _order_bytes,
    _same_bytes,
    _search,
)
from .string_column import StringColumn
from .string_view import StringView, StringViewStorage
from std.memory import ArcPointer, Pointer

# Predicate codes beside the comparison operators.
comptime IS_IN = -1
comptime STARTS_WITH = -2
comptime ENDS_WITH = -3
comptime CONTAINS = -4
comptime LIKE = -5

# `_Literals.flags` for LIKE: the pattern is anchored at the start (does not
# begin with '%'), at the end (does not end with '%'), and has '_'.
comptime _ANCHOR_START = 1
comptime _ANCHOR_END = 2
comptime _WILDCARDS = 4


@always_inline
def _span(
    bytes: List[UInt8], start: Int, end: Int
) -> Span[UInt8, ImmutAnyOrigin]:
    return Span[UInt8, ImmutAnyOrigin](
        unsafe_ptr=bytes.unsafe_ptr()
        .unsafe_offset(start)
        .unsafe_mut_cast[False]()
        .unsafe_origin_cast[ImmutAnyOrigin](),
        length=end - start,
    )


struct _Literals(Copyable, Movable):
    """The values a row is compared with, packed into one buffer, and for
    more than eight of them a table of their hashes (linear probing)."""

    var bytes: List[UInt8]
    var starts: List[Int]
    var slots: List[Int32]
    var flags: Int

    def __init__(out self, values: List[String], flags: Int = 0):
        self.flags = flags
        self.bytes = List[UInt8]()
        self.starts = List[Int](capacity=len(values) + 1)
        self.starts.append(0)
        for value in values:
            self.bytes.extend(value.as_bytes())
            self.starts.append(len(self.bytes))
        self.slots = List[Int32]()
        var count = len(values)
        if count > 8:
            var size = 16
            while size < 2 * count:
                size *= 2
            self.slots = List[Int32](length=size, fill=-1)
            for k in range(count):
                var slot = Int(_hash_bytes(self.get(k)) & UInt64(size - 1))
                while self.slots[slot] >= 0:
                    slot = (slot + 1) & (size - 1)
                self.slots[slot] = Int32(k)

    @always_inline
    def get(self, k: Int) -> Span[UInt8, ImmutAnyOrigin]:
        return _span(self.bytes, self.starts[k], self.starts[k + 1])

    @always_inline
    def contains(self, value: Span[UInt8, ImmutAnyOrigin]) -> Bool:
        if len(self.slots) == 0:
            for k in range(len(self.starts) - 1):
                if _same_bytes(value, self.get(k)):
                    return True
            return False
        var mask = len(self.slots) - 1
        var slot = Int(_hash_bytes(value) & UInt64(mask))
        while True:
            var k = Int(self.slots[slot])
            if k < 0:
                return False
            if _same_bytes(value, self.get(k)):
                return True
            slot = (slot + 1) & mask


struct _PredicateJob[op: Int](Job):
    """Rows [first, last) of a predicate, written as whole bitmap bytes:
    `first` is a multiple of 8 and `last` is too unless it ends the column."""

    var rows: _Bytes
    var nulls: Bool
    var literals: _Literals
    var output: Int
    var first: Int
    var last: Int

    def __init__(
        out self,
        column: StringColumn,
        nulls: Bool,
        literals: _Literals,
        output: Int,
        first: Int,
        last: Int,
    ):
        self.rows = _Bytes(column)
        self.nulls = nulls
        self.literals = literals.copy()
        self.output = output
        self.first = first
        self.last = last

    def run(mut self) raises:
        var bits = Pointer[List[UInt8], MutAnyOrigin](
            unsafe_from_address=self.output
        )[].unsafe_ptr()
        var literal = self.literals.get(0)
        var row = self.first
        while row < self.last:
            var end = min(row + 8, self.last)
            var byte = UInt8(0)
            for i in range(row, end):
                if self.nulls and not self.rows.column._valid(i):
                    continue
                var value = self.rows.get(i)
                var hit: Bool
                comptime if Self.op == IS_IN:
                    hit = self.literals.contains(value)
                elif Self.op == STARTS_WITH:
                    hit = len(value) >= len(literal) and _same_bytes(
                        Span[UInt8, ImmutAnyOrigin](
                            unsafe_ptr=value.unsafe_ptr(), length=len(literal)
                        ),
                        literal,
                    )
                elif Self.op == ENDS_WITH:
                    hit = len(value) >= len(literal) and _same_bytes(
                        Span[UInt8, ImmutAnyOrigin](
                            unsafe_ptr=value.unsafe_ptr().unsafe_offset(
                                len(value) - len(literal)
                            ),
                            length=len(literal),
                        ),
                        literal,
                    )
                elif Self.op == CONTAINS:
                    hit = _find_bytes(value, literal, 0) >= 0
                elif Self.op == LIKE:
                    hit = _like(value, self.literals)
                elif Self.op == EQ:
                    hit = _same_bytes(value, literal)
                elif Self.op == NE:
                    hit = not _same_bytes(value, literal)
                elif Self.op == LT:
                    hit = _order_bytes(value, literal) < 0
                elif Self.op == LE:
                    hit = _order_bytes(value, literal) <= 0
                elif Self.op == GT:
                    hit = _order_bytes(value, literal) > 0
                else:
                    hit = _order_bytes(value, literal) >= 0
                if hit:
                    byte |= UInt8(1) << UInt8(i - row)
            bits.unsafe_offset(row // 8)[] = byte
            row = end


def _like(value: Span[UInt8, ImmutAnyOrigin], pattern: _Literals) -> Bool:
    """SQL LIKE: the pattern's pieces between '%' (the literals) matched left
    to right, each at its earliest position, which is exact for '%'; a
    piece anchored at the end is matched only where it ends the value."""
    var pieces = len(pattern.starts) - 1
    var wildcards = (pattern.flags & _WILDCARDS) != 0
    var anchor_start = (pattern.flags & _ANCHOR_START) != 0
    var anchor_end = (pattern.flags & _ANCHOR_END) != 0
    if pieces == 1 and anchor_start and anchor_end:
        return _match_whole(value, pattern.get(0), wildcards)
    var at = 0
    var first = 0
    if anchor_start:
        var head = _search(value, 0, pattern.get(0), wildcards)
        if head[0] != 0:
            # The first match is not at 0; with no wildcards it is the only
            # candidate start, and with them a match at 0 is found first.
            return False
        at = head[1]
        first = 1
    var last = pieces - 1 if anchor_end else pieces
    for k in range(first, last):
        var piece = pattern.get(k)
        if len(piece) == 0:
            continue
        var found = _search(value, at, piece, wildcards)
        if found[0] < 0:
            return False
        at = found[1]
    if anchor_end:
        var tail = pattern.get(pieces - 1)
        if not wildcards:
            var start = len(value) - len(tail)
            return start >= at and _same_bytes(
                Span[UInt8, ImmutAnyOrigin](
                    unsafe_ptr=value.unsafe_ptr().unsafe_offset(start),
                    length=len(tail),
                ),
                tail,
            )
        for p in range(at, len(value) + 1):
            if _match_whole(
                Span[UInt8, ImmutAnyOrigin](
                    unsafe_ptr=value.unsafe_ptr().unsafe_offset(p),
                    length=len(value) - p,
                ),
                tail,
                True,
            ):
                return True
        return False
    return True


def _match_whole(
    value: Span[UInt8, ImmutAnyOrigin],
    piece: Span[UInt8, ImmutAnyOrigin],
    wildcards: Bool,
) -> Bool:
    if not wildcards:
        return _same_bytes(value, piece)
    var found = _search(value, 0, piece, True)
    # With '_' the piece can match at 0 with different end positions only
    # if code point widths differ; _match_at follows the value's widths.
    return found[0] == 0 and found[1] == len(value)


def _predicate[
    op: Int
](column: StringColumn, literals: _Literals) raises -> Series:
    var n = len(column)
    # Every byte is written by exactly one job below.
    var values = List[UInt8](unsafe_uninit_length=(n + 7) // 8)
    var nulls = column.null_count() > 0
    var workers = worker_count(n)
    var jobs = List[_PredicateJob[op]](capacity=workers)
    for w in range(workers):
        var first = (n * w // workers) // 8 * 8
        var last = n if w == workers - 1 else (n * (w + 1) // workers) // 8 * 8
        jobs.append(
            _PredicateJob[op](
                column, nulls, literals, Int(Pointer(to=values)), first, last
            )
        )
    if workers == 1:
        jobs[0].run()
    else:
        run_jobs(jobs)
    # The jobs wrote through the list's address; keep it alive until then.
    var valid = _copy_validity(column._bits[], column._offset, n)
    return Series("", BoolColumn(values=values^, bits=valid^, length=n))


def _flip(op: Int) -> Int:
    """The operator with its operands swapped: lit < col is col > lit."""
    if op == LT:
        return GT
    if op == LE:
        return GE
    if op == GT:
        return LT
    if op == GE:
        return LE
    return op


def compare_with_literal(
    column: StringColumn, op: Int, literal: StringSlice, literal_left: Bool
) raises -> Series:
    """`column <op> literal` (or `literal <op> column` with `literal_left`)
    for a valid literal; null rows stay null."""
    var literals = _Literals([String(literal)])
    var actual = _flip(op) if literal_left else op
    if actual == EQ:
        return _predicate[EQ](column, literals)
    if actual == NE:
        return _predicate[NE](column, literals)
    if actual == LT:
        return _predicate[LT](column, literals)
    if actual == LE:
        return _predicate[LE](column, literals)
    if actual == GT:
        return _predicate[GT](column, literals)
    return _predicate[GE](column, literals)


def is_in_literals(column: StringColumn, values: List[String]) raises -> Series:
    """Whether each row equals any of `values`; null rows stay null."""
    if len(values) == 0:
        var n = len(column)
        var falses = List[UInt8](length=(n + 7) // 8, fill=0)
        var valid = _copy_validity(column._bits[], column._offset, n)
        return Series("", BoolColumn(values=falses^, bits=valid^, length=n))
    return _predicate[IS_IN](column, _Literals(values))


def decode_values(text: String) raises -> List[String]:
    """The values `Expr.is_in` packed into a node's `text`: each is its
    byte length, ':' and its bytes."""
    var values = List[String]()
    var bytes = text.as_bytes()
    var at = 0
    while at < len(bytes):
        var length = 0
        while bytes[at] != 58:
            length = length * 10 + Int(bytes[at]) - 48
            at += 1
        at += 1
        values.append(
            String(
                unsafe_from_utf8=Span[UInt8, ImmutAnyOrigin](
                    unsafe_ptr=bytes.unsafe_ptr()
                    .unsafe_offset(at)
                    .unsafe_mut_cast[False]()
                    .unsafe_origin_cast[ImmutAnyOrigin](),
                    length=length,
                )
            )
        )
        at += length
    return values^


def string_starts_with(column: StringColumn, prefix: String) raises -> Series:
    return _predicate[STARTS_WITH](column, _Literals([prefix]))


def string_ends_with(column: StringColumn, suffix: String) raises -> Series:
    return _predicate[ENDS_WITH](column, _Literals([suffix]))


def string_contains(column: StringColumn, literal: String) raises -> Series:
    return _predicate[CONTAINS](column, _Literals([literal]))


def string_like(column: StringColumn, pattern: String) raises -> Series:
    """SQL LIKE: '%' matches any run of characters, '_' exactly one."""
    var pieces = List[String]()
    for part in pattern.split("%"):
        pieces.append(String(part))
    var flags = 0
    if not pattern.startswith("%"):
        flags |= _ANCHOR_START
    if not pattern.endswith("%") or len(pieces) == 1:
        flags |= _ANCHOR_END
    if "_" in pattern:
        flags |= _WILDCARDS
    return _predicate[LIKE](column, _Literals(pieces, flags))


@always_inline
def _codepoint_end(
    bytes: Pointer[UInt8, ImmutAnyOrigin], length: Int, at: Int, count: Int
) -> Int:
    """The byte position `count` code points after byte `at`, clipped."""
    var i = at
    var left = count
    while left > 0 and i < length:
        i += 1
        while i < length and (bytes.unsafe_offset(i)[] & 0xC0) == 0x80:
            i += 1
        left -= 1
    return i


def slice_strings(
    column: StringColumn, offset: Int, length: Int
) raises -> StringColumn:
    """`str.slice(offset, length)` as views into the source's bytes (#374):
    code points [offset, offset + length), a negative offset counting from
    the end and length -1 taking the rest, clipped. A result of up to 12
    bytes is copied into its view; a longer one points into the source
    buffer, so no row's bytes are copied."""
    var n = len(column)
    var views_storage = column._is_view_storage()
    if not views_storage and len(column._bytes[]) > 4_294_967_295:
        raise Error("slice source too large for views")
    var views = List[StringView](unsafe_uninit_length=n)
    var words = views.unsafe_ptr().unsafe_bitcast[UInt64]()
    var buffers = List[ArcPointer[List[UInt8]]]()
    var views_at = 0
    if views_storage:
        var storage = column._view_storage_unchecked()
        for buffer in storage._buffers[]:
            buffers.append(buffer.copy())
        views_at = Int(storage._views[].unsafe_ptr())
    else:
        buffers.append(column._bytes.copy())
    var nulls = column.null_count() > 0
    var total = 0
    var total_buffer = 0
    for k in range(n):
        if nulls and not column._valid(k):
            words.unsafe_offset(2 * k)[] = 0
            words.unsafe_offset(2 * k + 1)[] = 0
            continue
        var text = column._get(k).as_bytes()
        var data = text.unsafe_ptr()
        var size = len(text)
        var buffer = UInt64(0)
        var base = 0
        if views_storage:
            var view = (
                Pointer[StringView, ImmutAnyOrigin](
                    unsafe_from_address=views_at
                )
                .unsafe_offset(column._offset + k)[]
                .copy()
            )
            buffer = UInt64(view.buffer_index)
            base = Int(view.offset)
        else:
            base = column._start(k)
        var b0: Int
        if offset >= 0:
            b0 = _codepoint_end(data, size, 0, offset)
        else:
            var points = 0
            for i in range(size):
                if (data.unsafe_offset(i)[] & 0xC0) != 0x80:
                    points += 1
            b0 = _codepoint_end(data, size, 0, max(points + offset, 0))
        var b1 = size if length < 0 else _codepoint_end(data, size, b0, length)
        var part = b1 - b0
        total += part
        var low = UInt64(part)
        var high = UInt64(0)
        var p = data.unsafe_offset(b0)
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
            total_buffer += part
        words.unsafe_offset(2 * k)[] = low
        words.unsafe_offset(2 * k + 1)[] = high
    var bits = _copy_validity(column._bits[], column._offset, n)
    return StringColumn(
        StringViewStorage(views^, buffers^, bits^, n, total, total_buffer)
    )
