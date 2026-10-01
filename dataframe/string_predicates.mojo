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
from .string_bytes import _Bytes, _order_bytes, _same_bytes
from .string_column import StringColumn

# `is_in` as a predicate code beside the comparison operators.
comptime IS_IN = -1


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

    def __init__(out self, values: List[String]):
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
