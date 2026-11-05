"""DuckDB 2.0's join hash table (`execution/join_hashtable.cpp`,
`execution/ht_entry.hpp`, `common/row_operations/row_matcher.cpp`).

The build side is laid out as fixed-width rows, as DuckDB's
TupleDataLayout lays out a TupleDataCollection:

    [next pointer][key fields][build row][keys valid][padding to 8 bytes]

- The next pointer holds the row's hash until the row is inserted, then the
  row after it in its chain (0 ends a chain), as DuckDB's NEXT_PTR holds the
  hash before Finalize.
- Each key field is an 8-byte word (integers sign- or zero-extended, floats
  by `float_key` so every NaN is one value and -0.0 equals 0.0, booleans as
  1 and 2), two words for a 128-bit decimal, or a 16-byte string_t: the
  length, then up to twelve bytes inline, or four prefix bytes and the
  address of the bytes, which stay in the build's own column (DuckDB copies
  them into its collection's heap; the build frame outlives the table here).
- The build row is the index DuckDB's dictionary emission stores in
  NEXT_PTR (`BuildDictionaryArrays`): the probe emits the build side as
  dictionaries over the build frame's columns through it.

The pointer table is an array of 8-byte `ht_entry_t`: the hash's upper 16
bits (the salt) over the row's 48-bit address. It holds at least twice as
many entries as rows, and at least 16,384 (`PointerTableCapacity`). A row is
inserted by linear probing from its hash's low bits to an empty entry or to
the entry of a row with the same salt and keys, whose chain it joins:
every chain holds rows with equal keys, so a probe compares keys once per
chain. Inserting is single-threaded below 1,048,576 rows and lock-free
above (compare-and-swap on the entry), as DuckDB's Finalize. Single-
threaded inserts go in descending row order, so each chain lists its rows
in ascending order.

A probe (`ScanStructure`) hashes a chunk of up to 2,048 rows, finds each
row's chain by linear probing (salts compared once the table is larger than
8,192 entries, `USE_SALT_THRESHOLD`), compares the keys of the candidates
(`RowMatcher`), and emits at most 2,048 matches per call to `next`, walking
each probe row's chain before the next row's, so the matches come out in
probe order. Rows with a null key match nothing.

Right and full joins flag the build rows they matched (`found`, one byte
per build row; DuckDB keeps the flag in the row), and `scan_full_outer`
then emits the build rows no probe matched, as DuckDB's ScanFullOuter. A
semi or anti join built on its left input runs as DuckDB's RIGHT_SEMI or
RIGHT_ANTI: the probe only flags build rows, and the scan emits the flagged
(semi) or unflagged (anti) ones.
"""
from std.atomic import Atomic
from std.collections import Optional
from std.ffi import external_call
from std.memory import ArcPointer, Pointer, bitcast, unsafe_memcpy

from dataframe.aggregation.reducer import float_key
from dataframe.dtype import NUMERIC_DTYPES
from dataframe.join_type import (
    JOIN_ANTI,
    JOIN_FULL,
    JOIN_INNER,
    JOIN_LEFT,
    JOIN_MARK,
    JOIN_RIGHT,
    JOIN_RIGHT_ANTI,
    JOIN_RIGHT_SEMI,
    JOIN_SEMI,
)
from dataframe.kernels.partition import _hash_column
from dataframe.parallel import Job, partitions, run_jobs, worker_count
from dataframe.series import Series
from dataframe.storage.allocation import huge_list, huge_uninit
from dataframe.storage.boolean import BoolColumn
from dataframe.storage.buffer import _Buffer
from dataframe.storage.column import Column, _validity_at
from dataframe.storage.string import StringColumn
from dataframe.storage.string_view import StringView, _mapped
from dataframe.trace import trace_path
from dataframe.vector.types import STANDARD_VECTOR_SIZE

# ht_entry_t (execution/ht_entry.hpp): the upper 16 bits are the salt, the
# lower 48 the row's address.
comptime SALT_MASK = UInt64(0xFFFF000000000000)
comptime POINTER_MASK = UInt64(0x0000FFFFFFFFFFFF)
# JoinHashTable::USE_SALT_THRESHOLD: salts are compared on probes only once
# the table no longer fits the CPU cache.
comptime USE_SALT_THRESHOLD = 8192
# JoinHashTable::DEFAULT_LOAD_FACTOR and MINIMUM_CAPACITY.
comptime DEFAULT_LOAD_FACTOR = 2.0
comptime MINIMUM_CAPACITY = 16384
# physical_hash_join.cpp PARALLEL_CONSTRUCT_THRESHOLD: smaller tables are
# finalized on one thread.
comptime PARALLEL_CONSTRUCT_THRESHOLD = 1048576
# PerfectHashJoinExecutor::CanDoPerfectHashJoin MAX_BUILD_SIZE: the widest
# key range a perfect hash join addresses directly.
comptime PERFECT_MAX_BUILD_SIZE = 1048576

# Key field kinds and the row's fixed offsets.
comptime _WORD = 0
comptime _WIDE = 1
comptime _TEXT = 2
comptime _NEXT = 0
comptime _KEYS = 8
# A string_t's inline capacity (common/types/string_type.hpp INLINE_LENGTH).
comptime _INLINE = 12


@always_inline
def _load64(address: Int) -> UInt64:
    return Pointer[UInt64, MutAnyOrigin](unsafe_from_address=address)[]


@always_inline
def _store64(address: Int, value: UInt64):
    Pointer[UInt64, MutAnyOrigin](unsafe_from_address=address)[] = value


@always_inline
def _load8(address: Int) -> UInt8:
    return Pointer[UInt8, MutAnyOrigin](unsafe_from_address=address)[]


@always_inline
def _store8(address: Int, value: UInt8):
    Pointer[UInt8, MutAnyOrigin](unsafe_from_address=address)[] = value


@always_inline
def _load32(address: Int) -> UInt32:
    return Pointer[UInt32, MutAnyOrigin](unsafe_from_address=address)[]


@always_inline
def _store32(address: Int, value: UInt32):
    Pointer[UInt32, MutAnyOrigin](unsafe_from_address=address)[] = value


struct _StringBytes:
    """Where a string column's rows start and how long they are, from
    pointers taken once: offsets and a byte buffer, or Arrow views (inline
    bytes, or a buffer and an offset)."""

    var views: Int
    var offsets: Int
    var base: Int
    var buffers: Int
    # Views that select their descriptors (`StringViewStorage._row_map`).
    var rows: Int
    var shift: Int

    def __init__(out self, column: StringColumn):
        self.rows = 0
        self.shift = 0
        if column._is_view_storage():
            ref storage = column._view_storage.value()
            var map = storage._row_map(column._offset)
            self.views = map[0]
            self.rows = map[1]
            self.shift = map[2]
            self.buffers = Int(storage._buffers[].unsafe_ptr())
            self.offsets = 0
            self.base = 0
        else:
            self.views = 0
            self.buffers = 0
            self.offsets = Int(
                column._offsets[].unsafe_ptr().unsafe_offset(column._offset)
            )
            self.base = Int(column._base())

    @always_inline
    def at(self, i: Int) -> Tuple[Int, Int]:
        """Row i's first byte's address and its length."""
        if self.views == 0:
            var offsets = Pointer[Int64, MutAnyOrigin](
                unsafe_from_address=self.offsets
            )
            var start = Int(offsets[unsafe_offset=i])
            return (
                self.base + start,
                Int(offsets[unsafe_offset=i + 1]) - start,
            )
        var view = Pointer[StringView, MutAnyOrigin](
            unsafe_from_address=self.views
        ).unsafe_offset(_mapped(self.rows, self.shift, i))
        var length = Int(view[].length)
        if length <= 12:
            return (Int(view) + 4, length)
        var buffers = Pointer[_Buffer[UInt8], MutAnyOrigin](
            unsafe_from_address=self.buffers
        )
        return (
            Int(buffers[unsafe_offset=Int(view[].buffer_index)][].unsafe_ptr())
            + Int(view[].offset),
            length,
        )


def _field_kind(key: Series) -> Int:
    """How a key column's values are stored in a row."""
    if key._data.isa[Column[Int128]]():
        return _WIDE
    if key._data.isa[StringColumn]():
        return _TEXT
    return _WORD


def _field_bytes(kind: Int) -> Int:
    return 8 if kind == _WORD else 16


def _validity(key: Series) -> Tuple[Int, Int]:
    """A key's validity bitmap address (0 when every row is valid) and the
    bit of its row 0."""
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if key._data.isa[Column[Scalar[D]]]():
            ref column = key._data[Column[Scalar[D]]]
            if len(column._bits[]) == 0 or column.null_count() == 0:
                return (0, 0)
            return (Int(column._bits[].unsafe_ptr()), column._offset)
    if key._data.isa[Column[Int128]]():
        ref column = key._data[Column[Int128]]
        if len(column._bits[]) == 0 or column.null_count() == 0:
            return (0, 0)
        return (Int(column._bits[].unsafe_ptr()), column._offset)
    if key._data.isa[BoolColumn]():
        ref column = key._data[BoolColumn]
        if len(column._bits[]) == 0 or key.null_count() == 0:
            return (0, 0)
        return (Int(column._bits[].unsafe_ptr()), column._offset)
    ref column = key._data[StringColumn]
    if len(column._bits[]) == 0 or key.null_count() == 0:
        return (0, 0)
    return (Int(column._bits[].unsafe_ptr()), column._offset)


def _write_fields(
    key: Series,
    start: Int,
    count: Int,
    kind: Int,
    base: Int,
    stride: Int,
    valid: Int,
    valid_stride: Int,
) raises:
    """Rows [start, start + count) of an unchunked key as row fields: row i's
    field at `base + i * stride`, its validity ANDed into the byte at
    `valid + i * valid_stride`. A null row's field is left undefined."""
    var bits = _validity(key)
    if bits[0] != 0:
        var bitmap = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=bits[0])
        for i in range(count):
            if not _validity_at(bitmap, bits[1] + start + i):
                _store8(valid + i * valid_stride, 0)
    if kind == _TEXT:
        var strings = _StringBytes(key._data[StringColumn])
        for i in range(count):
            var at = base + i * stride
            var found = strings.at(start + i)
            var length = found[1]
            _store64(at, UInt64(length))
            _store64(at + 8, 0)
            if length <= _INLINE:
                if length > 0:
                    unsafe_memcpy(
                        dest=Pointer[UInt8, MutAnyOrigin](
                            unsafe_from_address=at + 4
                        ),
                        src=Pointer[UInt8, MutAnyOrigin](
                            unsafe_from_address=found[0]
                        ),
                        count=length,
                    )
            else:
                unsafe_memcpy(
                    dest=Pointer[UInt8, MutAnyOrigin](
                        unsafe_from_address=at + 4
                    ),
                    src=Pointer[UInt8, MutAnyOrigin](
                        unsafe_from_address=found[0]
                    ),
                    count=4,
                )
                _store64(at + 8, UInt64(found[0]))
        return
    if kind == _WIDE:
        var values = key._data[Column[Int128]]._ptr()
        for i in range(count):
            var value = bitcast[DType.uint128](values[unsafe_offset=start + i])
            _store64(base + i * stride, UInt64(value))
            _store64(base + i * stride + 8, UInt64(value >> 64))
        return
    if key._data.isa[BoolColumn]():
        ref bools = key._data[BoolColumn]
        for i in range(count):
            _store64(
                base + i * stride,
                UInt64(1) if bools._get(start + i) else UInt64(2),
            )
        return
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if key._data.isa[Column[Scalar[D]]]():
            var values = key._data[Column[Scalar[D]]]._ptr()
            for i in range(count):
                var value = values[unsafe_offset=start + i]
                var word: UInt64

                comptime if D.is_floating_point():
                    word = float_key(Float64(value))
                elif D.is_unsigned():
                    word = UInt64(value)
                else:
                    word = bitcast[DType.uint64](Int64(value))
                _store64(base + i * stride, word)
            return
    raise Error("unsupported join key type: " + key.dtype().name())


@fieldwise_init
struct KeyLayout(Copyable, Movable):
    """The key fields of a row (DuckDB's TupleDataLayout over the equality
    columns) and the row's other offsets."""

    var kinds: List[Int]
    # Each field's offset from the first key field.
    var offsets: List[Int]
    var key_bytes: Int
    var text: Bool
    var index_offset: Int
    var valid_offset: Int
    var width: Int

    @staticmethod
    def of(keys: List[Series]) -> KeyLayout:
        var kinds = List[Int](capacity=len(keys))
        var offsets = List[Int](capacity=len(keys))
        var key_bytes = 0
        var text = False
        for key in keys:
            var kind = _field_kind(key)
            kinds.append(kind)
            offsets.append(key_bytes)
            key_bytes += _field_bytes(kind)
            text = text or kind == _TEXT
        var index_offset = _KEYS + key_bytes
        var valid_offset = index_offset + 4
        var width = (valid_offset + 1 + 7) // 8 * 8
        return KeyLayout(
            kinds^,
            offsets^,
            key_bytes,
            text,
            index_offset,
            valid_offset,
            width,
        )


@always_inline
def _same_text(a: Int, b: Int) -> Bool:
    """Whether two string_t fields hold the same string."""
    var head = _load64(a)
    if head != _load64(b):
        return False
    var length = Int(head & 0xFFFFFFFF)
    if length <= _INLINE:
        return _load64(a + 8) == _load64(b + 8)
    return (
        external_call["memcmp", Int32](
            Int(_load64(a + 8)), Int(_load64(b + 8)), length
        )
        == 0
    )


@always_inline
def _same_keys(
    kinds: Int, offsets: Int, fields: Int, key_bytes: Int, text: Bool, a: Int, b: Int
) -> Bool:
    """Whether the key fields at `a` and `b` are equal (RowMatcher's
    equality over every key column). `kinds` and `offsets` address the
    layout's lists."""
    if not text:
        var w = 0
        while w < key_bytes:
            if _load64(a + w) != _load64(b + w):
                return False
            w += 8
        return True
    var kind = Pointer[Int, MutAnyOrigin](unsafe_from_address=kinds)
    var offset = Pointer[Int, MutAnyOrigin](unsafe_from_address=offsets)
    for f in range(fields):
        var at = offset[unsafe_offset=f]
        var k = kind[unsafe_offset=f]
        if k == _TEXT:
            if not _same_text(a + at, b + at):
                return False
        elif k == _WIDE:
            if _load64(a + at) != _load64(b + at) or _load64(
                a + at + 8
            ) != _load64(b + at + 8):
                return False
        elif _load64(a + at) != _load64(b + at):
            return False
    return True


struct _TableData(Movable):
    """What a table owns: its rows, its pointer table, the flags of the build
    rows a probe matched, and the build keys whose string bytes rows point
    into."""

    var rows: List[UInt8]
    var entries: List[UInt64]
    var found: List[UInt8]
    var keys: List[Series]
    var slots: List[Int32]

    def __init__(out self, var keys: List[Series]):
        self.rows = List[UInt8]()
        self.entries = List[UInt64]()
        self.found = List[UInt8]()
        self.keys = keys^
        self.slots = List[Int32]()


struct _ScatterJob(Job):
    """Lay out rows [start, end) of the build: header, key fields, hash."""

    var keys: List[Series]
    var layout: KeyLayout
    var rows: Int
    var start: Int
    var end: Int
    var invalid: Int

    def __init__(
        out self,
        keys: List[Series],
        layout: KeyLayout,
        rows: Int,
        start: Int,
        end: Int,
    ):
        self.keys = keys.copy()
        self.layout = layout.copy()
        self.rows = rows
        self.start = start
        self.end = end
        self.invalid = 0

    def run(mut self) raises:
        var width = self.layout.width
        var hashes = List[UInt64](unsafe_uninit_length=STANDARD_VECTOR_SIZE)
        var hashed = hashes.unsafe_ptr()
        var low = self.start
        while low < self.end:
            var count = min(STANDARD_VECTOR_SIZE, self.end - low)
            var first = self.rows + low * width
            for i in range(count):
                var row = first + i * width
                _store32(row + self.layout.index_offset, UInt32(low + i))
                _store8(row + self.layout.valid_offset, 1)
            for k in range(len(self.keys)):
                _write_fields(
                    self.keys[k],
                    low,
                    count,
                    self.layout.kinds[k],
                    first + _KEYS + self.layout.offsets[k],
                    width,
                    first + self.layout.valid_offset,
                    width,
                )
                _hash_column(
                    self.keys[k],
                    low,
                    low + count,
                    Int(hashed),
                    k == 0,
                    output_offset=low,
                )
            for i in range(count):
                var row = first + i * width
                _store64(row + _NEXT, hashed[unsafe_offset=i])
                if _load8(row + self.layout.valid_offset) == 0:
                    self.invalid += 1
            low += count


struct _InsertJob(Job):
    """Insert rows [start, end) into the pointer table (DuckDB's
    InsertHashesLoop); `parallel` inserts with compare-and-swap."""

    var layout: KeyLayout
    var rows: Int
    var entries: Int
    var bitmask: UInt64
    var start: Int
    var end: Int
    var parallel: Bool
    var chained: Bool

    def __init__(
        out self,
        layout: KeyLayout,
        rows: Int,
        entries: Int,
        bitmask: UInt64,
        start: Int,
        end: Int,
        parallel: Bool,
    ):
        self.layout = layout.copy()
        self.rows = rows
        self.entries = entries
        self.bitmask = bitmask
        self.start = start
        self.end = end
        self.parallel = parallel
        self.chained = False

    def run(mut self) raises:
        var width = self.layout.width
        var valid_offset = self.layout.valid_offset
        var kinds = Int(self.layout.kinds.unsafe_ptr())
        var offsets = Int(self.layout.offsets.unsafe_ptr())
        var fields = len(self.layout.kinds)
        var key_bytes = self.layout.key_bytes
        var text = self.layout.text
        var table = Pointer[UInt64, MutAnyOrigin](unsafe_from_address=self.entries)
        # Descending on one thread, so every chain lists its rows in
        # ascending order.
        var r = self.end - 1
        while r >= self.start:
            var row = self.rows + r * width
            r -= 1
            if _load8(row + valid_offset) == 0:
                continue
            var hash = _load64(row + _NEXT)
            var offset = Int(hash & self.bitmask)
            var salt = hash | POINTER_MASK
            var desired = (hash & SALT_MASK) | UInt64(row)
            while True:
                if not self.parallel:
                    var entry = table[unsafe_offset=offset]
                    if entry == 0:
                        _store64(row + _NEXT, 0)
                        table[unsafe_offset=offset] = desired
                        break
                    if (entry | POINTER_MASK) == salt:
                        var other = Int(entry & POINTER_MASK)
                        if _same_keys(
                            kinds,
                            offsets,
                            fields,
                            key_bytes,
                            text,
                            other + _KEYS,
                            row + _KEYS,
                        ):
                            _store64(row + _NEXT, UInt64(other))
                            table[unsafe_offset=offset] = desired
                            self.chained = True
                            break
                    offset = Int((UInt64(offset) + 1) & self.bitmask)
                    continue
                ref atom = Pointer[Atomic[Int64], MutAnyOrigin](
                    unsafe_from_address=self.entries + 8 * offset
                )[]
                var entry = bitcast[DType.uint64](atom.load())
                if entry == 0:
                    # Expect an empty entry: on a lost race, the row that
                    # took it is compared as an occupied entry.
                    _store64(row + _NEXT, 0)
                    var expected = Int64(0)
                    if atom.compare_exchange(
                        expected, bitcast[DType.int64](desired)
                    ):
                        break
                    entry = bitcast[DType.uint64](expected)
                if (entry | POINTER_MASK) == salt and _same_keys(
                    kinds,
                    offsets,
                    fields,
                    key_bytes,
                    text,
                    Int(entry & POINTER_MASK) + _KEYS,
                    row + _KEYS,
                ):
                    # Same keys: push onto the chain's head until it holds.
                    var current = entry
                    while True:
                        _store64(row + _NEXT, current & POINTER_MASK)
                        var expected = bitcast[DType.int64](current)
                        if atom.compare_exchange(
                            expected, bitcast[DType.int64](desired)
                        ):
                            break
                        current = bitcast[DType.uint64](expected)
                    self.chained = True
                    break
                offset = Int((UInt64(offset) + 1) & self.bitmask)


@always_inline
def _word[D: DType](value: Scalar[D]) -> UInt64:
    """An integer as the 64-bit word its row field holds: unsigned values
    zero-extended, signed ones sign-extended (two's complement), so the
    unsigned difference of two words is their distance."""
    comptime if D.is_unsigned():
        return UInt64(value)
    else:
        return bitcast[DType.uint64](Int64(value))


def _perfect_slots(
    key: Series, inserted: Int
) -> Optional[Tuple[UInt64, Int, List[Int32]]]:
    """PerfectHashJoinExecutor::CanDoPerfectHashJoin and its build: an
    integer key whose valid values span at most PERFECT_MAX_BUILD_SIZE, with
    no more of them than the span (DuckDB's `Count() > build_range` check),
    each value once. Gives the least value's word, the span and each slot's
    build row (-1 for none); None otherwise."""
    var bits = _validity(key)
    var bitmap = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=bits[0])
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        comptime if not D.is_floating_point():
            if key._data.isa[Column[Scalar[D]]]():
                var values = key._data[Column[Scalar[D]]]._ptr()
                var n = len(key)
                var least = Scalar[D].MAX
                var most = Scalar[D].MIN
                for i in range(n):
                    if bits[0] != 0 and not _validity_at(bitmap, bits[1] + i):
                        continue
                    var value = values[unsafe_offset=i]
                    least = min(least, value)
                    most = max(most, value)
                if least > most:
                    return None
                var base = _word[D](least)
                var span = _word[D](most) - base
                if span > UInt64(PERFECT_MAX_BUILD_SIZE):
                    return None
                if UInt64(inserted) > span:
                    return None
                var slots = List[Int32](length=Int(span) + 1, fill=-1)
                var table = slots.unsafe_ptr()
                for i in range(n):
                    if bits[0] != 0 and not _validity_at(bitmap, bits[1] + i):
                        continue
                    var at = Int(_word[D](values[unsafe_offset=i]) - base)
                    if table[unsafe_offset=at] >= 0:
                        return None
                    table[unsafe_offset=at] = Int32(i)
                return (base, Int(span), slots^)
    return None


def _perfect_heads(
    key: Series,
    start: Int,
    count: Int,
    base: UInt64,
    span: Int,
    slots: Int,
    rows: Int,
    width: Int,
    heads: Int,
):
    """Each probe row's build row, as the address of its row (0 for none),
    looked up by `key - base` (PerfectHashJoinExecutor's probe)."""
    var bits = _validity(key)
    var bitmap = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=bits[0])
    var table = Pointer[Int32, MutAnyOrigin](unsafe_from_address=slots)
    var out = Pointer[UInt64, MutAnyOrigin](unsafe_from_address=heads)
    var limit = UInt64(span)
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        comptime if not D.is_floating_point():
            if key._data.isa[Column[Scalar[D]]]():
                var values = key._data[Column[Scalar[D]]]._ptr()
                for i in range(count):
                    var at = _word[D](values[unsafe_offset=start + i]) - base
                    if at > limit:
                        continue
                    if bits[0] != 0 and not _validity_at(
                        bitmap, bits[1] + start + i
                    ):
                        continue
                    var b = table[unsafe_offset=Int(at)]
                    if b >= 0:
                        out[unsafe_offset=i] = UInt64(rows + Int(b) * width)
                return


def _next_power_of_two(value: Int) -> Int:
    var power = 1
    while power < value:
        power <<= 1
    return power


def pointer_table_capacity(count: Int) -> Int:
    """JoinHashTable::PointerTableCapacity."""
    return max(
        _next_power_of_two(Int(Float64(count) * DEFAULT_LOAD_FACTOR)),
        MINIMUM_CAPACITY,
    )


struct JoinHashTable(Copyable, Movable):
    """A finalized join hash table over a build side's key columns, shared
    by every probing thread (copies share the same rows and entries)."""

    var data: ArcPointer[_TableData]
    var layout: KeyLayout
    # DuckDB's PerfectHashJoinExecutor: an inner join on one integer key
    # whose build keys are unique within PERFECT_MAX_BUILD_SIZE values
    # looks each probe key up at `key - perfect_base` in `data.slots`
    # (the build row, -1 for none) instead of the pointer table.
    var perfect: Bool
    var perfect_base: UInt64
    var perfect_span: Int
    # Build rows, and those whose keys are all valid (in the table).
    var count: Int
    var inserted: Int
    var capacity: Int
    var bitmask: UInt64
    var chains_longer_than_one: Bool
    # Whether a build key is null (a mark join's unknown results).
    var has_null: Bool

    def __init__(out self, keys: List[Series], perfect: Bool = False) raises:
        """Build and finalize the table over `keys` (one column per
        equality condition, all of one length). `perfect`: the join is an
        inner join, which may run as a perfect hash join."""
        var unchunked = List[Series](capacity=len(keys))
        for key in keys:
            if key.dtype().is_nested():
                raise Error(
                    "list and struct columns cannot be join keys: "
                    + key.name()
                )
            unchunked.append(key.rechunk() if key.is_chunked() else key.copy())
        self.layout = KeyLayout.of(unchunked)
        self.count = len(unchunked[0]) if len(unchunked) > 0 else 0
        if self.count > Int(Int32.MAX):
            raise Error("A join build exceeds 2^31 - 1 rows")
        self.data = ArcPointer(_TableData(unchunked.copy()))
        self.data[].rows = huge_uninit[UInt8](
            max(self.count, 1) * self.layout.width
        )
        self.data[].found = List[UInt8](length=max(self.count, 1), fill=0)
        var rows = Int(self.data[].rows.unsafe_ptr())
        # Sink: every row laid out and hashed, on the crew.
        var workers = worker_count(self.count)
        var bounds = partitions(self.count, workers, STANDARD_VECTOR_SIZE)
        var scatter = List[_ScatterJob](capacity=workers)
        for w in range(workers):
            if bounds[w] < bounds[w + 1]:
                scatter.append(
                    _ScatterJob(
                        unchunked, self.layout, rows, bounds[w], bounds[w + 1]
                    )
                )
        run_jobs(scatter)
        var invalid = 0
        for w in range(len(scatter)):
            invalid += scatter[w].invalid
        self.has_null = invalid > 0
        self.inserted = self.count - invalid
        self.perfect = False
        self.perfect_base = 0
        self.perfect_span = 0
        self.chains_longer_than_one = False
        self.capacity = 0
        self.bitmask = 0
        if perfect and len(unchunked) == 1 and self.inserted > 0:
            # PerfectHashJoinExecutor::BuildPerfectHashTable, else (a
            # repeated key, a wide range) the hash table.
            var slots = _perfect_slots(unchunked[0], self.inserted)
            if slots:
                trace_path("join.perfect_hash")
                self.perfect = True
                self.perfect_base = slots.value()[0]
                self.perfect_span = slots.value()[1]
                self.data[].slots = slots.value()[2].copy()
                # Each row ends its chain: a probe emits it alone.
                for r in range(self.count):
                    _store64(rows + r * self.layout.width + _NEXT, 0)
                return
        # Finalize: the pointer table, then every row inserted.
        self.capacity = pointer_table_capacity(self.inserted)
        self.bitmask = UInt64(self.capacity - 1)
        self.data[].entries = huge_list[UInt64](self.capacity, 0)
        var entries = Int(self.data[].entries.unsafe_ptr())
        var parallel = (
            self.count >= PARALLEL_CONSTRUCT_THRESHOLD and workers > 1
        )
        var inserts = List[_InsertJob](capacity=workers)
        if parallel:
            for w in range(workers):
                if bounds[w] < bounds[w + 1]:
                    inserts.append(
                        _InsertJob(
                            self.layout,
                            rows,
                            entries,
                            self.bitmask,
                            bounds[w],
                            bounds[w + 1],
                            True,
                        )
                    )
            run_jobs(inserts)
        else:
            var job = _InsertJob(
                self.layout, rows, entries, self.bitmask, 0, self.count, False
            )
            job.run()
            inserts.append(job^)
        for w in range(len(inserts)):
            self.chains_longer_than_one = (
                self.chains_longer_than_one or inserts[w].chained
            )

    def keys(self) -> List[Series]:
        return self.data[].keys.copy()

    def rows_address(self) -> Int:
        return Int(self.data[].rows.unsafe_ptr())

    def entries_address(self) -> Int:
        return Int(self.data[].entries.unsafe_ptr())

    def found_address(self) -> Int:
        """One byte per build row, set once a probe matched it."""
        return Int(self.data[].found.unsafe_ptr())

    def reset_found(self):
        """Clear every build row's match flag before a new probe phase."""
        var found = self.data[].found.unsafe_ptr()
        for i in range(len(self.data[].found)):
            found[unsafe_offset=i] = 0

    def use_salt(self) -> Bool:
        """JoinHashTable::UseSalt."""
        return self.capacity > USE_SALT_THRESHOLD

    def build_row(self, pointer: UInt64) -> Int:
        """The build row of the row at `pointer`."""
        return Int(_load32(Int(pointer) + self.layout.index_offset))

    def scan_full_outer(
        self, mut position: Int, want: UInt8, mut rows: List[Int]
    ) -> Int:
        """DuckDB's ScanFullOuter: from build row `position` on, append up to
        STANDARD_VECTOR_SIZE build rows whose match flag is `want` (0: the
        rows no probe matched, for right, full and right-anti joins; 1:
        the matched ones, for right-semi joins) to `rows`, and advance
        `position`. Returns how many were appended."""
        var found = Pointer[UInt8, MutAnyOrigin](
            unsafe_from_address=self.found_address()
        )
        var appended = 0
        while position < self.count and appended < STANDARD_VECTOR_SIZE:
            if found[unsafe_offset=position] == want:
                rows.append(position)
                appended += 1
            position += 1
        return appended


struct ScanStructure(Movable):
    """One thread's probe of a JoinHashTable, a chunk of at most
    STANDARD_VECTOR_SIZE probe rows at a time (DuckDB's ScanStructure).

    `probe` hashes the chunk's keys and finds each row's chain
    (`GetRowPointers`); `next` then emits the join's rows for the chunk,
    at most STANDARD_VECTOR_SIZE per call, into `probe_rows` (the chunk's
    row, from 0) and `build_rows` (the build row, -1 for none), until it
    returns 0. A mark join emits every probe row, its result in `marks`
    (1 matched, 0 not, 2 unknown)."""

    var table: JoinHashTable
    var how: Int
    var count: Int
    var hashes: List[UInt64]
    var valid: List[UInt8]
    var records: List[UInt8]
    var heads: List[UInt64]
    var offsets: List[UInt64]
    var todo: List[Int32]
    var compare: List[Int32]
    var no_match: List[Int32]
    # Emission state: the probe row emitting, and its chain's next row.
    var row: Int
    var cursor: UInt64
    var started: Bool
    var finished: Bool
    var probe_rows: List[Int32]
    var build_rows: List[Int32]
    var marks: List[UInt8]
    # The join flags matched build rows (right, full, right-semi/anti).
    var propagate: Bool

    def __init__(out self, table: JoinHashTable, how: Int):
        self.table = table.copy()
        self.how = how
        self.count = 0
        var n = STANDARD_VECTOR_SIZE
        self.hashes = List[UInt64](unsafe_uninit_length=n)
        self.valid = List[UInt8](unsafe_uninit_length=n)
        self.records = List[UInt8](
            unsafe_uninit_length=n * max(table.layout.key_bytes, 8)
        )
        self.heads = List[UInt64](length=n, fill=0)
        self.offsets = List[UInt64](unsafe_uninit_length=n)
        self.todo = List[Int32](unsafe_uninit_length=n)
        self.compare = List[Int32](unsafe_uninit_length=n)
        self.no_match = List[Int32](unsafe_uninit_length=n)
        self.row = 0
        self.cursor = 0
        self.started = False
        self.finished = True
        self.probe_rows = List[Int32](unsafe_uninit_length=n)
        self.build_rows = List[Int32](unsafe_uninit_length=n)
        self.marks = List[UInt8](unsafe_uninit_length=n)
        self.propagate = (
            how == JOIN_RIGHT
            or how == JOIN_FULL
            or how == JOIN_RIGHT_SEMI
            or how == JOIN_RIGHT_ANTI
        )

    def probe(mut self, keys: List[Series], start: Int, count: Int) raises:
        """Probe rows [start, start + count) of the unchunked `keys` (at
        most STANDARD_VECTOR_SIZE rows): JoinHashTable::Probe."""
        self.count = count
        self.row = 0
        self.cursor = 0
        self.started = False
        self.finished = False
        ref layout = self.table.layout
        var heads = self.heads.unsafe_ptr()
        for i in range(count):
            heads[unsafe_offset=i] = 0
        if self.table.perfect:
            # ProbePerfectHashTable: the key's slot holds its build row.
            _perfect_heads(
                keys[0],
                start,
                count,
                self.table.perfect_base,
                self.table.perfect_span,
                Int(self.table.data[].slots.unsafe_ptr()),
                self.table.rows_address(),
                layout.width,
                Int(heads),
            )
            return
        var valid = self.valid.unsafe_ptr()
        for i in range(count):
            valid[unsafe_offset=i] = 1
        var records = Int(self.records.unsafe_ptr())
        var key_bytes = layout.key_bytes
        for k in range(len(keys)):
            _write_fields(
                keys[k],
                start,
                count,
                layout.kinds[k],
                records + layout.offsets[k],
                key_bytes,
                Int(valid),
                1,
            )
            _hash_column(
                keys[k],
                start,
                start + count,
                Int(self.hashes.unsafe_ptr()),
                k == 0,
                output_offset=start,
            )
        if self.table.inserted == 0:
            return
        # PrepareKeys: rows with a null key match nothing.
        var todo = self.todo.unsafe_ptr()
        var pending = 0
        for i in range(count):
            todo[unsafe_offset=pending] = Int32(i)
            pending += Int(valid[unsafe_offset=i])
        self._row_pointers(pending)

    def _row_pointers(mut self, var pending: Int):
        """GetRowPointers: each pending row's chain, by linear probing to an
        empty entry or one whose salt matches and whose keys are equal."""
        var table = Pointer[UInt64, MutAnyOrigin](
            unsafe_from_address=self.table.entries_address()
        )
        var bitmask = self.table.bitmask
        var salted = self.table.use_salt()
        var hashes = self.hashes.unsafe_ptr()
        var offsets = self.offsets.unsafe_ptr()
        var heads = self.heads.unsafe_ptr()
        var todo = self.todo.unsafe_ptr()
        var compare = self.compare.unsafe_ptr()
        var no_match = self.no_match.unsafe_ptr()
        ref layout = self.table.layout
        var kinds = Int(layout.kinds.unsafe_ptr())
        var key_offsets = Int(layout.offsets.unsafe_ptr())
        var fields = len(layout.kinds)
        var key_bytes = layout.key_bytes
        var text = layout.text
        var records = Int(self.records.unsafe_ptr())
        for t in range(pending):
            var i = Int(todo[unsafe_offset=t])
            offsets[unsafe_offset=i] = hashes[unsafe_offset=i] & bitmask
        while pending > 0:
            # ProbeForPointers: walk to an empty entry or a salt match.
            var candidates = 0
            for t in range(pending):
                var i = Int(todo[unsafe_offset=t])
                var hash = hashes[unsafe_offset=i]
                var salt = hash | POINTER_MASK
                var offset = offsets[unsafe_offset=i]
                while True:
                    var entry = table[unsafe_offset=Int(offset)]
                    if entry == 0:
                        break
                    if not salted or (entry | POINTER_MASK) == salt:
                        heads[unsafe_offset=i] = entry & POINTER_MASK
                        compare[unsafe_offset=candidates] = Int32(i)
                        candidates += 1
                        break
                    offset = (offset + 1) & bitmask
                offsets[unsafe_offset=i] = offset
            # RowMatcher: compare the candidates' keys; a mismatch goes on
            # probing from the next entry.
            var misses = 0
            if not text and key_bytes == 8:
                for c in range(candidates):
                    var i = Int(compare[unsafe_offset=c])
                    var head = Int(heads[unsafe_offset=i])
                    if _load64(head + _KEYS) != _load64(records + i * 8):
                        heads[unsafe_offset=i] = 0
                        no_match[unsafe_offset=misses] = Int32(i)
                        misses += 1
            else:
                for c in range(candidates):
                    var i = Int(compare[unsafe_offset=c])
                    var head = Int(heads[unsafe_offset=i])
                    if not _same_keys(
                        kinds,
                        key_offsets,
                        fields,
                        key_bytes,
                        text,
                        head + _KEYS,
                        records + i * key_bytes,
                    ):
                        heads[unsafe_offset=i] = 0
                        no_match[unsafe_offset=misses] = Int32(i)
                        misses += 1
            for m in range(misses):
                var i = Int(no_match[unsafe_offset=m])
                offsets[unsafe_offset=i] = (
                    offsets[unsafe_offset=i] + 1
                ) & bitmask
                todo[unsafe_offset=m] = Int32(i)
            pending = misses

    def matched(self, i: Int) -> Bool:
        """Whether chunk row i found a chain."""
        return self.heads[i] != 0

    def next(mut self) -> Int:
        """The next at most STANDARD_VECTOR_SIZE output rows of the probed
        chunk (DuckDB's ScanStructure::Next), or 0 once it is done."""
        if self.finished:
            return 0
        var heads = self.heads.unsafe_ptr()
        var out_probe = self.probe_rows.unsafe_ptr()
        var out_build = self.build_rows.unsafe_ptr()
        var index_offset = self.table.layout.index_offset
        var found = Pointer[UInt8, MutAnyOrigin](
            unsafe_from_address=self.table.found_address()
        )
        var out = 0
        if (
            self.how == JOIN_SEMI
            or self.how == JOIN_ANTI
            or self.how == JOIN_MARK
        ):
            # NextSemiJoin / NextAntiJoin / NextMarkJoin: one pass.
            var keep = self.how == JOIN_SEMI
            var unknown = self.table.has_null
            var valid = self.valid.unsafe_ptr()
            var marks = self.marks.unsafe_ptr()
            for i in range(self.count):
                var hit = heads[unsafe_offset=i] != 0
                if self.how == JOIN_MARK:
                    out_probe[unsafe_offset=out] = Int32(i)
                    out_build[unsafe_offset=out] = -1
                    marks[unsafe_offset=out] = UInt8(1) if hit else (
                        UInt8(2) if (
                            unknown or valid[unsafe_offset=i] == 0
                        ) else UInt8(0)
                    )
                    out += 1
                elif hit == keep:
                    out_probe[unsafe_offset=out] = Int32(i)
                    out_build[unsafe_offset=out] = -1
                    out += 1
            self.finished = True
            return out
        if self.how == JOIN_RIGHT_SEMI or self.how == JOIN_RIGHT_ANTI:
            # NextRightSemiOrAntiJoin: flag every row of each matched chain;
            # a chain whose head is flagged was flagged whole before.
            for i in range(self.count):
                var p = heads[unsafe_offset=i]
                if p == 0:
                    continue
                if found[unsafe_offset=Int(_load32(Int(p) + index_offset))] != 0:
                    continue
                while p != 0:
                    found[unsafe_offset=Int(_load32(Int(p) + index_offset))] = 1
                    p = _load64(Int(p) + _NEXT)
            self.finished = True
            return 0
        # NextInnerJoin / NextLeftJoin, one probe row's chain at a time.
        var outer = self.how == JOIN_LEFT or self.how == JOIN_FULL
        var flag = self.propagate
        while self.row < self.count:
            if out == STANDARD_VECTOR_SIZE:
                return out
            var i = self.row
            if not self.started:
                self.started = True
                self.cursor = heads[unsafe_offset=i]
                if self.cursor == 0:
                    if outer:
                        out_probe[unsafe_offset=out] = Int32(i)
                        out_build[unsafe_offset=out] = -1
                        out += 1
                    self.row += 1
                    self.started = False
                    continue
            var p = self.cursor
            while p != 0 and out < STANDARD_VECTOR_SIZE:
                var b = _load32(Int(p) + index_offset)
                out_probe[unsafe_offset=out] = Int32(i)
                out_build[unsafe_offset=out] = Int32(b)
                if flag:
                    found[unsafe_offset=Int(b)] = 1
                out += 1
                p = _load64(Int(p) + _NEXT)
            self.cursor = p
            if p == 0:
                self.row += 1
                self.started = False
        self.finished = True
        return out


struct _ProbeJob(Job):
    """One thread's probe of rows [start, end), chunk by chunk, keeping the
    pairs it emits (absolute probe rows; build rows, -1 for none)."""

    var table: JoinHashTable
    var keys: List[Series]
    var how: Int
    var start: Int
    var end: Int
    var both: Bool
    var probe: List[Int]
    var build: List[Int]

    def __init__(
        out self,
        table: JoinHashTable,
        keys: List[Series],
        how: Int,
        start: Int,
        end: Int,
    ):
        self.table = table.copy()
        self.keys = keys.copy()
        self.how = how
        self.start = start
        self.end = end
        self.both = not (
            how == JOIN_SEMI or how == JOIN_ANTI or how == JOIN_MARK
        )
        self.probe = List[Int]()
        self.build = List[Int]()

    def run(mut self) raises:
        var scan = ScanStructure(self.table, self.how)
        var low = self.start
        while low < self.end:
            var count = min(STANDARD_VECTOR_SIZE, self.end - low)
            scan.probe(self.keys, low, count)
            while True:
                var n = scan.next()
                if n == 0:
                    break
                var rows = scan.probe_rows.unsafe_ptr()
                var builds = scan.build_rows.unsafe_ptr()
                # Grown geometrically: an exact reservation per chunk
                # copied the list again for every chunk.
                if self.probe.capacity() < len(self.probe) + n:
                    self.probe.reserve(
                        max(len(self.probe) + n, 2 * self.probe.capacity())
                    )
                for k in range(n):
                    self.probe.append(low + Int(rows[unsafe_offset=k]))
                if self.both:
                    if self.build.capacity() < len(self.build) + n:
                        self.build.reserve(
                            max(len(self.build) + n, 2 * self.build.capacity())
                        )
                    for k in range(n):
                        self.build.append(Int(builds[unsafe_offset=k]))
            low += count


struct _AppendJob(Job):
    """Copy one probe job's pairs to their place in the joined lists."""

    var source: Int
    var target: Int
    var at: Int
    var count: Int

    def __init__(out self, source: Int, target: Int, at: Int, count: Int):
        self.source = source
        self.target = target
        self.at = at
        self.count = count

    def run(mut self) raises:
        if self.count == 0:
            return
        unsafe_memcpy(
            dest=Pointer[Int, MutAnyOrigin](
                unsafe_from_address=self.target
            ).unsafe_offset(self.at),
            src=Pointer[Int, MutAnyOrigin](unsafe_from_address=self.source),
            count=self.count,
        )


def probe_pairs(
    table: JoinHashTable,
    keys: List[Series],
    how: Int,
    workers: Int,
    scan: Bool = True,
) raises -> Tuple[List[Int], List[Int]]:
    """Every row `keys` (the probe side's key columns) joins against
    `table` with physical join type `how`: the probe rows and the build
    rows (-1 for none), in probe order, on `workers` threads (0: by the
    row count). A semi, anti or mark join gives probe rows only. With
    `scan`, a right or full join then appends the build rows no probe
    matched (probe row -1), and a right-semi or right-anti join gives its
    build rows only (`scan_full_outer`)."""
    var unchunked = List[Series](capacity=len(keys))
    for key in keys:
        unchunked.append(key.rechunk() if key.is_chunked() else key.copy())
    var n = len(unchunked[0]) if len(unchunked) > 0 else 0
    var lanes = workers if workers > 0 else worker_count(n)
    var bounds = partitions(n, lanes, STANDARD_VECTOR_SIZE)
    var jobs = List[_ProbeJob](capacity=lanes)
    for w in range(lanes):
        if bounds[w] < bounds[w + 1]:
            jobs.append(
                _ProbeJob(table, unchunked, how, bounds[w], bounds[w + 1])
            )
    if len(jobs) == 1:
        jobs[0].run()
    elif len(jobs) > 1:
        run_jobs(jobs)
    var probe = List[Int]()
    var build = List[Int]()
    if len(jobs) == 1:
        swap(probe, jobs[0].probe)
        swap(build, jobs[0].build)
    elif len(jobs) > 1:
        var total = 0
        for w in range(len(jobs)):
            total += len(jobs[w].probe)
        probe = huge_uninit[Int](total)
        var both = jobs[0].both
        if both:
            build = huge_uninit[Int](total)
        var copies = List[_AppendJob](capacity=2 * len(jobs))
        var at = 0
        for w in range(len(jobs)):
            ref job = jobs[w]
            copies.append(
                _AppendJob(
                    Int(job.probe.unsafe_ptr()),
                    Int(probe.unsafe_ptr()),
                    at,
                    len(job.probe),
                )
            )
            if both:
                copies.append(
                    _AppendJob(
                        Int(job.build.unsafe_ptr()),
                        Int(build.unsafe_ptr()),
                        at,
                        len(job.build),
                    )
                )
            at += len(job.probe)
        run_jobs(copies)
    _ = jobs^
    if not scan:
        return (probe^, build^)
    if how == JOIN_RIGHT or how == JOIN_FULL:
        var position = 0
        while table.scan_full_outer(position, 0, build) > 0:
            pass
        while len(probe) < len(build):
            probe.append(-1)
    elif how == JOIN_RIGHT_SEMI or how == JOIN_RIGHT_ANTI:
        var rows = List[Int]()
        var position = 0
        var want = UInt8(1) if how == JOIN_RIGHT_SEMI else UInt8(0)
        while table.scan_full_outer(position, want, rows) > 0:
            pass
        return (List[Int](), rows^)
    return (probe^, build^)
