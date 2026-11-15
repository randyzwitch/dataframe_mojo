"""Partition and stably order rows for window evaluation.

Hash the partition keys, order each partition by its sort keys and record
partition and peer boundaries. Stable ordering keeps input order among
equal keys, which makes positional window functions deterministic. The
result maps computed values back to the original row positions.
"""
from std.memory import Pointer, unsafe_memcmp, unsafe_memcpy
from std.sys import size_of

from dataframe.storage.allocation import huge_uninit
from dataframe.storage.boolean import BoolColumn
from dataframe.storage.column import Column
from dataframe.storage.string import StringColumn
from dataframe.dtype import NUMERIC_DTYPES
from dataframe.kernels.group_rows import (
    GroupRows,
    ROW_INDEX,
    WIDE_ROW_INDEX,
    group_rows_of,
    wide_row_indices,
)
from dataframe.kernels.partition import Partitioner, _same_value
from dataframe.aggregation.reducer import float_key
from dataframe.parallel import Job, configured_workers, partitions, run_jobs
from dataframe.series import Series
from dataframe.sorting.sort import sort_rows
from dataframe.kernels.binary import validity
from dataframe.sorting.sort_key import STANDARD_VECTOR_SIZE

# Boundary flags per sorted position.
comptime PARTITION_START = UInt8(1)
comptime PEER_START = UInt8(2)


struct WindowLayout(Movable):
    """Rows grouped by partition, each partition ordered by the order
    keys, with the partition and peer boundaries of the sorted order."""

    # Sorted position -> input row.
    var order: List[Int]
    # Input row -> sorted position (empty when `in_place`).
    var inverse: List[Int]
    # Whether `order` is the identity: nothing to gather or scatter.
    var in_place: Bool
    # Per sorted position: PARTITION_START and PEER_START flags.
    var flags: List[UInt8]
    # Where each partition starts in the sorted order (count + 1 entries).
    var starts: List[Int]
    var count: Int
    # Partition index bounds of the runs of whole partitions that were
    # ordered by one task (runs + 1 entries).
    var group_parts: List[Int]

    def __init__(out self):
        self.order = List[Int]()
        self.inverse = List[Int]()
        self.in_place = True
        self.flags = List[UInt8]()
        self.starts = List[Int]()
        self.count = 0
        self.group_parts = List[Int]()

    def row_partitions(self) -> List[Int]:
        """Input row -> its partition's index (in sorted partition order),
        for results with one value per partition."""
        var rows = List[Int](unsafe_uninit_length=len(self.order))
        var rp = rows.unsafe_ptr()
        var op = self.order.unsafe_ptr()
        for p in range(self.count):
            for k in range(self.starts[p], self.starts[p + 1]):
                rp[unsafe_offset=op[unsafe_offset=k]] = p
        return rows^

    def partition_ids(self) -> List[Int]:
        """Each sorted position's partition index."""
        var ids = List[Int](unsafe_uninit_length=len(self.order))
        var p = ids.unsafe_ptr()
        for g in range(self.count):
            for k in range(self.starts[g], self.starts[g + 1]):
                p[k] = g
        return ids^


@always_inline
def _row_at(rows: Int, k: Int) -> Int:
    """Sorted position k's row: rows[k], or k when `rows` is 0."""
    if rows == 0:
        return k
    return Pointer[Int, MutAnyOrigin](unsafe_from_address=rows)[unsafe_offset=k]


@always_inline
def _numeric_changes[
    D: DType
](
    column: Column[Scalar[D]],
    flags: Pointer[UInt8, MutAnyOrigin],
    lo: Int,
    hi: Int,
    bit: UInt8,
    rows: Int,
):
    """Mark positions of [lo + 1, hi) whose value differs from the one
    before (nulls equal to each other; floats as the sort compares them:
    every NaN one value, -0.0 equal to 0.0)."""
    var values = column._ptr()
    var bits = len(column._bits[]) != 0
    var previous = _row_at(rows, lo)
    for k in range(lo + 1, hi):
        var row = _row_at(rows, k)
        var differs: Bool
        comptime if D.is_floating_point():
            differs = float_key(
                Float64(values[unsafe_offset=row])
            ) != float_key(Float64(values[unsafe_offset=previous]))
        else:
            differs = (
                values[unsafe_offset=row] != values[unsafe_offset=previous]
            )
        if bits:
            var a = column._valid(previous)
            var b = column._valid(row)
            differs = a != b or (b and differs)
        if differs:
            flags[unsafe_offset=k] |= bit
        previous = row


def _string_changes(
    strings: StringColumn,
    flags: Pointer[UInt8, MutAnyOrigin],
    lo: Int,
    hi: Int,
    bit: UInt8,
    rows: Int,
):
    """String keys: lengths first, then bytes, with the column's buffers
    read through pointers hoisted out of the loop."""
    var bits = strings.null_count() > 0
    var previous = _row_at(rows, lo)
    if strings._is_view_storage():
        ref views = strings._view_storage.value()
        var base = strings._offset
        for k in range(lo + 1, hi):
            var row = _row_at(rows, k)
            var differs: Bool
            var a = views._view_unchecked(base + previous)
            var b = views._view_unchecked(base + row)
            if a.length != b.length or a.prefix != b.prefix:
                differs = True
            elif a.is_inline():
                differs = a.buffer_index != b.buffer_index or (
                    a.offset != b.offset
                )
            else:
                var x = views._get_unchecked(base + previous).as_bytes()
                var y = views._get_unchecked(base + row).as_bytes()
                differs = (
                    unsafe_memcmp(
                        Pointer[UInt8, ImmutAnyOrigin](
                            unsafe_from_address=Int(x.unsafe_ptr())
                        ),
                        Pointer[UInt8, ImmutAnyOrigin](
                            unsafe_from_address=Int(y.unsafe_ptr())
                        ),
                        len(x),
                    )
                    != 0
                )
            if bits:
                var u = strings._valid(previous)
                var v = strings._valid(row)
                differs = u != v or (v and differs)
            if differs:
                flags[unsafe_offset=k] |= bit
            previous = row
        return
    var offsets = strings._offsets[].unsafe_ptr().unsafe_offset(strings._offset)
    var data = strings._base()
    for k in range(lo + 1, hi):
        var row = _row_at(rows, k)
        var a0 = Int(offsets[unsafe_offset=previous])
        var a1 = Int(offsets[unsafe_offset=previous + 1])
        var b0 = Int(offsets[unsafe_offset=row])
        var b1 = Int(offsets[unsafe_offset=row + 1])
        var differs = a1 - a0 != b1 - b0 or (
            unsafe_memcmp(
                Pointer[UInt8, ImmutAnyOrigin](
                    unsafe_from_address=Int(data.unsafe_offset(a0))
                ),
                Pointer[UInt8, ImmutAnyOrigin](
                    unsafe_from_address=Int(data.unsafe_offset(b0))
                ),
                a1 - a0,
            )
            != 0
        )
        if bits:
            var u = strings._valid(previous)
            var v = strings._valid(row)
            differs = u != v or (v and differs)
        if differs:
            flags[unsafe_offset=k] |= bit
        previous = row


def mark_changes(
    key: Series,
    flags_address: Int,
    lo: Int,
    hi: Int,
    bit: UInt8,
    rows: Int = 0,
) raises:
    """Set `bit` in flags[k] for every k in (lo, hi) where the key's row at
    sorted position k differs from the row at k - 1, as the sort compares
    them: nulls equal, NaN one value, -0.0 equal to 0.0. Position k's row
    is rows[k] (the address of the sorted order), or k itself when `rows`
    is 0."""
    var flags = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=flags_address)
    var column = key.rechunk() if key.is_chunked() else key.copy()
    comptime for t in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[t]
        if column._data.isa[Column[Scalar[D]]]():
            _numeric_changes[D](
                column._data[Column[Scalar[D]]], flags, lo, hi, bit, rows
            )
            return
    if column._data.isa[StringColumn]():
        _string_changes(column._data[StringColumn], flags, lo, hi, bit, rows)
        return
    for k in range(lo + 1, hi):
        if not _same_value(column, _row_at(rows, k), _row_at(rows, k - 1)):
            flags[unsafe_offset=k] |= bit


@always_inline
def _strings_equal(strings: StringColumn, a: Int, b: Int) -> Bool:
    """Rows a and b of a string column hold equal values (nulls equal)."""
    var u = strings._valid(a)
    if u != strings._valid(b):
        return False
    if not u:
        return True
    if strings._is_view_storage():
        ref views = strings._view_storage.value()
        var x = views._view_unchecked(strings._offset + a)
        var y = views._view_unchecked(strings._offset + b)
        if x.length != y.length or x.prefix != y.prefix:
            return False
        if x.is_inline():
            return x.buffer_index == y.buffer_index and x.offset == y.offset
        var p = views._get_unchecked(strings._offset + a).as_bytes()
        var q = views._get_unchecked(strings._offset + b).as_bytes()
        return (
            unsafe_memcmp(
                Pointer[UInt8, ImmutAnyOrigin](
                    unsafe_from_address=Int(p.unsafe_ptr())
                ),
                Pointer[UInt8, ImmutAnyOrigin](
                    unsafe_from_address=Int(q.unsafe_ptr())
                ),
                len(p),
            )
            == 0
        )
    var p = strings._row_bytes(a)
    var q = strings._row_bytes(b)
    return len(p) == len(q) and (
        unsafe_memcmp(
            Pointer[UInt8, ImmutAnyOrigin](
                unsafe_from_address=Int(p.unsafe_ptr())
            ),
            Pointer[UInt8, ImmutAnyOrigin](
                unsafe_from_address=Int(q.unsafe_ptr())
            ),
            len(p),
        )
        == 0
    )


def _rows_equal(keys: List[Series], a: Int, b: Int) -> Bool:
    """Rows a and b hold equal partition keys (nulls equal, NaN one value,
    -0.0 equal to 0.0)."""
    for c in range(len(keys)):
        ref key = keys[c]
        if key._data.isa[StringColumn]():
            if not _strings_equal(key._data[StringColumn], a, b):
                return False
        elif not _same_value(key, a, b):
            return False
    return True


@always_inline
def _hash_partition(hash: UInt64, parts: Int) -> Int:
    """The partition of `parts` a row hash falls in: the high word of
    hash * parts, which maps hashes evenly onto [0, parts) from their top
    bits with one multiply and no division."""
    return Int((UInt128(hash) * UInt128(parts)) >> 64)


struct _Numbering(Movable):
    """A dictionary of partition keys, probed with row hashes, that numbers
    partitions in first-occurrence order, comparing keys row against row
    (`_rows_equal`) when hashes match. It serves any keys but one integer
    column, which `_IntGroups` numbers by value. The table holds one slot
    per partition, not per row, grown (rehashing the stored hashes) to
    stay at most half full, so it stays in cache. Slots come from a
    multiplicative mix of the whole hash: its top bits chose the hash
    group or partition."""

    var slots: List[Int]
    var shift: UInt64
    var mask: UInt64
    var first_rows: List[Int]
    var first_hashes: List[UInt64]

    def __init__(out self):
        self.slots = List[Int](length=256, fill=-1)
        self.shift = UInt64(56)
        self.mask = UInt64(255)
        self.first_rows = List[Int]()
        self.first_hashes = List[UInt64]()

    @always_inline
    def number(mut self, keys: List[Series], row: Int, hash: UInt64) -> Int:
        """Row `row`'s partition number, numbering its keys if new."""
        var at = Int((hash * 0x9E3779B97F4A7C15) >> self.shift)
        while True:
            var slot = self.slots[at]
            if slot < 0:
                var id = len(self.first_rows)
                self.slots[at] = id
                self.first_rows.append(row)
                self.first_hashes.append(hash)
                if 2 * len(self.first_rows) > len(self.slots):
                    self._grow()
                return id
            if self.first_hashes[slot] == hash and _rows_equal(
                keys, self.first_rows[slot], row
            ):
                return slot
            at = Int((UInt64(at) + 1) & self.mask)

    def _grow(mut self):
        var count = 2 * len(self.slots)
        self.shift -= 1
        self.mask = UInt64(count - 1)
        self.slots = List[Int](length=count, fill=-1)
        for id in range(len(self.first_hashes)):
            var to = Int(
                (self.first_hashes[id] * 0x9E3779B97F4A7C15) >> self.shift
            )
            while self.slots[to] >= 0:
                to = Int((UInt64(to) + 1) & self.mask)
            self.slots[to] = id


# Multiplying by a fixed odd constant spreads integer keys over the high
# bits of the product, which is all `_hash_partition` and `_IntGroups`
# read; equal keys give equal hashes, so it is a valid partition hash
# that costs one multiply per row.
comptime _ODD_MIX: UInt64 = 0x55FBFD6BFC5458E9


@always_inline
def _key_hash[D: DType](value: Scalar[D], valid: Bool) -> UInt64:
    """An integer key's partition hash; every null hashes to 0, so all
    nulls share a partition."""
    if not valid:
        return 0
    return value.cast[DType.uint64]() * _ODD_MIX


struct _IntGroups[D: DType](Movable):
    """A dictionary from one integer key's values to group numbers, given
    in first-arrival order. A slot holds a group number; the group's key
    is compared directly, so equal hashes need no second check, and a
    null key gets a number of its own outside the table. The table is
    grown (rehashing the stored keys) to stay at most half full; it holds
    one slot per group, not per row, so it stays in cache. Slots come from
    a second multiplicative mix of the key's hash, so they do not repeat
    the top bits that may have chosen the rows' partition. Both the over()
    groups and the sorted windows' partitions are numbered with it."""

    var slots: List[Int]
    var keys: List[Scalar[Self.D]]
    var shift: UInt64
    var mask: UInt64
    var null_group: Int

    def __init__(out self):
        self.slots = List[Int](length=256, fill=-1)
        self.keys = List[Scalar[Self.D]]()
        self.shift = UInt64(56)
        self.mask = UInt64(255)
        self.null_group = -1

    def count(self) -> Int:
        return len(self.keys)

    @always_inline
    def number(mut self, value: Scalar[Self.D], valid: Bool) -> Int:
        """The group number of key `value` (null when not `valid`),
        numbering it if new."""
        if not valid:
            if self.null_group < 0:
                # The null group's key place holds a placeholder.
                self.null_group = len(self.keys)
                self.keys.append(Scalar[Self.D](0))
            return self.null_group
        var slots = self.slots.unsafe_ptr()
        var keys = self.keys.unsafe_ptr()
        var at = Int(
            (_key_hash[Self.D](value, True) * 0x9E3779B97F4A7C15) >> self.shift
        )
        while True:
            var slot = slots[at]
            if slot < 0:
                return self._insert(at, value)
            if keys[slot] == value:
                return slot
            at = Int((UInt64(at) + 1) & self.mask)

    def _insert(mut self, at: Int, value: Scalar[Self.D]) -> Int:
        var id = len(self.keys)
        self.slots[at] = id
        self.keys.append(value)
        if 2 * len(self.keys) > len(self.slots):
            self._grow()
        return id

    def _grow(mut self):
        var count = 2 * len(self.slots)
        self.shift -= 1
        self.mask = UInt64(count - 1)
        self.slots = List[Int](length=count, fill=-1)
        for id in range(len(self.keys)):
            if id == self.null_group:
                continue
            var to = Int(
                (_key_hash[Self.D](self.keys[id], True) * 0x9E3779B97F4A7C15)
                >> self.shift
            )
            while self.slots[to] >= 0:
                to = Int((UInt64(to) + 1) & self.mask)
            self.slots[to] = id


struct _ScatterJob[I: DType](Job):
    """Rows [lo, hi): find each row's hash partition, then copy the rows
    into this range's own stretch of the entry arrays (also [lo, hi)),
    grouped by partition and in row order within each. A partition's
    rows are then read from the stretches of every range, one after
    another, so each row is read and moved once, by one task, rather than
    every partition task scanning every row.

    An entry is the row, as an `I` row index, and a word: for one integer
    key, its value, so the partition task reads only the entries and
    never goes back to the key column; for any other keys, the hash made
    beforehand. A null integer key has no value to store, so its rows go
    to one more bucket after the partitions (`parts`), whose entries need
    no word: the bucket says the key is null. The partition is found
    twice, once to count each bucket and once to place the row, rather
    than kept per row: finding it again is a multiply on a value the
    second pass reads anyway, while keeping it would write and read back
    a scratch value per row, and the scatter is bound by memory traffic,
    not arithmetic."""

    var keys: List[Series]
    var hashes: Int
    var lo: Int
    var hi: Int
    var parts: Int
    var entry_rows: Int
    var entry_words: Int
    # Where each bucket's entries start in the entry arrays (parts + 2
    # values: the partitions, the null bucket, and the end `hi`).
    var starts: List[Int]

    def __init__(
        out self,
        keys: List[Series],
        hashes: Int,
        lo: Int,
        hi: Int,
        parts: Int,
        entry_rows: Int,
        entry_words: Int,
    ):
        self.keys = keys.copy()
        self.hashes = hashes
        self.lo = lo
        self.hi = hi
        self.parts = parts
        self.entry_rows = entry_rows
        self.entry_words = entry_words
        self.starts = List[Int]()

    def run(mut self) raises:
        if self.hashes == 0:
            comptime for t in range(len(NUMERIC_DTYPES)):
                comptime D = NUMERIC_DTYPES[t]
                comptime if not D.is_floating_point():
                    if self.keys[0]._data.isa[Column[Scalar[D]]]():
                        var column = (
                            self.keys[0]._data[Column[Scalar[D]]].copy()
                        )
                        if len(column._bits[]) == 0:
                            self._scatter[D, False, False](column)
                        else:
                            self._scatter[D, False, True](column)
                        return
        self._scatter[DType.int64, True, False](
            Column[Scalar[DType.int64]](List[Int64]())
        )

    @always_inline
    def _bucket[
        D: DType, GENERIC: Bool, NULLS: Bool
    ](self, column: Column[Scalar[D]], row: Int) -> Int:
        """Row `row`'s bucket: its hash partition, or `parts` for a null
        integer key."""
        comptime if GENERIC:
            var hashes = Pointer[UInt64, MutAnyOrigin](
                unsafe_from_address=self.hashes
            )
            return _hash_partition(hashes[unsafe_offset=row], self.parts)
        else:
            comptime if NULLS:
                if not column._valid(row):
                    return self.parts
            return _hash_partition(
                _key_hash[D](column._ptr()[unsafe_offset=row], True),
                self.parts,
            )

    def _scatter[
        D: DType, GENERIC: Bool, NULLS: Bool
    ](mut self, column: Column[Scalar[D]]):
        var hashes = Pointer[UInt64, MutAnyOrigin](
            unsafe_from_address=self.hashes
        )
        var values = column._ptr()
        var rows = Pointer[Scalar[Self.I], MutAnyOrigin](
            unsafe_from_address=self.entry_rows
        )
        var words = Pointer[UInt64, MutAnyOrigin](
            unsafe_from_address=self.entry_words
        )
        var parts = self.parts
        var cursor = List[Int](length=parts + 2, fill=0)
        var at = cursor.unsafe_ptr()
        for row in range(self.lo, self.hi):
            at[
                unsafe_offset=self._bucket[D, GENERIC, NULLS](column, row) + 1
            ] += 1
        at[unsafe_offset=0] = self.lo
        for p in range(parts + 1):
            at[unsafe_offset=p + 1] += at[unsafe_offset=p]
        self.starts = cursor.copy()
        for row in range(self.lo, self.hi):
            var p = self._bucket[D, GENERIC, NULLS](column, row)
            var to = at[unsafe_offset=p]
            at[unsafe_offset=p] = to + 1
            rows[unsafe_offset=to] = Scalar[Self.I](row)
            comptime if GENERIC:
                words[unsafe_offset=to] = hashes[unsafe_offset=row]
            else:
                words[unsafe_offset=to] = values[unsafe_offset=row].cast[
                    DType.uint64
                ]()


struct _PartitionGroupsJob[I: DType](Job):
    """Hash partition `part`: number its rows' keys in one thread-local
    dictionary, then write each group's rows, ascending, into the
    partition's own stretch of the shared row array (from `row_base`).
    Equal keys have equal hashes, so each key belongs to one partition:
    no two tasks see the same key and nothing is merged afterwards. The
    entries come from every range's stretch in range order, so rows
    arrive ascending and every group's list stays ascending. Partition 0
    also takes the null-key bucket, where a null integer key's rows went
    (a null hashes to 0, which falls in partition 0), as one group.

    Each entry's group number replaces its word, which is not needed
    again, packed as an `I` index at the front of the stretch: entry k of
    a stretch starting at `lo` writes its number to index k - lo, which
    never passes the words not yet read, so the two later passes over the
    numbers read half the bytes when indices are 32 bits."""

    var keys: List[Series]
    var generic: Bool
    var part: Int
    # Address of every range's `starts` (ranges x (parts + 2) values).
    var starts: Int
    var ranges: Int
    var parts: Int
    var entry_rows: Int
    var entry_words: Int
    var row_base: Int
    var all_rows: Int
    # Output: where each group's rows start (count + 1 values, the last
    # one past the partition's rows) and the group count.
    var offsets: List[Scalar[Self.I]]
    var count: Int

    def __init__(
        out self,
        keys: List[Series],
        generic: Bool,
        part: Int,
        starts: Int,
        ranges: Int,
        parts: Int,
        entry_rows: Int,
        entry_words: Int,
        row_base: Int,
        all_rows: Int,
    ):
        self.keys = keys.copy()
        self.generic = generic
        self.part = part
        self.starts = starts
        self.ranges = ranges
        self.parts = parts
        self.entry_rows = entry_rows
        self.entry_words = entry_words
        self.row_base = row_base
        self.all_rows = all_rows
        self.offsets = List[Scalar[Self.I]]()
        self.count = 0

    def _stretches(self) -> List[Int]:
        """The entry stretches this task reads, as [lo, hi) pairs: the
        partition's stretch of every range, then (partition 0 only) every
        range's null-key bucket."""
        var starts = Pointer[Int, MutAnyOrigin](unsafe_from_address=self.starts)
        var width = self.parts + 2
        var out = List[Int](capacity=4 * self.ranges)
        for r in range(self.ranges):
            out.append(starts[unsafe_offset=r * width + self.part])
            out.append(starts[unsafe_offset=r * width + self.part + 1])
        if self.part == 0:
            for r in range(self.ranges):
                out.append(starts[unsafe_offset=r * width + self.parts])
                out.append(starts[unsafe_offset=r * width + self.parts + 1])
        return out^

    def run(mut self) raises:
        if not self.generic:
            comptime for t in range(len(NUMERIC_DTYPES)):
                comptime D = NUMERIC_DTYPES[t]
                comptime if not D.is_floating_point():
                    if self.keys[0]._data.isa[Column[Scalar[D]]]():
                        self._group[D, False]()
                        return
        self._group[DType.int64, True]()

    def _group[D: DType, GENERIC: Bool](mut self):
        var rows = Pointer[Scalar[Self.I], MutAnyOrigin](
            unsafe_from_address=self.entry_rows
        )
        var words = Pointer[UInt64, MutAnyOrigin](
            unsafe_from_address=self.entry_words
        )
        var all_rows = Pointer[Scalar[Self.I], MutAnyOrigin](
            unsafe_from_address=self.all_rows
        )
        var stretches = self._stretches()
        var pairs = len(stretches) // 2
        # The null-key bucket's stretches follow the partition's own.
        var own = self.ranges
        # Number each entry's group, packed over the stretch's words.
        var count: Int
        comptime if GENERIC:
            var keys = self.keys.copy()
            var numbering = _Numbering()
            for s in range(pairs):
                var lo = stretches[2 * s]
                var ids = Pointer[Scalar[Self.I], MutAnyOrigin](
                    unsafe_from_address=self.entry_words
                    + lo * size_of[UInt64]()
                )
                for k in range(lo, stretches[2 * s + 1]):
                    var g = numbering.number(
                        keys, Int(rows[unsafe_offset=k]), words[unsafe_offset=k]
                    )
                    ids[unsafe_offset=k - lo] = Scalar[Self.I](g)
            count = len(numbering.first_rows)
        else:
            var table = _IntGroups[D]()
            for s in range(pairs):
                var lo = stretches[2 * s]
                var ids = Pointer[Scalar[Self.I], MutAnyOrigin](
                    unsafe_from_address=self.entry_words
                    + lo * size_of[UInt64]()
                )
                if s < own:
                    for k in range(lo, stretches[2 * s + 1]):
                        ids[unsafe_offset=k - lo] = Scalar[Self.I](
                            table.number(words[unsafe_offset=k].cast[D](), True)
                        )
                else:
                    for k in range(lo, stretches[2 * s + 1]):
                        ids[unsafe_offset=k - lo] = Scalar[Self.I](
                            table.number(Scalar[D](0), False)
                        )
            count = table.count()
        # A counting sort of the entries by group into the partition's
        # stretch: entries arrive in row order, so each list is ascending.
        var cursors = List[Scalar[Self.I]](length=count + 1, fill=0)
        var at = cursors.unsafe_ptr()
        for s in range(pairs):
            var lo = stretches[2 * s]
            var ids = Pointer[Scalar[Self.I], MutAnyOrigin](
                unsafe_from_address=self.entry_words + lo * size_of[UInt64]()
            )
            for k in range(stretches[2 * s + 1] - lo):
                at[unsafe_offset=Int(ids[unsafe_offset=k]) + 1] += 1
        at[unsafe_offset=0] = Scalar[Self.I](self.row_base)
        for g in range(count):
            at[unsafe_offset=g + 1] += at[unsafe_offset=g]
        self.offsets = cursors.copy()
        for s in range(pairs):
            var lo = stretches[2 * s]
            var ids = Pointer[Scalar[Self.I], MutAnyOrigin](
                unsafe_from_address=self.entry_words + lo * size_of[UInt64]()
            )
            for k in range(lo, stretches[2 * s + 1]):
                var g = Int(ids[unsafe_offset=k - lo])
                var to = at[unsafe_offset=g]
                all_rows[unsafe_offset=Int(to)] = rows[unsafe_offset=k]
                at[unsafe_offset=g] = to + 1
        self.count = count

    def row_base_end(self) -> Int:
        """Where the partition's rows end in the shared row array."""
        var stretches = self._stretches()
        var end = self.row_base
        for s in range(len(stretches) // 2):
            end += stretches[2 * s + 1] - stretches[2 * s]
        return end


struct _GroupOffsetsJob[I: DType](Job):
    """Copy one partition's group starts into the shared offsets, from
    its first group's number."""

    var source: Int
    var count: Int
    var dest: Int

    def __init__(out self, source: Int, count: Int, dest: Int):
        self.source = source
        self.count = count
        self.dest = dest

    def run(mut self) raises:
        if self.count > 0:
            unsafe_memcpy(
                dest=Pointer[Scalar[Self.I], MutAnyOrigin](
                    unsafe_from_address=self.dest
                ),
                src=Pointer[Scalar[Self.I], MutAnyOrigin](
                    unsafe_from_address=self.source
                ),
                count=self.count,
            )


def _integer_key(keys: List[Series]) -> Bool:
    """Whether the keys are one integer column, which the numbering tasks
    hash themselves as they read it."""
    if len(keys) != 1:
        return False
    comptime for t in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[t]
        comptime if not D.is_floating_point():
            if keys[0]._data.isa[Column[Scalar[D]]]():
                return True
    return False


def partition_groups(keys: List[Series], height: Int) raises -> GroupRows:
    """The groups of the partition keys' rows, each group's rows in
    ascending order, with no sort and no rows moved, for aggregates over
    whole partitions. Rows are first moved, by row range in parallel,
    into buckets by key-hash partition (one per thread). Then one task per
    partition numbers its keys and writes its groups' row lists straight
    into its own stretch of one array: each partition's row count is
    known from the buckets, so every stretch is placed before any task
    starts. No group number is kept per row: the lists are what the
    aggregates and the write back to rows read. Groups are numbered in
    partition order, not by first row: aggregates over partitions do not
    depend on group order. Row indices are 32 bits unless the input has
    more rows than that numbers (`wide_row_indices`)."""
    if height == 0:
        return GroupRows()
    if wide_row_indices(height):
        return _partition_groups[WIDE_ROW_INDEX](keys, height)
    return _partition_groups[ROW_INDEX](keys, height)


def _partition_groups[
    I: DType
](keys: List[Series], height: Int) raises -> GroupRows:
    var workers = configured_workers()
    var parts = workers
    var flat = List[Series](capacity=len(keys))
    var hashes = List[UInt64]()
    var generic = not _integer_key(keys)
    if not generic:
        flat.append(
            keys[0].rechunk() if keys[0].is_chunked() else keys[0].copy()
        )
    else:
        var partitioner = Partitioner(keys, workers)
        flat = partitioner.keys.copy()
        hashes = partitioner^.into_hashes()
    var entry_rows = huge_uninit[Scalar[I]](height)
    var entry_words = huge_uninit[UInt64](height)
    # The groups' row lists, filled by the partition tasks.
    var rows = huge_uninit[Scalar[I]](height)
    var bounds = partitions(height, workers, 1)
    var scatter = List[_ScatterJob[I]](capacity=workers)
    for r in range(workers):
        scatter.append(
            _ScatterJob[I](
                flat,
                Int(hashes.unsafe_ptr()) if generic else 0,
                bounds[r],
                bounds[r + 1],
                parts,
                Int(entry_rows.unsafe_ptr()),
                Int(entry_words.unsafe_ptr()),
            )
        )
    if workers == 1:
        scatter[0].run()
    else:
        run_jobs(scatter)
    var starts = List[Int](capacity=workers * (parts + 2))
    for r in range(workers):
        starts.extend(scatter[r].starts.copy())
    var jobs = List[_PartitionGroupsJob[I]](capacity=parts)
    var row_base = 0
    for p in range(parts):
        jobs.append(
            _PartitionGroupsJob[I](
                flat,
                generic,
                p,
                Int(starts.unsafe_ptr()),
                workers,
                parts,
                Int(entry_rows.unsafe_ptr()),
                Int(entry_words.unsafe_ptr()),
                row_base,
                Int(rows.unsafe_ptr()),
            )
        )
        row_base = jobs[p].row_base_end()
    if parts == 1:
        jobs[0].run()
    else:
        run_jobs(jobs)
    var total = 0
    for p in range(parts):
        total += jobs[p].count
    var offsets = List[Scalar[I]](unsafe_uninit_length=total + 1)
    offsets[total] = Scalar[I](height)
    var copies = List[_GroupOffsetsJob[I]](capacity=parts)
    var base = 0
    for p in range(parts):
        copies.append(
            _GroupOffsetsJob[I](
                Int(jobs[p].offsets.unsafe_ptr()),
                jobs[p].count,
                Int(offsets.unsafe_ptr()) + base * size_of[Scalar[I]](),
            )
        )
        base += jobs[p].count
    if parts == 1:
        copies[0].run()
    else:
        run_jobs(copies)
    # Jobs read these by address: keep them alive until here.
    _ = jobs^
    _ = starts^
    _ = entry_rows^
    _ = entry_words^
    _ = hashes^
    _ = flat^
    return group_rows_of[I](offsets^, rows^)


def sort_key_rows(
    keys: List[Series],
    descending: List[Bool],
    nulls_last: List[Bool],
    stable: Bool = True,
) raises -> List[Int]:
    """The sorted order of the key columns' rows: normalize keys and sort
    (`sort_rows`), with ties in input order when `stable`, as
    over(order_by=...) keeps them. Threads come from the shared budget, so
    groups sorted on several workers at once share it."""
    return sort_rows(keys, descending, nulls_last, maintain_order=stable)


struct _RankOut(ImplicitlyCopyable, Movable):
    """Where a window's ranks go: the method (-1: no ranks), the input's
    validity by row (Bool, 0: every row valid) and the outputs, by input
    row (`ints` or, for average, `floats`; `out_valid` 0 when no nulls)."""

    var method: Int
    var valid: Int
    var ints: Int
    var floats: Int
    var out_valid: Int

    def __init__(out self):
        self.method = -1
        self.valid = 0
        self.ints = 0
        self.floats = 0
        self.out_valid = 0


struct _GroupSortJob[I: DType](Job):
    """Order one task's groups by their order keys and mark their
    boundaries. The task holds whole groups [g0, g1) of the partition
    groups, whose rows lie at [lo, hi) of the group row lists, each list
    ascending. Only the order keys are read, for these rows: the groups
    were already found, so no partition key is read again, hashed or
    numbered, and every position's group is known from the offsets."""

    var keys: List[Series]
    var descending: List[Bool]
    var nulls_last: List[Bool]
    # The group row lists (`I` indices): rows and offsets, by address.
    var rows: Int
    var offsets: Int
    var g0: Int
    var g1: Int
    var lo: Int
    var hi: Int
    # Output: order[lo:hi] and flags[lo:hi].
    var order: Int
    var flags: Int
    var stable: Bool
    # Ranks evaluated over the task's rows as soon as they are sorted, by
    # the same task: no other task's work has to finish first.
    var rank: _RankOut

    def __init__(
        out self,
        keys: List[Series],
        descending: List[Bool],
        nulls_last: List[Bool],
        rows: Int,
        offsets: Int,
        g0: Int,
        g1: Int,
        lo: Int,
        hi: Int,
        order: Int,
        flags: Int,
        stable: Bool,
        rank: _RankOut,
    ):
        self.keys = keys.copy()
        self.descending = descending.copy()
        self.nulls_last = nulls_last.copy()
        self.rows = rows
        self.offsets = offsets
        self.g0 = g0
        self.g1 = g1
        self.lo = lo
        self.hi = hi
        self.order = order
        self.flags = flags
        self.stable = stable
        self.rank = rank

    def run(mut self) raises:
        self._sort()
        if self.rank.method < 0 or self.hi == self.lo:
            return
        # The task holds whole partitions, so its ranks start fresh at lo
        # and need no state from other tasks.
        var scan = _RankScanJob(
            self.flags,
            self.order,
            self.rank.valid,
            self.lo,
            self.hi,
            self.rank.method,
            self.rank.ints,
            self.rank.floats,
            self.rank.out_valid,
        )
        scan.write = True
        scan.run()

    def _sort(mut self) raises:
        var n = self.hi - self.lo
        if n == 0:
            return
        var rows = Pointer[Scalar[Self.I], MutAnyOrigin](
            unsafe_from_address=self.rows
        )
        var offsets = Pointer[Scalar[Self.I], MutAnyOrigin](
            unsafe_from_address=self.offsets
        )
        var order = Pointer[Int, MutAnyOrigin](unsafe_from_address=self.order)
        var flags_at = self.flags + self.lo
        var flags = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=flags_at)
        # Partition starts come from the offsets: every group is one
        # partition, and the sort below keeps groups where they are.
        for k in range(n):
            flags[unsafe_offset=k] = 0
        for g in range(self.g0, self.g1):
            flags[unsafe_offset=Int(offsets[unsafe_offset=g]) - self.lo] = (
                PARTITION_START | PEER_START
            )
        if len(self.keys) == 0:
            # No order keys: each group's rows, ascending, are its order.
            for k in range(self.lo, self.hi):
                order[unsafe_offset=k] = Int(rows[unsafe_offset=k])
            return
        # The order keys of the task's rows, in group-list order.
        var gathered = List[Series](capacity=len(self.keys))
        for key in self.keys:
            gathered.append(_gather_rows[Self.I](key, self.rows, self.lo, n))
        # Groups are sorted in runs of consecutive groups holding at least
        # one STANDARD_VECTOR_SIZE block of rows: a sort call has a fixed
        # setup that a run of tiny groups would pay for each group, while
        # a run of one group needs no group number in its sort key.
        var g = self.g0
        while g < self.g1:
            var first = g
            var a = Int(offsets[unsafe_offset=g]) - self.lo
            g += 1
            while (
                g < self.g1
                and Int(offsets[unsafe_offset=g]) - self.lo - a
                < STANDARD_VECTOR_SIZE
            ):
                g += 1
            var b = Int(offsets[unsafe_offset=g]) - self.lo
            self._sort_run(gathered, first, g, a, b)

    def _sort_run(
        self, gathered: List[Series], g0: Int, g1: Int, a: Int, b: Int
    ) raises:
        """Order the run of groups [g0, g1), at task positions [a, b)."""
        var n = b - a
        var rows = Pointer[Scalar[Self.I], MutAnyOrigin](
            unsafe_from_address=self.rows
        )
        var offsets = Pointer[Scalar[Self.I], MutAnyOrigin](
            unsafe_from_address=self.offsets
        )
        var order = Pointer[Int, MutAnyOrigin](unsafe_from_address=self.order)
        var base = self.lo + a
        var sort_keys = List[Series](capacity=1 + len(gathered))
        var descending = List[Bool](capacity=1 + len(gathered))
        var nulls_last = List[Bool](capacity=1 + len(gathered))
        if g1 - g0 > 1:
            # Each position's group within the run leads the sort key, so
            # the groups stay in place and each is ordered on its own. The
            # numbers fit four bytes, keeping the sort's records narrow.
            var local = huge_uninit[UInt32](n)
            var at = 0
            for g in range(g1 - g0):
                var end = Int(offsets[unsafe_offset=g0 + g + 1]) - base
                while at < end:
                    local[at] = UInt32(g)
                    at += 1
            sort_keys.append(Series("", Column[UInt32](local^)))
            descending.append(False)
            nulls_last.append(False)
        var keys = List[Series](capacity=len(gathered))
        for c in range(len(gathered)):
            keys.append(gathered[c].slice(a, n))
            sort_keys.append(keys[c].copy())
            descending.append(self.descending[c])
            nulls_last.append(self.nulls_last[c])
        var perm = sort_key_rows(sort_keys, descending, nulls_last, self.stable)
        for k in range(n):
            order[unsafe_offset=base + k] = Int(
                rows[unsafe_offset=base + perm[k]]
            )
        for c in range(len(keys)):
            mark_changes(
                keys[c],
                self.flags + base,
                0,
                n,
                PEER_START,
                Int(perm.unsafe_ptr()),
            )
        _ = perm^


def _gather_rows[
    I: DType
](key: Series, rows_address: Int, lo: Int, n: Int) raises -> Series:
    """The key's values at rows[lo:lo + n] (`I` row indices at
    `rows_address`), in that order."""
    var rows = Pointer[Scalar[I], MutAnyOrigin](
        unsafe_from_address=rows_address
    )
    comptime for t in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[t]
        if key._data.isa[Column[Scalar[D]]]():
            ref column = key._data[Column[Scalar[D]]]
            if len(column._bits[]) == 0:
                var values = column._ptr()
                var out = huge_uninit[Scalar[D]](n)
                var to = out.unsafe_ptr()
                for k in range(n):
                    to[unsafe_offset=k] = values[
                        unsafe_offset=Int(rows[unsafe_offset=lo + k])
                    ]
                var result = Series("", Column[Scalar[D]](out^))
                result._dtype = key._dtype
                return result^
    var indices = List[Int](unsafe_uninit_length=n)
    for k in range(n):
        indices[k] = Int(rows[unsafe_offset=lo + k])
    return key.take(indices)


struct _MaskJob(Job):
    """Mark peer boundaries over one range of a fully sorted key set (one
    partition's peers, split across threads)."""

    var keys: List[Series]
    var lo: Int
    var hi: Int
    var flags: Int
    # The sorted order (address): keys are read through it.
    var rows: Int

    def __init__(
        out self, keys: List[Series], lo: Int, hi: Int, flags: Int, rows: Int
    ):
        self.keys = keys.copy()
        self.lo = lo
        self.hi = hi
        self.flags = flags
        self.rows = rows

    def run(mut self) raises:
        var flags = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=self.flags)
        for k in range(self.lo, self.hi):
            flags[unsafe_offset=k] = 0
        var from_ = max(self.lo - 1, 0)
        for key in self.keys:
            mark_changes(key, self.flags, from_, self.hi, PEER_START, self.rows)


def _sort_groups[
    I: DType
](
    groups: GroupRows,
    keys: List[Series],
    descending: List[Bool],
    nulls_last: List[Bool],
    order: List[Int],
    flags: List[UInt8],
    stable: Bool,
    rank: _RankOut,
) raises -> List[Int]:
    """Order every group's rows by the order keys into `order` and mark
    `flags`, in tasks of whole groups and about equal rows; returns the
    tasks' first positions. A task holds at least one STANDARD_VECTOR_SIZE
    block of rows: a task smaller than one vector costs more to schedule
    than its work, so small inputs get fewer tasks."""
    var height = len(order)
    var blocks = (height + STANDARD_VECTOR_SIZE - 1) // STANDARD_VECTOR_SIZE
    var tasks = max(1, min(configured_workers(), blocks))
    var cuts = groups.slices(tasks)
    var jobs = List[_GroupSortJob[I]](capacity=len(cuts) - 1)
    var bounds = List[Int](capacity=len(cuts))
    for t in range(len(cuts) - 1):
        var lo = groups.offset(cuts[t])
        jobs.append(
            _GroupSortJob[I](
                keys,
                descending,
                nulls_last,
                groups.rows_address(),
                groups.offsets_address(),
                cuts[t],
                cuts[t + 1],
                lo,
                groups.offset(cuts[t + 1]),
                Int(order.unsafe_ptr()),
                Int(flags.unsafe_ptr()),
                stable,
                rank,
            )
        )
        bounds.append(lo)
    if len(jobs) == 1:
        jobs[0].run()
    elif len(jobs) > 1:
        run_jobs(jobs)
    return bounds^


def window_layout(
    partition_keys: List[Series],
    order_keys: List[Series],
    descending: List[Bool],
    nulls_last: List[Bool],
    height: Int,
    stable: Bool = True,
) raises -> WindowLayout:
    """Sort `height` rows by partition and order keys and mark their
    boundaries. `descending` and `nulls_last` hold one flag per
    order key. Rows with equal keys keep input order when `stable`."""
    return _window_layout(
        partition_keys,
        order_keys,
        descending,
        nulls_last,
        height,
        stable,
        _RankOut(),
        True,
    )


def _window_layout(
    partition_keys: List[Series],
    order_keys: List[Series],
    descending: List[Bool],
    nulls_last: List[Bool],
    height: Int,
    stable: Bool,
    rank: _RankOut,
    inverse: Bool,
) raises -> WindowLayout:
    """`window_layout`; with `rank.method` set and partition keys, the hash
    group tasks also write every row's rank, and the layout comes back
    empty. Without `inverse`, rows' sorted positions are not recorded."""
    var layout = WindowLayout()
    layout.group_parts.append(0)
    if height == 0:
        layout.starts.append(0)
        return layout^
    var workers = configured_workers()
    var order = huge_uninit[Int](height)
    var flags = huge_uninit[UInt8](height)
    var bounds = List[Int]()
    if len(partition_keys) > 0:
        # Each partition's rows, ascending, found by hashing the keys once
        # (the over() groups' lists). Tasks then order whole groups by
        # their order keys alone, writing their positions of the layout.
        var groups = partition_groups(partition_keys, height)
        var keys = List[Series](capacity=len(order_keys))
        for key in order_keys:
            keys.append(key.rechunk() if key.is_chunked() else key.copy())
        if groups.wide:
            bounds = _sort_groups[WIDE_ROW_INDEX](
                groups, keys, descending, nulls_last, order, flags, stable, rank
            )
        else:
            bounds = _sort_groups[ROW_INDEX](
                groups, keys, descending, nulls_last, order, flags, stable, rank
            )
        if rank.method >= 0:
            # Ranks are written; nothing reads the layout.
            _ = order^
            _ = flags^
            return layout^
    elif len(order_keys) > 0:
        # No partition keys: one group, sorted on every thread; peers are marked
        # in parallel ranges, reading the keys through the sorted order.
        order = sort_key_rows(order_keys, descending, nulls_last, stable)
        var keys = List[Series](capacity=len(order_keys))
        for key in order_keys:
            keys.append(key.rechunk() if key.is_chunked() else key.copy())
        var ranges = partitions(height, workers, STANDARD_VECTOR_SIZE)
        var jobs = List[_MaskJob](capacity=workers)
        for w in range(workers):
            if ranges[w + 1] > ranges[w]:
                jobs.append(
                    _MaskJob(
                        keys,
                        ranges[w],
                        ranges[w + 1],
                        Int(flags.unsafe_ptr()),
                        Int(order.unsafe_ptr()),
                    )
                )
        run_jobs(jobs)
        flags[0] = PARTITION_START | PEER_START
        bounds.append(0)
    else:
        # No keys at all: input order, one partition of peers.
        for k in range(height):
            order[k] = k
            flags[k] = 0
        flags[0] = PARTITION_START | PEER_START
        bounds.append(0)
    bounds.append(height)
    # Partition starts, then each row's partition and sorted position, over
    # ranges of sorted positions in parallel.
    var ranges = partitions(height, workers, STANDARD_VECTOR_SIZE)
    var finish = List[_FinishJob](capacity=workers)
    for w in range(workers):
        if ranges[w + 1] > ranges[w]:
            finish.append(
                _FinishJob(
                    Int(flags.unsafe_ptr()),
                    Int(order.unsafe_ptr()),
                    ranges[w],
                    ranges[w + 1],
                )
            )
    if len(finish) > 1:
        run_jobs(finish)
    else:
        finish[0].run()
    layout.in_place = True
    for j in range(len(finish)):
        layout.in_place = layout.in_place and finish[j].in_place
        layout.starts.extend(Span(finish[j].starts))
    layout.count = len(layout.starts)
    layout.starts.append(height)
    # The groups' partition ranges: every group starts a partition.
    var g = 1
    for p in range(layout.count):
        if g < len(bounds) - 1 and layout.starts[p] == bounds[g]:
            layout.group_parts.append(p)
            g += 1
    layout.group_parts.append(layout.count)
    if inverse and not layout.in_place:
        # Each row's sorted position, to put row results back in order.
        layout.inverse = huge_uninit[Int](height)
        for j in range(len(finish)):
            finish[j].inverse = Int(layout.inverse.unsafe_ptr())
        if len(finish) > 1:
            run_jobs(finish)
        else:
            finish[0].run()
    layout.order = order^
    layout.flags = flags^
    return layout^


struct _FinishJob(Job):
    """One range of sorted positions [lo, hi): first its partition starts
    and whether order is the identity there; then (`inverse` set) each
    row's sorted position."""

    var flags: Int
    var order: Int
    var lo: Int
    var hi: Int
    var starts: List[Int]
    var in_place: Bool
    # Second pass: the output.
    var inverse: Int

    def __init__(out self, flags: Int, order: Int, lo: Int, hi: Int):
        self.flags = flags
        self.order = order
        self.lo = lo
        self.hi = hi
        self.starts = List[Int]()
        self.in_place = True
        self.inverse = 0

    def run(mut self) raises:
        var flags = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=self.flags)
        var order = Pointer[Int, MutAnyOrigin](unsafe_from_address=self.order)
        if self.inverse == 0:
            var identity = True
            for k in range(self.lo, self.hi):
                if (flags[unsafe_offset=k] & PARTITION_START) != 0:
                    self.starts.append(k)
                identity = identity and order[unsafe_offset=k] == k
            self.in_place = identity
            return
        var inverse = Pointer[Int, MutAnyOrigin](
            unsafe_from_address=self.inverse
        )
        for k in range(self.lo, self.hi):
            inverse[unsafe_offset=order[unsafe_offset=k]] = k


# Rank methods (rank()'s `method`).
comptime RANK_ORDINAL = 0
comptime RANK_MIN = 1
comptime RANK_MAX = 2
comptime RANK_DENSE = 3
comptime RANK_AVERAGE = 4


def rank_method(method: String) raises -> Int:
    if method == "ordinal":
        return RANK_ORDINAL
    if method == "min":
        return RANK_MIN
    if method == "max":
        return RANK_MAX
    if method == "dense":
        return RANK_DENSE
    if method == "average":
        return RANK_AVERAGE
    raise Error("unknown rank method: " + method)


def rank_order_keys(
    input: Series,
    descending: Bool,
    mut keys: List[Series],
    mut key_descending: List[Bool],
    mut key_nulls_last: List[Bool],
) raises:
    """Append the order keys that rank `input`: its value (a struct's
    fields in turn), nulls first in each partition so that ranks count
    from the first value. A struct field's nulls are its smallest values; a null struct sorts before every value.
    """
    var column = input.rechunk() if input.is_chunked() else input.copy()
    if column.dtype().is_struct():
        var structs = column.struct_column()
        if column.null_count() > 0:
            var present = validity(column)
            var flags = List[UInt8](unsafe_uninit_length=len(present))
            for k in range(len(present)):
                flags[k] = UInt8(1) if present[k] else UInt8(0)
            keys.append(Series("", Column[UInt8](flags^)))
            key_descending.append(False)
            key_nulls_last.append(False)
        for k in range(structs.field_count()):
            keys.append(structs.field(k))
            key_descending.append(descending)
            key_nulls_last.append(descending)
        return
    keys.append(column^)
    key_descending.append(descending)
    key_nulls_last.append(False)


struct _RankScanJob(Job):
    """Ranks over sorted positions [lo, hi) from the partition and peer
    boundaries (a rank restarts at each partition start and advances at
    each peer start; max and average also need the peer end). The first
    pass summarizes the range; the second writes ranks from the state
    carried in from the ranges before it."""

    var flags: Int
    var order: Int
    # Input row -> Bool validity (0: every row valid).
    var valid: Int
    var lo: Int
    var hi: Int
    var method: Int
    var write: Bool
    # Carried in (second pass): the partition's first valid position (-1
    # while in its nulls), the current peer group's start, the dense rank.
    var base: Int
    var peer: Int
    var dense: Int
    # The first peer boundary at or after hi.
    var next_start: Int
    # Summary (first pass).
    var has_partition: Bool
    var first_peer: Int
    var last_peer: Int
    var first_valid_peer: Int
    var local_base: Int
    var local_dense: Int
    # Outputs, by input row.
    var ints: Int
    var floats: Int
    var out_valid: Int

    def __init__(
        out self,
        flags: Int,
        order: Int,
        valid: Int,
        lo: Int,
        hi: Int,
        method: Int,
        ints: Int,
        floats: Int,
        out_valid: Int,
    ):
        self.flags = flags
        self.order = order
        self.valid = valid
        self.lo = lo
        self.hi = hi
        self.method = method
        self.write = False
        self.base = -1
        self.peer = lo
        self.dense = 0
        self.next_start = hi
        self.has_partition = False
        self.first_peer = -1
        self.last_peer = -1
        self.first_valid_peer = -1
        self.local_base = -1
        self.local_dense = 0
        self.ints = ints
        self.floats = floats
        self.out_valid = out_valid

    @always_inline
    def _valid_at(self, position: Int) -> Bool:
        if self.valid == 0:
            return True
        var order = Pointer[Int, MutAnyOrigin](unsafe_from_address=self.order)
        var valid = Pointer[Bool, MutAnyOrigin](unsafe_from_address=self.valid)
        return valid[unsafe_offset=order[unsafe_offset=position]]

    def run(mut self) raises:
        var flags = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=self.flags)
        if not self.write:
            for k in range(self.lo, self.hi):
                var f = flags[unsafe_offset=k]
                if (f & PEER_START) == 0:
                    continue
                if self.first_peer < 0:
                    self.first_peer = k
                self.last_peer = k
                if (f & PARTITION_START) != 0:
                    self.has_partition = True
                    self.local_base = -1
                    self.local_dense = 0
                if self._valid_at(k):
                    if self.has_partition:
                        if self.local_base < 0:
                            self.local_base = k
                    elif self.first_valid_peer < 0:
                        self.first_valid_peer = k
                    self.local_dense += 1
            return
        var order = Pointer[Int, MutAnyOrigin](unsafe_from_address=self.order)
        var ints = Pointer[Int64, MutAnyOrigin](unsafe_from_address=self.ints)
        var floats = Pointer[Float64, MutAnyOrigin](
            unsafe_from_address=self.floats
        )
        var out_valid = Pointer[Bool, MutAnyOrigin](
            unsafe_from_address=self.out_valid
        )
        var method = self.method
        var base = self.base
        var peer = self.peer
        var dense = self.dense
        var k = self.lo
        while k < self.hi:
            var f = flags[unsafe_offset=k]
            var present: Bool
            if (f & PEER_START) != 0:
                if (f & PARTITION_START) != 0:
                    base = -1
                    dense = 0
                peer = k
                present = self._valid_at(k)
                if present:
                    if base < 0:
                        base = k
                    dense += 1
            else:
                # The peer group carried in from the range before.
                present = base >= 0 and self._valid_at(k)
            var e = k + 1
            while e < self.hi and (flags[unsafe_offset=e] & PEER_START) == 0:
                e += 1
            var end = e if e < self.hi else self.next_start
            if self.out_valid != 0:
                for j in range(k, e):
                    out_valid[unsafe_offset=order[unsafe_offset=j]] = present
            if not present:
                k = e
                continue
            if method == RANK_AVERAGE:
                var value = Float64(peer + 1 + end - 2 * base) / 2
                for j in range(k, e):
                    floats[unsafe_offset=order[unsafe_offset=j]] = value
            elif method == RANK_ORDINAL:
                for j in range(k, e):
                    ints[unsafe_offset=order[unsafe_offset=j]] = Int64(
                        j - base + 1
                    )
            else:
                var value: Int64
                if method == RANK_MIN:
                    value = Int64(peer - base + 1)
                elif method == RANK_MAX:
                    value = Int64(end - base)
                else:
                    value = Int64(dense)
                for j in range(k, e):
                    ints[unsafe_offset=order[unsafe_offset=j]] = value
            k = e


def rank_window(
    partition_keys: List[Series],
    input: Series,
    descending: Bool,
    method: String,
) raises -> Series:
    """Every row's rank of `input` within its partition of
    `partition_keys` (none: one partition): rank() OVER (PARTITION BY keys
    ORDER BY input). Rows are sorted by `rank_order_keys(input)`: runs of
    equal values are peers, identified by adjacent equal sort keys. Nulls
    rank as null; ranks count from a partition's first value. Ordinal
    ranks peers in input order.

    Ranks are computed in sorted order and written straight to their
    rows. With partition keys, each task sorts the rows of its whole
    partitions, marks their partition and peer boundaries and ranks them
    in one go: its ranks depend on no other task and no thread waits for
    the others. One
    partition is ranked in parallel ranges of the full sort, each starting
    from the state its predecessors carry in."""
    var code = rank_method(method)
    var n = len(input)
    var keys = List[Series]()
    var key_descending = List[Bool]()
    var key_nulls_last = List[Bool]()
    rank_order_keys(input, descending, keys, key_descending, key_nulls_last)
    var average = code == RANK_AVERAGE
    var ints = huge_uninit[Int64](0 if average else n)
    var floats = huge_uninit[Float64](n if average else 0)
    var present = List[Bool]()
    var has_nulls = input.null_count() > 0
    if has_nulls:
        present = validity(input)
    var out_valid = huge_uninit[Bool](n if has_nulls else 0)
    var rank = _RankOut()
    rank.method = code
    rank.valid = Int(present.unsafe_ptr()) if has_nulls else 0
    rank.ints = Int(ints.unsafe_ptr())
    rank.floats = Int(floats.unsafe_ptr())
    rank.out_valid = Int(out_valid.unsafe_ptr()) if has_nulls else 0
    var layout = _window_layout(
        partition_keys,
        keys,
        key_descending,
        key_nulls_last,
        n,
        method == "ordinal",
        rank if len(partition_keys) > 0 else _RankOut(),
        False,
    )
    if len(partition_keys) == 0:
        _rank_ranges(layout, rank)
    _ = layout^
    _ = present^
    if average and has_nulls:
        return Series("", Column[Float64](floats^, out_valid))
    if average:
        return Series("", Column[Float64](floats^))
    if has_nulls:
        return Series("", Column[Int64](ints^, out_valid))
    return Series("", Column[Int64](ints^))


def _rank_ranges(layout: WindowLayout, rank: _RankOut) raises:
    """Rank one sorted layout in parallel ranges of sorted positions: a
    first pass summarizes each range, the second writes ranks from the
    state the ranges before it carry in."""
    var n = len(layout.order)
    var workers = configured_workers()
    var bounds = partitions(n, workers, STANDARD_VECTOR_SIZE)
    var jobs = List[_RankScanJob](capacity=workers)
    for w in range(workers):
        if bounds[w + 1] > bounds[w]:
            jobs.append(
                _RankScanJob(
                    Int(layout.flags.unsafe_ptr()),
                    Int(layout.order.unsafe_ptr()),
                    rank.valid,
                    bounds[w],
                    bounds[w + 1],
                    rank.method,
                    rank.ints,
                    rank.floats,
                    rank.out_valid,
                )
            )
    if len(jobs) > 1:
        run_jobs(jobs)
        # Carry the state across ranges, then the next boundary backwards.
        var base = -1
        var peer = 0
        var dense = 0
        for j in range(len(jobs)):
            jobs[j].base = base
            jobs[j].peer = peer
            jobs[j].dense = dense
            if jobs[j].has_partition:
                base = jobs[j].local_base
                dense = jobs[j].local_dense
            else:
                if base < 0 and jobs[j].first_valid_peer >= 0:
                    base = jobs[j].first_valid_peer
                dense += jobs[j].local_dense
            if jobs[j].last_peer >= 0:
                peer = jobs[j].last_peer
        var next_start = n
        for r in range(len(jobs)):
            var j = len(jobs) - 1 - r
            jobs[j].next_start = next_start
            if jobs[j].first_peer >= 0:
                next_start = jobs[j].first_peer
    elif len(jobs) == 1:
        jobs[0].next_start = n
    for j in range(len(jobs)):
        jobs[j].write = True
    if len(jobs) > 1:
        run_jobs(jobs)
    elif len(jobs) == 1:
        jobs[0].run()


def rank_rows(
    input: Series, ids: List[Int], method: String, descending: Bool
) raises -> Series:
    """Rank `input` within the partitions `ids` gives (none: one
    partition): rank() OVER (PARTITION BY ids ORDER BY input)."""
    var n = len(input)
    var partition_keys = List[Series]()
    if len(ids) > 0:
        var codes = huge_uninit[Int64](n)
        for k in range(n):
            codes[k] = Int64(ids[k])
        partition_keys.append(Series("", Column[Int64](codes^)))
    return rank_window(partition_keys, input, descending, method)
