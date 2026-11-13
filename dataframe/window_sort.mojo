"""Partition and stably order rows for window evaluation.

Hash the partition keys, order each partition by its sort keys and record
partition and peer boundaries. Stable ordering keeps input order among
equal keys, which makes positional window functions deterministic. The
result maps computed values back to the original row positions.
"""
from std.memory import Pointer, unsafe_memcmp
from std.sys import size_of

from dataframe.storage.allocation import huge_uninit
from dataframe.storage.boolean import BoolColumn
from dataframe.storage.column import Column
from dataframe.storage.string import StringColumn
from dataframe.dtype import NUMERIC_DTYPES
from dataframe.kernels.group_rows import GroupRows
from dataframe.kernels.partition import Partitioner, _same_value
from dataframe.aggregation.reducer import float_key
from dataframe.parallel import Job, configured_workers, partitions, run_jobs
from dataframe.series import Series
from dataframe.sorting.sort import sort_rows
from dataframe.kernels.binary import validity
from dataframe.kernels.gather import take_parallel
from dataframe.sorting.sort_key import STANDARD_VECTOR_SIZE

# hashed_sort.cpp, HashedSortGlobalSinkState::Rehash: partitions start at
# 4 radix bits and grow by 2 while the average hash group holds more than
# DEFAULT_ROW_GROUP_SIZE rows, up to max_bits (8 for any thread memory
# above 512 MB: thread_pages >> max_bits > 1).
comptime _START_BITS = 4
comptime _MAX_BITS = 8
comptime DEFAULT_ROW_GROUP_SIZE = 122880
# Boundary flags per sorted position.
comptime PARTITION_START = UInt8(1)
comptime PEER_START = UInt8(2)


def sink_threads(rows: Int) -> Int:
    """Threads sinking `rows` rows into the window: one per source morsel
    (DEFAULT_ROW_GROUP_SIZE rows), each hashing its own rows."""
    return max(
        1,
        min(
            configured_workers(),
            (rows + DEFAULT_ROW_GROUP_SIZE - 1) // DEFAULT_ROW_GROUP_SIZE,
        ),
    )


def radix_bits(rows: Int) -> Int:
    """Hash-group radix bits for `rows` rows (HashedSort Rehash)."""
    var bits = _START_BITS
    while bits < _MAX_BITS and rows // (1 << bits) > DEFAULT_ROW_GROUP_SIZE:
        bits = min(bits + 2, _MAX_BITS)
    return bits


struct WindowLayout(Movable):
    """Rows sorted by hash group, then partition and order keys, with the
    partition and peer boundaries of the sorted order."""

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
    # Partition index bounds of the hash groups (groups + 1 entries).
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
    them: nulls equal, NaN one value, -0.0 equal to 0.0 (VectorOperations::
    DistinctFrom in ComputeMasks). Position k's row is rows[k] (the
    address of the sorted order), or k itself when `rows` is 0."""
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


struct _Numbering[D: DType, GENERIC: Bool](Movable):
    """A dictionary of partition keys, probed with row hashes, that numbers
    partitions in first-occurrence order. One integer key (GENERIC off)
    compares its values directly; any other keys compare through
    `_rows_equal`. The table holds one slot per partition, not per row,
    grown (rehashing the stored hashes) to stay at most half full, so it
    stays in cache. Slots come from a multiplicative mix of the whole hash:
    its top bits chose the hash group or partition."""

    var slots: List[Int]
    var shift: UInt64
    var mask: UInt64
    var first_rows: List[Int]
    var first_hashes: List[UInt64]
    var first_values: List[Scalar[Self.D]]
    var first_valid: List[Bool]
    # The integer key's values (address) and whether it has a bitmap.
    var values: Int
    var bits: Bool

    def __init__(out self, column: Column[Scalar[Self.D]]):
        self.slots = List[Int](length=256, fill=-1)
        self.shift = UInt64(56)
        self.mask = UInt64(255)
        self.first_rows = List[Int]()
        self.first_hashes = List[UInt64]()
        self.first_values = List[Scalar[Self.D]]()
        self.first_valid = List[Bool]()
        self.values = Int(column._ptr())
        self.bits = len(column._bits[]) != 0

    @always_inline
    def number(
        mut self,
        keys: List[Series],
        column: Column[Scalar[Self.D]],
        row: Int,
        hash: UInt64,
    ) -> Int:
        """Row `row`'s partition number, numbering its keys if new."""
        var values = Pointer[Scalar[Self.D], MutAnyOrigin](
            unsafe_from_address=self.values
        )
        var at = Int((hash * 0x9E3779B97F4A7C15) >> self.shift)
        while True:
            var slot = self.slots[at]
            if slot < 0:
                var id = len(self.first_rows)
                self.slots[at] = id
                self.first_rows.append(row)
                self.first_hashes.append(hash)
                comptime if not Self.GENERIC:
                    self.first_values.append(values[unsafe_offset=row])
                    self.first_valid.append(not self.bits or column._valid(row))
                if 2 * len(self.first_rows) > len(self.slots):
                    self._grow()
                return id
            if self.first_hashes[slot] == hash:
                var same: Bool
                comptime if Self.GENERIC:
                    same = _rows_equal(keys, self.first_rows[slot], row)
                else:
                    var u = not self.bits or column._valid(row)
                    same = u == self.first_valid[slot] and (
                        not u
                        or self.first_values[slot] == values[unsafe_offset=row]
                    )
                if same:
                    return slot
            at = Int((UInt64(at) + 1) & self.mask)

    def into_firsts(deinit self) -> List[Int]:
        """Each number's first row, moved out."""
        return self.first_rows^

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


def number_partitions(
    keys: List[Series],
    hashes_address: Int,
    n: Int,
    ids_address: Int,
    mut firsts: List[Int],
) -> Int:
    """Number the partitions of the keys' n rows by a dictionary of their
    keys, probed with the sink's row hashes (`hashes_address`, one per
    row, in the same order); ids[k] (at `ids_address`) gets row k's number,
    in first-occurrence order, and firsts each number's first row. Returns
    the count."""
    if len(keys) == 1:
        comptime for t in range(len(NUMERIC_DTYPES)):
            comptime D = NUMERIC_DTYPES[t]
            comptime if not D.is_floating_point():
                if keys[0]._data.isa[Column[Scalar[D]]]():
                    return _number_partitions[D, False](
                        keys,
                        keys[0]._data[Column[Scalar[D]]],
                        hashes_address,
                        n,
                        ids_address,
                        firsts,
                    )
    return _number_partitions[DType.int64, True](
        keys,
        Column[Scalar[DType.int64]](List[Int64]()),
        hashes_address,
        n,
        ids_address,
        firsts,
    )


def _number_partitions[
    D: DType, GENERIC: Bool
](
    keys: List[Series],
    column: Column[Scalar[D]],
    hashes_address: Int,
    n: Int,
    ids_address: Int,
    mut firsts: List[Int],
) -> Int:
    var hashes = Pointer[UInt64, MutAnyOrigin](
        unsafe_from_address=hashes_address
    )
    var ids = Pointer[Int, MutAnyOrigin](unsafe_from_address=ids_address)
    var numbering = _Numbering[D, GENERIC](column)
    for k in range(n):
        ids[unsafe_offset=k] = numbering.number(
            keys, column, k, hashes[unsafe_offset=k]
        )
    var count = len(numbering.first_rows)
    firsts = numbering^.into_firsts()
    return count


# Multiplying by a fixed odd constant spreads integer keys over the high
# bits of the product, which is all `_hash_partition` and `_Numbering`
# read; equal keys give equal hashes, so it is a valid partition hash
# that costs one multiply per row.
comptime _ODD_MIX: UInt64 = 0x55FBFD6BFC5458E9


struct _PartitionNumberJob(Job):
    """Partition `part` of `parts`: scan every row and number only the rows
    whose hash falls in this partition, in one thread-local dictionary, so
    no two tasks ever see the same key and nothing is merged afterwards.
    Keeps the partition's rows in row order with their local numbers.

    One integer key is hashed here, as it is read (a multiply, cheaper
    than reading a stored hash), so the keys are scanned once. Any other
    keys read hashes made beforehand (`hashes`), since hashing strings or
    several columns once per partition would repeat real work."""

    var keys: List[Series]
    var hashes: Int
    var height: Int
    var parts: Int
    var part: Int
    # Output: the partition's rows, ascending, their local numbers, and the
    # partition's count.
    var rows: List[Int]
    var locals: List[Int]
    var count: Int

    def __init__(
        out self,
        keys: List[Series],
        hashes: Int,
        height: Int,
        parts: Int,
        part: Int,
    ):
        self.keys = keys.copy()
        self.hashes = hashes
        self.height = height
        self.parts = parts
        self.part = part
        self.rows = List[Int]()
        self.locals = List[Int]()
        self.count = 0

    def run(mut self) raises:
        if self.hashes == 0:
            comptime for t in range(len(NUMERIC_DTYPES)):
                comptime D = NUMERIC_DTYPES[t]
                comptime if not D.is_floating_point():
                    if self.keys[0]._data.isa[Column[Scalar[D]]]():
                        var column = (
                            self.keys[0]._data[Column[Scalar[D]]].copy()
                        )
                        self._number[D, False](column)
                        return
        self._number[DType.int64, True](
            Column[Scalar[DType.int64]](List[Int64]())
        )

    def _number[D: DType, GENERIC: Bool](mut self, column: Column[Scalar[D]]):
        var hashes = Pointer[UInt64, MutAnyOrigin](
            unsafe_from_address=self.hashes
        )
        var values = column._ptr()
        var bits = len(column._bits[]) != 0
        var keys = self.keys.copy()
        var numbering = _Numbering[D, GENERIC](column)
        var rows = List[Int]()
        var locals = List[Int]()
        var parts = self.parts
        var part = self.part
        for k in range(self.height):
            var hash: UInt64
            comptime if GENERIC:
                hash = hashes[unsafe_offset=k]
            else:
                # Every null hashes to 0, so all nulls share a partition.
                if bits and not column._valid(k):
                    hash = 0
                else:
                    hash = (
                        values[unsafe_offset=k].cast[DType.uint64]() * _ODD_MIX
                    )
            if _hash_partition(hash, parts) == part:
                locals.append(numbering.number(keys, column, k, hash))
                rows.append(k)
        self.count = len(numbering.first_rows)
        self.rows = rows^
        self.locals = locals^


struct _GroupRowsJob(Job):
    """One hash partition's groups, numbered from `base`: count each
    group's rows, then write every row into its group's list (rows from
    `row_base` in the shared array), in row order, so each list stays
    ascending. The partition's lists fill one contiguous stretch of the
    array, so no two tasks write the same cache line. Groups are numbered
    in partition order, not by first row: aggregates over partitions do
    not depend on group order."""

    var rows: Int
    var locals: Int
    var entries: Int
    var count: Int
    var base: Int
    var row_base: Int
    var offsets: Int
    var all_rows: Int

    def __init__(
        out self,
        rows: Int,
        locals: Int,
        entries: Int,
        count: Int,
        base: Int,
        row_base: Int,
        offsets: Int,
        all_rows: Int,
    ):
        self.rows = rows
        self.locals = locals
        self.entries = entries
        self.count = count
        self.base = base
        self.row_base = row_base
        self.offsets = offsets
        self.all_rows = all_rows

    def run(mut self) raises:
        var rows = Pointer[Int, MutAnyOrigin](unsafe_from_address=self.rows)
        var locals = Pointer[Int, MutAnyOrigin](unsafe_from_address=self.locals)
        var offsets = Pointer[Int, MutAnyOrigin](
            unsafe_from_address=self.offsets
        )
        var all_rows = Pointer[Int, MutAnyOrigin](
            unsafe_from_address=self.all_rows
        )
        var cursors = List[Int](length=self.count + 1, fill=0)
        var at = cursors.unsafe_ptr()
        for e in range(self.entries):
            at[unsafe_offset=locals[unsafe_offset=e] + 1] += 1
        at[unsafe_offset=0] = self.row_base
        for g in range(self.count):
            at[unsafe_offset=g + 1] += at[unsafe_offset=g]
        for g in range(self.count):
            offsets[unsafe_offset=self.base + g] = at[unsafe_offset=g]
        for e in range(self.entries):
            var local = locals[unsafe_offset=e]
            var row = rows[unsafe_offset=e]
            all_rows[unsafe_offset=at[unsafe_offset=local]] = row
            at[unsafe_offset=local] += 1


def _task_cuts(bounds: List[Int], height: Int) -> List[Int]:
    """Cut hash groups (their row bounds) into tasks of about equal rows,
    each at least one STANDARD_VECTOR_SIZE block, as CreateTaskList sizes
    window tasks in blocks per thread."""
    var blocks = (height + STANDARD_VECTOR_SIZE - 1) // STANDARD_VECTOR_SIZE
    var tasks = max(1, min(configured_workers(), blocks))
    var share = (height + tasks - 1) // tasks
    var cuts = List[Int]()
    cuts.append(0)
    for b in range(1, len(bounds) - 1):
        var at = bounds[b]
        if at < height and at - cuts[len(cuts) - 1] >= share:
            cuts.append(at)
    cuts.append(height)
    return cuts^


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
    whole partitions. One task per thread numbers the rows whose key hash
    falls in its partition. Equal keys have equal hashes, so each key is
    numbered by exactly one task and no tables are merged. Each partition
    then writes its groups' row lists into its own stretch of one array.
    No group number is kept per row: the lists are what the aggregates
    and the write back to rows read."""
    if height == 0:
        return GroupRows(List[Int](length=1, fill=0), List[Int]())
    var workers = configured_workers()
    var parts = workers
    var flat = List[Series](capacity=len(keys))
    var hashes = List[UInt64]()
    if _integer_key(keys):
        flat.append(
            keys[0].rechunk() if keys[0].is_chunked() else keys[0].copy()
        )
    else:
        var partitioner = Partitioner(keys, workers)
        flat = partitioner.keys.copy()
        hashes = partitioner^.into_hashes()
    var jobs = List[_PartitionNumberJob](capacity=parts)
    for p in range(parts):
        jobs.append(
            _PartitionNumberJob(
                flat,
                Int(hashes.unsafe_ptr()) if len(hashes) > 0 else 0,
                height,
                parts,
                p,
            )
        )
    if parts == 1:
        jobs[0].run()
    else:
        run_jobs(jobs)
    var total = 0
    for p in range(parts):
        total += jobs[p].count
    var offsets = List[Int](unsafe_uninit_length=total + 1)
    offsets[total] = height
    var rows = huge_uninit[Int](height)
    var fill = List[_GroupRowsJob](capacity=parts)
    var base = 0
    var row_base = 0
    for p in range(parts):
        fill.append(
            _GroupRowsJob(
                Int(jobs[p].rows.unsafe_ptr()),
                Int(jobs[p].locals.unsafe_ptr()),
                len(jobs[p].rows),
                jobs[p].count,
                base,
                row_base,
                Int(offsets.unsafe_ptr()),
                Int(rows.unsafe_ptr()),
            )
        )
        base += jobs[p].count
        row_base += len(jobs[p].rows)
    if parts == 1:
        fill[0].run()
    else:
        run_jobs(fill)
    # Jobs read these by address: keep them alive until here.
    _ = jobs^
    _ = hashes^
    _ = flat^
    return GroupRows(offsets^, rows^)


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


struct _GroupSortJob(Job):
    """Sort one task's hash groups and mark their boundaries (the SORT,
    MATERIALIZE and MASK stages of a WindowHashGroup). The keys and hashes
    are in hash group order, as the sink materialized them."""

    var keys: List[Series]
    var partition_keys: Int
    var descending: List[Bool]
    var nulls_last: List[Bool]
    # Hash group position -> input row (the scatter's order), and each
    # position's partition key hash.
    var rows: Int
    var hashes: Int
    var lo: Int
    var hi: Int
    # Output: order[lo:hi] and flags[lo:hi].
    var order: Int
    var flags: Int
    var stable: Bool
    # Ranks evaluated over the task's rows as soon as they are sorted, by
    # the same task: no other group's work has to finish first.
    var rank: _RankOut

    def __init__(
        out self,
        keys: List[Series],
        partition_keys: Int,
        descending: List[Bool],
        nulls_last: List[Bool],
        rows: Int,
        hashes: Int,
        lo: Int,
        hi: Int,
        order: Int,
        flags: Int,
        stable: Bool,
        rank: _RankOut,
    ):
        self.keys = keys.copy()
        self.partition_keys = partition_keys
        self.descending = descending.copy()
        self.nulls_last = nulls_last.copy()
        self.rows = rows
        self.hashes = hashes
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
        var rows = Pointer[Int, MutAnyOrigin](unsafe_from_address=self.rows)
        var order = Pointer[Int, MutAnyOrigin](unsafe_from_address=self.order)
        var flags_at = self.flags + self.lo
        var flags = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=flags_at)
        var count = len(self.keys)
        var slices = List[Series](capacity=count)
        for key in self.keys:
            slices.append(key.slice(self.lo, n))
        # The task's partitions numbered by a dictionary of their keys:
        # partitions need only be contiguous, so the sort orders rows by
        # that number (one integer) rather than by the key values.
        var part_keys = List[Series](capacity=self.partition_keys)
        for c in range(self.partition_keys):
            part_keys.append(slices[c].copy())
        var ids = huge_uninit[Int](n)
        var firsts = List[Int]()
        var partitions_here = number_partitions(
            part_keys,
            self.hashes + self.lo * size_of[UInt64](),
            n,
            Int(ids.unsafe_ptr()),
            firsts,
        )
        if partitions_here == 1 and count == self.partition_keys:
            # PartitionKeyTracker::CanBypass: one key and no order keys.
            for k in range(n):
                order[unsafe_offset=self.lo + k] = rows[
                    unsafe_offset=self.lo + k
                ]
                flags[unsafe_offset=k] = 0
            flags[unsafe_offset=0] = PARTITION_START | PEER_START
            return
        if count == self.partition_keys:
            # No order keys: the sort key is the dense partition number
            # alone, so it sorts by counting (stable, rows in input order).
            var at = List[Int](length=partitions_here + 1, fill=0)
            for k in range(n):
                at[ids[k] + 1] += 1
            for g in range(partitions_here):
                at[g + 1] += at[g]
            for k in range(n):
                flags[unsafe_offset=k] = 0
            for g in range(partitions_here):
                flags[unsafe_offset=at[g]] = PARTITION_START | PEER_START
            for k in range(n):
                var to = at[ids[k]]
                at[ids[k]] = to + 1
                order[unsafe_offset=self.lo + to] = rows[
                    unsafe_offset=self.lo + k
                ]
            return
        var sort_keys = List[Series](capacity=1 + count - self.partition_keys)
        var descending = List[Bool](capacity=1 + count - self.partition_keys)
        var nulls_last = List[Bool](capacity=1 + count - self.partition_keys)
        if partitions_here > 1 and partitions_here <= Int(UInt32.MAX):
            # Four bytes hold the partition numbers, so the sort's records
            # stay narrow.
            var narrow = huge_uninit[UInt32](n)
            for k in range(n):
                narrow[k] = UInt32(ids[k])
            sort_keys.append(Series("", Column[UInt32](narrow^)))
        elif partitions_here > 1:
            var wide = huge_uninit[Int64](n)
            for k in range(n):
                wide[k] = Int64(ids[k])
            sort_keys.append(Series("", Column[Int64](wide^)))
        if partitions_here > 1:
            descending.append(False)
            nulls_last.append(False)
        for c in range(self.partition_keys, count):
            sort_keys.append(slices[c].copy())
            descending.append(self.descending[c])
            nulls_last.append(self.nulls_last[c])
        var perm = sort_key_rows(sort_keys, descending, nulls_last, self.stable)
        var previous = -1
        for k in range(n):
            var at = perm[k]
            order[unsafe_offset=self.lo + k] = rows[unsafe_offset=self.lo + at]
            flags[unsafe_offset=k] = (PARTITION_START | PEER_START) if ids[
                at
            ] != previous else 0
            previous = ids[at]
        for c in range(self.partition_keys, count):
            mark_changes(
                slices[c], flags_at, 0, n, PEER_START, Int(perm.unsafe_ptr())
            )
        _ = perm^


struct _MaskJob(Job):
    """Mark peer boundaries over one range of a fully sorted key set (the
    MASK stage, split across threads for one large hash group)."""

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


def window_layout(
    partition_keys: List[Series],
    order_keys: List[Series],
    descending: List[Bool],
    nulls_last: List[Bool],
    height: Int,
    stable: Bool = True,
) raises -> WindowLayout:
    """Sort `height` rows by partition and order keys into hash groups and
    mark their boundaries. `descending` and `nulls_last` hold one flag per
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
        # HashedSort: radix-partition rows by their keys' hash.
        var sink = sink_threads(height)
        var partitioner = Partitioner(partition_keys, sink)
        var bits = radix_bits(height)
        var parts = partitioner.scatter(
            sink, with_hashes=True, bucket_count=1 << bits
        )
        var keys = partitioner.keys.copy()
        for key in order_keys:
            keys.append(key.rechunk() if key.is_chunked() else key.copy())
        # The sink materializes each hash group's rows (partition and order
        # keys) with their hashes (RadixPartitionedTupleData), so the tasks
        # read them in order.
        keys = take_parallel(keys, parts.order.copy(), workers)
        var desc = List[Bool](length=len(partition_keys), fill=False)
        var last = List[Bool](length=len(partition_keys), fill=False)
        for k in range(len(order_keys)):
            desc.append(descending[k])
            last.append(nulls_last[k])
        # Hash groups are sorted by tasks of about equal rows, each at
        # least one STANDARD_VECTOR_SIZE block (CreateTaskList sizes window
        # tasks in blocks per thread). A task takes consecutive hash groups
        # and sorts their rows together: every partition lies in one hash
        # group, so the task's range is a hash group of the layout.
        var cuts = _task_cuts(parts.bounds, height)
        var jobs = List[_GroupSortJob](capacity=len(cuts) - 1)
        for t in range(len(cuts) - 1):
            jobs.append(
                _GroupSortJob(
                    keys,
                    len(partition_keys),
                    desc,
                    last,
                    Int(parts.order.unsafe_ptr()),
                    Int(parts.hashes.unsafe_ptr()),
                    cuts[t],
                    cuts[t + 1],
                    Int(order.unsafe_ptr()),
                    Int(flags.unsafe_ptr()),
                    stable,
                    rank,
                )
            )
            bounds.append(cuts[t])
        if len(jobs) == 1:
            jobs[0].run()
        else:
            run_jobs(jobs)
        _ = parts^
        if rank.method >= 0:
            # Ranks are written; nothing reads the layout.
            _ = order^
            _ = flags^
            return layout^
    elif len(order_keys) > 0:
        # FullSort: one group, sorted on every thread; peers are marked
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
        # NaturalSort: input order, one partition of peers.
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
    boundaries (WindowPeerLocalState::NextRank, and the peer end for max
    and average). The first pass summarizes the range; the second writes
    ranks from the state carried in from the ranges before it."""

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
    rows. With partition keys, each hash group task sorts its rows, marks
    their partition and peer boundaries and ranks them in one go: a hash
    group holds whole partitions, so its ranks depend on no other group
    and no thread waits for the others. One
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
