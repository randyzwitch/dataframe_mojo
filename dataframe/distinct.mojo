"""Count distinct values, overall or per group, by hash partition (#336).

`n_unique` used to keep one standard-library Dict per group as a set. Each
worker filled its own sets over a range of rows and the sets were then
merged one key at a time on one thread, so a column with many distinct
values paid for every key twice, and a skewed group -- one holding most rows
-- got no parallelism at the merge.

Here every row becomes a 128-bit key and a hash: the value's equality key in
the low half (its bits for numbers, every NaN one value and -0.0 equal to
0.0; its row for a string, compared by bytes), and in the high half the
row's group and a flag for null, so a group's null is one value distinct
from all others. Rows are scattered by the top bits of their hash into
partitions small enough for a cache-resident table, in two parallel passes
(count, then write), and each partition counts its distinct keys in an
open-addressing table on its own worker. Equal keys share a partition, so
partitions never need merging: a distinct count is the sum of theirs, and a
group's is the number of new keys each partition saw for it. This is how
DuckDB and Polars count distinct, as grouping on (group, value).
"""
from std.bit import count_leading_zeros
from std.memory import bitcast

from .aggregate import float_key
from .bool_column import BoolColumn
from .column import Column
from .dtype import NUMERIC_DTYPES
from .parallel import Job, Pool, configured_workers
from .partition import _combine, _hash_bytes, _mix
from .series import Series
from .string_column import StringColumn
from .string_bytes import _Bytes, _same_bytes

# Rows per partition to aim for: a table of twice that many 16-byte keys and
# 8-byte hashes stays within a core's L2.
comptime _PARTITION_ROWS = 1 << 13
comptime _MAX_PARTITION_BITS = 12
comptime _NULL_FLAG = UInt128(1) << 64
comptime _LOW = (UInt128(1) << 64) - 1
# Slots in each worker's filter of recently seen keys (see _RowsJob).
comptime _SEEN_SLOTS = 1 << 12


@always_inline
def _bit_length(x: UInt64) -> Int:
    if x == 0:
        return 0
    return 64 - Int(count_leading_zeros(x))


struct _Piece(ImplicitlyCopyable, Movable):
    """Rows [first, last) of one chunk, which starts at row `base`."""

    var chunk: Int
    var base: Int
    var first: Int
    var last: Int

    def __init__(out self, chunk: Int, base: Int, first: Int, last: Int):
        self.chunk = chunk
        self.base = base
        self.first = first
        self.last = last


@always_inline
def _numeric_key[D: DType](value: Scalar[D]) -> UInt64:
    comptime if D.is_floating_point():
        return float_key(Float64(value))
    elif D.is_unsigned():
        return UInt64(value)
    else:
        return bitcast[DType.uint64](Int64(value))


struct _RowsJob(Job):
    """Key and hash every row of one piece, then either count them per
    partition (first pass) or write them to their partition's next slots
    (second pass, with `next` from the counts)."""

    var chunk: Series
    var piece: _Piece
    var groups: Int
    var grouped: Bool
    var shift: Int
    var counts: List[Int]
    var next: List[Int]
    var keys: Int
    var hashes: Int
    var write: Bool
    # A direct-mapped filter of keys this job has already emitted: a row
    # whose key sits in its hash's slot is a duplicate and is dropped here.
    # Without it, a key repeated across most rows (a low-cardinality column,
    # or one dominant value such as an empty string) sends every copy to
    # one partition, and one worker. Both passes read the rows in the same
    # order, so they drop the same rows.
    var seen_hash: List[UInt64]
    var seen_key: List[UInt128]
    var seen: List[Bool]

    def __init__(
        out self,
        chunk: Series,
        piece: _Piece,
        groups: Int,
        grouped: Bool,
        shift: Int,
        partitions: Int,
    ):
        self.chunk = chunk.copy()
        self.piece = piece
        self.groups = groups
        self.grouped = grouped
        self.shift = shift
        self.counts = List[Int](length=partitions, fill=0)
        self.next = List[Int]()
        self.keys = 0
        self.hashes = 0
        self.write = False
        self.seen_hash = List[UInt64]()
        self.seen_key = List[UInt128]()
        self.seen = List[Bool]()

    @always_inline
    def _repeat(
        mut self, key: UInt128, hash: UInt64, bytes: _Bytes, strings: Bool
    ) -> Bool:
        """Whether this key is the one in its filter slot; if not, it takes
        the slot."""
        var slot = Int(hash) & (_SEEN_SLOTS - 1)
        if (
            self.seen.unsafe_ptr()[unsafe_offset=slot]
            and self.seen_hash.unsafe_ptr()[unsafe_offset=slot] == hash
        ):
            var other = self.seen_key.unsafe_ptr()[unsafe_offset=slot]
            if (other >> 64) == (key >> 64):
                if not strings or (key & _NULL_FLAG) != 0:
                    if other == key:
                        return True
                elif _same_bytes(
                    bytes.get(Int(other & _LOW) - self.piece.base),
                    bytes.get(Int(key & _LOW) - self.piece.base),
                ):
                    return True
        self.seen.unsafe_ptr()[unsafe_offset=slot] = True
        self.seen_hash.unsafe_ptr()[unsafe_offset=slot] = hash
        self.seen_key.unsafe_ptr()[unsafe_offset=slot] = key
        return False

    @always_inline
    def _emit(mut self, key: UInt128, hash: UInt64):
        var p = Int(hash >> UInt64(self.shift)) if self.shift < 64 else 0
        if not self.write:
            self.counts.unsafe_ptr()[unsafe_offset=p] += 1
            return
        var slot = self.next.unsafe_ptr()[unsafe_offset=p]
        self.next.unsafe_ptr()[unsafe_offset=p] = slot + 1
        Pointer[List[UInt128], MutAnyOrigin](
            unsafe_from_address=self.keys
        )[].unsafe_ptr()[unsafe_offset=slot] = key
        Pointer[List[UInt64], MutAnyOrigin](
            unsafe_from_address=self.hashes
        )[].unsafe_ptr()[unsafe_offset=slot] = hash

    @always_inline
    def _group(self, row: Int) -> UInt128:
        """The key's high half for this row, less the null flag."""
        if not self.grouped:
            return 0
        var groups = Pointer[List[Int], MutAnyOrigin](
            unsafe_from_address=self.groups
        )[].unsafe_ptr()
        return UInt128(groups[unsafe_offset=self.piece.base + row]) << 65

    def run(mut self) raises:
        # A local handle (buffers are shared): references into self.chunk
        # would be invalidated by _emit, which mutates self.
        var chunk = self.chunk.copy()
        self.seen = List[Bool](length=_SEEN_SLOTS, fill=False)
        self.seen_hash = List[UInt64](length=_SEEN_SLOTS, fill=0)
        self.seen_key = List[UInt128](length=_SEEN_SLOTS, fill=0)
        var no_strings = _Bytes(StringColumn(List[String]()))
        comptime for k in range(len(NUMERIC_DTYPES)):
            comptime D = NUMERIC_DTYPES[k]
            if chunk._data.isa[Column[Scalar[D]]]():
                ref column = chunk._data[Column[Scalar[D]]]
                var nulls = column.null_count() > 0
                for i in range(self.piece.first, self.piece.last):
                    var high = self._group(i)
                    var key: UInt128
                    var hash: UInt64
                    if nulls and not column._valid(i):
                        key = high | _NULL_FLAG
                        hash = _mix(UInt64(key >> 64))
                    else:
                        var value = _numeric_key[D](column._get(i))
                        key = high | UInt128(value)
                        hash = _mix(value) if not self.grouped else _combine(
                            UInt64(high >> 64), value
                        )
                    if not self._repeat(key, hash, no_strings, False):
                        self._emit(key, hash)
                return
        if chunk._data.isa[BoolColumn]():
            ref column = chunk._data[BoolColumn]
            for i in range(self.piece.first, self.piece.last):
                var high = self._group(i)
                var key: UInt128
                var hash: UInt64
                if not column._valid(i):
                    key = high | _NULL_FLAG
                    hash = _mix(UInt64(key >> 64))
                else:
                    var value = UInt64(2) if column._get(i) else UInt64(1)
                    key = high | UInt128(value)
                    hash = _combine(UInt64(high >> 64), value)
                if not self._repeat(key, hash, no_strings, False):
                    self._emit(key, hash)
            return
        ref column = chunk._data[StringColumn]
        var bytes = _Bytes(column)
        var nulls = column.null_count() > 0
        for i in range(self.piece.first, self.piece.last):
            var high = self._group(i)
            var key: UInt128
            var hash: UInt64
            if nulls and not column._valid(i):
                key = high | _NULL_FLAG
                hash = _mix(UInt64(key >> 64))
            else:
                # A string's key holds its global row; equal keys are equal
                # bytes, compared when hashes match.
                key = high | UInt128(self.piece.base + i)
                hash = _combine(UInt64(high >> 64), _hash_bytes(bytes.get(i)))
            if not self._repeat(key, hash, bytes, True):
                self._emit(key, hash)


struct _CountJob(Job):
    """Distinct keys of partitions [first, last); for grouped keys, the
    group of each key seen for the first time."""

    var keys: Int
    var hashes: Int
    var offsets: Int
    var first: Int
    var last: Int
    var strings: Int
    var bases: Int
    var grouped: Bool
    var distinct: Int
    var new_groups: List[Int]

    def __init__(
        out self,
        keys: Int,
        hashes: Int,
        offsets: Int,
        first: Int,
        last: Int,
        strings: Int,
        bases: Int,
        grouped: Bool,
    ):
        self.keys = keys
        self.hashes = hashes
        self.offsets = offsets
        self.first = first
        self.last = last
        self.strings = strings
        self.bases = bases
        self.grouped = grouped
        self.distinct = 0
        self.new_groups = List[Int]()

    def _bytes(
        self, chunks: List[_Bytes], row: Int
    ) -> Span[UInt8, ImmutAnyOrigin]:
        """The bytes of a string at global `row`."""
        ref bases = Pointer[List[Int], MutAnyOrigin](
            unsafe_from_address=self.bases
        )[]
        var c = 0
        while c + 1 < len(bases) and bases[c + 1] <= row:
            c += 1
        return chunks[c].get(row - bases[c])

    def run(mut self) raises:
        var keys = Pointer[List[UInt128], MutAnyOrigin](
            unsafe_from_address=self.keys
        )[].unsafe_ptr()
        var hashes = Pointer[List[UInt64], MutAnyOrigin](
            unsafe_from_address=self.hashes
        )[].unsafe_ptr()
        ref offsets = Pointer[List[Int], MutAnyOrigin](
            unsafe_from_address=self.offsets
        )[]
        var strings = self.strings != 0
        var chunks = List[_Bytes]()
        if strings:
            for chunk in Pointer[List[Series], MutAnyOrigin](
                unsafe_from_address=self.strings
            )[]:
                chunks.append(_Bytes(chunk._data[StringColumn]))
        var table = List[Int32]()
        for p in range(self.first, self.last):
            var start = offsets[p]
            var size = offsets[p + 1] - start
            if size == 0:
                continue
            var capacity = 1 << _bit_length(UInt64(2 * size - 1))
            var mask = capacity - 1
            # Slots hold a key's position in the partition, -1 when empty.
            table.clear()
            table.resize(capacity, -1)
            var slots = table.unsafe_ptr()
            for j in range(size):
                var key = keys[unsafe_offset=start + j]
                var hash = hashes[unsafe_offset=start + j]
                var at = Int(hash) & mask
                while True:
                    var held = Int(slots[unsafe_offset=at])
                    if held < 0:
                        slots[unsafe_offset=at] = Int32(j)
                        self.distinct += 1
                        if self.grouped:
                            self.new_groups.append(Int(key >> 65))
                        break
                    var other = keys[unsafe_offset=start + held]
                    # Equal hashes and equal groups and null flags; then a
                    # number or a null compares its low half, and a string
                    # its bytes.
                    if hashes[unsafe_offset=start + held] == hash and (
                        other >> 64
                    ) == (key >> 64):
                        if not strings or (key & _NULL_FLAG) != 0:
                            if other == key:
                                break
                        elif _same_bytes(
                            self._bytes(chunks, Int(other & _LOW)),
                            self._bytes(chunks, Int(key & _LOW)),
                        ):
                            break
                    at = (at + 1) & mask


def distinct_counts(
    values: Series, groups: List[Int], group_count: Int, grouped: Bool
) raises -> Optional[List[Int64]]:
    """Distinct values of `values` (nulls counting as one), overall or per
    group of `groups`; None for dtypes this does not handle (nested and
    decimal values), which keep the reducer's sets."""
    var dtype = values.dtype()
    if dtype.is_nested() or dtype.is_decimal():
        return None
    var chunks = List[Series]()
    if values.is_chunked():
        for part in values.chunks():
            chunks.append(part.copy())
    else:
        chunks.append(values.copy())
    var n = len(values)
    var bases = List[Int]()
    var base = 0
    for chunk in chunks:
        bases.append(base)
        base += len(chunk)
    var workers = configured_workers() if n >= (1 << 16) else 1
    var bits = min(
        _MAX_PARTITION_BITS, max(0, _bit_length(UInt64(n // _PARTITION_ROWS)))
    )
    var partitions = 1 << bits
    var shift = 64 - bits
    # Pieces of about equal rows, never spanning a chunk boundary.
    var step = max(1, (n + workers - 1) // workers)
    var pieces = List[_Piece]()
    for c in range(len(chunks)):
        var length = len(chunks[c])
        var at = 0
        while at < length:
            var end = min(length, at + step)
            pieces.append(_Piece(c, bases[c], at, end))
            at = end
    var shared_groups = groups.copy()
    var pool = Pool(workers)
    var jobs = List[_RowsJob](capacity=len(pieces))
    for piece in pieces:
        jobs.append(
            _RowsJob(
                chunks[piece.chunk],
                piece,
                Int(Pointer(to=shared_groups)),
                grouped,
                shift,
                partitions,
            )
        )
    pool.run(jobs)
    var offsets = List[Int](length=partitions + 1, fill=0)
    for p in range(partitions):
        var total = 0
        for j in range(len(jobs)):
            total += jobs[j].counts[p]
        offsets[p + 1] = offsets[p] + total
    # Only rows the jobs' duplicate filters kept, and every slot is written.
    var emitted = offsets[partitions]
    var keys = List[UInt128](unsafe_uninit_length=emitted)
    var hashes = List[UInt64](unsafe_uninit_length=emitted)
    var running = offsets.copy()
    for j in range(len(jobs)):
        var next = List[Int](capacity=partitions)
        for p in range(partitions):
            next.append(running[p])
            running[p] += jobs[j].counts[p]
        jobs[j].next = next^
        jobs[j].keys = Int(Pointer(to=keys))
        jobs[j].hashes = Int(Pointer(to=hashes))
        jobs[j].write = True
    pool.run(jobs)
    var strings = values._data.isa[StringColumn]() or (
        values.is_chunked() and chunks[0]._data.isa[StringColumn]()
    )
    var tasks = max(1, min(partitions, 4 * workers))
    var counters = List[_CountJob](capacity=tasks)
    for t in range(tasks):
        counters.append(
            _CountJob(
                Int(Pointer(to=keys)),
                Int(Pointer(to=hashes)),
                Int(Pointer(to=offsets)),
                partitions * t // tasks,
                partitions * (t + 1) // tasks,
                Int(Pointer(to=chunks)) if strings else 0,
                Int(Pointer(to=bases)),
                grouped,
            )
        )
    pool.run(counters, claim=True)
    pool.release()
    var result = List[Int64](length=group_count if grouped else 1, fill=0)
    for t in range(len(counters)):
        if grouped:
            for g in counters[t].new_groups:
                result[g] += 1
        else:
            result[0] += Int64(counters[t].distinct)
    _ = keys^
    _ = hashes^
    _ = offsets^
    _ = chunks^
    _ = bases^
    _ = shared_groups^
    return result^
