"""A parallel, radix-partitioned grouped aggregate hash table.

Each group occupies a stable row of 64-bit words: its hash, key null bits,
first input row, key words, and inline reduction states (such as a sum and
count, a minimum and its seen flag, or Welford moments). Empty rows are
zeroed. Integer and float keys use canonical values. A string key stores
its length and up to twelve inline bytes; longer strings keep a pointer to
their input bytes, which remain alive through aggregation. Equal keys are
confirmed by comparing their canonical words or string bytes.

The algorithm has two phases:

1. Sink. Workers take morsels (row ranges within a chunk) from a shared
   cursor. Each hashes a batch of 2048 rows where they are, with the hash
   the partitioner uses (`partition._hash_column`), and probes its own
   table: a flat array of 8-byte entries, the hash's top 16 bits (a salt)
   over the address of a group's row. A salt that differs rejects a slot
   without reading the row. A new group's row is appended to its
   partition (by the hash bits below the salt) in blocks that never move,
   so the finish reads a partition's rows in sequence with no reordering.
   Each batch updates reduction states through the corresponding group
   addresses. Partitions start at one per worker (at most 16); when a
   worker's rows exceed half a block per partition, all workers move to
   256 partitions.

   With more than two workers the probe table stays fixed-size: at
   two-thirds occupancy its entries are cleared while group rows remain in
   stable storage. This bounds the working set; groups encountered again
   are merged during finish. With two or fewer workers the table doubles
   instead. After a million rows, a worker estimates key cardinality from
   a HyperLogLog sketch; above 95% distinct it stops probing and appends
   each subsequent row as a group.

2. Finish. Consecutive partitions holding about half a 256 KiB block of
   rows form a run. Each run merges every worker's rows for its partitions
   in a table sized for them, folding repeated groups' states into their
   first rows. Consecutive runs holding an equal share of rows per worker
   form jobs on separate threads; each job produces one chunk of the
   result. During vectorized output, counts, sums, and means are written
   directly into their result columns. Other states are loaded into
   `Reducer`s for finalization, and keys are read from group rows, so each
   row is read once. A run containing one worker's rows from a table that
   never cleared its entries has each group only once, so its rows can be
   copied directly. Groups are emitted by partition at the greatest bit
   depth reached by the workers (in first-row order for one worker), then
   by first row, giving stable order for a fixed worker count.

A string longer than twelve bytes is compared through the address of its
bytes in the input, which outlives the aggregation; the input is never
copied.
"""
from dataframe.storage.buffer import _Buffer
from std.atomic import Atomic
from std.bit import count_trailing_zeros
from std.collections import Optional
from std.math import log
from std.ffi import external_call
from std.memory import ArcPointer, Pointer
from std.sys import size_of

from dataframe.aggregation.reducer import Reducer
from dataframe.aggregation.function import (
    AggregateFunction,
    ReducerSpec,
    TOP_ROWS_INLINE,
    _ARENA_BYTES,
    _BATCH,
    _Finalizer,
    _Groups,
    _HEAD,
    _HeldBuffers,
    _KeyColumn,
    _K_COMOMENT,
    _K_DECIMAL_SUM,
    _K_DISTINCT,
    _K_LEN,
    _K_TOP_ROWS,
    _K_VALUES,
    _VALUE_NONE,
    _ValueArena,
    _batch_nulls,
    _column_words,
    _storage_kind,
    _update_states,
    _numeric_part,
    _pair_kind,
    _second_input,
    _slot_kind,
    _words_at,
    combine,
    reaggregable,
    reduction_supported,
    states_supported,
)
from dataframe.storage.boolean import BoolColumn
from dataframe.storage.column import Column
from dataframe.dtype import DataType, NUMERIC_DTYPES
from dataframe.expr import (
    CORR,
    COV,
    LEN,
)
from dataframe.storage.allocation import huge_list, huge_uninit
from dataframe.storage.nested import _leak, _reclaim
from dataframe.parallel import Job, configured_workers, run_jobs
from dataframe.kernels.partition import (
    _NULL_KEY,
    _combine as _hash_combine,
    _hash_bytes,
    _hash_column,
    _mix,
)
from dataframe.series import Series, Storage
from dataframe.kernels.hashing import RowKeys
from dataframe.trace import trace_path
from dataframe.storage.string import StringColumn

# A worker's table when more than two work: DuckDB's 32K-entry sink table,
# cleared at two-thirds full rather than grown.
comptime _SINK_CAPACITY = 1 << 15
# Workers up to which tables grow instead (DuckDB's
# GROW_STRATEGY_THREAD_THRESHOLD).
comptime _GROW_WORKERS = 2
# Rows a worker sinks before it may stop probing, and between retries
# (DuckDB's LOOKUP_SAMPLE_SIZE); the share of a cycle's rows a table must
# remove to keep probing (MINIMUM_SMALL_TABLE_REDUCTION).
comptime _LOOKUP_SAMPLE = 1 << 20
comptime _MIN_REDUCTION = 0.05
# A table grows when the bytes of the duplicate group rows it made exceed
# this many times the larger table's (GROWTH_DUPLICATE_BYTES_FACTOR).
comptime _GROWTH_DUPLICATE_BYTES = 4.0
# HyperLogLog registers.
comptime _HLL_BITS = 10
# Bytes of groups a finish partition holds at most: half of a 256 KiB
# block, where DuckDB adds partition bits.
comptime _PARTITION_BYTES = 128 * 1024
# DuckDB's partition bits (`RadixHTConfig`): log2 of the workers to start,
# at most 4 (MAXIMUM_INITIAL_SINK_RADIX_BITS); more once a worker's rows per
# partition pass half a 256 KiB block (BLOCK_FILL_FACTOR), up to 8.
comptime _MAX_INITIAL_BITS = 4
comptime _MAX_PARTITION_BITS = 8
comptime _BLOCK_BYTES = 256 * 1024
comptime _SALT_SHIFT = UInt64(48)
comptime _ID_MASK = (UInt64(1) << 48) - 1


@always_inline
def _entry(hash: UInt64, at: Int) -> UInt64:
    """A table entry: the hash's top 16 bits as a salt above a group-row
    address or a nonzero group number; zero marks an empty slot."""
    return ((hash >> _SALT_SHIFT) << _SALT_SHIFT) | UInt64(at)


@always_inline
def _stride(hash: UInt64) -> Int:
    """The odd probe step after a collision, derived from the hash salt."""
    return Int((hash >> 59) | 1)


def key_supported(column: Series) -> Bool:
    """Whether a key column's storage has key words here: any integer,
    float, boolean or string column (temporal, decimal and categorical keys
    included, by their storage)."""
    if column.dtype().is_nested():
        return False
    if column.is_chunked():
        return key_supported(column.chunks()[0])
    if column._data.isa[StringColumn]() or column._data.isa[BoolColumn]():
        return True
    if column._data.isa[Column[Int128]]():
        return True
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if column._data.isa[Column[Scalar[D]]]():
            return True
    return False


def layout_widths(layout: AggregateLayout) -> List[Int]:
    """Each key column's word count in a layout."""
    var widths = List[Int](capacity=len(layout.key_templates))
    for key in layout.key_templates:
        widths.append(_width(key))
    return widths^


def _width(column: Series) -> Int:
    """Key words a column takes: two for a string or a 128-bit decimal."""
    var part = column.chunks()[
        0
    ].copy() if column.is_chunked() else column.copy()
    if part._data.isa[StringColumn]() or part._data.isa[Column[Int128]]():
        return 2
    return 1


struct _Partitioned(Movable):
    """A worker's groups are appended to partitions selected by hash bits
    below the salt. Rows occupy stable blocks, so table entries can hold
    row addresses and finish can scan each partition sequentially. Two
    small arrays track the next row and block end for each partition,
    letting an append touch only its destination row."""

    var bits: Int
    var stride: Int
    var states_at: Int
    # Per partition, its blocks of rows: each one's address and rows.
    var blocks: List[List[Int]]
    var block_rows: List[List[Int]]
    # Per partition: the address of its next row, and its last block's end.
    var next: List[Int]
    var end: List[Int]
    # Arenas the blocks are cut from, on huge pages where granted: one
    # allocation per few MiB rather than per block, so a worker's fresh rows
    # do not fault page by page.
    var arenas: List[List[UInt8]]
    var arena_next: Int
    var arena_end: Int

    def __init__(out self, bits: Int, addressed: List[Bool], state_words: Int):
        var width = len(addressed)
        self.bits = bits
        self.states_at = _HEAD + width
        self.stride = _HEAD + width + state_words
        self.blocks = List[List[Int]](capacity=1 << bits)
        self.block_rows = List[List[Int]](capacity=1 << bits)
        for _ in range(1 << bits):
            self.blocks.append(List[Int]())
            self.block_rows.append(List[Int]())
        self.next = List[Int](length=1 << bits, fill=0)
        self.end = List[Int](length=1 << bits, fill=0)
        self.arenas = List[List[UInt8]]()
        self.arena_next = 0
        self.arena_end = 0

    @no_inline
    def _new_block(mut self, p: Int):
        """Another block for partition p: twice the last one's rows, from
        64 up to a 256 KiB block, cut from the current arena."""
        var rows = 64
        var count = len(self.blocks[p])
        if count > 0:
            rows = min(
                2 * self.block_rows[p][count - 1],
                max(64, (256 * 1024) // (8 * self.stride)),
            )
        var bytes = 8 * rows * self.stride
        if self.arena_next + bytes > self.arena_end:
            var size = max(_ARENA_BYTES, bytes)
            self.arenas.append(huge_uninit[UInt8](size))
            self.arena_next = Int(
                self.arenas[len(self.arenas) - 1].unsafe_ptr()
            )
            self.arena_end = self.arena_next + size
        var base = self.arena_next
        self.arena_next += bytes
        self.blocks[p].append(base)
        self.block_rows[p].append(rows)
        self.next[p] = base
        self.end[p] = base + bytes

    @always_inline
    def append(
        mut self,
        hash: UInt64,
        null: UInt64,
        words: List[Int],
        row: Int,
        first: Int,
    ) -> Int:
        """A new group from a batch row (`words` holds each word buffer's
        address), with empty states; its row's address."""
        var p = Int(
            (hash >> UInt64(48 - self.bits)) & UInt64((1 << self.bits) - 1)
        )
        var bytes = 8 * self.stride
        if (
            self.next.unsafe_ptr()[unsafe_offset=p] + bytes
            > self.end.unsafe_ptr()[unsafe_offset=p]
        ):
            self._new_block(p)
        var address = self.next.unsafe_ptr()[unsafe_offset=p]
        self.next.unsafe_ptr()[unsafe_offset=p] = address + bytes
        var at = _words_at(address)
        at[unsafe_offset=0] = hash
        at[unsafe_offset=1] = null
        at[unsafe_offset=2] = UInt64(first)
        for w in range(len(words)):
            at[unsafe_offset=_HEAD + w] = _words_at(words[w])[unsafe_offset=row]
        # Empty states are all zero words.
        for k in range(self.states_at, self.stride):
            at[unsafe_offset=k] = 0
        return address

    def rows_in(self, p: Int, b: Int) -> Int:
        """Rows written to block b of partition p: every row of an earlier
        block (a block is replaced only when no row fits), up to the next
        row in the last."""
        if b < len(self.blocks[p]) - 1:
            return self.block_rows[p][b]
        return (self.next[p] - self.blocks[p][b]) // (8 * self.stride)

    def repartitioned(self, bits: Int) -> _Partitioned:
        """Move these rows into `bits` partitions: each partition's rows
        go, in order, to the
        partitions its hash bits extend to, so a partition's rows keep
        their first-row order."""
        var out = _Partitioned(
            bits,
            List[Bool](length=self.states_at - _HEAD, fill=False),
            self.stride - self.states_at,
        )
        var stride = self.stride
        var bytes = 8 * stride
        var shift = UInt64(48 - bits)
        var mask = UInt64((1 << bits) - 1)
        for p in range(len(self.blocks)):
            for b in range(len(self.blocks[p])):
                var at = self.blocks[p][b]
                for _ in range(self.rows_in(p, b)):
                    var row = _words_at(at)
                    var q = Int((row[] >> shift) & mask)
                    var to = out.next.unsafe_ptr()[unsafe_offset=q]
                    if to + bytes > out.end.unsafe_ptr()[unsafe_offset=q]:
                        out._new_block(q)
                        to = out.next.unsafe_ptr()[unsafe_offset=q]
                    var into = _words_at(to)
                    for w in range(stride):
                        into[unsafe_offset=w] = row[unsafe_offset=w]
                    out.next.unsafe_ptr()[unsafe_offset=q] = to + bytes
                    at += bytes
        return out^


@always_inline
def _same(
    address: Int,
    addressed: List[Bool],
    words: List[Int],
    row: Int,
    null: UInt64,
) -> Bool:
    """Whether the group row at `address` has a batch row's key: null bits
    and every word, and for a long string the bytes behind the address.
    `words` holds each word buffer's address."""
    var base = _words_at(address)
    if base[unsafe_offset=1] != null:
        return False
    # The lists through their pointers: indexing checked bounds on every
    # word of every probed row.
    var long_words = addressed.unsafe_ptr()
    var buffers = words.unsafe_ptr()
    for w in range(len(addressed)):
        var stored = base[unsafe_offset=_HEAD + w]
        var mine = _words_at(buffers[unsafe_offset=w])[unsafe_offset=row]
        if stored == mine:
            continue
        if not long_words[unsafe_offset=w]:
            return False
        # Unequal addresses of a long string (equal lengths, by the word
        # before): compare the bytes.
        var length = Int(base[unsafe_offset=_HEAD - 1 + w] & 0xFFFFFFFF)
        if length <= 12:
            return False
        if external_call["memcmp", Int32](Int(stored), Int(mine), length) != 0:
            return False
    return True


@always_inline
def _same_rows(x_at: Int, y_at: Int, addressed: List[Bool]) -> Bool:
    """Whether two group rows have equal keys (word by word; a long
    string's bytes when its addresses differ)."""
    var x = _words_at(x_at)
    var y = _words_at(y_at)
    if x[unsafe_offset=1] != y[unsafe_offset=1]:
        return False
    for w in range(len(addressed)):
        var u = x[unsafe_offset=_HEAD + w]
        var v = y[unsafe_offset=_HEAD + w]
        if u == v:
            continue
        if not addressed[w]:
            return False
        var length = Int(x[unsafe_offset=_HEAD - 1 + w] & 0xFFFFFFFF)
        if length <= 12:
            return False
        if external_call["memcmp", Int32](Int(u), Int(v), length) != 0:
            return False
    return True


struct _HyperLogLog(Movable):
    """Distinct hashes estimated in 2^_HLL_BITS one-byte registers."""

    var registers: List[UInt8]

    def __init__(out self):
        self.registers = List[UInt8](length=1 << _HLL_BITS, fill=0)

    @always_inline
    def add(mut self, hash: UInt64):
        var index = Int(hash & ((1 << _HLL_BITS) - 1))
        var rest = (hash >> _HLL_BITS) | (UInt64(1) << (64 - _HLL_BITS))
        var rank = UInt8(Int(count_trailing_zeros(rest)) + 1)
        var at = self.registers.unsafe_ptr()
        if rank > at[unsafe_offset=index]:
            at[unsafe_offset=index] = rank

    def estimate(self) -> Float64:
        var m = Float64(1 << _HLL_BITS)
        var total = Float64(0)
        var zeros = 0
        for r in self.registers:
            total += 1.0 / Float64(UInt64(1) << UInt64(r))
            if r == 0:
                zeros += 1
        var raw = 0.7213 / (1.0 + 1.079 / m) * m * m / total
        if raw <= 2.5 * m and zeros > 0:
            return m * log(m / Float64(zeros))
        return raw


def _grown_capacity(
    capacity: Int,
    live: Int,
    made: Int,
    estimate: Float64,
    stride: Int,
    memory: Int,
) -> Int:
    """Return a larger capacity for a full sink table, or 0 when growth
    is not justified. Fit the estimated distinct groups and at least
    double capacity when duplicate group rows created after clearing
    exceed four times the larger table, subject to the worker's memory
    budget."""
    if live == made:
        # Never cleared: no group was made twice.
        return 0
    var distinct = min(Int(estimate), made)
    if distinct == 0:
        return 0
    var minimum = 2 * capacity
    var wanted = 64
    # GetCapacityForCount: the count at a 1.5 load factor, a power of two.
    while 2 * wanted < 3 * distinct:
        wanted *= 2
    var next = max(wanted, minimum)
    if Float64(made - distinct) * Float64(8 * stride) < (
        _GROWTH_DUPLICATE_BYTES * Float64(8 * next)
    ):
        return 0
    var current = 8 * (made * stride + capacity)
    while next >= minimum:
        if current <= memory and next <= (memory - current) // 8:
            break
        next //= 2
    if next < minimum:
        return 0
    return next


def _worker_memory(workers: Int) -> Int:
    """A worker's share of sink memory: 80% of physical memory divided
    among the workers."""
    var pages = external_call["sysconf", Int](Int32(85))  # _SC_PHYS_PAGES
    var size = external_call["sysconf", Int](Int32(30))  # _SC_PAGESIZE
    if pages <= 0 or size <= 0:
        return 1 << 30
    return Int(0.8 * Float64(pages) * Float64(size)) // max(1, workers)


@fieldwise_init
struct _Cursor(Copyable, Movable):
    """The next morsel, shared by every worker (an atomic in C memory)."""

    var address: Int

    @staticmethod
    def new() -> _Cursor:
        return _Cursor(
            external_call["calloc", Int](1, size_of[Atomic[Int64]]())
        )

    def take(self) -> Int:
        return Int(
            Pointer[Atomic[Int64], MutAnyOrigin](
                unsafe_from_address=self.address
            )[].fetch_add(1)
        )

    def free(self):
        _ = external_call["free", NoneType](self.address)


struct _SinkBits(Movable):
    """The partition bit count used by every worker's sink. It only
    increases and is unset until a sink starts."""

    var bits: Atomic[Int64]

    def __init__(out self):
        self.bits = Atomic[Int64](-1)

    def start(mut self, bits: Int) -> Int:
        """The bits now, `bits` if no sink has started."""
        var expected = Int64(-1)
        if self.bits.compare_exchange(expected, Int64(bits)):
            return bits
        return Int(self.bits.load())

    def raise_to(mut self, bits: Int) -> Int:
        """At least `bits` from now on; the bits now."""
        while True:
            var now = self.bits.load()
            if now >= Int64(bits):
                return Int(now)
            var expected = now
            if self.bits.compare_exchange(expected, Int64(bits)):
                return bits


def _initial_bits(workers: Int) -> Int:
    """Choose log2 of the worker count rounded up to a power of two, with
    a maximum of 4 bits."""
    var bits = 0
    while (1 << bits) < workers:
        bits += 1
    return min(bits, _MAX_INITIAL_BITS)


def _maximum_bits(workers: Int) -> Int:
    """Choose the maximum partition depth: the initial depth when tables
    grow (two workers or fewer), otherwise 8 bits. The finish sizes runs
    using whole partitions, so limiting partition depth avoids oversized
    runs and keeps their working tables cache-friendly."""
    if workers <= _GROW_WORKERS:
        return _initial_bits(workers)
    return _MAX_PARTITION_BITS


@fieldwise_init
struct AggregateLayout(Copyable, Movable):
    """What every worker's sink and the finish share: each reduction's
    state slot and template, which key words are long strings' addresses,
    the state words, the partition bits and empty key columns."""

    var slots: List[AggregateFunction]
    var templates: List[ReducerSpec]
    var addressed: List[Bool]
    var state_words: Int
    var radix: ArcPointer[_SinkBits]
    var key_templates: List[Series]
    # The distinct counts (DuckDB's `DistinctAggregateCollectionInfo`):
    # each one's reduction, its distinct table's key columns (the keys,
    # then the counted input) and that table's partition bits, shared by
    # every worker's distinct table.
    var distinct: List[Int]
    var distinct_keys: List[List[Series]]
    var distinct_addressed: List[List[Bool]]
    var distinct_radix: List[ArcPointer[_SinkBits]]

    def distinct_layout(self, d: Int) -> AggregateLayout:
        """Distinct count d's table: its keys and no states."""
        return AggregateLayout(
            List[AggregateFunction](),
            List[ReducerSpec](),
            self.distinct_addressed[d].copy(),
            0,
            self.distinct_radix[d],
            self.distinct_keys[d].copy(),
            List[Int](),
            List[List[Series]](),
            List[List[Bool]](),
            List[ArcPointer[_SinkBits]](),
        )


def aggregate_layout(
    keys: List[Series], inputs: List[Series], templates: List[ReducerSpec]
) raises -> AggregateLayout:
    """The layout of a table grouping columns like `keys` and reducing
    columns like `inputs` (only their types are read). Every reduction's
    state must fit inline (`states_supported`)."""
    var width = 0
    var addressed = List[Bool]()
    for key in keys:
        var part = key.chunks()[0].copy() if key.is_chunked() else key.copy()
        var strings = part._data.isa[StringColumn]()
        if _width(key) == 2:
            addressed.append(False)
            addressed.append(strings)
            width += 2
        else:
            addressed.append(False)
            width += 1
    if width > 64:
        raise Error("aggregate table: more than 64 key words")
    var slots = List[AggregateFunction](capacity=len(templates))
    var state_words = 0
    for s in range(len(templates)):
        var second = 0
        var kind: Int
        if templates[s].op == CORR or templates[s].op == COV:
            second = _second_input(templates, s)
            kind = (
                _pair_kind(templates[s], inputs[s], inputs[second]) if second
                < len(inputs) else -1
            )
        else:
            kind = _slot_kind(templates[s], inputs[s])
        if kind < 0:
            raise Error("aggregate table: a state that is not inline")
        var floating = False
        if kind != _K_LEN and kind != _K_DECIMAL_SUM and kind != _K_COMOMENT:
            # None for a decimal's counts, which hold no values.
            var part = _numeric_part(inputs[s])
            if part:
                floating = part.value().dtype().is_float()
        var storage = _VALUE_NONE
        if kind != _K_LEN and kind != _K_COMOMENT:
            storage = _storage_kind(inputs[s])
        slots.append(
            AggregateFunction(
                kind,
                state_words,
                floating,
                templates[s].make(0).dtype,
                second,
                False,
                storage,
            )
        )
        state_words += slots[s].state_size()
    var key_templates = List[Series](capacity=len(keys))
    for key in keys:
        key_templates.append(key.slice(0, 0))
    var distinct = List[Int]()
    var distinct_keys = List[List[Series]]()
    var distinct_addressed = List[List[Bool]]()
    var distinct_radix = List[ArcPointer[_SinkBits]]()
    for s in range(len(slots)):
        if slots[s].kind != _K_DISTINCT:
            continue
        var columns = key_templates.copy()
        columns.append(inputs[s].slice(0, 0))
        var words = addressed.copy()
        words.append(False)
        if _width(inputs[s]) == 2:
            var part = inputs[s].chunks()[0].copy() if inputs[
                s
            ].is_chunked() else inputs[s].copy()
            words.append(part._data.isa[StringColumn]())
        if len(words) > 64:
            raise Error("aggregate table: more than 64 key words")
        distinct.append(s)
        distinct_keys.append(columns^)
        distinct_addressed.append(words^)
        distinct_radix.append(ArcPointer(_SinkBits()))
    return AggregateLayout(
        slots^,
        templates.copy(),
        addressed^,
        state_words,
        ArcPointer(_SinkBits()),
        key_templates^,
        distinct^,
        distinct_keys^,
        distinct_addressed^,
        distinct_radix^,
    )


struct _DictionaryKeys(Movable):
    """Cache each dictionary entry's string hash and two key words, and
    compute them only the first time a row uses its code for a given
    dictionary identity."""

    var identity: Int
    # The dictionary's offsets and bytes, by address (each read through
    # its shared pointer cost an atomic count a row on every worker).
    var offsets: Int
    var bytes: Int
    var done: List[Bool]
    var keys: List[UInt64]
    var first: List[UInt64]
    var second: List[UInt64]
    # The morsel's codes, one per row.
    var codes: List[UInt32]

    def __init__(out self):
        self.identity = 0
        self.offsets = 0
        self.bytes = 0
        self.done = List[Bool]()
        self.keys = List[UInt64]()
        self.first = List[UInt64]()
        self.second = List[UInt64]()
        self.codes = List[UInt32]()

    def start(mut self, strings: StringColumn):
        """Ready for a morsel of `strings`, which carries codes: their
        dictionary's entries (reset when it is a new one) and its codes."""
        ref held = strings._codes.value()[]
        ref dictionary = held.dtype.dictionary()[]
        var identity = Int(Pointer(to=dictionary))
        if identity != self.identity:
            self.identity = identity
            self.offsets = Int(dictionary.offsets.unsafe_ptr())
            self.bytes = Int(dictionary.bytes.unsafe_ptr())
            var n = len(dictionary)
            self.done = List[Bool](length=n, fill=False)
            self.keys = List[UInt64](length=n, fill=0)
            self.first = List[UInt64](length=n, fill=0)
            self.second = List[UInt64](length=n, fill=0)
        if len(self.codes) < len(strings):
            self.codes = List[UInt32](length=len(strings), fill=0)
        strings._write_codes(Int(self.codes.unsafe_ptr()))

    @no_inline
    def _entry(mut self, strings: StringColumn, code: Int):
        """Entry `code`'s hash key and words: as `_hash_column` and
        `_string_words` give them for the same string."""
        var offsets = Pointer[Int64, ImmutAnyOrigin](
            unsafe_from_address=self.offsets
        )
        var start = Int(offsets[unsafe_offset=code])
        var length = Int(offsets[unsafe_offset=code + 1]) - start
        var bytes = Pointer[UInt8, MutAnyOrigin](
            unsafe_from_address=self.bytes + start
        )
        self.keys[code] = _hash_bytes(
            Span[UInt8, ImmutAnyOrigin](
                unsafe_ptr=Pointer[UInt8, ImmutAnyOrigin](
                    unsafe_from_address=self.bytes + start
                ),
                length=length,
            )
        )
        var head = UInt64(length)
        for b in range(min(length, 4)):
            head |= UInt64(bytes[unsafe_offset=b]) << UInt64(32 + 8 * b)
        self.first[code] = head
        if length <= 12:
            var tail = UInt64(0)
            for b in range(4, length):
                tail |= UInt64(bytes[unsafe_offset=b]) << UInt64(8 * (b - 4))
            self.second[code] = tail
        else:
            self.second[code] = UInt64(self.bytes + start)
        self.done[code] = True

    def write(
        mut self,
        strings: StringColumn,
        lo: Int,
        count: Int,
        hashes_at: Int,
        first_word: Int,
        second_word: Int,
        nulls_at: Int,
        bit: UInt64,
        leading: Bool,
    ):
        """Batch rows [lo, lo + count) of the morsel: hashes (combined as
        `_hash_column` combines, `leading` for the first key) and key words
        through their codes."""
        var first = Pointer[UInt64, MutAnyOrigin](
            unsafe_from_address=first_word
        )
        var second = Pointer[UInt64, MutAnyOrigin](
            unsafe_from_address=second_word
        )
        var nulls = Pointer[UInt64, MutAnyOrigin](unsafe_from_address=nulls_at)
        var hashes = Pointer[UInt64, MutAnyOrigin](
            unsafe_from_address=hashes_at
        )
        var has_nulls = _batch_nulls(
            strings._bits[], strings._offset, lo, count
        )
        var codes = self.codes.unsafe_ptr()
        for k in range(count):
            var key: UInt64
            if has_nulls and not strings._valid(lo + k):
                key = _NULL_KEY
                first[unsafe_offset=k] = 0
                second[unsafe_offset=k] = 0
                nulls[unsafe_offset=k] |= bit
            else:
                var code = Int(codes[unsafe_offset=lo + k])
                if not self.done.unsafe_ptr()[unsafe_offset=code]:
                    self._entry(strings, code)
                key = self.keys.unsafe_ptr()[unsafe_offset=code]
                first[unsafe_offset=k] = self.first.unsafe_ptr()[
                    unsafe_offset=code
                ]
                second[unsafe_offset=k] = self.second.unsafe_ptr()[
                    unsafe_offset=code
                ]
            hashes[unsafe_offset=k] = _mix(key) if leading else _hash_combine(
                hashes[unsafe_offset=k], key
            )


struct _DistinctSinks(Movable):
    """A worker's distinct tables (`AggregateSink`s), held by address: a
    sink cannot hold its own kind in a field."""

    var addresses: List[Int]

    def __init__(out self):
        self.addresses = List[Int]()

    def __deinit__(deinit self):
        for address in self.addresses:
            if address != 0:
                _ = _reclaim[AggregateSink](address)

    def add(mut self, var sink: AggregateSink):
        self.addresses.append(_leak[AggregateSink](sink^))

    def at(self, d: Int) -> ref[MutAnyOrigin] AggregateSink:
        return Pointer[AggregateSink, MutAnyOrigin](
            unsafe_from_address=self.addresses[d]
        )[]

    def take(mut self, d: Int) -> AggregateSink:
        """Table d, moved out (once)."""
        var sink = _reclaim[AggregateSink](self.addresses[d])
        self.addresses[d] = 0
        return sink^


struct AggregateSink(Movable):
    """One worker's sink state: its table of salted row addresses,
    partitioned group rows, and counters used to adapt fill cycles. Rows
    are supplied a morsel at a time by `sink`."""

    var slots: List[AggregateFunction]
    var addressed: List[Bool]
    var parts: _Partitioned
    # Every sink's partition bits, and the most this one's may reach.
    var radix: ArcPointer[_SinkBits]
    var max_bits: Int
    var grow: Bool
    # This worker's share of memory, in bytes.
    var memory: Int
    var capacity: Int
    var entries: List[UInt64]
    # Groups in the table this cycle, and every group row this worker
    # made (DuckDB's materialized count).
    var live: Int
    var made: Int
    # Rows sunk; rows probed and skipped since the table was last
    # cleared, and since lookups stopped.
    var sunk: Int
    var cycle: Int
    var skipped: Int
    var skip: Bool
    # Whether a full table ever lost its entries (`Abandon`): until then
    # every group has one row here.
    var abandoned: Bool
    var hll: _HyperLogLog
    # Batch buffers, kept across batches.
    var hashes: List[UInt64]
    var nulls: List[UInt64]
    var groups: List[Int]
    var word_rows: List[List[UInt64]]
    # Per key column, its dictionary's entries when its strings carry
    # codes.
    var dictionaries: List[_DictionaryKeys]
    # Every morsel's string key columns, held until the finish: a group's
    # long string key is the address of its bytes there, as DuckDB's rows
    # point into their collection's heap (a morsel's column holds its
    # buffers by reference, so nothing is copied). The finished key
    # columns are views into the same buffers (`_HeldBuffers`).
    var held: List[Series]
    # Numbering mode (`table_row_keys`): each new group's dense id in its
    # first state word, every row's id, and each group's first row.
    var numbering: Bool
    var row_ids: List[Int]
    var representatives: List[Int]
    # A group row's worth of words that rows a filter dropped update
    # (`sink`'s `keep`): read by nothing.
    var discard: List[UInt64]
    # The segments of the quantile values, which the finish reads.
    var values: _ValueArena
    # Three vectors of words the value states' updates write their rows'
    # value words into.
    var scratch: List[UInt64]
    # Per distinct count (`AggregateLayout.distinct`), this worker's table
    # of its (keys, value) groups, which every morsel also sinks into
    # (DuckDB's `SinkDistinct`); and, in a table the distinct groups are
    # sunk into at the finish, the one reduction rows count into.
    var distinct: _DistinctSinks
    var distinct_slots: List[Int]
    var only: Int

    def __init__(
        out self,
        layout: AggregateLayout,
        workers: Int,
        shares: Int = 1,
        distinct: Bool = True,
    ):
        self.slots = layout.slots.copy()
        self.addressed = layout.addressed.copy()
        self.radix = layout.radix
        self.parts = _Partitioned(
            self.radix[].start(_initial_bits(workers)),
            layout.addressed,
            layout.state_words,
        )
        self.max_bits = _maximum_bits(workers)
        self.grow = workers <= _GROW_WORKERS
        # A worker with several tables (one per grouping set) divides its
        # share among them.
        self.memory = _worker_memory(workers) // max(1, shares)
        self.capacity = _SINK_CAPACITY
        self.entries = huge_list(self.capacity, UInt64(0))
        self.live = 0
        self.made = 0
        self.sunk = 0
        self.cycle = 0
        self.skipped = 0
        self.skip = False
        self.abandoned = False
        self.hll = _HyperLogLog()
        self.hashes = List[UInt64](length=_BATCH, fill=0)
        self.nulls = List[UInt64](length=_BATCH, fill=0)
        self.groups = List[Int](length=_BATCH, fill=0)
        self.word_rows = List[List[UInt64]](capacity=len(layout.addressed))
        for _ in range(len(layout.addressed)):
            self.word_rows.append(List[UInt64](length=_BATCH, fill=0))
        self.dictionaries = List[_DictionaryKeys](
            capacity=len(layout.key_templates)
        )
        for _ in range(len(layout.key_templates)):
            self.dictionaries.append(_DictionaryKeys())
        self.held = List[Series]()
        self.numbering = False
        self.discard = List[UInt64](length=self.parts.stride + 8, fill=0)
        self.values = _ValueArena()
        self.scratch = List[UInt64](length=3 * _BATCH, fill=0)
        self.distinct = _DistinctSinks()
        self.distinct_slots = List[Int]()
        self.only = -1
        if distinct:
            self.distinct_slots = layout.distinct.copy()
            for d in range(len(layout.distinct)):
                self.distinct.add(
                    AggregateSink(
                        layout.distinct_layout(d),
                        workers,
                        shares,
                        distinct=False,
                    )
                )
        self.row_ids = List[Int]()
        self.representatives = List[Int]()

    def _full(mut self, count: Int):
        """Handle a table without room for `count` more groups: grow it
        when duplicates justify the added capacity; otherwise clear its
        entries but retain group rows, and decide whether to skip lookups."""
        if self.grow:
            while 3 * (self.live + count) >= 2 * self.capacity:
                self.capacity *= 2
            self.entries = huge_list(self.capacity, UInt64(0))
            self._reinsert()
            return
        self.abandoned = True
        var grown = 0
        if not self.skip:
            grown = _grown_capacity(
                self.capacity,
                self.live,
                self.made,
                self.hll.estimate(),
                self.parts.stride,
                self.memory,
            )
        if grown > 0:
            self.capacity = grown
            self.entries = huge_list(self.capacity, UInt64(0))
        else:
            if (
                not self.skip
                and self.sunk >= _LOOKUP_SAMPLE
                and self.capacity == _SINK_CAPACITY
                and Float64(self.cycle - self.live)
                < _MIN_REDUCTION * Float64(self.cycle)
            ):
                # The cycle removed almost no rows: append without
                # lookups (`MaybeSkipLookups`).
                self.skip = True
                self.skipped = 0
            var table = self.entries.unsafe_ptr()
            for e in range(self.capacity):
                table[unsafe_offset=e] = 0
            if self.skip and self.skipped >= _LOOKUP_SAMPLE:
                # Retry lookups (`MaybeResumeLookups`).
                self.skip = False
        self.live = 0
        self.cycle = 0
        # With its entries gone, the rows may move (`MaybeRepartition`):
        # the most bits for every sink once this one's rows fill half a
        # block per partition. DuckDB adds two bits at a time; each step
        # copies every row made so far, which cost a fifth of a 100K-group
        # aggregation here, so one step copies them once.
        var bits = self.parts.bits
        if 8 * self.made * self.parts.stride > (_BLOCK_BYTES // 2) << bits:
            bits = self.radix[].raise_to(self.max_bits)
        self.repartition(max(bits, Int(self.radix[].bits.load())))

    def repartition(mut self, bits: Int):
        """This sink's rows in `bits` partitions, when it has fewer."""
        if bits > self.parts.bits:
            self.parts = self.parts.repartitioned(bits)

    def sink(
        mut self,
        given: List[Series],
        given_inputs: List[Series],
        first: Int,
        keep: List[Bool] = List[Bool](),
    ) raises:
        """A morsel's rows into the table: `keys` and `inputs` (a row count
        reads none) as unchunked columns of its rows. When `keep` is given
        (one flag a row), only its rows join groups: the others (a filter's
        dropped rows, read in place rather than compacted away) update a
        discarded group row, so no group appears for them. `first` is the source
        row of its first."""
        if len(given) == 0:
            raise Error("aggregate table: no keys")
        # A string key without codes whose views select another column's
        # descriptors (a join's build side) is read a batch at a time
        # through pointers: its descriptors are copied out once here, not
        # once a batch. One with codes is read through them alone.
        var keys = List[Series](capacity=len(given))
        for c in range(len(given)):
            if given[c]._data.isa[StringColumn]():
                ref strings = given[c]._data[StringColumn]
                if not strings._codes and strings._is_view_storage():
                    if strings._view_storage.value()._selects():
                        var flat = given[c].copy()
                        flat._data = Storage(strings._flat_views())
                        keys.append(flat^)
                        continue
            keys.append(given[c].copy())
        # Value states over strings hold their bytes' addresses: their
        # inputs are flattened the same way, and held until the finish.
        var inputs = given_inputs.copy()
        for s in range(len(self.slots)):
            if self.only >= 0 or not self.slots[s].holds_strings():
                continue
            ref strings = inputs[s]._data[StringColumn]
            if strings._is_view_storage():
                if strings._view_storage.value()._selects():
                    var flat = inputs[s].copy()
                    flat._data = Storage(strings._flat_views())
                    inputs[s] = flat^
            self.held.append(inputs[s].copy())
        var length = len(keys[0])
        var width = len(self.addressed)
        var word_ptrs = List[Int](capacity=width)
        for w in range(width):
            word_ptrs.append(Int(self.word_rows[w].unsafe_ptr()))
        var group_at = self.groups.unsafe_ptr()
        var nulls = self.nulls.unsafe_ptr()
        var hash_ptr = self.hashes.unsafe_ptr()
        # String keys carrying their source's dictionary codes are read
        # through the dictionary's entries.
        var coded = List[Bool](capacity=len(keys))
        for c in range(len(keys)):
            if keys[c]._data.isa[StringColumn]():
                self.held.append(keys[c].copy())
            var by_codes = keys[c]._data.isa[StringColumn]() and Bool(
                keys[c]._data[StringColumn]._codes
            )
            if by_codes:
                # DuckDB's TryAddDictionaryGroups: a stored dictionary's
                # entries are looked up once only while it holds fewer
                # than MAX_DICTIONARY_SIZE_THRESHOLD (20,000); a larger one
                # is hashed row by row like any string.
                ref held = keys[c]._data[StringColumn]._codes.value()[]
                by_codes = len(held.dtype.dictionary()[]) < 20_000
            coded.append(by_codes)
            if by_codes:
                self.dictionaries[c].start(keys[c]._data[StringColumn])
        for lo in range(0, length, _BATCH):
            var count = min(_BATCH, length - lo)
            for k in range(count):
                nulls[unsafe_offset=k] = 0
            # Hashes and key words, where the rows are, column by column
            # (each combines into the hashes of the ones before).
            var w = 0
            for c in range(len(keys)):
                var second = w + 1 if _width(keys[c]) == 2 else w
                if coded[c]:
                    self.dictionaries[c].write(
                        keys[c]._data[StringColumn],
                        lo,
                        count,
                        Int(hash_ptr),
                        word_ptrs[w],
                        word_ptrs[second],
                        Int(nulls),
                        UInt64(1) << UInt64(c),
                        c == 0,
                    )
                else:
                    _hash_column(
                        keys[c],
                        lo,
                        lo + count,
                        Int(hash_ptr),
                        c == 0,
                        lo,
                    )
                    _column_words(
                        keys[c],
                        lo,
                        count,
                        word_ptrs[w],
                        word_ptrs[second],
                        Int(nulls),
                        UInt64(1) << UInt64(c),
                    )
                w = second + 1
            var kept = len(keep) > 0
            # Rows a filter dropped are skipped in their own instantiation,
            # so a sink without a selection runs the loops it always had.
            if kept:
                self._insert[True](
                    count,
                    first + lo,
                    Int(hash_ptr),
                    Int(nulls),
                    word_ptrs,
                    Int(group_at),
                    Int(keep.unsafe_ptr().unsafe_offset(lo)),
                )
            else:
                self._insert[False](
                    count,
                    first + lo,
                    Int(hash_ptr),
                    Int(nulls),
                    word_ptrs,
                    Int(group_at),
                    0,
                )
            if self.numbering:
                var states_at = 8 * self.parts.states_at
                var ids = self.row_ids.unsafe_ptr()
                for k in range(count):
                    ids[unsafe_offset=first + lo + k] = Int(
                        _words_at(group_at[unsafe_offset=k] + states_at)[]
                    )
            # The batch's states, by the address of each row's group.
            _update_states(
                self.slots,
                inputs,
                lo,
                count,
                group_at,
                self.parts.states_at,
                first,
                Pointer[_ValueArena, MutAnyOrigin](
                    unsafe_from_address=Int(Pointer(to=self.values))
                ),
                Int(self.discard.unsafe_ptr()),
                Int(self.scratch.unsafe_ptr()),
                self.only,
            )
        self.sunk += length
        # Each distinct count's (keys, value) pairs into its own table.
        for d in range(len(self.distinct_slots)):
            var pairs = keys.copy()
            pairs.append(inputs[self.distinct_slots[d]].copy())
            self.distinct.at(d).sink(pairs, List[Series](), first, keep)

    def _insert[
        masked: Bool
    ](
        mut self,
        count: Int,
        row: Int,
        hashes: Int,
        nulls: Int,
        word_ptrs: List[Int],
        groups: Int,
        keep: Int,
    ) raises:
        """A batch's `count` rows (source rows from `row`) into the table:
        each row's group address into `groups`. With `masked`, rows whose
        `keep` flag is off join no group and point at the discarded row."""
        var hash_ptr = Pointer[UInt64, MutAnyOrigin](unsafe_from_address=hashes)
        var null_ptr = Pointer[UInt64, MutAnyOrigin](unsafe_from_address=nulls)
        var group_at = Pointer[Int, MutAnyOrigin](unsafe_from_address=groups)
        var keep_at = Pointer[Bool, MutAnyOrigin](unsafe_from_address=keep)
        var discard = Int(self.discard.unsafe_ptr())
        if 3 * (self.live + count) >= 2 * self.capacity:
            self._full(count)
        self.cycle += count
        if self.skip:
            var appended = 0
            for k in range(count):
                comptime if masked:
                    if not keep_at[unsafe_offset=k]:
                        group_at[unsafe_offset=k] = discard
                        continue
                appended += 1
                group_at[unsafe_offset=k] = self.parts.append(
                    hash_ptr[unsafe_offset=k],
                    null_ptr[unsafe_offset=k],
                    word_ptrs,
                    k,
                    row + k,
                )
            if not self.grow:
                # While lookups are skipped every row is a group, and the
                # sketch sees them all (DuckDB's `ObserveHashes`).
                for k in range(count):
                    comptime if masked:
                        if not keep_at[unsafe_offset=k]:
                            continue
                    self.hll.add(hash_ptr[unsafe_offset=k])
            self.made += appended
            self.live += appended
            self.skipped += appended
            return
        var mask = self.capacity - 1
        var table = _words_at(Int(self.entries.unsafe_ptr()))
        for k in range(count):
            comptime if masked:
                if not keep_at[unsafe_offset=k]:
                    group_at[unsafe_offset=k] = discard
                    continue
            var hash = hash_ptr[unsafe_offset=k]
            var salt = (hash >> _SALT_SHIFT) << _SALT_SHIFT
            var slot = Int(hash) & mask
            var null = null_ptr[unsafe_offset=k]
            while True:
                var entry = table[unsafe_offset=slot]
                if entry == 0:
                    var at = self.parts.append(
                        hash, null, word_ptrs, k, row + k
                    )
                    if self.numbering:
                        _words_at(at + 8 * self.parts.states_at)[] = UInt64(
                            self.made
                        )
                        self.representatives.append(row + k)
                    table[unsafe_offset=slot] = _entry(hash, at)
                    # The sketch sees only new groups: a row whose group
                    # is in the table has had its hash seen already.
                    if not self.grow:
                        self.hll.add(hash)
                    self.live += 1
                    self.made += 1
                    group_at[unsafe_offset=k] = at
                    break
                if (entry & ~_ID_MASK) == salt:
                    var at = Int(entry & _ID_MASK)
                    if _words_at(at)[] == hash and _same(
                        at, self.addressed, word_ptrs, k, null
                    ):
                        group_at[unsafe_offset=k] = at
                        break
                slot = (slot + _stride(hash)) & mask

    def sink_group_rows(
        mut self,
        source: AggregateSink,
        columns: List[Int],
        widths: List[Int],
        masks: List[Int64] = List[Int64](),
    ) raises:
        """The group rows of another sink, with this table's keys the
        source's `columns` (in this order; none for a constant key of one
        zero word), each one-word key with only its `masks` entry's bits
        kept when given (-1 keeps it whole), into this table with their states merged rather than
        rebuilt from values: a coarser grouping from a finer one's groups
        (DuckDB merges grouped states the same way across threads, and
        `combine` is the finish's merge of the workers'). `widths` is each
        source key column's word count. Both tables hold the same reductions,
        all of which `reaggregable`. Rows are read where they are; the
        source's key buffers are kept for the long strings they point at.
        This does not touch `sink`'s per-row update."""
        var starts = List[Int](capacity=len(widths))
        var at = 0
        for c in range(len(widths)):
            starts.append(at)
            at += widths[c]
        var width = len(self.addressed)
        var word_ptrs = List[Int](capacity=width)
        for w in range(width):
            word_ptrs.append(Int(self.word_rows[w].unsafe_ptr()))
        var sources = List[Int](length=_BATCH, fill=0)
        var firsts = List[UInt64](length=_BATCH, fill=0)
        var source_at = sources.unsafe_ptr()
        var first_at = firsts.unsafe_ptr()
        var hashes = self.hashes.unsafe_ptr()
        var nulls = self.nulls.unsafe_ptr()
        var stride = source.parts.stride
        var count = 0
        for p in range(len(source.parts.blocks)):
            for b in range(len(source.parts.blocks[p])):
                var base = source.parts.blocks[p][b]
                for r in range(source.parts.rows_in(p, b)):
                    var address = base + 8 * r * stride
                    var row = _words_at(address)
                    var hash = UInt64(0)
                    var null = UInt64(0)
                    var to = 0
                    if len(columns) == 0:
                        _words_at(word_ptrs[0])[unsafe_offset=count] = 0
                        hash = _mix(UInt64(0))
                    for j in range(len(columns)):
                        var c = columns[j]
                        var from_word = _HEAD + starts[c]
                        var part: UInt64
                        var first = row[unsafe_offset=from_word]
                        if len(masks) > 0 and widths[c] == 1:
                            # A packed key's fields this set rolls up.
                            first &= masks[j].cast[DType.uint64]()
                        _words_at(word_ptrs[to])[unsafe_offset=count] = first
                        if (row[unsafe_offset=1] >> UInt64(c)) & 1 != 0:
                            null |= UInt64(1) << UInt64(j)
                            part = _NULL_KEY
                        else:
                            part = first
                        if widths[c] == 2:
                            var second = row[unsafe_offset=from_word + 1]
                            _words_at(word_ptrs[to + 1])[
                                unsafe_offset=count
                            ] = second
                            var length = Int(first & 0xFFFFFFFF)
                            if (
                                source.addressed[starts[c] + 1]
                                and length > 12
                                and part != _NULL_KEY
                            ):
                                # A long string's bytes, not where they are.
                                part = _hash_combine(
                                    _hash_bytes(
                                        Span[UInt8, ImmutAnyOrigin](
                                            unsafe_ptr=Pointer[
                                                UInt8, ImmutAnyOrigin
                                            ](unsafe_from_address=Int(second)),
                                            length=length,
                                        )
                                    ),
                                    UInt64(length),
                                )
                            else:
                                part = _hash_combine(part, second)
                        to += widths[c]
                        hash = _mix(part) if j == 0 else _hash_combine(
                            hash, part
                        )
                    hashes[unsafe_offset=count] = hash
                    nulls[unsafe_offset=count] = null
                    source_at[unsafe_offset=count] = address
                    first_at[unsafe_offset=count] = row[unsafe_offset=2]
                    count += 1
                    if count == _BATCH:
                        self._merge_group_rows(
                            count, source.parts.states_at, sources, firsts
                        )
                        count = 0
        if count > 0:
            self._merge_group_rows(
                count, source.parts.states_at, sources, firsts
            )
        for held in source.held:
            self.held.append(held.copy())

    def _merge_group_rows(
        mut self,
        count: Int,
        source_states_at: Int,
        sources: List[Int],
        firsts: List[UInt64],
    ) raises:
        """A batch of source group rows, their keys and hashes in this
        sink's batch buffers: each into its group (made when new), the
        states merged and the earlier first row kept."""
        var word_ptrs = List[Int](capacity=len(self.addressed))
        for w in range(len(self.addressed)):
            word_ptrs.append(Int(self.word_rows[w].unsafe_ptr()))
        var group_at = self.groups.unsafe_ptr()
        # A new group's first row starts past any real row, so the earlier
        # of it and the source's wins.
        self._insert[False](
            count,
            1 << 50,
            Int(self.hashes.unsafe_ptr()),
            Int(self.nulls.unsafe_ptr()),
            word_ptrs,
            Int(group_at),
            0,
        )
        var into_states = 8 * self.parts.states_at
        var from_states = 8 * source_states_at
        for k in range(count):
            var group = group_at[unsafe_offset=k]
            combine(self.slots, group + into_states, sources[k] + from_states)
            var head = _words_at(group)
            if firsts[k] < head[unsafe_offset=2]:
                head[unsafe_offset=2] = firsts[k]
        self.sunk += count

    def _reinsert(mut self):
        """Every row back into a grown table (growing strategy only, so no
        group has a duplicate)."""
        var mask = self.capacity - 1
        var table = self.entries.unsafe_ptr()
        var stride = self.parts.stride
        for p in range(len(self.parts.blocks)):
            for b in range(len(self.parts.blocks[p])):
                var base = self.parts.blocks[p][b]
                for r in range(self.parts.rows_in(p, b)):
                    var at = base + 8 * r * stride
                    var hash = _words_at(at)[]
                    var slot = Int(hash) & mask
                    while table[unsafe_offset=slot] != 0:
                        slot = (slot + _stride(hash)) & mask
                    table[unsafe_offset=slot] = _entry(hash, at)


struct _SinkJob(Job):
    """Phase 1 over a frame: one worker's morsels (row ranges within
    chunks, from a shared cursor) into its own sink."""

    var keys: List[Series]
    var inputs: List[Series]
    var slots: List[AggregateFunction]
    var ranges: List[Int]
    var cursor: _Cursor
    var sink: AggregateSink

    def __init__(
        out self,
        keys: List[Series],
        inputs: List[Series],
        layout: AggregateLayout,
        ranges: List[Int],
        cursor: _Cursor,
        workers: Int,
    ):
        self.keys = keys.copy()
        self.inputs = inputs.copy()
        self.slots = layout.slots.copy()
        self.ranges = ranges.copy()
        self.cursor = cursor.copy()
        self.sink = AggregateSink(layout, workers)

    def run(mut self) raises:
        var morsels = len(self.ranges) // 2
        while True:
            var m = self.cursor.take()
            if m >= morsels:
                break
            var offset = self.ranges[2 * m]
            var length = self.ranges[2 * m + 1]
            var keys = List[Series](capacity=len(self.keys))
            for key in self.keys:
                keys.append(_one_chunk(key.slice(offset, length)))
            var inputs = List[Series](capacity=len(self.inputs))
            for s in range(len(self.inputs)):
                # A row count reads no input.
                var size = (
                    0 if s < len(self.slots)
                    and self.slots[s].kind == _K_LEN else length
                )
                inputs.append(_one_chunk(self.inputs[s].slice(offset, size)))
            self.sink.sink(keys, inputs, offset)

    def take_sink(deinit self) -> AggregateSink:
        return self.sink^


def _one_chunk(column: Series) -> Series:
    """A morsel's slice of a chunked column is one chunk, maybe still
    wrapped as chunked: that chunk."""
    if column.is_chunked() and column.n_chunks() == 1:
        return column.chunks()[0].copy()
    return column.copy()


def _next_first(parts: _Partitioned, p: Int, block: Int, row: Int) -> Int:
    """The first row of a worker's next row of partition p, at `row` of
    `block` or later, or Int.MAX when it has none."""
    var b = block
    var r = row
    while b < len(parts.blocks[p]):
        if r < parts.rows_in(p, b):
            var at = parts.blocks[p][b] + 8 * r * parts.stride
            return Int(_words_at(at)[unsafe_offset=2])
        b += 1
        r = 0
    return Int.MAX


@always_inline
def _merge_row(
    at: Int,
    table: Pointer[UInt64, MutAnyOrigin],
    mask: Int,
    mut final: _Groups,
    addressed: List[Bool],
    slots: List[AggregateFunction],
) raises -> Bool:
    """Row `at` into the finish's table: a new group (True), or its states
    folded into its group's, which keeps the earlier first row."""
    var row = _words_at(at)
    var hash = row[]
    var salt = (hash >> _SALT_SHIFT) << _SALT_SHIFT
    var slot = Int(hash) & mask
    var states_bytes = 8 * final.states_at
    while True:
        var entry = table[unsafe_offset=slot]
        if entry == 0:
            var f = final.copy_row(at)
            table[unsafe_offset=slot] = _entry(hash, f + 1)
            return True
        if (entry & ~_ID_MASK) == salt:
            var f = Int(entry & _ID_MASK) - 1
            var mine = final.at(f)
            if _words_at(mine)[] == hash and _same_rows(mine, at, addressed):
                combine(slots, mine + states_bytes, at + states_bytes)
                var first = row[unsafe_offset=2]
                if first < _words_at(mine)[unsafe_offset=2]:
                    _words_at(mine)[unsafe_offset=2] = first
                return False
        slot = (slot + _stride(hash)) & mask


def _by_partition(
    spans: List[Int], sizes: List[Int], bits: Int, mut into: _Groups
):
    """The rows of `spans` (each an address and its `sizes` rows) appended
    to `into` by the top `bits` bits of their hash: a counting sort, stable
    for rows of one partition."""
    var parts = 1 << bits
    var shift = UInt64(48 - bits)
    var mask = UInt64(parts - 1)
    var bytes = 8 * into.stride
    var starts = List[Int](length=parts + 1, fill=0)
    for k in range(len(spans)):
        for r in range(sizes[k]):
            var hash = _words_at(spans[k] + r * bytes)[]
            starts[Int((hash >> shift) & mask) + 1] += 1
    starts[0] = into.n
    for p in range(parts):
        starts[p + 1] += starts[p]
    var stride = into.stride
    var place = starts.unsafe_ptr()
    for k in range(len(spans)):
        var at = spans[k]
        for _ in range(sizes[k]):
            var row = _words_at(at)
            var p = Int((row[] >> shift) & mask)
            var to = _words_at(into.at(place[unsafe_offset=p]))
            for w in range(stride):
                to[unsafe_offset=w] = row[unsafe_offset=w]
            place[unsafe_offset=p] += 1
            at += bytes
    into.n = starts[parts]


def _capacity_for(rows: Int) -> Int:
    """Choose a power-of-two finalize-table capacity holding `rows` at a
    1.5 load factor, with a minimum capacity of 64."""
    var capacity = 64
    while 2 * capacity < 3 * max(rows, 1):
        capacity *= 2
    return capacity


struct GroupFilter(Copyable, Movable):
    """Which finished groups to keep, decided from the results of some of
    their reductions before the rest are read out: a filter over an
    aggregate's outputs (SQL's HAVING) run in the finish, so the groups it
    drops never have their keys or other results written. `slots` are the
    reductions it reads; `keep` gets those reductions' finished `Reducer`s
    over a job's groups (in `slots` order) with `context`, and returns the
    kept groups' indices in increasing order. It runs on the finishing
    workers at once, so it must only read what `context` points at."""

    var slots: List[Int]
    var context: Int
    var keep: def(Int, List[Reducer]) raises thin -> List[Int]

    def __init__(
        out self,
        var slots: List[Int],
        context: Int,
        keep: def(Int, List[Reducer]) raises thin -> List[Int],
    ):
        self.slots = slots^
        self.context = context
        self.keep = keep


struct _FinishJob(Job):
    """Phase 2: a range of partitions' rows from every worker, merged one
    run of partitions at a time in a table sized for the run, states
    folded into each group's row, into one chunk of groups."""

    var sinks: Int
    # The partitions where each run starts, and each run's rows.
    var run_starts: List[Int]
    var run_rows: List[Int]
    var bits: Int
    # The partition bits groups come out by: the most the sinks could
    # reach, the same on every run.
    var order_bits: Int
    var lo: Int
    var hi: Int
    var total: Int
    var templates: List[ReducerSpec]
    var slots: List[AggregateFunction]
    var key_templates: List[Series]
    var states: List[Reducer]
    var firsts: List[Int]
    var keys: List[Series]
    # The rows every top-rows state holds.
    var top_rows: List[Int]
    # The address of the sinks' `_HeldBuffers`.
    var held: Int
    var filter: Optional[GroupFilter]
    # The groups merged, before `filter` dropped any.
    var merged: Int

    def __init__(
        out self,
        sinks: Int,
        held: Int,
        lo: Int,
        hi: Int,
        total: Int,
        templates: List[ReducerSpec],
        slots: List[AggregateFunction],
        key_templates: List[Series],
        bits: Int,
        order_bits: Int,
        var run_starts: List[Int],
        var run_rows: List[Int],
        filter: Optional[GroupFilter],
    ):
        self.filter = filter.copy()
        self.merged = 0
        self.sinks = sinks
        self.run_starts = run_starts^
        self.run_rows = run_rows^
        self.held = held
        self.bits = bits
        self.order_bits = order_bits
        self.lo = lo
        self.hi = hi
        self.total = total
        self.templates = templates.copy()
        self.slots = slots.copy()
        self.key_templates = key_templates.copy()
        self.states = List[Reducer]()
        self.firsts = List[Int]()
        self.keys = List[Series]()
        self.top_rows = List[Int]()

    def run(mut self) raises:
        ref jobs = Pointer[List[AggregateSink], MutAnyOrigin](
            unsafe_from_address=self.sinks
        )[]
        ref addressed = jobs[0].addressed
        var final = _Groups(
            len(addressed),
            addressed.copy(),
            jobs[0].parts.stride - jobs[0].parts.states_at,
            self.total,
        )
        var stride = final.stride
        # Groups come out by partition, then by first row, so the output
        # order depends on the rows alone, not on which worker took which
        # morsel. A worker takes morsels in order and makes a group's row
        # at its first sight in a fill cycle, so each worker's rows of a
        # partition are in first-row order: merging them makes every group
        # at its earliest row, in that order, with no sort and no second
        # copy of the groups (both were a tenth of near-unique group-bys).
        var sinks = len(jobs)
        # One worker's rows alone, from a table that never abandoned its
        # entries, are its groups already, each once and in first-row order
        # within a partition: copied as they are, with no table (DuckDB's
        # `single_ht`, per run of partitions rather than per query).
        var only = -1
        for j in range(sinks):
            for p in range(self.lo, self.hi):
                if len(jobs[j].parts.blocks[p]) > 0:
                    only = j if only == -1 else -2
                    break
        var in_order = True
        # Whether the groups are in the output order already.
        var ordered = False
        if only >= 0 and not jobs[only].abandoned:
            trace_path("aggregate_table.single_sink_finish")
            ref parts = jobs[only].parts
            var spans = List[Int]()
            var sizes = List[Int]()
            for p in range(self.lo, self.hi):
                var last = -1
                for b in range(len(parts.blocks[p])):
                    var rows = parts.rows_in(p, b)
                    spans.append(parts.blocks[p][b])
                    sizes.append(rows)
                    var words = _words_at(parts.blocks[p][b])
                    for r in range(rows):
                        var first = Int(words[unsafe_offset=r * stride + 2])
                        in_order = in_order and first > last
                        last = first
            if in_order and self.bits < self.order_bits:
                # Into the output order as they are copied (see below).
                _by_partition(spans, sizes, self.order_bits, final)
                ordered = True
            else:
                for k in range(len(spans)):
                    _ = external_call["memcpy", Int](
                        final.at(final.n), spans[k], 8 * sizes[k] * stride
                    )
                    final.n += sizes[k]
        else:
            # Each run's table at DuckDB's finalize capacity (its rows at a
            # 1.5 load factor), in one buffer sized for the largest run.
            var largest = 0
            for r in range(len(self.run_rows)):
                largest = max(largest, self.run_rows[r])
            var entries = huge_list(_capacity_for(largest), UInt64(0))
            var table = entries.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
            var mask = 0
            var run = 0
            var block = List[Int](length=sinks, fill=0)
            var row_in = List[Int](length=sinks, fill=0)
            # Each worker's next row's first row, Int.MAX when it has none.
            var head = List[Int](length=sinks, fill=Int.MAX)
            for p in range(self.lo, self.hi):
                if run < len(self.run_starts) and p == self.run_starts[run]:
                    # A run's first partition: an empty table for its rows.
                    var capacity = _capacity_for(self.run_rows[run])
                    if run > 0:
                        for e in range(capacity):
                            table[unsafe_offset=e] = 0
                    mask = capacity - 1
                    run += 1
                for j in range(sinks):
                    block[j] = 0
                    row_in[j] = 0
                    head[j] = _next_first(jobs[j].parts, p, 0, 0)
                var last = -1
                while True:
                    # The worker whose next row comes first, and the first
                    # row next after it among the others.
                    var best = -1
                    var best_first = Int.MAX
                    var after = Int.MAX
                    for j in range(sinks):
                        var first = head[j]
                        if first < best_first:
                            after = best_first
                            best_first = first
                            best = j
                        elif first < after:
                            after = first
                    if best < 0:
                        break
                    ref parts = jobs[best].parts
                    # Its rows up to the others' next, block by block.
                    while block[best] < len(parts.blocks[p]):
                        var rows = parts.rows_in(p, block[best])
                        var base = parts.blocks[p][block[best]]
                        var stop = False
                        while row_in[best] < rows:
                            var at = base + 8 * row_in[best] * stride
                            var first = Int(_words_at(at)[unsafe_offset=2])
                            if first > after:
                                stop = True
                                break
                            if _merge_row(
                                at, table, mask, final, addressed, self.slots
                            ):
                                in_order = in_order and first > last
                                last = first
                            row_in[best] += 1
                        if stop:
                            break
                        block[best] += 1
                        row_in[best] = 0
                    head[best] = _next_first(
                        parts, p, block[best], row_in[best]
                    )
            _ = entries^
        if not in_order:
            # A worker's rows were not in first-row order: sort the groups.
            trace_path("aggregate_table.sorted_groups")
            # Each as one 128-bit key: partition and first row above, the
            # group below (first rows stay under 2^56).
            var shift = UInt64(48 - self.order_bits)
            var part_mask = UInt64((1 << self.order_bits) - 1)
            var keyed = List[UInt128](capacity=final.count())
            for g in range(final.count()):
                var part = (_words_at(final.at(g))[] >> shift) & part_mask
                var place = (UInt128(part) << 56) | UInt128(final.first(g))
                keyed.append((place << 64) | UInt128(g))
            sort(keyed)
            var sorted = _Groups(
                final.width(),
                final.addressed.copy(),
                final.stride - final.states_at,
                final.count(),
            )
            for key in keyed:
                _ = sorted.copy_row(
                    final.at(Int(UInt64(key & 0xFFFFFFFFFFFFFFFF)))
                )
            final = sorted^
        elif self.bits < self.order_bits and not ordered:
            # Fewer partitions than the output order's (the most the sinks
            # could have reached): each group to its place among those,
            # stably, so groups keep their first-row order within each and
            # every run returns the same order whatever partitions the
            # sinks reached.
            var into = _Groups(
                final.width(),
                final.addressed.copy(),
                final.stride - final.states_at,
                final.count(),
            )
            _by_partition([final.at(0)], [final.count()], self.order_bits, into)
            # The old groups stay alive until their rows are copied.
            swap(final, into)
            _ = into^
        self.merged = final.count()
        ref held = Pointer[_HeldBuffers, MutAnyOrigin](
            unsafe_from_address=self.held
        )[]
        if self.filter and final.count() > 0:
            # The reductions the filter reads, finished over every group,
            # then only the kept groups' rows, in order, for the scan.
            ref filter = self.filter.value()
            var results = List[Reducer](capacity=len(filter.slots))
            for s in filter.slots:
                var one = _Finalizer(
                    self.slots[s], self.templates[s], final.count(), held
                )
                for lo in range(0, final.count(), _BATCH):
                    one.scan(final, lo, min(final.count(), lo + _BATCH), held)
                results.append(one^.reducer())
            var kept = filter.keep(filter.context, results)
            if len(kept) < final.count():
                var into = _Groups(
                    final.width(),
                    final.addressed.copy(),
                    final.stride - final.states_at,
                    len(kept),
                )
                for g in kept:
                    _ = into.copy_row(final.at(g))
                swap(final, into)
                _ = into^
        # The groups are scanned out a vector of rows at a time (DuckDB's
        # `Scan`: each chunk of rows gathered into every output column, its
        # states finalized), so each group's row is read from memory once,
        # not once per reduction and key.
        var count = final.count()
        var outputs = List[_Finalizer](capacity=len(self.templates))
        for s in range(len(self.templates)):
            if self.slots[s].kind == _K_TOP_ROWS:
                var w0 = final.states_at + self.slots[s].offset
                var words = final.rows.unsafe_ptr()
                for g in range(count):
                    var w = g * final.stride + w0
                    for p in range(Int(words[unsafe_offset=w])):
                        self.top_rows.append(
                            Int(words[unsafe_offset=w + 2 + 2 * p])
                        )
            outputs.append(
                _Finalizer(self.slots[s], self.templates[s], count, held)
            )
        var key_columns = List[_KeyColumn](capacity=len(self.key_templates))
        var w = 0
        for c in range(len(self.key_templates)):
            key_columns.append(
                _KeyColumn(
                    self.key_templates[c],
                    w,
                    UInt64(1) << UInt64(c),
                    count,
                    held,
                )
            )
            w += _width(self.key_templates[c])
        self.firsts = List[Int](unsafe_uninit_length=count)
        var firsts = self.firsts.unsafe_ptr()
        for lo in range(0, count, _BATCH):
            var hi = min(count, lo + _BATCH)
            for s in range(len(outputs)):
                outputs[s].scan(final, lo, hi, held)
            for c in range(len(key_columns)):
                key_columns[c].scan(final, lo, hi, held)
            for g in range(lo, hi):
                firsts[unsafe_offset=g] = final.first(g)
        while len(outputs) > 0:
            self.states.append(outputs.pop(0).reducer())
        while len(key_columns) > 0:
            self.keys.append(key_columns.pop(0).column())


def aggregate_table(
    keys: List[Series],
    inputs: List[Series],
    templates: List[ReducerSpec],
    ranges: List[Int],
    workers: Int,
) raises -> AggregateResult:
    """Group `keys` and reduce each of `inputs` with its `templates` state,
    over the (offset, length) morsels `ranges` (within chunks) on `workers`
    threads. Returns per finish partition its groups' keys, first rows and
    states (unfinished), partitions in hash order. Every reduction's state
    must fit inline (`states_supported`)."""
    var layout = aggregate_layout(keys, inputs, templates)
    var cursor = _Cursor.new()
    var jobs = List[_SinkJob](capacity=workers)
    for _ in range(max(1, workers)):
        jobs.append(_SinkJob(keys, inputs, layout, ranges, cursor, workers))
    try:
        run_jobs(jobs)
    except e:
        cursor.free()
        raise e^
    cursor.free()
    var sinks = List[AggregateSink](capacity=len(jobs))
    while len(jobs) > 0:
        var job = jobs.pop()
        sinks.append(job^.take_sink())
    return aggregate_finish(sinks, layout)


struct _CombineJob(Job):
    """One sink's rows moved to the shared partition bits."""

    var sinks: Int
    var index: Int
    var bits: Int

    def __init__(out self, sinks: Int, index: Int, bits: Int):
        self.sinks = sinks
        self.index = index
        self.bits = bits

    def run(mut self) raises:
        Pointer[List[AggregateSink], MutAnyOrigin](
            unsafe_from_address=self.sinks
        )[][self.index].repartition(self.bits)


def _valid_rows(column: Series) raises -> List[Bool]:
    """Whether each row of an unchunked column of any storage is valid."""
    var n = len(column)
    var valid = List[Bool](capacity=n)
    if column._data.isa[StringColumn]():
        ref strings = column._data[StringColumn]
        for i in range(n):
            valid.append(strings._valid(i))
    elif column._data.isa[BoolColumn]():
        ref bools = column._data[BoolColumn]
        for i in range(n):
            valid.append(bools._valid(i))
    elif column._data.isa[Column[Int128]]():
        ref wide = column._data[Column[Int128]]
        for i in range(n):
            valid.append(wide._valid(i))
    else:
        comptime for k in range(len(NUMERIC_DTYPES)):
            comptime D = NUMERIC_DTYPES[k]
            if column._data.isa[Column[Scalar[D]]]():
                ref typed = column._data[Column[Scalar[D]]]
                for i in range(n):
                    valid.append(typed._valid(i))
    if len(valid) != n:
        raise Error("aggregate table: unsupported distinct value storage")
    return valid^


struct _DistinctJob(Job):
    """One partition of a distinct table's finished (keys, value) groups
    sunk into a new sink of the main table, each adding one to its group's
    distinct count (DuckDB's `HashAggregateDistinctFinalizeTask`)."""

    var keys: List[Series]
    var value: Series
    var slot: Int
    var nulls: Bool
    var layout: AggregateLayout
    var workers: Int
    var sinks: List[AggregateSink]

    def __init__(
        out self,
        var keys: List[Series],
        value: Series,
        slot: Int,
        nulls: Bool,
        layout: AggregateLayout,
        workers: Int,
    ):
        self.keys = keys^
        self.value = value.copy()
        self.slot = slot
        self.nulls = nulls
        self.layout = layout.copy()
        self.workers = workers
        self.sinks = List[AggregateSink]()

    def run(mut self) raises:
        var sink = AggregateSink(self.layout, self.workers, distinct=False)
        sink.only = self.slot
        var n = len(self.value)
        var valid = List[Bool]()
        if not self.nulls:
            # A null value is no distinct value: its group adds nothing.
            valid = _valid_rows(self.value)
        var step = 1 << 16
        for lo in range(0, n, step):
            var length = min(step, n - lo)
            var part = List[Series](capacity=len(self.keys))
            for key in self.keys:
                part.append(_one_chunk(key.slice(lo, length)))
            var keep = List[Bool]()
            if not self.nulls:
                keep = List[Bool](capacity=length)
                for k in range(length):
                    keep.append(valid[lo + k])
            # First rows past every input row's: a group's first row stays
            # the one its own rows gave it.
            sink.sink(part, List[Series](), (1 << 50) + lo, keep)
        self.sinks.append(sink^)


def _finalize_distinct(
    mut sinks: List[AggregateSink], layout: AggregateLayout
) raises:
    """DuckDB's `FinalizeDistinct`: each distinct count's tables (one per
    worker) finished into their (keys, value) groups, which are sunk into
    new sinks of the main table, one per finished partition, on their own
    threads; those sinks join `sinks`, and the main finish merges their
    counts into the groups."""
    # The workers' sinks; the new sinks join the list after them.
    var workers = len(sinks)
    for d in range(len(layout.distinct)):
        var slot = layout.distinct[d]
        var tables = List[AggregateSink](capacity=workers)
        for j in range(workers):
            tables.append(sinks[j].distinct.take(d))
        var pairs = aggregate_finish(tables, layout.distinct_layout(d))
        var nulls = layout.templates[slot].integer == 0
        var keys = len(layout.key_templates)
        var jobs = List[_DistinctJob]()
        for p in range(len(pairs.keys)):
            if len(pairs.firsts[p]) == 0:
                continue
            var columns = List[Series](capacity=keys)
            for c in range(keys):
                columns.append(pairs.keys[p][c].copy())
            jobs.append(
                _DistinctJob(
                    columns^,
                    pairs.keys[p][keys],
                    slot,
                    nulls,
                    layout,
                    workers,
                )
            )
        run_jobs(jobs)
        _ = tables^
        for j in range(len(jobs)):
            while len(jobs[j].sinks) > 0:
                sinks.append(jobs[j].sinks.pop())


def aggregate_finish(
    mut sinks: List[AggregateSink],
    layout: AggregateLayout,
    filter: Optional[GroupFilter] = None,
) raises -> AggregateResult:
    """Phase 2 over every worker's sink: runs of consecutive partitions
    holding about half a 256 KiB block of rows each. This bounds each
    working set so its table and rows can stay in cache while runs merge
    on separate threads. With `filter`, only the groups it keeps come
    out (`GroupFilter`); the result's `merged` counts them all."""
    var result = AggregateResult()
    if len(sinks) == 0:
        return result^
    if len(layout.distinct) > 0:
        _finalize_distinct(sinks, layout)
    # Every sink's rows at the shared bits first, each on its own thread
    # (DuckDB's `Combine`, which repartitions to the global bits).
    var bits = Int(layout.radix[].bits.load())
    var combines = List[_CombineJob]()
    for j in range(len(sinks)):
        if sinks[j].parts.bits < bits:
            combines.append(_CombineJob(Int(Pointer(to=sinks)), j, bits))
    if len(combines) > 0:
        run_jobs(combines)
    var stride = sinks[0].parts.stride
    var columns = List[List[Series]](capacity=len(sinks))
    for j in range(len(sinks)):
        columns.append(sinks[j].held.copy())
    var held = _HeldBuffers(columns)
    _ = columns^
    # Runs: consecutive partitions holding about half a 256 KiB block of
    # rows, each merged in its own cache-sized table. Jobs: consecutive
    # runs holding about an equal share of the rows per worker, and at
    # least a run's, each one chunk of the result, as Polars finishes a partition per thread (a
    # chunk per run made later steps pay their per-chunk cost hundreds of
    # times).
    var per_run = max(1, _PARTITION_BYTES // (8 * stride))
    var parts = 1 << bits
    var rows = List[Int](length=parts, fill=0)
    # What finishing a partition reads, in bytes: its group rows, and the
    # values of every quantile state in them, which a job flattens and
    # selects from. Jobs are cut by this, so a few groups of many values
    # (a median over ten groups of a million values) still spread over
    # the workers.
    var weight = List[Int](length=parts, fill=0)
    var counted = List[Int]()
    for slot in layout.slots:
        if slot.kind == _K_VALUES:
            counted.append(sinks[0].parts.states_at + slot.offset + 2)
    var total = 0
    for p in range(parts):
        for j in range(len(sinks)):
            for b in range(len(sinks[j].parts.blocks[p])):
                var held = sinks[j].parts.rows_in(p, b)
                rows[p] += held
                if len(counted) > 0:
                    var at = sinks[j].parts.blocks[p][b]
                    for _ in range(held):
                        for w in counted:
                            weight[p] += 8 * Int(_words_at(at)[unsafe_offset=w])
                        at += 8 * stride
        weight[p] += 8 * stride * rows[p]
        total += weight[p]
    var per_job = max(
        _PARTITION_BYTES, -(-total // max(1, configured_workers()))
    )
    var finishes = List[_FinishJob]()
    var lo = 0
    var run = 0
    var job = 0
    var job_rows = 0
    var starts = List[Int]()
    var run_rows = List[Int]()
    var open = False
    for p in range(parts):
        if not open:
            starts.append(p)
            open = True
        run += rows[p]
        job += weight[p]
        job_rows += rows[p]
        if run >= per_run or job >= per_job or p == parts - 1:
            run_rows.append(run)
            run = 0
            open = False
        if job >= per_job or p == parts - 1:
            if job_rows > 0:
                finishes.append(
                    _FinishJob(
                        Int(Pointer(to=sinks)),
                        Int(Pointer(to=held)),
                        lo,
                        p + 1,
                        job_rows,
                        layout.templates,
                        layout.slots,
                        layout.key_templates,
                        bits,
                        sinks[0].max_bits,
                        starts^,
                        run_rows^,
                        filter,
                    )
                )
            starts = List[Int]()
            run_rows = List[Int]()
            lo = p + 1
            job = 0
            job_rows = 0
    run_jobs(finishes)
    _ = held^
    for p in range(len(finishes)):
        var firsts = List[Int]()
        var states = List[Reducer]()
        var keys_out = List[Series]()
        result.top_rows.extend(Span(finishes[p].top_rows))
        result.merged += finishes[p].merged
        swap(firsts, finishes[p].firsts)
        swap(states, finishes[p].states)
        swap(keys_out, finishes[p].keys)
        result.firsts.append(firsts^)
        result.states.append(states^)
        result.keys.append(keys_out^)
    return result^


struct AggregateResult(Movable):
    """Per finish partition, in hash order: its groups' first rows and
    their states (one per reduction, unfinished)."""

    var firsts: List[List[Int]]
    var states: List[List[Reducer]]
    # Per partition, its groups' key columns.
    var keys: List[List[Series]]
    # The rows held by top-rows states, every partition's.
    var top_rows: List[Int]
    # The groups merged, including any a `GroupFilter` dropped.
    var merged: Int

    def __init__(out self):
        self.merged = 0
        self.firsts = List[List[Int]]()
        self.states = List[List[Reducer]]()
        self.keys = List[List[Series]]()
        self.top_rows = List[Int]()


def aggregate_top_rows(
    keys: List[Series],
    value: Series,
    k: Int,
    descending: Bool,
    ranges: List[Int],
    workers: Int,
) raises -> List[Int]:
    """The rows of each group of `keys` whose `value` ranks among its k
    first by ordinal rank (ties: the earlier row), over the (offset,
    length) morsels `ranges` on `workers` threads, in no order: each
    group's row in the table holds its k best (value, row) pairs, merged
    across workers at the finish. `k` is at most `TOP_ROWS_INLINE`; nulls
    are never kept."""
    var count_spec = ReducerSpec(
        LEN, DataType.INT64, 0, 0, 0.0, String(), DataType.INT64
    )
    var layout = aggregate_layout(keys, [value.copy()], [count_spec^])
    var dtype = layout.slots[0].dtype
    layout.slots[0] = AggregateFunction(
        _K_TOP_ROWS, 0, value.dtype().is_float(), dtype, k, descending
    )
    layout.state_words = layout.slots[0].state_size()
    var cursor = _Cursor.new()
    var jobs = List[_SinkJob](capacity=workers)
    for _ in range(max(1, workers)):
        jobs.append(
            _SinkJob(keys, [value.copy()], layout, ranges, cursor, workers)
        )
    try:
        run_jobs(jobs)
    except e:
        cursor.free()
        raise e^
    cursor.free()
    var sinks = List[AggregateSink](capacity=len(jobs))
    while len(jobs) > 0:
        var job = jobs.pop()
        sinks.append(job^.take_sink())
    var result = aggregate_finish(sinks, layout)
    var rows = List[Int]()
    swap(rows, result.top_rows)
    return rows^


def table_row_keys(keys: List[Series]) raises -> RowKeys:
    """Dense ids of the rows' key tuples in first-occurrence order, with
    nulls equal (`encode_rows` with `nulls_equal`), by the aggregate table's
    hashing and in-place key comparison on one thread: no row-encoded copy
    of the keys. Every key must be `key_supported`."""
    var n = len(keys[0])
    var flat = List[Series](capacity=len(keys))
    for key in keys:
        flat.append(key.rechunk() if key.is_chunked() else key.copy())
    var layout = aggregate_layout(flat, List[Series](), List[ReducerSpec]())
    layout.state_words = 1
    var sink = AggregateSink(layout, 1)
    sink.numbering = True
    sink.row_ids = List[Int](length=n, fill=0)
    var step = 1 << 16
    for lo in range(0, n, step):
        var length = min(step, n - lo)
        var part = List[Series](capacity=len(flat))
        for key in flat:
            part.append(_one_chunk(key.slice(lo, length)))
        sink.sink(part, List[Series](), lo)
    var ids = List[Int]()
    var representatives = List[Int]()
    swap(ids, sink.row_ids)
    swap(representatives, sink.representatives)
    _ = sink^
    return RowKeys(ids^, representatives^)
