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
   copied directly. Without maintain_order, groups come out in whatever
   order they merge, as DuckDB's GROUP BY and Polars' group_by give no
   order. With it, groups are emitted by partition at the greatest bit
   depth the workers could reach, then by first row, which the caller
   merges into first-occurrence order.

A string longer than twelve bytes is compared through the address of its
bytes in the input, which outlives the aggregation; the input is never
copied.
"""
from std.atomic import Atomic
from std.bit import count_trailing_zeros
from std.collections import Optional
from std.math import log
from std.ffi import external_call
from std.memory import ArcPointer, Pointer, bitcast
from std.sys import size_of

from dataframe.aggregation.reducer import Reducer, float_key
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
    _K_COUNT,
    _K_DECIMAL_SUM,
    _K_DISTINCT,
    _K_LEN,
    _K_NULLS,
    _K_ORDERED,
    _K_STATE,
    _K_TOP_ROWS,
    _K_VALUES,
    _OrderedBuffer,
    _VALUE_NONE,
    _ValueArena,
    _storage_kind,
    _update_states,
    _numeric_part,
    _pair_kind,
    _second_input,
    _slot_kind,
    _words_at,
    combine,
    reducer_input,
    reaggregable,
    reduction_supported,
    states_supported,
    table_kind,
    table_reduction,
    table_states_supported,
)
from dataframe.aggregation.perfect_hash import (
    PerfectHash,
    group_locations,
    key_word,
    perfect_hash,
)
from dataframe.storage.boolean import BoolColumn
from dataframe.storage.column import Column
from dataframe.dtype import CategoricalDictionary, DataType, NUMERIC_DTYPES
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
    _mix,
    _short_hash,
)
from dataframe.series import Series
from dataframe.trace import trace_path
from dataframe.storage.string import StringColumn
from dataframe.vector.types import (
    PHYSICAL_BOOL,
    PHYSICAL_INT128,
    PHYSICAL_VARCHAR,
    STRING_INLINE_BYTES,
    StringT,
    UnifiedVectorFormat,
    physical_of,
    zero_selection,
)

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
    # The perfect hash table's shape when the table is one (none by
    # default): every worker's table then holds a row for every key
    # combination, found from the keys' bits (`perfect_hash.mojo`).
    var perfect: PerfectHash

    def use_perfect_hash(mut self, keys: List[Series]) raises:
        """Make this a perfect hash table when DuckDB's planner would
        (`CanUsePerfectHashAggregate`) for the grouping keys `keys`, the
        layout's own key columns carrying their statistics: never with a
        distinct count, which perfect hash aggregates do not take."""
        if len(self.distinct) > 0 or len(keys) != len(self.key_templates):
            return
        for key in self.key_templates:
            if _width(key) != 1:
                return
        self.perfect = perfect_hash(keys)
        if self.perfect.used():
            trace_path("aggregate_table.perfect_hash")

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
            PerfectHash.none(),
        )


def aggregate_layout(
    keys: List[Series], inputs: List[Series], templates: List[ReducerSpec]
) raises -> AggregateLayout:
    """The layout of a table grouping columns like `keys` and reducing
    columns like `inputs` (only their types are read). Every reduction
    has a state here, inline or by address (`table_states_supported`)."""
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
        if templates[s].op == CORR or templates[s].op == COV:
            second = _second_input(templates, s)
        var kind = table_kind(templates, inputs, s)
        if kind < 0:
            raise Error("aggregate table: a reduction with no state here")
        if kind == _K_STATE:
            trace_path("aggregate_table.state_by_address")
        elif kind == _K_ORDERED:
            trace_path("aggregate_table.ordered_state")
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
            var part = (
                inputs[s]
                .chunks()[0]
                .copy() if inputs[s]
                .is_chunked() else inputs[s]
                .copy()
            )
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
        PerfectHash.none(),
    )


@fieldwise_init
struct KeyVector(Copyable, ImplicitlyCopyable, Movable):
    """A group key column of a data chunk as the table reads it: its
    `UnifiedVectorFormat` (DuckDB's group chunk vectors, which
    `GroupedAggregateHashTable::AddChunk` reads through their unified
    format), and how its values become key words.

    - `codes`: the address of one dictionary code a row, for a string key
      whose source stored it dictionary-encoded (the format then gives
      only each row's validity); its dictionary is the
      `CategoricalDictionary` at `dictionary`. 0 for none.
    - `map`: the address of `map_length` UInt32 codes a categorical key's
      codes are replaced by (equal labels group together); 0 for none.
    - `mask`: the bits of an integer key's value kept (a grouping set
      groups by some fields of a packed key); every bit by default.
    """

    var format: UnifiedVectorFormat
    var codes: Int
    var dictionary: Int
    var map: Int
    var map_length: Int
    var mask: UInt64

    @staticmethod
    def of(format: UnifiedVectorFormat) -> KeyVector:
        """A key read as its format gives it."""
        return KeyVector(format, 0, 0, 0, 0, UInt64.MAX)

    def width(self) -> Int:
        """Key words a row takes: two for a string or a 128-bit decimal."""
        if (
            self.format.physical == PHYSICAL_VARCHAR
            or self.format.physical == PHYSICAL_INT128
        ):
            return 2
        return 1


def _vector_key(
    key: KeyVector,
    count: Int,
    hashes_at: Int,
    first_at: Int,
    second_at: Int,
    nulls_at: Int,
    bit: UInt64,
    leading: Bool,
) raises:
    """A chunk's `count` rows of one key vector: each row's hash (combined
    into the hashes of the keys before unless `leading`, as
    VectorOperations::CombineHash) and key words, a null row setting `bit`
    in its null word. A row's hash is the one the partitioner's
    `_hash_column` gives the same value in a column."""
    ref format = key.format
    var physical = format.physical
    if physical == PHYSICAL_VARCHAR:
        if format.sel.is_set():
            _string_key[True](
                format,
                count,
                hashes_at,
                first_at,
                second_at,
                nulls_at,
                bit,
                leading,
            )
        else:
            _string_key[False](
                format,
                count,
                hashes_at,
                first_at,
                second_at,
                nulls_at,
                bit,
                leading,
            )
        return
    var hashes = Pointer[UInt64, MutAnyOrigin](unsafe_from_address=hashes_at)
    var first = Pointer[UInt64, MutAnyOrigin](unsafe_from_address=first_at)
    var second = Pointer[UInt64, MutAnyOrigin](unsafe_from_address=second_at)
    var nulls = Pointer[UInt64, MutAnyOrigin](unsafe_from_address=nulls_at)
    var sel = format.sel
    var validity = format.validity
    var has_nulls = not validity.all_valid()
    if physical == PHYSICAL_BOOL:
        var data = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=format.data)
        for k in range(count):
            var i = sel.get_index(k)
            var hash_key: UInt64
            if has_nulls and not validity.row_is_valid_unsafe(i):
                hash_key = _NULL_KEY
                first[unsafe_offset=k] = 0
                nulls[unsafe_offset=k] |= bit
            else:
                hash_key = UInt64(1) if data[unsafe_offset=i] != 0 else UInt64(
                    2
                )
                first[unsafe_offset=k] = hash_key
            hashes[unsafe_offset=k] = _mix(
                hash_key
            ) if leading else _hash_combine(hashes[unsafe_offset=k], hash_key)
        return
    if physical == PHYSICAL_INT128:
        var data = format.values[DType.int128]()
        for k in range(count):
            var i = sel.get_index(k)
            var hash_key: UInt64
            if has_nulls and not validity.row_is_valid_unsafe(i):
                hash_key = _NULL_KEY
                first[unsafe_offset=k] = 0
                second[unsafe_offset=k] = 0
                nulls[unsafe_offset=k] |= bit
            else:
                var value = bitcast[DType.uint128](data[unsafe_offset=i])
                first[unsafe_offset=k] = UInt64(value)
                second[unsafe_offset=k] = UInt64(value >> 64)
                hash_key = UInt64(value) ^ UInt64(value >> 64)
            hashes[unsafe_offset=k] = _mix(
                hash_key
            ) if leading else _hash_combine(hashes[unsafe_offset=k], hash_key)
        return
    comptime for d in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[d]
        comptime if D != DType.int128:
            if physical == physical_of[D]():
                if sel.is_set():
                    if has_nulls:
                        _fixed_key[D, True, True](
                            key,
                            count,
                            hashes_at,
                            first_at,
                            nulls_at,
                            bit,
                            leading,
                        )
                    else:
                        _fixed_key[D, True, False](
                            key,
                            count,
                            hashes_at,
                            first_at,
                            nulls_at,
                            bit,
                            leading,
                        )
                elif has_nulls:
                    _fixed_key[D, False, True](
                        key, count, hashes_at, first_at, nulls_at, bit, leading
                    )
                else:
                    _fixed_key[D, False, False](
                        key, count, hashes_at, first_at, nulls_at, bit, leading
                    )
                return
    raise Error("aggregate table: unsupported key vector")


def _fixed_key[
    D: DType, selected: Bool, nullable: Bool
](
    key: KeyVector,
    count: Int,
    hashes_at: Int,
    first_at: Int,
    nulls_at: Int,
    bit: UInt64,
    leading: Bool,
) raises:
    """`_vector_key` over fixed-width values of `D`: the value's word (a
    float's canonical key), through a selection when `selected` and a
    validity mask when `nullable`."""
    var hashes = Pointer[UInt64, MutAnyOrigin](unsafe_from_address=hashes_at)
    var first = Pointer[UInt64, MutAnyOrigin](unsafe_from_address=first_at)
    var nulls = Pointer[UInt64, MutAnyOrigin](unsafe_from_address=nulls_at)
    var data = key.format.values[D]()
    var sel = key.format.sel
    var validity = key.format.validity
    var mask = key.mask
    var map = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=key.map)
    var mapped = key.map != 0
    for k in range(count):
        var i: Int
        comptime if selected:
            i = sel.get_index_unsafe(k)
        else:
            i = k
        var hash_key: UInt64
        var valid = True
        comptime if nullable:
            valid = validity.row_is_valid_unsafe(i)
        if not valid:
            hash_key = _NULL_KEY
            first[unsafe_offset=k] = 0
            nulls[unsafe_offset=k] |= bit
        else:
            var value = data[unsafe_offset=i]
            comptime if D.is_floating_point():
                hash_key = float_key(Float64(value))
            elif D.is_unsigned():
                hash_key = UInt64(value)
                comptime if D == DType.uint32:
                    if mapped:
                        if Int(value) >= key.map_length:
                            raise Error(
                                "Categorical code exceeds its dictionary"
                            )
                        hash_key = UInt64(map[unsafe_offset=Int(value)])
                hash_key &= mask
            else:
                hash_key = bitcast[DType.uint64](Int64(value)) & mask
            first[unsafe_offset=k] = hash_key
        hashes[unsafe_offset=k] = _mix(hash_key) if leading else _hash_combine(
            hashes[unsafe_offset=k], hash_key
        )


def _string_key[
    selected: Bool
](
    format: UnifiedVectorFormat,
    count: Int,
    hashes_at: Int,
    first_at: Int,
    second_at: Int,
    nulls_at: Int,
    bit: UInt64,
    leading: Bool,
):
    """`_vector_key` over string_t values. A string_t is already the two
    key words (its length and first four bytes, then its next eight bytes
    or the address of all of them); the inline bytes past the length are
    masked off."""
    var hashes = Pointer[UInt64, MutAnyOrigin](unsafe_from_address=hashes_at)
    var first = Pointer[UInt64, MutAnyOrigin](unsafe_from_address=first_at)
    var second = Pointer[UInt64, MutAnyOrigin](unsafe_from_address=second_at)
    var nulls = Pointer[UInt64, MutAnyOrigin](unsafe_from_address=nulls_at)
    var strings = Pointer[StringT, MutAnyOrigin](
        unsafe_from_address=format.data
    )
    var sel = format.sel
    var validity = format.validity
    var has_nulls = not validity.all_valid()
    for k in range(count):
        var i: Int
        comptime if selected:
            i = sel.get_index_unsafe(k)
        else:
            i = k
        var hash_key: UInt64
        if has_nulls and not validity.row_is_valid_unsafe(i):
            hash_key = _NULL_KEY
            first[unsafe_offset=k] = 0
            second[unsafe_offset=k] = 0
            nulls[unsafe_offset=k] |= bit
        else:
            var s = strings[unsafe_offset=i]
            var length = Int(s.length)
            if length <= STRING_INLINE_BYTES:
                var prefix = UInt64(s.prefix)
                if length < 4:
                    prefix &= (UInt64(1) << UInt64(8 * length)) - 1
                var keep = 8 * max(length - 4, 0)
                var rest = s.rest & (
                    (UInt64(1) << UInt64(keep)) - 1 if keep < 64 else UInt64.MAX
                )
                first[unsafe_offset=k] = UInt64(length) | (prefix << 32)
                second[unsafe_offset=k] = rest
                if length <= 8:
                    hash_key = _short_hash(prefix | (rest << 32), length)
                else:
                    hash_key = _hash_bytes(
                        Span[UInt8, ImmutAnyOrigin](
                            unsafe_ptr=Pointer[UInt8, ImmutAnyOrigin](
                                unsafe_from_address=format.data + 16 * i + 4
                            ),
                            length=length,
                        )
                    )
            else:
                first[unsafe_offset=k] = UInt64(length) | (
                    UInt64(s.prefix) << 32
                )
                second[unsafe_offset=k] = s.rest
                hash_key = _hash_bytes(
                    Span[UInt8, ImmutAnyOrigin](
                        unsafe_ptr=Pointer[UInt8, ImmutAnyOrigin](
                            unsafe_from_address=Int(s.rest)
                        ),
                        length=length,
                    )
                )
        hashes[unsafe_offset=k] = _mix(hash_key) if leading else _hash_combine(
            hashes[unsafe_offset=k], hash_key
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

    def __init__(out self):
        self.identity = 0
        self.offsets = 0
        self.bytes = 0
        self.done = List[Bool]()
        self.keys = List[UInt64]()
        self.first = List[UInt64]()
        self.second = List[UInt64]()

    def use(mut self, dictionary: Int):
        """Ready for codes into the `CategoricalDictionary` at address
        `dictionary`: its entries, reset when it is a new one."""
        if dictionary == self.identity:
            return
        ref entries = Pointer[CategoricalDictionary, ImmutAnyOrigin](
            unsafe_from_address=dictionary
        )[]
        self.identity = dictionary
        self.offsets = Int(entries.offsets.unsafe_ptr())
        self.bytes = Int(entries.bytes.unsafe_ptr())
        var n = len(entries)
        self.done = List[Bool](length=n, fill=False)
        self.keys = List[UInt64](length=n, fill=0)
        self.first = List[UInt64](length=n, fill=0)
        self.second = List[UInt64](length=n, fill=0)

    @no_inline
    def _entry(mut self, code: Int):
        """Entry `code`'s hash key and words: as `_vector_key` gives
        them for the same string."""
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

    def write_vector(
        mut self,
        key: KeyVector,
        count: Int,
        hashes_at: Int,
        first_word: Int,
        second_word: Int,
        nulls_at: Int,
        bit: UInt64,
        leading: Bool,
    ):
        """A chunk's `count` rows of a coded string key (`key.codes`, one
        code a row; its format gives each row's validity): hashes and key
        words through their codes, as `_vector_key` gives them for the
        same strings."""
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
        var codes = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=key.codes)
        var sel = key.format.sel
        var validity = key.format.validity
        var has_nulls = not validity.all_valid()
        for k in range(count):
            var hash_key: UInt64
            if has_nulls and not validity.row_is_valid_unsafe(sel.get_index(k)):
                hash_key = _NULL_KEY
                first[unsafe_offset=k] = 0
                second[unsafe_offset=k] = 0
                nulls[unsafe_offset=k] |= bit
            else:
                var code = Int(codes[unsafe_offset=k])
                if not self.done.unsafe_ptr()[unsafe_offset=code]:
                    self._entry(code)
                hash_key = self.keys.unsafe_ptr()[unsafe_offset=code]
                first[unsafe_offset=k] = self.first.unsafe_ptr()[
                    unsafe_offset=code
                ]
                second[unsafe_offset=k] = self.second.unsafe_ptr()[
                    unsafe_offset=code
                ]
            hashes[unsafe_offset=k] = _mix(
                hash_key
            ) if leading else _hash_combine(hashes[unsafe_offset=k], hash_key)

    def write_entries(
        mut self,
        entries: Int,
        count: Int,
        hashes_at: Int,
        first_word: Int,
        second_word: Int,
        nulls_at: Int,
    ):
        """The sole key's hashes and words for `count` dictionary entries
        (`entries`, one index a group; the dictionary's size is the null
        key): DuckDB hashing the sliced dictionary's unique values."""
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
        var at = Pointer[Int, MutAnyOrigin](unsafe_from_address=entries)
        var size = len(self.keys)
        for k in range(count):
            var code = at[unsafe_offset=k]
            var hash_key: UInt64
            if code == size:
                hash_key = _NULL_KEY
                first[unsafe_offset=k] = 0
                second[unsafe_offset=k] = 0
                nulls[unsafe_offset=k] = 1
            else:
                if not self.done.unsafe_ptr()[unsafe_offset=code]:
                    self._entry(code)
                hash_key = self.keys.unsafe_ptr()[unsafe_offset=code]
                first[unsafe_offset=k] = self.first.unsafe_ptr()[
                    unsafe_offset=code
                ]
                second[unsafe_offset=k] = self.second.unsafe_ptr()[
                    unsafe_offset=code
                ]
                nulls[unsafe_offset=k] = 0
            hashes[unsafe_offset=k] = _mix(hash_key)


struct _DictionaryGroups(Movable):
    """DuckDB's `AggregateDictionaryState`: for the dictionary whose
    entries a sole coded key's codes index (`identity`, its address, as
    DuckDB's dictionary id), each entry's group row once it has been
    looked up (one past the entries is the null key), and how many are.
    Rows whose entries were looked up before take their group from here
    and are neither hashed nor probed. Reset when the table's entries are
    cleared or its rows move (DuckDB resets the id in `ClearPointerTable`'s
    callers)."""

    var identity: Int
    var found: List[Bool]
    var addresses: List[Int]
    var resolved: Int
    # A chunk's rows' entries, and its entries not found before with the
    # source row of each one's first row.
    var row_entries: List[Int]
    var entries: List[Int]
    var rows: List[Int]

    def __init__(out self):
        self.identity = 0
        self.found = List[Bool]()
        self.addresses = List[Int]()
        self.resolved = 0
        self.row_entries = List[Int](length=_BATCH, fill=0)
        self.entries = List[Int](length=_BATCH, fill=0)
        self.rows = List[Int](length=_BATCH, fill=0)

    def reset(mut self):
        self.identity = 0

    def use(mut self, dictionary: Int, size: Int):
        """Ready for codes into the dictionary at address `dictionary` of
        `size` entries: nothing found yet when it is a new one."""
        if dictionary == self.identity:
            return
        self.identity = dictionary
        self.found = List[Bool](length=size + 1, fill=False)
        if len(self.addresses) < size + 1:
            self.addresses = List[Int](length=size + 1, fill=0)
        self.resolved = 0


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
    # codes; and a sole coded key's groups by entry.
    var dictionaries: List[_DictionaryKeys]
    var dictionary_groups: _DictionaryGroups
    # Every morsel's string key columns, held until the finish: a group's
    # long string key is the address of its bytes there, as DuckDB's rows
    # point into their collection's heap (a morsel's column holds its
    # buffers by reference, so nothing is copied). The finished key
    # columns are views into the same buffers (`_HeldBuffers`).
    var held: List[Series]
    # A group row's worth of words the state updates may write for rows
    # that join no group: read by nothing.
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
    # Whether this sink's rows of each partition are in first-row order,
    # as a worker taking morsels in order makes them; a sink of distinct
    # groups (`count_group_rows`) makes them in the distinct table's
    # order instead.
    var in_order: Bool
    # The reductions whose states the rows hold by address, and per
    # reduction the owner of those states: a `Reducer` of every group's
    # state (`_K_STATE`) or a buffer of input rows (`_K_ORDERED`). Each
    # list is made once, at its full length, so an owner's address is
    # fixed while the sink moves.
    var by_address: List[Int]
    var owned: List[Reducer]
    var buffered: List[_OrderedBuffer]
    # A perfect hash table (`AggregateLayout.perfect`): its shape, and a
    # row of `parts.stride` words for every group slot (DuckDB's `data`),
    # all empty until a row finds it; word 1 of a slot's row is then its
    # `group_is_set` flag and word 2 its first row. None for a hashed
    # table.
    var perfect: PerfectHash
    var perfect_rows: List[UInt64]

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
        # Made at the first probe, as DuckDB makes a local table at its
        # first `Sink` (`if (!lstate.ht)`): a worker that sinks nothing
        # here (every reduction a distinct count) never clears one.
        self.entries = List[UInt64]()
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
        self.dictionary_groups = _DictionaryGroups()
        self.held = List[Series]()
        self.discard = List[UInt64](length=self.parts.stride + 8, fill=0)
        self.values = _ValueArena()
        self.scratch = List[UInt64](length=3 * _BATCH, fill=0)
        self.distinct = _DistinctSinks()
        self.distinct_slots = List[Int]()
        self.only = -1
        self.in_order = True
        self.by_address = List[Int]()
        self.owned = List[Reducer](capacity=len(layout.slots))
        self.buffered = List[_OrderedBuffer](capacity=len(layout.slots))
        for s in range(len(layout.slots)):
            if layout.slots[s].by_address():
                self.by_address.append(s)
            self.owned.append(layout.templates[s].make(0))
            self.buffered.append(_OrderedBuffer())
        self.perfect = layout.perfect.copy()
        self.perfect_rows = List[UInt64]()
        if self.perfect.used():
            self.perfect_rows = List[UInt64](
                length=self.perfect.total_groups() * self.parts.stride, fill=0
            )
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

    @always_inline
    def _ensure_table(mut self):
        """The table's entries, made at the first probe."""
        if len(self.entries) == 0:
            self._make_table()

    @no_inline
    def _make_table(mut self):
        self.entries = huge_list(self.capacity, UInt64(0))

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
        # The cleared entries take the dictionary's groups with them
        # (`Abandon` resets the dictionary id).
        self.dictionary_groups.reset()
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
            self.dictionary_groups.reset()

    def _word_ptrs(self) -> List[Int]:
        var word_ptrs = List[Int](capacity=len(self.addressed))
        for w in range(len(self.addressed)):
            word_ptrs.append(Int(self.word_rows[w].unsafe_ptr()))
        return word_ptrs^

    def sink_vectors(
        mut self,
        keys: List[KeyVector],
        inputs: List[UnifiedVectorFormat],
        hold: List[Series],
        first: Int,
        count: Int,
        columns: List[Series] = List[Series](),
    ) raises:
        """A data chunk's `count` rows (at most a batch) into the table
        (`GroupedAggregateHashTable::AddChunk`): each row's group from the
        group chunk's vectors `keys`, then the payload chunk's vectors
        `inputs` (a row count reads none) update the rows' states where
        they are (`UpdateStates`). `hold` are the columns the keys' and
        inputs' strings point into, held until the finish; `first` is the
        source row of the chunk's first.

        As `TryAddCompressedGroups`, keys that are all constant find their
        one group once (`TryAddConstantGroups`), and a sole key whose
        strings index a stored dictionary finds a group once per entry
        (`TryAddDictionaryGroups`); otherwise every row is hashed and
        probed (`FindOrCreateGroups`). The reductions whose states the rows
        hold by address read their input as `columns` (by input, the
        chunk's rows as a column; DuckDB's states that own memory update
        through the same addresses). DuckDB's dictionary path also takes
        a dictionary without an id when it has fewer than half the chunk's
        rows; here every dictionary-coded key comes from a stored
        dictionary (an id), so that case does not arise.

        When every reduction is a distinct count, only the distinct tables
        take the chunk (`CanSkipRegularSink`): the main table's groups and
        counts all come from their finished groups."""
        if len(keys) == 0:
            raise Error("aggregate table: no keys")
        if count > _BATCH:
            raise Error("aggregate table: a chunk is at most a batch")
        if len(self.distinct_slots) > 0 and len(self.distinct_slots) == len(
            self.slots
        ):
            self._sink_distinct(keys, inputs, hold, first, count)
            return
        for column in hold:
            self.held.append(column.copy())
        if self.perfect.used():
            self._perfect_groups(keys, first, count)
            self._update_groups(inputs, first, count, columns)
            return
        var constant = count > 1
        var zero = zero_selection().data
        for c in range(len(keys)):
            constant = constant and keys[c].format.sel.data == zero
        if constant:
            self._constant_groups(keys, first, count)
        elif len(keys) == 1 and keys[0].codes != 0:
            self._dictionary_groups(keys[0], first, count)
        else:
            self._find_or_create(keys, first, count)
        self._update_groups(inputs, first, count, columns)
        self._sink_distinct(keys, inputs, hold, first, count)

    def _update_groups(
        mut self,
        inputs: List[UnifiedVectorFormat],
        first: Int,
        count: Int,
        columns: List[Series],
    ) raises:
        """The chunk's states, by the address of each row's group in
        `groups` (`UpdateStates`)."""
        _update_states(
            self.slots,
            inputs,
            0,
            count,
            self.groups.unsafe_ptr(),
            self.parts.states_at,
            first,
            Pointer[_ValueArena, MutAnyOrigin](
                unsafe_from_address=Int(Pointer(to=self.values))
            ),
            Int(self.discard.unsafe_ptr()),
            Int(self.scratch.unsafe_ptr()),
            self.only,
        )
        if len(self.by_address) > 0 and self.only < 0:
            self._update_by_address(columns, first, count)
        self.sunk += count

    def _perfect_groups(
        mut self, keys: List[KeyVector], first: Int, count: Int
    ) raises:
        """`PerfectAggregateHashTable::AddChunk`'s group addresses: each
        row's slot from its keys' bits (`ComputeGroupLocation`), checked
        against the table's size (statistics that do not hold for the
        data raise, as DuckDB's do), the slot marked set (with the row, its
        first) and its row's address into `groups`."""
        var slots = self.hashes.unsafe_ptr().unsafe_origin_cast[
            MutAnyOrigin
        ]()
        for k in range(count):
            slots[unsafe_offset=k] = 0
        var shift = self.perfect.total_bits
        for c in range(len(keys)):
            shift -= self.perfect.bits[c]
            group_locations(
                keys[c].format,
                keys[c].map,
                keys[c].map_length,
                self.perfect.minima[c],
                shift,
                count,
                Pointer[UInt64, MutAnyOrigin](unsafe_from_address=Int(slots)),
            )
        var total = UInt64(self.perfect.total_groups())
        var base = Int(self.perfect_rows.unsafe_ptr())
        var bytes = 8 * self.parts.stride
        var groups = self.groups.unsafe_ptr()
        for k in range(count):
            var slot = slots[unsafe_offset=k]
            if slot >= total:
                raise Error(
                    "Perfect hash aggregate: aggregate group "
                    + String(slot)
                    + " exceeded total groups "
                    + String(total)
                    + ", so the key statistics do not hold for the data"
                )
            var at = base + Int(slot) * bytes
            var row = _words_at(at)
            if row[unsafe_offset=1] == 0:
                row[unsafe_offset=1] = 1
                row[unsafe_offset=2] = UInt64(first + k)
            groups[unsafe_offset=k] = at

    def _update_by_address(
        mut self, columns: List[Series], first: Int, count: Int
    ) raises:
        """The batch's states held by address: a group row's state with no
        owner yet is made (DuckDB's `initialize` of a new group's state),
        its owner's address and its ordinal there written into the row;
        then each owner takes the batch's rows by their states' ordinals,
        a `Reducer` updating them and a buffer keeping the rows with their
        sequence numbers (`first` onward)."""
        var rows = self.groups.unsafe_ptr()
        var discard = Int(self.discard.unsafe_ptr())
        for s in self.by_address:
            ref slot = self.slots[s]
            var base = 8 * (self.parts.states_at + slot.offset)
            var ordered = slot.kind == _K_ORDERED
            var owner = Int(Pointer(to=self.buffered[s])) if ordered else Int(
                Pointer(to=self.owned[s])
            )
            var made = (
                self.buffered[s]
                .groups if ordered else self.owned[s]
                .group_count
            )
            var ordinals = List[Int](unsafe_uninit_length=count)
            var spare = -1
            for k in range(count):
                var row = rows[unsafe_offset=k]
                if row == discard:
                    # A row that joins no group updates a state no row holds.
                    if not ordered and spare < 0:
                        spare = made
                        made += 1
                    ordinals[k] = spare
                    continue
                var state = _words_at(row + base)
                if state[] == 0:
                    state[] = UInt64(owner)
                    state[unsafe_offset=1] = UInt64(made + 1)
                    state[unsafe_offset=2] = 0
                    made += 1
                ordinals[k] = Int(state[unsafe_offset=1]) - 1
            if ordered:
                self.buffered[s].add(columns[s], ordinals, first, made)
                continue
            self.owned[s].grow(made)
            if slot.kind == _K_STATE and (
                self.owned[s].op == CORR or self.owned[s].op == COV
            ):
                self.owned[s].update_pair(
                    columns[s], columns[slot.k], 0, True, ordinals
                )
            else:
                self.owned[s].update(
                    reducer_input(self.owned[s], columns[s]), 0, True, ordinals
                )

    def prepare_ordered(mut self) raises:
        """Every buffer of input rows ready for the finish to read."""
        for s in self.by_address:
            if self.slots[s].kind == _K_ORDERED:
                self.buffered[s].prepare()

    def _sink_distinct(
        mut self,
        keys: List[KeyVector],
        inputs: List[UnifiedVectorFormat],
        hold: List[Series],
        first: Int,
        count: Int,
    ) raises:
        """Each distinct count's (keys, value) pairs into its own table,
        the value's vector read as a key where it is (`SinkDistinct`)."""
        for d in range(len(self.distinct_slots)):
            var pairs = keys.copy()
            pairs.append(KeyVector.of(inputs[self.distinct_slots[d]]))
            self.distinct.at(d).sink_vectors(
                pairs, List[UnifiedVectorFormat](), hold, first, count
            )

    def _find_or_create(
        mut self, keys: List[KeyVector], first: Int, count: Int
    ) raises:
        """`count` rows' groups into `groups` (`FindOrCreateGroups`): the
        key vectors hashed and turned into key words where they are (the
        key words being the row layout's), then probed."""
        var word_ptrs = self._word_ptrs()
        var nulls = self.nulls.unsafe_ptr()
        var hash_ptr = self.hashes.unsafe_ptr()
        for k in range(count):
            nulls[unsafe_offset=k] = 0
        var w = 0
        for c in range(len(keys)):
            var second = w + 1 if keys[c].width() == 2 else w
            if keys[c].codes != 0:
                self.dictionaries[c].use(keys[c].dictionary)
                self.dictionaries[c].write_vector(
                    keys[c],
                    count,
                    Int(hash_ptr),
                    word_ptrs[w],
                    word_ptrs[second],
                    Int(nulls),
                    UInt64(1) << UInt64(c),
                    c == 0,
                )
            else:
                _vector_key(
                    keys[c],
                    count,
                    Int(hash_ptr),
                    word_ptrs[w],
                    word_ptrs[second],
                    Int(nulls),
                    UInt64(1) << UInt64(c),
                    c == 0,
                )
            w = second + 1
        self._insert[False](
            count,
            first,
            Int(hash_ptr),
            Int(nulls),
            word_ptrs,
            Int(self.groups.unsafe_ptr()),
            0,
        )

    def _constant_groups(
        mut self, keys: List[KeyVector], first: Int, count: Int
    ) raises:
        """DuckDB's `TryAddConstantGroups`: every key is constant, so the
        chunk's first row finds the one group and every row takes it."""
        self._find_or_create(keys, first, 1)
        var group_at = self.groups.unsafe_ptr()
        var group = group_at[]
        for k in range(1, count):
            group_at[unsafe_offset=k] = group
        # The adaptive counters see every row (`AddInput`).
        self.cycle += count - 1

    def _dictionary_groups(
        mut self, key: KeyVector, first: Int, count: Int
    ) raises:
        """DuckDB's `TryAddDictionaryGroups`: the sole key's rows are
        codes into a stored dictionary. The entries the chunk uses that no
        row looked up before (none once every entry has been) are hashed
        and probed once each, in first-row order, and their groups kept by
        entry; every row then takes its entry's group."""
        # Room first for the chunk's rows, so no table clearing within
        # the probe below moves the groups already found.
        self._ensure_table()
        if 3 * (self.live + count) >= 2 * self.capacity:
            self._full(count)
        self.dictionaries[0].use(key.dictionary)
        var size = len(self.dictionaries[0].keys)
        self.dictionary_groups.use(key.dictionary, size)
        var codes = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=key.codes)
        var sel = key.format.sel
        var validity = key.format.validity
        var has_nulls = not validity.all_valid()
        var row_entries = self.dictionary_groups.row_entries.unsafe_ptr()
        var entries = self.dictionary_groups.entries.unsafe_ptr()
        var rows = self.dictionary_groups.rows.unsafe_ptr()
        var addresses = self.dictionary_groups.addresses.unsafe_ptr()
        for k in range(count):
            if has_nulls and not validity.row_is_valid_unsafe(sel.get_index(k)):
                row_entries[unsafe_offset=k] = size
            else:
                row_entries[unsafe_offset=k] = Int(codes[unsafe_offset=k])
        var unique = 0
        if self.dictionary_groups.resolved < size + 1:
            var found = self.dictionary_groups.found.unsafe_ptr()
            for k in range(count):
                var e = row_entries[unsafe_offset=k]
                if not found[unsafe_offset=e]:
                    found[unsafe_offset=e] = True
                    entries[unsafe_offset=unique] = e
                    rows[unsafe_offset=unique] = first + k
                    unique += 1
            self.dictionary_groups.resolved += unique
        var group_at = self.groups.unsafe_ptr()
        if unique > 0:
            var word_ptrs = self._word_ptrs()
            self.dictionaries[0].write_entries(
                Int(entries),
                unique,
                Int(self.hashes.unsafe_ptr()),
                word_ptrs[0],
                word_ptrs[1],
                Int(self.nulls.unsafe_ptr()),
            )
            self._insert[True](
                unique,
                first,
                Int(self.hashes.unsafe_ptr()),
                Int(self.nulls.unsafe_ptr()),
                word_ptrs,
                Int(group_at),
                Int(rows),
            )
            for i in range(unique):
                addresses[unsafe_offset=entries[unsafe_offset=i]] = group_at[
                    unsafe_offset=i
                ]
        # The adaptive counters see every row (`AddInput`).
        self.cycle += count - unique
        for k in range(count):
            group_at[unsafe_offset=k] = addresses[
                unsafe_offset=row_entries[unsafe_offset=k]
            ]

    def _insert[
        listed: Bool
    ](
        mut self,
        count: Int,
        row: Int,
        hashes: Int,
        nulls: Int,
        word_ptrs: List[Int],
        groups: Int,
        rows: Int,
    ) raises:
        """A batch's `count` rows (source rows from `row`, or with `listed`
        each one's at `rows`) into the table: each row's group address
        into `groups`."""
        var hash_ptr = Pointer[UInt64, MutAnyOrigin](unsafe_from_address=hashes)
        var null_ptr = Pointer[UInt64, MutAnyOrigin](unsafe_from_address=nulls)
        var group_at = Pointer[Int, MutAnyOrigin](unsafe_from_address=groups)
        var row_at = Pointer[Int, MutAnyOrigin](unsafe_from_address=rows)
        self._ensure_table()
        if 3 * (self.live + count) >= 2 * self.capacity:
            self._full(count)
        self.cycle += count
        if self.skip:
            for k in range(count):
                var source = row + k
                comptime if listed:
                    source = row_at[unsafe_offset=k]
                group_at[unsafe_offset=k] = self.parts.append(
                    hash_ptr[unsafe_offset=k],
                    null_ptr[unsafe_offset=k],
                    word_ptrs,
                    k,
                    source,
                )
            if not self.grow:
                # While lookups are skipped every row is a group, and the
                # sketch sees them all (DuckDB's `ObserveHashes`).
                for k in range(count):
                    self.hll.add(hash_ptr[unsafe_offset=k])
            self.made += count
            self.live += count
            self.skipped += count
            return
        var mask = self.capacity - 1
        var table = _words_at(Int(self.entries.unsafe_ptr()))
        for k in range(count):
            var hash = hash_ptr[unsafe_offset=k]
            var salt = (hash >> _SALT_SHIFT) << _SALT_SHIFT
            var slot = Int(hash) & mask
            var null = null_ptr[unsafe_offset=k]
            while True:
                var entry = table[unsafe_offset=slot]
                if entry == 0:
                    var source = row + k
                    comptime if listed:
                        source = row_at[unsafe_offset=k]
                    var at = self.parts.append(hash, null, word_ptrs, k, source)
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

    def count_group_rows(
        mut self, groups: _Groups, widths: List[Int], nulls_count: Bool
    ) raises:
        """A distinct table's finished (keys, value) groups into this
        table, each adding one to its group's distinct count (`only`): its
        rows' key words read in place (the keys are the first
        `len(widths) - 1` columns, each `widths` words; the value the last)
        and hashed as `_hash_column` hashes the same values, so the groups
        meet the workers' rows of the same keys at the finish. A group whose
        value is null makes its group but adds nothing unless
        `nulls_count`, so a group of only nulls counts 0 even when the
        workers sank no rows into the main table (`CanSkipRegularSink`);
        each group's first row is its pair's, so the earliest pair's row is
        the group's. DuckDB scans the distinct table a vector at a time and
        sinks it into the main table (`HashAggregateDistinctFinalizeTask`).
        """
        self.in_order = False
        var keys = len(widths) - 1
        var width = len(self.addressed)
        var word_ptrs = List[Int](capacity=width)
        for w in range(width):
            word_ptrs.append(Int(self.word_rows[w].unsafe_ptr()))
        var hashes = self.hashes.unsafe_ptr()
        var nulls = self.nulls.unsafe_ptr()
        var group_at = self.groups.unsafe_ptr()
        var value_null = UInt64(1) << UInt64(keys)
        var base = 8 * (self.parts.states_at + self.slots[self.only].offset)
        var firsts = List[Int](length=_BATCH, fill=0)
        var adds = List[UInt64](length=_BATCH, fill=0)
        var first_at = firsts.unsafe_ptr()
        var add_at = adds.unsafe_ptr()
        var count = 0
        for g in range(groups.count()):
            var address = groups.at(g)
            var row = _words_at(address)
            add_at[unsafe_offset=count] = (
                UInt64(1) if nulls_count
                or row[unsafe_offset=1] & value_null == 0 else 0
            )
            first_at[unsafe_offset=count] = Int(row[unsafe_offset=2])
            var hash = UInt64(0)
            var w = 0
            for c in range(keys):
                var first = row[unsafe_offset=_HEAD + w]
                var part: UInt64
                _words_at(word_ptrs[w])[unsafe_offset=count] = first
                if widths[c] == 2:
                    var second = row[unsafe_offset=_HEAD + w + 1]
                    _words_at(word_ptrs[w + 1])[unsafe_offset=count] = second
                    if (row[unsafe_offset=1] >> UInt64(c)) & 1 != 0:
                        part = _NULL_KEY
                    elif groups.addressed[w + 1]:
                        # A string: its bytes inline in the two words
                        # (after the length) or at the address.
                        var length = Int(first & 0xFFFFFFFF)
                        var at = address + 8 * (_HEAD + w) + 4 if length <= (
                            12
                        ) else Int(second)
                        part = _hash_bytes(
                            Span[UInt8, ImmutAnyOrigin](
                                unsafe_ptr=Pointer[UInt8, ImmutAnyOrigin](
                                    unsafe_from_address=at
                                ),
                                length=length,
                            )
                        )
                    else:
                        # A 128-bit value.
                        part = first ^ second
                elif (row[unsafe_offset=1] >> UInt64(c)) & 1 != 0:
                    part = _NULL_KEY
                else:
                    part = first
                w += widths[c]
                hash = _mix(part) if c == 0 else _hash_combine(hash, part)
            hashes[unsafe_offset=count] = hash
            nulls[unsafe_offset=count] = row[unsafe_offset=1] & (value_null - 1)
            count += 1
            if count == _BATCH:
                self._count_batch(count, word_ptrs, base, firsts, adds)
                count = 0
        if count > 0:
            self._count_batch(count, word_ptrs, base, firsts, adds)

    def _count_batch(
        mut self,
        count: Int,
        word_ptrs: List[Int],
        base: Int,
        firsts: List[Int],
        adds: List[UInt64],
    ) raises:
        """A batch of keys in the batch buffers into their groups (made
        when new, with the first rows `firsts`; one found again keeps the
        earlier), each adding its `adds` to the count at byte `base` of its
        group's row."""
        var group_at = self.groups.unsafe_ptr()
        self._insert[True](
            count,
            0,
            Int(self.hashes.unsafe_ptr()),
            Int(self.nulls.unsafe_ptr()),
            word_ptrs,
            Int(group_at),
            Int(firsts.unsafe_ptr()),
        )
        var add_at = adds.unsafe_ptr()
        var first_at = firsts.unsafe_ptr()
        for k in range(count):
            var group = group_at[unsafe_offset=k]
            _words_at(group + base)[] += add_at[unsafe_offset=k]
            # A group found again keeps the earlier first row.
            var head = _words_at(group)
            var first = UInt64(first_at[unsafe_offset=k])
            if first < head[unsafe_offset=2]:
                head[unsafe_offset=2] = first
        self.sunk += count

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
    # Whether the job stops at its merged group rows (`rows`), unread: a
    # distinct table's, which the main table counts from.
    var rows_only: Bool
    var rows: List[_Groups]
    # Whether the groups come out in the output order (by partition, then
    # by first row); otherwise in whatever order they merge, as DuckDB's
    # GROUP BY and Polars' group_by without maintain_order give them.
    var ordered: Bool

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
        rows_only: Bool,
        ordered: Bool,
    ):
        self.filter = filter.copy()
        self.merged = 0
        self.rows_only = rows_only
        # A distinct table's groups are only counted: never in order.
        self.ordered = ordered and not rows_only
        self.rows = List[_Groups]()
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
        # Ordered (`ordered`), groups come out by partition, then by first
        # row, so the output order depends on the rows alone, not on which
        # worker took which morsel. A worker takes morsels in order and makes a group's row
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
            if in_order and self.bits < self.order_bits and self.ordered:
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
            # Rows merged in first-row order only where that order comes
            # out: not for groups in no order (`ordered` false), nor when
            # some sink's rows are out of it, whose groups are sorted below.
            # Those merge sink by sink, as DuckDB's finalize does.
            var sequential = not self.ordered
            for j in range(sinks):
                sequential = sequential or not jobs[j].in_order
            if sequential:
                in_order = False
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
                if sequential:
                    for j in range(sinks):
                        ref parts = jobs[j].parts
                        for b in range(len(parts.blocks[p])):
                            var at = parts.blocks[p][b]
                            for _ in range(parts.rows_in(p, b)):
                                _ = _merge_row(
                                    at,
                                    table,
                                    mask,
                                    final,
                                    addressed,
                                    self.slots,
                                )
                                at += 8 * stride
                    continue
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
        # Groups in no order skip the output order: a distinct table's
        # (keys, value) groups (`rows_only`), only counted into the main
        # table, and a group-by's without maintain_order.
        if not self.ordered:
            pass
        elif not in_order:
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
        if self.rows_only:
            self.rows.append(final^)
            return
        ref held = Pointer[_HeldBuffers, MutAnyOrigin](
            unsafe_from_address=self.held
        )[]
        _scan_groups(
            final,
            self.slots,
            self.templates,
            self.key_templates,
            held,
            self.filter,
            self.states,
            self.keys,
            self.firsts,
            self.top_rows,
        )


def _scan_groups(
    mut final: _Groups,
    slots: List[AggregateFunction],
    templates: List[ReducerSpec],
    key_templates: List[Series],
    held: _HeldBuffers,
    filter: Optional[GroupFilter],
    mut states: List[Reducer],
    mut keys: List[Series],
    mut firsts_out: List[Int],
    mut top_rows: List[Int],
) raises:
    """A finish job's groups (`final`) read out: with `filter`, only the
    groups it keeps; then every reduction's state finalized and every key
    column read, a vector of groups at a time, into `states`, `keys`,
    `firsts_out` (each group's first row) and `top_rows`."""
    if filter and final.count() > 0:
        # The reductions the filter reads, finished over every group,
        # then only the kept groups' rows, in order, for the scan.
        ref chosen = filter.value()
        var results = List[Reducer](capacity=len(chosen.slots))
        for s in chosen.slots:
            var one = _Finalizer(
                slots[s], templates[s], final.count(), held
            )
            for lo in range(0, final.count(), _BATCH):
                one.scan(final, lo, min(final.count(), lo + _BATCH), held)
            results.append(one^.reducer())
        var kept = chosen.keep(chosen.context, results)
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
    var outputs = List[_Finalizer](capacity=len(templates))
    for s in range(len(templates)):
        if slots[s].kind == _K_TOP_ROWS:
            var w0 = final.states_at + slots[s].offset
            var words = final.rows.unsafe_ptr()
            for g in range(count):
                var w = g * final.stride + w0
                for p in range(Int(words[unsafe_offset=w])):
                    top_rows.append(
                        Int(words[unsafe_offset=w + 2 + 2 * p])
                    )
        outputs.append(
            _Finalizer(slots[s], templates[s], count, held)
        )
    var key_columns = List[_KeyColumn](capacity=len(key_templates))
    var w = 0
    for c in range(len(key_templates)):
        key_columns.append(
            _KeyColumn(
                key_templates[c],
                w,
                UInt64(1) << UInt64(c),
                count,
                held,
            )
        )
        w += _width(key_templates[c])
    firsts_out = List[Int](unsafe_uninit_length=count)
    var firsts = firsts_out.unsafe_ptr()
    for lo in range(0, count, _BATCH):
        var hi = min(count, lo + _BATCH)
        for s in range(len(outputs)):
            outputs[s].scan(final, lo, hi, held)
        for c in range(len(key_columns)):
            key_columns[c].scan(final, lo, hi, held)
        for g in range(lo, hi):
            firsts[unsafe_offset=g] = final.first(g)
    while len(outputs) > 0:
        states.append(outputs.pop(0).reducer())
    while len(key_columns) > 0:
        keys.append(key_columns.pop(0).column())


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


struct _DistinctJob(Job):
    """Finish runs' merged (keys, value) groups of a distinct table sunk
    into one new sink of the main table, each adding one to its group's
    distinct count (DuckDB's `HashAggregateDistinctFinalizeTask`, whose
    local sink takes every partition the task scans)."""

    var groups: List[Int]
    var widths: List[Int]
    var slot: Int
    var nulls: Bool
    var layout: AggregateLayout
    var workers: Int
    var sinks: List[AggregateSink]

    def __init__(
        out self,
        groups: List[Int],
        widths: List[Int],
        slot: Int,
        nulls: Bool,
        layout: AggregateLayout,
        workers: Int,
    ):
        self.groups = groups.copy()
        self.widths = widths.copy()
        self.slot = slot
        self.nulls = nulls
        self.layout = layout.copy()
        self.workers = workers
        self.sinks = List[AggregateSink]()

    def run(mut self) raises:
        var sink = AggregateSink(self.layout, self.workers, distinct=False)
        sink.only = self.slot
        for groups in self.groups:
            sink.count_group_rows(
                Pointer[_Groups, MutAnyOrigin](unsafe_from_address=groups)[],
                self.widths,
                self.nulls,
            )
        self.sinks.append(sink^)


def _finalize_distinct(
    mut sinks: List[AggregateSink], layout: AggregateLayout
) raises:
    """DuckDB's `FinalizeDistinct`: each distinct count's tables (one per
    worker) merged into their (keys, value) groups, whose rows are sunk
    into new sinks of the main table, on their own threads: as many as
    there are runs, at most one per worker (`CreateGlobalSources` and
    `MaxThreads`), each taking every run its task scans. Those sinks join
    `sinks`, and the main finish merges their counts into the groups. The long strings their keys point at stay in
    the tables' held columns, which the new sinks hold too."""
    # The workers' sinks; the new sinks join the list after them.
    var workers = len(sinks)
    for d in range(len(layout.distinct)):
        var slot = layout.distinct[d]
        var tables = List[AggregateSink](capacity=workers)
        for j in range(workers):
            tables.append(sinks[j].distinct.take(d))
        var table_layout = layout.distinct_layout(d)
        var pairs = aggregate_finish(
            tables, table_layout, rows_only=True, ordered=False
        )
        var widths = layout_widths(table_layout)
        var nulls = layout.templates[slot].integer == 0
        var runs = List[Int]()
        for p in range(len(pairs.rows)):
            if pairs.rows[p].count() > 0:
                runs.append(Int(Pointer(to=pairs.rows[p])))
        var tasks = min(len(runs), workers)
        var jobs = List[_DistinctJob](capacity=tasks)
        for t in range(tasks):
            var share = List[Int]()
            for r in range(t, len(runs), tasks):
                share.append(runs[r])
            jobs.append(
                _DistinctJob(share, widths, slot, nulls, layout, workers)
            )
        run_jobs(jobs)
        var held = List[Series]()
        for j in range(len(tables)):
            for column in tables[j].held:
                held.append(column.copy())
        for j in range(len(jobs)):
            while len(jobs[j].sinks) > 0:
                var sink = jobs[j].sinks.pop()
                for column in held:
                    sink.held.append(column.copy())
                sinks.append(sink^)
        _ = pairs^
        _ = tables^


struct _PerfectScanJob(Job):
    """A range of a perfect hash table's group slots, [lo, hi): every
    worker's rows of those slots combined into the first worker's
    (`PerfectAggregateHashTable::Combine`), then the occupied slots read out
    in slot order, each key rebuilt from the slot's bits (`Scan`)."""

    var sinks: Int
    var held: Int
    var lo: Int
    var hi: Int
    var templates: List[ReducerSpec]
    var slots: List[AggregateFunction]
    var key_templates: List[Series]
    var perfect: PerfectHash
    var filter: Optional[GroupFilter]
    var merged: Int
    var states: List[Reducer]
    var firsts: List[Int]
    var keys: List[Series]
    var top_rows: List[Int]

    def __init__(
        out self,
        sinks: Int,
        held: Int,
        lo: Int,
        hi: Int,
        layout: AggregateLayout,
        filter: Optional[GroupFilter],
    ):
        self.sinks = sinks
        self.held = held
        self.lo = lo
        self.hi = hi
        self.templates = layout.templates.copy()
        self.slots = layout.slots.copy()
        self.key_templates = layout.key_templates.copy()
        self.perfect = layout.perfect.copy()
        self.filter = filter.copy()
        self.merged = 0
        self.states = List[Reducer]()
        self.firsts = List[Int]()
        self.keys = List[Series]()
        self.top_rows = List[Int]()

    def run(mut self) raises:
        ref jobs = Pointer[List[AggregateSink], MutAnyOrigin](
            unsafe_from_address=self.sinks
        )[]
        var stride = jobs[0].parts.stride
        var states_bytes = 8 * jobs[0].parts.states_at
        var bytes = 8 * stride
        var base = Int(jobs[0].perfect_rows.unsafe_ptr())
        var count = 0
        for slot in range(self.lo, self.hi):
            var into = base + slot * bytes
            var mine = _words_at(into)
            for j in range(1, len(jobs)):
                var other = Int(jobs[j].perfect_rows.unsafe_ptr()) + slot * bytes
                var theirs = _words_at(other)
                if theirs[unsafe_offset=1] == 0:
                    continue
                if mine[unsafe_offset=1] == 0:
                    _ = external_call["memcpy", Int](into, other, bytes)
                    continue
                combine(self.slots, into + states_bytes, other + states_bytes)
                if theirs[unsafe_offset=2] < mine[unsafe_offset=2]:
                    mine[unsafe_offset=2] = theirs[unsafe_offset=2]
            count += Int(mine[unsafe_offset=1] != 0)
        var final = _Groups(
            len(jobs[0].addressed),
            jobs[0].addressed.copy(),
            stride - jobs[0].parts.states_at,
            count,
        )
        var key_count = len(self.key_templates)
        for slot in range(self.lo, self.hi):
            var into = base + slot * bytes
            if _words_at(into)[unsafe_offset=1] == 0:
                continue
            var row = _words_at(final.at(final.copy_row(into)))
            # The row's null bits and key words, from the slot's bits.
            var nulls = UInt64(0)
            for c in range(key_count):
                var key = key_word(self.perfect, c, slot)
                if not key[0]:
                    nulls |= UInt64(1) << UInt64(c)
                row[unsafe_offset=_HEAD + c] = key[1]
            row[] = 0
            row[unsafe_offset=1] = nulls
        self.merged = count
        ref held = Pointer[_HeldBuffers, MutAnyOrigin](
            unsafe_from_address=self.held
        )[]
        _scan_groups(
            final,
            self.slots,
            self.templates,
            self.key_templates,
            held,
            self.filter,
            self.states,
            self.keys,
            self.firsts,
            self.top_rows,
        )


def _perfect_finish(
    mut sinks: List[AggregateSink],
    layout: AggregateLayout,
    filter: Optional[GroupFilter],
) raises -> AggregateResult:
    """The perfect hash table's finish (`PhysicalPerfectHashAggregate`'s
    `Combine` and source): the group slots cut into ranges holding about an
    equal share of what finishing reads (each occupied slot's rows and the
    values of its quantile states, as `aggregate_finish` cuts partitions),
    and each range combined and read out on its own thread, one chunk of
    the result each, in slot order."""
    var result = AggregateResult()
    for j in range(len(sinks)):
        sinks[j].prepare_ordered()
    var columns = List[List[Series]](capacity=len(sinks))
    for j in range(len(sinks)):
        columns.append(sinks[j].held.copy())
    var held = _HeldBuffers(columns)
    _ = columns^
    var stride = sinks[0].parts.stride
    var counted = List[Int]()
    for slot in layout.slots:
        if slot.kind == _K_VALUES:
            counted.append(sinks[0].parts.states_at + slot.offset + 2)
    var groups = layout.perfect.total_groups()
    var weight = List[Int](length=groups, fill=0)
    var total = 0
    for j in range(len(sinks)):
        var rows = sinks[j].perfect_rows.unsafe_ptr()
        for g in range(groups):
            var row = g * stride
            if rows[unsafe_offset=row + 1] == 0:
                continue
            var bytes = 8 * stride
            for w in counted:
                bytes += 8 * Int(rows[unsafe_offset=row + w])
            weight[g] += bytes
            total += bytes
    var per_job = max(
        _PARTITION_BYTES, -(-total // max(1, configured_workers()))
    )
    var jobs = List[_PerfectScanJob]()
    var lo = 0
    var job = 0
    for g in range(groups):
        job += weight[g]
        if job >= per_job or g == groups - 1:
            if job > 0:
                jobs.append(
                    _PerfectScanJob(
                        Int(Pointer(to=sinks)),
                        Int(Pointer(to=held)),
                        lo,
                        g + 1,
                        layout,
                        filter,
                    )
                )
            lo = g + 1
            job = 0
    run_jobs(jobs)
    _ = held^
    for p in range(len(jobs)):
        var firsts = List[Int]()
        var states = List[Reducer]()
        var keys_out = List[Series]()
        result.top_rows.extend(Span(jobs[p].top_rows))
        result.merged += jobs[p].merged
        swap(firsts, jobs[p].firsts)
        swap(states, jobs[p].states)
        swap(keys_out, jobs[p].keys)
        result.firsts.append(firsts^)
        result.states.append(states^)
        result.keys.append(keys_out^)
    return result^


def aggregate_finish(
    mut sinks: List[AggregateSink],
    layout: AggregateLayout,
    filter: Optional[GroupFilter] = None,
    rows_only: Bool = False,
    *,
    ordered: Bool,
) raises -> AggregateResult:
    """Phase 2 over every worker's sink: runs of consecutive partitions
    holding about half a 256 KiB block of rows each. This bounds each
    working set so its table and rows can stay in cache while runs merge
    on separate threads. With `filter`, only the groups it keeps come
    out (`GroupFilter`); the result's `merged` counts them all. With
    `ordered`, each partition's groups come out in first-row order (by
    partition at the most bits the sinks could reach, then first row),
    which `table_frame` merges into first-occurrence order; otherwise in
    no particular order."""
    var result = AggregateResult()
    if len(sinks) == 0:
        return result^
    if layout.perfect.used():
        return _perfect_finish(sinks, layout, filter)
    if len(layout.distinct) > 0:
        _finalize_distinct(sinks, layout)
    for j in range(len(sinks)):
        sinks[j].prepare_ordered()
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
                        rows_only,
                        ordered,
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
        while len(finishes[p].rows) > 0:
            result.rows.append(finishes[p].rows.pop())
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
    # With `rows_only`, each finish job's merged group rows, unread.
    var rows: List[_Groups]

    def __init__(out self):
        self.merged = 0
        self.firsts = List[List[Int]]()
        self.states = List[List[Reducer]]()
        self.keys = List[List[Series]]()
        self.top_rows = List[Int]()
        self.rows = List[_Groups]()
