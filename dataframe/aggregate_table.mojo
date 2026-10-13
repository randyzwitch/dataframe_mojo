"""The grouped aggregate hash table, after DuckDB's
(`src/execution/aggregate_hashtable.cpp`, `radix_partitioned_hashtable.cpp`,
`tuple_data_collection.cpp`).

Each group is one row of 64-bit words, as DuckDB's TupleData rows: its
hash, its key null bits, its first row, its key words, then its
reductions' states inline (a sum's total and count, a minimum and whether
one was seen, Welford moments), all zero when empty. A key word is a
canonical integer or float (the hash's own), or a string's 16-byte form
(DuckDB's `string_t`): its length and first four bytes, then its next eight
bytes, or for a longer string the address of its bytes, compared with
memcmp when the words agree.

Two phases, as there:

1. Sink. Workers take morsels (row ranges within a chunk) from a shared
   cursor. Each hashes a batch of 2048 rows where they are, with the hash
   the partitioner uses (`partition._hash_column`), and probes its own
   table: a flat array of 8-byte entries, the hash's top 16 bits (a salt)
   over the address of a group's row. A salt that differs rejects a slot
   without reading the row. A new group's row is appended to its
   partition (256, by the hash bits below the salt) in blocks that never
   move, so the finish reads a partition's rows in sequence with no
   reordering. The batch's states then update through each row's group
   address (DuckDB's `UpdateStates`).

   With more than two workers the table never grows: two-thirds full, its
   entries are cleared and its rows kept (DuckDB's `Abandon`), so the
   table probed stays cache-sized however many groups there are, and a
   group seen again later is merged at the finish. With two or fewer it
   doubles instead. After a million rows a worker checks a HyperLogLog of
   its hashes: past 95% distinct it stops probing and appends every row as
   a group (DuckDB's `SkipLookups`).

2. Finish. Consecutive partitions holding about half a 256 KiB block of
   rows form a run; each run merges every worker's rows of its partitions
   in a fresh table sized for them, on its own thread, folding a repeated
   group's states into its first row. Its states then load into `Reducer`s,
   which finish them as every other route does, and its keys are read
   back out of the rows.

A string longer than twelve bytes is compared through the address of its
bytes in the input, which outlives the aggregation; the input is never
copied.
"""
from std.atomic import Atomic
from std.bit import count_trailing_zeros
from std.collections import Optional
from std.math import isnan, log
from std.ffi import external_call
from std.memory import ArcPointer, Pointer, bitcast
from std.sys import size_of

from .aggregate import Reducer, float_key
from .decimal import check_limit, precision_limit
from .bool_column import BoolColumn
from .column import Column, _pack_bits
from .dtype import DataType, NUMERIC_DTYPES
from .expr import (
    COUNT,
    LEN,
    MAX,
    MEAN,
    MIN,
    NULL_COUNT,
    STD,
    SUM,
    VAR,
)
from .indexed_reduce import _canonical_int
from .huge_pages import huge_list, huge_uninit
from .parallel import Job, run_jobs
from .partition import (
    _NULL_KEY,
    _combine as _hash_combine,
    _hash_bytes,
    _hash_column,
    _mix,
)
from .series import Series
from .hashing import RowKeys
from .trace import trace_path
from .string_bytes import _Bytes
from .string_column import StringColumn
from .string_view import (
    STRING_VIEW_INLINE_BYTES,
    StringView,
    StringViewStorage,
    _STRING_VIEW_BUFFER_LIMIT,
)

# Rows hashed and probed at a time (DuckDB's STANDARD_VECTOR_SIZE).
comptime _BATCH = 2048
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
comptime _MAX_PARTITION_BITS = 8
# Bytes of a worker's row arena: two huge pages.
comptime _ARENA_BYTES = 4 << 20

comptime _SALT_SHIFT = UInt64(48)
comptime _ID_MASK = (UInt64(1) << 48) - 1


@always_inline
def _entry(hash: UInt64, at: Int) -> UInt64:
    """A table entry: the hash's top 16 bits over a group row's address
    (DuckDB's salted pointer) or a nonzero group number; zero is empty."""
    return ((hash >> _SALT_SHIFT) << _SALT_SHIFT) | UInt64(at)


@always_inline
def _stride(hash: UInt64) -> Int:
    """The probe step after a collision: odd, from the salt, as DuckDB's."""
    return Int((hash >> 59) | 1)


def reduction_supported(op: Int) -> Bool:
    """A reduction whose state is a few fixed words merged in any order
    (DuckDB's aggregate states inline in the group's row): workers take
    morsels as they come, so first, last and positional reductions, and
    states that grow (distinct sets, samples), keep the other routes."""
    return op in [SUM, MEAN, MIN, MAX, COUNT, LEN, STD, VAR, NULL_COUNT]


# What a state slot holds, in words after the group's key words.
comptime _K_LEN = 0  # rows
comptime _K_COUNT = 1  # valid values
comptime _K_NULLS = 2  # null values
comptime _K_INT_SUM = 3  # 128-bit total (two words), count
comptime _K_FLOAT_SUM = 4  # total, count
comptime _K_MIN = 5  # best, flags (1 seen, 2 NaN seen)
comptime _K_MAX = 6
comptime _K_MOMENTS = 7  # count, mean, m2 (Welford, as `VarState`)
# Int64 total, count: an integer input of at most 32 bits over fewer than
# 2^31 rows cannot overflow it (DuckDB's `sum_no_overflow`).
comptime _K_NARROW_SUM = 8
# A decimal sum or mean: 128-bit total, count; the total is held to the
# result's precision on every add, as the `Reducer`'s decimal sums are.
comptime _K_DECIMAL_SUM = 9


@fieldwise_init
struct _Slot(Copyable, Movable):
    """One reduction's state in a group row: its kind, its first word after
    the keys, and whether its values are floats (minima and maxima
    compare as `Float64` then, else as canonical `Int64`)."""

    var kind: Int
    var offset: Int
    var floating: Bool
    # The state's type (a decimal sum's result type, for its precision).
    var dtype: DataType

    def words(self) -> Int:
        if (
            self.kind == _K_INT_SUM
            or self.kind == _K_MOMENTS
            or self.kind == _K_DECIMAL_SUM
        ):
            return 3
        if (
            self.kind == _K_FLOAT_SUM
            or self.kind == _K_NARROW_SUM
            or self.kind == _K_MIN
            or (self.kind == _K_MAX)
        ):
            return 2
        return 1


def _numeric_part(column: Series) -> Optional[Series]:
    """A column's storage as one numeric column (a chunked one's first
    chunk, for its type), or None when it is not a plain number."""
    var dtype = column.dtype()
    if dtype.is_decimal() or dtype.is_categorical() or dtype.is_nested():
        return None
    var part = column.chunks()[
        0
    ].copy() if column.is_chunked() else column.copy()
    comptime for t in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[t]
        if part._data.isa[Column[Scalar[D]]]():
            return part^
    return None


def _slot_kind(spec: ReducerSpec, input: Series) -> Int:
    """The inline state kind of a reduction over `input`, or -1 when its
    state is not a few fixed words here."""
    if spec.op == LEN:
        return _K_LEN
    if (spec.op == SUM or spec.op == MEAN) and input.dtype().is_decimal():
        var state = spec.make(0).dtype
        if state.is_decimal() and _decimal_part(input):
            return _K_DECIMAL_SUM
        return -1
    var part = _numeric_part(input)
    if not part:
        return -1
    var floating = part.value().dtype().is_float()
    var dtype = spec.make(0).dtype
    if dtype != DataType.INT64 and dtype != DataType.FLOAT64:
        return -1
    if (dtype == DataType.FLOAT64) != floating:
        return -1
    if spec.op == COUNT:
        return _K_COUNT
    if spec.op == NULL_COUNT:
        return _K_NULLS
    if spec.op == SUM or spec.op == MEAN:
        if floating:
            return _K_FLOAT_SUM
        if part.value().dtype().bit_width() <= 32 and len(input) < (1 << 31):
            return _K_NARROW_SUM
        return _K_INT_SUM
    if spec.op == MIN:
        return _K_MIN
    if spec.op == MAX:
        return _K_MAX
    if spec.op == STD or spec.op == VAR:
        return _K_MOMENTS
    return -1


def states_supported(specs: List[ReducerSpec], inputs: List[Series]) -> Bool:
    """Whether every reduction's state fits inline in a group row."""
    for s in range(len(specs)):
        if _slot_kind(specs[s], inputs[s]) < 0:
            return False
    return True


@always_inline
def _words_at(address: Int) -> Pointer[UInt64, MutAnyOrigin]:
    return Pointer[UInt64, MutAnyOrigin](unsafe_from_address=address)


def _update_slot[
    D: DType
](
    slot: _Slot,
    column: Column[Scalar[D]],
    start: Int,
    count: Int,
    rows: Pointer[Int, _],
    base: Int,
):
    """A batch's values (column rows [start, start + count)) into their
    groups' state words, `base` bytes into each group's row (`rows` holds
    their addresses; DuckDB's `UpdateStates` by address)."""
    var values = column._ptr()
    var nulls = column.null_count() > 0
    var kind = slot.kind
    if kind == _K_COUNT or kind == _K_NULLS:
        if not nulls:
            if kind == _K_COUNT:
                for k in range(count):
                    _words_at(rows[unsafe_offset=k] + base)[] += 1
            return
        for k in range(count):
            if column._valid(start + k) == (kind == _K_COUNT):
                _words_at(rows[unsafe_offset=k] + base)[] += 1
        return
    for k in range(count):
        var i = start + k
        if nulls and not column._valid(i):
            continue
        var at = _words_at(rows[unsafe_offset=k] + base)
        var w = 0
        var value = values[unsafe_offset=i]
        if kind == _K_INT_SUM:
            var total = (UInt128(at[unsafe_offset=w + 1]) << 64) | UInt128(
                at[unsafe_offset=w]
            )
            total += bitcast[DType.uint128](value.cast[DType.int128]())
            at[unsafe_offset=w] = UInt64(total)
            at[unsafe_offset=w + 1] = UInt64(total >> 64)
            at[unsafe_offset=w + 2] += 1
        elif kind == _K_NARROW_SUM:
            at[unsafe_offset=w] += bitcast[DType.uint64](
                value.cast[DType.int64]()
            )
            at[unsafe_offset=w + 1] += 1
        elif kind == _K_FLOAT_SUM:
            at[unsafe_offset=w] = bitcast[DType.uint64](
                bitcast[DType.float64](at[unsafe_offset=w]) + Float64(value)
            )
            at[unsafe_offset=w + 1] += 1
        elif kind == _K_MOMENTS:
            var x: Float64
            comptime if D.is_floating_point():
                x = Float64(value)
            else:
                x = Float64(_canonical_int[D](value))
            var n = at[unsafe_offset=w] + 1
            var mean = bitcast[DType.float64](at[unsafe_offset=w + 1])
            var delta = x - mean
            mean += delta / Float64(n)
            var m2 = bitcast[DType.float64](at[unsafe_offset=w + 2]) + delta * (
                x - mean
            )
            at[unsafe_offset=w] = n
            at[unsafe_offset=w + 1] = bitcast[DType.uint64](mean)
            at[unsafe_offset=w + 2] = bitcast[DType.uint64](m2)
        else:
            var is_max = kind == _K_MAX
            var seen = at[unsafe_offset=w + 1] & 1 != 0
            comptime if D.is_floating_point():
                var x = Float64(value)
                if isnan(x):
                    at[unsafe_offset=w + 1] |= 2
                    continue
                var best = bitcast[DType.float64](at[unsafe_offset=w])
                if (
                    not seen
                    or (is_max and x > best)
                    or (not is_max and x < best)
                ):
                    at[unsafe_offset=w] = bitcast[DType.uint64](x)
                    at[unsafe_offset=w + 1] |= 1
            else:
                # Ties keep the earlier value, as `_extreme` does.
                var x = _canonical_int[D](value)
                var best = bitcast[DType.int64](at[unsafe_offset=w])
                if (
                    not seen
                    or (is_max and x > best)
                    or (not is_max and x < best)
                ):
                    at[unsafe_offset=w] = bitcast[DType.uint64](x)
                    at[unsafe_offset=w + 1] |= 1


def _decimal_part(column: Series) -> Bool:
    """Whether a decimal column's storage is 32-, 64- or 128-bit integers."""
    var part = column.chunks()[
        0
    ].copy() if column.is_chunked() else column.copy()
    return (
        part._data.isa[Column[Int32]]()
        or part._data.isa[Column[Int64]]()
        or part._data.isa[Column[Int128]]()
    )


def _update_decimal[
    D: DType
](
    slot: _Slot,
    column: Column[Scalar[D]],
    start: Int,
    count: Int,
    rows: Pointer[Int, _],
    base: Int,
) raises:
    """A batch's decimal values (unscaled, at their storage width) into
    their groups' 128-bit totals and counts, each total held to the
    state's precision."""
    var values = column._ptr()
    var nulls = column.null_count() > 0
    var limit = precision_limit(slot.dtype.precision())
    for k in range(count):
        var i = start + k
        if nulls and not column._valid(i):
            continue
        var at = _words_at(rows[unsafe_offset=k] + base)
        var total = bitcast[DType.int128](
            (UInt128(at[unsafe_offset=1]) << 64) | UInt128(at[unsafe_offset=0])
        )
        total = check_limit(
            total + Int128(values[unsafe_offset=i]), limit, slot.dtype
        )
        var word = bitcast[DType.uint128](total)
        at[unsafe_offset=0] = UInt64(word)
        at[unsafe_offset=1] = UInt64(word >> 64)
        at[unsafe_offset=2] += 1


def _update_states(
    slots: List[_Slot],
    inputs: List[Series],
    start: Int,
    count: Int,
    rows: Pointer[Int, _],
    states_at: Int,
) raises:
    """Every reduction's states for a batch (`inputs` are the morsel's
    slices, one chunk each; `rows` the addresses of the rows' groups, whose
    states start at word `states_at`)."""
    for s in range(len(slots)):
        ref slot = slots[s]
        var base = 8 * (states_at + slot.offset)
        if slot.kind == _K_LEN:
            for k in range(count):
                _words_at(rows[unsafe_offset=k] + base)[] += 1
            continue
        ref part = inputs[s]
        if slot.kind == _K_DECIMAL_SUM:
            if part._data.isa[Column[Int64]]():
                _update_decimal[DType.int64](
                    slot, part._data[Column[Int64]], start, count, rows, base
                )
            elif part._data.isa[Column[Int32]]():
                _update_decimal[DType.int32](
                    slot, part._data[Column[Int32]], start, count, rows, base
                )
            else:
                _update_decimal[DType.int128](
                    slot, part._data[Column[Int128]], start, count, rows, base
                )
            continue
        var done = False
        comptime for t in range(len(NUMERIC_DTYPES)):
            comptime D = NUMERIC_DTYPES[t]
            if not done and part._data.isa[Column[Scalar[D]]]():
                _update_slot[D](
                    slot,
                    part._data[Column[Scalar[D]]],
                    start,
                    count,
                    rows,
                    base,
                )
                done = True
        if not done:
            raise Error("aggregate table: unsupported reduction input")


def _combine(slots: List[_Slot], into: Int, other: Int) raises:
    """Group row `other`'s states folded into group row `into` (both the
    addresses of their first state word's row)."""
    var a = _words_at(into)
    var b = _words_at(other)
    for s in range(len(slots)):
        var w = slots[s].offset
        var kind = slots[s].kind
        if kind <= _K_NULLS:
            a[unsafe_offset=w] += b[unsafe_offset=w]
        elif kind == _K_NARROW_SUM:
            a[unsafe_offset=w] += b[unsafe_offset=w]
            a[unsafe_offset=w + 1] += b[unsafe_offset=w + 1]
        elif kind == _K_DECIMAL_SUM:
            var x = bitcast[DType.int128](
                (UInt128(a[unsafe_offset=w + 1]) << 64)
                | UInt128(a[unsafe_offset=w])
            )
            var y = bitcast[DType.int128](
                (UInt128(b[unsafe_offset=w + 1]) << 64)
                | UInt128(b[unsafe_offset=w])
            )
            var total = bitcast[DType.uint128](
                check_limit(
                    x + y,
                    precision_limit(slots[s].dtype.precision()),
                    slots[s].dtype,
                )
            )
            a[unsafe_offset=w] = UInt64(total)
            a[unsafe_offset=w + 1] = UInt64(total >> 64)
            a[unsafe_offset=w + 2] += b[unsafe_offset=w + 2]
        elif kind == _K_INT_SUM:
            var x = (UInt128(a[unsafe_offset=w + 1]) << 64) | UInt128(
                a[unsafe_offset=w]
            )
            var y = (UInt128(b[unsafe_offset=w + 1]) << 64) | UInt128(
                b[unsafe_offset=w]
            )
            x += y
            a[unsafe_offset=w] = UInt64(x)
            a[unsafe_offset=w + 1] = UInt64(x >> 64)
            a[unsafe_offset=w + 2] += b[unsafe_offset=w + 2]
        elif kind == _K_FLOAT_SUM:
            a[unsafe_offset=w] = bitcast[DType.uint64](
                bitcast[DType.float64](a[unsafe_offset=w])
                + bitcast[DType.float64](b[unsafe_offset=w])
            )
            a[unsafe_offset=w + 1] += b[unsafe_offset=w + 1]
        elif kind == _K_MOMENTS:
            var m = b[unsafe_offset=w]
            if m == 0:
                continue
            var n = a[unsafe_offset=w]
            if n == 0:
                a[unsafe_offset=w] = m
                a[unsafe_offset=w + 1] = b[unsafe_offset=w + 1]
                a[unsafe_offset=w + 2] = b[unsafe_offset=w + 2]
                continue
            var total = n + m
            var mean = bitcast[DType.float64](a[unsafe_offset=w + 1])
            var delta = bitcast[DType.float64](b[unsafe_offset=w + 1]) - mean
            mean += delta * (Float64(m) / Float64(total))
            var m2 = (
                bitcast[DType.float64](a[unsafe_offset=w + 2])
                + bitcast[DType.float64](b[unsafe_offset=w + 2])
                + delta * delta * Float64(n) * Float64(m) / Float64(total)
            )
            a[unsafe_offset=w] = total
            a[unsafe_offset=w + 1] = bitcast[DType.uint64](mean)
            a[unsafe_offset=w + 2] = bitcast[DType.uint64](m2)
        else:
            var flags = b[unsafe_offset=w + 1]
            a[unsafe_offset=w + 1] |= flags & 2
            if flags & 1 == 0:
                continue
            var is_max = kind == _K_MAX
            var better: Bool
            if a[unsafe_offset=w + 1] & 1 == 0:
                better = True
            elif slots[s].floating:
                var x = bitcast[DType.float64](b[unsafe_offset=w])
                var y = bitcast[DType.float64](a[unsafe_offset=w])
                better = (is_max and x > y) or (not is_max and x < y)
            else:
                var x = bitcast[DType.int64](b[unsafe_offset=w])
                var y = bitcast[DType.int64](a[unsafe_offset=w])
                better = (is_max and x > y) or (not is_max and x < y)
            if better:
                a[unsafe_offset=w] = b[unsafe_offset=w]
                a[unsafe_offset=w + 1] |= 1


def _unload(slot: _Slot, groups: _Groups, spec: ReducerSpec) raises -> Reducer:
    """A reduction's states of every group, from their rows into a
    `Reducer`, which finishes them as every other route does."""
    var n = groups.count()
    var reducer = spec.make(n)
    var at = groups.rows.unsafe_ptr()
    var stride = groups.stride
    var w0 = groups.states_at + slot.offset
    var kind = slot.kind
    for g in range(n):
        var w = g * stride + w0
        if kind <= _K_NULLS:
            reducer.counts[g] = bitcast[DType.int64](at[unsafe_offset=w])
        elif kind == _K_INT_SUM:
            var total = (UInt128(at[unsafe_offset=w + 1]) << 64) | UInt128(
                at[unsafe_offset=w]
            )
            reducer.int_sums[g].total = bitcast[DType.int128](total)
            reducer.int_sums[g].count = bitcast[DType.int64](
                at[unsafe_offset=w + 2]
            )
        elif kind == _K_DECIMAL_SUM:
            reducer.decimals[g] = bitcast[DType.int128](
                (UInt128(at[unsafe_offset=w + 1]) << 64)
                | UInt128(at[unsafe_offset=w])
            )
            reducer.counts[g] = bitcast[DType.int64](at[unsafe_offset=w + 2])
        elif kind == _K_NARROW_SUM:
            reducer.int_sums[g].total = bitcast[DType.int64](
                at[unsafe_offset=w]
            ).cast[DType.int128]()
            reducer.int_sums[g].count = bitcast[DType.int64](
                at[unsafe_offset=w + 1]
            )
        elif kind == _K_FLOAT_SUM:
            reducer.float_sums[g].total = bitcast[DType.float64](
                at[unsafe_offset=w]
            )
            reducer.float_sums[g].count = bitcast[DType.int64](
                at[unsafe_offset=w + 1]
            )
        elif kind == _K_MOMENTS:
            reducer.moments[g].count = bitcast[DType.int64](at[unsafe_offset=w])
            reducer.moments[g].mean = bitcast[DType.float64](
                at[unsafe_offset=w + 1]
            )
            reducer.moments[g].m2 = bitcast[DType.float64](
                at[unsafe_offset=w + 2]
            )
        else:
            var flags = at[unsafe_offset=w + 1]
            reducer.seen[g] = flags & 1 != 0
            if slot.floating:
                reducer.nan_seen[g] = flags & 2 != 0
                reducer.floats[g] = bitcast[DType.float64](at[unsafe_offset=w])
            else:
                reducer.ints[g] = bitcast[DType.int64](at[unsafe_offset=w])
    return reducer^


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


def _width(column: Series) -> Int:
    """Key words a column takes: two for a string or a 128-bit decimal."""
    var part = column.chunks()[
        0
    ].copy() if column.is_chunked() else column.copy()
    if part._data.isa[StringColumn]() or part._data.isa[Column[Int128]]():
        return 2
    return 1


def _string_words(
    strings: StringColumn,
    start: Int,
    count: Int,
    first_at: Int,
    second_at: Int,
    nulls_at: Int,
    bit: UInt64,
):
    """Rows [start, start + count) of a string column as DuckDB's 16-byte
    `string_t`: length and first four bytes, then the next eight bytes up to
    twelve, or the address of the bytes of a longer string."""
    var first = Pointer[UInt64, MutAnyOrigin](unsafe_from_address=first_at)
    var second = Pointer[UInt64, MutAnyOrigin](unsafe_from_address=second_at)
    var nulls = Pointer[UInt64, MutAnyOrigin](unsafe_from_address=nulls_at)
    var has_nulls = strings.null_count() > 0
    var views = strings._is_view_storage()
    if views:
        # A view is already these words: its first word, then its inline
        # bytes or (past twelve bytes) where its bytes are.
        var rows = _Bytes(strings)
        var words = Pointer[UInt64, ImmutAnyOrigin](
            unsafe_from_address=rows.views
        )
        for k in range(count):
            var i = start + k
            if has_nulls and not strings._valid(i):
                first[unsafe_offset=k] = 0
                second[unsafe_offset=k] = 0
                nulls[unsafe_offset=k] |= bit
                continue
            var head = rows.head(i)
            var length = Int(head & 0xFFFFFFFF)
            first[unsafe_offset=k] = head
            if length <= 12:
                var keep = 8 * max(length - 4, 0)
                second[unsafe_offset=k] = words[unsafe_offset=2 * i + 1] & (
                    (UInt64(1) << UInt64(keep)) - 1 if keep < 64 else UInt64.MAX
                )
            else:
                second[unsafe_offset=k] = UInt64(Int(rows.get(i).unsafe_ptr()))
        return
    var size = len(strings._bytes[])
    for k in range(count):
        var i = start + k
        if has_nulls and not strings._valid(i):
            first[unsafe_offset=k] = 0
            second[unsafe_offset=k] = 0
            nulls[unsafe_offset=k] |= bit
            continue
        var bytes = strings._row_bytes(i) if views else Span[
            UInt8, ImmutAnyOrigin
        ](
            unsafe_ptr=strings._base().unsafe_offset(strings._start(i)),
            length=strings._end(i) - strings._start(i),
        )
        var length = len(bytes)
        var start_ptr = bytes.unsafe_ptr()
        var head = UInt64(length)
        if not views and strings._start(i) + 16 <= size:
            # Sixteen readable bytes from the string's start: its words by
            # two loads, masked to its length.
            var low = start_ptr.unsafe_bitcast[UInt64]().unsafe_load[
                alignment=1
            ]()
            head |= (low & 0xFFFFFFFF) << 32
            first[unsafe_offset=k] = head
            if length <= 12:
                var high = (low >> 32) | (
                    UInt64(
                        start_ptr.unsafe_offset(8)
                        .unsafe_bitcast[UInt32]()
                        .unsafe_load[alignment=1]()
                    )
                    << 32
                )
                var keep = 8 * max(length - 4, 0)
                second[unsafe_offset=k] = high & (
                    (UInt64(1) << UInt64(keep)) - 1 if keep < 64 else UInt64.MAX
                )
                if length < 4:
                    first[unsafe_offset=k] = UInt64(length) | (
                        (low & ((UInt64(1) << UInt64(8 * length)) - 1)) << 32
                    )
            else:
                second[unsafe_offset=k] = UInt64(Int(start_ptr))
            continue
        for b in range(min(length, 4)):
            head |= UInt64(start_ptr[unsafe_offset=b]) << UInt64(32 + 8 * b)
        first[unsafe_offset=k] = head
        if length <= 12:
            var tail = UInt64(0)
            for b in range(4, length):
                tail |= UInt64(start_ptr[unsafe_offset=b]) << UInt64(
                    8 * (b - 4)
                )
            second[unsafe_offset=k] = tail
        else:
            second[unsafe_offset=k] = UInt64(Int(start_ptr))


def _column_words(
    column: Series,
    start: Int,
    count: Int,
    first_at: Int,
    second_at: Int,
    nulls_at: Int,
    bit: UInt64,
) raises:
    """A key column's words for rows [start, start + count) of a column
    that is not chunked, written at the given buffer addresses; a null row
    sets `bit` in its null word."""
    if column._data.isa[StringColumn]():
        _string_words(
            column._data[StringColumn],
            start,
            count,
            first_at,
            second_at,
            nulls_at,
            bit,
        )
        return
    var first = Pointer[UInt64, MutAnyOrigin](unsafe_from_address=first_at)
    var second = Pointer[UInt64, MutAnyOrigin](unsafe_from_address=second_at)
    var nulls = Pointer[UInt64, MutAnyOrigin](unsafe_from_address=nulls_at)
    if column._data.isa[BoolColumn]():
        ref bools = column._data[BoolColumn]
        for k in range(count):
            if not bools._valid(start + k):
                first[unsafe_offset=k] = 0
                nulls[unsafe_offset=k] |= bit
            else:
                first[unsafe_offset=k] = UInt64(1) if bools._get(
                    start + k
                ) else UInt64(2)
        return
    if column._data.isa[Column[Int128]]():
        ref wide = column._data[Column[Int128]]
        var has_nulls = wide.null_count() > 0
        for k in range(count):
            if has_nulls and not wide._valid(start + k):
                first[unsafe_offset=k] = 0
                second[unsafe_offset=k] = 0
                nulls[unsafe_offset=k] |= bit
                continue
            var value = bitcast[DType.uint128](wide._get(start + k))
            first[unsafe_offset=k] = UInt64(value)
            second[unsafe_offset=k] = UInt64(value >> 64)
        return
    comptime for d in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[d]
        if column._data.isa[Column[Scalar[D]]]():
            ref typed = column._data[Column[Scalar[D]]]
            var values = typed._ptr()
            var has_nulls = typed.null_count() > 0
            for k in range(count):
                var i = start + k
                if has_nulls and not typed._valid(i):
                    first[unsafe_offset=k] = 0
                    nulls[unsafe_offset=k] |= bit
                    continue
                var value = values[unsafe_offset=i]
                comptime if D.is_floating_point():
                    # The hash's canonical float: -0.0 is 0.0, NaNs are one.
                    first[unsafe_offset=k] = float_key(Float64(value))
                elif D.is_unsigned():
                    first[unsafe_offset=k] = UInt64(value)
                else:
                    first[unsafe_offset=k] = bitcast[DType.uint64](Int64(value))
            return
    raise Error("aggregate table: unsupported key storage")


# Row words before the key words: hash, null bits, first row.
comptime _HEAD = 3


struct _Groups(Movable):
    """A partition's final groups in rows of 64-bit words, as DuckDB's
    TupleData rows: each group's hash, null bits and first row, its key
    words (two per string or 128-bit column), then its state words."""

    var rows: List[UInt64]
    var stride: Int
    # The row's first state word.
    var states_at: Int
    # Per key word: whether it is a long string's address (its bytes
    # compared), the word before it holding the length.
    var addressed: List[Bool]

    var n: Int

    def __init__(
        out self,
        width: Int,
        var addressed: List[Bool],
        state_words: Int,
        capacity: Int,
    ):
        """Room for `capacity` groups, allocated once."""
        self.states_at = _HEAD + width
        self.stride = _HEAD + width + state_words
        self.rows = List[UInt64](
            unsafe_uninit_length=max(1, capacity) * self.stride
        )
        self.addressed = addressed^
        self.n = 0

    def count(self) -> Int:
        return self.n

    def width(self) -> Int:
        return self.states_at - _HEAD

    @always_inline
    def at(self, g: Int) -> Int:
        """The address of group g's row."""
        return Int(self.rows.unsafe_ptr().unsafe_offset(g * self.stride))

    @always_inline
    def null(self, g: Int) -> UInt64:
        return self.rows.unsafe_ptr()[unsafe_offset=g * self.stride + 1]

    @always_inline
    def first(self, g: Int) -> Int:
        return Int(self.rows.unsafe_ptr()[unsafe_offset=g * self.stride + 2])

    @always_inline
    def word(self, g: Int, w: Int) -> UInt64:
        return self.rows.unsafe_ptr()[unsafe_offset=g * self.stride + _HEAD + w]

    @always_inline
    def copy_row(mut self, address: Int) -> Int:
        """A new group with the row at `address`; its id."""
        var g = self.n
        _ = external_call["memcpy", Int](self.at(g), address, 8 * self.stride)
        self.n += 1
        return g


struct _Partitioned(Movable):
    """A worker's groups as DuckDB's partitioned TupleData: rows appended
    to their partition (the hash bits below the salt) as they are made, in
    blocks that never move, so a table entry holds a row's address and the
    finish reads each partition's rows in sequence. Each partition's next
    row and block end sit in two small arrays (DuckDB's append state), so
    an append touches only the row it writes."""

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
    """A full sink table's next capacity, or 0 to keep it (DuckDB main's
    `RadixHTAdaptivity::TryGrow`): it grows, to fit every group its
    HyperLogLog counts and at least double, once the group rows it made
    again after clearing (`made` against the distinct count) take more
    bytes than four times the larger table, within the worker's share of
    memory (`memory` bytes)."""
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
    """A worker's share of memory for its sink: 80% of physical memory (as
    DuckDB's default memory limit) divided among the workers."""
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


@fieldwise_init
struct ReducerSpec(Copyable, Movable):
    """What a reduction's state is (a `Reducer` without its groups), so
    every worker and partition makes its own."""

    var op: Int
    var input: DataType
    var min_count: Int
    var integer: Int64
    var floating: Float64
    var text: String
    var logical: DataType

    @staticmethod
    def of(template: Reducer) -> ReducerSpec:
        return ReducerSpec(
            template.op,
            template.input,
            template.min_count,
            template.integer,
            template.floating,
            template.text,
            template.logical,
        )

    def make(self, groups: Int) -> Reducer:
        return Reducer(
            self.op,
            self.input,
            groups,
            self.min_count,
            self.integer,
            self.floating,
            self.text,
            Optional(self.logical),
        )


@fieldwise_init
struct AggregateLayout(Copyable, Movable):
    """What every worker's sink and the finish share: each reduction's
    state slot and template, which key words are long strings' addresses,
    the state words, the partition bits and empty key columns."""

    var slots: List[_Slot]
    var templates: List[ReducerSpec]
    var addressed: List[Bool]
    var state_words: Int
    var bits: Int
    var key_templates: List[Series]


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
    var slots = List[_Slot](capacity=len(templates))
    var state_words = 0
    for s in range(len(templates)):
        var kind = _slot_kind(templates[s], inputs[s])
        if kind < 0:
            raise Error("aggregate table: a state that is not inline")
        var floating = False
        if kind != _K_LEN and kind != _K_DECIMAL_SUM:
            floating = _numeric_part(inputs[s]).value().dtype().is_float()
        slots.append(
            _Slot(kind, state_words, floating, templates[s].make(0).dtype)
        )
        state_words += slots[s].words()
    var key_templates = List[Series](capacity=len(keys))
    for key in keys:
        key_templates.append(key.slice(0, 0))
    return AggregateLayout(
        slots^,
        templates.copy(),
        addressed^,
        state_words,
        _MAX_PARTITION_BITS,
        key_templates^,
    )


struct _DictionaryKeys(Movable):
    """One key column's dictionary entries as the sink reads them (after
    DuckDB's `TryAddDictionaryGroups`, which looks each dictionary entry up
    once): each entry's string hash and its two key words, computed the
    first time a row holds its code, for the dictionary at `identity`."""

    var identity: Int
    var done: List[Bool]
    var keys: List[UInt64]
    var first: List[UInt64]
    var second: List[UInt64]
    # The morsel's codes, one per row.
    var codes: List[UInt32]

    def __init__(out self):
        self.identity = 0
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
        ref dictionary = strings._codes.value()[].dtype.dictionary()[]
        var start = Int(dictionary.offsets[code])
        var length = Int(dictionary.offsets[code + 1]) - start
        var bytes = dictionary.bytes.unsafe_ptr().unsafe_offset(start)
        self.keys[code] = _hash_bytes(
            Span[UInt8, ImmutAnyOrigin](
                unsafe_ptr=bytes.unsafe_mut_cast[False]().unsafe_origin_cast[
                    ImmutAnyOrigin
                ](),
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
            self.second[code] = UInt64(Int(bytes))
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
        var has_nulls = strings.null_count() > 0
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


struct AggregateSink(Movable):
    """One worker's sink (DuckDB's `RadixHTLocalSinkState`): its table of
    salted row addresses, its partitioned group rows, and the counts its
    fill cycles adapt by, fed a morsel at a time (`sink`)."""

    var slots: List[_Slot]
    var addressed: List[Bool]
    var parts: _Partitioned
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

    def __init__(out self, layout: AggregateLayout, workers: Int):
        self.slots = layout.slots.copy()
        self.addressed = layout.addressed.copy()
        self.parts = _Partitioned(
            layout.bits, layout.addressed, layout.state_words
        )
        self.grow = workers <= _GROW_WORKERS
        self.memory = _worker_memory(workers)
        self.capacity = _SINK_CAPACITY
        self.entries = huge_list(self.capacity, UInt64(0))
        self.live = 0
        self.made = 0
        self.sunk = 0
        self.cycle = 0
        self.skipped = 0
        self.skip = False
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
        self.row_ids = List[Int]()
        self.representatives = List[Int]()

    def _full(mut self, count: Int):
        """A table without room for `count` more groups (DuckDB main's fill
        cycle, `Sink` and `radix_ht_adaptivity.cpp`): grown if its
        duplicates justify a larger one, else its entries cleared and its
        rows kept (`Abandon`), deciding whether to skip lookups."""
        if self.grow:
            while 3 * (self.live + count) >= 2 * self.capacity:
                self.capacity *= 2
            self.entries = huge_list(self.capacity, UInt64(0))
            self._reinsert()
            return
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

    def sink(
        mut self, keys: List[Series], inputs: List[Series], first: Int
    ) raises:
        """A morsel's rows into the table: `keys` and `inputs` (a row count
        reads none) as unchunked columns of its rows, `first` the source
        row of its first."""
        if len(keys) == 0:
            raise Error("aggregate table: no keys")
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
            if not self.grow:
                for k in range(count):
                    self.hll.add(hash_ptr[unsafe_offset=k])
            if 3 * (self.live + count) >= 2 * self.capacity:
                self._full(count)
            self.cycle += count
            if self.skip:
                self.made += count
                self.live += count
                self.skipped += count
                for k in range(count):
                    group_at[unsafe_offset=k] = self.parts.append(
                        hash_ptr[unsafe_offset=k],
                        nulls[unsafe_offset=k],
                        word_ptrs,
                        k,
                        first + lo + k,
                    )
            else:
                var mask = self.capacity - 1
                var table = _words_at(Int(self.entries.unsafe_ptr()))
                for k in range(count):
                    var hash = hash_ptr[unsafe_offset=k]
                    var salt = (hash >> _SALT_SHIFT) << _SALT_SHIFT
                    var slot = Int(hash) & mask
                    var null = nulls[unsafe_offset=k]
                    while True:
                        var entry = table[unsafe_offset=slot]
                        if entry == 0:
                            var at = self.parts.append(
                                hash, null, word_ptrs, k, first + lo + k
                            )
                            if self.numbering:
                                _words_at(
                                    at + 8 * self.parts.states_at
                                )[] = UInt64(self.made)
                                self.representatives.append(first + lo + k)
                            table[unsafe_offset=slot] = _entry(hash, at)
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
            )
        self.sunk += length

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
    var slots: List[_Slot]
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
                var size = 0 if self.slots[s].kind == _K_LEN else length
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
    slots: List[_Slot],
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
                _combine(slots, mine + states_bytes, at + states_bytes)
                var first = row[unsafe_offset=2]
                if first < _words_at(mine)[unsafe_offset=2]:
                    _words_at(mine)[unsafe_offset=2] = first
                return False
        slot = (slot + _stride(hash)) & mask


struct _FinishJob(Job):
    """Phase 2: a run of partitions' rows from every worker merged in a
    fresh table sized for them, states folded into each group's row."""

    var sinks: Int
    var bits: Int
    var lo: Int
    var hi: Int
    var total: Int
    var templates: List[ReducerSpec]
    var slots: List[_Slot]
    var key_templates: List[Series]
    var states: List[Reducer]
    var firsts: List[Int]
    var keys: List[Series]
    # The address of the sinks' `_HeldBuffers`.
    var held: Int

    def __init__(
        out self,
        sinks: Int,
        held: Int,
        lo: Int,
        hi: Int,
        total: Int,
        templates: List[ReducerSpec],
        slots: List[_Slot],
        key_templates: List[Series],
        bits: Int,
    ):
        self.sinks = sinks
        self.held = held
        self.bits = bits
        self.lo = lo
        self.hi = hi
        self.total = total
        self.templates = templates.copy()
        self.slots = slots.copy()
        self.key_templates = key_templates.copy()
        self.states = List[Reducer]()
        self.firsts = List[Int]()
        self.keys = List[Series]()

    def run(mut self) raises:
        ref jobs = Pointer[List[AggregateSink], MutAnyOrigin](
            unsafe_from_address=self.sinks
        )[]
        # DuckDB's finalize capacity: the rows' count at a 1.5 load factor.
        var capacity = 64
        while 2 * capacity < 3 * max(self.total, 1):
            capacity *= 2
        var mask = capacity - 1
        var entries = huge_list(capacity, UInt64(0))
        var table = entries.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
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
        var block = List[Int](length=sinks, fill=0)
        var row_in = List[Int](length=sinks, fill=0)
        # Each worker's next row's first row, Int.MAX when it has none.
        var head = List[Int](length=sinks, fill=Int.MAX)
        var in_order = True
        for p in range(self.lo, self.hi):
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
                head[best] = _next_first(parts, p, block[best], row_in[best])
        _ = entries^
        if not in_order:
            # A worker's rows were not in first-row order: sort the groups.
            trace_path("aggregate_table.sorted_groups")
            # Each as one 128-bit key: partition and first row above, the
            # group below (first rows stay under 2^56).
            var shift = UInt64(48 - self.bits)
            var part_mask = UInt64((1 << self.bits) - 1)
            var keyed = List[UInt128](capacity=final.count())
            for g in range(final.count()):
                var part = (_words_at(final.at(g))[] >> shift) & part_mask
                var place = (UInt128(part) << 56) | UInt128(final.first(g))
                keyed.append((place << 64) | UInt128(g))
            sort(keyed)
            var ordered = _Groups(
                final.width(),
                final.addressed.copy(),
                final.stride - final.states_at,
                final.count(),
            )
            for key in keyed:
                _ = ordered.copy_row(
                    final.at(Int(UInt64(key & 0xFFFFFFFFFFFFFFFF)))
                )
            final = ordered^
        for s in range(len(self.templates)):
            self.states.append(_unload(self.slots[s], final, self.templates[s]))
        # Keys scanned out of the stored words (DuckDB's gather from its
        # rows), not gathered from the input.
        var w = 0
        for c in range(len(self.key_templates)):
            self.keys.append(
                _key_column(
                    self.key_templates[c],
                    w,
                    UInt64(1) << UInt64(c),
                    final,
                    Pointer[_HeldBuffers, MutAnyOrigin](
                        unsafe_from_address=self.held
                    )[],
                )
            )
            w += _width(self.key_templates[c])
        var count = final.count()
        self.firsts = List[Int](capacity=count)
        for g in range(count):
            self.firsts.append(final.first(g))


struct _HeldBuffers(Movable):
    """The byte buffers of every string key the sinks hold, by address, so
    a finished string key column is Arrow views into them: no group's
    bytes are copied or even read, as Polars gathers a BinaryView column
    and DuckDB's `string_t` points into its heap. A group's long key words
    hold the address of its bytes, found here by binary search."""

    var buffers: List[ArcPointer[List[UInt8]]]
    # Buffer ranges in address order: start, end and buffer.
    var starts: List[Int]
    var ends: List[Int]
    var index: List[Int]

    def __init__(out self, sinks: List[AggregateSink]):
        self.buffers = List[ArcPointer[List[UInt8]]]()
        self.starts = List[Int]()
        self.ends = List[Int]()
        self.index = List[Int]()
        var keyed = List[UInt128]()
        for j in range(len(sinks)):
            for column in sinks[j].held:
                ref strings = column._data[StringColumn]
                if strings._is_view_storage():
                    ref storage = strings._view_storage.value()
                    for b in range(len(storage._buffers[])):
                        self._add(storage._buffers[][b], keyed)
                else:
                    self._add(strings._bytes, keyed)
        sort(keyed)
        var last = -1
        for key in keyed:
            var b = Int(UInt64(key & 0xFFFFFFFFFFFFFFFF))
            var start = Int(self.buffers[b][].unsafe_ptr())
            if start == last:
                # Morsels slice one column: its buffers repeat.
                continue
            last = start
            self.starts.append(start)
            self.ends.append(start + len(self.buffers[b][]))
            self.index.append(b)

    def _add(
        mut self, buffer: ArcPointer[List[UInt8]], mut keyed: List[UInt128]
    ):
        if len(buffer[]) == 0:
            return
        var start = UInt64(Int(buffer[].unsafe_ptr()))
        keyed.append((UInt128(start) << 64) | UInt128(len(self.buffers)))
        self.buffers.append(buffer.copy())

    def find(self, address: Int, length: Int) -> Tuple[Int, Int]:
        """The buffer holding `length` bytes at `address`, and their offset
        in it; (-1, 0) when none does."""
        var lo = 0
        var hi = len(self.starts)
        while lo < hi:
            var mid = (lo + hi) // 2
            if self.starts[mid] <= address:
                lo = mid + 1
            else:
                hi = mid
        if lo == 0:
            return (-1, 0)
        var r = lo - 1
        if address + length > self.ends[r]:
            return (-1, 0)
        return (self.index[r], address - self.starts[r])


def _string_keys(
    w: Int, bit: UInt64, groups: _Groups, held: _HeldBuffers
) -> StringColumn:
    """A string key column of finished groups as Arrow views, from each
    group's words: a key up to twelve bytes is its two words exactly (the
    view's length, prefix and inline bytes, zero past the end); a longer
    one is its first word and the buffer and offset its address falls in.
    A key held in no shared buffer (a dictionary's entry) is copied."""
    var n = groups.count()
    var views = List[StringView](capacity=n)
    var valid = List[Bool](capacity=n)
    var nulls = False
    var buffers = List[ArcPointer[List[UInt8]]]()
    # Each held buffer's place among this column's, once it is used.
    var local = List[Int](length=len(held.buffers), fill=-1)
    var spill = List[UInt8]()
    var spill_at = -1
    var total_bytes = 0
    var buffer_bytes = 0
    for g in range(n):
        if groups.null(g) & bit != 0:
            views.append(StringView(0, 0, 0, 0))
            valid.append(False)
            nulls = True
            continue
        valid.append(True)
        var head = groups.word(g, w)
        var tail = groups.word(g, w + 1)
        var length = Int(head & 0xFFFFFFFF)
        total_bytes += length
        if length <= STRING_VIEW_INLINE_BYTES:
            views.append(
                StringView(
                    UInt32(head & 0xFFFFFFFF),
                    UInt32(head >> 32),
                    UInt32(tail & 0xFFFFFFFF),
                    UInt32(tail >> 32),
                )
            )
            continue
        buffer_bytes += length
        var found = held.find(Int(tail), length)
        var b: Int
        var offset = found[1]
        if found[0] >= 0 and offset <= _STRING_VIEW_BUFFER_LIMIT - length:
            b = local[found[0]]
            if b < 0:
                b = len(buffers)
                local[found[0]] = b
                buffers.append(held.buffers[found[0]].copy())
        else:
            if spill_at >= 0 and (
                len(spill) + length > _STRING_VIEW_BUFFER_LIMIT
            ):
                buffers[spill_at] = ArcPointer(spill^)
                spill = List[UInt8]()
                spill_at = -1
            if spill_at < 0:
                spill_at = len(buffers)
                buffers.append(ArcPointer(List[UInt8]()))
            b = spill_at
            offset = len(spill)
            spill.extend(
                Span[UInt8, ImmutAnyOrigin](
                    unsafe_ptr=Pointer[UInt8, ImmutAnyOrigin](
                        unsafe_from_address=Int(tail)
                    ),
                    length=length,
                )
            )
        views.append(
            StringView(
                UInt32(head & 0xFFFFFFFF),
                UInt32(head >> 32),
                UInt32(b),
                UInt32(offset),
            )
        )
    if spill_at >= 0:
        buffers[spill_at] = ArcPointer(spill^)
    var bits = _pack_bits(valid) if nulls else List[UInt8]()
    return StringColumn(
        StringViewStorage(views^, buffers^, bits^, n, total_bytes, buffer_bytes)
    )


def _key_column(
    template: Series, w: Int, bit: UInt64, groups: _Groups, held: _HeldBuffers
) raises -> Series:
    """One key column of a partition's final groups, built from the words
    of each group's first sight: strings as views (`_string_keys`), a
    number from its value, as the template's type (temporal, decimal and
    categorical keys keep theirs)."""
    var n = groups.count()
    var name = template.name()
    var dtype = template.dtype()
    var part = template.chunks()[0].copy() if template.is_chunked() else (
        template.copy()
    )
    if part._data.isa[StringColumn]():
        return Series(name, _string_keys(w, bit, groups, held)).with_dtype(
            dtype
        )
    var valid = List[Bool](capacity=n)
    if part._data.isa[BoolColumn]():
        var values = List[Bool](capacity=n)
        for f in range(n):
            var g = f
            valid.append(groups.null(g) & bit == 0)
            values.append(groups.word(g, w) == 1)
        return Series(name, BoolColumn(values, valid))
    if part._data.isa[Column[Int128]]():
        var values = List[Int128](capacity=n)
        for f in range(n):
            var g = f
            valid.append(groups.null(g) & bit == 0)
            var wide = (UInt128(groups.word(g, w + 1)) << 64) | UInt128(
                groups.word(g, w)
            )
            values.append(bitcast[DType.int128](wide))
        return Series(name, Column[Int128](values^, valid)).with_dtype(dtype)
    comptime for d in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[d]
        if part._data.isa[Column[Scalar[D]]]():
            var values = List[Scalar[D]](capacity=n)
            for f in range(n):
                var g = f
                valid.append(groups.null(g) & bit == 0)
                var word = groups.word(g, w)
                comptime if D.is_floating_point():
                    values.append(Scalar[D](bitcast[DType.float64](word)))
                elif D.is_unsigned():
                    values.append(Scalar[D](word))
                else:
                    values.append(Scalar[D](bitcast[DType.int64](word)))
            return Series(name, Column[Scalar[D]](values^, valid)).with_dtype(
                dtype
            )
    raise Error("aggregate table: unsupported key storage")


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


def aggregate_finish(
    mut sinks: List[AggregateSink], layout: AggregateLayout
) raises -> AggregateResult:
    """Phase 2 over every worker's sink: runs of consecutive partitions
    holding about half a 256 KiB block of rows each (where DuckDB adds
    partition bits), so each run's table and rows stay in cache, merged on
    their own threads."""
    var result = AggregateResult()
    if len(sinks) == 0:
        return result^
    var bits = layout.bits
    var stride = sinks[0].parts.stride
    var held = _HeldBuffers(sinks)
    var per_run = max(1, _PARTITION_BYTES // (8 * stride))
    var finishes = List[_FinishJob]()
    var lo = 0
    var run = 0
    for p in range(1 << bits):
        for j in range(len(sinks)):
            ref blocks = sinks[j].parts.blocks[p]
            for b in range(len(blocks)):
                run += sinks[j].parts.rows_in(p, b)
        if run >= per_run or p == (1 << bits) - 1:
            if run > 0:
                finishes.append(
                    _FinishJob(
                        Int(Pointer(to=sinks)),
                        Int(Pointer(to=held)),
                        lo,
                        p + 1,
                        run,
                        layout.templates,
                        layout.slots,
                        layout.key_templates,
                        bits,
                    )
                )
            lo = p + 1
            run = 0
    run_jobs(finishes)
    _ = held^
    for p in range(len(finishes)):
        var firsts = List[Int]()
        var states = List[Reducer]()
        var keys_out = List[Series]()
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

    def __init__(out self):
        self.firsts = List[List[Int]]()
        self.states = List[List[Reducer]]()
        self.keys = List[List[Series]]()


def table_row_keys(keys: List[Series]) raises -> RowKeys:
    """Dense ids of the rows' key tuples in first-occurrence order, with
    nulls equal (`encode_rows` with `nulls_equal`), by the aggregate table's
    hashing and in-place key compare on one thread: no row-encoded copy of
    the keys (Polars numbers window groups with its hash groupers the same
    way). Every key must be `key_supported`."""
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
