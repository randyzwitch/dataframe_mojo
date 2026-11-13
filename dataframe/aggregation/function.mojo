"""Aggregate state stored inline in each group's row.

Each reduction reserves `state_size` 64-bit words after the keys.
`initialize` zeros an empty state, `update` folds input rows into it,
`combine` merges workers' partial states, and `finalize` writes results.
Variable-size state lives in arenas or reducers owned by the sink, so
destroying the sink releases it without a separate allocation per row.

Counts, sums, moments and numeric extrema keep fixed-size state. String
extrema hold a length, prefix and inline bytes or a pointer into retained
input buffers. First and last also keep sequence numbers, so merging
workers preserves input order. Boolean reductions track true, false and
null flags. Quantiles retain value chains. Reductions requiring input order
buffer values and sequence numbers, then sort before finalizing.
"""
from dataframe.storage.buffer import _Buffer, _BufferData
from std.collections import Optional
from std.math import isnan
from std.memory import (
    ArcPointer,
    Pointer,
    bitcast,
    unsafe_memcmp,
    unsafe_memcpy,
)

from dataframe.aggregation.reducer import (
    Reducer,
    _from_canonical,
    quantile_in,
)
from dataframe.categorical import decode
from dataframe.decimal import check_limit, decimal_mean, precision_limit
from dataframe.storage.boolean import BoolColumn
from dataframe.storage.nested import ListColumn
from dataframe.storage.column import (
    Column,
    _count_set,
    _pack_bits,
)
from dataframe.dtype import DataType, NUMERIC_DTYPES
from dataframe.expr import (
    ALL,
    ANY,
    ARG_MAX,
    ARG_MIN,
    CORR,
    N_UNIQUE,
    COUNT,
    COV,
    FIRST,
    IMPLODE,
    LAST,
    LEN,
    MAX,
    MEAN,
    MEDIAN,
    MIN,
    MODE,
    NULL_COUNT,
    PRODUCT,
    QUANTILE,
    KURTOSIS,
    SKEW,
    STD,
    SUM,
    VAR,
)
from dataframe.aggregation.selected_rows import _canonical_int
from dataframe.storage.allocation import huge_uninit
from dataframe.kernels.rank import _key
from dataframe.series import Series
from dataframe.text.bytes import _Bytes
from dataframe.storage.string import StringColumn
from dataframe.vector.types import (
    PHYSICAL_BOOL,
    PHYSICAL_INT128,
    PHYSICAL_INT32,
    PHYSICAL_INT64,
    PHYSICAL_VARCHAR,
    STRING_INLINE_BYTES,
    StringT,
    UnifiedVectorFormat,
    physical_of,
    string_data,
)
from dataframe.storage.string_view import (
    STRING_VIEW_INLINE_BYTES,
    StringView,
    StringViewStorage,
    _STRING_VIEW_BUFFER_LIMIT,
)

# Rows updated at a time.
comptime _BATCH = 2048
# Bytes of a sink's arena: two huge pages.
comptime _ARENA_BYTES = 4 << 20


def reaggregable(op: Int) -> Bool:
    """Whether a reduction's state, merged with another group's state,
    gives the state of the union of their rows: a coarser grouping's states
    come from a finer one's group rows (`sink_group_rows`). Totals, counts,
    extremes, logic flags and first or last values (by their rows'
    sequence numbers) do (a mean is its total and count); a quantile's
    values and the Welford moments are left to the rows."""
    return op in [
        SUM,
        MEAN,
        MIN,
        MAX,
        COUNT,
        LEN,
        NULL_COUNT,
        FIRST,
        LAST,
        ANY,
        ALL,
    ]


def reduction_supported(op: Int) -> Bool:
    """Whether a reduction has an `AggregateFunction` here: a fixed-size
    state, or a quantile's chain of values. Positional reductions, distinct
    sets and holistic reductions use other execution routes."""
    return op in [
        SUM,
        MEAN,
        MIN,
        MAX,
        COUNT,
        LEN,
        STD,
        VAR,
        NULL_COUNT,
        MEDIAN,
        QUANTILE,
        CORR,
        COV,
        FIRST,
        LAST,
        ANY,
        ALL,
        N_UNIQUE,
    ]


def table_reduction(op: Int) -> Bool:
    """Whether the aggregate table holds a reduction's state for groups:
    an `AggregateFunction` of its own (`reduction_supported`), or a state by
    address (`_K_STATE`, `_K_ORDERED`) for the positional, product and
    moment reductions, mode and
    implode."""
    return reduction_supported(op) or op in [
        ARG_MIN,
        ARG_MAX,
        SKEW,
        KURTOSIS,
        PRODUCT,
        MODE,
        IMPLODE,
    ]


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
# 2^31 rows cannot overflow it.
comptime _K_NARROW_SUM = 8
# A decimal sum or mean: 128-bit total, count; the total is held to the
# result's precision on every add, as the `Reducer`'s decimal sums are.
comptime _K_DECIMAL_SUM = 9
# A group's first k (value, row) pairs by ordinal rank: how many are
# held, then each pair's order key and row, kept sorted, first best.
comptime _K_TOP_ROWS = 10
# A median or quantile: three words, the newest and the oldest segment of
# the group's chain of valid values and how many it holds; all 0 when empty.
comptime _K_VALUES = 11
# A correlation or covariance of two columns: pairs counted, the two
# means, the co-moment and the two
# second moments (Welford's update, merged pairwise). The second column is
# an input after the reductions' own (`AggregateFunction.k` is its index).
comptime _K_COMOMENT = 12
# A string's minimum or maximum: the value
# as a string_t (its length and first four bytes, then up to eight more
# inline bytes or the address of its bytes in the held input), then a word
# that is 1 once a value is held.
comptime _K_STRING_MIN = 13
comptime _K_STRING_MAX = 14
# A first or last value: its value words (as a key
# column's: one, or two for a 128-bit value or a string), the sequence
# number of its row, then flags (1 set, 2 the value is valid). A worker
# sees its rows in increasing sequence, so an update keeps its first (or
# overwrites with every later) row, and a combine keeps the lower (or
# higher) sequence number.
comptime _K_FIRST = 15
comptime _K_LAST = 16
# Kleene flags of Boolean values: 1 a true, 2 a false, 4 a null seen (any,
# all, and a Boolean's minimum and maximum).
comptime _K_LOGIC = 17
# A decimal's minimum or maximum: its 128-bit unscaled value, then 1 once a
# value is held.
comptime _K_WIDE_MIN = 18
comptime _K_WIDE_MAX = 19
# A distinct count: the number of the
# group's distinct values, which the sink's distinct table of (keys, value)
# groups adds to its group (`AggregateSink.distinct`); a row's update
# leaves it alone.
comptime _K_DISTINCT = 20
# A state held by its owner rather than inline in the group row: the address
# of its owner, the worker's `Reducer` of such states (one per reduction),
# the state's ordinal there plus one, and the address of the next state
# merged into this group's (0 for none). A combine links the other
# group's states after this one's, as quantile value chains are linked,
# so no state is copied until the finish folds the chain into one
# `Reducer` of the job's groups. The owner's states live as long as the
# worker's sink, and go with it.
comptime _K_STATE = 21
# An order-sensitive reduction buffers arguments and sort keys: the same three
# words, the owner being the worker's `_OrderedBuffer` of the reduction's
# input rows and their sequence numbers. The finish sorts the group's
# buffered rows by sequence (their input order) and only then reduces them:
# first and last take the row at either end, the others replay the rows
# into a `Reducer` in that order.
comptime _K_ORDERED = 22
# The most pairs a group row holds inline: 33 state words, four cache lines.
comptime TOP_ROWS_INLINE = 16


# A median or quantile's values are kept in segments cut from the sink's
# value arena, chained from the newest back: a segment is its previous
# segment's address, the values it holds, its capacity, then the values as
# Float64. Segments double from two values to 1024, so a group's values
# are never copied as they grow; merging groups links their chains.
comptime _VALUES_FIRST = 2
comptime _VALUES_LAST = 1024
comptime _SEGMENT_HEAD = 3


struct _ValueArena(Movable):
    """The segments of a sink's quantile values, cut from arenas on huge
    pages where granted."""

    var arenas: List[List[UInt8]]
    var next: Int
    var end: Int

    def __init__(out self):
        self.arenas = List[List[UInt8]]()
        self.next = 0
        self.end = 0

    @no_inline
    def _refill(mut self, bytes: Int):
        var size = max(_ARENA_BYTES, bytes)
        self.arenas.append(huge_uninit[UInt8](size))
        self.next = Int(self.arenas[len(self.arenas) - 1].unsafe_ptr())
        self.end = self.next + size

    @always_inline
    def segment(mut self, capacity: Int, previous: Int) -> Int:
        """A new empty segment after `previous`; its address."""
        var bytes = 8 * (_SEGMENT_HEAD + capacity)
        if self.next + bytes > self.end:
            self._refill(bytes)
        var at = self.next
        self.next += bytes
        var head = _words_at(at)
        head[unsafe_offset=0] = UInt64(previous)
        head[unsafe_offset=1] = 0
        head[unsafe_offset=2] = UInt64(capacity)
        return at


@always_inline
def _values_push(
    state: Pointer[UInt64, MutAnyOrigin],
    value: Float64,
    arena: Pointer[_ValueArena, MutAnyOrigin],
):
    """A value onto the chain whose newest segment, oldest segment and
    value count are the three words at `state` (all 0 for none)."""
    var tail = Int(state[])
    var head = _words_at(tail)
    if tail == 0 or head[unsafe_offset=1] == head[unsafe_offset=2]:
        var capacity = _VALUES_FIRST
        if tail != 0:
            capacity = min(2 * Int(head[unsafe_offset=2]), _VALUES_LAST)
        var fresh = arena[].segment(capacity, tail)
        if tail == 0:
            state[unsafe_offset=1] = UInt64(fresh)
        state[] = UInt64(fresh)
        tail = fresh
        head = _words_at(tail)
    var fill = Int(head[unsafe_offset=1])
    head[unsafe_offset=_SEGMENT_HEAD + fill] = bitcast[DType.uint64](value)
    head[unsafe_offset=1] = UInt64(fill + 1)
    state[unsafe_offset=2] += 1


def _values_append(into: Pointer[UInt64, MutAnyOrigin], other: Int):
    """The chain whose three state words start at `other` appended to the
    one at `into`: the older chain's oldest segment now follows the other's
    newest."""
    var added = _words_at(other)
    if added[unsafe_offset=2] == 0:
        return
    if into[unsafe_offset=2] == 0:
        for w in range(3):
            into[unsafe_offset=w] = added[unsafe_offset=w]
        return
    _words_at(Int(into[unsafe_offset=1]))[] = added[]
    into[unsafe_offset=1] = added[unsafe_offset=1]
    into[unsafe_offset=2] += added[unsafe_offset=2]


def _values_gather(
    state: Pointer[UInt64, MutAnyOrigin], into: Pointer[Float64, MutAnyOrigin]
):
    """The chain's values copied to `into`, which holds room for them."""
    var at = Int(state[])
    var end = Int(state[unsafe_offset=2])
    while at != 0:
        var segment = _words_at(at)
        var fill = Int(segment[unsafe_offset=1])
        end -= fill
        unsafe_memcpy(
            dest=Pointer[UInt8, MutAnyOrigin](
                unsafe_from_address=Int(into) + 8 * end
            ),
            src=Pointer[UInt8, ImmutAnyOrigin](
                unsafe_from_address=at + 8 * _SEGMENT_HEAD
            ),
            count=8 * fill,
        )
        at = Int(segment[])


@fieldwise_init
struct AggregateFunction(Copyable, Movable):
    """One reduction's state in a group row: its kind, its first
    word after the keys, and what its values are. A state of all zero words
    is empty (`initialize`); `update`, `combine` and `finalize` are the
    `_update_states`, `combine` and `_Finalizer` of its kind."""

    var kind: Int
    var offset: Int
    # Minima and maxima compare as `Float64` when set, else as canonical
    # `Int64`.
    var floating: Bool
    # The state's type (a decimal sum's result type, for its precision).
    var dtype: DataType
    # A top-rows state's pair count and direction; a correlation's second
    # input.
    var k: Int
    var descending: Bool
    # A value state's input storage (`_VALUE_STRING`, `_VALUE_BOOL`,
    # `_VALUE_WIDE`, or an index into NUMERIC_DTYPES) and its value words.
    var storage: Int
    var width: Int

    def __init__(
        out self,
        kind: Int,
        offset: Int,
        floating: Bool,
        dtype: DataType,
        k: Int,
        descending: Bool,
        storage: Int = _VALUE_NONE,
    ):
        self.kind = kind
        self.offset = offset
        self.floating = floating
        self.dtype = dtype
        self.k = k
        self.descending = descending
        self.storage = storage
        self.width = 2 if (
            storage == _VALUE_STRING or storage == _VALUE_WIDE
        ) else 1

    def state_size(self) -> Int:
        """The state's words."""
        if self.kind == _K_TOP_ROWS:
            return 1 + 2 * self.k
        if self.kind == _K_COMOMENT:
            return 6
        if self.kind == _K_FIRST or self.kind == _K_LAST:
            return self.width + 2
        if self.kind == _K_LOGIC:
            return 1
        if (
            self.kind == _K_STATE
            or self.kind == _K_ORDERED
            or self.kind == _K_INT_SUM
            or self.kind == _K_VALUES
            or self.kind == _K_MOMENTS
            or self.kind == _K_DECIMAL_SUM
            or self.kind == _K_STRING_MIN
            or self.kind == _K_STRING_MAX
            or self.kind == _K_WIDE_MIN
            or self.kind == _K_WIDE_MAX
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

    def initialize(self, state: Int):
        """An empty state at address `state`: all zero words."""
        var at = _words_at(state)
        for w in range(self.state_size()):
            at[unsafe_offset=w] = 0

    def holds_strings(self) -> Bool:
        """Whether the state holds addresses of its input's string bytes,
        so the sink keeps that input alive."""
        return self.storage == _VALUE_STRING and (
            self.kind == _K_FIRST
            or self.kind == _K_LAST
            or self.kind == _K_STRING_MIN
            or self.kind == _K_STRING_MAX
        )

    def by_address(self) -> Bool:
        """Whether the row holds the state's address (`_K_STATE`,
        `_K_ORDERED`) rather than the state."""
        return self.kind == _K_STATE or self.kind == _K_ORDERED

    def destroy(self, state: Int):
        """Release state: no per-row destruction is needed. Value chains live in
        the sink's arena, and states by address in the sink's `Reducer`s
        and buffers, all freed with the sink."""
        pass


# Row words before the key words: hash, null bits, first row.
comptime _HEAD = 3


struct _Groups(Movable):
    """A partition's final groups in rows of 64-bit words: each group's
    hash, null bits and first row, key words (two per string or 128-bit
    column), then reduction state words."""

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
        unsafe_memcpy(
            dest=Pointer[UInt8, MutAnyOrigin](unsafe_from_address=self.at(g)),
            src=Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=address),
            count=8 * self.stride,
        )
        self.n += 1
        return g


# A value column's storage (`_ValueColumn`, `AggregateFunction.storage`):
# strings as views, Booleans, 128-bit values, nothing, or (from 0) a number
# of type NUMERIC_DTYPES[kind].
comptime _VALUE_STRING = -1
comptime _VALUE_BOOL = -2
comptime _VALUE_WIDE = -3
comptime _VALUE_NONE = -4


def _storage_kind(column: Series) -> Int:
    """A column's storage as a `_VALUE_` kind (a chunked column's by its
    first chunk); `_VALUE_NONE` for nested values."""
    if column.dtype().is_nested():
        return _VALUE_NONE
    var part = column.chunks()[
        0
    ].copy() if column.is_chunked() else column.copy()
    if part._data.isa[StringColumn]():
        return _VALUE_STRING
    if part._data.isa[BoolColumn]():
        return _VALUE_BOOL
    if part._data.isa[Column[Int128]]():
        return _VALUE_WIDE
    comptime for d in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[d]
        if part._data.isa[Column[Scalar[D]]]():
            return d
    return _VALUE_NONE


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
    var storage = _storage_kind(input)
    if (spec.op == COUNT or spec.op == NULL_COUNT) and storage != _VALUE_NONE:
        # A count reads only validity, whatever the values' type.
        return _K_COUNT if spec.op == COUNT else _K_NULLS
    if spec.op == N_UNIQUE:
        return _K_DISTINCT if storage != _VALUE_NONE else -1
    if input.dtype().is_categorical():
        return -1
    if (spec.op == FIRST or spec.op == LAST) and storage != _VALUE_NONE:
        return _K_FIRST if spec.op == FIRST else _K_LAST
    if (spec.op == ANY or spec.op == ALL) and storage == _VALUE_BOOL:
        return _K_LOGIC
    if spec.op == MIN or spec.op == MAX:
        if storage == _VALUE_BOOL:
            return _K_LOGIC
        if storage == _VALUE_STRING:
            return _K_STRING_MIN if spec.op == MIN else _K_STRING_MAX
        if input.dtype().is_decimal() and _decimal_part(input):
            return _K_WIDE_MIN if spec.op == MIN else _K_WIDE_MAX
    if (spec.op == SUM or spec.op == MEAN) and input.dtype().is_decimal():
        var state = spec.make(0).dtype
        if state.is_decimal() and _decimal_part(input):
            return _K_DECIMAL_SUM
        return -1
    var part = _numeric_part(input)
    if not part:
        return -1
    if spec.op == MEDIAN or spec.op == QUANTILE:
        return _K_VALUES
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


def _pair_kind(spec: ReducerSpec, x: Series, y: Series) -> Int:
    """The inline state kind of a correlation or covariance of two numeric
    columns (Pearson only: Spearman ranks every value), or -1."""
    if spec.op == CORR and spec.text == "spearman":
        return -1
    if not _numeric_part(x) or not _numeric_part(y):
        return -1
    return _K_COMOMENT


def _second_input(specs: List[ReducerSpec], s: Int) -> Int:
    """The index of reduction s's second input among the inputs: those of
    the two-input reductions follow the reductions' own, in order."""
    var at = len(specs)
    for r in range(s):
        if specs[r].op == CORR or specs[r].op == COV:
            at += 1
    return at


def states_supported(specs: List[ReducerSpec], inputs: List[Series]) -> Bool:
    """Whether every reduction's state fits inline in a group row."""
    for s in range(len(specs)):
        if specs[s].op == CORR or specs[s].op == COV:
            var at = _second_input(specs, s)
            if (
                at >= len(inputs)
                or _pair_kind(specs[s], inputs[s], inputs[at]) < 0
            ):
                return False
        elif _slot_kind(specs[s], inputs[s]) < 0:
            return False
    return True


def _address_kind(op: Int) -> Int:
    """The state by address of a reduction with no inline state:
    `_K_ORDERED` for one whose answer depends on the order of the group's
    rows (first, last, the position of an extreme), `_K_STATE` for any
    other a `Reducer` holds, or -1."""
    if op in [FIRST, LAST, ARG_MIN, ARG_MAX, IMPLODE]:
        return _K_ORDERED
    if op in [
        SUM,
        MEAN,
        MIN,
        MAX,
        COUNT,
        NULL_COUNT,
        STD,
        VAR,
        MEDIAN,
        QUANTILE,
        ANY,
        ALL,
        SKEW,
        KURTOSIS,
        PRODUCT,
        CORR,
        COV,
        MODE,
    ]:
        return _K_STATE
    return -1


def table_kind(specs: List[ReducerSpec], inputs: List[Series], s: Int) -> Int:
    """The state kind of reduction s in a group row: its inline kind, else
    its state by address (`_address_kind`), else -1."""
    var kind: Int
    if specs[s].op == CORR or specs[s].op == COV:
        var at = _second_input(specs, s)
        if at >= len(inputs):
            return -1
        kind = _pair_kind(specs[s], inputs[s], inputs[at])
    else:
        kind = _slot_kind(specs[s], inputs[s])
    if kind >= 0:
        return kind
    return _address_kind(specs[s].op)


def table_states_supported(
    specs: List[ReducerSpec], inputs: List[Series]
) -> Bool:
    """Whether the aggregate table holds every reduction's state for
    groups, inline or by address (`table_kind`)."""
    for s in range(len(specs)):
        if table_kind(specs, inputs, s) < 0:
            return False
    return True


@always_inline
def _words_at(address: Int) -> Pointer[UInt64, MutAnyOrigin]:
    return Pointer[UInt64, MutAnyOrigin](unsafe_from_address=address)


def _format_nulls(format: UnifiedVectorFormat, start: Int, count: Int) -> Bool:
    """Whether rows [start, start + count) of a vector may be null: never without a mask. A flat vector's mask is counted over
    those rows, so an all-valid bitmap (as Parquet columns carry) reads
    as none."""
    var validity = format.validity
    if validity.all_valid():
        return False
    if format.sel.is_set():
        return True
    var bits = Span[UInt8, ImmutAnyOrigin](
        unsafe_ptr=Pointer[UInt8, ImmutAnyOrigin](
            unsafe_from_address=validity.bits
        ),
        length=(validity.offset + start + count + 7) // 8,
    )
    return _count_set(bits, validity.offset + start, count) < count


def _update_counts(
    format: UnifiedVectorFormat,
    kind: Int,
    start: Int,
    count: Int,
    rows: Pointer[Int, _],
    base: Int,
):
    """Count a batch's valid (`_K_COUNT`) or null (`_K_NULLS`) rows
    [start, start + count) of a vector into their groups' state words,
    reading only its validity (count(ANY)'s scatter)."""
    if not _format_nulls(format, start, count):
        if kind == _K_COUNT:
            for k in range(count):
                _words_at(rows[unsafe_offset=k] + base)[] += 1
        return
    var sel = format.sel
    var validity = format.validity
    var counted = kind == _K_COUNT
    for k in range(count):
        if validity.row_is_valid_unsafe(sel.get_index(start + k)) == counted:
            _words_at(rows[unsafe_offset=k] + base)[] += 1


@always_inline
def _string_order(at: Int, length: Int, state: Int) -> Int:
    """The order of `length` bytes at `at` against the string_t held at
    `state` (negative, zero or positive): bytes compared as unsigned, then
    lengths, so a shorter equal prefix sorts first."""
    var held = _words_at(state)
    var size = Int(held[] & 0xFFFFFFFF)
    var bytes = state + 4 if size <= 12 else Int(held[unsafe_offset=1])
    var c = Int(
        unsafe_memcmp(
            Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=at),
            Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=bytes),
            min(length, size),
        )
    )
    if c != 0:
        return c
    return length - size


def _value_words(
    format: UnifiedVectorFormat,
    start: Int,
    count: Int,
    first_at: Int,
    second_at: Int,
    nulls_at: Int,
) raises:
    """A value state's words for rows [start, start + count) of a vector,
    at the given buffer addresses: a string_t's two words (its inline
    bytes past its length masked off), 1 for true and 2 for false, a
    128-bit value's halves, or a number's bits (a float's exactly, an
    integer as Int64 or UInt64). A null row's word is 0 and its null word
    nonzero. These are the key words `_vector_key` gives the same rows."""
    var first = Pointer[UInt64, MutAnyOrigin](unsafe_from_address=first_at)
    var second = Pointer[UInt64, MutAnyOrigin](unsafe_from_address=second_at)
    var nulls = Pointer[UInt64, MutAnyOrigin](unsafe_from_address=nulls_at)
    var sel = format.sel
    var validity = format.validity
    var has_nulls = _format_nulls(format, start, count)
    var physical = format.physical
    if physical == PHYSICAL_VARCHAR:
        var strings = Pointer[StringT, MutAnyOrigin](
            unsafe_from_address=format.data
        )
        for k in range(count):
            var i = sel.get_index(start + k)
            if has_nulls and not validity.row_is_valid_unsafe(i):
                first[unsafe_offset=k] = 0
                second[unsafe_offset=k] = 0
                nulls[unsafe_offset=k] = 1
                continue
            var s = strings[unsafe_offset=i]
            var length = Int(s.length)
            var prefix = UInt64(s.prefix)
            var rest = s.rest
            if length <= STRING_INLINE_BYTES:
                if length < 4:
                    prefix &= (UInt64(1) << UInt64(8 * length)) - 1
                var keep = 8 * max(length - 4, 0)
                rest &= (
                    UInt64(1) << UInt64(keep)
                ) - 1 if keep < 64 else UInt64.MAX
            first[unsafe_offset=k] = UInt64(length) | (prefix << 32)
            second[unsafe_offset=k] = rest
        return
    if physical == PHYSICAL_BOOL:
        var data = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=format.data)
        for k in range(count):
            var i = sel.get_index(start + k)
            if has_nulls and not validity.row_is_valid_unsafe(i):
                first[unsafe_offset=k] = 0
                nulls[unsafe_offset=k] = 1
                continue
            first[unsafe_offset=k] = UInt64(1) if data[
                unsafe_offset=i
            ] != 0 else UInt64(2)
        return
    if physical == PHYSICAL_INT128:
        var data = format.values[DType.int128]()
        for k in range(count):
            var i = sel.get_index(start + k)
            if has_nulls and not validity.row_is_valid_unsafe(i):
                first[unsafe_offset=k] = 0
                second[unsafe_offset=k] = 0
                nulls[unsafe_offset=k] = 1
                continue
            var value = bitcast[DType.uint128](data[unsafe_offset=i])
            first[unsafe_offset=k] = UInt64(value)
            second[unsafe_offset=k] = UInt64(value >> 64)
        return
    comptime for d in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[d]
        if physical == physical_of[D]():
            var data = format.values[D]()
            for k in range(count):
                var i = sel.get_index(start + k)
                if has_nulls and not validity.row_is_valid_unsafe(i):
                    first[unsafe_offset=k] = 0
                    nulls[unsafe_offset=k] = 1
                    continue
                var value = data[unsafe_offset=i]
                comptime if D.is_floating_point():
                    first[unsafe_offset=k] = bitcast[DType.uint64](
                        Float64(value)
                    )
                elif D.is_unsigned():
                    first[unsafe_offset=k] = UInt64(value)
                else:
                    first[unsafe_offset=k] = bitcast[DType.uint64](Int64(value))
            return
    raise Error("aggregate table: unsupported value vector")


def _update_values(
    slot: AggregateFunction,
    format: UnifiedVectorFormat,
    start: Int,
    count: Int,
    rows: Pointer[Int, _],
    base: Int,
    first: Int,
    scratch: Int,
) raises:
    """A batch into its groups' value states (first, last, a string's
    minimum or maximum, logic flags). The rows' value words
    (`_value_words`) are written into `scratch` (three vectors of words:
    first words, second words, null flags), then each row's state takes
    them or keeps its own."""
    var first_words = _words_at(scratch)
    var second_words = _words_at(scratch + 8 * _BATCH)
    var nulls = _words_at(scratch + 16 * _BATCH)
    for k in range(count):
        nulls[unsafe_offset=k] = 0
    _value_words(
        format,
        start,
        count,
        scratch,
        scratch + 8 * _BATCH,
        scratch + 16 * _BATCH,
    )
    var kind = slot.kind
    if kind == _K_LOGIC:
        # A Boolean's word is 1 for true and 2 for false: the flags.
        for k in range(count):
            var at = _words_at(rows[unsafe_offset=k] + base)
            at[] |= (
                UInt64(4) if nulls[unsafe_offset=k]
                != 0 else first_words[unsafe_offset=k]
            )
        return
    if kind == _K_FIRST or kind == _K_LAST:
        var width = slot.width
        var last = kind == _K_LAST
        for k in range(count):
            var at = _words_at(rows[unsafe_offset=k] + base)
            if not last and at[unsafe_offset=width + 1] != 0:
                continue
            at[] = first_words[unsafe_offset=k]
            if width == 2:
                at[unsafe_offset=1] = second_words[unsafe_offset=k]
            at[unsafe_offset=width] = UInt64(first + start + k)
            at[unsafe_offset=width + 1] = UInt64(1) if nulls[
                unsafe_offset=k
            ] != 0 else UInt64(3)
        return
    # A string's minimum or maximum: a row replaces the held value only when it
    # orders before (after) it.
    var is_max = kind == _K_STRING_MAX
    var sel = format.sel
    for k in range(count):
        if nulls[unsafe_offset=k] != 0:
            continue
        var state = rows[unsafe_offset=k] + base
        var at = _words_at(state)
        if at[unsafe_offset=2] != 0:
            var entry = format.data + 16 * sel.get_index(start + k)
            var length = Int(
                Pointer[StringT, MutAnyOrigin](
                    unsafe_from_address=entry
                )[].length
            )
            var order = _string_order(string_data(entry), length, state)
            if (is_max and order <= 0) or (not is_max and order >= 0):
                continue
        at[] = first_words[unsafe_offset=k]
        at[unsafe_offset=1] = second_words[unsafe_offset=k]
        at[unsafe_offset=2] = 1


def _update_wide[
    D: DType
](
    slot: AggregateFunction,
    format: UnifiedVectorFormat,
    start: Int,
    count: Int,
    rows: Pointer[Int, _],
    base: Int,
):
    """A batch's decimal values (unscaled, at their storage width) into
    their groups' 128-bit minima or maxima."""
    var values = format.values[D]()
    var sel = format.sel
    var validity = format.validity
    var nulls = _format_nulls(format, start, count)
    var is_max = slot.kind == _K_WIDE_MAX
    for k in range(count):
        var i = sel.get_index(start + k)
        if nulls and not validity.row_is_valid_unsafe(i):
            continue
        var at = _words_at(rows[unsafe_offset=k] + base)
        var x = values[unsafe_offset=i].cast[DType.int128]()
        if at[unsafe_offset=2] != 0:
            var best = bitcast[DType.int128](
                (UInt128(at[unsafe_offset=1]) << 64)
                | UInt128(at[unsafe_offset=0])
            )
            if (is_max and x <= best) or (not is_max and x >= best):
                continue
        var word = bitcast[DType.uint128](x)
        at[unsafe_offset=0] = UInt64(word)
        at[unsafe_offset=1] = UInt64(word >> 64)
        at[unsafe_offset=2] = 1


def _update_slot[
    D: DType, selected: Bool
](
    slot: AggregateFunction,
    format: UnifiedVectorFormat,
    start: Int,
    count: Int,
    rows: Pointer[Int, _],
    base: Int,
    arena: Pointer[_ValueArena, MutAnyOrigin],
    discard: Int,
):
    """Update group state words with a batch's values of `D`, rows [start, start
    + count) of a vector. `base` is the byte offset in each group row, and
    `rows` holds the group-row addresses. Each state kind has its own loop, and
    a vector without nulls takes one that reads no validity."""
    var nulls = _format_nulls(format, start, count)
    var kind = slot.kind
    if kind == _K_VALUES:
        _scatter_kind[D, selected, _K_VALUES](
            format, start, count, rows, base, arena, discard, nulls
        )
    elif kind == _K_INT_SUM:
        _scatter_kind[D, selected, _K_INT_SUM](
            format, start, count, rows, base, arena, discard, nulls
        )
    elif kind == _K_NARROW_SUM:
        _scatter_kind[D, selected, _K_NARROW_SUM](
            format, start, count, rows, base, arena, discard, nulls
        )
    elif kind == _K_FLOAT_SUM:
        _scatter_kind[D, selected, _K_FLOAT_SUM](
            format, start, count, rows, base, arena, discard, nulls
        )
    elif kind == _K_MOMENTS:
        _scatter_kind[D, selected, _K_MOMENTS](
            format, start, count, rows, base, arena, discard, nulls
        )
    elif kind == _K_MAX:
        _scatter_kind[D, selected, _K_MAX](
            format, start, count, rows, base, arena, discard, nulls
        )
    else:
        _scatter_kind[D, selected, _K_MIN](
            format, start, count, rows, base, arena, discard, nulls
        )


@always_inline
def _scatter_kind[
    D: DType, selected: Bool, kind: Int
](
    format: UnifiedVectorFormat,
    start: Int,
    count: Int,
    rows: Pointer[Int, _],
    base: Int,
    arena: Pointer[_ValueArena, MutAnyOrigin],
    discard: Int,
    nulls: Bool,
):
    if nulls:
        _scatter[D, selected, kind, True](
            format, start, count, rows, base, arena, discard
        )
    else:
        _scatter[D, selected, kind, False](
            format, start, count, rows, base, arena, discard
        )


def _scatter[
    D: DType, selected: Bool, kind: Int, nulls: Bool
](
    format: UnifiedVectorFormat,
    start: Int,
    count: Int,
    rows: Pointer[Int, _],
    base: Int,
    arena: Pointer[_ValueArena, MutAnyOrigin],
    discard: Int,
):
    """One state kind's scatter loop: each row's value into the state at its
    group row's address, a null skipped only when the vector may hold one
    (`nulls`)."""
    var values = format.values[D]()
    var sel = format.sel
    var validity = format.validity
    for k in range(count):
        var i: Int
        comptime if selected:
            i = sel.get_index_unsafe(start + k)
        else:
            i = start + k
        comptime if nulls:
            if not validity.row_is_valid_unsafe(i):
                continue
        var value = values[unsafe_offset=i]
        var at = _words_at(rows[unsafe_offset=k] + base)
        comptime if kind == _K_VALUES:
            # Rows a filter dropped keep no value.
            if rows[unsafe_offset=k] != discard:
                _values_push(at, Float64(value), arena)
        elif kind == _K_INT_SUM:
            var total = (UInt128(at[unsafe_offset=1]) << 64) | UInt128(
                at[unsafe_offset=0]
            )
            total += bitcast[DType.uint128](value.cast[DType.int128]())
            at[unsafe_offset=0] = UInt64(total)
            at[unsafe_offset=1] = UInt64(total >> 64)
            at[unsafe_offset=2] += 1
        elif kind == _K_NARROW_SUM:
            at[unsafe_offset=0] += bitcast[DType.uint64](
                value.cast[DType.int64]()
            )
            at[unsafe_offset=1] += 1
        elif kind == _K_FLOAT_SUM:
            at[unsafe_offset=0] = bitcast[DType.uint64](
                bitcast[DType.float64](at[unsafe_offset=0]) + Float64(value)
            )
            at[unsafe_offset=1] += 1
        elif kind == _K_MOMENTS:
            # Every integer as its value (an ordering key would shift a
            # UInt64 by 2^63 and lose its spread).
            var x = value.cast[DType.float64]()
            var n = at[unsafe_offset=0] + 1
            var mean = bitcast[DType.float64](at[unsafe_offset=1])
            var delta = x - mean
            mean += delta / Float64(n)
            var m2 = bitcast[DType.float64](at[unsafe_offset=2]) + delta * (
                x - mean
            )
            at[unsafe_offset=0] = n
            at[unsafe_offset=1] = bitcast[DType.uint64](mean)
            at[unsafe_offset=2] = bitcast[DType.uint64](m2)
        else:
            comptime is_max = kind == _K_MAX
            var seen = at[unsafe_offset=1] & 1 != 0
            comptime if D.is_floating_point():
                var x = Float64(value)
                if isnan(x):
                    at[unsafe_offset=1] |= 2
                    continue
                var best = bitcast[DType.float64](at[unsafe_offset=0])
                var better: Bool
                comptime if is_max:
                    better = x > best
                else:
                    better = x < best
                if not seen or better:
                    at[unsafe_offset=0] = bitcast[DType.uint64](x)
                    at[unsafe_offset=1] |= 1
            else:
                # Ties keep the earlier value, as `_extreme` does.
                var x = _canonical_int[D](value)
                var best = bitcast[DType.int64](at[unsafe_offset=0])
                var better: Bool
                comptime if is_max:
                    better = x > best
                else:
                    better = x < best
                if not seen or better:
                    at[unsafe_offset=0] = bitcast[DType.uint64](x)
                    at[unsafe_offset=1] |= 1


def _decimal_part(column: Series) -> Bool:
    """Whether a column is a decimal stored as 32-, 64- or 128-bit
    integers."""
    if not column.dtype().is_decimal():
        return False
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
    slot: AggregateFunction,
    format: UnifiedVectorFormat,
    start: Int,
    count: Int,
    rows: Pointer[Int, _],
    base: Int,
) raises:
    """A batch's decimal values (unscaled, at their storage width) into
    their groups' 128-bit totals and counts, each total held to the
    state's precision."""
    var values = format.values[D]()
    var sel = format.sel
    var validity = format.validity
    var nulls = _format_nulls(format, start, count)
    var limit = precision_limit(slot.dtype.precision())
    for k in range(count):
        var i = sel.get_index(start + k)
        if nulls and not validity.row_is_valid_unsafe(i):
            continue
        var at = _words_at(rows[unsafe_offset=k] + base)
        var total = bitcast[DType.int128](
            (UInt128(at[unsafe_offset=1]) << 64) | UInt128(at[unsafe_offset=0])
        )
        total = check_limit(
            total + values[unsafe_offset=i].cast[DType.int128](),
            limit,
            slot.dtype,
        )
        var word = bitcast[DType.uint128](total)
        at[unsafe_offset=0] = UInt64(word)
        at[unsafe_offset=1] = UInt64(word >> 64)
        at[unsafe_offset=2] += 1


def _load_floats(
    format: UnifiedVectorFormat,
    start: Int,
    count: Int,
    values_at: Int,
    valid_at: Int,
) raises:
    """Rows [start, start + count) of a numeric vector as Float64 and
    validity, at the addresses `values_at` and `valid_at`."""
    var values = Pointer[Float64, MutAnyOrigin](unsafe_from_address=values_at)
    var valid = Pointer[Bool, MutAnyOrigin](unsafe_from_address=valid_at)
    var sel = format.sel
    var validity = format.validity
    var nulls = _format_nulls(format, start, count)
    comptime for t in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[t]
        if format.physical == physical_of[D]():
            var data = format.values[D]()
            for k in range(count):
                var i = sel.get_index(start + k)
                values[unsafe_offset=k] = data[unsafe_offset=i].cast[
                    DType.float64
                ]()
                valid[
                    unsafe_offset=k
                ] = not nulls or validity.row_is_valid_unsafe(i)
            return
    raise Error("aggregate table: unsupported pair input")


def _update_comoment(
    x: UnifiedVectorFormat,
    y: UnifiedVectorFormat,
    start: Int,
    count: Int,
    rows: Pointer[Int, _],
    base: Int,
    scratch: Int,
) raises:
    """A batch's (x, y) pairs into their groups' co-moment states; a pair
    with a null on either side is dropped.
    The pairs are loaded into `scratch` (three vectors of words: x values,
    y values, then both validity flags), allocated once per sink."""
    var xs = Pointer[Float64, MutAnyOrigin](unsafe_from_address=scratch)
    var ys = Pointer[Float64, MutAnyOrigin](
        unsafe_from_address=scratch + 8 * _BATCH
    )
    var xv = Pointer[Bool, MutAnyOrigin](
        unsafe_from_address=scratch + 16 * _BATCH
    )
    var yv = Pointer[Bool, MutAnyOrigin](
        unsafe_from_address=scratch + 17 * _BATCH
    )
    _load_floats(x, start, count, Int(xs), Int(xv))
    _load_floats(y, start, count, Int(ys), Int(yv))
    for k in range(count):
        if not (xv[unsafe_offset=k] and yv[unsafe_offset=k]):
            continue
        var at = _words_at(rows[unsafe_offset=k] + base)
        var n = at[unsafe_offset=0] + 1
        var mean_x = bitcast[DType.float64](at[unsafe_offset=1])
        var mean_y = bitcast[DType.float64](at[unsafe_offset=2])
        var dx = xs[unsafe_offset=k] - mean_x
        mean_x += dx / Float64(n)
        var dy = ys[unsafe_offset=k] - mean_y
        mean_y += dy / Float64(n)
        at[unsafe_offset=0] = n
        at[unsafe_offset=1] = bitcast[DType.uint64](mean_x)
        at[unsafe_offset=2] = bitcast[DType.uint64](mean_y)
        at[unsafe_offset=3] = bitcast[DType.uint64](
            bitcast[DType.float64](at[unsafe_offset=3])
            + dx * (ys[unsafe_offset=k] - mean_y)
        )
        at[unsafe_offset=4] = bitcast[DType.uint64](
            bitcast[DType.float64](at[unsafe_offset=4])
            + dx * (xs[unsafe_offset=k] - mean_x)
        )
        at[unsafe_offset=5] = bitcast[DType.uint64](
            bitcast[DType.float64](at[unsafe_offset=5])
            + dy * (ys[unsafe_offset=k] - mean_y)
        )


def _update_states(
    slots: List[AggregateFunction],
    inputs: List[UnifiedVectorFormat],
    start: Int,
    count: Int,
    rows: Pointer[Int, _],
    states_at: Int,
    first: Int,
    arena: Pointer[_ValueArena, MutAnyOrigin],
    discard: Int,
    scratch: Int,
    only: Int = -1,
    skip: List[Bool] = List[Bool](),
) raises:
    """Every reduction's `update` for a batch: `inputs` are the reductions' vectors,
    rows [start, start + count) of each; `rows` the addresses of the
    rows' groups, whose states start at word `states_at`; `first` the
    source row of the vectors' first, rows numbered on from it; `scratch`
    three vectors of words for value states. With `only`, each row adds
    one to that reduction's state alone (a distinct table's groups into
    their distinct counts). Reductions marked in `skip` (the clustered
    update's, `update_clustered`) are left alone."""
    if only >= 0:
        var base = 8 * (states_at + slots[only].offset)
        for k in range(count):
            _words_at(rows[unsafe_offset=k] + base)[] += 1
        return
    for s in range(len(slots)):
        ref slot = slots[s]
        var base = 8 * (states_at + slot.offset)
        if slot.kind == _K_DISTINCT or slot.by_address():
            # States by address are the sink's (`AggregateSink`).
            continue
        if len(skip) > 0 and skip[s]:
            continue
        if slot.kind == _K_LEN:
            for k in range(count):
                _words_at(rows[unsafe_offset=k] + base)[] += 1
            continue
        var format = inputs[s]
        var physical = format.physical
        if slot.kind == _K_COUNT or slot.kind == _K_NULLS:
            _update_counts(format, slot.kind, start, count, rows, base)
            continue
        if (
            slot.kind == _K_FIRST
            or slot.kind == _K_LAST
            or slot.kind == _K_STRING_MIN
            or slot.kind == _K_STRING_MAX
            or slot.kind == _K_LOGIC
        ):
            _update_values(
                slot, format, start, count, rows, base, first, scratch
            )
            continue
        if slot.kind == _K_WIDE_MIN or slot.kind == _K_WIDE_MAX:
            if physical == PHYSICAL_INT64:
                _update_wide[DType.int64](
                    slot, format, start, count, rows, base
                )
            elif physical == PHYSICAL_INT32:
                _update_wide[DType.int32](
                    slot, format, start, count, rows, base
                )
            elif physical == PHYSICAL_INT128:
                _update_wide[DType.int128](
                    slot, format, start, count, rows, base
                )
            else:
                raise Error("aggregate table: unsupported decimal input")
            continue
        if slot.kind == _K_COMOMENT:
            _update_comoment(
                format, inputs[slot.k], start, count, rows, base, scratch
            )
            continue
        if slot.kind == _K_TOP_ROWS:
            var placed = False
            comptime for t in range(len(NUMERIC_DTYPES)):
                comptime D = NUMERIC_DTYPES[t]
                if not placed and physical == physical_of[D]():
                    _update_top_rows[D](
                        slot, format, start, count, rows, base, first
                    )
                    placed = True
            if not placed:
                raise Error("aggregate table: unsupported top-rows input")
            continue
        if slot.kind == _K_DECIMAL_SUM:
            if physical == PHYSICAL_INT64:
                _update_decimal[DType.int64](
                    slot, format, start, count, rows, base
                )
            elif physical == PHYSICAL_INT32:
                _update_decimal[DType.int32](
                    slot, format, start, count, rows, base
                )
            elif physical == PHYSICAL_INT128:
                _update_decimal[DType.int128](
                    slot, format, start, count, rows, base
                )
            else:
                raise Error("aggregate table: unsupported decimal input")
            continue
        var done = False
        var selected = format.sel.is_set()
        comptime for t in range(len(NUMERIC_DTYPES)):
            comptime D = NUMERIC_DTYPES[t]
            if not done and physical == physical_of[D]():
                if selected:
                    _update_slot[D, True](
                        slot, format, start, count, rows, base, arena, discard
                    )
                else:
                    _update_slot[D, False](
                        slot, format, start, count, rows, base, arena, discard
                    )
                done = True
        if not done:
            raise Error("aggregate table: unsupported reduction input")


@always_inline
def _retain_top_rows(
    state: Pointer[UInt64, MutAnyOrigin], k: Int, key: UInt64, row: UInt64
):
    """Retain the best k pairs in relocatable inline aggregate storage.

    At most TOP_ROWS_INLINE pairs live here. Owning a heap in this raw state
    would require destruction and relocation hooks, so sort its bounded span.
    """
    if k <= 0:
        return
    var count = Int(state[])
    if count == k:
        var last_key = state[unsafe_offset=2 * k - 1]
        var last_row = state[unsafe_offset=2 * k]
        if key > last_key or (key == last_key and row >= last_row):
            return
        count -= 1
    state[unsafe_offset=2 * count + 1] = key
    state[unsafe_offset=2 * count + 2] = row
    count += 1
    state[] = UInt64(count)
    var pairs = Span(
        unsafe_ptr=Pointer[Array[UInt64, 2], MutAnyOrigin](
            unsafe_from_address=Int(state.unsafe_offset(1))
        ),
        length=count,
    )

    def less(a: Array[UInt64, 2], b: Array[UInt64, 2]) -> Bool:
        return a[0] < b[0] or (a[0] == b[0] and a[1] < b[1])

    sort(pairs, less)


def _update_top_rows[
    D: DType
](
    slot: AggregateFunction,
    format: UnifiedVectorFormat,
    start: Int,
    count: Int,
    rows: Pointer[Int, _],
    base: Int,
    first: Int,
):
    """A batch's values into their groups' top-rows states; a null value
    has no rank and is not kept."""
    var values = format.values[D]()
    var sel = format.sel
    var validity = format.validity
    var nulls = _format_nulls(format, start, count)
    for k in range(count):
        var i = sel.get_index(start + k)
        if nulls and not validity.row_is_valid_unsafe(i):
            continue
        _retain_top_rows(
            _words_at(rows[unsafe_offset=k] + base),
            slot.k,
            _key[D](values[unsafe_offset=i], slot.descending),
            UInt64(first + start + k),
        )


def combine(slots: List[AggregateFunction], into: Int, other: Int) raises:
    """Every reduction's `combine`: group row
    `other`'s states folded into group row `into` (both the addresses of
    their first state word's row)."""
    var a = _words_at(into)
    var b = _words_at(other)
    for s in range(len(slots)):
        var w = slots[s].offset
        var kind = slots[s].kind
        if kind == _K_LOGIC:
            a[unsafe_offset=w] |= b[unsafe_offset=w]
        elif kind == _K_STATE or kind == _K_ORDERED:
            # The other group's chain of states, then the rest of this
            # one's, after this one's first: linked, never copied. The
            # other row stays where it is (a sink's block) until the finish
            # reads the chain.
            if b[unsafe_offset=w] == 0:
                continue
            if a[unsafe_offset=w] == 0:
                for i in range(3):
                    a[unsafe_offset=w + i] = b[unsafe_offset=w + i]
                continue
            var tail = other + 8 * w
            while _words_at(tail)[unsafe_offset=2] != 0:
                tail = Int(_words_at(tail)[unsafe_offset=2])
            _words_at(tail)[unsafe_offset=2] = a[unsafe_offset=w + 2]
            a[unsafe_offset=w + 2] = UInt64(other + 8 * w)
        elif kind == _K_FIRST or kind == _K_LAST:
            # The value of the earlier (first) or later (last) row.
            var width = slots[s].width
            var theirs = b[unsafe_offset=w + width + 1]
            if theirs == 0:
                continue
            var mine = a[unsafe_offset=w + width + 1]
            var row = b[unsafe_offset=w + width]
            var held = a[unsafe_offset=w + width]
            if (
                mine == 0
                or (kind == _K_FIRST and row < held)
                or (kind == _K_LAST and row > held)
            ):
                for i in range(width + 2):
                    a[unsafe_offset=w + i] = b[unsafe_offset=w + i]
        elif kind == _K_STRING_MIN or kind == _K_STRING_MAX:
            if b[unsafe_offset=w + 2] == 0:
                continue
            if a[unsafe_offset=w + 2] != 0:
                var source = other + 8 * w
                var size = Int(b[unsafe_offset=w] & 0xFFFFFFFF)
                var bytes = source + 4 if size <= 12 else Int(
                    b[unsafe_offset=w + 1]
                )
                var order = _string_order(bytes, size, into + 8 * w)
                if (kind == _K_STRING_MAX and order <= 0) or (
                    kind == _K_STRING_MIN and order >= 0
                ):
                    continue
            for i in range(3):
                a[unsafe_offset=w + i] = b[unsafe_offset=w + i]
        elif kind == _K_WIDE_MIN or kind == _K_WIDE_MAX:
            if b[unsafe_offset=w + 2] == 0:
                continue
            if a[unsafe_offset=w + 2] != 0:
                var x = bitcast[DType.int128](
                    (UInt128(b[unsafe_offset=w + 1]) << 64)
                    | UInt128(b[unsafe_offset=w])
                )
                var y = bitcast[DType.int128](
                    (UInt128(a[unsafe_offset=w + 1]) << 64)
                    | UInt128(a[unsafe_offset=w])
                )
                if (kind == _K_WIDE_MAX and x <= y) or (
                    kind == _K_WIDE_MIN and x >= y
                ):
                    continue
            for i in range(3):
                a[unsafe_offset=w + i] = b[unsafe_offset=w + i]
        elif kind == _K_TOP_ROWS:
            var held = Int(b[unsafe_offset=w])
            var into_state = Pointer[UInt64, MutAnyOrigin](
                unsafe_from_address=into + 8 * w
            )
            for p in range(held):
                _retain_top_rows(
                    into_state,
                    slots[s].k,
                    b[unsafe_offset=w + 1 + 2 * p],
                    b[unsafe_offset=w + 2 + 2 * p],
                )
        elif kind == _K_COMOMENT:
            var m = b[unsafe_offset=w]
            if m == 0:
                continue
            var n = a[unsafe_offset=w]
            if n == 0:
                for i in range(6):
                    a[unsafe_offset=w + i] = b[unsafe_offset=w + i]
                continue
            var wa = Float64(n)
            var wb = Float64(m)
            var total = wa + wb
            var dx = bitcast[DType.float64](b[unsafe_offset=w + 1]) - bitcast[
                DType.float64
            ](a[unsafe_offset=w + 1])
            var dy = bitcast[DType.float64](b[unsafe_offset=w + 2]) - bitcast[
                DType.float64
            ](a[unsafe_offset=w + 2])
            var cxy = (
                bitcast[DType.float64](a[unsafe_offset=w + 3])
                + bitcast[DType.float64](b[unsafe_offset=w + 3])
                + dx * dy * wa * wb / total
            )
            var m2x = (
                bitcast[DType.float64](a[unsafe_offset=w + 4])
                + bitcast[DType.float64](b[unsafe_offset=w + 4])
                + dx * dx * wa * wb / total
            )
            var m2y = (
                bitcast[DType.float64](a[unsafe_offset=w + 5])
                + bitcast[DType.float64](b[unsafe_offset=w + 5])
                + dy * dy * wa * wb / total
            )
            a[unsafe_offset=w + 1] = bitcast[DType.uint64](
                bitcast[DType.float64](a[unsafe_offset=w + 1]) + dx * wb / total
            )
            a[unsafe_offset=w + 2] = bitcast[DType.uint64](
                bitcast[DType.float64](a[unsafe_offset=w + 2]) + dy * wb / total
            )
            a[unsafe_offset=w + 3] = bitcast[DType.uint64](cxy)
            a[unsafe_offset=w + 4] = bitcast[DType.uint64](m2x)
            a[unsafe_offset=w + 5] = bitcast[DType.uint64](m2y)
            a[unsafe_offset=w] = n + m
        elif kind == _K_VALUES:
            _values_append(
                Pointer[UInt64, MutAnyOrigin](unsafe_from_address=into + 8 * w),
                other + 8 * w,
            )
        elif kind <= _K_NULLS or kind == _K_DISTINCT:
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


def _unload(
    slot: AggregateFunction,
    groups: _Groups,
    mut reducer: Reducer,
    lo: Int,
    hi: Int,
) raises:
    """A reduction's states of groups [lo, hi), from their rows into a
    `Reducer` sized for every group, which finishes them as every other
    route does."""
    if slot.kind == _K_STATE:
        _unload_states(slot, groups, reducer, lo, hi)
        return
    if slot.kind == _K_ORDERED:
        _replay_ordered(slot, groups, reducer, lo, hi)
        return
    var at = groups.rows.unsafe_ptr()
    var stride = groups.stride
    var w0 = groups.states_at + slot.offset
    var kind = slot.kind
    for g in range(lo, hi):
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
        elif kind == _K_COMOMENT:
            reducer.comoments[g].count = bitcast[DType.int64](
                at[unsafe_offset=w]
            )
            reducer.comoments[g].mean_x = bitcast[DType.float64](
                at[unsafe_offset=w + 1]
            )
            reducer.comoments[g].mean_y = bitcast[DType.float64](
                at[unsafe_offset=w + 2]
            )
            reducer.comoments[g].cxy = bitcast[DType.float64](
                at[unsafe_offset=w + 3]
            )
            reducer.comoments[g].m2x = bitcast[DType.float64](
                at[unsafe_offset=w + 4]
            )
            reducer.comoments[g].m2y = bitcast[DType.float64](
                at[unsafe_offset=w + 5]
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


struct _OrderedBuffer(Movable):
    """A worker's buffered input rows of one order-sensitive reduction: each row's value, its group's ordinal among this buffer's
    groups and its sequence number (its place in the input). `prepare`
    makes the values one column and indexes the rows by group, once the
    worker's run is over."""

    var batches: List[Series]
    var ordinals: List[Int]
    var sequences: List[Int]
    var groups: Int
    var values: Optional[Series]
    # Each group's rows, in input order: those of ordinal o are
    # rows[starts[o]:starts[o + 1]].
    var starts: List[Int]
    var rows: List[Int]

    def __init__(out self):
        self.batches = List[Series]()
        self.ordinals = List[Int]()
        self.sequences = List[Int]()
        self.groups = 0
        self.values = None
        self.starts = List[Int]()
        self.rows = List[Int]()

    def add(
        mut self, batch: Series, ordinals: List[Int], first: Int, groups: Int
    ) raises:
        """A batch's rows, the k-th of group ordinal `ordinals[k]` (-1 for a
        row that joins no group) and sequence number `first + k`."""
        var kept = List[Int]()
        for k in range(len(ordinals)):
            if ordinals[k] < 0:
                continue
            kept.append(k)
            self.ordinals.append(ordinals[k])
            self.sequences.append(first + k)
        if len(kept) == len(ordinals):
            self.batches.append(batch.copy())
        elif len(kept) > 0:
            self.batches.append(batch.take(kept))
        self.groups = groups

    def prepare(mut self) raises:
        """The values as one column and the rows by group (a stable
        counting sort, so each group's rows stay in input order)."""
        if self.values or len(self.batches) == 0:
            return
        self.values = (
            Series._from_chunks(self.batches).rechunk() if len(self.batches)
            > 1 else self.batches[0].copy()
        )
        self.batches = List[Series]()
        self.starts = List[Int](length=self.groups + 1, fill=0)
        for o in self.ordinals:
            self.starts[o + 1] += 1
        for g in range(self.groups):
            self.starts[g + 1] += self.starts[g]
        var place = self.starts.copy()
        self.rows = List[Int](length=len(self.ordinals), fill=0)
        for r in range(len(self.ordinals)):
            var o = self.ordinals[r]
            self.rows[place[o]] = r
            place[o] += 1


def reducer_input(reducer: Reducer, chunk: Series) raises -> Series:
    """A batch as `reducer` reads it: categorical labels as strings when
    its type is not categorical (the eager evaluator's `_feed` does the
    same)."""
    if chunk.dtype().is_categorical() and not reducer.logical.is_categorical():
        return decode(chunk)
    return chunk.copy()


@always_inline
def _owner_index(mut owners: List[Int], address: Int) -> Int:
    """The index of `address` among `owners` (a sink's each), added when
    new: a job's groups have at most one owner per worker."""
    for o in range(len(owners)):
        if owners[o] == address:
            return o
    owners.append(address)
    return len(owners) - 1


def _unload_states(
    slot: AggregateFunction,
    groups: _Groups,
    mut reducer: Reducer,
    lo: Int,
    hi: Int,
) raises:
    """States by address (`_K_STATE`) of groups [lo, hi): every state of a
    group's chain merged into the group's state in `reducer`, owner by
    owner."""
    var owners = List[Int]()
    var sources = List[List[Int]]()
    var targets = List[List[Int]]()
    var base = 8 * (groups.states_at + slot.offset)
    for g in range(lo, hi):
        var at = groups.at(g) + base
        while at != 0:
            var state = _words_at(at)
            if state[] != 0:
                var o = _owner_index(owners, Int(state[]))
                if o == len(sources):
                    sources.append(List[Int]())
                    targets.append(List[Int]())
                sources[o].append(Int(state[unsafe_offset=1]) - 1)
                targets[o].append(g)
            at = Int(state[unsafe_offset=2])
    for o in range(len(owners)):
        reducer.merge(
            Pointer[Reducer, MutAnyOrigin](unsafe_from_address=owners[o])[],
            groups=targets[o],
            sources=sources[o],
        )


def _ordered_rows(
    slot: AggregateFunction,
    groups: _Groups,
    lo: Int,
    hi: Int,
    mut owners: List[Int],
    mut owner_of: List[Int],
    mut positions: List[Int],
    mut targets: List[Int],
):
    """The buffered rows of groups [lo, hi) (`_K_ORDERED`), every one
    of each group's chain, sorted by sequence number: each row's buffer
    (an index into `owners`), its position in that buffer's values and
    its group."""
    var keyed = List[UInt128]()
    var found_owner = List[Int]()
    var found_row = List[Int]()
    var found_group = List[Int]()
    var base = 8 * (groups.states_at + slot.offset)
    for g in range(lo, hi):
        var at = groups.at(g) + base
        while at != 0:
            var state = _words_at(at)
            if state[] != 0:
                var o = _owner_index(owners, Int(state[]))
                ref buffer = Pointer[_OrderedBuffer, MutAnyOrigin](
                    unsafe_from_address=Int(state[])
                )[]
                var ordinal = Int(state[unsafe_offset=1]) - 1
                for r in range(
                    buffer.starts[ordinal], buffer.starts[ordinal + 1]
                ):
                    var row = buffer.rows[r]
                    keyed.append(
                        (UInt128(buffer.sequences[row]) << 64)
                        | UInt128(len(found_row))
                    )
                    found_owner.append(o)
                    found_row.append(row)
                    found_group.append(g)
            at = Int(state[unsafe_offset=2])
    sort(keyed)
    for key in keyed:
        var e = Int(UInt64(key & 0xFFFFFFFFFFFFFFFF))
        owner_of.append(found_owner[e])
        positions.append(found_row[e])
        targets.append(found_group[e])


def _gather_rows(
    owners: List[Int], owner_of: List[Int], positions: List[Int]
) raises -> Series:
    """The values of buffered rows, the k-th at `positions[k]` of buffer
    `owners[owner_of[k]]`, in that order: gathered buffer by buffer, then
    put in order when they come from more than one."""
    var picks = List[List[Int]](capacity=len(owners))
    for _ in range(len(owners)):
        picks.append(List[Int]())
    for k in range(len(positions)):
        picks[owner_of[k]].append(positions[k])
    var pieces = List[Series](capacity=len(owners))
    for o in range(len(owners)):
        ref buffer = Pointer[_OrderedBuffer, MutAnyOrigin](
            unsafe_from_address=owners[o]
        )[]
        pieces.append(buffer.values.value().take(picks[o]))
    if len(pieces) == 1:
        return pieces[0].copy()
    var starts = List[Int](capacity=len(owners))
    var at = 0
    for o in range(len(owners)):
        starts.append(at)
        at += len(picks[o])
    var order = List[Int](capacity=len(positions))
    for k in range(len(positions)):
        order.append(starts[owner_of[k]])
        starts[owner_of[k]] += 1
    return Series._from_chunks(pieces).rechunk().take(order)


def _replay_ordered(
    slot: AggregateFunction,
    groups: _Groups,
    mut reducer: Reducer,
    lo: Int,
    hi: Int,
) raises:
    """An order-sensitive reduction of groups [lo, hi) (`_K_ORDERED`):
    their buffered rows sorted into input order, then into `reducer`."""
    var owners = List[Int]()
    var owner_of = List[Int]()
    var positions = List[Int]()
    var targets = List[Int]()
    _ordered_rows(slot, groups, lo, hi, owners, owner_of, positions, targets)
    if len(positions) == 0:
        return
    reducer.update(
        reducer_input(reducer, _gather_rows(owners, owner_of, positions)),
        0,
        True,
        targets,
    )


# How a reduction's result is made from its group rows (`_Finalizer`):
# its states unloaded into a `Reducer` that finishes them; nothing (top
# rows, read separately); or the result column written straight from the
# rows: counts, an integer, float or decimal sum, a mean.
comptime _F_STATES = 0
comptime _F_SKIP = 1
comptime _F_COUNTS = 2
comptime _F_INT_SUM = 3
comptime _F_FLOAT_SUM = 4
comptime _F_MEAN = 5
comptime _F_QUANTILE = 6
comptime _F_DECIMAL_SUM = 7
comptime _F_DECIMAL_MEAN = 8
# A value state's result in its input's type (first, last, a string's
# minimum or maximum), a Boolean from logic flags, a decimal's extreme.
comptime _F_VALUE = 9
comptime _F_LOGIC = 10
comptime _F_WIDE = 11
# An ordered first or last (`_K_ORDERED`): the value of the group's
# earliest or latest buffered row, of any type.
comptime _F_PICK = 12
# An ordered implode (`_K_ORDERED`): every row of
# the group in input order, as a list.
comptime _F_LIST = 13


def _finalize_mode(slot: AggregateFunction, template: Reducer) -> Int:
    """The `_F_` mode of a reduction whose state is `slot`: a result
    written from the rows where it is the one `template.finish` makes from
    the same state (counts, Int64 or narrow integer sums, float sums,
    means), else its states unloaded."""
    var kind = slot.kind
    if kind == _K_TOP_ROWS:
        return _F_SKIP
    if kind == _K_ORDERED and (template.op == FIRST or template.op == LAST):
        return _F_PICK
    if kind == _K_ORDERED and template.op == IMPLODE:
        return _F_LIST
    if kind == _K_STATE or kind == _K_ORDERED:
        return _F_STATES
    if (
        kind == _K_FIRST
        or kind == _K_LAST
        or kind == _K_STRING_MIN
        or kind == _K_STRING_MAX
    ):
        return _F_VALUE
    if kind == _K_LOGIC:
        return _F_LOGIC
    if kind == _K_WIDE_MIN or kind == _K_WIDE_MAX:
        return _F_WIDE
    if (
        kind == _K_LEN
        or kind == _K_COUNT
        or kind == _K_NULLS
        or kind == _K_DISTINCT
    ):
        return _F_COUNTS
    if kind == _K_VALUES:
        return _F_QUANTILE
    if kind == _K_DECIMAL_SUM:
        # The 128-bit total is the result's unscaled value (its precision
        # held on every add), so finalization needs no further rounding.
        return _F_DECIMAL_SUM if template.op == SUM else _F_DECIMAL_MEAN
    var summed = (
        kind == _K_INT_SUM or kind == _K_NARROW_SUM or kind == _K_FLOAT_SUM
    )
    if template.op == MEAN and summed:
        return _F_MEAN
    if template.op != SUM:
        return _F_STATES
    if kind == _K_FLOAT_SUM:
        return _F_FLOAT_SUM
    # A narrow sum's total fits Int64; a wide one only when its input is
    # Int64 (a UInt64 total may not).
    if kind == _K_NARROW_SUM or (
        kind == _K_INT_SUM and template.input == DataType.INT64
    ):
        return _F_INT_SUM
    return _F_STATES


struct _Finalizer(Movable):
    """An `AggregateFunction`'s `finalize` over a finish job's groups, a
    vector at a time: results written straight into the result column
    where the mode allows, else states unloaded into a `Reducer`, which
    finishes them as every other route does."""

    var slot: AggregateFunction
    var mode: Int
    var op: Int
    var min_count: Int64
    var integer: Int64
    var q: Float64
    var method: String
    var logical: DataType
    var gathered: List[Float64]
    var state: Reducer
    var ints: List[Int64]
    var floats: List[Float64]
    var wides: List[Int128]
    var bools: List[Bool]
    var valid: List[Bool]
    var nulls: Bool
    # A value state's column (one, for `_F_VALUE`).
    var values: List[_ValueColumn]
    # An ordered first or last's picked rows (`_F_PICK`): the buffers,
    # and each group's row as a buffer and a position in it. An implode's
    # (`_F_LIST`) are every row of each group in turn, and `list_counts`
    # each group's number of rows.
    var pick_owners: List[Int]
    var pick_owner: List[Int]
    var pick_rows: List[Int]
    var list_counts: List[Int]

    def __init__(
        out self,
        slot: AggregateFunction,
        spec: ReducerSpec,
        n: Int,
        held: _HeldBuffers,
    ) raises:
        self.slot = slot.copy()
        var template = spec.make(0)
        self.mode = _finalize_mode(slot, template)
        self.op = spec.op
        self.min_count = Int64(template.min_count)
        self.integer = spec.integer
        self.q = template.floating
        self.method = template.text
        self.logical = spec.logical
        self.gathered = List[Float64]()
        var direct = self.mode >= _F_COUNTS and self.mode < _F_VALUE
        var integer = self.mode == _F_COUNTS or self.mode == _F_INT_SUM
        var wide = self.mode == _F_DECIMAL_SUM or self.mode == _F_WIDE
        self.ints = List[Int64](
            unsafe_uninit_length=n if direct and integer else 0
        )
        self.floats = List[Float64](
            unsafe_uninit_length=n if direct and not integer and not wide else 0
        )
        self.wides = List[Int128](unsafe_uninit_length=n if wide else 0)
        self.bools = List[Bool](
            unsafe_uninit_length=n if self.mode == _F_LOGIC else 0
        )
        self.valid = List[Bool](
            unsafe_uninit_length=n if (direct and self.mode != _F_COUNTS)
            or (self.mode >= _F_LOGIC and self.mode < _F_PICK) else 0
        )
        self.pick_owners = List[Int]()
        self.pick_owner = List[Int](capacity=n if self.mode == _F_PICK else 0)
        self.pick_rows = List[Int](capacity=n if self.mode == _F_PICK else 0)
        self.list_counts = List[Int](capacity=n if self.mode == _F_LIST else 0)
        self.nulls = False
        self.values = List[_ValueColumn]()
        if self.mode == _F_VALUE:
            self.values.append(
                _ValueColumn("", spec.logical, slot.storage, n, held)
            )
        if self.mode == _F_STATES:
            self.state = spec.make(n)
        else:
            self.state = template^

    def scan(
        mut self, groups: _Groups, lo: Int, hi: Int, held: _HeldBuffers
    ) raises:
        """Groups [lo, hi)'s results from their rows."""
        if self.mode == _F_SKIP:
            return
        if self.mode == _F_STATES:
            _unload(self.slot, groups, self.state, lo, hi)
            return
        if self.mode == _F_PICK:
            # Every group has a row, so each takes the first (or last) of
            # its rows in input order.
            var owner_of = List[Int]()
            var positions = List[Int]()
            var targets = List[Int]()
            _ordered_rows(
                self.slot,
                groups,
                lo,
                hi,
                self.pick_owners,
                owner_of,
                positions,
                targets,
            )
            var first = self.op == FIRST
            # Rows in input order: a group's first is the earliest of its
            # rows here, its last the latest.
            var chosen = List[Int](length=hi - lo, fill=-1)
            for e in range(len(targets)):
                var g = targets[e] - lo
                if chosen[g] < 0 or not first:
                    chosen[g] = e
            for g in range(hi - lo):
                var e = chosen[g]
                self.pick_owner.append(owner_of[e])
                self.pick_rows.append(positions[e])
            return
        if self.mode == _F_LIST:
            # The rows in input order, then by group (a stable counting
            # sort), so each group's list keeps its input order.
            var owner_of = List[Int]()
            var positions = List[Int]()
            var targets = List[Int]()
            _ordered_rows(
                self.slot,
                groups,
                lo,
                hi,
                self.pick_owners,
                owner_of,
                positions,
                targets,
            )
            var starts = List[Int](length=hi - lo + 1, fill=0)
            for e in range(len(targets)):
                starts[targets[e] - lo + 1] += 1
            for g in range(hi - lo):
                self.list_counts.append(starts[g + 1])
                starts[g + 1] += starts[g]
            var base = len(self.pick_rows)
            self.pick_owner.resize(base + len(targets), 0)
            self.pick_rows.resize(base + len(targets), 0)
            for e in range(len(targets)):
                var g = targets[e] - lo
                var at = base + starts[g]
                starts[g] += 1
                self.pick_owner[at] = owner_of[e]
                self.pick_rows[at] = positions[e]
            return
        var at = groups.rows.unsafe_ptr()
        var stride = groups.stride
        var w0 = groups.states_at + self.slot.offset
        var kind = self.slot.kind
        var ints = self.ints.unsafe_ptr()
        var floats = self.floats.unsafe_ptr()
        var valid = self.valid.unsafe_ptr()
        var nulls = False
        if self.mode == _F_VALUE:
            # First and last are valid when their row's value is; an
            # extreme once it holds one.
            var picked = kind == _K_FIRST or kind == _K_LAST
            var flags = w0 + (self.slot.width + 1 if picked else 2)
            var bit = UInt64(2) if picked else UInt64(1)
            ref column = self.values[0]
            var marks = column.valid.unsafe_ptr()
            for g in range(lo, hi):
                marks[unsafe_offset=g] = (
                    at[unsafe_offset=g * stride + flags] & bit != 0
                )
            column.write(at, stride, w0, lo, hi, held)
            return
        if self.mode == _F_LOGIC:
            var bools = self.bools.unsafe_ptr()
            var ignore_nulls = self.integer != 0
            for g in range(lo, hi):
                var flags = at[unsafe_offset=g * stride + w0]
                var saw_true = flags & 1 != 0
                var saw_false = flags & 2 != 0
                var saw_null = flags & 4 != 0
                var ok: Bool
                var value: Bool
                if self.op == ANY:
                    value = saw_true
                    ok = saw_true or ignore_nulls or not saw_null
                elif self.op == ALL:
                    value = not saw_false
                    ok = saw_false or ignore_nulls or not saw_null
                elif self.op == MIN:
                    value = not saw_false
                    ok = saw_true or saw_false
                else:
                    value = saw_true
                    ok = saw_true or saw_false
                bools[unsafe_offset=g] = value and ok
                valid[unsafe_offset=g] = ok
                nulls = nulls or not ok
            self.nulls = self.nulls or nulls
            return
        if self.mode == _F_WIDE:
            var wides = self.wides.unsafe_ptr()
            for g in range(lo, hi):
                var w = g * stride + w0
                var ok = at[unsafe_offset=w + 2] != 0
                valid[unsafe_offset=g] = ok
                nulls = nulls or not ok
                wides[unsafe_offset=g] = bitcast[DType.int128](
                    (UInt128(at[unsafe_offset=w + 1]) << 64)
                    | UInt128(at[unsafe_offset=w])
                ) if ok else 0
            self.nulls = self.nulls or nulls
            return
        if self.mode == _F_COUNTS:
            for g in range(lo, hi):
                ints[unsafe_offset=g] = bitcast[DType.int64](
                    at[unsafe_offset=g * stride + w0]
                )
        elif self.mode == _F_FLOAT_SUM:
            for g in range(lo, hi):
                var w = g * stride + w0
                var ok = (
                    bitcast[DType.int64](at[unsafe_offset=w + 1])
                    >= self.min_count
                )
                valid[unsafe_offset=g] = ok
                nulls = nulls or not ok
                floats[unsafe_offset=g] = bitcast[DType.float64](
                    at[unsafe_offset=w]
                ) if ok else 0
        elif self.mode == _F_QUANTILE:
            for g in range(lo, hi):
                var state = _words_at(Int(at.unsafe_offset(g * stride + w0)))
                var n = Int(state[unsafe_offset=2])
                var ok = n > 0
                floats[unsafe_offset=g] = 0
                if ok:
                    if len(self.gathered) < n:
                        self.gathered = List[Float64](
                            unsafe_uninit_length=2 * n
                        )
                    var values = Pointer[Float64, MutAnyOrigin](
                        unsafe_from_address=Int(self.gathered.unsafe_ptr())
                    )
                    _values_gather(state, values)
                    var result = quantile_in(
                        Span[Float64, MutAnyOrigin](
                            unsafe_ptr=values, length=n
                        ),
                        self.q,
                        self.method,
                    )
                    ok = Bool(result)
                    if ok:
                        floats[unsafe_offset=g] = result.value()
                valid[unsafe_offset=g] = ok
                nulls = nulls or not ok
        elif self.mode == _F_DECIMAL_SUM or self.mode == _F_DECIMAL_MEAN:
            var wides = self.wides.unsafe_ptr()
            var scale = self.state.dtype.scale()
            for g in range(lo, hi):
                var w = g * stride + w0
                var total = bitcast[DType.int128](
                    (UInt128(at[unsafe_offset=w + 1]) << 64)
                    | UInt128(at[unsafe_offset=w])
                )
                var count = bitcast[DType.int64](at[unsafe_offset=w + 2])
                if self.mode == _F_DECIMAL_SUM:
                    var ok = count >= self.min_count
                    valid[unsafe_offset=g] = ok
                    nulls = nulls or not ok
                    wides[unsafe_offset=g] = total if ok else 0
                else:
                    var ok = count > 0
                    valid[unsafe_offset=g] = ok
                    nulls = nulls or not ok
                    floats[unsafe_offset=g] = decimal_mean(
                        total, count, scale
                    ) if ok else 0
        elif self.mode == _F_MEAN:
            for g in range(lo, hi):
                var w = g * stride + w0
                var count: Int64
                var total: Float64
                if kind == _K_FLOAT_SUM:
                    count = bitcast[DType.int64](at[unsafe_offset=w + 1])
                    total = bitcast[DType.float64](at[unsafe_offset=w])
                elif kind == _K_NARROW_SUM:
                    count = bitcast[DType.int64](at[unsafe_offset=w + 1])
                    total = (
                        bitcast[DType.int64](at[unsafe_offset=w])
                        .cast[DType.int128]()
                        .cast[DType.float64]()
                    )
                else:
                    count = bitcast[DType.int64](at[unsafe_offset=w + 2])
                    total = bitcast[DType.int128](
                        (UInt128(at[unsafe_offset=w + 1]) << 64)
                        | UInt128(at[unsafe_offset=w])
                    ).cast[DType.float64]()
                var ok = count > 0
                valid[unsafe_offset=g] = ok
                nulls = nulls or not ok
                floats[unsafe_offset=g] = total / Float64(count) if ok else 0
        else:
            for g in range(lo, hi):
                var w = g * stride + w0
                var count: Int64
                var total: Int64
                if kind == _K_NARROW_SUM:
                    count = bitcast[DType.int64](at[unsafe_offset=w + 1])
                    total = bitcast[DType.int64](at[unsafe_offset=w])
                else:
                    count = bitcast[DType.int64](at[unsafe_offset=w + 2])
                    var wide = bitcast[DType.int128](
                        (UInt128(at[unsafe_offset=w + 1]) << 64)
                        | UInt128(at[unsafe_offset=w])
                    )
                    var ok = count >= self.min_count
                    if ok and (
                        wide > Int64.MAX.cast[DType.int128]()
                        or wide < Int64.MIN.cast[DType.int128]()
                    ):
                        raise Error("Int64 expression sum overflow")
                    total = wide.cast[DType.int64]()
                var ok = count >= self.min_count
                valid[unsafe_offset=g] = ok
                nulls = nulls or not ok
                ints[unsafe_offset=g] = total if ok else 0
        self.nulls = self.nulls or nulls

    def reducer(deinit self) raises -> Reducer:
        """The reduction's `Reducer`: its unloaded states, or one holding
        the finished column, of the type and validity its `finish` gives
        (an integer sum in its input's sum type, a Float32 sum as Float32)."""
        if self.mode < _F_COUNTS:
            return self.state^
        var state = self.state^
        if self.mode == _F_PICK:
            if len(self.pick_rows) == 0:
                # No groups: an empty part, which is never read.
                return state^
            state.finished = reducer_input(
                state,
                _gather_rows(self.pick_owners, self.pick_owner, self.pick_rows),
            )
            return state^
        if self.mode == _F_LIST:
            if len(self.list_counts) == 0:
                return state^
            var offsets = List[Int64](capacity=len(self.list_counts) + 1)
            offsets.append(0)
            var total = 0
            for count in self.list_counts:
                total += count
                offsets.append(Int64(total))
            var child = _gather_rows(
                self.pick_owners, self.pick_owner, self.pick_rows
            ).renamed("item")
            child._field_metadata = state.metadata
            state.finished = Series("", ListColumn(offsets^, child^))
            return state^
        if self.mode == _F_VALUE:
            var values = self.values^
            state.finished = values.pop().column()
            return state^
        if self.mode == _F_LOGIC:
            state.finished = Series("", BoolColumn(self.bools, self.valid))
            return state^
        var bits = _pack_bits(self.valid) if self.nulls else List[UInt8]()
        var result: Series
        if self.mode == _F_WIDE:
            # At the decimal's own storage width, as the input's.
            var width = self.logical.decimal_width()
            if width == 64:
                var narrow = List[Int64](capacity=len(self.wides))
                for v in self.wides:
                    narrow.append(Int64(v))
                result = Series("", Column[Int64](values=narrow^, bits=bits^))
            elif width == 32:
                var narrow = List[Int32](capacity=len(self.wides))
                for v in self.wides:
                    narrow.append(Int32(v))
                result = Series("", Column[Int32](values=narrow^, bits=bits^))
            else:
                result = Series(
                    "", Column[Int128](values=self.wides^, bits=bits^)
                )
            state.finished = result.with_dtype(self.logical)
            return state^
        if self.mode == _F_COUNTS:
            result = Series("", Column[Int64](self.ints^))
        elif self.mode == _F_INT_SUM:
            result = _sum_typed(self.ints^, bits^, state.input)
        elif self.mode == _F_DECIMAL_SUM:
            result = Series(
                "", Column[Int128](values=self.wides^, bits=bits^)
            ).with_dtype(state.dtype)
        else:
            result = Series(
                "", Column[Float64](values=self.floats^, bits=bits^)
            )
            if self.mode == _F_FLOAT_SUM and state.input != DataType.FLOAT64:
                result = _from_canonical(result, state.input)
        state.finished = result^
        return state^


def _sum_typed(
    var totals: List[Int64], var bits: List[UInt8], input: DataType
) raises -> Series:
    """Integer sums as their input's sum type (`Reducer._integer_sum`):
    Int64 for Int64 and 8- or 16-bit inputs, else the input's own type,
    which every total must fit."""
    var target = input.sum_type()
    if input == DataType.INT64 or target == DataType.INT64:
        return Series("", Column[Int64](values=totals^, bits=bits^))
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        comptime if D.is_integral():
            if target == DataType.of(D):
                var output = List[Scalar[D]](unsafe_uninit_length=len(totals))
                var at = output.unsafe_ptr()
                for g in range(len(totals)):
                    var total = totals[g]
                    if (
                        total.cast[DType.int128]()
                        > Scalar[D].MAX.cast[DType.int128]()
                        or total.cast[DType.int128]()
                        < Scalar[D].MIN.cast[DType.int128]()
                    ):
                        raise Error(target.name() + " expression sum overflow")
                    at[unsafe_offset=g] = total.cast[D]()
                return Series("", Column[Scalar[D]](values=output^, bits=bits^))
    raise Error("Unsupported sum type " + target.name())


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
    # An implode's input field metadata (`Reducer.metadata`).
    var metadata: Optional[ArcPointer[List[UInt8]]]

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
            template.metadata,
        )

    def make(self, groups: Int) -> Reducer:
        var reducer = Reducer(
            self.op,
            self.input,
            groups,
            self.min_count,
            self.integer,
            self.floating,
            self.text,
            Optional(self.logical),
        )
        reducer.metadata = self.metadata
        return reducer^


struct _HeldBuffers(Movable):
    """Keep the byte buffers referenced by string keys, indexed by
    address. Finished string columns can use Arrow views into those buffers
    without copying key bytes. A group's long key words hold the address
    of its bytes, found here by binary search."""

    var buffers: List[_Buffer[UInt8]]
    # Buffer ranges in address order: start, end and buffer.
    var starts: List[Int]
    var ends: List[Int]
    var index: List[Int]

    def __init__(out self):
        """No buffers held."""
        self.buffers = List[_Buffer[UInt8]]()
        self.starts = List[Int]()
        self.ends = List[Int]()
        self.index = List[Int]()

    def __init__(out self, held: List[List[Series]]):
        """The buffers of every sink's held string columns."""
        self.buffers = List[_Buffer[UInt8]]()
        self.starts = List[Int]()
        self.ends = List[Int]()
        self.index = List[Int]()
        var keyed = List[UInt128]()
        for j in range(len(held)):
            for column in held[j]:
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

    def _add(mut self, buffer: _Buffer[UInt8], mut keyed: List[UInt128]):
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


struct _ValueColumn(Movable):
    """One column of a finish job's groups, built from words in their rows
    (a key's, or a value state's) a vector of group rows at a time: a
    number or 128-bit value written in place in its result buffer, as the
    column's type (temporal, decimal and categorical values keep theirs); a
    string as an Arrow view, one up to twelve bytes its two words exactly
    (the view's length, prefix and inline bytes, zero past the end), a
    longer one its first word and the held buffer and offset its address
    falls in, so no bytes are copied unless required. A string held in no
    shared buffer (a dictionary's entry) is copied. Rows are valid as the
    caller marks them in `valid`."""

    var name: String
    var dtype: DataType
    var kind: Int
    var n: Int
    # A number's or 128-bit value's result, written through `values`.
    var series: Series
    var values: Int
    var valid: List[Bool]
    var nulls: Bool
    var bools: List[Bool]
    var views: List[StringView]
    var buffers: List[_Buffer[UInt8]]
    # Each held buffer's place among this column's, once it is used.
    var local: List[Int]
    var spill: List[UInt8]
    var spill_at: Int
    var buffer_bytes: Int

    def __init__(
        out self,
        name: String,
        dtype: DataType,
        kind: Int,
        n: Int,
        held: _HeldBuffers,
    ) raises:
        self.name = name
        self.dtype = dtype
        self.kind = kind
        self.n = n
        self.valid = List[Bool](unsafe_uninit_length=n)
        self.nulls = False
        self.bools = List[Bool]()
        self.views = List[StringView]()
        self.buffers = List[_Buffer[UInt8]]()
        self.local = List[Int]()
        self.spill = List[UInt8]()
        self.spill_at = -1
        self.buffer_bytes = 0
        self.series = Series(self.name, Column[Int64](List[Int64]()))
        self.values = 0
        if kind == _VALUE_STRING:
            self.views = List[StringView](unsafe_uninit_length=n)
            self.local = List[Int](length=len(held.buffers), fill=-1)
            return
        if kind == _VALUE_BOOL:
            self.bools = List[Bool](unsafe_uninit_length=n)
            return
        if kind == _VALUE_WIDE:
            var column = Column[Int128](List[Int128](unsafe_uninit_length=n))
            self.values = Int(column._ptr())
            self.series = Series(self.name, column^)
            return
        comptime for d in range(len(NUMERIC_DTYPES)):
            comptime D = NUMERIC_DTYPES[d]
            if kind == d:
                var column = Column[Scalar[D]](
                    List[Scalar[D]](unsafe_uninit_length=n)
                )
                self.values = Int(column._ptr())
                self.series = Series(self.name, column^)
                return
        raise Error("aggregate table: unsupported value storage")

    def write(
        mut self,
        at: Pointer[UInt64, _],
        stride: Int,
        w0: Int,
        lo: Int,
        hi: Int,
        held: _HeldBuffers,
    ):
        """Groups [lo, hi)'s values from their rows' words at `w0` (each
        row `stride` words from the last), `valid` already marked."""
        var valid = self.valid.unsafe_ptr()
        for g in range(lo, hi):
            if not valid[unsafe_offset=g]:
                self.nulls = True
                break
        if self.kind == _VALUE_STRING:
            self._strings(at, stride, w0, lo, hi, held)
            return
        if self.kind == _VALUE_BOOL:
            var bools = self.bools.unsafe_ptr()
            for g in range(lo, hi):
                bools[unsafe_offset=g] = at[unsafe_offset=g * stride + w0] == 1
            return
        if self.kind == _VALUE_WIDE:
            var into = Pointer[Int128, MutAnyOrigin](
                unsafe_from_address=self.values
            )
            for g in range(lo, hi):
                var row = g * stride + w0
                into[unsafe_offset=g] = bitcast[DType.int128](
                    (UInt128(at[unsafe_offset=row + 1]) << 64)
                    | UInt128(at[unsafe_offset=row])
                )
            return
        comptime for d in range(len(NUMERIC_DTYPES)):
            comptime D = NUMERIC_DTYPES[d]
            if self.kind == d:
                var into = Pointer[Scalar[D], MutAnyOrigin](
                    unsafe_from_address=self.values
                )
                for g in range(lo, hi):
                    var word = at[unsafe_offset=g * stride + w0]
                    comptime if D.is_floating_point():
                        into[unsafe_offset=g] = Scalar[D](
                            bitcast[DType.float64](word)
                        )
                    elif D.is_unsigned():
                        into[unsafe_offset=g] = Scalar[D](word)
                    else:
                        into[unsafe_offset=g] = Scalar[D](
                            bitcast[DType.int64](word)
                        )
                return

    def _strings(
        mut self,
        at: Pointer[UInt64, _],
        stride: Int,
        w0: Int,
        lo: Int,
        hi: Int,
        held: _HeldBuffers,
    ):
        var views = self.views.unsafe_ptr()
        var valid = self.valid.unsafe_ptr()
        for g in range(lo, hi):
            if not valid[unsafe_offset=g]:
                views[unsafe_offset=g] = StringView(0, 0, 0, 0)
                continue
            var head = at[unsafe_offset=g * stride + w0]
            var tail = at[unsafe_offset=g * stride + w0 + 1]
            var length = Int(head & 0xFFFFFFFF)
            if length <= STRING_VIEW_INLINE_BYTES:
                views[unsafe_offset=g] = StringView(
                    UInt32(head & 0xFFFFFFFF),
                    UInt32(head >> 32),
                    UInt32(tail & 0xFFFFFFFF),
                    UInt32(tail >> 32),
                )
                continue
            self.buffer_bytes += length
            var found = held.find(Int(tail), length)
            var b: Int
            var offset = found[1]
            if found[0] >= 0 and offset <= _STRING_VIEW_BUFFER_LIMIT - length:
                b = self.local[found[0]]
                if b < 0:
                    b = len(self.buffers)
                    self.local[found[0]] = b
                    self.buffers.append(held.buffers[found[0]].copy())
            else:
                if self.spill_at >= 0 and (
                    len(self.spill) + length > _STRING_VIEW_BUFFER_LIMIT
                ):
                    var full = List[UInt8]()
                    swap(full, self.spill)
                    self.buffers[self.spill_at] = _Buffer(full^)
                    self.spill_at = -1
                if self.spill_at < 0:
                    self.spill_at = len(self.buffers)
                    self.buffers.append(_Buffer(List[UInt8]()))
                b = self.spill_at
                offset = len(self.spill)
                self.spill.extend(
                    Span[UInt8, ImmutAnyOrigin](
                        unsafe_ptr=Pointer[UInt8, ImmutAnyOrigin](
                            unsafe_from_address=Int(tail)
                        ),
                        length=length,
                    )
                )
            views[unsafe_offset=g] = StringView(
                UInt32(head & 0xFFFFFFFF),
                UInt32(head >> 32),
                UInt32(b),
                UInt32(offset),
            )

    def column(deinit self) raises -> Series:
        """The finished column."""
        var bits = _pack_bits(self.valid) if self.nulls else List[UInt8]()
        if self.kind == _VALUE_STRING:
            if self.spill_at >= 0:
                self.buffers[self.spill_at] = _Buffer(self.spill^)
            return Series(
                self.name,
                StringColumn(
                    StringViewStorage(
                        self.views^,
                        self.buffers^,
                        bits^,
                        self.n,
                        self.buffer_bytes,
                    )
                ),
            ).with_dtype(self.dtype)
        if self.kind == _VALUE_BOOL:
            return Series(self.name, BoolColumn(self.bools, self.valid))
        var series = self.series^
        if self.nulls:
            var shared = _Buffer(bits^)
            if self.kind == _VALUE_WIDE:
                series._data[Column[Int128]]._bits = shared.copy()
            else:
                comptime for d in range(len(NUMERIC_DTYPES)):
                    comptime D = NUMERIC_DTYPES[d]
                    if self.kind == d:
                        series._data[Column[Scalar[D]]]._bits = shared.copy()
        return series.with_dtype(self.dtype)


struct _KeyColumn(Movable):
    """One key column of a finish job's groups (`_ValueColumn`), a row
    valid unless its null bit is set."""

    var values: _ValueColumn
    var w: Int
    var bit: UInt64
    # The key's source field metadata, which its groups' column keeps.
    var metadata: Optional[ArcPointer[List[UInt8]]]

    def __init__(
        out self,
        template: Series,
        w: Int,
        bit: UInt64,
        n: Int,
        held: _HeldBuffers,
    ) raises:
        self.values = _ValueColumn(
            template.name(), template.dtype(), _storage_kind(template), n, held
        )
        self.w = w
        self.bit = bit
        self.metadata = template._field_metadata

    def scan(mut self, groups: _Groups, lo: Int, hi: Int, held: _HeldBuffers):
        """Groups [lo, hi)'s keys from their rows."""
        var at = groups.rows.unsafe_ptr()
        var stride = groups.stride
        var valid = self.values.valid.unsafe_ptr()
        for g in range(lo, hi):
            valid[unsafe_offset=g] = (
                at[unsafe_offset=g * stride + 1] & (self.bit) == 0
            )
        self.values.write(at, stride, _HEAD + self.w, lo, hi, held)

    def column(deinit self) raises -> Series:
        """The finished key column, with its source's field metadata."""
        var metadata = self.metadata
        var column = self.values^.column()
        column._field_metadata = metadata
        return column^
