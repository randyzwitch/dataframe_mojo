"""Thread-owned pipelines over morsels (docs/executor-design.md, #538).

A pipeline is a source, a list of row-local steps and a sink. Every worker
thread runs the whole pipeline on the morsels it takes from the source's
cursor and pushes into its own sink state; the states are combined once
when the source is exhausted. The main thread does no per-morsel work.
DuckDB's `PipelineExecutor` runs one per thread the same way, and Polars'
streaming engine one task per lane per node.

A morsel is a frame and a selection. A filter narrows the selection and
copies nothing; a column is gathered through the selection when a step
reads it, once, and the gathered column replaces the shared one. DuckDB
slices a chunk into dictionary vectors the same way.
"""
from std.atomic import Atomic
from std.ffi import external_call
from std.memory import ArcPointer, Pointer, bitcast
from std.sys import size_of
from std.time import perf_counter_ns

from .bool_column import BoolColumn, both_true, true_count
from .expr import COL, SELECTOR, Expr
from .aggregate import Reducer
from .binding import BoundExpr, bind
from .column import Column
from .dtype import DataType, NUMERIC_DTYPES
from .execution import _ReduceJob, _new_reducer
from .expr import (
    COUNT,
    FIRST,
    LAST,
    LEN,
    MAX,
    MEAN,
    MIN,
    N_UNIQUE,
    NULL_COUNT,
    STD,
    SUM,
    VAR,
    is_reduction,
    subtree,
)
from .frame import (
    DataFrame,
    _StreamReduction,
    _equality_words,
    _expand_struct_keys,
    _finish_parts,
    _rows_mask,
    _word_hash,
    concat,
)
from .indexed_reduce import _reduce_floats, _reduce_ints
from .join_hash import PreparedHashIndex
from .row_encode import encodable
from .gather import can_filter_aligned_chunks, true_rows
from .huge_pages import huge_list
from .mask_filter import filter_columns
from .parallel import Job, run_jobs
from .series import Series
from .trace import trace_path

# Step kinds, the row-local plan nodes a pipeline runs between its source
# and its sink. Numbered here so this module does not import the planner.
comptime STEP_FILTER = 0
comptime STEP_SELECT = 1
comptime STEP_WITH_COLUMNS = 2
comptime STEP_DROP = 3
# A hash join against a build frame prepared before the pipeline runs
# (the index is shared by every worker): the morsel is the probe side.
comptime STEP_JOIN = 4
# Hash parts of a worker's grouped state (DuckDB's radix bits): 16 tables
# of a few thousand groups each while inserting; while collecting, one
# gather per morsel and part, so few parts keep that cheap.
comptime _PART_BITS = 4
# Groups a worker holds in one table before splitting it into
# 2^_PART_BITS parts: a few groups (ClickBench q7's eight) pay nothing
# per part and morsel, many fit the parts' tables in cache.
comptime _SPLIT_GROUPS = 4096
# The column a top-k sink adds to its candidates: the row's place in the
# source's order (morsel sequence, then row within the morsel's output),
# the last sort key, so ties come out as a stable sort of the whole input
# would place them, whichever worker held them.
comptime _ORDER_COLUMN = "__pipeline_order"
# Candidate rows a worker holds before folding them to the k best: the
# fold's setup is worth about this many rows of comparisons.
comptime _FOLD_ROWS = 8192


@fieldwise_init
struct Step(Copyable, Movable):
    var kind: Int
    var exprs: List[Expr]
    var names: List[String]
    # STEP_JOIN: the build frame and index (an offset into the lists the
    # pipeline is given), the right keys (`names` are the left keys), the
    # join kind, the suffix and whether keys coalesce.
    var join: Int
    var right_keys: List[String]
    var how: Int
    var suffix: String
    var coalesce: Bool


def _reads(exprs: List[Expr]) -> Optional[List[String]]:
    """The columns the expressions read, or None when a selector may read
    any column."""
    var names = List[String]()
    for expr in exprs:
        for node in expr._nodes:
            if node.op == SELECTOR:
                return None
            if node.op == COL and node.text not in names:
                names.append(node.text)
    return Optional(names^)


comptime _BOUND_NONE = 0
comptime _BOUND_INT = 1
comptime _BOUND_FLOAT = 2


struct _TopBound(Copyable, Movable):
    """The first sort key's value at a streamed top-k's k-th row so far.

    A row whose key is worse than it (after it in the sort's direction)
    cannot enter the top k, so a batch keeps only rows at or better than
    the bound when at most one in eight are, ties included (later keys decide those) and nulls included
    (wherever the sort puts them). Integer-backed keys (integers, dates,
    datetimes, durations) compare as Int64, floats as Float64 with NaN
    kept; other dtypes get no bound.
    """

    var kind: Int
    var name: String
    var descending: Bool
    var integer: Int64
    var floating: Float64

    def __init__(out self):
        self.kind = _BOUND_NONE
        self.name = String()
        self.descending = False
        self.integer = 0
        self.floating = 0.0

    def __init__(
        out self, candidates: DataFrame, name: String, descending: Bool, k: Int
    ) raises:
        """The bound from `candidates`, sorted best first, once it holds
        `k` rows; none before then or when its k-th key is null."""
        self = Self()
        if candidates.height() < k or k <= 0:
            return
        var key = candidates.column(name)
        if key.is_chunked():
            key = key.rechunk()
        comptime for d in range(len(NUMERIC_DTYPES)):
            comptime D = NUMERIC_DTYPES[d]
            if key._data.isa[Column[Scalar[D]]]():
                ref column = key._data[Column[Scalar[D]]]
                if not column._valid(k - 1):
                    return
                var value = column._get(k - 1)
                comptime if D.is_floating_point():
                    self.floating = Float64(value)
                    if self.floating != self.floating:
                        # NaN sorts past every number: no row is worse.
                        return
                    self.kind = _BOUND_FLOAT
                elif D == DType.uint64:
                    return
                else:
                    self.integer = Int64(value)
                    self.kind = _BOUND_INT
                self.name = name
                self.descending = descending
                return

    def rows(self, frame: DataFrame) raises -> Optional[List[Int]]:
        """The rows of `frame` that can still enter the top k, or None
        when every row can (or the batch lacks the key)."""
        if frame.height() == 0 or self.name not in frame.columns():
            return None
        var key = frame.column(self.name)
        if key.is_chunked():
            key = key.rechunk()
        var rows = List[Int]()
        comptime for d in range(len(NUMERIC_DTYPES)):
            comptime D = NUMERIC_DTYPES[d]
            if key._data.isa[Column[Scalar[D]]]():
                ref column = key._data[Column[Scalar[D]]]
                var values = column.unsafe_values()
                var nulls = column.null_count() > 0
                for i in range(len(column)):
                    if nulls and not column._valid(i):
                        rows.append(i)
                        continue
                    var value = values.unsafe_offset(i)[]
                    var ok: Bool
                    comptime if D.is_floating_point():
                        var x = Float64(value)
                        ok = x != x or (
                            x
                            >= self.floating if self.descending else x
                            <= self.floating
                        )
                    elif D == DType.uint64:
                        ok = True
                    else:
                        var x = Int64(value)
                        ok = (
                            x
                            >= self.integer if self.descending else x
                            <= self.integer
                        )
                    if ok:
                        rows.append(i)
                if len(rows) == frame.height():
                    return None
                return rows^
        return None


struct Morsel(Movable):
    """A frame and a selection over its rows.

    Columns not yet `compact` are the source's, shared, and read through
    `mask`; a compact column already holds only the selected rows.
    """

    var frame: DataFrame
    var mask: BoolColumn
    var selected: Bool
    var count: Int
    var compact: List[Bool]
    var sequence: Int

    def __init__(out self, var frame: DataFrame, sequence: Int):
        self.count = frame.height()
        self.compact = List[Bool](length=frame.width(), fill=True)
        self.frame = frame^
        self.mask = BoolColumn(List[Bool]())
        self.selected = False
        self.sequence = sequence

    def height(self) -> Int:
        return self.count

    def _index(self, name: String) raises -> Int:
        for i in range(self.frame.width()):
            if self.frame._columns[i].name() == name:
                return i
        raise Error("Column not found: " + name)

    def gather(mut self, names: List[String]) raises:
        """Make the named columns compact: gathered through the selection
        in one pass over the mask for all of them."""
        if not self.selected:
            return
        var indices = List[Int]()
        var pending = List[Series]()
        for name in names:
            var i = self._index(name)
            if not self.compact[i]:
                indices.append(i)
                pending.append(self.frame._columns[i].copy())
        if len(pending) == 0:
            return
        var filtered = filter_columns(pending, self.mask)
        ref others = filtered[1]
        if len(others) > 0:
            # String and nested columns: the row-list route.
            var rows = true_rows(self.mask)
            var next_fixed = 0
            var next_other = 0
            for k in range(len(indices)):
                if next_other < len(others) and others[next_other] == k:
                    self.frame._columns[indices[k]] = pending[k].take(rows)
                    next_other += 1
                else:
                    self.frame._columns[indices[k]] = filtered[0][
                        next_fixed
                    ].copy()
                    next_fixed += 1
        else:
            for k in range(len(indices)):
                self.frame._columns[indices[k]] = filtered[0][k].copy()
        for i in indices:
            self.compact[i] = True

    def gather_all(mut self) raises:
        """Every column compact; the selection is then dropped."""
        if not self.selected:
            return
        var names = List[String]()
        for i in range(self.frame.width()):
            if not self.compact[i]:
                names.append(self.frame._columns[i].name())
        self.gather(names)
        self.frame._height = self.count
        self.selected = False

    def _compact_frame(mut self, names: List[String]) raises -> DataFrame:
        """The named columns, compact, as a frame of `count` rows."""
        self.gather(names)
        var columns = List[Series](capacity=len(names))
        for name in names:
            columns.append(self.frame._columns[self._index(name)].copy())
        return DataFrame(columns^, height=self.count)

    def filter(mut self, predicate: Expr, batch_size: Int) raises:
        """Narrow the selection to the rows `predicate` keeps."""
        var reads = _reads([predicate.copy()])
        var names: List[String]
        if reads:
            names = reads.take()
        else:
            names = self.frame.columns()
        var narrowed: BoolColumn
        if not self.selected:
            var input = self._compact_frame(names)
            narrowed = input._selection(predicate, batch_size)
            var kept = true_count(narrowed)
            if kept == self.count:
                return
            self.mask = narrowed^
            self.selected = True
            self.count = kept
            for i in range(len(self.compact)):
                self.compact[i] = False
            return
        var input = self._compact_frame(names)
        narrowed = input._selection(predicate, batch_size)
        var kept = true_count(narrowed)
        if kept == self.count:
            return
        # Compact columns are at the old selection's rows: filter them by
        # the new mask directly. Shared columns are read through the
        # composed mask: the new mask's bits spread over the old mask's
        # set bits.
        var compact_columns = List[Series]()
        var compact_at = List[Int]()
        for i in range(self.frame.width()):
            if self.compact[i]:
                compact_at.append(i)
                compact_columns.append(self.frame._columns[i].copy())
        if len(compact_columns) > 0:
            var sliced = DataFrame(compact_columns^, height=self.count).filter(
                narrowed
            )
            for k in range(len(compact_at)):
                self.frame._columns[compact_at[k]] = sliced._columns[k].copy()
        self.mask = _spread(self.mask, narrowed)
        self.count = kept

    def select(mut self, exprs: List[Expr], batch_size: Int) raises:
        var reads = _reads(exprs)
        var names: List[String]
        if reads:
            names = reads.take()
        else:
            names = self.frame.columns()
        var input = self._compact_frame(names)
        var result = input.select_exprs(exprs, batch_size=batch_size)
        self.frame = result^
        self.count = self.frame.height()
        self.compact = List[Bool](length=self.frame.width(), fill=True)
        self.selected = False

    def with_columns(mut self, exprs: List[Expr], batch_size: Int) raises:
        var reads = _reads(exprs)
        var names: List[String]
        if reads:
            names = reads.take()
        else:
            names = self.frame.columns()
        var input = self._compact_frame(names)
        var result = input.with_columns(exprs, batch_size=batch_size)
        # Columns the expressions produced or replaced join the frame as
        # compact; the rest stay as they are.
        for c in range(result.width()):
            ref column = result._columns[c]
            var found = -1
            for i in range(self.frame.width()):
                if self.frame._columns[i].name() == column.name():
                    found = i
            if found >= 0:
                self.frame._columns[found] = column.copy()
                self.compact[found] = True
            else:
                self.frame._columns.append(column.copy())
                self.compact.append(True)

    def drop(mut self, names: List[String]) raises:
        var kept = List[Series]()
        var kept_compact = List[Bool]()
        for i in range(self.frame.width()):
            if self.frame._columns[i].name() not in names:
                kept.append(self.frame._columns[i].copy())
                kept_compact.append(self.compact[i])
        var height = self.frame._height
        self.frame = DataFrame(kept^, height=height)
        self.compact = kept_compact^

    def join(
        mut self,
        step: Step,
        joins: ArcPointer[List[DataFrame]],
        indexes: ArcPointer[List[Optional[PreparedHashIndex]]],
    ) raises:
        """Probe the join's prepared index with this morsel's rows: the
        output is the matched rows of both sides (DuckDB's `NextInnerJoin`
        over one chunk; the probe runs on this worker alone)."""
        var input = self.materialized()
        var joined = input._join_impl(
            joins[][step.join],
            left_on=step.names,
            right_on=step.right_keys,
            how=step.how,
            suffix=step.suffix,
            coalesce=step.coalesce,
            prepared=indexes[][step.join],
            workers=0,
        )
        self.count = joined.height()
        self.compact = List[Bool](length=joined.width(), fill=True)
        self.frame = joined^
        self.mask = BoolColumn(List[Bool]())
        self.selected = False

    def apply(
        mut self,
        step: Step,
        batch_size: Int,
        joins: ArcPointer[List[DataFrame]],
        indexes: ArcPointer[List[Optional[PreparedHashIndex]]],
    ) raises:
        if step.kind == STEP_FILTER:
            self.filter(step.exprs[0], batch_size)
        elif step.kind == STEP_SELECT:
            self.select(step.exprs, batch_size)
        elif step.kind == STEP_WITH_COLUMNS:
            self.with_columns(step.exprs, batch_size)
        elif step.kind == STEP_JOIN:
            self.join(step, joins, indexes)
        else:
            self.drop(step.names)

    def narrow(mut self, rows: List[Int]) raises:
        """Select only `rows` (ascending) of a fresh morsel: a top-k's
        bound applied before any step reads a column."""
        self.mask = _rows_mask(rows, self.frame._height)
        self.selected = True
        self.count = len(rows)
        for i in range(len(self.compact)):
            self.compact[i] = False

    def materialized(mut self) raises -> DataFrame:
        self.gather_all()
        return self.frame.copy()


def _spread(outer: BoolColumn, inner: BoolColumn) raises -> BoolColumn:
    """`outer` with its set bits replaced, in order, by `inner`'s bits:
    the selection of a filter applied to an already selected morsel."""
    var values = List[Bool](capacity=len(outer))
    var next = 0
    for r in range(len(outer)):
        if outer._get(r) and outer._valid(r):
            values.append(inner._get(next) and inner._valid(next))
            next += 1
        else:
            values.append(False)
    return BoolColumn(values^)


@fieldwise_init
struct _Cursor(Copyable, Movable):
    """The source's next morsel offset, shared by every worker: an atomic
    in C-allocated memory, as the thread budget of `parallel` keeps its
    counters."""

    var address: Int

    @staticmethod
    def new() -> _Cursor:
        return _Cursor(
            external_call["calloc", Int](1, size_of[Atomic[Int64]]())
        )

    def take(self, rows: Int) -> Int:
        return Int(
            Pointer[Atomic[Int64], MutAnyOrigin](
                unsafe_from_address=self.address
            )[].fetch_add(Int64(rows))
        )

    def free(self):
        _ = external_call["free", NoneType](self.address)


def _float_order(value: Float64) -> Int64:
    """An Int64 that orders as the float does (its own inverse)."""
    var bits = bitcast[DType.int64](value)
    return bits if bits >= 0 else bits ^ Int64(0x7FFFFFFFFFFFFFFF)


@fieldwise_init
struct _SharedBound(Copyable, Movable):
    """The tightest top-k bound any worker has found, shared by all: a
    worker folding its candidates publishes its k-th key, and every
    worker narrows its next morsel by the tightest (DuckDB's top-n shares
    its heap boundary across threads). Two atomics: whether a bound is
    set, and the key as an Int64 that orders as the key does."""

    var address: Int

    @staticmethod
    def new() -> _SharedBound:
        return _SharedBound(
            external_call["calloc", Int](2, size_of[Atomic[Int64]]())
        )

    def _slot(self, k: Int) -> Pointer[Atomic[Int64], MutAnyOrigin]:
        return Pointer[Atomic[Int64], MutAnyOrigin](
            unsafe_from_address=self.address + k * size_of[Atomic[Int64]]()
        )

    def publish(self, bound: _TopBound):
        if bound.kind == _BOUND_NONE:
            return
        var key = bound.integer if bound.kind == _BOUND_INT else _float_order(
            bound.floating
        )
        var value = self._slot(1)
        var state = self._slot(0)
        while True:
            var set = state[].load()
            var current = value[].load()
            if set != 0:
                var tighter = (
                    key > current if bound.descending else key < current
                )
                if not tighter:
                    return
                if value[].compare_exchange(current, key):
                    return
                continue
            # Unset: store the key, then the flag; a reader sees the flag
            # only after the key.
            if value[].compare_exchange(current, key):
                state[].store(1)
                return

    def tighten(
        self,
        mut bound: _TopBound,
        name: String,
        descending: Bool,
        floating: Bool,
    ):
        """Adopt the shared bound into `bound` when it is tighter."""
        if self._slot(0)[].load() == 0:
            return
        var key = self._slot(1)[].load()
        if bound.kind != _BOUND_NONE:
            var mine = (
                bound.integer if bound.kind
                == _BOUND_INT else _float_order(bound.floating)
            )
            var tighter = key > mine if descending else key < mine
            if not tighter:
                return
        bound.name = name
        bound.descending = descending
        if floating:
            bound.kind = _BOUND_FLOAT
            bound.floating = bitcast[DType.float64](_float_order_inverse(key))
        else:
            bound.kind = _BOUND_INT
            bound.integer = key

    def free(self):
        _ = external_call["free", NoneType](self.address)


def _float_order_inverse(key: Int64) -> Int64:
    return key if key >= 0 else key ^ Int64(0x7FFFFFFFFFFFFFFF)


def _plain_op(op: Int) -> Bool:
    """A reduction `_reduce_ints`/`_reduce_floats` update in place."""
    return op in [
        SUM,
        MEAN,
        MIN,
        MAX,
        COUNT,
        FIRST,
        LAST,
        STD,
        VAR,
        N_UNIQUE,
        NULL_COUNT,
    ]


def _plain_column(column: Series) -> Bool:
    """A numeric, non-decimal column those updates read."""
    if column.dtype().is_decimal() or column.dtype().is_categorical():
        return False
    comptime for t in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[t]
        if column._data.isa[Column[Scalar[D]]]():
            return True
    return False


def _plain_reduction(bound: BoundExpr, columns: List[Series]) -> Int:
    """The source column of a reduction `_reduce_ints`/`_reduce_floats`
    update in place (`col op` over a numeric, non-decimal column), or -1."""
    ref nodes = bound.expr._nodes
    if len(nodes) != 2 or nodes[0].op != COL:
        return -1
    if not _plain_op(nodes[1].op):
        return -1
    var source = bound.sources[0]
    if source < 0 or not _plain_column(columns[source]):
        return -1
    return source


def _update_plain(
    mut reducer: Reducer,
    column: Series,
    rows: Pointer[Int, _],
    ids: Pointer[Int, _],
    n: Int,
    op: Int,
) raises:
    comptime for t in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[t]
        if column._data.isa[Column[Scalar[D]]]():
            comptime if D.is_floating_point():
                _reduce_floats[D](
                    reducer, column._data[Column[Scalar[D]]], rows, ids, n, op
                )
            else:
                _reduce_ints[D](
                    reducer, column._data[Column[Scalar[D]]], rows, ids, n, op
                )
            return
    raise Error("grouped pipeline: unexpected column storage")


struct _PartIndex(Movable):
    """Group ids of one hash part by the partitioner's key hash and key
    words: an open-addressing slot table of ids, each group's hash and
    words stored once. A probe compares the hash, then the words through
    pointers; no key is hashed again and nothing is bounds-checked per
    row. `_KeyIndex` served the same purpose at 100 to 180 ns a probe."""

    var slots: List[Int32]
    var hashes: List[UInt64]
    var words: List[List[Int]]
    var count: Int

    def __init__(out self, width: Int):
        self.slots = huge_list(1024, Int32(-1))
        self.hashes = List[UInt64]()
        self.words = List[List[Int]](capacity=width)
        for _ in range(width):
            self.words.append(List[Int]())
        self.count = 0

    def _grow(mut self):
        var capacity = 2 * len(self.slots)
        self.slots = huge_list(capacity, Int32(-1))
        var table = self.slots.unsafe_ptr()
        var mask = capacity - 1
        var hashes = self.hashes.unsafe_ptr()
        for id in range(self.count):
            var slot = Int(hashes[unsafe_offset=id]) & mask
            while table[unsafe_offset=slot] >= 0:
                slot = (slot + 1) & mask
            table[unsafe_offset=slot] = Int32(id)

    @always_inline
    def find_or_insert(
        mut self, hash: UInt64, ptrs: List[Int], width: Int, row: Int
    ) -> Int:
        """The id of the key at `row` of the word buffers `ptrs`, added as
        the next id if new; True in the second value when added."""
        if 2 * (self.count + 1) > len(self.slots):
            self._grow()
        var table = self.slots.unsafe_ptr()
        var mask = len(self.slots) - 1
        var slot = Int(hash) & mask
        var hashes = self.hashes.unsafe_ptr()
        while True:
            var id = Int(table[unsafe_offset=slot])
            if id < 0:
                table[unsafe_offset=slot] = Int32(self.count)
                self.hashes.append(hash)
                for w in range(width):
                    self.words[w].append(
                        Pointer[Int, MutAnyOrigin](unsafe_from_address=ptrs[w])[
                            unsafe_offset=row
                        ]
                    )
                self.count += 1
                return self.count - 1
            if hashes[unsafe_offset=id] == hash:
                var same = True
                for w in range(width):
                    var stored = self.words[w].unsafe_ptr()[unsafe_offset=id]
                    var mine = Pointer[Int, MutAnyOrigin](
                        unsafe_from_address=ptrs[w]
                    )[unsafe_offset=row]
                    if stored != mine:
                        same = False
                        break
                if same:
                    return id
            slot = (slot + 1) & mask


struct _GroupPart(Movable):
    """One hash part of a worker's inserted groups: their keys in an
    index, reducer states, first rows, and key values (the rows that
    introduced them, taken from each morsel)."""

    var index: _PartIndex
    var states: List[Reducer]
    var firsts: List[Int]
    var keys: List[DataFrame]

    def __init__(out self, width: Int):
        self.index = _PartIndex(width)
        self.states = List[Reducer]()
        self.firsts = List[Int]()
        self.keys = List[DataFrame]()

    def absorb(mut self, var other: Self, width: Int) raises:
        """Merge another worker's groups of the same hash part: each is
        probed by its stored hash and words, new ones appended, states
        merged by id, first rows the earliest."""
        var n = other.index.count
        if n == 0:
            return
        var ptrs = List[Int](capacity=width)
        for w in range(width):
            ptrs.append(Int(other.index.words[w].unsafe_ptr()))
        var mapping = List[Int](capacity=n)
        var fresh = List[Int]()
        var before = self.index.count
        for g in range(n):
            var id = self.index.find_or_insert(
                other.index.hashes[g], ptrs, width, g
            )
            mapping.append(id)
            if id >= before + len(fresh):
                fresh.append(g)
                self.firsts.append(other.firsts[g])
            elif other.firsts[g] < self.firsts[id]:
                self.firsts[id] = other.firsts[g]
        if len(fresh) > 0:
            var keys: DataFrame
            if len(other.keys) == 1:
                keys = other.keys[0].copy()
            else:
                keys = concat(other.keys)
            self.keys.append(keys.take(fresh))
        var count = self.index.count
        for e in range(len(self.states)):
            self.states[e].grow(count)
            self.states[e].merge(other.states[e], groups=mapping)

    def into_state(
        mut self,
        names: List[String],
        dtypes: List[DataType],
        outputs: List[Expr],
        schema: DataFrame,
    ) raises -> _StreamReduction:
        var keys: DataFrame
        if len(self.keys) == 0:
            keys = schema.copy()
        elif len(self.keys) == 1:
            keys = self.keys[0].copy()
        else:
            keys = concat(self.keys)
        var reducers = List[Reducer]()
        swap(reducers, self.states)
        var firsts = List[Int]()
        swap(firsts, self.firsts)
        return _StreamReduction(
            keys=keys^,
            states=reducers^,
            names=names,
            dtypes=dtypes,
            outputs=outputs,
            grouped=True,
            firsts=firsts^,
            rows=0,
        )


struct _GroupedSink(Movable):
    """A worker's grouped reduce sink, in one of two modes the plan picks
    for every worker alike from a sample of the keys.

    Few groups: every morsel's rows are hashed on their keys, bucketed by
    hash part, and inserted into that part's index; the reducer states of
    plain reductions update in place per row, other reductions reduce the
    part's rows into a local state merged by id. No state is built per
    morsel: DuckDB's thread-local aggregate table, radix-partitioned.

    Many groups: the morsels' rows are kept as they come, and the finish
    runs the eager partitioned group-by over all the workers' rows: one
    scatter of every row into buckets and cache-local bucket encodes,
    which inserting rows one at a time into tables that outgrow the
    caches does not match (ClickBench q32, 10M groups: 850 ms inserted
    against 300 scattered). Polars' streaming group-by and DuckDB's sink
    partition and finalize the same way once their tables grow. When the
    steps only filter, what is kept is each morsel's selection over the
    source frame, and the rows are gathered once at the finish (late
    materialization); otherwise the morsels' output frames.
    """

    var names: List[String]
    var dtypes: List[DataType]
    var outputs: List[Expr]
    var expressions: List[Expr]
    # One reducer state per reduction node, in the order the template's
    # outputs name them: the state's expression and node.
    var state_exprs: List[Int]
    var state_nodes: List[Int]
    var key_names: List[String]
    var key_schema: DataFrame
    var parts: List[_GroupPart]
    var count: Int
    var collecting: Bool
    var collected: List[DataFrame]
    var batch_size: Int
    # Selections over the source frame: a morsel's offset, length and
    # mask (an empty mask: every row of the morsel).
    var offsets: List[Int]
    var lengths: List[Int]
    var masks: List[BoolColumn]
    var width: Int
    # Scratch for a morsel's rows, kept across morsels: fresh lists of
    # this size are returned to the kernel by the allocator and faulted
    # in again every morsel.
    var part_of: List[Int]
    var order: List[Int]
    var ids: List[Int]

    def __init__(
        out self,
        sample: DataFrame,
        expressions: List[Expr],
        keys: List[String],
        collecting: Bool,
        batch_size: Int,
    ) raises:
        var empty = sample.clear()
        var template = _StreamReduction(empty, expressions, keys)
        self.names = template.names.copy()
        self.dtypes = template.dtypes.copy()
        self.outputs = template.outputs.copy()
        self.expressions = expressions.copy()
        self.state_exprs = List[Int]()
        self.state_nodes = List[Int]()
        for e in range(len(expressions)):
            for i in range(len(expressions[e]._nodes)):
                if is_reduction(expressions[e]._nodes[i].op):
                    self.state_exprs.append(e)
                    self.state_nodes.append(i)
        self.key_names = keys.copy()
        var key_columns = _expand_struct_keys(empty._subset_keys(keys))
        self.key_schema = DataFrame(key_columns.copy(), height=0)
        self.count = 0
        self.collecting = collecting
        self.collected = List[DataFrame]()
        self.batch_size = batch_size
        self.offsets = List[Int]()
        self.lengths = List[Int]()
        self.masks = List[BoolColumn]()
        self.width = (
            len(_equality_words(key_columns)) if len(key_columns) > 0 else 0
        )
        self.part_of = List[Int]()
        self.order = List[Int]()
        self.ids = List[Int]()
        self.parts = List[_GroupPart](capacity=1 << _PART_BITS)
        if not collecting:
            self.parts.append(self._empty_part(empty))

    def _empty_part(self, sample: DataFrame) raises -> _GroupPart:
        """A part with one empty state per expression, so the workers'
        parts merge expression by expression."""
        var part = _GroupPart(self.width)
        for s in range(len(self.state_exprs)):
            var bound = bind(
                self.expressions[self.state_exprs[s]], sample._columns
            )
            part.states.append(
                _new_reducer(bound, bound.expr._nodes[self.state_nodes[s]], 0)
            )
        return part^

    def _repartition(mut self, sample: DataFrame) raises:
        """Split the single part into 2^_PART_BITS by the groups' stored
        hashes, the hashes every row's part is taken from."""
        var whole = self.parts.pop(0)
        var keys: DataFrame
        if len(whole.keys) == 0:
            keys = self.key_schema.copy()
        elif len(whole.keys) == 1:
            keys = whole.keys[0].copy()
        else:
            keys = concat(whole.keys).rechunk()
        var n = keys.height()
        var count = 1 << _PART_BITS
        var members = List[List[Int]](length=count, fill=List[Int]())
        var shift = UInt64(64 - _PART_BITS)
        for g in range(n):
            members[Int(whole.index.hashes[g] >> shift)].append(g)
        var ptrs = List[Int](capacity=self.width)
        for w in range(self.width):
            ptrs.append(Int(whole.index.words[w].unsafe_ptr()))
        for p in range(count):
            var part = self._empty_part(sample)
            if len(members[p]) > 0:
                var part_keys = keys.take(members[p])
                for g in members[p]:
                    _ = part.index.find_or_insert(
                        whole.index.hashes[g], ptrs, self.width, g
                    )
                    part.firsts.append(whole.firsts[g])
                part.keys.append(part_keys^)
                for s in range(len(part.states)):
                    part.states[s].grow(len(members[p]))
                    part.states[s].merge(whole.states[s], sources=members[p])
            self.parts.append(part^)

    def select(mut self, offset: Int, length: Int, mask: BoolColumn):
        """Keep a filter-only morsel's selection over the source frame."""
        self.offsets.append(offset)
        self.lengths.append(length)
        self.masks.append(mask.copy())

    def absorb(mut self, input: DataFrame, offset: Int) raises -> Bool:
        """Take the morsel's rows (frame rows `offset` on). False, with
        nothing taken, when the keys cannot be partitioned (nested)."""
        if self.collecting:
            self.collected.append(input.copy())
            return True
        var key_columns = _expand_struct_keys(
            input._subset_keys(self.key_names)
        )
        for column in key_columns:
            if column.dtype().is_nested() or not encodable(column):
                return False
        var n = input.height()
        if n == 0:
            return True
        if len(self.parts) == 1 and self.count >= _SPLIT_GROUPS:
            self._repartition(input)
        var part_count = len(self.parts)
        var shift = UInt64(64 - _PART_BITS)
        if len(self.part_of) < n:
            self.part_of = List[Int](unsafe_uninit_length=n)
            self.order = List[Int](unsafe_uninit_length=n)
            self.ids = List[Int](unsafe_uninit_length=n)
        ref part_of = self.part_of
        var starts = List[Int](length=part_count + 1, fill=0)
        # The words define key equality; their hash is the index's hash
        # and, by its top bits, the row's part, so the same key lands in
        # the same part with the same hash on every worker and morsel.
        var words = _equality_words(key_columns)
        var row_hashes = List[UInt64](capacity=n)
        for r in range(n):
            row_hashes.append(_word_hash(words, r))
        var hashes_ptr = row_hashes.unsafe_ptr()
        var parts_ptr = part_of.unsafe_ptr()
        var starts_ptr = starts.unsafe_ptr()
        if part_count == 1:
            for r in range(n):
                parts_ptr[unsafe_offset=r] = 0
            starts[1] = n
        else:
            for r in range(n):
                var p = Int(hashes_ptr[unsafe_offset=r] >> shift)
                parts_ptr[unsafe_offset=r] = p
                starts_ptr[unsafe_offset=p + 1] += 1
        for p in range(part_count):
            starts[p + 1] += starts[p]
        var cursor = starts.copy()
        var cursor_ptr = cursor.unsafe_ptr()
        ref order = self.order
        var order_ptr = order.unsafe_ptr()
        for r in range(n):
            var p = parts_ptr[unsafe_offset=r]
            var at = cursor_ptr[unsafe_offset=p]
            order_ptr[unsafe_offset=at] = r
            cursor_ptr[unsafe_offset=p] = at + 1
        ref ids = self.ids
        var rows_ptr = order.unsafe_ptr()
        var ids_ptr = ids.unsafe_ptr()
        # Each expression bound to this morsel's columns once; the plain
        # reductions' source columns found once.
        var bounds = List[BoundExpr](capacity=len(self.expressions))
        var sources = List[Int](capacity=len(self.expressions))
        for e in range(len(self.expressions)):
            bounds.append(bind(self.expressions[e], input._columns))
            sources.append(_plain_reduction(bounds[e], input._columns))
        # A plain reduction of a computed input (`(price * (1 - discount))
        # .sum()`): the input evaluated once over the morsel's rows, then
        # the same in-place update as a column's. Otherwise each part
        # gathers its rows and runs the general reduce job (PDS-H q15 at
        # 600K rows: 3.4 ms against 1.7).
        var computed = List[Series](capacity=len(self.state_exprs))
        var has_computed = List[Bool](capacity=len(self.state_exprs))
        for s in range(len(self.state_exprs)):
            var e = self.state_exprs[s]
            ref node = bounds[e].expr._nodes[self.state_nodes[s]]
            has_computed.append(False)
            computed.append(input._columns[0].slice(0, 0))
            if sources[e] >= 0 or node.op == LEN or node.left < 0:
                continue
            if not _plain_op(node.op) or node.right >= 0:
                continue
            var evaluated = input.select_exprs(
                [
                    subtree(self.expressions[e], node.left).alias(
                        "__pipeline_input"
                    )
                ],
                batch_size=self.batch_size,
            )
            if evaluated.height() != n or not _plain_column(
                evaluated._columns[0]
            ):
                continue
            computed[s] = evaluated._columns[0].copy()
            has_computed[s] = True
        var ptrs = List[Int](capacity=self.width)
        for w in range(self.width):
            ptrs.append(Int(words[w].unsafe_ptr()))
        for p in range(part_count):
            var lo = starts[p]
            var hi = starts[p + 1]
            if hi == lo:
                continue
            ref part = self.parts[p]
            var before = part.index.count
            var fresh = List[Int]()
            for k in range(lo, hi):
                var r = rows_ptr[unsafe_offset=k]
                var id = part.index.find_or_insert(
                    hashes_ptr[unsafe_offset=r], ptrs, self.width, r
                )
                ids_ptr[unsafe_offset=k] = id
                if id >= before + len(fresh):
                    fresh.append(r)
                    part.firsts.append(offset + r)
            var count = part.index.count
            self.count += count - before
            if len(fresh) > 0:
                var columns = List[Series](capacity=len(key_columns))
                for column in key_columns:
                    columns.append(column.take(fresh))
                part.keys.append(DataFrame(columns^, height=len(fresh)))
            for s in range(len(self.state_exprs)):
                var e = self.state_exprs[s]
                ref node = bounds[e].expr._nodes[self.state_nodes[s]]
                part.states[s].grow(count)
                if node.op == LEN:
                    var counts = part.states[s].counts.unsafe_ptr()
                    for k in range(lo, hi):
                        counts[unsafe_offset=ids_ptr[unsafe_offset=k]] += 1
                    continue
                var source = sources[e]
                if source >= 0:
                    _update_plain(
                        part.states[s],
                        input._columns[source],
                        rows_ptr.unsafe_offset(lo),
                        ids_ptr.unsafe_offset(lo),
                        hi - lo,
                        node.op,
                    )
                    continue
                if has_computed[s]:
                    _update_plain(
                        part.states[s],
                        computed[s],
                        rows_ptr.unsafe_offset(lo),
                        ids_ptr.unsafe_offset(lo),
                        hi - lo,
                        node.op,
                    )
                    continue
                # Any other reduction: the part's rows reduced into a local
                # state over the same ids, merged into the part's by id.
                var selected = List[Int](capacity=hi - lo)
                var local_ids = List[Int](capacity=hi - lo)
                for k in range(lo, hi):
                    selected.append(order[k])
                    local_ids.append(ids[k])
                var local = input.take(selected)
                var job = _ReduceJob[8](
                    bind(self.expressions[e], local._columns),
                    local._columns,
                    List[Series](),
                    node,
                    0,
                    local.height(),
                    4096,
                    True,
                    ArcPointer(local_ids^),
                    count,
                )
                job.run()
                part.states[s].merge(job^.into_reducer())
        # Read through pointers above; destroyed here, not at their last
        # named use.
        _ = words^
        _ = row_hashes^
        _ = cursor^
        return True

    def into_parts(
        mut self, sample: DataFrame, split: Bool
    ) raises -> List[_GroupPart]:
        """The inserted parts, in part order, for combining: one when no
        worker's table grew past the split (small aggregates combine
        and finish as one table, as DuckDB's do before they partition),
        otherwise 2^_PART_BITS, so every worker's parts line up."""
        if split and len(self.parts) == 1:
            self._repartition(sample)
        var parts = List[_GroupPart]()
        swap(parts, self.parts)
        return parts^


struct _PartJob(Job):
    """Finish some hash parts: merge the workers' parts of each through
    the first one's index (stored hashes and words, nothing encoded
    again). Parts never share a key, so they finish independently."""

    var parts: List[List[_GroupPart]]
    var states: List[_StreamReduction]
    var width: Int
    var names: List[String]
    var dtypes: List[DataType]
    var outputs: List[Expr]
    var schema: DataFrame

    def __init__(
        out self,
        width: Int,
        names: List[String],
        dtypes: List[DataType],
        outputs: List[Expr],
        schema: DataFrame,
    ):
        self.parts = List[List[_GroupPart]]()
        self.states = List[_StreamReduction]()
        self.width = width
        self.names = names.copy()
        self.dtypes = dtypes.copy()
        self.outputs = outputs.copy()
        self.schema = schema.copy()

    def run(mut self) raises:
        while len(self.parts) > 0:
            var workers = self.parts.pop(0)
            var base = workers.pop(0)
            while len(workers) > 0:
                base.absorb(workers.pop(0), self.width)
            self.states.append(
                base.into_state(
                    self.names, self.dtypes, self.outputs, self.schema
                )
            )


struct _PipelineJob(Job):
    """One worker's run of the pipeline: morsels from the shared cursor
    through every step into this worker's own sink state."""

    var frame: ArcPointer[DataFrame]
    var cursor: Int
    # Morsel ranges as (offset, length) pairs, none crossing a chunk
    # boundary of the frame's columns (a slice within one Parquet row
    # group is a view; one across two is copied when a kernel reads it).
    var ranges: ArcPointer[List[Int]]
    var steps: List[Step]
    # Build frames and prepared indexes of the join steps, shared.
    var joins: ArcPointer[List[DataFrame]]
    var indexes: ArcPointer[List[Optional[PreparedHashIndex]]]
    var expressions: List[Expr]
    # Group keys of a grouped reduce sink (none: ungrouped or materialize).
    var keys: List[String]
    var ordered: Bool
    # Grouped sink: inserted into part tables, or collected for the
    # eager partitioned group-by (the plan decides for every worker).
    var insert: Bool
    # Every step is a filter: a collecting sink keeps the morsels'
    # selections over the source frame rather than their rows.
    var filter_only: Bool
    var batch_size: Int
    # Top-k sink (a materialize sink keeping the k best rows by the sort
    # keys): this worker's candidates in `parts`, their row count, and
    # the running bound on the first key once k candidates are held
    # (`bounded`: the key is an input column no step rewrites, so the
    # bound narrows a morsel before any step reads it).
    var top: Int
    var top_names: List[String]
    var top_descending: List[Bool]
    var top_nulls_last: List[Bool]
    var bounded: Bool
    var candidate_rows: Int
    # Candidate rows that trigger the next fold; doubles after each.
    var fold_at: Int
    var bound: _TopBound
    var shared_bound: _SharedBound
    # Whether the first sort key is a float (the shared bound's key kind).
    var floating_key: Bool
    # Materialize sink: the morsels' frames and their sequence numbers.
    var parts: List[DataFrame]
    var sequences: List[Int]
    # Reduce sink: this worker's merged state (ungrouped: one state;
    # grouped keys the key index cannot word-encode: one state per hash
    # part, merged from each morsel's state).
    var reduction: List[_StreamReduction]
    # Grouped sink: rows inserted straight into per-part key indexes and
    # reducer states (`_GroupedSink`).
    var grouped: List[_GroupedSink]
    var rows: Int
    # Rows fed, then each step's input and output rows, for the report.
    var counts: List[Int]

    def __init__(
        out self,
        frame: ArcPointer[DataFrame],
        cursor: Int,
        ranges: ArcPointer[List[Int]],
        steps: List[Step],
        joins: ArcPointer[List[DataFrame]],
        indexes: ArcPointer[List[Optional[PreparedHashIndex]]],
        expressions: List[Expr],
        keys: List[String],
        ordered: Bool,
        insert: Bool,
        batch_size: Int,
        top: Int,
        top_names: List[String],
        top_descending: List[Bool],
        top_nulls_last: List[Bool],
        bounded: Bool,
        shared_bound: _SharedBound,
        floating_key: Bool,
    ):
        self.shared_bound = shared_bound.copy()
        self.floating_key = floating_key
        self.top = top
        self.top_names = top_names.copy()
        self.top_descending = top_descending.copy()
        self.top_nulls_last = top_nulls_last.copy()
        self.bounded = bounded
        self.candidate_rows = 0
        self.fold_at = 4 * max(1, top)
        self.bound = _TopBound()
        self.frame = frame.copy()
        self.cursor = cursor
        self.ranges = ranges.copy()
        self.steps = steps.copy()
        self.joins = joins.copy()
        self.indexes = indexes.copy()
        self.expressions = expressions.copy()
        self.keys = keys.copy()
        self.ordered = ordered
        self.insert = insert
        self.filter_only = True
        for step in steps:
            if step.kind != STEP_FILTER:
                self.filter_only = False
        self.batch_size = batch_size
        self.parts = List[DataFrame]()
        self.sequences = List[Int]()
        self.reduction = List[_StreamReduction]()
        self.grouped = List[_GroupedSink]()
        self.rows = 0
        # Rows fed, then each step's rows in and out, then each step's
        # busy nanoseconds on this worker.
        self.counts = List[Int](length=3 * len(steps) + 1, fill=0)

    def _fold_candidates(mut self) raises:
        """Fold this worker's candidates to the k best (DuckDB's heap), and
        tighten the bound from the k-th, shared with every worker."""
        var merged = _top_rows(
            concat(self.parts),
            self.top_names,
            self.top_descending,
            self.top_nulls_last,
            self.top,
            1,
        )
        self.candidate_rows = merged.height()
        if self.bounded:
            self.bound = _TopBound(
                merged, self.top_names[0], self.top_descending[0], self.top
            )
            self.shared_bound.publish(self.bound)
        self.parts = [merged^]

    def run(mut self) raises:
        var count = len(self.ranges[]) // 2
        var cursor = _Cursor(self.cursor)
        while True:
            var k = cursor.take(1)
            if k >= count:
                # The worker's leftover candidates folded to k, so the
                # finish sorts at most k rows a worker.
                if self.top > 0 and self.candidate_rows > self.top:
                    self._fold_candidates()
                return
            var offset = self.ranges[][2 * k]
            var length = self.ranges[][2 * k + 1]
            var morsel = Morsel(self.frame[].slice(offset, length), k)
            self.counts[0] += length
            if self.top > 0 and self.bounded:
                self.shared_bound.tighten(
                    self.bound,
                    self.top_names[0],
                    self.top_descending[0],
                    self.floating_key,
                )
            if self.top > 0 and self.bound.kind != _BOUND_NONE:
                # Narrow to the rows that can still enter the top k when
                # at most one in eight can: gathering a step's columns at
                # scattered rows costs more than reading them whole past
                # that (ClickBench q26's string filter on 15% of the rows:
                # 31 ms busy against 6 reading every row).
                var kept = self.bound.rows(morsel.frame)
                if kept and 8 * len(kept.value()) <= morsel.height():
                    morsel.narrow(kept.value())
            for i in range(len(self.steps)):
                self.counts[2 * i + 1] += morsel.height()
                var began = Int(perf_counter_ns())
                morsel.apply(
                    self.steps[i], self.batch_size, self.joins, self.indexes
                )
                self.counts[2 * len(self.steps) + 1 + i] += (
                    Int(perf_counter_ns()) - began
                )
                self.counts[2 * i + 2] += morsel.height()
                if morsel.height() == 0:
                    break
            if morsel.height() == 0:
                continue
            if (
                len(self.keys) > 0
                and len(self.expressions) > 0
                and not self.insert
                and self.filter_only
            ):
                if len(self.grouped) == 0:
                    self.grouped.append(
                        _GroupedSink(
                            morsel.frame,
                            self.expressions,
                            self.keys,
                            True,
                            self.batch_size,
                        )
                    )
                if morsel.selected:
                    self.grouped[0].select(offset, length, morsel.mask)
                else:
                    self.grouped[0].select(
                        offset, length, BoolColumn(List[Bool]())
                    )
                self.rows += morsel.height()
                continue
            if len(self.expressions) > 0:
                var reads = _reads(self.expressions)
                var input: DataFrame
                if reads:
                    var names = reads.take()
                    for key in self.keys:
                        if key not in names:
                            names.append(key)
                    input = morsel._compact_frame(names)
                else:
                    input = morsel.materialized()
                if len(self.keys) > 0 and len(self.reduction) == 0:
                    # Rows go straight into this worker's per-part key
                    # indexes and reducer states (DuckDB's thread-local
                    # partitioned aggregate table), when the keys encode
                    # as words; otherwise each morsel's state is split
                    # into parts and merged below.
                    if len(self.grouped) == 0:
                        self.grouped.append(
                            _GroupedSink(
                                input,
                                self.expressions,
                                self.keys,
                                not self.insert,
                                self.batch_size,
                            )
                        )
                    if self.grouped[0].absorb(input, offset):
                        self.rows += input.height()
                        continue
                    if self.grouped[0].count > 0:
                        raise Error(
                            "grouped pipeline: keys stopped partitioning"
                        )
                    self.grouped = List[_GroupedSink]()
                # Group first rows are the frame's rows, so first-occurrence
                # order survives morsels taken out of order.
                var state = _StreamReduction(input, self.expressions, self.keys)
                state.shift_firsts(offset)
                self.rows += state.rows
                if len(self.keys) == 0:
                    if len(self.reduction) == 0:
                        self.reduction.append(state^)
                    else:
                        self.reduction[0].merge(state)
                elif len(self.reduction) == 0:
                    # The worker's state is kept in hash parts, so every
                    # morsel's groups go into tables small enough to stay
                    # in cache; one table over all the worker's groups
                    # cost ClickBench q32 (10M groups) 280 ns a probe.
                    self.reduction = state.split(_PART_BITS)
                else:
                    var pieces = state.split(_PART_BITS)
                    var p = 0
                    while len(pieces) > 0:
                        var piece = pieces.pop(0)
                        if piece.group_count() > 0:
                            var batch = List[_StreamReduction]()
                            batch.append(piece^)
                            self.reduction[p].merge_all(batch, parallel=False)
                        p += 1
            elif self.top > 0:
                var output = morsel.materialized()
                var h = output.height()
                var places = List[Int64](capacity=h)
                var base = Int64(morsel.sequence) << 40
                for r in range(h):
                    places.append(base + Int64(r))
                output = output.with_column(
                    Series(_ORDER_COLUMN, Column[Int64](places^))
                )
                self.candidate_rows += h
                self.parts.append(output^)
                # Folds come at doubling candidate counts, from a few rows
                # past k up to a few thousand: the first folds are cheap
                # and tighten the shared bound fast (a bound from 40 rows
                # kept 10% of ClickBench q26's rows; from 8K rows, 0.1%),
                # and the fold's setup per call (dense ranks of a string
                # key) is then paid once per thousands of rows, not per
                # morsel (650 folds cost q26 44 ms).
                if self.candidate_rows > self.fold_at and len(self.parts) > 1:
                    self._fold_candidates()
                    self.fold_at = min(_FOLD_ROWS, 2 * self.fold_at)
            else:
                self.parts.append(morsel.materialized())
                self.sequences.append(morsel.sequence)


def _top_rows(
    candidates: DataFrame,
    names: List[String],
    descending: List[Bool],
    nulls_last: List[Bool],
    k: Int,
    threads: Int,
) raises -> DataFrame:
    """The k best candidate rows in order, ties by their place in the
    source. With several keys, the rows are first cut to those whose first
    key is at or better than the k-th best first key (a numeric key, no
    null there): later keys only settle ties, so ranking a string key over
    every candidate (what a multi-key selection does) is spared for all
    but the tied rows, as a lexicographic heap compares them."""
    var pool = candidates.copy()
    if len(names) > 1 and pool.height() > k:
        var first = pool.take(
            pool._arg_sort_head(
                [names[0]], [descending[0]], [nulls_last[0]], k, threads
            )
        )
        var bound = _TopBound(first, names[0], descending[0], k)
        if bound.kind != _BOUND_NONE:
            var kept = bound.rows(pool)
            if kept:
                pool = pool.take(kept.value())
    return pool.take(
        pool._arg_sort_head(
            _with_order(names),
            _with_false(descending),
            _with_false(nulls_last),
            k,
            threads,
        )
    )


def _with_order(names: List[String]) -> List[String]:
    var out = names.copy()
    out.append(_ORDER_COLUMN)
    return out^


def _with_false(flags: List[Bool]) -> List[Bool]:
    var out = flags.copy()
    out.append(False)
    return out^


def _bits_at(address: Int, count: Int, bit: Int, n: Int) -> UInt64:
    """`n` (at most 64) bits of a `count`-byte bitmap from `bit` on, bit
    0 lowest; bits past the bitmap read as zero."""
    var bytes = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=address)
    var byte = bit >> 3
    var shift = bit & 7
    var low = UInt64(0)
    if byte + 8 <= count:
        low = Pointer[UInt64, MutAnyOrigin](
            unsafe_from_address=address + byte
        )[]
    else:
        for b in range(8):
            if byte + b < count:
                low |= UInt64(bytes[unsafe_offset=byte + b]) << UInt64(8 * b)
    var word = low >> UInt64(shift)
    if shift > 0 and byte + 8 < count:
        word |= UInt64(bytes[unsafe_offset=byte + 8]) << UInt64(64 - shift)
    if n < 64:
        word &= (UInt64(1) << UInt64(n)) - 1
    return word


def _or_bits_at(address: Int, count: Int, bit: Int, word: UInt64):
    """OR `word`'s bits into a `count`-byte bitmap from `bit` on."""
    var bytes = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=address)
    var byte = bit >> 3
    var shift = bit & 7
    var low = word << UInt64(shift)
    for b in range(8):
        if byte + b < count:
            bytes[unsafe_offset=byte + b] |= UInt8(low >> UInt64(8 * b))
    if shift > 0 and byte + 8 < count:
        bytes[unsafe_offset=byte + 8] |= UInt8(word >> UInt64(64 - shift))


def _merged_mask(
    offsets: List[Int],
    lengths: List[Int],
    masks: List[BoolColumn],
    height: Int,
) raises -> BoolColumn:
    """One mask over `height` rows from morsel selections: each morsel's
    rows [offset, offset + length) set as its mask says (an empty mask:
    every row), 64 bits at a time whatever the alignment of either."""
    var count = (height + 7) // 8
    var values = List[UInt8](length=count, fill=0)
    var bits = Int(values.unsafe_ptr())
    for m in range(len(offsets)):
        var low = offsets[m]
        var length = lengths[m]
        ref mask = masks[m]
        var pos = 0
        if len(mask) == 0:
            while pos < length:
                var n = min(64, length - pos)
                var word = (
                    UInt64.MAX if n == 64 else (UInt64(1) << UInt64(n)) - 1
                )
                _or_bits_at(bits, count, low + pos, word)
                pos += n
            continue
        var source = Int(mask._data[].unsafe_ptr())
        var source_count = len(mask._data[])
        var valid = Int(mask._bits[].unsafe_ptr())
        var valid_count = len(mask._bits[])
        while pos < length:
            var n = min(64, length - pos)
            var word = _bits_at(source, source_count, mask._offset + pos, n)
            if valid_count > 0:
                word &= _bits_at(valid, valid_count, mask._offset + pos, n)
            if word != 0:
                _or_bits_at(bits, count, low + pos, word)
            pos += n
    return BoolColumn(values=values^, bits=List[UInt8](), length=height)


def _morsel_ranges(
    frame: DataFrame, morsel_rows: Int, workers: Int
) -> List[Int]:
    """(offset, length) pairs covering the frame: each chunk of columns
    chunked at the same rows is cut into even pieces of about
    `morsel_rows`, and a frame without such chunks likewise. When that
    gives fewer than four morsels a worker, the pieces are cut finer so
    their count is a multiple of `workers` and the last round of morsels
    is as full as the first: ten even pieces of 600K rows on eight
    workers left two workers a whole second round (PDS-H q12 at scale
    0.1, 1.8 to 2.1 ms). Many morsels are left as they are: each costs
    its steps' setup (ClickBench q29's ninety sums, 36 morsels of 312K
    rows against 144 of 69K, 4 to 6 ms)."""
    var ends = List[Int]()
    if can_filter_aligned_chunks(frame._columns):
        ends = frame._columns[0]._chunked.value()[].ends.copy()
    else:
        ends.append(frame.height())
    var lanes = max(1, workers)
    var lengths = List[Int]()
    var starts = List[Int]()
    var pieces = List[Int]()
    var total = 0
    var start = 0
    for end in ends:
        var length = end - start
        if length > 0:
            starts.append(start)
            lengths.append(length)
            # Pieces of about `morsel_rows`: a chunk a little longer than
            # the target stays one morsel rather than two short ones, each
            # paying the steps' setup (TPC-DS q9's fifteen conditional
            # sums over 123K-row groups cut at 90K: 145 to 174 ms on one
            # thread).
            pieces.append(
                max(1, (2 * length + morsel_rows) // (2 * morsel_rows))
            )
            total += pieces[len(pieces) - 1]
        start = end
    if total > 1 and total < 4 * lanes and total % lanes != 0:
        # Cut every chunk finer by the same factor, then give the longest
        # chunks one more piece each until the count is a multiple.
        var wanted = total + lanes - total % lanes
        var scaled = 0
        for c in range(len(pieces)):
            pieces[c] = min(lengths[c], pieces[c] * wanted // total)
            scaled += pieces[c]
        while scaled < wanted:
            var longest = -1
            for c in range(len(pieces)):
                if pieces[c] < lengths[c] and (
                    longest < 0
                    or lengths[c] // pieces[c]
                    > lengths[longest] // pieces[longest]
                ):
                    longest = c
            if longest < 0:
                break
            pieces[longest] += 1
            scaled += 1
    # Boundaries inside a chunk fall on multiples of 64 rows, so a
    # morsel's validity bitmaps start on a byte: kernels that AND or
    # window bitmaps take their slow path from a bit offset (TPC-DS q9's
    # masked sums, 145 to 174 ms on one thread at 87,381-row morsels).
    var ranges = List[Int]()
    for c in range(len(pieces)):
        for p in range(pieces[c]):
            var low = starts[c] + lengths[c] * p // pieces[c]
            var high = starts[c] + lengths[c] * (p + 1) // pieces[c]
            if p > 0:
                low = starts[c] + (low - starts[c]) // 64 * 64
            if p + 1 < pieces[c]:
                high = starts[c] + (high - starts[c]) // 64 * 64
            if high > low:
                ranges.append(low)
                ranges.append(high - low)
    return ranges^


def run_pipeline(
    frame: DataFrame,
    steps: List[Step],
    joins: ArcPointer[List[DataFrame]],
    indexes: ArcPointer[List[Optional[PreparedHashIndex]]],
    expressions: List[Expr],
    keys: List[String],
    ordered: Bool,
    insert: Bool,
    workers: Int,
    morsel_rows: Int,
    batch_size: Int,
    mut counts: List[Int],
    top: Int = -1,
    top_names: List[String] = List[String](),
    top_descending: List[Bool] = List[Bool](),
    top_nulls_last: List[Bool] = List[Bool](),
    bounded: Bool = False,
) raises -> DataFrame:
    """Run the steps over `frame` in morsels on `workers` threads, into a
    materialize sink, or a reduce sink when `expressions` are reductions
    over the steps' output, grouped by `keys` when given (`ordered`: groups
    in first-occurrence order; `insert`: rows go into per-worker group
    tables rather than being collected for the eager partitioned
    group-by). `counts` receives the rows fed, then each step's input and
    output rows."""
    trace_path("pipeline.run")
    var shared = ArcPointer(frame.copy())
    var ranges = ArcPointer(
        _morsel_ranges(frame, max(1, morsel_rows), max(1, workers))
    )
    var cursor = _Cursor.new()
    var shared_bound = _SharedBound.new()
    var floating_key = False
    if top > 0 and len(top_names) > 0 and top_names[0] in frame.columns():
        floating_key = frame.column(top_names[0]).dtype().physical() in (
            DataType.FLOAT64,
            DataType.FLOAT32,
        )
    # One job per morsel at most: a tiny source (PDS-H q2's five-row
    # region frame) runs on the caller, with no crew dispatch to wait on.
    var job_count = max(1, min(workers, len(ranges[]) // 2))
    var jobs = List[_PipelineJob](capacity=job_count)
    for _ in range(job_count):
        jobs.append(
            _PipelineJob(
                shared,
                cursor.address,
                ranges,
                steps,
                joins,
                indexes,
                expressions,
                keys,
                ordered,
                insert,
                batch_size,
                top,
                top_names,
                top_descending,
                top_nulls_last,
                bounded,
                shared_bound,
                floating_key,
            )
        )
    try:
        run_jobs(jobs)
    except e:
        cursor.free()
        shared_bound.free()
        raise e
    cursor.free()
    shared_bound.free()
    counts = List[Int](length=3 * len(steps) + 1, fill=0)
    for j in range(len(jobs)):
        for i in range(len(counts)):
            counts[i] += jobs[j].counts[i]
    if len(expressions) > 0 and len(keys) > 0:
        var sink = -1
        for j in range(len(jobs)):
            if len(jobs[j].grouped) > 0:
                sink = j
        if sink < 0:
            var empty = frame.clear()
            for step in steps:
                var morsel = Morsel(empty.copy(), 0)
                morsel.apply(step, batch_size, joins, indexes)
                empty = morsel.materialized()
            return _StreamReduction(empty, expressions, keys).finish()
        if jobs[sink].grouped[0].collecting:
            # Every worker's rows, grouped once by the eager partitioned
            # group-by: one scatter of the rows into buckets, each encoded
            # on its own worker.
            # Only the columns the keys and expressions read are gathered
            # and grouped; the steps may have read others.
            var whole: DataFrame
            if len(steps) == 0 or jobs[sink].filter_only:
                var source = frame.copy()
                var reads = _reads(expressions)
                if reads:
                    var names = keys.copy()
                    for name in reads.take():
                        if name not in names:
                            names.append(name)
                    source = frame.select(names)
                if len(steps) == 0:
                    whole = source^
                else:
                    # The morsels' selections as one mask over the source
                    # frame, gathered once.
                    var offsets = List[Int]()
                    var lengths = List[Int]()
                    var masks = List[BoolColumn]()
                    for j in range(len(jobs)):
                        if len(jobs[j].grouped) == 0:
                            continue
                        ref kept = jobs[j].grouped[0]
                        for m in range(len(kept.offsets)):
                            offsets.append(kept.offsets[m])
                            lengths.append(kept.lengths[m])
                            masks.append(kept.masks[m].copy())
                    whole = source.filter(
                        _merged_mask(offsets, lengths, masks, frame.height())
                    )
            else:
                var collected = List[DataFrame]()
                for j in range(len(jobs)):
                    if len(jobs[j].grouped) == 0:
                        continue
                    var rows = List[DataFrame]()
                    swap(rows, jobs[j].grouped[0].collected)
                    while len(rows) > 0:
                        collected.append(rows.pop(0))
                # One chunk per column (merged in parallel): the
                # partitioned group-by copies chunked keys per chunk
                # (#478), and the morsels' outputs are many small ones.
                whole = concat(collected).rechunk()
            return whole.group_by(keys, maintain_order=ordered).agg(expressions)
        # Each worker's parts, part by part across the workers; every
        # worker splits into hash parts when any did.
        var split = False
        for j in range(len(jobs)):
            if len(jobs[j].grouped) > 0 and len(jobs[j].grouped[0].parts) > 1:
                split = True
        var parts = List[List[_GroupPart]]()
        for j in range(len(jobs)):
            if len(jobs[j].grouped) == 0:
                continue
            var pieces = jobs[j].grouped[0].into_parts(frame.clear(), split)
            var p = 0
            while len(pieces) > 0:
                if len(parts) <= p:
                    parts.append(List[_GroupPart]())
                parts[p].append(pieces.pop(0))
                p += 1
        ref meta = jobs[sink].grouped[0]
        return _combine_grouped(
            parts^,
            meta.width,
            meta.names,
            meta.dtypes,
            meta.outputs,
            meta.key_schema,
            ordered,
            workers,
        )
    if len(expressions) > 0:
        var merged = List[_StreamReduction]()
        for j in range(len(jobs)):
            if len(jobs[j].reduction) == 0:
                continue
            if len(merged) == 0:
                merged.append(jobs[j].reduction.pop())
            else:
                merged[0].merge(jobs[j].reduction[0])
        if len(merged) == 0:
            # No row survived: reduce an empty frame of the right schema.
            var empty = frame.clear()
            for step in steps:
                var morsel = Morsel(empty.copy(), 0)
                morsel.apply(step, batch_size, joins, indexes)
                empty = morsel.materialized()
            return _StreamReduction(empty, expressions, List[String]()).finish()
        return merged[0].finish()
    # Morsels in source order: each worker's parts are in the order it
    # took them, so a merge by sequence over the workers' lists orders all.
    # A top-k's candidates have no order to keep.
    var parts = List[DataFrame]()
    var next = List[Int](length=len(jobs), fill=0)
    if top > 0:
        for j in range(len(jobs)):
            for part in jobs[j].parts:
                parts.append(part.copy())
    while top <= 0:
        var best = -1
        var best_sequence = 0
        for j in range(len(jobs)):
            if next[j] < len(jobs[j].sequences):
                var sequence = jobs[j].sequences[next[j]]
                if best < 0 or sequence < best_sequence:
                    best = j
                    best_sequence = sequence
        if best < 0:
            break
        parts.append(jobs[best].parts[next[best]].copy())
        next[best] += 1
    if len(parts) == 0:
        var empty = frame.clear()
        for step in steps:
            var morsel = Morsel(empty.copy(), 0)
            morsel.apply(step, batch_size, joins, indexes)
            empty = morsel.materialized()
        return empty^
    if top > 0:
        # Every worker's candidates, sorted once to the k best, ties by
        # their place in the source.
        var merged = parts[0].copy() if len(parts) == 1 else concat(parts)
        merged = _top_rows(
            merged, top_names, top_descending, top_nulls_last, top, 0
        )
        return merged.drop([_ORDER_COLUMN])
    if len(parts) == 1:
        return parts[0].copy()
    return concat(parts)


def _combine_grouped(
    var parts: List[List[_GroupPart]],
    width: Int,
    names: List[String],
    dtypes: List[DataType],
    outputs: List[Expr],
    schema: DataFrame,
    ordered: Bool,
    workers: Int,
) raises -> DataFrame:
    """Combine the workers' inserted parts, the same hash parts on every
    worker: the parts of one hash merge on their own worker (they never
    share a key) and finish in parallel, interleaved by first occurrence
    when `ordered`. DuckDB combines its thread-local aggregate tables
    partition by partition the same way."""
    var count = len(parts)
    var job_count = max(1, min(workers, count))
    var jobs = List[_PartJob](capacity=job_count)
    for _ in range(job_count):
        jobs.append(_PartJob(width, names, dtypes, outputs, schema))
    for p in range(count):
        jobs[p % job_count].parts.append(parts.pop(0))
    run_jobs(jobs)
    var merged = List[_StreamReduction](capacity=count)
    for p in range(count):
        merged.append(jobs[p % job_count].states.pop(0))
    if ordered:
        for p in range(count):
            merged[p].order_by_firsts()
        return _finish_parts(merged^)
    var frames = List[DataFrame](capacity=count)
    for p in range(count):
        if merged[p].group_count() > 0:
            frames.append(merged[p].finish())
    if len(frames) == 0:
        return merged[0].finish()
    if len(frames) == 1:
        return frames[0].copy()
    return concat(frames)
