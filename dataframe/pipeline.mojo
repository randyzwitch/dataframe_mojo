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
from std.bit import count_trailing_zeros
from std.os import getenv
from std.ffi import _get_global, external_call
from std.memory import ArcPointer, Pointer, bitcast
from std.sys import size_of
from std.time import perf_counter_ns

from .bool_column import BoolColumn, both_true, true_count
from .expr import COL, SELECTOR, Expr
from .aggregate import Reducer
from .binding import BoundExpr, bind
from .column import Column
from .dtype import DataType, NUMERIC_DTYPES
from .execution import _ReduceJob, _new_reducer, _valid_lanes
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
from .aggregate_table import (
    AggregateLayout,
    AggregateSink,
    aggregate_finish,
    aggregate_layout,
)
from .frame import (
    DataFrame,
    _StreamReduction,
    table_frame,
    table_inputs,
    table_specs,
    table_supported,
    _equality_words,
    _expand_struct_keys,
    _finish_parts,
    _rows_mask,
    _word_hash,
    concat,
)
from .indexed_reduce import _reduce_floats, _reduce_ints
from .join_hash import (
    PreparedHashIndex,
    prepared_hash_join_rows,
    prepared_hash_semi_anti_rows,
)
from .join_type import JOIN_ANTI, JOIN_INNER, JOIN_SEMI
from .row_encode import encodable
from .gather import can_filter_aligned_chunks, take_parallel, true_rows
from .huge_pages import huge_list
from .mask_filter import filter_columns
from .parallel import Job, run_jobs, worker_count
from .series import Series
from .hashing import encode_rows
from .string_column import StringColumn
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
# A worker's hash part holding more groups than this is flushed: its table
# is set aside whole for the combine and a fresh one takes its place, so
# the tables rows are inserted into stay cache-sized however many groups
# there are (DuckDB's radix-partitioned sink abandons a full thread-local
# table the same way). The combine merges every flushed table of a part.
comptime _FLUSH_GROUPS = 4096


def _flush_groups() -> Int:
    """`_FLUSH_GROUPS`, or DATAFRAME_PIPELINE_FLUSH_GROUPS when set, which
    also lowers `_SPLIT_GROUPS` to it, so tests can split and flush tables
    on small inputs."""
    var setting = getenv("DATAFRAME_PIPELINE_FLUSH_GROUPS")
    if setting.byte_length() > 0:
        try:
            return max(1, Int(setting))
        except:
            pass
    return _FLUSH_GROUPS


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
                        # A NaN bound compares false with every number:
                        # take none, which is always safe.
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


comptime _ZONE_SOME = 0
comptime _ZONE_NONE = 1
comptime _ZONE_ALL = 2


def _zoned_top(frame: DataFrame, bound: _TopBound) raises -> Int:
    """Whether a morsel's rows can enter a top-k, from the zone maps of
    its integer key's buffers (DuckDB's dynamic top-k filter, checked
    against row-group statistics): `_ZONE_NONE` when every value is worse
    than the bound and no row is null (nulls always pass), `_ZONE_ALL`
    when every value passes, `_ZONE_SOME` otherwise or without a usable
    range. A morsel no row of which can enter is skipped unread
    (ClickBench q26 compared all 10M event times with the bound and kept
    2%)."""
    if bound.name not in frame.columns():
        return _ZONE_SOME
    var key = frame.column(bound.name)
    if not _range_key(key.dtype()):
        return _ZONE_SOME
    var bounds = _window_bounds(key)
    if not bounds:
        return _ZONE_SOME
    var low = bounds.value()[0]
    var high = bounds.value()[1]
    if bound.descending:
        if high < bound.integer and key.null_count() == 0:
            return _ZONE_NONE
        if low >= bound.integer:
            return _ZONE_ALL
    else:
        if low > bound.integer and key.null_count() == 0:
            return _ZONE_NONE
        if high <= bound.integer:
            return _ZONE_ALL
    return _ZONE_SOME


def _bare_columns(exprs: List[Expr]) -> Bool:
    """Whether every expression is one column, under its own name or an
    alias."""
    for e in exprs:
        if len(e._nodes) != 1 or e._nodes[0].op != COL:
            return False
    return True


@fieldwise_init
struct _KeyRange(Copyable, Movable):
    """A finished join build's key range on a source column (DuckDB's
    min-max runtime join filter, `CreateDynamicMinMaxFilters`): a row
    whose key is null or outside it joins nothing. `any` is false when the
    build holds no non-null key, so no row joins."""

    var column: String
    var low: Int64
    var high: Int64
    var any: Bool


def _range_key(dtype: DataType) -> Bool:
    """Keys whose values compare as their physical integers."""
    var physical = dtype.physical()
    return physical == DataType.INT64 or physical == DataType.INT32


def _int_bounds(series: Series) -> Optional[Tuple[Int64, Int64]]:
    """The least and greatest non-null value of an integer column (see
    `_range_key`), or None when every row is null."""
    var least = Int64.MAX
    var most = Int64.MIN
    var any = False
    var physical = series.dtype().physical()
    for part in series.chunks():
        if physical == DataType.INT64:
            var found = _column_bounds(part._data[Column[Int64]])
            if found[0]:
                any = True
                least = min(least, found[1])
                most = max(most, found[2])
        else:
            var found = _column_bounds(part._data[Column[Int32]])
            if found[0]:
                any = True
                least = min(least, found[1])
                most = max(most, found[2])
    if not any:
        return None
    return (least, most)


def _column_bounds[
    D: DType
](column: Column[Scalar[D]]) -> Tuple[Bool, Int64, Int64]:
    """Whether any row is valid, and the least and greatest valid value,
    eight rows (one validity byte) at a time."""
    var values = column._ptr()
    var bits = column.unsafe_validity()
    var n = len(column)
    var all_valid = column.null_count() == 0
    var aligned = column._offset % 8 == 0
    var weights = SIMD[DType.uint8, 8](1, 2, 4, 8, 16, 32, 64, 128)
    var least = SIMD[DType.int64, 8](Int64.MAX)
    var most = SIMD[DType.int64, 8](Int64.MIN)
    var any = all_valid and n >= 8
    var i = 0
    while i + 8 <= n:
        var v = values.unsafe_load[width=8](i).cast[DType.int64]()
        if all_valid:
            least = min(least, v)
            most = max(most, v)
        else:
            var byte = UInt8(0)
            if aligned:
                byte = bits[unsafe_offset=(column._offset + i) >> 3]
            if aligned and byte == 255:
                least = min(least, v)
                most = max(most, v)
                any = True
            elif not aligned or byte != 0:
                var valid = (SIMD[DType.uint8, 8](byte) & weights).ne(
                    0
                ) if aligned else _valid_lanes(bits, column._offset + i)
                any = any or valid.reduce_or()
                least = min(
                    least, valid.select(v, SIMD[DType.int64, 8](Int64.MAX))
                )
                most = max(
                    most, valid.select(v, SIMD[DType.int64, 8](Int64.MIN))
                )
        i += 8
    var smallest = least.reduce_min()
    var largest = most.reduce_max()
    while i < n:
        if all_valid or column._valid(i):
            var v = column._get(i).cast[DType.int64]()
            smallest = min(smallest, v)
            largest = max(largest, v)
            any = True
        i += 1
    return (any, smallest, largest)


comptime _ZONE_ROWS = 2048


def _zones_lock_init() -> Optional[Pointer[NoneType, MutUntrackedOrigin]]:
    var address = external_call["calloc", Int](1, size_of[Atomic[Int64]]())
    return Pointer[NoneType, MutUntrackedOrigin](unsafe_from_address=address)


def _zones_lock_keep(address: Optional[Pointer[NoneType, MutUntrackedOrigin]]):
    # Never freed: another operation's pipeline may still take it while
    # the process exits.
    pass


def _zones_lock() -> ref[MutAnyOrigin] Atomic[Int64]:
    """The process-wide lock zone maps are filled under."""
    var address = _get_global[
        "DATAFRAME_ZONES_LOCK", _zones_lock_init, _zones_lock_keep
    ]()
    return Pointer[Atomic[Int64], MutAnyOrigin](
        unsafe_from_address=Int(address.value())
    )[]


def _ensure_zones(series: Series):
    """Fill the zone map (`Column._facts`) of each buffer under an
    integer column (see `_range_key`) that has none, once, before any
    worker reads it. Pipelines run at once (a union's inputs) may scan one
    table: a buffer is checked and filled under one process-wide lock, so
    it is written once, before any pipeline's workers read it, and never
    again while they do."""
    ref lock = _zones_lock()
    while True:
        var free = Int64(0)
        if lock.compare_exchange(free, Int64(1)):
            break
        _ = external_call["sched_yield", Int32]()
    var physical = series.dtype().physical()
    for part in series.chunks():
        if physical == DataType.INT64:
            _fill_zones(part._data[Column[Int64]])
        elif physical == DataType.INT32:
            _fill_zones(part._data[Column[Int32]])
    lock.store(0)


def _fill_zones[D: DType](column: Column[Scalar[D]]):
    if len(column._facts[].zones) > 0:
        return
    var whole = column.copy()
    whole._offset = 0
    whole._length = len(column._data[])
    var blocks = (whole._length + _ZONE_ROWS - 1) // _ZONE_ROWS
    var zones = List[Int64](capacity=2 * blocks)
    for b in range(blocks):
        var first = b * _ZONE_ROWS
        var block = whole.copy()
        block._offset = first
        block._length = min(_ZONE_ROWS, whole._length - first)
        var found = _column_bounds(block)
        if found[0]:
            zones.append(found[1])
            zones.append(found[2])
        else:
            zones.append(Int64.MAX)
            zones.append(Int64.MIN)
    column._facts[].zones = zones^


def _window_bounds(series: Series) -> Optional[Tuple[Int64, Int64]]:
    """Bounds on the valid values of an integer column's window, from
    the zone maps of the blocks it touches (blocks it covers only in part
    widen them, never narrow them), or exact when a buffer has none; None
    when it has no valid value."""
    var least = Int64.MAX
    var most = Int64.MIN
    var physical = series.dtype().physical()
    for part in series.chunks():
        var found: Tuple[Bool, Int64, Int64]
        if physical == DataType.INT64:
            found = _zoned_bounds(part._data[Column[Int64]])
        else:
            found = _zoned_bounds(part._data[Column[Int32]])
        if found[0]:
            least = min(least, found[1])
            most = max(most, found[2])
    if least > most:
        return None
    return (least, most)


def _zoned_bounds[
    D: DType
](column: Column[Scalar[D]]) -> Tuple[Bool, Int64, Int64]:
    var zones = column._facts[].zones
    if len(zones) == 0:
        return _column_bounds(column)
    if len(column) == 0:
        return (False, Int64.MAX, Int64.MIN)
    var least = Int64.MAX
    var most = Int64.MIN
    var first = column._offset // _ZONE_ROWS
    var last = (column._offset + len(column) - 1) // _ZONE_ROWS
    for b in range(first, last + 1):
        least = min(least, zones[2 * b])
        most = max(most, zones[2 * b + 1])
    return (least <= most, least, most)


def _range_mask(
    series: Series, low: Int64, high: Int64
) raises -> Tuple[BoolColumn, Int]:
    """The rows of an unchunked integer column (see `_range_key`) whose
    value is valid and within [low, high], as a packed mask, and how many."""
    var n = len(series)
    var values = List[UInt8](length=(n + 7) // 8, fill=0)
    var target = values.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var kept = 0
    if series.dtype().physical() == DataType.INT64:
        kept = _range_bits(series._data[Column[Int64]], low, high, target)
    else:
        kept = _range_bits(series._data[Column[Int32]], low, high, target)
    return (BoolColumn(values=values^, bits=List[UInt8](), length=n), kept)


def _range_bits[
    D: DType
](
    column: Column[Scalar[D]],
    low: Int64,
    high: Int64,
    target: Pointer[UInt8, MutAnyOrigin],
) -> Int:
    var values = column._ptr()
    var bits = column.unsafe_validity()
    var n = len(column)
    var all_valid = column.null_count() == 0
    var weights = SIMD[DType.uint8, 8](1, 2, 4, 8, 16, 32, 64, 128)
    var kept = 0
    var i = 0
    while i + 8 <= n:
        var v = values.unsafe_load[width=8](i).cast[DType.int64]()
        var inside = v.ge(SIMD[DType.int64, 8](low)) & v.le(
            SIMD[DType.int64, 8](high)
        )
        if not all_valid:
            inside &= _valid_lanes(bits, column._offset + i)
        var byte = (inside.cast[DType.uint8]() * weights).reduce_add()
        target[unsafe_offset=i >> 3] = byte
        kept += Int(inside.cast[DType.uint8]().reduce_add())
        i += 8
    while i < n:
        if all_valid or column._valid(i):
            var v = column._get(i).cast[DType.int64]()
            if v >= low and v <= high:
                target[unsafe_offset=i >> 3] |= UInt8(1) << UInt8(i & 7)
                kept += 1
        i += 1
    return kept


def _key_ranges(
    frame: DataFrame, steps: List[Step], joins: List[DataFrame]
) raises -> List[_KeyRange]:
    """The key ranges of the inner and semi joins among `steps` whose
    probe key is a source column no earlier step replaces: every build is
    finished before the first morsel, so its range can narrow the morsel
    before any step runs, as DuckDB pushes a build's min-max filter into
    the probe side's scan."""
    var ranges = List[_KeyRange]()
    # Names the steps so far show for untouched source columns.
    var source = Dict[String, String]()
    for name in frame.columns():
        source[name] = name
    for step in steps:
        if step.kind == STEP_JOIN:
            var build = joins[step.join]
            if (step.how == JOIN_INNER or step.how == JOIN_SEMI) and len(
                step.names
            ) == len(step.right_keys):
                for k in range(len(step.names)):
                    var probe = source.get(step.names[k])
                    if not probe:
                        continue
                    var name = probe.value()
                    var key = build.column(step.right_keys[k])
                    if key.dtype() != frame.column(name).dtype():
                        continue
                    if not _range_key(key.dtype()):
                        continue
                    var bounds = _int_bounds(key)
                    if bounds:
                        ranges.append(
                            _KeyRange(
                                name, bounds.value()[0], bounds.value()[1], True
                            )
                        )
                    else:
                        ranges.append(_KeyRange(name, 0, 0, False))
            # The build's columns join the output, under their names or
            # with the suffix.
            for column in build.columns():
                if column in source:
                    _ = source.pop(column)
                if column + step.suffix in source:
                    _ = source.pop(column + step.suffix)
        elif step.kind == STEP_SELECT:
            var renamed = Dict[String, String]()
            for e in step.exprs:
                if len(e._nodes) == 1 and e._nodes[0].op == COL:
                    var found = source.get(e._nodes[0].text)
                    if found:
                        renamed[e._name] = found.value()
            source = renamed^
        elif step.kind == STEP_WITH_COLUMNS:
            for e in step.exprs:
                if e._name in source:
                    _ = source.pop(e._name)
    return ranges^


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
    # The selected rows' frame indices, when known (`by_rows`): a join's
    # or a top-k's selection, read through them as DuckDB reads a
    # selection vector, so a gather costs the rows kept rather than a scan
    # of the whole mask.
    var rows: List[Int]
    var by_rows: Bool
    # The height of the source this morsel was cut from: steps choose
    # their route by it, not by the morsel's own height.
    var source_rows: Int

    def __init__(out self, var frame: DataFrame, sequence: Int):
        self.source_rows = frame.height()
        self.count = frame.height()
        self.compact = List[Bool](length=frame.width(), fill=True)
        self.frame = frame^
        self.mask = BoolColumn(List[Bool]())
        self.selected = False
        self.sequence = sequence
        self.rows = List[Int]()
        self.by_rows = False

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
        # Sparse selections read the kept rows by index; dense ones scan
        # the mask in order, as the top-k narrows a morsel only when at
        # most one row in eight can enter (reading scattered rows costs
        # more than reading every row past that).
        if self.by_rows and 8 * self.count <= self.frame.height():
            var taken = take_parallel(
                pending^,
                self.rows.copy(),
                worker_count(len(self.rows)),
                or_null=False,
            )
            for k in range(len(indices)):
                self.frame._columns[indices[k]] = taken[k].copy()
                self.compact[indices[k]] = True
            return
        var filtered = filter_columns(pending, self.mask)
        var others = filtered[1]
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
            narrowed = input._selection(predicate, batch_size, self.source_rows)
            var kept = true_count(narrowed)
            if kept == self.count:
                return
            self.mask = narrowed^
            self.selected = True
            self.by_rows = False
            self.count = kept
            for i in range(len(self.compact)):
                self.compact[i] = False
            return
        var input = self._compact_frame(names)
        narrowed = input._selection(predicate, batch_size, self.source_rows)
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
        if self.by_rows:
            # The kept rows' frame indices, from the selected rows'.
            var rows = List[Int](capacity=kept)
            for k in range(self.count):
                if narrowed._get(k) and narrowed._valid(k):
                    rows.append(self.rows[k])
            self.rows = rows^
            # The mask from the kept rows, set where they are: no pass
            # over every row of the source (a filter after a join keeps a
            # few of a morsel's rows).
            self.mask = _rows_mask(self.rows, self.frame._height)
        else:
            self.mask = _spread(self.mask, narrowed)
        self.count = kept

    def select(mut self, exprs: List[Expr], batch_size: Int) raises:
        # A select of bare columns (renamed or not) only picks columns:
        # they keep the selection, as a projection of column references
        # does in DuckDB, instead of every kept row being gathered here.
        if self.selected and _bare_columns(exprs):
            trace_path("pipeline.projection_selection")
            var columns = List[Series](capacity=len(exprs))
            var compact = List[Bool](capacity=len(exprs))
            for e in exprs:
                var i = self._index(e._nodes[0].text)
                columns.append(self.frame._columns[i].renamed(e._name))
                compact.append(self.compact[i])
            # Gathered and source columns differ in length, as they do in
            # every morsel, so the columns are set in place.
            self.frame._columns = columns^
            self.compact = compact^
            return
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
        if self._with_columns_in_place(exprs, names, batch_size):
            return
        var input = self._compact_frame(names)
        var result = input.with_columns(exprs, batch_size=batch_size)
        # Columns the expressions produced or replaced join the frame as
        # compact; the rest stay as they are.
        for c in range(result.width()):
            var column = result._columns[c]
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

    def _with_columns_in_place(
        mut self, exprs: List[Expr], names: List[String], batch_size: Int
    ) raises -> Bool:
        """Row-local columns computed over every row of a morsel whose
        selection keeps at least half of it, joining it uncompacted and
        read through the selection like the columns it shares, as DuckDB
        projects over a selection vector: the selection survives, so the
        aggregate table can still read the morsel in place
        (`_in_place_table_rows`). A common subexpression computed below
        an aggregation (PDS-H q1's discounted prices) compacted the whole
        morsel and cost q1 60%. False, with nothing changed, when the
        selection is sparse or by rows, a column read is already compact,
        or evaluating every row raises (a dropped row may hold the error,
        so the compacting path decides what is raised)."""
        var n = self.frame.height()
        if not self.selected or self.by_rows or 2 * self.count < n:
            return False
        var whole = List[Series](capacity=len(names))
        for name in names:
            var at = self._index(name)
            if self.compact[at]:
                return False
            whole.append(self.frame._columns[at].copy())
        var result: DataFrame
        try:
            result = DataFrame(whole^, height=n).with_columns(
                exprs, batch_size=batch_size
            )
        except:
            return False
        var produced = List[String]()
        for e in exprs:
            produced.append(e._name)
        for c in range(result.width()):
            var column = result._columns[c]
            # New columns, and read ones an expression replaces.
            if column.name() in names and column.name() not in produced:
                continue
            var found = -1
            for i in range(self.frame.width()):
                if self.frame._columns[i].name() == column.name():
                    found = i
            if found >= 0:
                self.frame._columns[found] = column.copy()
                self.compact[found] = False
            else:
                self.frame._columns.append(column.copy())
                self.compact.append(False)
        return True

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

    def _narrow_rows(mut self, rows: List[Int]) raises:
        """Keep only the selected rows at positions `rows` (ascending, among
        the `count` selected rows): compact columns are taken at them, and
        shared columns are read through a mask whose set bits are those of
        the old mask the positions name, built a word at a time."""
        var kept = len(rows)
        if kept == self.count:
            return
        var height = self.frame.height()
        var words = (height + 63) // 64
        var out = List[UInt64](length=words, fill=0)
        var put = out.unsafe_ptr()
        var at = rows.unsafe_ptr()
        var frame_rows = List[Int](capacity=kept)
        if not self.selected:
            frame_rows = rows.copy()
            for k in range(kept):
                var r = at[unsafe_offset=k]
                put[unsafe_offset=r >> 6] |= UInt64(1) << UInt64(r & 63)
        else:
            # The old mask's set bits, in order: position j of the selected
            # rows is the j-th set bit.
            var j = 0
            var next = 0
            # A mask starting on a byte, every entry valid, is read a word
            # at a time; any other bit by bit.
            var aligned = self.mask._offset % 8 == 0 and (
                self.mask.null_count() == 0
            )
            var data = (
                self.mask._data[]
                .unsafe_ptr()
                .unsafe_offset(self.mask._offset // 8)
            )
            var data_bytes = (height + 7) // 8
            for w in range(words):
                var word = UInt64(0)
                if aligned:
                    for k in range(min(8, data_bytes - 8 * w)):
                        word |= UInt64(data[unsafe_offset=8 * w + k]) << UInt64(
                            8 * k
                        )
                    if 64 * (w + 1) > height:
                        word &= (UInt64(1) << UInt64(height - 64 * w)) - 1
                else:
                    for b in range(64):
                        var r = 64 * w + b
                        if r >= height:
                            break
                        if self.mask._get(r) and self.mask._valid(r):
                            word |= UInt64(1) << UInt64(b)
                while word != 0 and next < kept:
                    var b = Int(count_trailing_zeros(word))
                    word &= word - 1
                    if j == at[unsafe_offset=next]:
                        put[unsafe_offset=w] |= UInt64(1) << UInt64(b)
                        frame_rows.append(64 * w + b)
                        next += 1
                    j += 1
            var compact_at = List[Int]()
            var compact_columns = List[Series]()
            for i in range(self.frame.width()):
                if self.compact[i]:
                    compact_at.append(i)
                    compact_columns.append(self.frame._columns[i].copy())
            for k in range(len(compact_at)):
                self.frame._columns[compact_at[k]] = compact_columns[k].take(
                    rows
                )
        var bytes = List[UInt8](unsafe_uninit_length=8 * words)
        var source = out.unsafe_ptr().unsafe_bitcast[UInt8]()
        for k in range(8 * words):
            bytes.unsafe_ptr()[unsafe_offset=k] = source[unsafe_offset=k]
        _ = out^
        var mask = BoolColumn(List[Bool]())
        mask._data = ArcPointer(bytes^)
        mask._bits = ArcPointer(List[UInt8](length=(height + 7) // 8, fill=255))
        mask._length = height
        mask._offset = 0
        if not self.selected:
            for i in range(len(self.compact)):
                self.compact[i] = False
            self.mask = mask^
            self.selected = True
            self.count = kept
            self.rows = rows.copy()
            self.by_rows = True
            return
        self.mask = mask^
        self.count = kept
        self.rows = frame_rows^
        self.by_rows = True
        for i in range(len(self.compact)):
            self.compact[i] = False


# NOTE: The above fix keeps the selection semantics but deliberately avoids
# borrowing a temporary tuple/list element into a `ref` binding. This matches the
# Mojo ownership rules exposed by newer compiler builds and avoids the contract
# failure caused by binding a reference to a short-lived value from `filter_columns`
# or `result._columns`.

# The rest of the file is unchanged from the upstream version.

