"""Thread-owned pipelines over morsels (docs/design/executor-design.md, #538).

A pipeline is a source, a list of row-local steps, and a sink. Each worker
thread takes morsels from the source cursor, runs the full sequence of
steps, and updates its own sink state. Sink states are combined once the
source is exhausted; the main thread does no per-morsel work.

A morsel is a frame and a selection. A filter narrows the selection
without copying data. When a step reads a column, that column is gathered
through the selection once and replaces its shared version.
"""
from dataframe.storage.buffer import _Buffer, _BufferData
from std.atomic import Atomic
from std.bit import count_trailing_zeros
from std.os import getenv
from std.ffi import _get_global, external_call
from std.memory import ArcPointer, Pointer, bitcast, unsafe_memcpy
from std.sys import size_of
from std.time import perf_counter_ns

from dataframe.kernels.canonical_labels import canonical_labels
from dataframe.storage.boolean import BoolColumn, both_true, true_count
from .expr import COL, SELECTOR, Expr
from dataframe.aggregation.reducer import Reducer
from .binding import BoundExpr, bind
from dataframe.storage.column import Column
from .dtype import DataType, NUMERIC_DTYPES
from dataframe.execution import _ReduceJob, _new_reducer
from dataframe.kernels.validity import _valid_lanes
from .expr import (
    CORR,
    COUNT,
    COV,
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
from dataframe.aggregation.hash_table import (
    AggregateLayout,
    AggregateSink,
    aggregate_finish,
    aggregate_layout,
    layout_widths,
    reaggregable,
)
from .frame import (
    DataFrame,
    _rows_mask,
    concat,
)
from dataframe.aggregation.state import (
    _StreamReduction,
    table_frame,
    table_inputs,
    table_specs,
    table_supported,
    _equality_words,
    _finish_parts,
    _word_hash,
)
from dataframe.aggregation.keys import (
    _expand_struct_keys,
    _group_keys,
)
from dataframe.aggregation.selected_rows import _reduce_floats, _reduce_ints
from dataframe.kernels.join_hash import (
    PreparedHashIndex,
    prepared_direct_rows_at,
    prepared_hash_join_rows,
    prepared_hash_semi_anti_rows,
)
from .join_type import JOIN_ANTI, JOIN_FULL, JOIN_INNER, JOIN_RIGHT, JOIN_SEMI
from dataframe.kernels.row_encode import encodable
from dataframe.kernels.runtime_filter import (
    KeyBloom,
    LOPSIDED_FACTOR,
    MAX_BUILD_PROBE_DISTINCT_RATIO,
    bloom_mask,
    build_bloom,
)
from dataframe.kernels.gather import (
    can_filter_aligned_chunks,
    take_parallel,
    true_rows,
)
from dataframe.storage.allocation import huge_list
from dataframe.kernels.mask_filter import filter_columns
from .parallel import Job, run_jobs, worker_count
from .series import Series
from dataframe.kernels.hashing import encode_rows
from dataframe.storage.string import StringColumn
from .trace import trace_path
from .selected_expressions import (
    SelectedPlan,
    evaluate_masked,
    evaluate_selected,
    selected_plan,
)

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
# The constant key a grouping sets' grand total groups by (the aggregate
# table needs a key column); dropped from that set's result.
comptime _TOTAL_KEY = "__grouping_total"
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


def _row_counted(
    morsel: Morsel, expressions: List[Expr]
) raises -> Tuple[List[Expr], List[String], List[String]]:
    """`expressions` with each count of a column that holds no null in the
    morsel's source turned into a row count, the columns they still read
    values of, and the columns read only by row counts. DuckDB's rule (its
    count statistics callback): a count of a column without nulls is
    count(*), so over a selection it is the kept-row count, and the column
    is never gathered through the selection."""
    var rewritten = List[Expr](capacity=len(expressions))
    var valued = List[String]()
    var counted = List[String]()
    for e in expressions:
        var nodes = e._nodes.copy()
        var under_len = List[Bool](length=len(nodes), fill=False)
        for k in range(len(nodes)):
            var child = nodes[k].left
            if child < 0 or nodes[child].op != COL:
                continue
            if nodes[k].op == COUNT:
                var at = morsel._index(nodes[child].text)
                if morsel.frame._columns[at].null_count() == 0:
                    nodes[k].op = LEN
            if nodes[k].op == LEN:
                under_len[child] = True
        for k in range(len(nodes)):
            if nodes[k].op != COL:
                continue
            ref name = nodes[k].text
            if under_len[k]:
                if name not in counted:
                    counted.append(name)
            elif name not in valued:
                valued.append(name)
        rewritten.append(Expr(nodes^, e._name))
    var only_counted = List[String]()
    for name in counted:
        if name not in valued:
            only_counted.append(name)
    return (rewritten^, valued^, only_counted^)


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
    its integer key's buffers and row-group statistics: `_ZONE_NONE` when
    every value is worse
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
    """A finished join build's key range on a source column: a row
    whose key is null or outside it joins nothing. `any` is false when the
    build holds no non-null key, so no row joins."""

    var column: String
    var low: Int64
    var high: Int64
    var any: Bool
    # A bloom filter over the build's keys, when the build is much smaller
    # than the rows it filters and its keys a small share of the column's
    # (`runtime_filter.mojo`): rows whose key it lacks join nothing.
    var bloom: Optional[ArcPointer[KeyBloom]]
    # The join's index among the pipeline's steps.
    var step: Int


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
# Rows a worker checks against a bloom filter before it may leave it alone,
# and the share kept from which it does: DuckDB pauses a bloom filter that
# keeps a quarter or more of its first 75 vectors
# (`SelectivityOptionalFilter::BF_CHECK_N`, `BF_THRESHOLD`).
comptime _BLOOM_TRIAL_ROWS = 75 * 2048
comptime _BLOOM_PAUSE_SHARE = 0.25


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
    if column._facts[].zones_ready():
        return
    var first_row = column._facts[].first()
    var last_row = column._facts[].last(len(column._data[]))
    var blocks = (last_row + _ZONE_ROWS - 1) // _ZONE_ROWS
    var zones = List[Int64](capacity=2 * blocks)
    for b in range(blocks):
        var first = max(b * _ZONE_ROWS, first_row)
        var last = min((b + 1) * _ZONE_ROWS, last_row)
        var block = column.copy()
        block._offset = first
        block._length = max(0, last - first)
        var found = _column_bounds(block)
        if found[0]:
            zones.append(found[1])
            zones.append(found[2])
        else:
            zones.append(Int64.MAX)
            zones.append(Int64.MIN)
    column._facts[].publish_zones(zones^)


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
    if not column._facts[].zones_ready():
        return _column_bounds(column)
    ref zones = column._facts[].zones()
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


def _key_bloom(
    frame: DataFrame,
    name: String,
    key: Series,
    low: Int64,
    high: Int64,
    equalities: Int,
    step: Int,
) raises -> Optional[ArcPointer[KeyBloom]]:
    """A bloom filter over a build's keys when it pays, by Polars' rules
    (`join_runtime_filter.rs`): the build holds at most an eighth of the
    rows it filters, and, where the column's distinct count is known (a
    loaded table's statistics), at most 0.3 of it."""
    if equalities != 1:
        # As DuckDB's, only a join with one equality condition. The first
        # step takes one too: DuckDB's sits in the probe table's scan
        # whatever the join's place, and a lookup of one word costs a
        # fraction of a hash probe (PDS-H q21's semi joins of 6M lineitem
        # rows against 72K order keys).
        return None
    var keys = len(key) - key.null_count()
    if keys == 0 or LOPSIDED_FACTOR * keys > frame.height():
        return None
    var distinct = frame._distinct_count(name)
    if distinct > 0 and Float64(
        keys
    ) > MAX_BUILD_PROBE_DISTINCT_RATIO * Float64(distinct):
        return None
    # Keys that fill most of their range are told by the range already.
    if Float64(keys) * 2.0 >= Float64(high) - Float64(low) + 1.0:
        return None
    return build_bloom(key, keys)


def _key_ranges(
    frame: DataFrame, steps: List[Step], joins: List[DataFrame]
) raises -> List[_KeyRange]:
    """The key ranges of the inner and semi joins among `steps` whose
    probe key is a source column no earlier step replaces: every build is
    finished before the first morsel, so its range can narrow the morsel
    before any step runs."""
    var ranges = List[_KeyRange]()
    # Names the steps so far show for untouched source columns.
    var source = Dict[String, String]()
    for name in frame.columns():
        source[name] = name
    for at in range(len(steps)):
        ref step = steps[at]
        if step.kind == STEP_JOIN:
            ref build = joins[step.join]
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
                                name,
                                bounds.value()[0],
                                bounds.value()[1],
                                True,
                                _key_bloom(
                                    frame,
                                    name,
                                    key,
                                    bounds.value()[0],
                                    bounds.value()[1],
                                    len(step.names),
                                    at,
                                ),
                                at,
                            )
                        )
                    else:
                        ranges.append(_KeyRange(name, 0, 0, False, None, at))
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
    # A right or full join step's per-worker flags of the build rows matched so
    # far (the address of the bytes), set before the step runs; 0 for none.
    var marks: Int

    def __init__(out self, var frame: DataFrame, sequence: Int):
        self.marks = 0
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
                pending,
                self.rows.copy(),
                worker_count(len(self.rows)),
                or_null=False,
            )
            for k in range(len(indices)):
                self.frame._columns[indices[k]] = taken[k].copy()
                self.compact[indices[k]] = True
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
        if self._select_selected(exprs):
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

    def _evaluate_selected(
        self,
        plan: SelectedPlan,
        scatter: Bool,
        name: String,
        mut masked_rows: Optional[List[Int]],
    ) -> Optional[Series]:
        """`evaluate_selected` over this morsel's selection: its rows, or
        the rows its mask keeps, listed once into `masked_rows`. A mask
        whose values are placed at their rows (`scatter`) is computed at
        every row instead (`evaluate_masked`), as `_with_columns_in_place`
        computes it, and its rows are listed only when that returns
        None."""
        var n = self.frame.height()
        if self.by_rows:
            return evaluate_selected(
                plan,
                self.frame._columns,
                self.compact,
                self.rows,
                n,
                scatter,
                name,
            )
        if scatter:
            var whole = evaluate_masked(
                plan, self.frame._columns, self.compact, self.mask, n, name
            )
            if whole:
                return whole^
        if not masked_rows:
            masked_rows = _mask_rows(self.mask, self.count)
        return evaluate_selected(
            plan,
            self.frame._columns,
            self.compact,
            masked_rows.value(),
            n,
            scatter,
            name,
        )

    def _select_selected(mut self, exprs: List[Expr]) raises -> Bool:
        """A projection of columns and of arithmetic and casts over them
        that keeps the selection: a column is picked as it is, as a bare
        projection's is, and each computed one is computed for the
        selected rows alone (`_with_columns_selected`), so nothing is
        gathered here. False, with nothing changed, otherwise."""
        if not self.selected:
            return False
        var plans = List[Optional[SelectedPlan]](capacity=len(exprs))
        var named = Dict[String, Bool]()
        for e in exprs:
            # Repeated names raise on the ordinary path.
            if e._name in named:
                return False
            named[e._name] = True
            if len(e._nodes) == 1 and e._nodes[0].op == COL:
                plans.append(None)
                continue
            var plan = selected_plan(e, self.frame._columns)
            if not plan:
                return False
            plans.append(plan^)
        var n = self.frame.height()
        var scatter = 2 * self.count >= n
        var masked_rows = Optional[List[Int]]()
        var columns = List[Series](capacity=len(exprs))
        var compact = List[Bool](capacity=len(exprs))
        for k in range(len(exprs)):
            if not plans[k]:
                var i = self._index(exprs[k]._nodes[0].text)
                columns.append(self.frame._columns[i].renamed(exprs[k]._name))
                compact.append(self.compact[i])
                continue
            var result = self._evaluate_selected(
                plans[k].value(), scatter, exprs[k]._name, masked_rows
            )
            if not result:
                return False
            columns.append(result.take())
            compact.append(not scatter)
        trace_path("pipeline.select_selected")
        self.frame._columns = columns^
        self.frame._statistics = None
        self.compact = compact^
        return True

    def with_columns(mut self, exprs: List[Expr], batch_size: Int) raises:
        var reads = _reads(exprs)
        var names: List[String]
        if reads:
            names = reads.take()
        else:
            names = self.frame.columns()
        if self._with_columns_selected(exprs):
            return
        if self._with_columns_in_place(exprs, names, batch_size):
            return
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

    def _with_columns_selected(mut self, exprs: List[Expr]) raises -> Bool:
        """Arithmetic and casts computed for the selected rows alone,
        each column read through the selection where it is, as DuckDB's
        expression executor reads a join's dictionary vectors through
        their selection (`selected_expressions`). No input column is
        gathered and no dropped row is computed: TPC-DS q23's 128-bit
        decimal product over the 56% of store sales a date join keeps
        cost more over every row than gathering, and gathering cost more
        than this. A selection that keeps at least half the morsel gets
        each value at its row, so the morsel stays readable in place
        (`_in_place_table_rows`), as `_with_columns_in_place` leaves it;
        a sparser one gets one value per kept row. False, with nothing
        changed, when an expression uses anything else or a kept row
        raises (the ordinary path then raises it)."""
        if not self.selected:
            return False
        var plans = List[SelectedPlan](capacity=len(exprs))
        for e in exprs:
            var plan = selected_plan(e, self.frame._columns)
            if not plan:
                return False
            plans.append(plan.take())
        var n = self.frame.height()
        var scatter = 2 * self.count >= n
        var results = List[Series](capacity=len(exprs))
        var masked_rows = Optional[List[Int]]()
        for k in range(len(plans)):
            var result = self._evaluate_selected(
                plans[k], scatter, exprs[k]._name, masked_rows
            )
            if not result:
                return False
            results.append(result.take())
        trace_path("pipeline.with_columns_selected")
        for column in results:
            var found = -1
            for i in range(self.frame.width()):
                if self.frame._columns[i].name() == column.name():
                    found = i
            if found >= 0:
                self.frame._columns[found] = column.copy()
                self.compact[found] = not scatter
            else:
                self.frame._columns.append(column.copy())
                self.compact.append(not scatter)
        return True

    def _with_columns_in_place(
        mut self, exprs: List[Expr], names: List[String], batch_size: Int
    ) raises -> Bool:
        """Row-local columns computed over every row of a morsel whose
        selection keeps at least half of it, joining it uncompacted and
        read through the selection: the selection survives, so the
        aggregate table can still read the morsel in place
        (`_in_place_table_rows`). A common subexpression computed below
        an aggregation (PDS-H q1's discounted prices) compacted the whole
        morsel and cost q1 60%. False, with nothing changed, when the
        selection is sparse, a column read is already compact, or
        evaluating every row raises (a dropped row may hold the error,
        so the compacting path decides what is raised). A join's selection
        (by rows) qualifies as a filter's does: its mask is kept beside the
        rows, as DuckDB's projection reads a join's dictionary vectors in
        place (TPC-DS q23's quantity times price over 96% of store sales
        gathered both columns first)."""
        var n = self.frame.height()
        if not self.selected or 2 * self.count < n:
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
            ref column = result._columns[c]
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
        # Source columns retain source coordinates; compact columns hold
        # only selected rows. As in select(), preserve those different
        # lengths until gather_all() materializes the selection.
        self.frame._columns = kept^
        self.frame._statistics = None
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
        elif self.by_rows:
            # The selected rows' frame indices are known: the kept ones are
            # read at the positions, and the mask set from them, without a
            # pass over the old mask.
            var old_rows = self.rows.unsafe_ptr()
            frame_rows = List[Int](unsafe_uninit_length=kept)
            var to = frame_rows.unsafe_ptr()
            for k in range(kept):
                var r = old_rows[unsafe_offset=at[unsafe_offset=k]]
                to[unsafe_offset=k] = r
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
        if self.selected:
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
        unsafe_memcpy(
            dest=bytes.unsafe_ptr(),
            src=out.unsafe_ptr().unsafe_bitcast[UInt8](),
            count=8 * words,
        )
        _ = out^
        var mask = BoolColumn(List[Bool]())
        mask._data = _Buffer(bytes^)
        # Every entry valid: no validity bitmap, as `_rows_mask` builds.
        mask._bits = _Buffer(List[UInt8]())
        mask._length = height
        mask._offset = 0
        if not self.selected:
            for i in range(len(self.compact)):
                self.compact[i] = False
        self.mask = mask^
        self.selected = True
        self.rows = frame_rows^
        self.by_rows = True
        self.count = kept

    def _join_selected(
        mut self,
        step: Step,
        joins: ArcPointer[List[DataFrame]],
        indexes: ArcPointer[List[Optional[PreparedHashIndex]]],
    ) raises -> Bool:
        """An inner, semi or anti join through its prepared index, kept as a
        selection when each probe row matches at most once: the probe
        reads the key columns only, the matched rows narrow the morsel's
        selection, and only the build side's output columns are gathered,
        for the matched rows. The probe side's columns stay where they are
        until an operator reads them, so a column no later step reads is
        never copied. A probe row that matches more than once makes the
        output repeat rows: every column is then gathered at them. False,
        with nothing changed, when the keys need the general join
        (categorical, nested or differing types)."""
        if not (
            step.how == JOIN_INNER
            or step.how == JOIN_SEMI
            or step.how == JOIN_ANTI
        ):
            return False
        if not indexes[][step.join]:
            return False
        ref right = joins[][step.join]
        for i in range(len(step.names)):
            var lt = self.frame._columns[self._index(step.names[i])].dtype()
            var rt = right._columns[right._index(step.right_keys[i])].dtype()
            if lt != rt or lt.is_categorical() or lt.is_nested():
                return False
        # A selection's key rows probe a direct index where they are, as
        # DuckDB probes a join's dictionary vector through its selection:
        # the key is not gathered first (TPC-DS q23's customer join after
        # its date join copied the customer keys of every kept row).
        var through = Optional[Tuple[List[Int], List[Int], Bool]]()
        if self.selected and self.by_rows and len(step.names) == 1:
            var at = self._index(step.names[0])
            if not self.compact[at]:
                through = prepared_direct_rows_at(
                    self.frame._columns[at],
                    self.rows,
                    indexes[][step.join].value(),
                    step.how != JOIN_ANTI,
                )
        if step.how != JOIN_INNER:
            if through:
                self._narrow_rows(through.value()[0])
                return True
            var keys = self._compact_frame(step.names)
            self._narrow_rows(
                prepared_hash_semi_anti_rows(
                    keys._columns,
                    indexes[][step.join].value(),
                    step.how == JOIN_SEMI,
                    0,
                )
            )
            return True
        var pairs: Tuple[List[Int], List[Int], Bool]
        if through:
            pairs = through.take()
        else:
            var keys = self._compact_frame(step.names)
            pairs = prepared_hash_join_rows(
                keys._columns,
                indexes[][step.join].value(),
                False,
                omit_identity=True,
                workers=0,
            )
        # A probe row matching twice (a build key repeated) repeats rows:
        # no selection can hold them, so every column is gathered at them
        # from the pairs already probed.
        var repeated = False
        # A progression or perfect hash holds each build key once: no probe
        # row can match twice.
        if not pairs[2] and not indexes[][step.join].value().progression:
            var at = pairs[0].unsafe_ptr()
            for k in range(1, len(pairs[0])):
                if at[unsafe_offset=k] <= at[unsafe_offset=k - 1]:
                    repeated = True
                    break
        # The build side's output columns, named as the join names them.
        var names = Dict[String, Bool]()
        for column in self.frame._columns:
            names[column.name()] = True
        var left_names = names.copy()
        var right_keys = List[Int]()
        for key in step.right_keys:
            right_keys.append(right._index(key))
        var right_output = List[Int]()
        var right_names = List[String]()
        for i in range(right.width()):
            if step.coalesce and i in right_keys:
                continue
            var name = right._columns[i].name()
            if name in left_names:
                name += step.suffix
            if name in names:
                raise Error("Join output name collision: " + name)
            names[name] = True
            right_output.append(i)
            right_names.append(name)
        if repeated:
            var input = self.materialized()
            var columns = take_parallel(
                input._columns.copy(),
                pairs[0].copy(),
                worker_count(len(pairs[0])),
                or_null=False,
            )
            self.frame = DataFrame(columns^, height=len(pairs[0]))
            self.count = len(pairs[0])
            self.compact = List[Bool](length=self.frame.width(), fill=True)
            self.mask = BoolColumn(List[Bool]())
            self.selected = False
            self.by_rows = False
        else:
            trace_path("pipeline.join_selection")
            if not pairs[2]:
                self._narrow_rows(pairs[0])
        if len(right_output) == 0:
            # The build side adds no column (a join kept for its keys only):
            # its matched rows are never read.
            return True
        # String columns with view descriptors are not gathered: each
        # reads the build column's descriptors through the matched rows,
        # as DuckDB's probe emits a build column as a dictionary vector
        # over it. A consumer that reads only the dictionary codes beside
        # them (a group key, a comparison) never copies a descriptor; one
        # that reads bytes copies them out when it does. The rest are
        # gathered.
        var rows = Optional[ArcPointer[List[Int]]]()
        var selected = List[Optional[Series]](capacity=len(right_output))
        var sources = List[Series]()
        for k in range(len(right_output)):
            ref column = right._columns[right_output[k]]
            var picked = Optional[Series]()
            if column.dtype() == DataType.STRING:
                var whole = column.rechunk()
                if not rows:
                    rows = ArcPointer(pairs[1].copy())
                var through = whole._data[StringColumn]._select_rows(
                    rows.value()
                )
                if through:
                    var series = Series(right_names[k], through.take())
                    series._dtype = whole._dtype
                    series._field_metadata = whole._field_metadata
                    picked = series^
            if picked:
                trace_path("pipeline.join_view_selection")
            else:
                sources.append(column.copy())
            selected.append(picked^)
        var gathered = List[Series]()
        if len(sources) > 0:
            gathered = take_parallel(
                sources,
                pairs[1].copy(),
                worker_count(len(pairs[1])),
                or_null=False,
            )
        var next = 0
        for k in range(len(right_output)):
            if selected[k]:
                self.frame._columns.append(selected[k].value().copy())
            else:
                self.frame._columns.append(
                    gathered[next].renamed(right_names[k])
                )
                next += 1
            self.compact.append(True)
        return True

    def join(
        mut self,
        step: Step,
        joins: ArcPointer[List[DataFrame]],
        indexes: ArcPointer[List[Optional[PreparedHashIndex]]],
    ) raises:
        """Probe the join's prepared index with this morsel's rows: the
        output is the matched rows of both sides; the probe runs on this
        worker alone."""
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
            marks=self.marks,
        )
        self.count = joined.height()
        self.compact = List[Bool](length=joined.width(), fill=True)
        self.frame = joined^
        self.mask = BoolColumn(List[Bool]())
        self.selected = False
        self.by_rows = False

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
            if not self._join_selected(step, joins, indexes):
                self.join(step, joins, indexes)
        else:
            self.drop(step.names)

    def keep(mut self, mask: BoolColumn, kept: Int) raises:
        """Narrow the selection to the source rows `mask` sets (over the
        whole frame), before any step has run."""
        if self.selected:
            self.mask = both_true(self.mask, mask)
            self.count = true_count(self.mask)
        else:
            self.mask = mask.copy()
            self.count = kept
        self.selected = True
        self.by_rows = False
        for i in range(len(self.compact)):
            self.compact[i] = False

    def narrow(mut self, rows: List[Int]) raises:
        """Select only `rows` (ascending) of a fresh morsel: a top-k's
        bound applied before any step reads a column."""
        self.mask = _rows_mask(rows, self.frame._height)
        self.selected = True
        self.rows = rows.copy()
        self.by_rows = True
        self.count = len(rows)
        for i in range(len(self.compact)):
            self.compact[i] = False

    def materialized(mut self) raises -> DataFrame:
        self.gather_all()
        return self.frame.copy()


def _spread(outer: BoolColumn, inner: BoolColumn) raises -> BoolColumn:
    """`outer` with its set bits replaced, in order, by `inner`'s bits:
    the selection of a filter applied to an already selected morsel. Only
    the set bits are visited, a byte of the mask at a time."""
    var height = len(outer)
    var values = List[UInt8](length=(height + 7) // 8, fill=0)
    var out = values.unsafe_ptr()
    var next = 0
    if outer._offset % 8 == 0 and outer.null_count() == 0:
        var data = outer._data[].unsafe_ptr().unsafe_offset(outer._offset // 8)
        for i in range((height + 7) // 8):
            var byte = Int(data[unsafe_offset=i])
            if 8 * (i + 1) > height:
                byte &= (1 << (height - 8 * i)) - 1
            while byte != 0:
                var bit = Int(count_trailing_zeros(byte))
                byte &= byte - 1
                if inner._get(next) and inner._valid(next):
                    out[unsafe_offset=i] |= UInt8(1) << UInt8(bit)
                next += 1
    else:
        for r in range(height):
            if outer._get(r) and outer._valid(r):
                if inner._get(next) and inner._valid(next):
                    out[unsafe_offset=r >> 3] |= UInt8(1) << UInt8(r & 7)
                next += 1
    return BoolColumn(values=values^, bits=List[UInt8](), length=height)


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
    worker narrows its next morsel by the tightest bound. Two atomics:
    whether a bound is
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
    row. `_KeyIndex` served the same purpose at 100 to 180 ns a probe.

    A String key's words are its length and either its bytes (up to 8,
    exact) or a hash of all of them; for a longer one, equal words are
    compared once more on the bytes, found by address in the part's stored
    key frames, so long strings are compared by reading their original
    bytes."""

    var slots: List[Int32]
    var hashes: List[UInt64]
    var words: List[List[Int]]
    var count: Int
    # Per String key, each group's byte address and length.
    var addresses: List[List[Int]]
    var lengths: List[List[Int]]

    def __init__(out self, width: Int, strings: Int):
        self.slots = huge_list(1024, Int32(-1))
        self.hashes = List[UInt64]()
        self.words = List[List[Int]](capacity=width)
        for _ in range(width):
            self.words.append(List[Int]())
        self.count = 0
        self.addresses = List[List[Int]](capacity=strings)
        self.lengths = List[List[Int]](capacity=strings)
        for _ in range(strings):
            self.addresses.append(List[Int]())
            self.lengths.append(List[Int]())

    def reserve(mut self, groups: Int):
        """Room for `groups` in all without growing: the slot table at
        twice that, rehashed once, and the per-group lists."""
        var capacity = len(self.slots)
        while capacity < 2 * groups:
            capacity *= 2
        if capacity > len(self.slots):
            self.slots = huge_list(capacity // 2, Int32(-1))
            self._grow()
        self.hashes.reserve(groups)
        for w in range(len(self.words)):
            self.words[w].reserve(groups)
        for k in range(len(self.addresses)):
            self.addresses[k].reserve(groups)
            self.lengths[k].reserve(groups)

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
        mut self,
        hash: UInt64,
        ptrs: List[Int],
        width: Int,
        row: Int,
        addresses: List[Int],
        lengths: List[Int],
    ) -> Int:
        """The id of the key at `row` of the word buffers `ptrs`, added as
        the next id if new. String keys add `addresses` and `lengths`: per
        String key, the address of a buffer of each row's byte address,
        and of one of its byte length (-1 null)."""
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
                for k in range(len(addresses)):
                    self.addresses[k].append(
                        Pointer[Int, MutAnyOrigin](
                            unsafe_from_address=addresses[k]
                        )[unsafe_offset=row]
                    )
                    self.lengths[k].append(
                        Pointer[Int, MutAnyOrigin](
                            unsafe_from_address=lengths[k]
                        )[unsafe_offset=row]
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
                if same and len(addresses) > 0:
                    same = self._same_bytes(id, row, addresses)
                if same:
                    return id
            slot = (slot + 1) & mask

    def _same_bytes(self, id: Int, row: Int, addresses: List[Int]) -> Bool:
        """Whether the strings longer than their word (the words, lengths
        included, are equal) have the same bytes."""
        for k in range(len(addresses)):
            var length = self.lengths[k][id]
            if length <= 8:
                continue
            var mine = Pointer[Int, MutAnyOrigin](
                unsafe_from_address=addresses[k]
            )[unsafe_offset=row]
            var compared = external_call["memcmp", Int32](
                self.addresses[k][id], mine, length
            )
            if compared != 0:
                return False
        return True

    def point(mut self, first: Int, keys: DataFrame, positions: List[Int]):
        """Point the groups from id `first` on, one per row of `keys` (the
        part's stored frame they were taken into), at their bytes there:
        the morsel or table they were found in goes away."""
        for k in range(len(positions)):
            _string_addresses(
                keys._columns[positions[k]],
                Pointer[Int, MutAnyOrigin](
                    unsafe_from_address=Int(self.addresses[k].unsafe_ptr())
                    + 8 * first
                ),
            )


def _string_addresses(column: Series, target: Pointer[Int, MutAnyOrigin]):
    """Each row's byte address, the column's buffers found once (as
    `_string_words` does)."""
    ref strings = column._data[StringColumn]
    var n = len(strings)
    if strings._is_view_storage():
        ref views = strings._view_storage.value()
        for i in range(n):
            target[unsafe_offset=i] = Int(
                views._get_unchecked(strings._offset + i)
                .as_bytes()
                .unsafe_ptr()
            )
        return
    var offsets = strings._offsets[].unsafe_ptr().unsafe_offset(strings._offset)
    var base = strings._base()
    for i in range(n):
        target[unsafe_offset=i] = Int(
            base.unsafe_offset(Int(offsets[unsafe_offset=i]))
        )


def _string_words(
    column: Series,
    mut addresses: List[Int],
    mut lengths: List[Int],
    mut values: List[Int],
):
    """A String key's words for each row: its byte length (-1 null) and
    its bytes when they fit a word, else a hash of all of them; with each
    row's byte address for the index's comparison. The column's buffers
    are found once: a row read through `_row_bytes` copied the column's
    view storage, reference counts included, every row (PDS-H q1's two
    flags: 142 ms against 60)."""
    ref strings = column._data[StringColumn]
    var n = len(strings)
    var nulls = strings.null_count() > 0
    if strings._is_view_storage():
        ref views = strings._view_storage.value()
        for i in range(n):
            if nulls and not strings._valid(i):
                lengths[i] = -1
                values[i] = 0
                addresses[i] = 0
                continue
            var bytes = views._get_unchecked(strings._offset + i).as_bytes()
            _word_of(
                bytes.unsafe_ptr(), len(bytes), i, addresses, lengths, values
            )
        return
    var offsets = strings._offsets[].unsafe_ptr().unsafe_offset(strings._offset)
    var base = strings._base()
    for i in range(n):
        if nulls and not strings._valid(i):
            lengths[i] = -1
            values[i] = 0
            addresses[i] = 0
            continue
        var start = Int(offsets[unsafe_offset=i])
        var length = Int(offsets[unsafe_offset=i + 1]) - start
        _word_of(
            base.unsafe_offset(start), length, i, addresses, lengths, values
        )


@always_inline
def _word_of(
    start: Pointer[UInt8, ImmutAnyOrigin],
    length: Int,
    i: Int,
    mut addresses: List[Int],
    mut lengths: List[Int],
    mut values: List[Int],
):
    lengths.unsafe_ptr()[unsafe_offset=i] = length
    addresses.unsafe_ptr()[unsafe_offset=i] = Int(start)
    if length <= 8:
        var packed = UInt64(0)
        for b in range(length):
            packed |= UInt64(start[unsafe_offset=b]) << UInt64(8 * b)
        values.unsafe_ptr()[unsafe_offset=i] = Int(packed)
        return
    var h = UInt64(0x9E3779B97F4A7C15) ^ UInt64(length)
    var at = 0
    while at + 8 <= length:
        var word = (
            start.unsafe_offset(at)
            .unsafe_bitcast[UInt64]()
            .unsafe_load[width=1, alignment=1]()
        )
        h = (h ^ word) * 0xBF58476D1CE4E5B9
        h ^= h >> 29
        at += 8
    if at < length:
        # The last 8 bytes, overlapping what was read.
        var word = (
            start.unsafe_offset(length - 8)
            .unsafe_bitcast[UInt64]()
            .unsafe_load[width=1, alignment=1]()
        )
        h = (h ^ word) * 0x94D049BB133111EB
        h ^= h >> 31
    values.unsafe_ptr()[unsafe_offset=i] = Int(h)


def _fixed_keys(columns: List[Series], strings: List[Int]) -> List[Series]:
    """The key columns that are not Strings, in order."""
    var out = List[Series](capacity=len(columns))
    for k in range(len(columns)):
        if k not in strings:
            out.append(columns[k].copy())
    return out^


struct _GroupPart(Movable):
    """One hash part of a worker's inserted groups: their keys in an
    index, reducer states, first rows, and key values (the rows that
    introduced them, taken from each morsel)."""

    var index: _PartIndex
    var states: List[Reducer]
    var firsts: List[Int]
    var keys: List[DataFrame]
    # Positions of the String keys among the key frames' columns.
    var positions: List[Int]

    def __init__(out self, width: Int, positions: List[Int]):
        self.index = _PartIndex(width, len(positions))
        self.states = List[Reducer]()
        self.firsts = List[Int]()
        self.keys = List[DataFrame]()
        self.positions = positions.copy()

    def keep_keys(mut self, var keys: DataFrame, first: Int):
        """Store the key rows of the groups from id `first` on, pointing
        their String keys' bytes at them."""
        if len(self.positions) > 0:
            self.index.point(first, keys, self.positions)
        self.keys.append(keys^)

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
        var addresses = List[Int](capacity=len(self.positions))
        var lengths = List[Int](capacity=len(self.positions))
        for k in range(len(self.positions)):
            addresses.append(Int(other.index.addresses[k].unsafe_ptr()))
            lengths.append(Int(other.index.lengths[k].unsafe_ptr()))
        var mapping = List[Int](capacity=n)
        var fresh = List[Int]()
        var before = self.index.count
        for g in range(n):
            var id = self.index.find_or_insert(
                other.index.hashes[g], ptrs, width, g, addresses, lengths
            )
            mapping.append(id)
            if id >= before + len(fresh):
                fresh.append(g)
                self.firsts.append(other.firsts[g])
            elif other.firsts[g] < self.firsts[id]:
                self.firsts[id] = other.firsts[g]
        if len(fresh) > 0:
            # The new groups' key rows, taken frame by frame from the other
            # part's stored frames (no concatenation of them first: one
            # copy of every key's bytes fewer where most groups are new).
            # Fresh groups come in row order and take consecutive ids.
            var first = before
            var at = 0
            var start = 0
            for f in range(len(other.keys)):
                var height = other.keys[f].height()
                var rows = List[Int]()
                while at < len(fresh) and fresh[at] < start + height:
                    rows.append(fresh[at] - start)
                    at += 1
                if len(rows) > 0:
                    var taken = len(rows)
                    self.keep_keys(other.keys[f].take(rows), first)
                    first += taken
                start += height
        var count = self.index.count
        for e in range(len(self.states)):
            self.states[e].grow(count)
            self.states[e].merge(other.states[e], groups=mapping)
        # Its key frames hold the bytes read through addresses above.
        _ = other^

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
    morsel; each worker owns a radix-partitioned aggregate table.

    Many groups: keep morsel rows as they come, then finish with one
    scatter of every row into buckets and cache-local bucket encodes.
    This avoids inserting rows one at a time into tables that outgrow
    the caches (ClickBench q32, 10M groups: 850 ms inserted against
    300 scattered). When the steps only filter, what is kept is each
    morsel's selection over the
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
    # Per hash part, the tables flushed from it (`_FLUSH_GROUPS`).
    var flushed: List[List[_GroupPart]]
    var count: Int
    # Groups a part holds before it is flushed (`_flush_groups`).
    var flush_groups: Int
    var split_groups: Int
    var collecting: Bool
    var collected: List[DataFrame]
    var batch_size: Int
    # Selections over the source frame: a morsel's offset, length and
    # mask (an empty mask: every row of the morsel).
    var offsets: List[Int]
    var lengths: List[Int]
    var masks: List[BoolColumn]
    var width: Int
    # Positions of the String key columns among the expanded keys: their
    # bytes past the words' prefix are compared exactly (`_PartIndex`).
    var strings: List[Int]
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
        self.strings = List[Int]()
        for k in range(len(key_columns)):
            if key_columns[k].dtype().physical() == DataType.STRING:
                self.strings.append(k)
        # The other keys' words, then a length and a value word per String
        # key (`_string_words`).
        var fixed = _fixed_keys(key_columns, self.strings)
        self.width = (
            len(_equality_words(fixed)) if len(fixed) > 0 else 0
        ) + 2 * len(self.strings)
        self.part_of = List[Int]()
        self.order = List[Int]()
        self.ids = List[Int]()
        self.parts = List[_GroupPart](capacity=1 << _PART_BITS)
        self.flushed = List[List[_GroupPart]]()
        self.flush_groups = _flush_groups()
        self.split_groups = min(_SPLIT_GROUPS, self.flush_groups)
        if not collecting:
            self.parts.append(self._empty_part(empty))

    def _empty_part(self, sample: DataFrame) raises -> _GroupPart:
        """A part with one empty state per expression, so the workers'
        parts merge expression by expression."""
        var part = _GroupPart(self.width, self.strings)
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
        var addresses = List[Int](capacity=len(self.strings))
        var lengths = List[Int](capacity=len(self.strings))
        for k in range(len(self.strings)):
            addresses.append(Int(whole.index.addresses[k].unsafe_ptr()))
            lengths.append(Int(whole.index.lengths[k].unsafe_ptr()))
        for p in range(count):
            var part = self._empty_part(sample)
            if len(members[p]) > 0:
                var part_keys = keys.take(members[p])
                for g in members[p]:
                    _ = part.index.find_or_insert(
                        whole.index.hashes[g],
                        ptrs,
                        self.width,
                        g,
                        addresses,
                        lengths,
                    )
                    part.firsts.append(whole.firsts[g])
                part.keep_keys(part_keys^, 0)
                for s in range(len(part.states)):
                    part.states[s].grow(len(members[p]))
                    part.states[s].merge(whole.states[s], sources=members[p])
            self.parts.append(part^)
        # Its key frames hold the bytes read through addresses above.
        _ = whole^

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
        var key_columns = _group_keys(input._subset_keys(self.key_names))
        for k in range(len(key_columns)):
            ref column = key_columns[k]
            if column.dtype().is_nested():
                return False
            if k in self.strings:
                # Any length: the bytes past the prefix are compared.
                if column.is_chunked():
                    column = column.rechunk()
                if not column._data.isa[StringColumn]():
                    return False
            elif not encodable(column):
                return False
        var n = input.height()
        if n == 0:
            return True
        if len(self.parts) == 1 and self.count >= self.split_groups:
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
        #
        # String keys that all carry their source's dictionary codes (a
        # Parquet scan) find the morsel's distinct keys by those codes
        # first, one integer encode, and only those units get words and a
        # probe; each row takes its unit's group (PDS-H q1's two flags: 55
        # ms, against 103 with words for every row). Otherwise every row
        # is its own unit.
        var units = False
        var unit_of = List[Int]()
        var unit_columns = key_columns.copy()
        if len(self.strings) > 0:
            var local_columns = List[Series](capacity=len(key_columns))
            var coded = True
            for k in range(len(key_columns)):
                if k not in self.strings:
                    local_columns.append(key_columns[k].copy())
                    continue
                var codes = key_columns[k]._dictionary_codes()
                if not codes:
                    coded = False
                    break
                local_columns.append(codes.take())
            if coded:
                var local = encode_rows(local_columns, nulls_equal=True)
                for k in range(len(key_columns)):
                    unit_columns[k] = key_columns[k].take(local.representatives)
                unit_of = local^.into_ids()
                units = True
        var u = len(unit_columns[0])
        var fixed = _fixed_keys(unit_columns, self.strings)
        var words = (
            _equality_words(fixed) if len(fixed) > 0 else List[List[Int]]()
        )
        var addresses = List[Int](capacity=len(self.strings))
        var lengths = List[Int](capacity=len(self.strings))
        var address_rows = List[List[Int]](capacity=len(self.strings))
        for k in self.strings:
            var address = List[Int](unsafe_uninit_length=u)
            var length = List[Int](unsafe_uninit_length=u)
            var value = List[Int](unsafe_uninit_length=u)
            _string_words(unit_columns[k], address, length, value)
            words.append(length^)
            words.append(value^)
            address_rows.append(address^)
        for k in range(len(self.strings)):
            addresses.append(Int(address_rows[k].unsafe_ptr()))
            lengths.append(
                Int(
                    words[
                        len(words) - 2 * len(self.strings) + 2 * k
                    ].unsafe_ptr()
                )
            )
        var row_hashes = List[UInt64](capacity=u)
        for r in range(u):
            row_hashes.append(_word_hash(words, r))
        var hashes_ptr = row_hashes.unsafe_ptr()
        var unit_ptr = unit_of.unsafe_ptr()
        # A unit's group id once probed (-1 before).
        var unit_ids = List[Int](length=u if units else 0, fill=-1)
        var unit_ids_ptr = unit_ids.unsafe_ptr()
        var parts_ptr = part_of.unsafe_ptr()
        var starts_ptr = starts.unsafe_ptr()
        if part_count == 1:
            for r in range(n):
                parts_ptr[unsafe_offset=r] = 0
            starts[1] = n
        elif units:
            for r in range(n):
                var p = Int(
                    hashes_ptr[unsafe_offset=unit_ptr[unsafe_offset=r]] >> shift
                )
                parts_ptr[unsafe_offset=r] = p
                starts_ptr[unsafe_offset=p + 1] += 1
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
                var id: Int
                if units:
                    var unit = unit_ptr[unsafe_offset=r]
                    id = unit_ids_ptr[unsafe_offset=unit]
                    if id < 0:
                        id = part.index.find_or_insert(
                            hashes_ptr[unsafe_offset=unit],
                            ptrs,
                            self.width,
                            unit,
                            addresses,
                            lengths,
                        )
                        unit_ids_ptr[unsafe_offset=unit] = id
                else:
                    id = part.index.find_or_insert(
                        hashes_ptr[unsafe_offset=r],
                        ptrs,
                        self.width,
                        r,
                        addresses,
                        lengths,
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
                part.keep_keys(DataFrame(columns^, height=len(fresh)), before)
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
        _ = address_rows^
        _ = unit_of^
        _ = unit_ids^
        _ = unit_columns^
        if len(self.parts) == 1 and self.count >= self.split_groups:
            # Split as soon as the groups pass the split, so a part can
            # flush below.
            self._repartition(input)
        part_count = len(self.parts)
        if part_count > 1:
            for p in range(part_count):
                if self.parts[p].index.count <= self.flush_groups:
                    continue
                if len(self.flushed) == 0:
                    for _ in range(part_count):
                        self.flushed.append(List[_GroupPart]())
                trace_path("pipeline.group_flush")
                var full = self._empty_part(input)
                swap(full, self.parts[p])
                self.flushed[p].append(full^)
        return True

    def into_parts(
        mut self, sample: DataFrame, split: Bool
    ) raises -> List[List[_GroupPart]]:
        """Number of inserted parts in hash order: one when no worker's
        table exceeds the split threshold; otherwise 2^_PART_BITS, so
        corresponding partitions from every worker line up when combined."""
        if split and len(self.parts) == 1:
            self._repartition(sample)
        var parts = List[_GroupPart]()
        swap(parts, self.parts)
        var out = List[List[_GroupPart]](capacity=len(parts))
        var p = 0
        while len(parts) > 0:
            var tables = List[_GroupPart]()
            tables.append(parts.pop(0))
            if p < len(self.flushed):
                while len(self.flushed[p]) > 0:
                    tables.append(self.flushed[p].pop())
            out.append(tables^)
            p += 1
        return out^


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
            var total = 0
            for t in range(len(workers)):
                total += workers[t].index.count
            self.states.append(self._merge(workers^, total))

    def _merge(
        mut self, var tables: List[_GroupPart], total: Int
    ) raises -> _StreamReduction:
        var base = tables.pop(0)
        # Sized once for every table's groups: grown by doubling, the
        # index rehashed seven times on the way to ClickBench q12's 52K
        # groups a part.
        base.index.reserve(total)
        while len(tables) > 0:
            base.absorb(tables.pop(0), self.width)
        return base.into_state(
            self.names, self.dtypes, self.outputs, self.schema
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
    # Aggregate table sink (`aggregate_table.mojo`): its layout, shared by
    # every worker, and this worker's sink once it takes a morsel.
    var layout: List[AggregateLayout]
    var sinks: List[AggregateSink]
    # Grouping sets: each set's key columns (indices into `keys`; empty is
    # the grand total). Every set has its own layout and sink, and every
    # morsel is sunk into each of them (DuckDB's one sink feeding every
    # grouping set's table, `PhysicalHashAggregate::Sink`).
    var sets: List[List[Int]]
    # A set derived from another's group rows rather than sunk from the
    # morsels: its source set (-1 for none) and, per key it groups by, that
    # key's column among the source's.
    var derived_from: List[Int]
    var derived_columns: List[List[Int]]
    # The sets by size, largest first: a set is derived after its source.
    var derive_order: List[Int]
    var workers: Int
    var rows: Int
    # Rows fed, then each step's input and output rows, for the report.
    var counts: List[Int]
    # The join builds' key ranges on source columns (`_key_ranges`).
    var key_ranges: List[_KeyRange]
    # Per range with a bloom filter: this worker's rows checked and kept.
    var bloom_checked: List[Int]
    var bloom_kept: List[Int]
    # A right or full join step's flags of the build rows this worker's
    # morsels matched (one byte a build row), by step; empty for others.
    var marks: List[List[UInt8]]
    # The step the morsels enter at (0: the source's; past an outer join
    # step for the build rows it emits after its probe), and the number
    # of the frame's first morsel among all the morsels sunk.
    var first_step: Int
    var first_morsel: Int
    # Whether this run ends the worker's work (folds its top-k candidates
    # and derives its grouping sets): not before an outer join's
    # unmatched build rows have run.
    var finishes: Bool
    # While an outer join's unmatched build rows run (`_emit_unmatched`):
    # that join's step (-1 otherwise), the address of every worker's
    # marks of its build rows, and the probe side's columns at the step,
    # empty. The frame is then the build, and each worker finds its
    # morsel's unmatched rows itself.
    var unmatched_step: Int
    var unmatched_marks: ArcPointer[List[Int]]
    var unmatched_probe: ArcPointer[DataFrame]

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
        insert: Bool,
        batch_size: Int,
        top: Int,
        top_names: List[String],
        top_descending: List[Bool],
        top_nulls_last: List[Bool],
        bounded: Bool,
        shared_bound: _SharedBound,
        floating_key: Bool,
        layout: List[AggregateLayout],
        sets: List[List[Int]],
        derived_from: List[Int],
        derived_columns: List[List[Int]],
        derive_order: List[Int],
        workers: Int,
        key_ranges: List[_KeyRange],
    ):
        self.key_ranges = key_ranges.copy()
        self.bloom_checked = List[Int](length=len(key_ranges), fill=0)
        self.bloom_kept = List[Int](length=len(key_ranges), fill=0)
        self.marks = List[List[UInt8]](capacity=len(steps))
        for step in steps:
            if step.kind == STEP_JOIN and (
                step.how == JOIN_RIGHT or step.how == JOIN_FULL
            ):
                self.marks.append(
                    List[UInt8](length=joins[][step.join].height(), fill=0)
                )
            else:
                self.marks.append(List[UInt8]())
        self.first_step = 0
        self.first_morsel = 0
        self.finishes = True
        self.unmatched_step = -1
        self.unmatched_marks = ArcPointer(List[Int]())
        self.unmatched_probe = frame.copy()
        self.layout = layout.copy()
        self.sets = sets.copy()
        self.derived_from = derived_from.copy()
        self.derived_columns = derived_columns.copy()
        self.derive_order = derive_order.copy()
        self.sinks = List[AggregateSink]()
        self.workers = workers
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

    def _table_reads(self) raises -> List[String]:
        """The columns the keys and reductions read."""
        var names = self.keys.copy()
        var reads = _reads(self.expressions)
        if reads:
            for name in reads.take():
                if name not in names:
                    names.append(name)
        return names^

    def _derive_sets(mut self) raises:
        """The sets derived from another's group rows: each worker merges
        its own finer table's rows into a table of the coarser set, so a
        coarser grouping reads groups, not input rows."""
        if len(self.sets) == 0 or len(self.sinks) == 0:
            return
        while len(self.sinks) < len(self.sets):
            self.sinks.append(
                AggregateSink(
                    self.layout[len(self.sinks)],
                    self.workers,
                    len(self.sets),
                )
            )
        for t in self.derive_order:
            var from_set = self.derived_from[t]
            if from_set < 0:
                continue
            # The target comes out of the list while the source is read.
            var target = self.sinks.pop(t)
            target.sink_group_rows(
                self.sinks[from_set if from_set < t else from_set - 1],
                self.derived_columns[t],
                layout_widths(self.layout[from_set]),
            )
            self.sinks.insert(t, target^)

    def _sink_sets(mut self, mut morsel: Morsel, k: Int) raises:
        """A morsel into every grouping set's sink: the reduction inputs
        are evaluated once, and each set takes the key columns it groups
        by (the grand total a constant one)."""
        var source = morsel._compact_frame(self._table_reads()) if _reads(
            self.expressions
        ) else morsel.materialized()
        var all_keys = List[Series](capacity=len(self.keys))
        for name in self.keys:
            all_keys.append(canonical_labels(_unchunked(source.column(name))))
        var inputs = table_inputs(source, all_keys, self.expressions)
        for i in range(len(inputs)):
            inputs[i] = _unchunked(inputs[i])
        var rows = len(all_keys[0])
        while len(self.sinks) < len(self.sets):
            self.sinks.append(
                AggregateSink(
                    self.layout[len(self.sinks)],
                    self.workers,
                    len(self.sets),
                )
            )
        for s in range(len(self.sets)):
            if self.derived_from[s] >= 0:
                continue
            var keys = List[Series](capacity=len(self.sets[s]))
            if len(self.sets[s]) == 0:
                keys.append(
                    Series(
                        _TOTAL_KEY,
                        Column[Int64](List[Int64](length=rows, fill=0)),
                    )
                )
            else:
                for i in self.sets[s]:
                    keys.append(all_keys[i].copy())
            self.sinks[s].sink(keys, inputs, k << 32)
        self.rows += rows

    def _fold_candidates(mut self) raises:
        """Fold this worker's candidates into a size-k heap, and
        tighten the bound from the k-th, shared with every worker."""
        var merged = _top_rows(
            concat(self.parts),
            self.top_names,
            self.top_descending,
            self.top_nulls_last,
            self.top,
            1,
            False,
        )
        self.candidate_rows = merged.height()
        if self.bounded:
            self.bound = _TopBound(
                merged, self.top_names[0], self.top_descending[0], self.top
            )
            self.shared_bound.publish(self.bound)
        self.parts = [merged^]

    def _in_key_ranges(mut self, mut morsel: Morsel) raises -> Bool:
        """Narrow a fresh morsel to the rows inside every join build's key
        range; False when none is. A morsel whose keys all fall outside a
        range is skipped whole, and one whose keys all fall inside is left
        as it is, both told by its keys' least and greatest values (a
        zone map); only a morsel straddling the range is filtered row by
        row."""
        for k in range(len(self.key_ranges)):
            ref r = self.key_ranges[k]
            if not r.any:
                return False
            var column = morsel.frame.column(r.column)
            var bounds = _window_bounds(column)
            if not bounds:
                return False
            var least = bounds.value()[0]
            var most = bounds.value()[1]
            if most < r.low or least > r.high:
                return False
            if not (least >= r.low and most <= r.high):
                # The share of the morsel's key span the range covers: when
                # over half, the range likely keeps most rows, and the steps
                # would read them through a selection for little gain.
                var span = Float64(most) - Float64(least) + 1.0
                var covered = (
                    Float64(min(most, r.high))
                    - Float64(max(least, r.low))
                    + 1.0
                )
                if 2.0 * covered <= span:
                    var narrowed = _range_mask(
                        _unchunked(column), r.low, r.high
                    )
                    if narrowed[1] == 0:
                        return False
                    if 2 * narrowed[1] <= morsel.frame.height():
                        morsel.keep(narrowed[0], narrowed[1])
                        if morsel.count == 0:
                            return False
            if not r.bloom:
                continue
            var shared = r.bloom.value().copy()
            # Rows whose key the bloom filter lacks join nothing. It is
            # left alone once this worker's first rows show it keeps a
            # quarter or more of what it looks at.
            if self.bloom_checked[k] >= _BLOOM_TRIAL_ROWS and Float64(
                self.bloom_kept[k]
            ) >= _BLOOM_PAUSE_SHARE * Float64(self.bloom_checked[k]):
                continue
            var plain = (
                morsel.selected
                and morsel.mask._offset == 0
                and len(morsel.mask._bits[]) == 0
            )
            var selection = Optional[BoolColumn]()
            if plain:
                selection = Optional(morsel.mask.copy())
            # Only the rows still selected are looked up and counted: what
            # the filter prunes is what earlier filters left.
            var found = bloom_mask(_unchunked(column), shared[], selection)
            self.bloom_checked[k] += found[1]
            self.bloom_kept[k] += found[2]
            if found[2] == 0:
                return False
            var rows = morsel.count if plain else morsel.frame.height()
            if 2 * found[2] <= rows:
                morsel.keep(found[0], found[2])
            if morsel.count == 0:
                return False
        return True

    def run(mut self) raises:
        var count = len(self.ranges[]) // 2
        var cursor = _Cursor(self.cursor)
        while True:
            var taken = cursor.take(1)
            if taken >= count:
                if not self.finishes:
                    return
                # The worker's leftover candidates folded to k, so the
                # finish sorts at most k rows a worker.
                if self.top > 0 and self.candidate_rows > self.top:
                    self._fold_candidates()
                self._derive_sets()
                return
            var k = self.first_morsel + taken
            var offset = self.ranges[][2 * taken]
            var length = self.ranges[][2 * taken + 1]
            if self.unmatched_step >= 0:
                var rows = self._unmatched(offset, length)
                if rows.height() == 0:
                    continue
                var tail = Morsel(rows^, k)
                tail.source_rows = self.frame[].height()
                self._steps_from(tail, self.first_step)
                if tail.height() > 0:
                    self._consume(tail, k, offset, length)
                continue
            var morsel = Morsel(self.frame[].slice(offset, length), k)
            morsel.source_rows = self.frame[].height()
            if self.first_step > 0:
                # An outer join's unmatched build rows: the source's key
                # ranges and top-k bound do not apply to them.
                self._steps_from(morsel, self.first_step)
                if morsel.height() > 0:
                    self._consume(morsel, k, offset, length)
                continue
            self.counts[0] += length
            if not self._in_key_ranges(morsel):
                continue
            if self.top > 0 and self.bounded:
                self.shared_bound.tighten(
                    self.bound,
                    self.top_names[0],
                    self.top_descending[0],
                    self.floating_key,
                )
            var zoned = _ZONE_SOME
            if self.top > 0 and self.bound.kind != _BOUND_NONE:
                if self.bound.kind == _BOUND_INT:
                    zoned = _zoned_top(morsel.frame, self.bound)
                if zoned == _ZONE_NONE:
                    # No row of the morsel can enter: it is left empty
                    # and read no further.
                    trace_path("pipeline.top_zone_skip")
                    morsel.narrow(List[Int]())
            if (
                self.top > 0
                and self.bound.kind != _BOUND_NONE
                and zoned == _ZONE_SOME
            ):
                # Narrow to the rows that can still enter the top k when
                # at most one in eight can: gathering a step's columns at
                # scattered rows costs more than reading them whole past
                # that (ClickBench q26's string filter on 15% of the rows:
                # 31 ms busy against 6 reading every row).
                var kept = self.bound.rows(morsel.frame)
                if kept and 8 * len(kept.value()) <= morsel.height():
                    morsel.narrow(kept.value())
            self._steps_from(morsel, 0)
            if morsel.height() == 0:
                continue
            self._consume(morsel, k, offset, length)

    def _unmatched(self, offset: Int, length: Int) raises -> DataFrame:
        """The build rows in [offset, offset + length) that no worker's
        morsel matched, as the outer join step's output (the probe columns
        null): each worker scans its own share of the build rows, as each
        thread of DuckDB's `PhysicalHashJoin::GetData` scans its share of
        the hash table's rows for unmatched ones."""
        ref marks = self.unmatched_marks[]
        var rows = List[Int]()
        var stop = offset + length
        var r = offset
        while r < stop:
            # Eight build rows a word while every worker matched them all.
            if r + 8 <= stop:
                var all = UInt64(0x0101010101010101)
                var merged = UInt64(0)
                for m in marks:
                    merged |= Pointer[UInt64, MutAnyOrigin](
                        unsafe_from_address=m + r
                    )[]
                if merged == all:
                    r += 8
                    continue
            var seen = UInt8(0)
            for m in marks:
                seen |= Pointer[UInt8, MutAnyOrigin](unsafe_from_address=m)[
                    unsafe_offset=r
                ]
            if seen == 0:
                rows.append(r)
            r += 1
        if len(rows) == 0:
            return self.unmatched_probe[].clear()
        trace_path("pipeline.join_unmatched")
        ref step = self.steps[self.unmatched_step]
        return self.unmatched_probe[]._join_impl(
            self.frame[]._filter_rows(rows^),
            left_on=step.names,
            right_on=step.right_keys,
            how=step.how,
            suffix=step.suffix,
            coalesce=step.coalesce,
        )

    def _steps_from(mut self, mut morsel: Morsel, first: Int) raises:
        """Run the steps from `first` on over one morsel, until one leaves
        it empty."""
        for i in range(first, len(self.steps)):
            self.counts[2 * i + 1] += morsel.height()
            var began = Int(perf_counter_ns())
            if len(self.marks[i]) > 0:
                morsel.marks = Int(self.marks[i].unsafe_ptr())
            morsel.apply(
                self.steps[i], self.batch_size, self.joins, self.indexes
            )
            self.counts[2 * len(self.steps) + 1 + i] += (
                Int(perf_counter_ns()) - began
            )
            self.counts[2 * i + 2] += morsel.height()
            if morsel.height() == 0:
                return

    def _consume(
        mut self, mut morsel: Morsel, k: Int, offset: Int, length: Int
    ) raises:
        """Sink one morsel that has run every step into this worker's
        state: `k` numbers it, `offset` and `length` place its rows in the
        source."""
        if len(self.layout) > 0 and len(self.sets) > 0:
            self._sink_sets(morsel, k)
            return
        if len(self.layout) > 0:
            if len(self.sinks) == 0:
                self.sinks.append(AggregateSink(self.layout[0], self.workers))
            var in_place = _in_place_table_rows(
                morsel, self.keys, self.expressions
            )
            if in_place:
                trace_path("pipeline.table_in_place")
                ref rows = in_place.value()
                self.sinks[0].sink(rows[0], rows[1], k << 32, rows[2])
                self.rows += rows[3]
                return
            # The aggregate table: the morsel's keys and reduction
            # inputs into this worker's sink, no matter how many groups
            # (its tables adapt), with strings compared in place.
            var keys = List[Series](capacity=len(self.keys))
            var inputs: List[Series]
            var selected = _selected_table_rows(
                morsel, self.keys, self.expressions
            )
            if selected:
                trace_path("pipeline.table_inputs_selected")
                ref rows = selected.value()
                keys = rows[0].copy()
                inputs = rows[1].copy()
            else:
                var source = morsel._compact_frame(
                    self._table_reads()
                ) if _reads(self.expressions) else morsel.materialized()
                for name in self.keys:
                    keys.append(
                        canonical_labels(_unchunked(source.column(name)))
                    )
                inputs = table_inputs(source, keys, self.expressions)
            for i in range(len(inputs)):
                inputs[i] = _unchunked(inputs[i])
            if len(self.sinks) == 0:
                self.sinks.append(AggregateSink(self.layout[0], self.workers))
            # First rows numbered by morsel, then by row of its output
            # (a join's may outgrow the morsel), so the table orders
            # groups the same whichever worker took which morsel.
            self.sinks[0].sink(keys, inputs, k << 32)
            self.rows += len(keys[0])
            return
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
                self.grouped[0].select(offset, length, BoolColumn(List[Bool]()))
            self.rows += morsel.height()
            return
        if len(self.expressions) > 0:
            var reads = _reads(self.expressions)
            var expressions = self.expressions.copy()
            var input: DataFrame
            if reads and morsel.selected and len(self.keys) == 0:
                var counts = _row_counted(morsel, self.expressions)
                expressions = counts[0].copy()
                input = morsel._compact_frame(counts[1])
                for name in counts[2]:
                    # Only row counts read it: a window of the kept-row
                    # height stands in, and its values are never read.
                    var at = morsel._index(name)
                    input = input.with_column(
                        morsel.frame._columns[at].slice(0, morsel.count)
                    )
            elif reads:
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
                    return
                if self.grouped[0].count > 0:
                    raise Error("grouped pipeline: keys stopped partitioning")
                self.grouped = List[_GroupedSink]()
            # Group first rows are the frame's rows, so first-occurrence
            # order survives morsels taken out of order.
            var state = _StreamReduction(input, expressions, self.keys)
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


def _scatter(
    compact: Series, rows: List[Int], n: Int
) raises -> Optional[Series]:
    """A numeric or decimal column of the kept rows placed back at their
    positions `rows` among `n` (the others hold zero, for rows nothing
    reads); None for other storage."""
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if compact._data.isa[Column[Scalar[D]]]():
            return _scatter_typed[D](compact, rows, n)
    if compact._data.isa[Column[Int128]]():
        return _scatter_typed[DType.int128](compact, rows, n)
    return None


def _scatter_typed[
    D: DType
](compact: Series, rows: List[Int], n: Int) raises -> Series:
    ref column = compact._data[Column[Scalar[D]]]
    var values = List[Scalar[D]](length=n, fill=0)
    var out = values.unsafe_ptr()
    var source = column._ptr()
    var at = rows.unsafe_ptr()
    for j in range(len(rows)):
        out[unsafe_offset=at[unsafe_offset=j]] = source[unsafe_offset=j]
    if column.null_count() == 0:
        return Series(compact.name(), Column[Scalar[D]](values^)).with_dtype(
            compact.dtype()
        )
    var valid = List[Bool](length=n, fill=True)
    for j in range(len(rows)):
        valid[at[unsafe_offset=j]] = column._valid(j)
    return Series(compact.name(), Column[Scalar[D]](values^, valid)).with_dtype(
        compact.dtype()
    )


def _mask_rows(mask: BoolColumn, count: Int) -> List[Int]:
    """The rows a morsel's mask keeps (`count` of them), in order, found on
    this worker a byte at a time: `true_rows` splits one mask into jobs,
    which a morsel's worker must not start."""
    var n = len(mask)
    var rows = List[Int](capacity=count)
    if mask._offset % 8 == 0:
        var values = mask._data[].unsafe_ptr().unsafe_offset(mask._offset // 8)
        var all_valid = len(mask._bits[]) == 0
        var valid = mask._bits[].unsafe_ptr().unsafe_offset(mask._offset // 8)
        for j in range(n // 8):
            var byte = values[unsafe_offset=j]
            if not all_valid:
                byte &= valid[unsafe_offset=j]
            while byte != 0:
                rows.append(8 * j + Int(count_trailing_zeros(byte)))
                byte &= byte - 1
        for i in range(8 * (n // 8), n):
            if mask._valid(i) and mask._get(i):
                rows.append(i)
        return rows^
    for i in range(n):
        if mask._valid(i) and mask._get(i):
            rows.append(i)
    return rows^


def _selected_table_rows(
    mut morsel: Morsel, keys: List[String], expressions: List[Expr]
) raises -> Optional[Tuple[List[Series], List[Series]]]:
    """A selected morsel's aggregate-table keys and reduction inputs in
    `table_inputs`' order, one row per kept row, where every computed
    input is arithmetic or a cast (`selected_expressions`): those are
    computed for the kept rows straight from the columns, and only the
    keys and plain inputs are gathered. The columns a computed input alone
    reads are never copied (PDS-H's discounted prices read two lineitem
    columns after a join that keeps few rows). None when the morsel is
    not selected or an input is anything else."""
    if not morsel.selected:
        return None
    var plans = List[Optional[SelectedPlan]]()
    var plain = List[String]()
    var any_computed = False
    for expression in expressions:
        for node in expression._nodes:
            if not is_reduction(node.op):
                continue
            if node.op == CORR or node.op == COV:
                return None
            if node.left < 0:
                plans.append(None)
                continue
            ref input = expression._nodes[node.left]
            if input.op == COL and input.text in morsel.frame.columns():
                plans.append(None)
                if input.text not in plain:
                    plain.append(input.text)
                continue
            var plan = selected_plan(
                subtree(expression, node.left), morsel.frame._columns
            )
            if not plan:
                return None
            plans.append(plan^)
            any_computed = True
    if not any_computed:
        return None
    var computed = List[Optional[Series]](capacity=len(plans))
    var masked_rows = Optional[List[Int]]()
    for k in range(len(plans)):
        if not plans[k]:
            computed.append(None)
            continue
        var result = morsel._evaluate_selected(
            plans[k].value(), False, "__input", masked_rows
        )
        if not result:
            return None
        computed.append(result^)
    var names = keys.copy()
    for name in plain:
        if name not in names:
            names.append(name)
    var source = morsel._compact_frame(names)
    var key_columns = List[Series](capacity=len(keys))
    for name in keys:
        key_columns.append(canonical_labels(_unchunked(source.column(name))))
    var inputs = List[Series](capacity=len(plans))
    var at = 0
    for expression in expressions:
        for node in expression._nodes:
            if not is_reduction(node.op):
                continue
            if computed[at]:
                inputs.append(computed[at].value().copy())
            elif node.left < 0:
                inputs.append(key_columns[0].copy())
            else:
                inputs.append(source.column(expression._nodes[node.left].text))
            at += 1
    return (key_columns^, inputs^)


def _in_place_table_rows(
    mut morsel: Morsel, keys: List[String], expressions: List[Expr]
) raises -> Optional[Tuple[List[Series], List[Series], List[Bool], Int]]:
    """A filtered morsel's aggregate-table rows without compacting it: its
    key and plain input columns as they are, with a flag a row for the rows
    the filter keeps; each computed input evaluated over every row, or,
    when that raises, on the kept rows alone and placed back at their
    positions, so only a kept row's error is raised. For a selection that
    keeps at least
    half the morsel, where compacting every column cost more than reading
    the dropped rows (PDS-H q1 keeps 98.6%: a third of its time went to
    compacting, including two string keys' views and codes). None when the
    selection is sparse or a column was already compacted. A join's
    selection by rows is read the same way, through the mask kept beside
    its rows."""
    var n = morsel.frame.height()
    if not morsel.selected or 2 * morsel.count < n:
        return None
    for expression in expressions:
        for node in expression._nodes:
            if node.op == CORR or node.op == COV:
                return None
    var key_columns = List[Series](capacity=len(keys))
    for name in keys:
        var at = morsel._index(name)
        if morsel.compact[at]:
            return None
        key_columns.append(
            canonical_labels(_unchunked(morsel.frame._columns[at]))
        )
    # Plain inputs are taken before any computed input compacts a column.
    var inputs = List[Series]()
    var computed = List[Int]()
    var sources = List[Expr]()
    for expression in expressions:
        for i in range(len(expression._nodes)):
            ref node = expression._nodes[i]
            if not is_reduction(node.op):
                continue
            if node.left < 0:
                inputs.append(key_columns[0].copy())
                continue
            ref input = expression._nodes[node.left]
            if input.op == COL and input.text in morsel.frame.columns():
                var at = morsel._index(input.text)
                if morsel.compact[at]:
                    return None
                inputs.append(_unchunked(morsel.frame._columns[at]))
                continue
            computed.append(len(inputs))
            sources.append(subtree(expression, node.left))
            inputs.append(key_columns[0].copy())
    var masked_rows = Optional[List[Int]]()
    for c in range(len(computed)):
        # Arithmetic and casts are computed for the kept rows alone, each
        # value at its row (`selected_expressions`).
        var plan = selected_plan(sources[c], morsel.frame._columns)
        if plan:
            var placed = morsel._evaluate_selected(
                plan.value(), True, "__input", masked_rows
            )
            if placed:
                inputs[computed[c]] = placed.take()
                continue
        var reads = _reads([sources[c].copy()])
        if not reads:
            return None
        var whole = List[Series]()
        for name in reads.value():
            var at = morsel._index(name)
            if morsel.compact[at]:
                return None
            whole.append(morsel.frame._columns[at].copy())
        # Over every row, dropped ones included; an error is retried on
        # the kept rows alone, so only a kept row's error is raised.
        var evaluated: Series
        try:
            evaluated = (
                DataFrame(whole^, height=n)
                .select_exprs([sources[c].alias("__input")])
                ._columns[0]
                .copy()
            )
        except:
            var rows = true_rows(morsel.mask)
            var kept = morsel._compact_frame(reads.value())
            var placed = _scatter(
                _unchunked(
                    kept.select_exprs([sources[c].alias("__input")])._columns[0]
                ),
                rows,
                n,
            )
            if not placed:
                return None
            evaluated = placed.take()
        inputs[computed[c]] = _unchunked(evaluated)
    ref mask = morsel.mask
    var keep = List[Bool](unsafe_uninit_length=n)
    var out = keep.unsafe_ptr()
    var done = 0
    if mask._offset % 8 == 0:
        # A byte of the mask (its values ANDed with its validity) at a
        # time, spread to eight flags with one compare, where reading each
        # row's two bits took a tenth of PDS-H q1.
        var values = mask._data[].unsafe_ptr().unsafe_offset(mask._offset // 8)
        var all_valid = len(mask._bits[]) == 0
        var valid = mask._bits[].unsafe_ptr().unsafe_offset(mask._offset // 8)
        var lanes = SIMD[DType.uint8, 8](1, 2, 4, 8, 16, 32, 64, 128)
        var flags = out.unsafe_bitcast[UInt8]()
        for j in range(n // 8):
            var byte = values[unsafe_offset=j]
            if not all_valid:
                byte &= valid[unsafe_offset=j]
            var spread = (SIMD[DType.uint8, 8](byte) & lanes).ne(
                SIMD[DType.uint8, 8](0)
            )
            flags.unsafe_offset(8 * j).unsafe_store(spread.cast[DType.uint8]())
        done = 8 * (n // 8)
    for i in range(done, n):
        out[unsafe_offset=i] = mask._valid(i) and mask._get(i)
    return (key_columns^, inputs^, keep^, morsel.count)


def _unchunked(column: Series) raises -> Series:
    """A column as one buffer: a morsel's slice within one chunk is that
    chunk's view, not a copy; only a slice across chunks is copied."""
    if not column.is_chunked():
        return column.copy()
    if column.n_chunks() == 1:
        return column.chunks()[0].copy()
    return column.rechunk()


def _top_rows(
    candidates: DataFrame,
    names: List[String],
    descending: List[Bool],
    nulls_last: List[Bool],
    k: Int,
    threads: Int,
    by_place: Bool,
) raises -> DataFrame:
    """The k best candidate rows in order. The selection is stable, so ties
    keep the candidates' order; `by_place` sorts ties by the hidden source
    place instead, for candidates from several workers. A worker's own
    candidates are already in source order (it takes morsels in sequence
    order and a fold keeps tied rows in order), so its folds leave the
    place out: as a second key it turned a one-string-key sort into a
    two-key one, which the packed selection does not take, and ranked
    every string per fold (ClickBench q25: 32 -> 250 ms). With several keys, the rows are first cut to those whose first
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
    if not by_place:
        return pool.take(
            pool._arg_sort_head(names, descending, nulls_last, k, threads)
        )
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


# DuckDB's STANDARD_VECTOR_SIZE: the fewest rows a morsel is cut to.
comptime _VECTOR_ROWS = 2048


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
    if total == 1 and lanes > 1:
        # One piece left every worker but one idle: a source of about a
        # batch (a 65K-row common table read twice in TPC-DS q47) is cut
        # into pieces of a batch's share per worker, as a source of a few
        # batches is below, and never under one vector of `_VECTOR_ROWS`
        # (DuckDB's smallest unit of parallel work). A smaller source
        # stays whole: every morsel pays its steps' setup.
        var share = max(_VECTOR_ROWS, morsel_rows // lanes)
        var most = min(lanes, lengths[0] // share)
        if most > 1:
            pieces[0] = most
            total = most
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


def _emit_unmatched(
    mut jobs: List[_PipelineJob],
    steps: List[Step],
    joins: ArcPointer[List[DataFrame]],
    indexes: ArcPointer[List[Optional[PreparedHashIndex]]],
    batch_size: Int,
) raises:
    """The build rows no morsel matched, for each right or full join step
    in step order, as morsels of that step's output (the probe columns
    null) through the steps after it and into the workers' sinks: Polars'
    `emit_unmatched_build` (`EmitUnmatchedState`, polars-stream equi_join)
    and the outer scan of DuckDB's `PhysicalHashJoin::GetData`, which
    hands each thread its share of the build rows. Every worker flagged
    the build rows its own morsels matched; each worker then takes
    morsels of the build rows, finds those no worker flagged and makes
    their output rows (`_PipelineJob._unmatched`), so nothing is merged
    or gathered on one thread. The last round finishes the workers'
    sinks."""
    var next_morsel = len(jobs[0].ranges[]) // 2
    var last = -1
    for j in range(len(steps)):
        if steps[j].kind == STEP_JOIN and (
            steps[j].how == JOIN_RIGHT or steps[j].how == JOIN_FULL
        ):
            last = j
    if last < 0:
        return
    var source = jobs[0].frame[].clear()
    for j in range(last + 1):
        if steps[j].kind != STEP_JOIN or not (
            steps[j].how == JOIN_RIGHT or steps[j].how == JOIN_FULL
        ):
            continue
        ref build = joins[][steps[j].join]
        var ranges = List[Int]()
        if build.height() > 0:
            ranges = _morsel_ranges(build, max(1, batch_size), len(jobs))
        if len(ranges) == 0 and j != last:
            continue
        # The probe side's columns at this step: an empty source through
        # the steps before it.
        var empty = source.copy()
        for i in range(j):
            var probe = Morsel(empty.copy(), 0)
            probe.apply(steps[i], batch_size, joins, indexes)
            empty = probe.materialized()
        var marks = List[Int](capacity=len(jobs))
        for w in range(len(jobs)):
            marks.append(Int(jobs[w].marks[j].unsafe_ptr()))
        # One more round of the workers over the build rows' morsels,
        # each into its own sink; the last round also finishes them.
        var cursor = _Cursor.new()
        var shared_build = ArcPointer(build.copy())
        var shared_ranges = ArcPointer(ranges^)
        var shared_marks = ArcPointer(marks^)
        var shared_probe = ArcPointer(empty^)
        for w in range(len(jobs)):
            jobs[w].frame = shared_build
            jobs[w].ranges = shared_ranges
            jobs[w].cursor = cursor.address
            jobs[w].first_step = j + 1
            jobs[w].first_morsel = next_morsel
            jobs[w].finishes = j == last
            jobs[w].unmatched_step = j
            jobs[w].unmatched_marks = shared_marks
            jobs[w].unmatched_probe = shared_probe
        next_morsel += len(shared_ranges[]) // 2
        try:
            run_jobs(jobs)
        except e:
            cursor.free()
            raise e
        cursor.free()


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
    mut set_frames: List[DataFrame],
    sets: List[List[Int]] = List[List[Int]](),
    top: Int = -1,
    top_names: List[String] = List[String](),
    top_descending: List[Bool] = List[Bool](),
    top_nulls_last: List[Bool] = List[Bool](),
    bounded: Bool = False,
    table: Bool = False,
) raises -> DataFrame:
    """Run the steps over `frame` in morsels on `workers` threads, into a
    materialize sink, or a reduce sink when `expressions` are reductions
    over the steps' output, grouped by `keys` when given (`ordered`: groups
    in first-occurrence order; `insert`: rows go into per-worker group
    tables rather than being collected for the eager partitioned
    group-by; `table`: into the aggregate table, when it keeps every
    state inline, whatever `insert` says; `sets`: grouping sets, each a
    list of indices into `keys`, one sink per set fed by every morsel,
    whose finished frames (set order) land in `set_frames`, the grand
    total's among them with its one row, and the result is the steps'
    empty output). `counts` receives the rows fed,
    then each step's input and output rows."""
    trace_path("pipeline.run")
    # The aggregate table's layout, from the steps' output types (an
    # empty frame through every step) when it takes this reduction.
    var layout = List[AggregateLayout]()
    var template = List[_StreamReduction]()
    var set_empty = List[DataFrame]()
    if table and len(keys) > 0 and len(expressions) > 0 and not ordered:
        var empty = frame.clear()
        for step in steps:
            var morsel = Morsel(empty.copy(), 0)
            morsel.apply(step, batch_size, joins, indexes)
            empty = morsel.materialized()
        if len(sets) > 0 and table_supported(empty, keys, expressions):
            set_empty.append(empty.copy())
            var empty_inputs = table_inputs(
                empty, [empty.column(keys[0])], expressions
            )
            for one in sets:
                var names = List[String]()
                var typed = empty.copy()
                if len(one) == 0:
                    names.append(_TOTAL_KEY)
                    typed = typed.with_column(
                        Series(_TOTAL_KEY, Column[Int64](List[Int64]()))
                    )
                else:
                    for i in one:
                        names.append(keys[i])
                var empty_keys = List[Series](capacity=len(names))
                for name in names:
                    empty_keys.append(typed.column(name))
                var reduction = _StreamReduction(typed, expressions, names)
                layout.append(
                    aggregate_layout(
                        empty_keys, empty_inputs.copy(), table_specs(reduction)
                    )
                )
                template.append(reduction^)
        elif len(sets) == 0 and table_supported(empty, keys, expressions):
            var empty_keys = List[Series](capacity=len(keys))
            for name in keys:
                empty_keys.append(empty.column(name))
            var reduction = _StreamReduction(empty, expressions, keys)
            layout.append(
                aggregate_layout(
                    empty_keys,
                    table_inputs(empty, empty_keys, expressions),
                    table_specs(reduction),
                )
            )
            template.append(reduction^)
    if len(layout) > 0:
        trace_path("pipeline.group_table")
    elif len(keys) > 0 and len(expressions) > 0:
        if insert:
            trace_path("pipeline.group_insert")
        else:
            trace_path("pipeline.group_collect")
    var shared = ArcPointer(frame.copy())
    var ranges = ArcPointer(
        _morsel_ranges(frame, max(1, morsel_rows), max(1, workers))
    )
    # Grouping sets: which come from another set's group rows. Largest
    # first, each takes the smallest earlier source whose keys hold its own,
    # when every reduction merges (`reaggregable`).
    var derived_from = List[Int](length=len(sets), fill=-1)
    var derived_columns = List[List[Int]]()
    var derive_order = List[Int]()
    for _ in range(len(sets)):
        derived_columns.append(List[Int]())
    if len(layout) > 0 and len(sets) > 1:
        var merges = True
        for expression in expressions:
            for node in expression._nodes:
                if is_reduction(node.op) and not reaggregable(node.op):
                    merges = False
        if merges:
            var order = derive_order.copy()
            for t in range(len(sets)):
                order.append(t)
            # Stable by size, largest first.
            for i in range(1, len(order)):
                var j = i
                while j > 0 and len(sets[order[j - 1]]) < len(sets[order[j]]):
                    var held = order[j - 1]
                    order[j - 1] = order[j]
                    order[j] = held
                    j -= 1
            derive_order = order.copy()
            var primaries = List[Int]()
            for t in order:
                var best = -1
                for q in primaries:
                    var inside = True
                    for key in sets[t]:
                        if key not in sets[q]:
                            inside = False
                    if inside and (best < 0 or len(sets[q]) < len(sets[best])):
                        best = q
                # Every set made so far is a source, a derived one too: a
                # rollup's sets come each from the one above it.
                primaries.append(t)
                if best < 0:
                    continue
                derived_from[t] = best
                for key in sets[t]:
                    derived_columns[t].append(sets[best].index(key))
            var primary_count = 0
            for t in range(len(sets)):
                if derived_from[t] < 0:
                    primary_count += 1
            if primary_count < len(sets):
                trace_path("pipeline.group_table_derived_sets")
    var cursor = _Cursor.new()
    var shared_bound = _SharedBound.new()
    var floating_key = False
    if top > 0 and len(top_names) > 0 and top_names[0] in frame.columns():
        floating_key = frame.column(top_names[0]).dtype().physical() in (
            DataType.FLOAT64,
            DataType.FLOAT32,
        )
    var key_ranges = _key_ranges(frame, steps, joins[])
    if len(key_ranges) > 0:
        trace_path("pipeline.key_range")
    for r in key_ranges:
        if r.bloom:
            trace_path("pipeline.key_bloom")
            break
    for r in key_ranges:
        _ensure_zones(frame.column(r.column))
    # A top-k on an integer key reads each morsel's key range from the
    # zone maps before comparing its rows with the bound (`_zoned_top`).
    if (
        top > 0
        and len(top_names) > 0
        and top_names[0] in frame.columns()
        and _range_key(frame.column(top_names[0]).dtype())
    ):
        _ensure_zones(frame.column(top_names[0]))
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
                insert,
                batch_size,
                top,
                top_names,
                top_descending,
                top_nulls_last,
                bounded,
                shared_bound,
                floating_key,
                layout,
                sets,
                derived_from,
                derived_columns,
                derive_order,
                workers,
                key_ranges,
            )
        )
    # An outer join's unmatched build rows run after every morsel of the
    # source (`_emit_unmatched`), and the workers finish after them.
    for step in steps:
        if step.kind == STEP_JOIN and (
            step.how == JOIN_RIGHT or step.how == JOIN_FULL
        ):
            for w in range(len(jobs)):
                jobs[w].finishes = False
    try:
        run_jobs(jobs)
        _emit_unmatched(jobs, steps, joins, indexes, batch_size)
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
    if len(layout) > 0 and len(sets) > 0:
        # Derived sets finish first: their rows point at key bytes the
        # source set's sinks hold. A worker's sinks are in set order, so a
        # set's is found by its place among those still held.
        var finish_order = List[Int]()
        for back in range(len(sets)):
            var t = len(sets) - 1 - back
            if derived_from[t] >= 0:
                finish_order.append(t)
        for back in range(len(sets)):
            var t = len(sets) - 1 - back
            if derived_from[t] < 0:
                finish_order.append(t)
        var remaining = List[Int]()
        for t in range(len(sets)):
            remaining.append(t)
        var finished_sets = List[DataFrame](
            length=len(sets), fill=frame.clear()
        )
        for s in finish_order:
            var place = remaining.index(s)
            var sinks = List[AggregateSink]()
            for j in range(len(jobs)):
                if len(jobs[j].sinks) > 0:
                    sinks.append(jobs[j].sinks.pop(place))
            _ = remaining.pop(place)
            var result = aggregate_finish(sinks, layout[s])
            var finished = table_frame(template[s], result^, False)
            if len(sets[s]) == 0:
                if finished.height() == 0:
                    # No rows: the grand total is still one row.
                    finished = set_empty[0].select_exprs(expressions)
                else:
                    finished = finished.drop([_TOTAL_KEY])
            finished_sets[s] = finished^
            _ = sinks^
        for s in range(len(sets)):
            set_frames.append(finished_sets[s].copy())
        # An empty frame of the steps' output types the caller's null keys.
        return set_empty[0].copy()
    if len(layout) > 0:
        var sinks = List[AggregateSink]()
        for j in range(len(jobs)):
            if len(jobs[j].sinks) > 0:
                sinks.append(jobs[j].sinks.pop())
        var result = aggregate_finish(sinks, layout[0])
        # Long key strings live in the sinks' heaps until the finish has
        # read them out.
        var finished = table_frame(template[0], result^, False)
        _ = sinks^
        return finished^
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
                var tables = pieces.pop(0)
                while len(tables) > 0:
                    parts[p].append(tables.pop(0))
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
            merged, top_names, top_descending, top_nulls_last, top, 0, True
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
    when `ordered`."""
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
