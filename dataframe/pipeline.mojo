"""Thread-owned pipelines over data chunks (docs/design/executor-design.md,
#538).

A pipeline is a source, a list of row-local steps, and a sink. Each worker
thread takes morsels (ranges of the source) from the shared cursor, cuts
each into data chunks of STANDARD_VECTOR_SIZE rows (DuckDB's DataChunk,
`dataframe/vector`), runs the chunks through the steps, and updates its
own sink state. Sink states are combined once the source is exhausted;
the main thread does no per-morsel work.

A row-local step reads its input chunk's vectors through the unified
format: a filter's result is the same vectors as dictionaries through its
selection, so nothing is copied for it, and a projection computes only
the rows the selection keeps. Every pipeline ends in a sink taking the
last step's chunks one at a time (`dataframe/vector/sink.mojo`). Join
probes do not read vectors yet: a morsel's chunks are collected for them
(`_Collector`), each column kept as a range or a row list over the column
it reads, and gathered through it only when the probe reads a frame.
"""
from dataframe.storage.buffer import _Buffer, _BufferData
from std.atomic import Atomic
from std.bit import count_trailing_zeros, pop_count
from std.os import getenv
from std.ffi import external_call
from std.memory import ArcPointer, Pointer, bitcast, unsafe_memcpy
from std.sys import size_of
from std.time import perf_counter_ns

from dataframe.kernels.canonical_labels import canonical_labels
from dataframe.storage.boolean import BoolColumn
from .expr import COL, SELECTOR, WHEN, Expr
from dataframe.aggregation.reducer import Reducer
from .binding import BoundExpr, bind
from dataframe.storage.column import Column, _copy_validity
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
    col,
    is_reduction,
    node_children,
    subtree,
)
from dataframe.aggregation.hash_table import (
    AggregateLayout,
    AggregateSink,
    GroupFilter,
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
from dataframe.aggregation.table_sink import (
    TOTAL_KEY,
    GroupingSetsLayout,
    GroupingSetsSink,
    TableSink,
    TableSinkLayout,
)
from dataframe.aggregation.ungrouped import (
    UngroupedLayout,
    UngroupedSink,
    ungrouped_finish,
    ungrouped_layout,
)
from dataframe.sorting.top_n import TopNBoundary
from dataframe.sorting.top_n_sink import TopNSink
from dataframe.vector.column_data_collection import (
    MaterializeSink,
    append_raw,
    materialize_finish,
    raw_series,
)
from dataframe.vector.sink import Sink
from dataframe.storage.statistics import (
    STATISTICS_VECTOR_SIZE,
    COMPARE_GREATERTHAN,
    COMPARE_GREATERTHANOREQUALTO,
    COMPARE_LESSTHAN,
    COMPARE_LESSTHANOREQUALTO,
)
from dataframe.table_filter import (
    DynamicFilterData,
    TableFilterSet,
    column_domain,
)
from dataframe.table_scan import ScanFilterInfo
from dataframe.aggregation.state import (
    _StreamReduction,
    chunks_frame,
    table_frame,
    table_parts,
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
from dataframe.joins.hash_table import (
    JoinHashTable,
    ScanStructure,
)
from dataframe.joins.join_filter import (
    BF_CHECK_N,
    BF_THRESHOLD,
    JoinFilter,
    PushedJoinFilter,
    SelectivityStats,
    finalize_filters,
    keep_rows,
    supports_join_filters,
)
from dataframe.table_filter import TableFilterSet
from .join_type import JOIN_ANTI, JOIN_FULL, JOIN_INNER, JOIN_RIGHT, JOIN_SEMI
from dataframe.kernels.row_encode import encodable
from dataframe.kernels.gather import (
    can_filter_aligned_chunks,
    take_parallel,
    true_rows,
)
from dataframe.storage.allocation import huge_list
from dataframe.kernels.mask_filter import filter_columns
from .parallel import Job, run_jobs, worker_count
from .series import Series
from dataframe.kernels.segment_statistics import (
    ensure_statistics,
    series_statistics,
)
from dataframe.storage.statistics import NUMERIC_STATS
from dataframe.table_scan import ScanFilterInfo, scan_filter_info
from dataframe.kernels.hashing import encode_rows
from dataframe.storage.string import StringColumn
from .trace import trace_path
from .selectors import expand
from .expr import LIT_BOOL, conjuncts
from dataframe.aggregation.expressions import _bind_all, _row_local
from dataframe.vector.data_chunk import (
    ColumnArena,
    DataChunk,
    flat_over,
    to_series,
)
from dataframe.vector.expression import VectorExpression, _KIND_COLUMN
from dataframe.vector.types import (
    PHYSICAL_BOOL,
    PHYSICAL_VARCHAR,
    physical_of,
    CONSTANT_VECTOR,
    DICTIONARY_VECTOR,
    FLAT_VECTOR,
    SEQUENCE_VECTOR,
    STANDARD_VECTOR_SIZE,
    SelectionVector,
    Vector,
    physical_size,
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


# The column a top-k sink adds as its last key when the sort keeps input
# order: the row's place in the source's order (morsel sequence, then row
# within the morsel's output), so ties come out as a stable sort of the
# whole input would place them, whichever worker held them.
comptime _ORDER_COLUMN = "__pipeline_order"
# A late-materialized top-N scan's row id column (DuckDB's `rowid`).
comptime ROW_ID_COLUMN = "__dataframe_row_id"
# The constant key a grouping sets' grand total groups by; dropped from
# that set's result.
comptime _TOTAL_KEY = TOTAL_KEY


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


comptime _AGGREGATE_INPUT = "__aggregate_input_"


def _extract_aggregate_inputs(
    steps: List[Step],
    expressions: List[Expr],
    keys: List[String],
    schema: DataFrame,
) raises -> Tuple[List[Step], List[Expr]]:
    """DuckDB's ExtractAggregateExpressions (physical_plan/plan_aggregate.cpp):
    every aggregate's computed input becomes a column of a projection
    below the aggregate, and the aggregate reads that column. The
    projection runs on the chunks the steps before it make, through their
    selections, so a filter's kept rows reach the sink as the computed
    values only, never as every input column gathered first. `schema` is
    the steps' output; the steps are returned unchanged when no input is
    computed (a projection of column references only moves no data)."""
    var projected = List[Expr]()
    var rewritten = List[Expr](capacity=len(expressions))
    for e in expressions:
        var nodes = e._nodes.copy()
        var changed = False
        for i in range(len(nodes)):
            if not is_reduction(nodes[i].op):
                continue
            # An input `when(c).then(x)` with no otherwise is DuckDB's
            # aggregate FILTER (WHERE c) over x: its condition and its
            # value are projected each on its own, as the aggregate's
            # filter and child are (`bound_aggr.GetFilter()`).
            var targets = List[Int]()
            for child in node_children(nodes[i]):
                if nodes[child].op == WHEN and nodes[child].extra < 0:
                    targets.append(nodes[child].left)
                    targets.append(nodes[child].right)
                else:
                    targets.append(child)
            for child in targets:
                if nodes[child].op == COL:
                    continue
                var input = subtree(e, child)
                var reads = False
                var local = True
                for node in input._nodes:
                    reads = reads or node.op == COL
                    local = local and not (
                        is_reduction(node.op) or node.op == SELECTOR
                    )
                if not reads or not local or not _row_local([input.copy()]):
                    continue
                var name = _AGGREGATE_INPUT + String(len(projected))
                while name in schema.columns():
                    name += "_"
                projected.append(input.alias(name))
                nodes[child] = col(name)._nodes[0].copy()
                changed = True
        if changed:
            var compact = Expr(nodes^, e._name)
            rewritten.append(subtree(compact, len(compact._nodes) - 1))
        else:
            rewritten.append(e.copy())
    if len(projected) == 0:
        return (steps.copy(), expressions.copy())
    # The projection's columns: the group keys, the columns the aggregates
    # still read as they are, then the computed inputs.
    var exprs = List[Expr]()
    var names = List[String]()
    for key in keys:
        if key not in names:
            names.append(key)
    var reads = _reads(rewritten)
    if not reads:
        return (steps.copy(), expressions.copy())
    for name in reads.value():
        if name not in names and not name.startswith(_AGGREGATE_INPUT):
            names.append(name)
    for name in names:
        exprs.append(col(name))
    for p in projected:
        exprs.append(p.copy())
    var result = steps.copy()
    result.append(
        Step(
            STEP_SELECT,
            exprs^,
            List[String](),
            -1,
            List[String](),
            JOIN_INNER,
            String(""),
            False,
        )
    )
    return (result^, rewritten^)


def _row_counted(
    schema: DataFrame,
    expressions: List[Expr],
    null_free: List[Bool],
) raises -> Tuple[List[Expr], List[String], List[String]]:
    """`expressions` with each count of a column that holds no null
    (`null_free`, per column of `schema`) turned into a row count, the
    columns they still read values of, and the columns read only by row
    counts. DuckDB's rule (its count statistics callback): a count of a
    column without nulls is count(*), so it is the kept-row count, and the
    column is never gathered through its rows."""
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
                if null_free[schema._index(nodes[child].text)]:
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


def _range_key(dtype: DataType) -> Bool:
    """Keys whose values compare as their physical integers."""
    var physical = dtype.physical()
    return physical == DataType.INT64 or physical == DataType.INT32


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


def _ensure_zones(series: Series):
    """Compute the segment statistics of each buffer under an integer
    column (see `_range_key`) that has none, before any worker reads them
    (`ensure_statistics`; a loaded table has them already). Without them
    bounds are read from the rows."""
    if _range_key(series.dtype()):
        try:
            ensure_statistics(series)
        except:
            pass


def _window_bounds(series: Series) -> Optional[Tuple[Int64, Int64]]:
    """Bounds on the valid values of an integer column's window, from the
    statistics of the vectors it touches (vectors it covers only in part
    widen them, never narrow them), or exact when a buffer has none; None
    when it has no valid value."""
    var least = Int64.MAX
    var most = Int64.MIN
    for part in series.chunks():
        var found = _zoned_bounds(part)
        if found[0]:
            least = min(least, found[1])
            most = max(most, found[2])
    if least > most:
        return None
    return (least, most)


def _zoned_bounds(part: Series) -> Tuple[Bool, Int64, Int64]:
    """An unchunked integer window's bounds, from its buffer's statistics
    when it has them, else from its rows."""
    if len(part) == 0:
        return (False, Int64.MAX, Int64.MIN)
    var stats = series_statistics(part)
    if stats and stats.value().stats_type == NUMERIC_STATS:
        ref found = stats.value()
        if not found.has_min_max:
            return (False, Int64.MAX, Int64.MIN)
        return (True, Int64(found.int_min), Int64(found.int_max))
    if part.dtype().physical() == DataType.INT64:
        return _column_bounds(part._data[Column[Int64]])
    return _column_bounds(part._data[Column[Int32]])


def _join_filters(
    frame: DataFrame,
    steps: List[Step],
    joins: List[DataFrame],
    indexes: List[Optional[JoinHashTable]],
    pushed: List[PushedJoinFilter],
    mut scan: TableFilterSet,
) raises -> List[JoinFilter]:
    """DuckDB's join filter pushdown into the source: for each inner, semi
    or right join among `steps` whose probe key is a source column no
    earlier step replaces (every build is finished before the first
    morsel), and for each key `pushed` from a join above the aggregate
    these steps feed whose column of the steps' output is such a source
    column, its statistics filters go into `scan` and its row filters
    come back (`join_filter.finalize_filters`)."""
    var filters = List[JoinFilter]()
    # Names the steps so far show for untouched source columns.
    var source = Dict[String, String]()
    for name in frame.columns():
        source[name] = name
    for at in range(len(steps)):
        ref step = steps[at]
        if step.kind == STEP_JOIN:
            ref build = joins[step.join]
            if (
                supports_join_filters(step.how)
                and len(step.names) == len(step.right_keys)
                and indexes[step.join]
            ):
                ref table = indexes[step.join].value()
                for k in range(len(step.names)):
                    var probe = source.get(step.names[k])
                    if not probe:
                        continue
                    var name = probe.value()
                    var made = finalize_filters(
                        name,
                        frame.column(name).dtype(),
                        build.column(step.right_keys[k]),
                        at,
                        len(step.names),
                        table.inserted,
                        frame.height(),
                        table.build_filtered,
                        scan,
                    )
                    if made:
                        filters.append(made.take())
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
    for p in pushed:
        var probe = source.get(p.column)
        if not probe:
            continue
        var name = probe.value()
        var made = finalize_filters(
            name,
            frame.column(name).dtype(),
            p.key,
            -1,
            p.equalities,
            p.build_count,
            frame.height(),
            p.build_filtered,
            scan,
        )
        if made:
            trace_path("pipeline.join_filter_through_aggregate")
            filters.append(made.take())
    return filters^


struct _Columns(Copyable, Movable):
    """A morsel's columns between the steps that read vectors and the
    operators that do not yet (a join's probe, a sink): column c is the
    rows from `starts[c]` of arena column `handles[c]`, or, when
    `lists[c] >= 0`, the rows `row_lists[lists[c]]` of it, a dictionary
    vector over the column (of any length) as DuckDB's join emits its
    build side. Every column has `height` rows. Chunks of
    STANDARD_VECTOR_SIZE rows are cut from it without a copy."""

    var handles: List[Int]
    var starts: List[Int]
    var lists: List[Int]
    var row_lists: List[List[UInt32]]
    var height: Int

    def __init__(out self):
        self.handles = List[Int]()
        self.starts = List[Int]()
        self.lists = List[Int]()
        self.row_lists = List[List[UInt32]]()
        self.height = 0

    @staticmethod
    def of_frame(mut arena: ColumnArena, frame: DataFrame) raises -> _Columns:
        """Every column of `frame`, held in the arena, read whole."""
        var result = _Columns()
        for column in frame._columns:
            result.handles.append(arena.add(column))
            result.starts.append(0)
            result.lists.append(-1)
        result.height = frame.height()
        return result^

    def width(self) -> Int:
        return len(self.handles)

    def through(self, rows: List[Int]) -> _Columns:
        """Every column read at its rows `rows` (positions among this one's
        `height`), the dictionaries merged: a row list over a row list is
        one row list over the column (SelectionVector::Slice). Columns
        sharing a list, or a start, share the merged list."""
        var result = _Columns()
        result.height = len(rows)
        var from_list = List[Int]()
        var to_list = List[Int]()
        var from_start = List[Int]()
        var start_list = List[Int]()
        var at = rows.unsafe_ptr()
        var n = len(rows)
        for c in range(self.width()):
            result.handles.append(self.handles[c])
            result.starts.append(0)
            var made = -1
            if self.lists[c] >= 0:
                for m in range(len(from_list)):
                    if from_list[m] == self.lists[c]:
                        made = to_list[m]
                if made < 0:
                    var merged = List[UInt32](unsafe_uninit_length=n)
                    var out = merged.unsafe_ptr()
                    var source = self.row_lists[self.lists[c]].unsafe_ptr()
                    for k in range(n):
                        out[unsafe_offset=k] = source[
                            unsafe_offset=at[unsafe_offset=k]
                        ]
                    made = len(result.row_lists)
                    result.row_lists.append(merged^)
                    from_list.append(self.lists[c])
                    to_list.append(made)
            else:
                for m in range(len(from_start)):
                    if from_start[m] == self.starts[c]:
                        made = start_list[m]
                if made < 0:
                    var merged = List[UInt32](unsafe_uninit_length=n)
                    var out = merged.unsafe_ptr()
                    var start = self.starts[c]
                    for k in range(n):
                        out[unsafe_offset=k] = UInt32(
                            start + at[unsafe_offset=k]
                        )
                    made = len(result.row_lists)
                    result.row_lists.append(merged^)
                    from_start.append(self.starts[c])
                    start_list.append(made)
            result.lists.append(made)
        return result^

    def _picked(self, list: Int) -> List[Int]:
        ref rows = self.row_lists[list]
        var picked = List[Int](unsafe_uninit_length=self.height)
        var out = picked.unsafe_ptr()
        var at = rows.unsafe_ptr()
        for k in range(self.height):
            out[unsafe_offset=k] = Int(at[unsafe_offset=k])
        return picked^

    def series(
        self, arena: ColumnArena, c: Int, var name: String
    ) raises -> Series:
        """Column `c` as an at-rest column of `height` rows: a slice of its
        arena column when read whole, else its rows gathered."""
        ref column = arena.columns[self.handles[c]]
        var result: Series
        if self.lists[c] < 0:
            if self.starts[c] == 0 and len(column) == self.height:
                result = column.copy()
            else:
                result = column.slice(self.starts[c], self.height)
        else:
            ref rows = self.row_lists[self.lists[c]]
            var picked = List[Int](unsafe_uninit_length=self.height)
            var out = picked.unsafe_ptr()
            var at = rows.unsafe_ptr()
            for k in range(self.height):
                out[unsafe_offset=k] = Int(at[unsafe_offset=k])
            result = column.take(picked)
        return result.renamed(name^)

    def frame(
        self, arena: ColumnArena, schema: DataFrame, names: List[String]
    ) raises -> DataFrame:
        """The named columns as a frame of `height` rows (each gathered
        through its rows): the adapter for operators that read frames."""
        var columns = List[Series](capacity=len(names))
        # Each row list as gather indices once, for every column reading it.
        var picked = List[Optional[List[Int]]](
            length=len(self.row_lists), fill=None
        )
        for name in names:
            var c = schema._index(name)
            var l = self.lists[c]
            if l < 0:
                columns.append(self.series(arena, c, name))
                continue
            if not picked[l]:
                picked[l] = self._picked(l)
            columns.append(
                arena.columns[self.handles[c]]
                .take(picked[l].value())
                .renamed(name)
            )
        return DataFrame(columns^, height=self.height)

    def whole(self, arena: ColumnArena, schema: DataFrame) raises -> DataFrame:
        """Every column as a frame of `height` rows."""
        return self.frame(arena, schema, schema.columns())

    def bases(self, arena: ColumnArena) -> List[Vector]:
        """Per column, a flat vector over its arena column from row 0."""
        var result = List[Vector](capacity=self.width())
        for c in range(self.width()):
            result.append(flat_over(arena, self.handles[c], 0))
        return result^

    def fill(
        self,
        bases: List[Vector],
        arena: ColumnArena,
        mut chunk: DataChunk,
        low: Int,
        count: Int,
    ):
        """`chunk` as rows [low, low + count) of these columns: flat
        windows of the columns read whole, dictionaries through a window of
        the row lists of the others (no copy)."""
        chunk.reset()
        chunk.set_columns(self.width())
        for c in range(self.width()):
            var v = bases[c]
            if self.lists[c] < 0:
                var at = self.starts[c] + low
                if v.data != 0:
                    v.data += physical_size(v.physical) * at
                if not v.validity.all_valid():
                    v.validity.offset += at
                v.source_offset = at
            else:
                v.vector_type = DICTIONARY_VECTOR
                v.sel = SelectionVector(
                    Int(self.row_lists[self.lists[c]].unsafe_ptr()) + 4 * low
                )
                v.dictionary_size = len(arena.columns[self.handles[c]])
            chunk.data[c] = v
        chunk.count = count


def _append_rows(mut listed: List[UInt32], v: Vector, count: Int):
    """The at-rest rows of a vector over a source, appended."""
    # Grown geometrically: an exact reservation per chunk copied the
    # list again for every chunk.
    if listed.capacity() < len(listed) + count:
        listed.reserve(max(len(listed) + count, 2 * listed.capacity()))
    var base = UInt32(v.source_offset)
    if v.vector_type == DICTIONARY_VECTOR and v.sel.data != 0:
        var sel = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=v.sel.data)
        for i in range(count):
            listed.append(base + sel[unsafe_offset=i])
    elif v.vector_type == CONSTANT_VECTOR:
        for _ in range(count):
            listed.append(base)
    else:
        for i in range(count):
            listed.append(base + UInt32(i))


comptime _COLLECT_NONE = 0
comptime _COLLECT_RANGE = 1
comptime _COLLECT_ROWS = 2
comptime _COLLECT_VALUES = 3
# Values a step computed, as raw fixed-width bytes and a packed validity
# bitmap, appended a chunk at a time and made a column once.
comptime _COLLECT_RAW = 4


struct _Collector(Movable):
    """A morsel's chunks gathered into `_Columns` for a join step, which
    reads frames (the adapter the join port deletes): a
    column whose chunks read one arena column is kept as that column's
    range or row list, and only values a step computed are copied."""

    var schema: DataFrame
    var modes: List[Int]
    var handles: List[Int]
    var starts: List[Int]
    var rows: List[List[UInt32]]
    var parts: List[Series]
    var raw: List[List[UInt8]]
    var raw_bits: List[List[UInt8]]
    var raw_physical: List[Int]
    var height: Int
    # The most rows the morsel can give (its input's): raw values and
    # bitmaps are sized for it once rather than grown chunk by chunk.
    var capacity: Int

    def __init__(out self, schema: DataFrame, capacity: Int = 0):
        self.schema = schema.copy()
        self.capacity = capacity
        var width = schema.width()
        self.modes = List[Int](length=width, fill=_COLLECT_NONE)
        self.handles = List[Int](length=width, fill=-1)
        self.starts = List[Int](length=width, fill=0)
        self.rows = List[List[UInt32]](length=width, fill=List[UInt32]())
        self.parts = List[Series](capacity=width)
        for c in range(width):
            self.parts.append(schema._columns[c].copy())
        self.raw = List[List[UInt8]](length=width, fill=List[UInt8]())
        self.raw_bits = List[List[UInt8]](length=width, fill=List[UInt8]())
        self.raw_physical = List[Int](length=width, fill=0)
        self.height = 0

    def append(mut self, mut chunk: DataChunk, mut arena: ColumnArena) raises:
        var count = chunk.count
        for c in range(len(self.modes)):
            var v = chunk.data[c]
            var mode = self.modes[c]
            if (
                v.source >= 0
                and v.vector_type != SEQUENCE_VECTOR
                and (
                    mode == _COLLECT_NONE
                    or (mode != _COLLECT_VALUES and self.handles[c] == v.source)
                )
            ):
                if mode == _COLLECT_NONE:
                    self.handles[c] = v.source
                    if v.vector_type == FLAT_VECTOR:
                        self.modes[c] = _COLLECT_RANGE
                        self.starts[c] = v.source_offset
                        continue
                    self.modes[c] = _COLLECT_ROWS
                elif mode == _COLLECT_RANGE:
                    if v.vector_type == FLAT_VECTOR and (
                        v.source_offset == self.starts[c] + self.height
                    ):
                        continue
                    # The range so far as rows, then this chunk's.
                    var listed = List[UInt32](capacity=self.height + count)
                    for r in range(self.height):
                        listed.append(UInt32(self.starts[c] + r))
                    self.rows[c] = listed^
                    self.modes[c] = _COLLECT_ROWS
                _append_rows(self.rows[c], v, count)
                continue
            var size = physical_size(v.physical)
            var fixed = (
                v.source < 0
                and v.vector_type != SEQUENCE_VECTOR
                and v.physical != PHYSICAL_BOOL
                and v.physical != PHYSICAL_VARCHAR
                and size > 0
            )
            if fixed and (
                (mode == _COLLECT_NONE and self.height == 0)
                or (mode == _COLLECT_RAW and self.raw_physical[c] == v.physical)
            ):
                if mode == _COLLECT_NONE:
                    self.modes[c] = _COLLECT_RAW
                    self.raw_physical[c] = v.physical
                self._append_raw(c, chunk, arena, count, size)
                continue
            if mode == _COLLECT_RAW:
                self.parts[c] = self._raw_series(c)
                self.modes[c] = _COLLECT_VALUES
            elif mode != _COLLECT_VALUES:
                self.parts[c] = self._so_far(c, arena)
                self.modes[c] = _COLLECT_VALUES
            var piece = to_series(
                chunk,
                c,
                arena,
                self.schema._columns[c].name(),
                self.schema._columns[c].dtype(),
            )
            self.parts[c]._append_series(piece)
        self.height += count

    def _append_raw(
        mut self,
        c: Int,
        mut chunk: DataChunk,
        mut arena: ColumnArena,
        count: Int,
        size: Int,
    ) raises:
        append_raw(
            self.raw[c],
            self.raw_bits[c],
            self.height,
            self.capacity,
            chunk.unified(c, arena),
            count,
            size,
        )

    def _raw_series(self, c: Int) raises -> Series:
        """Column `c`'s raw values as a column of its schema type."""
        return raw_series(
            self.raw[c],
            self.raw_bits[c],
            self.raw_physical[c],
            self.schema._columns[c],
        )

    def _so_far(self, c: Int, arena: ColumnArena) raises -> Series:
        """Column `c`'s rows collected so far, as values."""
        var name = self.schema._columns[c].name()
        if self.modes[c] == _COLLECT_NONE or self.height == 0:
            return self.schema._columns[c].copy()
        ref column = arena.columns[self.handles[c]]
        var result: Series
        if self.modes[c] == _COLLECT_RANGE:
            result = column.slice(self.starts[c], self.height)
            # Owned, so later chunks append to it.
            if result.is_chunked():
                result = result.rechunk()
            var owned = self.schema._columns[c].copy()
            owned._append_series(result)
            return owned^
        var picked = List[Int](capacity=self.height)
        for r in self.rows[c]:
            picked.append(Int(r))
        result = column.take(picked).renamed(name)
        return result^

    def finish(mut self, mut arena: ColumnArena) raises -> _Columns:
        var result = _Columns()
        result.height = self.height
        for c in range(len(self.modes)):
            var mode = self.modes[c]
            if mode == _COLLECT_RANGE:
                result.handles.append(self.handles[c])
                result.starts.append(self.starts[c])
                result.lists.append(-1)
            elif mode == _COLLECT_RAW:
                result.handles.append(arena.add(self._raw_series(c)))
                result.starts.append(0)
                result.lists.append(-1)
            elif mode == _COLLECT_ROWS:
                result.handles.append(self.handles[c])
                result.starts.append(0)
                var listed = List[UInt32]()
                swap(listed, self.rows[c])
                # Columns a filter sliced alike share one row list (one
                # selection vector for the chunk, as DuckDB's slice).
                var shared = -1
                for l in range(len(result.row_lists)):
                    if len(result.row_lists[l]) == len(listed) and (
                        external_call["memcmp", Int32](
                            Int(result.row_lists[l].unsafe_ptr()),
                            Int(listed.unsafe_ptr()),
                            4 * len(listed),
                        )
                        == 0
                    ):
                        shared = l
                        break
                if shared >= 0:
                    result.lists.append(shared)
                else:
                    result.lists.append(len(result.row_lists))
                    result.row_lists.append(listed^)
            else:
                result.handles.append(arena.add(self.parts[c]))
                result.starts.append(0)
                result.lists.append(-1)
        return result^


struct _VectorStep(Copyable, Movable):
    """One row-local step compiled for its input columns, with its output
    chunk (DuckDB's operator state and output DataChunk), reused for
    every chunk a worker runs through it."""

    var kind: Int
    # FILTER: the predicate (ExpressionExecutor::Select; its AND / OR
    # nodes order their children with DuckDB's AdaptiveFilter); none for
    # a literal true. SELECT / WITH_COLUMNS: the computed columns.
    var expressions: List[VectorExpression]
    # SELECT / WITH_COLUMNS / DROP: per output column, the input column it
    # reads (>= 0) or the expression computing it (-1 - expression).
    var outputs: List[Int]
    var output: DataChunk

    def __init__(
        out self, step: Step, input: DataFrame, output: DataFrame
    ) raises:
        self.kind = step.kind
        self.expressions = List[VectorExpression]()
        self.outputs = List[Int]()
        self.output = DataChunk()
        var schema = input._columns.copy()
        if step.kind == STEP_FILTER:
            ref predicate = step.exprs[0]
            if (
                len(predicate._nodes) == 1
                and predicate._nodes[0].op == LIT_BOOL
                and predicate._nodes[0].integer != 0
            ):
                return
            var expanded = expand(predicate, schema)
            if len(expanded) != 1:
                raise Error("A filter selector must match exactly one column")
            trace_path("pipeline.vector_filter")
            var compiled = VectorExpression(expanded[0], schema)
            if compiled.root_dtype() != DataType.BOOL:
                raise Error("Filter expression must return Boolean values")
            self.expressions.append(compiled^)
            return
        trace_path("pipeline.vector_projection")
        if step.kind == STEP_DROP:
            for c in range(input.width()):
                if input._columns[c].name() not in step.names:
                    self.outputs.append(c)
            return
        var bound = _bind_all(step.exprs, schema)
        var produced = Dict[String, Int]()
        for b in bound:
            var compiled = VectorExpression(b.expr, schema)
            if compiled.root_kind() == _KIND_COLUMN:
                produced[b.expr._name] = compiled.root_column()
            else:
                produced[b.expr._name] = -1 - len(self.expressions)
                self.expressions.append(compiled^)
        for c in range(output.width()):
            var name = output._columns[c].name()
            var found = produced.get(name)
            if found:
                self.outputs.append(found.value())
            else:
                self.outputs.append(input._index(name))

    def run(mut self, mut input: DataChunk, mut arena: ColumnArena) raises:
        """This step over one chunk, into `output`."""
        self.output.reset()
        if self.kind == STEP_FILTER:
            var rows = SelectionVector()
            var count = input.count
            if len(self.expressions) > 0:
                rows = self.output.new_selection()
                count = self.expressions[0].select(
                    input, arena, SelectionVector(), count, rows
                )
            if count == input.count:
                self.output.set_columns(input.column_count())
                for c in range(input.column_count()):
                    self.output.data[c] = input.data[c]
                self.output.count = count
                return
            self.output.slice(input, rows, count)
            return
        self.output.set_columns(len(self.outputs))
        self.output.count = input.count
        for j in range(len(self.outputs)):
            var source = self.outputs[j]
            if source >= 0:
                self.output.data[j] = input.data[source]
            else:
                self.expressions[-1 - source].write(
                    input, arena, self.output, j
                )


def _step_schemas(
    frame: DataFrame, steps: List[Step], joins: List[DataFrame]
) raises -> List[DataFrame]:
    """The empty frame each step reads, then the steps' output: the
    eager operators over no rows decide every name and type."""
    var schemas = List[DataFrame](capacity=len(steps) + 1)
    var current = frame.clear()
    schemas.append(current.copy())
    for step in steps:
        if step.kind == STEP_FILTER:
            current = current.filter(step.exprs[0])
        elif step.kind == STEP_SELECT:
            current = current.select_exprs(step.exprs)
        elif step.kind == STEP_WITH_COLUMNS:
            current = current.with_columns(step.exprs)
        elif step.kind == STEP_JOIN:
            current = current._join_impl(
                joins[step.join].clear(),
                left_on=step.names,
                right_on=step.right_keys,
                how=step.how,
                suffix=step.suffix,
                coalesce=step.coalesce,
            )
        else:
            var kept = List[Series]()
            for column in current._columns:
                if column.name() not in step.names:
                    kept.append(column.copy())
            current = DataFrame(kept^, height=0)
        if current.height() != 0:
            current = current.clear()
        schemas.append(current.copy())
    return schemas^


def _compile_steps(
    steps: List[Step], schemas: List[DataFrame]
) raises -> List[Optional[_VectorStep]]:
    """Every row-local step compiled for its input columns."""
    var compiled = List[Optional[_VectorStep]](capacity=len(steps))
    for s in range(len(steps)):
        if steps[s].kind == STEP_JOIN:
            compiled.append(None)
        else:
            compiled.append(_VectorStep(steps[s], schemas[s], schemas[s + 1]))
    return compiled^


struct _ChunkRunner(Movable):
    """A worker's run of a pipeline's steps over data chunks of
    STANDARD_VECTOR_SIZE rows: each row-local step reads its input chunk
    through the unified vector format and writes its own output chunk;
    a join step, not yet on vectors, takes the morsel's chunks collected
    (`_Collector`) and gives its output back as columns the next steps
    cut chunks from. `counts` receives the rows fed, each step's input
    and output rows, and each step's busy nanoseconds."""

    var steps: List[Step]
    var schemas: ArcPointer[List[DataFrame]]
    var compiled: List[Optional[_VectorStep]]
    # The steps compiled once for every worker (bound, constants
    # evaluated); each worker copies its own (DuckDB's operator state is
    # per thread, its expressions bound once).
    var templates: ArcPointer[List[Optional[_VectorStep]]]
    var joins: ArcPointer[List[DataFrame]]
    var indexes: ArcPointer[List[Optional[JoinHashTable]]]
    var arena: ColumnArena
    # The arena's columns held for the whole pipeline (join builds); a
    # morsel's are dropped when the next one starts.
    var held: Int
    # Per join step, the arena handles of its build's columns.
    var builds: List[List[Int]]
    var source: DataChunk
    var counts: List[Int]
    # Whether to stop at a step that leaves no row (a pipeline's morsel),
    # or to run on so the output has every step's columns.
    var stop_empty: Bool
    # An ungrouped aggregate sink (`ungrouped.mojo`), when the pipeline
    # ends in one: the last step's chunks are sunk into its states, and
    # nothing is collected.
    var ungrouped: List[UngroupedSink]
    # The layout every worker's sink is made from, on its own thread
    # when its job starts.
    var ungrouped_layout: List[ArcPointer[UngroupedLayout]]
    # A grouped aggregate table sink (`table_sink.mojo`), likewise: the
    # last step's chunks go into the worker's table; its layout, and the
    # workers every table is sized for.
    var table: List[TableSink]
    var table_layout: List[ArcPointer[TableSinkLayout]]
    var table_workers: Int
    # Grouping sets' tables (`GroupingSetsSink`), likewise.
    var sets: List[GroupingSetsSink]
    var sets_layout: List[ArcPointer[GroupingSetsLayout]]
    # A reduction neither aggregate sink takes (`_ReduceSink`).
    var reduce: List[_ReduceSink]
    var reduce_layout: List[ArcPointer[_ReduceSinkLayout]]
    # A materialize sink (`MaterializeSink`), when the pipeline's rows are
    # its result.
    var materialize: List[MaterializeSink]
    var materializes: Bool
    # The source's table filters, and the source row of the morsel's first
    # row while the morsel's own rows run (-1 otherwise): a chunk whose
    # vector the filters' statistics rule out is not read (DuckDB's
    # `RowGroup::CheckZonemapSegments`).
    var scan: Optional[ScanFilterInfo]
    var scan_offset: Int

    def __init__(
        out self,
        steps: List[Step],
        schemas: ArcPointer[List[DataFrame]],
        joins: ArcPointer[List[DataFrame]],
        indexes: ArcPointer[List[Optional[JoinHashTable]]],
        stop_empty: Bool,
        templates: ArcPointer[List[Optional[_VectorStep]]] = ArcPointer(
            List[Optional[_VectorStep]]()
        ),
    ) raises:
        self.steps = steps.copy()
        self.schemas = schemas
        self.templates = templates
        if len(self.templates[]) != len(steps):
            self.templates = ArcPointer(
                List[Optional[_VectorStep]](length=len(steps), fill=None)
            )
        self.joins = joins
        self.indexes = indexes
        self.compiled = List[Optional[_VectorStep]](capacity=len(steps))
        self.arena = ColumnArena()
        self.builds = List[List[Int]](capacity=len(steps))
        for s in range(len(steps)):
            self.compiled.append(None)
            var handles = List[Int]()
            if steps[s].kind == STEP_JOIN:
                for column in joins[][steps[s].join]._columns:
                    handles.append(self.arena.add(column))
            self.builds.append(handles^)
        self.held = self.arena.mark()
        self.source = DataChunk()
        self.counts = List[Int](length=3 * len(steps) + 1, fill=0)
        self.stop_empty = stop_empty
        self.ungrouped = List[UngroupedSink]()
        self.ungrouped_layout = List[ArcPointer[UngroupedLayout]]()
        self.table = List[TableSink]()
        self.table_layout = List[ArcPointer[TableSinkLayout]]()
        self.table_workers = 1
        self.sets = List[GroupingSetsSink]()
        self.sets_layout = List[ArcPointer[GroupingSetsLayout]]()
        self.reduce = List[_ReduceSink]()
        self.reduce_layout = List[ArcPointer[_ReduceSinkLayout]]()
        self.materialize = List[MaterializeSink]()
        self.materializes = False
        self.scan = None
        self.scan_offset = -1

    def begin_morsel(mut self):
        """Drop the previous morsel's columns."""
        self.arena.truncate(self.held)

    def sinks_chunks(self) -> Bool:
        """Whether the worker's sink state is made (`begin_sink`): the
        last step's chunks go into it, and nothing is collected."""
        return (
            len(self.ungrouped) > 0
            or len(self.table) > 0
            or len(self.sets) > 0
            or len(self.reduce) > 0
            or len(self.materialize) > 0
        )

    def begin_sink(
        mut self, morsel: Int, offset: Int = 0, length: Int = 0
    ) raises:
        """Morsel `morsel` (source rows [offset, offset + length)) comes
        next: the worker's local sink state, made on its own thread at its
        first morsel (`GetLocalSinkState`), numbers its rows from the
        morsel's."""
        if len(self.ungrouped_layout) > 0:
            if len(self.ungrouped) == 0:
                self.ungrouped.append(UngroupedSink(self.ungrouped_layout[0][]))
            self.ungrouped[0].begin(morsel)
        elif len(self.table_layout) > 0:
            if len(self.table) == 0:
                self.table.append(
                    TableSink(self.table_layout[0][], self.table_workers)
                )
            self.table[0].begin(morsel)
        elif len(self.sets_layout) > 0:
            if len(self.sets) == 0:
                self.sets.append(
                    GroupingSetsSink(self.sets_layout[0][], self.table_workers)
                )
            self.sets[0].begin(morsel)
        elif len(self.reduce_layout) > 0:
            if len(self.reduce) == 0:
                self.reduce.append(_ReduceSink(self.reduce_layout[0][]))
            self.reduce[0].place(offset, length)
            self.reduce[0].begin(morsel)
        elif self.materializes:
            if len(self.materialize) == 0:
                self.materialize.append(
                    MaterializeSink(self.output_schema().clear())
                )
            self.materialize[0].begin(morsel)

    def release_sink(mut self) raises:
        """The worker's run is over: only its sink's states outlive it. A
        grouping sets sink derives its coarser sets here, on the worker's
        thread."""
        if len(self.ungrouped) > 0:
            self.ungrouped[0].release()
        if len(self.table) > 0:
            self.table[0].release()
        if len(self.sets) > 0:
            self.sets[0].release()
            self.sets[0].derive()
        if len(self.reduce) > 0:
            self.reduce[0].release()
        if len(self.materialize) > 0:
            self.materialize[0].release()

    def _sink(mut self, mut chunk: DataChunk) raises:
        """One output chunk into the worker's sink."""
        if len(self.ungrouped) > 0:
            _sink_chunk(self.ungrouped[0], chunk, self.arena)
        elif len(self.table) > 0:
            _sink_chunk(self.table[0], chunk, self.arena)
        elif len(self.sets) > 0:
            _sink_chunk(self.sets[0], chunk, self.arena)
        elif len(self.reduce) > 0:
            _sink_chunk(self.reduce[0], chunk, self.arena)
        else:
            _sink_chunk(self.materialize[0], chunk, self.arena)

    def schema(self, at: Int) -> DataFrame:
        return self.schemas[][at].copy()

    def output_schema(self) -> DataFrame:
        return self.schemas[][len(self.steps)].copy()

    def run(
        mut self, var columns: _Columns, first: Int, marks: List[Int]
    ) raises -> _Columns:
        """The steps from `first` on over `columns` (the input of step
        `first`); `marks` is per step the address of a right or full
        join's matched-row flags (0 for none)."""
        return self.run_until(columns^, first, len(self.steps), marks)

    def run_to_last(
        mut self, var columns: _Columns, first: Int, marks: List[Int]
    ) raises -> Tuple[_Columns, Int]:
        """The steps from `first` up to the last row-local segment: the
        columns that segment reads and the step it starts at (the step
        count when the last step is a join). The segment then runs one
        chunk at a time through `chunk`, so a sink takes each chunk as it
        is made (DuckDB's PipelineExecutor pushing a chunk to the sink)."""
        var n = len(self.steps)
        var last = n
        while last > first and self.steps[last - 1].kind != STEP_JOIN:
            last -= 1
        if last > first:
            columns = self.run_until(columns^, first, last, marks)
        for s in range(last, n):
            self._compiled(s)
        return (columns^, last)

    def run_until(
        mut self,
        var columns: _Columns,
        first: Int,
        stop: Int,
        marks: List[Int],
    ) raises -> _Columns:
        """Steps [first, stop) over `columns`, collected."""
        var at = first
        var n = len(self.steps)
        while at < stop:
            if self.stop_empty and columns.height == 0:
                return columns^
            if self.steps[at].kind == STEP_JOIN:
                self.counts[2 * at + 1] += columns.height
                var began = Int(perf_counter_ns())
                columns = self._join(columns, at, marks[at])
                self.counts[2 * n + 1 + at] += Int(perf_counter_ns()) - began
                self.counts[2 * at + 2] += columns.height
                at += 1
                continue
            var end = at
            while end < stop and self.steps[end].kind != STEP_JOIN:
                end += 1
            columns = self._vector_steps(columns, at, end)
            at = end
        if stop == n and self.sinks_chunks() and columns.height > 0:
            if len(self.ungrouped) > 0 and self.ungrouped[0].counts_only():
                # Row counts read no column: the rows are only counted.
                self.ungrouped[0].sink_rows(columns.height)
                return _Columns()
            # A join's output (or the source, with no steps) cut into
            # chunks for the sink.
            columns = self._vector_steps(columns, n, n)
        return columns^

    def chunk(
        mut self,
        columns: _Columns,
        bases: List[Vector],
        low: Int,
        count: Int,
        first: Int,
    ) raises -> Int:
        """Rows [low, low + count) of `columns` through steps [first, end)
        (all row-local, compiled by `run_to_last`): the address of the
        output chunk, or 0 when no row is left."""
        var n = len(self.steps)
        var compiled = Pointer[Optional[_VectorStep], MutAnyOrigin](
            unsafe_from_address=Int(self.compiled.unsafe_ptr())
        )
        var input = Pointer[DataChunk, MutAnyOrigin](
            unsafe_from_address=Int(Pointer(to=self.source))
        )
        columns.fill(bases, self.arena, input[], low, count)
        for s in range(first, n):
            self.counts[2 * s + 1] += input[].count
            var began = Int(perf_counter_ns())
            ref step = compiled[unsafe_offset=s].value()
            step.run(input[], self.arena)
            self.counts[2 * n + 1 + s] += Int(perf_counter_ns()) - began
            self.counts[2 * s + 2] += step.output.count
            input = Pointer[DataChunk, MutAnyOrigin](
                unsafe_from_address=Int(Pointer(to=step.output))
            )
            if step.output.count == 0:
                return 0
        return Int(input)

    def chunk_frame(mut self, address: Int) raises -> DataFrame:
        """An output chunk as a frame of the output schema (the adapter
        for sinks that read frames)."""
        var chunk = Pointer[DataChunk, MutAnyOrigin](
            unsafe_from_address=address
        )
        var schema = self.output_schema()
        var columns = List[Series](capacity=schema.width())
        for c in range(schema.width()):
            columns.append(
                to_series(
                    chunk[],
                    c,
                    self.arena,
                    schema._columns[c].name(),
                    schema._columns[c].dtype(),
                )
            )
        return DataFrame(columns^)

    def _compiled(mut self, s: Int) raises:
        if not self.compiled[s]:
            if self.templates[][s]:
                self.compiled[s] = self.templates[][s].value().copy()
            else:
                self.compiled[s] = _VectorStep(
                    self.steps[s], self.schemas[][s], self.schemas[][s + 1]
                )

    def _vector_steps(
        mut self, columns: _Columns, first: Int, stop: Int
    ) raises -> _Columns:
        """Steps [first, stop), all row-local, chunk by chunk."""
        for s in range(first, stop):
            self._compiled(s)
        var n = len(self.steps)
        var sinking = stop == n and self.sinks_chunks()
        # Chunks for a later join step are collected; a sink takes its own.
        var collector = List[_Collector]()
        if not sinking:
            collector.append(_Collector(self.schemas[][stop], columns.height))
        var bases = columns.bases(self.arena)
        var compiled = Pointer[Optional[_VectorStep], MutAnyOrigin](
            unsafe_from_address=Int(self.compiled.unsafe_ptr())
        )
        var source = Pointer[DataChunk, MutAnyOrigin](
            unsafe_from_address=Int(Pointer(to=self.source))
        )
        var skips = first == 0 and self.scan_offset >= 0 and Bool(self.scan)
        for low in range(0, columns.height, STANDARD_VECTOR_SIZE):
            var count = min(STANDARD_VECTOR_SIZE, columns.height - low)
            if skips:
                var at = self.scan_offset + low
                if self.scan.value().skip(at, at + count):
                    trace_path("scan.vector_skip")
                    continue
            columns.fill(bases, self.arena, source[], low, count)
            var input = source
            var empty = False
            for s in range(first, stop):
                self.counts[2 * s + 1] += input[].count
                var began = Int(perf_counter_ns())
                ref step = compiled[unsafe_offset=s].value()
                step.run(input[], self.arena)
                self.counts[2 * n + 1 + s] += Int(perf_counter_ns()) - began
                self.counts[2 * s + 2] += step.output.count
                input = Pointer[DataChunk, MutAnyOrigin](
                    unsafe_from_address=Int(Pointer(to=step.output))
                )
                if step.output.count == 0:
                    empty = True
                    break
            if empty:
                continue
            if sinking:
                self._sink(input[])
            else:
                collector[0].append(input[], self.arena)
        if sinking:
            return _Columns()
        return collector[0].finish(self.arena)

    def _join(
        mut self, columns: _Columns, s: Int, marks: Int
    ) raises -> _Columns:
        """A join step over a morsel's collected columns. An inner, semi or
        anti join probes its table (`ScanStructure`) with the key columns
        only and keeps every other column a dictionary over where it is:
        the probe side through the matched probe rows, the build side
        through the matched build rows (no payload is gathered, as
        DuckDB's probe slices the probe chunk and emits build columns as
        dictionary vectors, `EmitDictVectors`). Any other join probes the
        morsel's frame; a right or full join flags the build rows it
        matched in the table (`marks` is their address), for the rows no
        morsel matched to come out after the probe."""
        ref step = self.steps[s]
        ref input = self.schemas[][s]
        ref output = self.schemas[][s + 1]
        ref right = self.joins[][step.join]
        var dictionary = (
            step.how == JOIN_INNER
            or step.how == JOIN_SEMI
            or step.how == JOIN_ANTI
        ) and Bool(self.indexes[][step.join])
        if dictionary:
            for i in range(len(step.names)):
                var lt = input.column(step.names[i]).dtype()
                var rt = right.column(step.right_keys[i]).dtype()
                if lt != rt or lt.is_categorical() or lt.is_nested():
                    dictionary = False
        if not dictionary:
            var frame = columns.whole(self.arena, input)
            var joined = frame._join_impl(
                right,
                left_on=step.names,
                right_on=step.right_keys,
                how=step.how,
                suffix=step.suffix,
                coalesce=step.coalesce,
                prepared=self.indexes[][step.join],
                workers=1,
            )
            return _Columns.of_frame(self.arena, joined)
        trace_path("pipeline.join_dictionary")
        var keys = columns.frame(self.arena, input, step.names)._columns.copy()
        for k in range(len(keys)):
            if keys[k].is_chunked():
                keys[k] = keys[k].rechunk()
        var scan = ScanStructure(self.indexes[][step.join].value(), step.how)
        var probe_rows = List[Int](capacity=columns.height)
        var build_rows = List[UInt32]()
        var inner = step.how == JOIN_INNER
        if inner:
            build_rows.reserve(columns.height)
        for low in range(0, columns.height, STANDARD_VECTOR_SIZE):
            var count = min(STANDARD_VECTOR_SIZE, columns.height - low)
            scan.probe(keys, low, count)
            while True:
                var n = scan.next()
                if n == 0:
                    break
                var rows = scan.probe_rows.unsafe_ptr()
                var builds = scan.build_rows.unsafe_ptr()
                for k in range(n):
                    probe_rows.append(low + Int(rows[unsafe_offset=k]))
                if inner:
                    for k in range(n):
                        build_rows.append(UInt32(builds[unsafe_offset=k]))
        if not inner:
            if len(probe_rows) == columns.height:
                return columns.copy()
            return columns.through(probe_rows)
        var identity = len(probe_rows) == columns.height
        if identity:
            var at = probe_rows.unsafe_ptr()
            for k in range(len(probe_rows)):
                if at[unsafe_offset=k] != k:
                    identity = False
                    break
        var result: _Columns
        if identity:
            result = columns.copy()
        else:
            result = columns.through(probe_rows)
        result.height = len(build_rows)
        # The build side's output columns, in the join's output order.
        var list = len(result.row_lists)
        result.row_lists.append(build_rows^)
        var right_keys = List[Int]()
        for key in step.right_keys:
            right_keys.append(right._index(key))
        for i in range(right.width()):
            if step.coalesce and i in right_keys:
                continue
            result.handles.append(self.builds[s][i])
            result.starts.append(0)
            result.lists.append(list)
        if result.width() != output.width():
            raise Error("pipeline join: output columns differ from its schema")
        return result^


def _sink_chunk[
    S: Sink
](mut sink: S, mut chunk: DataChunk, mut arena: ColumnArena) raises:
    """One chunk into a worker's local sink state (DuckDB's `Sink`)."""
    sink.sink(chunk, arena)


def _combined[S: Sink](var locals: List[S]) raises -> List[S]:
    """Every worker's local sink state combined into the global state
    (DuckDB's `Combine`), in worker order."""
    var combined = List[S]()
    while len(locals) > 0:
        locals.pop(0).combine(combined)
    return combined^


def _steps_over_frame(
    frame: DataFrame,
    steps: List[Step],
    joins: ArcPointer[List[DataFrame]],
    indexes: ArcPointer[List[Optional[JoinHashTable]]],
    mut counts: List[Int],
) raises -> DataFrame:
    """`steps` over all of `frame` on this thread, chunk by chunk; `counts`
    receives each step's input and output rows."""
    var schemas = ArcPointer(_step_schemas(frame, steps, joins[]))
    var runner = _ChunkRunner(steps, schemas, joins, indexes, False)
    runner.materializes = True
    runner.begin_sink(0)
    var source = _Columns.of_frame(runner.arena, frame)
    var marks = List[Int](length=len(steps), fill=0)
    _ = runner.run(source^, 0, marks)
    runner.release_sink()
    for i in range(len(steps)):
        counts.append(runner.counts[2 * i + 1])
        counts.append(runner.counts[2 * i + 2])
    return materialize_finish(
        runner.materialize, runner.output_schema().clear()
    )


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


struct _ReduceSinkLayout(Copyable, Movable):
    """What every worker's reduce sink shares (a grouped reduction the
    aggregate table does not take, or an ungrouped one the ungrouped sink
    does not): the sink input's types, the reductions and group keys,
    whether groups are inserted or collected, whether every step only
    filters, and the input columns read."""

    var schema: DataFrame
    var expressions: List[Expr]
    var keys: List[String]
    var insert: Bool
    var filter_only: Bool
    var batch_size: Int
    # None: a selector may read any column, so every one is read.
    var reads: Optional[List[String]]

    def __init__(
        out self,
        schema: DataFrame,
        expressions: List[Expr],
        keys: List[String],
        insert: Bool,
        filter_only: Bool,
        batch_size: Int,
    ):
        self.schema = schema.copy()
        self.expressions = expressions.copy()
        self.keys = keys.copy()
        self.insert = insert
        self.filter_only = filter_only
        self.batch_size = batch_size
        self.reads = _reads(expressions)

    def selects(self) -> Bool:
        """Whether the sink keeps each morsel's selection over the source
        rather than its rows (a collected grouping of a pipeline that only
        filters: its rows are gathered once at the finish)."""
        return (
            len(self.keys) > 0
            and len(self.expressions) > 0
            and not self.insert
            and self.filter_only
        )


struct _ReduceSink(Sink):
    """One worker's reduce sink state, fed a chunk at a time: its grouped
    sink (`_GroupedSink`) while the keys partition, else its reducer
    states (ungrouped: one state; grouped: one per hash part, each chunk's
    state merged into them). A grouped sink of a filter-only pipeline
    keeps the source rows each morsel's chunks select."""

    var shared: _ReduceSinkLayout
    var grouped: List[_GroupedSink]
    var reduction: List[_StreamReduction]
    var sequence: Int
    # The morsel the chunks come from, as its source offset and length,
    # and the morsel rows they selected so far.
    var offset: Int
    var length: Int
    var selected: List[UInt32]

    def __init__(out self, shared: _ReduceSinkLayout):
        self.shared = shared.copy()
        self.grouped = List[_GroupedSink]()
        self.reduction = List[_StreamReduction]()
        self.sequence = 0
        self.offset = 0
        self.length = 0
        self.selected = List[UInt32]()

    def place(mut self, offset: Int, length: Int) raises:
        """The next morsel's rows are source rows [offset, offset +
        length)."""
        self._keep_selection()
        self.offset = offset
        self.length = length

    def _keep_selection(mut self) raises:
        """The morsel's selected rows as its mask over the source (empty:
        every row)."""
        if len(self.selected) == 0:
            return
        if len(self.grouped) == 0:
            self.grouped.append(
                _GroupedSink(
                    self.shared.schema,
                    self.shared.expressions,
                    self.shared.keys,
                    True,
                    self.shared.batch_size,
                )
            )
        var mask = BoolColumn(List[Bool]())
        if len(self.selected) != self.length:
            var rows = List[Int](capacity=len(self.selected))
            for r in self.selected:
                rows.append(Int(r))
            mask = _rows_mask(rows, self.length)
        self.grouped[0].select(self.offset, self.length, mask)
        self.selected = List[UInt32]()

    def begin(mut self, morsel: Int):
        self.sequence = morsel << 32

    def sink(mut self, mut chunk: DataChunk, mut arena: ColumnArena) raises:
        var n = chunk.count
        if n == 0:
            return
        if self.shared.selects():
            var v = chunk.data[0]
            if v.source < 0 or (
                v.vector_type != FLAT_VECTOR
                and v.vector_type != DICTIONARY_VECTOR
            ):
                raise Error("pipeline: a filter's chunk lost its source rows")
            _append_rows(self.selected, v, n)
            self.sequence += n
            return
        ref schema = self.shared.schema
        var expressions = self.shared.expressions.copy()
        var input: DataFrame
        if self.shared.reads and len(self.shared.keys) == 0:
            # A count of a column holding no null is the row count, and a
            # column only row counts read is not gathered.
            var null_free = List[Bool](capacity=schema.width())
            for c in range(schema.width()):
                var source = chunk.data[c].source
                null_free.append(
                    source >= 0 and arena.columns[source].null_count() == 0
                )
            var counts = _row_counted(schema, expressions, null_free)
            expressions = counts[0].copy()
            input = _chunk_frame(chunk, arena, schema, counts[1])
            for name in counts[2]:
                # Only row counts read it: a column of the chunk's height
                # stands in, and its values are never read.
                input = input.with_column(
                    Series(name, Column[Int8](List[Int8](length=n, fill=0)))
                )
        elif self.shared.reads:
            var names = self.shared.reads.value().copy()
            for key in self.shared.keys:
                if key not in names:
                    names.append(key)
            input = _chunk_frame(chunk, arena, schema, names)
        else:
            input = _chunk_frame(chunk, arena, schema, schema.columns())
        var first = self.sequence
        self.sequence += n
        if len(self.shared.keys) > 0 and len(self.reduction) == 0:
            # Rows go straight into this worker's per-part key indexes and
            # reducer states (DuckDB's thread-local partitioned aggregate
            # table), when the keys encode as words; otherwise each chunk's
            # state is split into parts and merged below.
            if len(self.grouped) == 0:
                self.grouped.append(
                    _GroupedSink(
                        input,
                        self.shared.expressions,
                        self.shared.keys,
                        not self.shared.insert,
                        self.shared.batch_size,
                    )
                )
            if self.grouped[0].absorb(input, first):
                return
            if self.grouped[0].count > 0:
                raise Error("grouped pipeline: keys stopped partitioning")
            self.grouped = List[_GroupedSink]()
        # Group first rows are numbered by morsel, so first-occurrence
        # order survives morsels taken out of order.
        var state = _StreamReduction(input, expressions, self.shared.keys)
        state.shift_firsts(first)
        if len(self.shared.keys) == 0:
            if len(self.reduction) == 0:
                self.reduction.append(state^)
            else:
                self.reduction[0].merge(state)
        elif len(self.reduction) == 0:
            # The worker's state is kept in hash parts, so every chunk's
            # groups go into tables small enough to stay in cache.
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

    def release(mut self) raises:
        self._keep_selection()

    def combine(var self, mut combined: List[_ReduceSink]) raises:
        """This worker's state into the global state, every worker's: the
        finish merges them (by hash part when grouped)."""
        combined.append(self^)


def _chunk_frame(
    mut chunk: DataChunk,
    mut arena: ColumnArena,
    schema: DataFrame,
    names: List[String],
) raises -> DataFrame:
    """The named columns of a chunk of `schema`'s columns as a frame (the
    adapter for sinks that read frames)."""
    var columns = List[Series](capacity=len(names))
    for name in names:
        var c = schema._index(name)
        columns.append(
            to_series(chunk, c, arena, name, schema._columns[c].dtype())
        )
    return DataFrame(columns^, height=chunk.count)


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
    var indexes: ArcPointer[List[Optional[JoinHashTable]]]
    # Whether the pipeline ends in a reduction (no top-N then).
    var reduces: Bool
    # Top-N sink (PhysicalTopN): this worker's heap over the output's
    # sort keys (made at its first chunk), the boundary every worker's
    # full heap shares, and whether ties keep the input order (the
    # source place is then the last key).
    var top: Int
    var top_names: List[String]
    var top_descending: List[Bool]
    var top_nulls_last: List[Bool]
    var top_maintain_order: Bool
    var topn: List[TopNSink]
    var boundary: TopNBoundary
    # The scan's dynamic filter on the first sort column (DuckDB's TopN
    # filter pushdown), when that column is the source's own.
    var top_filter: _TopScanFilter
    # Rows fed, then each step's input and output rows, for the report.
    var counts: List[Int]
    # The join builds' filters on source columns (`_join_filters`): row
    # filters, with this worker's selectivity of each, and the statistics
    # filters with their answers per vector.
    var join_filters: List[JoinFilter]
    var join_stats: List[SelectivityStats]
    var join_scan: Optional[ArcPointer[ScanFilterInfo]]
    # Per step, the address of a right or full join's flags of the build
    # rows any morsel matched (one byte a build row, in the step's shared
    # table; `JoinHashTable.found_address`), 0 for other steps.
    var marks: List[Int]
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
    # The steps over data chunks, with this worker's vectors and arena.
    var runner: _ChunkRunner

    def __init__(
        out self,
        frame: ArcPointer[DataFrame],
        cursor: Int,
        ranges: ArcPointer[List[Int]],
        steps: List[Step],
        joins: ArcPointer[List[DataFrame]],
        indexes: ArcPointer[List[Optional[JoinHashTable]]],
        reduces: Bool,
        top: Int,
        top_names: List[String],
        top_descending: List[Bool],
        top_nulls_last: List[Bool],
        top_maintain_order: Bool,
        boundary: TopNBoundary,
        top_filter: _TopScanFilter,
        join_filters: List[JoinFilter],
        join_scan: Optional[ArcPointer[ScanFilterInfo]],
        schemas: ArcPointer[List[DataFrame]],
        templates: ArcPointer[List[Optional[_VectorStep]]],
    ) raises:
        self.runner = _ChunkRunner(
            steps, schemas, joins, indexes, True, templates
        )
        self.join_filters = join_filters.copy()
        self.join_stats = List[SelectivityStats](
            length=len(join_filters),
            fill=SelectivityStats.of(BF_CHECK_N, BF_THRESHOLD),
        )
        self.join_scan = join_scan
        # A right or full join flags the build rows it matched in its
        # shared table, as DuckDB flags them in the build's rows.
        self.marks = List[Int](capacity=len(steps))
        for step in steps:
            if (
                step.kind == STEP_JOIN
                and (step.how == JOIN_RIGHT or step.how == JOIN_FULL)
                and indexes[][step.join]
            ):
                self.marks.append(indexes[][step.join].value().found_address())
            else:
                self.marks.append(0)
        self.first_step = 0
        self.first_morsel = 0
        self.finishes = True
        self.unmatched_step = -1
        self.unmatched_marks = ArcPointer(List[Int]())
        self.unmatched_probe = frame.copy()
        self.boundary = boundary
        self.top_filter = top_filter.copy()
        self.top_maintain_order = top_maintain_order
        self.topn = List[TopNSink]()
        self.top = top
        self.top_names = top_names.copy()
        self.top_descending = top_descending.copy()
        self.top_nulls_last = top_nulls_last.copy()
        self.frame = frame.copy()
        self.cursor = cursor
        self.ranges = ranges.copy()
        self.steps = steps.copy()
        self.joins = joins.copy()
        self.indexes = indexes.copy()
        self.reduces = reduces
        # Rows fed, then each step's rows in and out, then each step's
        # busy nanoseconds on this worker.
        self.counts = List[Int](length=3 * len(steps) + 1, fill=0)

    def _join_rows(
        mut self, frame: DataFrame, offset: Int, length: Int
    ) raises -> Tuple[Bool, Optional[List[Int]]]:
        """The rows of a fresh morsel (`frame`, at source row `offset`) the
        join filters keep: False when none is, else their rows (None:
        every row). Vectors the statistics filters rule out are not read
        (CheckZonemapSegments); the row filters run vector by vector, a
        bloom filter only while its selectivity keeps it active."""
        for f in self.join_filters:
            if not f.any:
                return (False, None)
        var vectors = Optional[List[Int]]()
        if self.join_scan:
            vectors = _vectors_kept(self.join_scan.value()[], offset, length)
            if vectors and len(vectors.value()) == 0:
                trace_path("pipeline.join_filter_skip")
                return (False, None)
        var rowwise = False
        for f in self.join_filters:
            rowwise = rowwise or f.equal or Bool(f.bloom)
        if not rowwise:
            if not vectors:
                return (True, None)
            return (True, _rows_in_vectors(None, vectors.value(), length))
        var keys = List[Series](capacity=len(self.join_filters))
        for f in self.join_filters:
            if f.equal or f.bloom:
                keys.append(_unchunked(frame.column(f.column)))
            else:
                keys.append(Series("", Column[Int64](List[Int64]())))
        var starts = List[Int]()
        if vectors:
            starts = vectors.value().copy()
        else:
            for at in range(0, length, STATISTICS_VECTOR_SIZE):
                starts.append(at)
        var rows = List[Int]()
        var keep = List[UInt8](length=STATISTICS_VECTOR_SIZE, fill=1)
        for start in starts:
            var end = min(length, start + STATISTICS_VECTOR_SIZE)
            var at = keep.unsafe_ptr()
            for i in range(end - start):
                at[unsafe_offset=i] = 1
            for k in range(len(self.join_filters)):
                ref f = self.join_filters[k]
                if f.equal:
                    _ = keep_rows(f, keys[k], start, end, keep)
                elif f.bloom:
                    if not self.join_stats[k].active:
                        self.join_stats[k].update(0, 0)
                        continue
                    var counted = keep_rows(f, keys[k], start, end, keep)
                    self.join_stats[k].update(counted[1], counted[0])
            for i in range(end - start):
                if at[unsafe_offset=i] != 0:
                    rows.append(start + i)
        if len(rows) == length:
            return (True, None)
        return (True, rows^)

    def _marks(self) -> List[Int]:
        """Per step, the address of a right or full join's matched-row
        flags (0 for none)."""
        return self.marks.copy()

    def run(mut self) raises:
        var count = len(self.ranges[]) // 2
        var cursor = _Cursor(self.cursor)
        var marks = self._marks()
        while True:
            var taken = cursor.take(1)
            if taken >= count:
                self.counts = self.runner.counts.copy()
                if not self.finishes:
                    return
                self.runner.release_sink()
                return
            var k = self.first_morsel + taken
            var offset = self.ranges[][2 * taken]
            var length = self.ranges[][2 * taken + 1]
            self.runner.begin_sink(k, offset, length)
            self.runner.begin_morsel()
            if self.unmatched_step >= 0:
                var rows = self._unmatched(offset, length)
                if rows.height() == 0:
                    continue
                var tail = _Columns.of_frame(self.runner.arena, rows)
                self._sink_from(tail^, k, offset, marks)
                continue
            var morsel = self.frame[].slice(offset, length)
            if self.top_filter.row_ids:
                morsel = morsel.with_column(_row_ids(offset, length))
            if self.first_step > 0:
                # An outer join's unmatched build rows: the source's key
                # ranges and top-k bound do not apply to them.
                var source = _Columns.of_frame(self.runner.arena, morsel)
                self._sink_from(source^, k, offset, marks)
                continue
            self.runner.counts[0] += length
            var vectors = Optional[List[Int]]()
            # A top-N taking chunks checks each vector as it is read.
            if self.top_filter.info and not self._chunk_top():
                # The TopN's dynamic filter against the scan's vector
                # statistics (CheckZonemapSegments): vectors it rules out
                # are not read.
                vectors = _vectors_kept(
                    self.top_filter.info.value()[], offset, length
                )
                if vectors and len(vectors.value()) == 0:
                    trace_path("pipeline.top_dynamic_skip")
                    continue
            var keyed = self._join_rows(morsel, offset, length)
            if not keyed[0]:
                continue
            var rows = keyed[1].copy()
            if vectors:
                rows = _rows_in_vectors(rows, vectors.value(), length)
            var source = _Columns.of_frame(self.runner.arena, morsel)
            if rows:
                if len(rows.value()) == 0:
                    continue
                source = source.through(rows.value())
            else:
                self.runner.scan_offset = offset
            if self._chunk_top():
                self._top_chunks(source^, rows, k, offset, marks, 0)
                self.runner.scan_offset = -1
                continue
            _ = self.runner.run(source^, 0, marks)
            self.runner.scan_offset = -1

    def _sink_from(
        mut self, var columns: _Columns, k: Int, offset: Int, marks: List[Int]
    ) raises:
        """An outer join's unmatched build rows (or a morsel of them past
        its step) through the steps after `first_step` into the sink."""
        if self._chunk_top():
            self._top_chunks(columns^, None, k, offset, marks, self.first_step)
        else:
            _ = self.runner.run(columns^, self.first_step, marks)

    def _chunk_top(self) -> Bool:
        """Whether the sink is a top-N, taking chunks one at a time."""
        return self.top > 0 and not self.reduces

    def _top_chunks(
        mut self,
        var source: _Columns,
        rows: Optional[List[Int]],
        k: Int,
        offset: Int,
        marks: List[Int],
        first_step: Int,
    ) raises:
        """A morsel into the top-N one chunk at a time, as DuckDB's
        PhysicalTopN sinks each chunk the scan makes, from step
        `first_step` on. Before each vector is read, the scan checks the
        dynamic filter again (the OptionalFilter on the TopN's boundary in
        CheckZonemap), so a boundary the chunks before tightened skips the
        vectors it rules out."""
        var ran = self.runner.run_to_last(source^, first_step, marks)
        var columns = ran[0].copy()
        var first = ran[1]
        # The vectors map to source rows only before any join.
        var direct = first == 0
        var bases = columns.bases(self.runner.arena)
        var made = 0
        var skipped = 0
        if len(self.topn) > 0:
            self.topn[0].begin(k)
        for low in range(0, columns.height, STANDARD_VECTOR_SIZE):
            var count = min(STANDARD_VECTOR_SIZE, columns.height - low)
            if direct and (self.top_filter.info or self.runner.scan):
                var start = low
                var end = low + count
                if rows:
                    start = rows.value()[low]
                    end = rows.value()[low + count - 1] + 1
                if self.runner.scan and self.runner.scan.value().skip(
                    offset + start, offset + end
                ):
                    trace_path("scan.vector_skip")
                    continue
                if self.top_filter.info and self.top_filter.info.value()[].skip(
                    offset + start, offset + end
                ):
                    skipped += 1
                    continue
            var address = self.runner.chunk(columns, bases, low, count, first)
            if address == 0:
                continue
            var chunk = Pointer[DataChunk, MutAnyOrigin](
                unsafe_from_address=address
            )
            if self.top_maintain_order:
                # Ties keep the input order: the order column is added to
                # the chunk's rows as a frame.
                var frame = self.runner.chunk_frame(address)
                var height = frame.height()
                self._sink_top(frame^, k, offset, made)
                made += height
                continue
            if len(self.topn) == 0:
                self._sink_top(self.runner.output_schema(), k, offset, 0)
            _sink_chunk(self.topn[0], chunk[], self.runner.arena)
        if skipped > 0:
            # Once a morsel: tracing reads the environment.
            trace_path("pipeline.top_dynamic_skip_vector")

    def _sink_top(
        mut self, var output: DataFrame, k: Int, offset: Int, first_row: Int
    ) raises:
        """Rows into this worker's top-N heap; `first_row` numbers the
        first among the morsel's output rows (ties keep that order)."""
        if self.top_maintain_order:
            var h = output.height()
            var places = List[Int64](capacity=h)
            var base = Int64(k) << 40
            if self.top_filter.reordered:
                # A row group's rows (at most ROW_GROUP_ROWS, no
                # join) after its source row.
                base = Int64(offset) << 20
            for r in range(h):
                places.append(base + Int64(first_row + r))
            output = output.with_column(
                Series(_ORDER_COLUMN, Column[Int64](places^))
            )
        if len(self.topn) == 0:
            self.topn.append(
                TopNSink(
                    output,
                    _top_keys(self.top_names, self.top_maintain_order),
                    _top_flags(self.top_descending, self.top_maintain_order),
                    _top_flags(self.top_nulls_last, self.top_maintain_order),
                    self.top,
                    self.boundary,
                    self.top_filter.dynamic,
                    self.top_filter.op,
                )
            )
        self.topn[0].sink(output)

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


def _unchunked(column: Series) raises -> Series:
    """A column as one buffer: a morsel's slice within one chunk is that
    chunk's view, not a copy; only a slice across chunks is copied."""
    if not column.is_chunked():
        return column.copy()
    if column.n_chunks() == 1:
        return column.chunks()[0].copy()
    return column.rechunk()


def _top_keys(names: List[String], maintain_order: Bool) -> List[String]:
    """A top-N's keys: the sort's, then the source place when ties keep
    input order."""
    var out = names.copy()
    if maintain_order:
        out.append(_ORDER_COLUMN)
    return out^


def _top_flags(flags: List[Bool], maintain_order: Bool) -> List[Bool]:
    var out = flags.copy()
    if maintain_order:
        out.append(False)
    return out^


@fieldwise_init
struct _TopScanFilter(Copyable, Movable):
    """A TopN's dynamic filter on its first sort column, pushed into the
    scan (`TopN::PushdownDynamicFilters`): `info` checks it against the
    source's statistics, `dynamic` is the shared constant the sinks set,
    `op` its comparison."""

    var info: Optional[ArcPointer[ScanFilterInfo]]
    var dynamic: Optional[ArcPointer[DynamicFilterData]]
    var op: Int
    # The source is read in its row groups' statistics order: rows are
    # placed by their source row, not by the morsel's turn.
    var reordered: Bool
    # The scan adds each row's row id (late materialization).
    var row_ids: Bool


def _top_scan_filter(
    frame: DataFrame,
    top: Int,
    names: List[String],
    descending: List[Bool],
    nulls_last: List[Bool],
    scan_key: Bool,
    maintain_order: Bool,
) raises -> _TopScanFilter:
    """The dynamic filter for a TopN whose first sort column is read from
    the source unchanged (`scan_key`): `c < boundary` ascending and
    `c > boundary` descending, `<=` / `>=` with later keys to settle ties.
    DuckDB ORs `IS NULL` for nulls first; a statistics check cannot keep
    a vector's nulls apart, so a nulls-first key pushes nothing. Columns
    whose order is not their storage's (categoricals) push nothing."""
    var none = _TopScanFilter(None, None, 0, False, False)
    if top <= 0 or not scan_key or len(names) == 0 or not nulls_last[0]:
        return none^
    if names[0] not in frame.columns():
        return none^
    var dtype = frame.column(names[0]).dtype()
    if column_domain(dtype) < 0:
        return none^
    var op: Int
    # `maintain_order` makes the source place a last key: a row tied on
    # the first key may still enter, so the comparison keeps ties.
    if len(names) == 1 and not maintain_order:
        op = COMPARE_GREATERTHAN if descending[0] else COMPARE_LESSTHAN
    else:
        op = COMPARE_GREATERTHANOREQUALTO if descending[
            0
        ] else COMPARE_LESSTHANOREQUALTO
    var dynamic = ArcPointer(DynamicFilterData(dtype, op))
    var filters = TableFilterSet()
    filters.push_dynamic(names[0], dynamic)
    var info = ScanFilterInfo(filters, [frame.column(names[0])])
    return _TopScanFilter(ArcPointer(info^), dynamic, op, False, False)


def _row_ids(offset: Int, length: Int) -> Series:
    """Rows [offset, offset + length)'s row ids: the scan's `rowid`."""
    var ids = List[Int64](unsafe_uninit_length=length)
    for i in range(length):
        ids[i] = Int64(offset + i)
    return Series(ROW_ID_COLUMN, Column[Int64](ids^))


def _vectors_kept(
    info: ScanFilterInfo, offset: Int, length: Int
) raises -> Optional[List[Int]]:
    """The starts (within the morsel) of the 2,048-row vectors the filter
    does not rule out; None when it rules none out."""
    var kept = List[Int]()
    var ruled_out = False
    var at = 0
    while at < length:
        var end = min(length, at + STATISTICS_VECTOR_SIZE)
        if info.skip(offset + at, offset + end):
            ruled_out = True
        else:
            kept.append(at)
        at = end
    if not ruled_out:
        return None
    return kept^


def _rows_in_vectors(
    rows: Optional[List[Int]], vectors: List[Int], length: Int
) -> List[Int]:
    """The rows of `rows` (None: every row of the morsel) inside the kept
    vectors."""
    var out = List[Int]()
    if not rows:
        for start in vectors:
            for r in range(start, min(length, start + STATISTICS_VECTOR_SIZE)):
                out.append(r)
        return out^
    ref mine = rows.value()
    var v = 0
    for r in mine:
        while v < len(vectors) and vectors[v] + STATISTICS_VECTOR_SIZE <= r:
            v += 1
        if v < len(vectors) and vectors[v] <= r:
            out.append(r)
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
    indexes: ArcPointer[List[Optional[JoinHashTable]]],
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
        var empty = jobs[0].runner.schema(j)
        # Every worker flagged its matches in the step's one table.
        var marks = List[Int]()
        marks.append(jobs[0].marks[j])
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


def _set_mask(set_masks: List[List[Int64]], s: Int, j: Int) -> Int64:
    """Set `s`'s mask for its `j`th key column (-1: the whole value)."""
    if len(set_masks) == 0:
        return -1
    return set_masks[s][j]


def run_pipeline(
    frame: DataFrame,
    steps: List[Step],
    joins: ArcPointer[List[DataFrame]],
    indexes: ArcPointer[List[Optional[JoinHashTable]]],
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
    top_maintain_order: Bool = False,
    top_scan_key: Bool = False,
    top_ranges: Optional[List[Int]] = None,
    top_row_ids: Bool = False,
    table: Bool = False,
    set_masks: List[List[Int64]] = List[List[Int64]](),
    finish_steps: List[Step] = List[Step](),
    pushed: List[PushedJoinFilter] = List[PushedJoinFilter](),
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
    empty output; `set_masks`: per set, each key column's mask, for keys
    that pack several coded keys). `finish_steps` are row-local steps (no
    joins) over the reduce sink's output, run where it is made: on each
    worker finishing a part of the aggregate table, else as a pipeline
    over the output. `pushed` are join keys from above an aggregate these
    steps feed, for the source's join filters (`_join_filters`).
    `counts` receives the rows fed, then each step's input and output
    rows, then each finish step's."""
    var finish_counts = List[Int]()
    var result = _run_pipeline(
        frame,
        steps,
        joins,
        indexes,
        expressions,
        keys,
        ordered,
        insert,
        workers,
        morsel_rows,
        batch_size,
        counts,
        set_frames,
        sets,
        top,
        top_names,
        top_descending,
        top_nulls_last,
        top_maintain_order,
        top_scan_key,
        top_ranges,
        top_row_ids,
        table,
        set_masks,
        finish_steps,
        finish_counts,
        pushed,
    )
    if len(finish_steps) > 0 and len(finish_counts) == 0:
        # The sink finished the output whole: the steps run over it as
        # a pipeline of their own, morsels on every worker.
        var inner = List[Int]()
        var none = List[DataFrame]()
        result = _run_pipeline(
            result,
            finish_steps,
            joins,
            indexes,
            List[Expr](),
            List[String](),
            False,
            False,
            workers,
            batch_size,
            batch_size,
            inner,
            none,
            List[List[Int]](),
            -1,
            List[String](),
            List[Bool](),
            List[Bool](),
            False,
            False,
            None,
            False,
            False,
            List[List[Int64]](),
            List[Step](),
            finish_counts,
            List[PushedJoinFilter](),
        )
        for k in range(len(finish_steps)):
            finish_counts.append(inner[2 * k + 1])
            finish_counts.append(inner[2 * k + 2])
    counts.extend(Span(finish_counts))
    return result^


struct _FinishStepsJob(Job):
    """One part of the aggregate table finished, then run through the
    row-local steps above the aggregate on the worker that finished it,
    as DuckDB runs a filter over an aggregate in the pipeline whose source
    is the aggregate table's scan: the groups it drops are never
    assembled into the whole result, nor filtered again over it."""

    var state: _StreamReduction
    var steps: List[Step]
    var joins: ArcPointer[List[DataFrame]]
    var indexes: ArcPointer[List[Optional[JoinHashTable]]]
    var batch_size: Int
    var frame: DataFrame
    # Each step's input and output rows.
    var counts: List[Int]

    def __init__(
        out self,
        var state: _StreamReduction,
        steps: List[Step],
        joins: ArcPointer[List[DataFrame]],
        indexes: ArcPointer[List[Optional[JoinHashTable]]],
        batch_size: Int,
    ) raises:
        self.state = state^
        self.steps = steps.copy()
        self.joins = joins
        self.indexes = indexes
        self.batch_size = batch_size
        self.frame = DataFrame(List[Series](), height=0)
        self.counts = List[Int]()

    def run(mut self) raises:
        self.frame = _steps_over_frame(
            self.state.finish(),
            self.steps,
            self.joins,
            self.indexes,
            self.counts,
        )


struct _HavingFilter(Movable):
    """A filter over an aggregate's outputs alone (SQL's HAVING), run by
    the aggregate table's finish before it reads out the groups
    (`GroupFilter`): `slots` are the reductions the outputs it reads come
    from, `names` and `dtypes` their states' names and types, and
    `outputs` those outputs."""

    var predicate: Expr
    var slots: List[Int]
    var names: List[String]
    var dtypes: List[DataType]
    var outputs: List[Expr]
    var batch_size: Int

    def __init__(
        out self,
        var predicate: Expr,
        var slots: List[Int],
        var names: List[String],
        var dtypes: List[DataType],
        var outputs: List[Expr],
        batch_size: Int,
    ):
        self.predicate = predicate^
        self.slots = slots^
        self.names = names^
        self.dtypes = dtypes^
        self.outputs = outputs^
        self.batch_size = batch_size


def _having(
    template: _StreamReduction,
    step: Step,
    keys: List[String],
    batch_size: Int,
) raises -> Optional[_HavingFilter]:
    """`step` as a filter the aggregate table's finish runs, when it is a
    filter reading only outputs of the reductions, no key."""
    if step.kind != STEP_FILTER:
        return None
    var reads = _reads(step.exprs)
    if not reads or len(reads.value()) == 0:
        return None
    var outputs = List[Expr]()
    var slots = List[Int]()
    for name in reads.value():
        if name in keys:
            return None
        var found = -1
        for o in range(len(template.outputs)):
            if template.outputs[o]._name == name:
                found = o
        if found < 0:
            return None
        outputs.append(template.outputs[found].copy())
        var states = _reads([template.outputs[found].copy()])
        if not states:
            return None
        for state in states.value():
            if state not in template.names:
                return None
            var i = template.names.index(state)
            if i not in slots:
                slots.append(i)
    sort(slots)
    var names = List[String](capacity=len(slots))
    var dtypes = List[DataType](capacity=len(slots))
    for i in slots:
        names.append(template.names[i])
        dtypes.append(template.dtypes[i])
    return _HavingFilter(
        step.exprs[0].copy(),
        slots^,
        names^,
        dtypes^,
        outputs^,
        batch_size,
    )


def _having_keep(context: Int, states: List[Reducer]) raises -> List[Int]:
    """The groups a `_HavingFilter` at `context` keeps, from its
    reductions' finished results over a finish job's groups."""
    ref having = Pointer[_HavingFilter, MutAnyOrigin](
        unsafe_from_address=context
    )[]
    var columns = List[Series](capacity=len(states))
    for i in range(len(states)):
        columns.append(
            states[i]
            .finish()
            .with_dtype(having.dtypes[i])
            .renamed(having.names[i])
        )
    var height = len(columns[0])
    var values = DataFrame(columns^, height=height).select_exprs(having.outputs)
    return true_rows(values._selection(having.predicate, having.batch_size))


def _finish_with_steps(
    template: _StreamReduction,
    var parts: List[_StreamReduction],
    steps: List[Step],
    joins: ArcPointer[List[DataFrame]],
    indexes: ArcPointer[List[Optional[JoinHashTable]]],
    batch_size: Int,
    mut counts: List[Int],
) raises -> DataFrame:
    """The aggregate table's `parts` finished and run through `steps`,
    each part on its own worker, the kept rows of each a chunk of the
    result; `counts` receives each step's input and output rows. With no
    part (no groups), the steps run over the empty finished output."""
    if len(parts) == 0:
        # No group: the steps over the empty result, for its columns.
        return _steps_over_frame(
            template.finish(), steps, joins, indexes, counts
        )
    trace_path("pipeline.aggregate_finish_steps")
    var finishes = List[_FinishStepsJob](capacity=len(parts))
    while len(parts) > 0:
        finishes.append(
            _FinishStepsJob(parts.pop(0), steps, joins, indexes, batch_size)
        )
    run_jobs(finishes)
    var frames = List[DataFrame](capacity=len(finishes))
    counts = List[Int](length=2 * len(steps), fill=0)
    for j in range(len(finishes)):
        frames.append(finishes[j].frame.copy())
        for k in range(len(counts)):
            counts[k] += finishes[j].counts[k]
    return chunks_frame(frames)


def _run_pipeline(
    frame: DataFrame,
    given_steps: List[Step],
    joins: ArcPointer[List[DataFrame]],
    indexes: ArcPointer[List[Optional[JoinHashTable]]],
    given_expressions: List[Expr],
    keys: List[String],
    ordered: Bool,
    insert: Bool,
    workers: Int,
    morsel_rows: Int,
    batch_size: Int,
    mut counts: List[Int],
    mut set_frames: List[DataFrame],
    sets: List[List[Int]],
    top: Int,
    top_names: List[String],
    top_descending: List[Bool],
    top_nulls_last: List[Bool],
    top_maintain_order: Bool,
    top_scan_key: Bool,
    top_ranges: Optional[List[Int]],
    top_row_ids: Bool,
    table: Bool,
    set_masks: List[List[Int64]],
    finish_steps: List[Step],
    mut finish_counts: List[Int],
    pushed: List[PushedJoinFilter],
) raises -> DataFrame:
    """`run_pipeline`; `finish_counts` receives each finish step's input
    and output rows when the aggregate table's finish ran them, and stays
    empty otherwise."""
    trace_path("pipeline.run")
    # Every step's input columns and the output, typed once, and the
    # row-local steps compiled once for every worker. A late-materialized
    # top-N's scan adds its row id column to every morsel.
    var schema_frame = frame.clear()
    if top_row_ids:
        schema_frame = schema_frame.with_column(
            Series(ROW_ID_COLUMN, Column[Int64](List[Int64]()))
        )
    var steps = given_steps.copy()
    var expressions = given_expressions.copy()
    var typed = _step_schemas(schema_frame, steps, joins[])
    if len(expressions) > 0:
        var extracted = _extract_aggregate_inputs(
            steps, expressions, keys, typed[len(steps)]
        )
        if len(extracted[0]) != len(steps):
            trace_path("pipeline.aggregate_projection")
            steps = extracted[0].copy()
            expressions = extracted[1].copy()
            typed.append(
                typed[len(typed) - 1].select_exprs(steps[len(steps) - 1].exprs)
            )
    var schemas = ArcPointer(typed^)
    var templates = ArcPointer(_compile_steps(steps, schemas[]))
    # The aggregate table's layout, from the steps' output types (an
    # empty frame through every step) when it takes this reduction.
    var layout = List[AggregateLayout]()
    var template = List[_StreamReduction]()
    var set_empty = List[DataFrame]()
    if table and len(keys) > 0 and len(expressions) > 0:
        var empty = schemas[][len(steps)].copy()
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
    # A top-N's row groups in their statistics order (RowGroupReorderer):
    # workers take them in turn, one row group each, as DuckDB's
    # NextParallelScan hands out the reorderer's next row group.
    var ranges = ArcPointer(
        top_ranges.value().copy() if top_ranges else _morsel_ranges(
            frame, max(1, morsel_rows), max(1, workers)
        )
    )
    # Grouping sets: which come from another set's group rows. Largest
    # first, each takes the smallest earlier source whose keys hold its own,
    # when every reduction merges (`reaggregable`).
    var derived_from = List[Int](length=len(sets), fill=-1)
    var derived_columns = List[List[Int]]()
    var derived_masks = List[List[Int64]]()
    var derive_order = List[Int]()
    for _ in range(len(sets)):
        derived_columns.append(List[Int]())
        derived_masks.append(List[Int64]())
    if len(layout) > 0 and len(sets) > 1:
        var merges = True
        for expression in expressions:
            for node in expression._nodes:
                if is_reduction(node.op) and not reaggregable(node.op):
                    merges = False
        if merges:
            # A set's size counts its key columns' mask bits (a whole key
            # all 64), so a set holding another's keys is the larger.
            var sizes = List[Int](capacity=len(sets))
            for t in range(len(sets)):
                var size = 0
                for j in range(len(sets[t])):
                    size += Int(pop_count(_set_mask(set_masks, t, j)))
                sizes.append(size)
            var order = derive_order.copy()
            for t in range(len(sets)):
                order.append(t)
            # Stable by size, largest first.
            for i in range(1, len(order)):
                var j = i
                while j > 0 and sizes[order[j - 1]] < sizes[order[j]]:
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
                    for j in range(len(sets[t])):
                        var key = sets[t][j]
                        if key not in sets[q]:
                            inside = False
                            break
                        # A packed key's fields: the source keeps every
                        # field this set does.
                        var kept = _set_mask(set_masks, q, sets[q].index(key))
                        if _set_mask(set_masks, t, j) & ~kept != 0:
                            inside = False
                            break
                    if inside and (best < 0 or sizes[q] < sizes[best]):
                        best = q
                # Every set made so far is a source, a derived one too: a
                # rollup's sets come each from the one above it.
                primaries.append(t)
                if best < 0:
                    continue
                derived_from[t] = best
                for j in range(len(sets[t])):
                    var key = sets[t][j]
                    var at = sets[best].index(key)
                    derived_columns[t].append(at)
                    # Only a mask the source does not already apply
                    # narrows the source's word.
                    var mask = _set_mask(set_masks, t, j)
                    derived_masks[t].append(
                        Int64(-1) if mask
                        == _set_mask(set_masks, best, at) else mask
                    )
            var primary_count = 0
            for t in range(len(sets)):
                if derived_from[t] < 0:
                    primary_count += 1
            if primary_count < len(sets):
                trace_path("pipeline.group_table_derived_sets")
    var cursor = _Cursor.new()
    var boundary = TopNBoundary.new() if top > 0 else TopNBoundary.none()
    var top_filter = _top_scan_filter(
        frame,
        top,
        top_names,
        top_descending,
        top_nulls_last,
        top_scan_key,
        top_maintain_order,
    )
    top_filter.reordered = Bool(top_ranges)
    top_filter.row_ids = top_row_ids
    var join_scan_filters = TableFilterSet()
    var join_filters = _join_filters(
        frame, steps, joins[], indexes[], pushed, join_scan_filters
    )
    if len(join_filters) > 0:
        trace_path("pipeline.join_filter")
    for f in join_filters:
        if f.bloom:
            trace_path("pipeline.join_bloom")
            break
    for f in join_scan_filters.filters:
        _ensure_zones(frame.column(f.column))
    var join_scan = Optional[ArcPointer[ScanFilterInfo]]()
    if join_scan_filters.has_filters():
        join_scan = ArcPointer(
            ScanFilterInfo(join_scan_filters, frame._columns)
        )
    # A right or full join's flags of matched build rows start clear.
    for step in steps:
        if (
            step.kind == STEP_JOIN
            and (step.how == JOIN_RIGHT or step.how == JOIN_FULL)
            and indexes[][step.join]
        ):
            indexes[][step.join].value().reset_found()
    # One job per morsel at most: a tiny source (PDS-H q2's five-row
    # region frame) runs on the caller, with no crew dispatch to wait on.
    var job_count = max(1, min(workers, len(ranges[]) // 2))
    # The source's table filters with their answers per vector, found once
    # for every worker's chunk loop.
    var scan = Optional[ScanFilterInfo]()
    if len(steps) > 0 and steps[0].kind == STEP_FILTER:
        scan = scan_filter_info(frame, conjuncts(steps[0].exprs[0]))
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
                len(expressions) > 0,
                top,
                top_names,
                top_descending,
                top_nulls_last,
                top_maintain_order,
                boundary,
                top_filter,
                join_filters,
                join_scan,
                schemas,
                templates,
            )
        )
    # An ungrouped reduce sink: each worker's states updated from the
    # last step's chunks (DuckDB's `PhysicalUngroupedAggregate`).
    var ungrouped = List[ArcPointer[UngroupedLayout]]()
    if len(keys) == 0 and len(expressions) > 0 and top <= 0:
        var made = ungrouped_layout(schemas[][len(steps)], expressions)
        if made:
            ungrouped.append(ArcPointer(made.take()))
    if len(ungrouped) > 0:
        trace_path("pipeline.ungrouped_sink")
        for w in range(len(jobs)):
            jobs[w].runner.ungrouped_layout.append(ungrouped[0])
    # The aggregate table (one grouping): each worker's table takes the
    # last step's chunks (`PhysicalHashAggregate::Sink`).
    if len(layout) > 0 and len(sets) == 0:
        var empty = schemas[][len(steps)].copy()
        var empty_keys = List[Series](capacity=len(keys))
        for name in keys:
            empty_keys.append(empty.column(name))
        var table_layout = ArcPointer(
            TableSinkLayout(
                layout[0],
                empty,
                keys,
                expressions,
                table_inputs(empty, empty_keys, expressions),
            )
        )
        for w in range(len(jobs)):
            jobs[w].runner.table_layout.append(table_layout)
            jobs[w].runner.table_workers = workers
    # Grouping sets: each worker's tables, one a set, take the last step's
    # chunks, every key of every set read once a chunk.
    if len(layout) > 0 and len(sets) > 0:
        var empty = schemas[][len(steps)].copy()
        var empty_keys = List[Series](capacity=len(keys))
        for name in keys:
            empty_keys.append(empty.column(name))
        var sets_layout = ArcPointer(
            GroupingSetsLayout(
                TableSinkLayout(
                    layout[0],
                    empty,
                    keys,
                    expressions,
                    table_inputs(empty, empty_keys, expressions),
                ),
                layout,
                sets,
                set_masks,
                derived_from,
                derived_columns,
                derived_masks,
                derive_order,
            )
        )
        for w in range(len(jobs)):
            jobs[w].runner.sets_layout.append(sets_layout)
            jobs[w].runner.table_workers = workers
    # Any other reduction: each worker's reduce sink takes the chunks.
    var reduce_layout = List[ArcPointer[_ReduceSinkLayout]]()
    if len(expressions) > 0 and len(layout) == 0 and len(ungrouped) == 0:
        var filter_only = True
        for step in steps:
            if step.kind != STEP_FILTER:
                filter_only = False
        reduce_layout.append(
            ArcPointer(
                _ReduceSinkLayout(
                    schemas[][len(steps)].clear(),
                    expressions,
                    keys,
                    insert,
                    filter_only,
                    batch_size,
                )
            )
        )
        for w in range(len(jobs)):
            jobs[w].runner.reduce_layout.append(reduce_layout[0])
    # No reduction and no top-N: the rows are the result, each worker's
    # chunks copied into its collections (PhysicalBatchCollector).
    if len(expressions) == 0 and top <= 0:
        for w in range(len(jobs)):
            jobs[w].runner.materializes = True
    if scan:
        for w in range(len(jobs)):
            jobs[w].runner.scan = scan.copy()
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
        boundary.free()
        raise e
    cursor.free()
    boundary.free()
    counts = List[Int](length=3 * len(steps) + 1, fill=0)
    for j in range(len(jobs)):
        for i in range(len(counts)):
            counts[i] += jobs[j].counts[i]
    var planned = len(given_steps)
    if len(steps) != planned:
        # The aggregate inputs' projection has no plan node: the counts
        # are the planned steps'.
        var n = len(steps)
        var kept = List[Int](capacity=3 * planned + 1)
        for i in range(2 * planned + 1):
            kept.append(counts[i])
        for i in range(planned):
            kept.append(counts[2 * n + 1 + i])
        counts = kept^
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
        var locals = List[GroupingSetsSink]()
        for j in range(len(jobs)):
            if len(jobs[j].runner.sets) > 0:
                locals.append(jobs[j].runner.sets.pop())
        var combined = _combined(locals^)
        for s in finish_order:
            var place = remaining.index(s)
            var sinks = List[AggregateSink]()
            for j in range(len(combined)):
                sinks.append(combined[j].tables.pop(place))
            _ = remaining.pop(place)
            var result = aggregate_finish(sinks, layout[s], ordered=False)
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
        var locals = List[TableSink]()
        for j in range(len(jobs)):
            if len(jobs[j].runner.table) > 0:
                locals.append(jobs[j].runner.table.pop())
        var combined = _combined(locals^)
        var sinks = List[AggregateSink](capacity=len(combined))
        while len(combined) > 0:
            sinks.append(combined.pop(0).take_table())
        # Long key strings live in the sinks' heaps until the finish has
        # read them out.
        var finished: DataFrame
        if len(finish_steps) == 0:
            var result = aggregate_finish(sinks, layout[0], ordered=ordered)
            finished = table_frame(template[0], result^, ordered)
        elif ordered:
            # The groups in first-occurrence order (each group row's first
            # sequence number), then the steps over them.
            var result = aggregate_finish(sinks, layout[0], ordered=True)
            finished = _steps_over_frame(
                table_frame(template[0], result^, True),
                finish_steps,
                joins,
                indexes,
                finish_counts,
            )
        else:
            # A first step filtering on the reductions' outputs alone runs
            # in the finish, before the groups are read out (DuckDB runs
            # it on each chunk the aggregate's scan yields); the rest run
            # on each part of the groups that remain.
            var having = _having(template[0], finish_steps[0], keys, batch_size)
            var rest = finish_steps.copy()
            var filter = Optional[GroupFilter]()
            if having:
                trace_path("pipeline.aggregate_finish_filter")
                _ = rest.pop(0)
                filter = GroupFilter(
                    having.value().slots.copy(),
                    Int(Pointer(to=having.value())),
                    _having_keep,
                )
            var result = aggregate_finish(
                sinks, layout[0], filter, ordered=False
            )
            _ = having^
            var merged = result.merged
            var kept = 0
            for p in range(len(result.firsts)):
                kept += len(result.firsts[p])
            var parts = table_parts(template[0], result^)
            var rest_counts = List[Int]()
            finished = _finish_with_steps(
                template[0],
                parts^,
                rest,
                joins,
                indexes,
                batch_size,
                rest_counts,
            )
            if filter:
                finish_counts.append(merged)
                finish_counts.append(kept)
            finish_counts.extend(Span(rest_counts))
        _ = sinks^
        return finished^
    var reduces = List[_ReduceSink]()
    if len(reduce_layout) > 0:
        var locals = List[_ReduceSink]()
        for j in range(len(jobs)):
            if len(jobs[j].runner.reduce) > 0:
                locals.append(jobs[j].runner.reduce.pop())
        reduces = _combined(locals^)
    if len(expressions) > 0 and len(keys) > 0:
        var sink = -1
        for j in range(len(reduces)):
            if len(reduces[j].grouped) > 0:
                sink = j
        if sink < 0:
            var empty = schemas[][len(steps)].copy()
            return _StreamReduction(empty, expressions, keys).finish()
        if reduces[sink].grouped[0].collecting:
            # Every worker's rows, grouped once by the eager partitioned
            # group-by: one scatter of the rows into buckets, each encoded
            # on its own worker.
            # Only the columns the keys and expressions read are gathered
            # and grouped; the steps may have read others.
            var whole: DataFrame
            if len(steps) == 0 or reduce_layout[0][].filter_only:
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
                    for j in range(len(reduces)):
                        if len(reduces[j].grouped) == 0:
                            continue
                        ref kept = reduces[j].grouped[0]
                        for m in range(len(kept.offsets)):
                            offsets.append(kept.offsets[m])
                            lengths.append(kept.lengths[m])
                            masks.append(kept.masks[m].copy())
                    whole = source.filter(
                        _merged_mask(offsets, lengths, masks, frame.height())
                    )
            else:
                var collected = List[DataFrame]()
                for j in range(len(reduces)):
                    if len(reduces[j].grouped) == 0:
                        continue
                    var rows = List[DataFrame]()
                    swap(rows, reduces[j].grouped[0].collected)
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
        for j in range(len(reduces)):
            if (
                len(reduces[j].grouped) > 0
                and len(reduces[j].grouped[0].parts) > 1
            ):
                split = True
        var parts = List[List[_GroupPart]]()
        for j in range(len(reduces)):
            if len(reduces[j].grouped) == 0:
                continue
            var pieces = reduces[j].grouped[0].into_parts(frame.clear(), split)
            var p = 0
            while len(pieces) > 0:
                if len(parts) <= p:
                    parts.append(List[_GroupPart]())
                var tables = pieces.pop(0)
                while len(tables) > 0:
                    parts[p].append(tables.pop(0))
                p += 1
        ref meta = reduces[sink].grouped[0]
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
    if len(ungrouped) > 0:
        var sinks = List[UngroupedSink](capacity=len(jobs))
        for j in range(len(jobs)):
            if len(jobs[j].runner.ungrouped) > 0:
                sinks.append(jobs[j].runner.ungrouped.pop())
        var combined = _combined(sinks^)
        return ungrouped_finish(ungrouped[0][], combined)
    if len(expressions) > 0:
        var merged = List[_StreamReduction]()
        for j in range(len(reduces)):
            if len(reduces[j].reduction) == 0:
                continue
            if len(merged) == 0:
                merged.append(reduces[j].reduction.pop())
            else:
                merged[0].merge(reduces[j].reduction[0])
        if len(merged) == 0:
            # No row survived: reduce an empty frame of the right schema.
            var empty = schemas[][len(steps)].copy()
            return _StreamReduction(empty, expressions, List[String]()).finish()
        return merged[0].finish()
    if top <= 0:
        # The workers' batches in source order.
        var locals = List[MaterializeSink]()
        for j in range(len(jobs)):
            if len(jobs[j].runner.materialize) > 0:
                locals.append(jobs[j].runner.materialize.pop())
        var combined = _combined(locals^)
        return materialize_finish(combined, schemas[][len(steps)].clear())
    # A top-N's parts are its workers' heaps, each sorted (Finalize).
    var parts = List[DataFrame]()
    var locals = List[TopNSink]()
    for j in range(len(jobs)):
        if len(jobs[j].topn) > 0:
            locals.append(jobs[j].topn.pop())
    var heaps = _combined(locals^)
    for h in range(len(heaps)):
        var part = heaps[h].finish()
        if part.height() > 0:
            parts.append(part^)
    if len(parts) == 0:
        var empty = schemas[][len(steps)].copy()
        return empty^
    # Combine: the workers' heaps into one top-N.
    var merged = parts[0].copy() if len(parts) == 1 else concat(parts)
    if len(parts) > 1:
        var names = _top_keys(top_names, top_maintain_order)
        merged = merged.take(
            merged._arg_sort_head(
                names,
                _top_flags(top_descending, top_maintain_order),
                _top_flags(top_nulls_last, top_maintain_order),
                top,
                maintain_order=False,
            )
        )
    if top_maintain_order:
        merged = merged.drop([_ORDER_COLUMN])
    return merged^


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
