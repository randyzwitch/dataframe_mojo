"""Thread-owned pipelines over data chunks (docs/design/executor-design.md,
#538).

A pipeline is a source, a list of row-local steps, and a sink. Each worker
thread takes morsels (ranges of the source) from the shared cursor, cuts
each into data chunks of STANDARD_VECTOR_SIZE rows, runs the chunks through the steps, and updates its
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
from dataframe.storage.bitmap import _bitmap_word
from dataframe.storage.buffer import _Buffer, _BufferData
from std.atomic import Atomic
from std.bit import count_trailing_zeros
from std.os import getenv
from std.memory import (
    Allocation,
    Layout,
    alloc,
    dealloc,
    ArcPointer,
    Pointer,
    bitcast,
    unsafe_memcpy,
    unsafe_memcmp,
)
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
from .join_type import (
    JOIN_ANTI,
    JOIN_FULL,
    JOIN_INNER,
    JOIN_LEFT,
    JOIN_RIGHT,
    JOIN_RIGHT_ANTI,
    JOIN_RIGHT_SEMI,
    JOIN_SEMI,
    join_emits_build_rows,
)
from dataframe.kernels.row_encode import encodable
from dataframe.kernels.gather import (
    can_filter_aligned_chunks,
    take_parallel,
    true_rows,
)
from dataframe.storage.allocation import huge_list
from dataframe.kernels.mask_filter import filter_columns
from .parallel import Job, morsel_ranges, run_jobs, worker_count
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
from dataframe.vector.tuple_data_scatter_gather import gather_rows_series
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
# The column a top-k sink adds as its last key when the sort keeps input
# order: the row's place in the source's order (morsel sequence, then row
# within the morsel's output), so ties come out as a stable sort of the
# whole input would place them, whichever worker held them.
comptime _ORDER_COLUMN = "__pipeline_order"
# A late-materialized top-N scan's row id column.
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


def _key_statistics(
    frame: DataFrame, steps: List[Step], keys: List[String]
) -> List[Series]:
    """The source column whose statistics hold for each grouping key after
    the steps; empty when some key has none (a computed key,
    or one a right or full join coalesces with its build side's)."""
    var names = frame.columns()
    var sources = frame.columns()
    for step in steps:
        if step.kind == STEP_SELECT or step.kind == STEP_WITH_COLUMNS:
            var made = List[String]()
            var made_sources = List[String]()
            if step.kind == STEP_WITH_COLUMNS:
                made = names.copy()
                made_sources = sources.copy()
            for expr in step.exprs:
                for i in range(len(made)):
                    if made[i] == expr._name:
                        _ = made.pop(i)
                        _ = made_sources.pop(i)
                        break
                if len(expr._nodes) != 1 or expr._nodes[0].op != COL:
                    continue
                for i in range(len(names)):
                    if names[i] == expr._nodes[0].text:
                        made.append(expr._name)
                        made_sources.append(sources[i])
                        break
            names = made^
            sources = made_sources^
        elif step.kind == STEP_JOIN and (
            step.how == JOIN_RIGHT_SEMI or step.how == JOIN_RIGHT_ANTI
        ):
            # The build side is the output: no probe column stays.
            names = List[String]()
            sources = List[String]()
        elif step.kind == STEP_DROP or (
            step.kind == STEP_JOIN
            and (step.how == JOIN_RIGHT or step.how == JOIN_FULL)
        ):
            var kept = List[String]()
            var kept_sources = List[String]()
            for i in range(len(names)):
                if names[i] not in step.names:
                    kept.append(names[i])
                    kept_sources.append(sources[i])
            names = kept^
            sources = kept_sources^
    var found = List[Series](capacity=len(keys))
    for key in keys:
        var at = -1
        for i in range(len(names)):
            if names[i] == key:
                at = i
                break
        if at < 0:
            return List[Series]()
        try:
            found.append(frame.column(sources[at]))
        except:
            return List[Series]()
    return found^


comptime _AGGREGATE_INPUT = "__aggregate_input_"


def _extract_aggregate_inputs(
    steps: List[Step],
    expressions: List[Expr],
    keys: List[String],
    schema: DataFrame,
) raises -> Tuple[List[Step], List[Expr]]:
    """Project computed aggregate inputs into columns below the aggregate.
    Evaluate the projection over preceding pipeline chunks, then let each
    reduction read its input column. This shares expression evaluation and
    avoids materializing the entire intermediate frame."""
    var projected = List[Expr]()
    var rewritten = List[Expr](capacity=len(expressions))
    for e in expressions:
        var nodes = e._nodes.copy()
        var changed = False
        for i in range(len(nodes)):
            if not is_reduction(nodes[i].op):
                continue
            # A conditional aggregate input with no otherwise can project its
            # condition and value separately. Update the reduction only for
            # rows where the condition is true, preserving which values and
            # errors are observed.
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
    """Identify columns read for values versus validity or row counts. COUNT
    over a proven non-null column is the retained-row count, so it need not
    gather that column through a row selection. Keep validity reads for
    nullable columns."""
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
    """Derive source filters from finished builds of inner, semi and right
    joins. Only push a probe key still referring to an unchanged source
    column. Rejecting a source row must preserve the join's treatment of
    unmatched probe rows and all preceding expression observations."""
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
    """A source frame plus optional per-column row lists. A selected column is
    a dictionary view over its original values, and every visible column has
    height rows. Cut STANDARD_VECTOR_SIZE chunks without copying payload
    buffers."""

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
                # Columns sliced by the same filter share one row list and
                # selection vector rather than allocating duplicate index
                # arrays.
                var shared = -1
                for l in range(len(result.row_lists)):
                    if len(result.row_lists[l]) == len(listed) and (
                        unsafe_memcmp(
                            Pointer[UInt8, ImmutAnyOrigin](
                                unsafe_from_address=Int(
                                    result.row_lists[l].unsafe_ptr()
                                )
                            ),
                            Pointer[UInt8, ImmutAnyOrigin](
                                unsafe_from_address=Int(listed.unsafe_ptr())
                            ),
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
    chunk, reused for
    every chunk a worker runs through it."""

    var kind: Int
    # FILTER holds the compiled predicate; AND/OR children can adapt order
    # when errors permit. Literal true needs no evaluator. SELECT/WITH_COLUMNS
    # hold their computed expressions.
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

    # Share immutable step descriptions. Construct each worker's mutable
    # compiled expressions and sink state on its own thread, and release them
    # there at the end of its run.
    var steps: ArcPointer[List[Step]]
    var schemas: ArcPointer[List[DataFrame]]
    var compiled: List[Optional[_VectorStep]]
    # Bind expressions and evaluate constants once as shared templates. Each
    # worker copies its own mutable evaluation state before processing chunks.
    var templates: ArcPointer[List[Optional[_VectorStep]]]
    var joins: ArcPointer[List[DataFrame]]
    var indexes: ArcPointer[List[Optional[JoinHashTable]]]
    var arena: ColumnArena
    # The arena's columns held for the whole pipeline (join builds); a
    # morsel's are dropped when the next one starts.
    var held: Int
    # Per join step, the arena handles of its build's columns.
    var builds: List[List[Int]]
    # Retain source build frames for String and nested output columns. Their
    # dictionary views read payloads by row number without copying variable-
    # size bytes into the join rows.
    var frame_builds: List[List[Int]]
    # Whether this worker's arena holds the builds' columns yet.
    var bound: Bool
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
    # Retain scan-filter state and its source offset while the morsel runs.
    # Statistics can skip a vector before any values are read; -1 denotes no
    # source offset.
    var scan: Optional[ArcPointer[ScanFilterInfo]]
    var scan_offset: Int

    def __init__(
        out self,
        steps: ArcPointer[List[Step]],
        schemas: ArcPointer[List[DataFrame]],
        joins: ArcPointer[List[DataFrame]],
        indexes: ArcPointer[List[Optional[JoinHashTable]]],
        stop_empty: Bool,
        templates: ArcPointer[List[Optional[_VectorStep]]] = ArcPointer(
            List[Optional[_VectorStep]]()
        ),
    ) raises:
        var n = len(steps[])
        self.steps = steps
        self.schemas = schemas
        self.templates = templates
        if len(self.templates[]) != n:
            self.templates = ArcPointer(
                List[Optional[_VectorStep]](length=n, fill=None)
            )
        self.joins = joins
        self.indexes = indexes
        self.compiled = List[Optional[_VectorStep]](length=n, fill=None)
        self.arena = ColumnArena()
        self.builds = List[List[Int]]()
        self.frame_builds = List[List[Int]]()
        self.bound = False
        self.held = 0
        self.source = DataChunk()
        self.counts = List[Int](length=3 * n + 1, fill=0)
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

    def begin_morsel(mut self) raises:
        """Drop the previous morsel's columns. At a worker's first morsel,
        hold the join builds' columns, on its own thread: a worker that
        claims no morsel never does."""
        if not self.bound:
            self.bound = True
            self.builds = List[List[Int]](capacity=len(self.steps[]))
            self.frame_builds = List[List[Int]](capacity=len(self.steps[]))
            for step in self.steps[]:
                var handles = List[Int]()
                var frame_handles = List[Int]()
                if step.kind == STEP_JOIN:
                    if (
                        self.indexes[][step.join]
                        and self.indexes[][step.join].value().use_dictionary
                    ):
                        # The table's dictionary columns (perfect or
                        # dictionary emission), one per output column.
                        ref dictionary = (
                            self.indexes[][step.join].value().data[].dictionary
                        )
                        for c in range(len(dictionary)):
                            handles.append(self.arena.add(dictionary[c]))
                    for column in self.joins[][step.join]._columns:
                        frame_handles.append(self.arena.add(column))
                self.builds.append(handles^)
                self.frame_builds.append(frame_handles^)
            self.held = self.arena.mark()
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

    def release_local(mut self):
        """Release worker-local compiled steps on the worker that used them.
        This distributes cleanup instead of serially freeing every worker's
        state on the caller after each round."""
        self.compiled = List[Optional[_VectorStep]]()
        self.arena = ColumnArena()
        self.source = DataChunk()

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
        return self.schemas[][len(self.steps[])].copy()

    def run(mut self, var columns: _Columns, first: Int) raises -> _Columns:
        """The steps from `first` on over `columns` (the input of step
        `first`)."""
        return self.run_until(columns^, first, len(self.steps[]))

    def run_to_last(
        mut self, var columns: _Columns, first: Int
    ) raises -> Tuple[_Columns, Int]:
        """The steps from `first` up to the last row-local segment: the
        columns that segment reads and the step it starts at (the step
        count when the last step is a join). The segment then runs one
        chunk at a time through `chunk`, so a sink takes each chunk as it
        is made."""
        var n = len(self.steps[])
        var last = n
        while last > first and self.steps[][last - 1].kind != STEP_JOIN:
            last -= 1
        if last > first:
            columns = self.run_until(columns^, first, last)
        for s in range(last, n):
            self._compiled(s)
        return (columns^, last)

    def run_until(
        mut self,
        var columns: _Columns,
        first: Int,
        stop: Int,
    ) raises -> _Columns:
        """Steps [first, stop) over `columns`, collected."""
        var at = first
        var n = len(self.steps[])
        while at < stop:
            if self.stop_empty and columns.height == 0:
                return columns^
            if self.steps[][at].kind == STEP_JOIN:
                self.counts[2 * at + 1] += columns.height
                var began = Int(perf_counter_ns())
                columns = self._join(columns, at)
                self.counts[2 * n + 1 + at] += Int(perf_counter_ns()) - began
                self.counts[2 * at + 2] += columns.height
                at += 1
                continue
            var end = at
            while end < stop and self.steps[][end].kind != STEP_JOIN:
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
        var n = len(self.steps[])
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
                    self.steps[][s], self.schemas[][s], self.schemas[][s + 1]
                )

    def _vector_steps(
        mut self, columns: _Columns, first: Int, stop: Int
    ) raises -> _Columns:
        """Steps [first, stop), all row-local, chunk by chunk."""
        for s in range(first, stop):
            self._compiled(s)
        var n = len(self.steps[])
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
                if self.scan.value()[].skip(at, at + count):
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

    def _join(mut self, columns: _Columns, s: Int) raises -> _Columns:
        """A join step over a morsel's collected columns: it probes the
        step's table (`ScanStructure`) with the key columns only and keeps
        the probe side a dictionary over where it is. An inner join leaves the build
        side a dictionary over the build's columns (`EmitDictVectors`); a
        left, full or right join gathers it (`GatherRHS`), and flags the
        build rows it matched for `begin_round`, as a right-semi or
        right-anti join does without emitting anything. A step with no
        table (nested keys) or whose keys need dictionaries unified joins
        the morsel's frame eagerly."""
        ref step = self.steps[][s]
        ref input = self.schemas[][s]
        ref output = self.schemas[][s + 1]
        ref right = self.joins[][step.join]
        var direct = Bool(self.indexes[][step.join])
        if direct:
            for i in range(len(step.names)):
                var lt = input.column(step.names[i]).dtype()
                var rt = right.column(step.right_keys[i]).dtype()
                if lt != rt or lt.is_nested():
                    direct = False
        if not direct:
            var frame = columns.whole(self.arena, input)
            var joined = frame._join_impl(
                right,
                left_on=step.names,
                right_on=step.right_keys,
                how=step.how,
                suffix=step.suffix,
                coalesce=step.coalesce,
            )
            return _Columns.of_frame(self.arena, joined)
        trace_path("pipeline.join_dictionary")
        var keys = columns.frame(self.arena, input, step.names)._columns.copy()
        for k in range(len(keys)):
            if keys[k].is_chunked():
                keys[k] = keys[k].rechunk()
        ref table = self.indexes[][step.join].value()
        var scan = ScanStructure(table, step.how)
        var how = step.how
        var semi_or_anti = how == JOIN_SEMI or how == JOIN_ANTI
        var right_semi = how == JOIN_RIGHT_SEMI or how == JOIN_RIGHT_ANTI
        var pairs = not semi_or_anti and not right_semi
        var probe_rows = List[Int](capacity=columns.height)
        # Per output row, the build row's pointer and its dictionary
        # index (the slot of a perfect join, a row number with dictionary
        # emission).
        var pointers = List[UInt64]()
        var dictionary_rows = List[UInt32]()
        if pairs:
            pointers.reserve(columns.height)
            if table.use_dictionary:
                dictionary_rows.reserve(columns.height)
        for low in range(0, columns.height, STANDARD_VECTOR_SIZE):
            var count = min(STANDARD_VECTOR_SIZE, columns.height - low)
            scan.probe(keys, low, count)
            while True:
                var n = scan.next()
                if n == 0:
                    break
                var rows = scan.probe_rows.unsafe_ptr()
                for k in range(n):
                    probe_rows.append(low + Int(rows[unsafe_offset=k]))
                if pairs:
                    var at = scan.row_pointers.unsafe_ptr()
                    for k in range(n):
                        pointers.append(at[unsafe_offset=k])
                    if table.use_dictionary:
                        var builds = scan.build_rows.unsafe_ptr()
                        for k in range(n):
                            dictionary_rows.append(
                                UInt32(builds[unsafe_offset=k])
                            )
        if right_semi:
            # A probe only flags build rows; `begin_round` emits them.
            return _Columns.of_frame(self.arena, output)
        if semi_or_anti:
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
        result.height = len(pointers)
        var right_keys = List[Int]()
        for key in step.right_keys:
            right_keys.append(right._index(key))
        # The build side's output columns, in the join's output order:
        # from the rows (GatherRHS), the table's dictionaries (a perfect
        # join or dictionary emission), or, for a column a plan replaced
        # after the build, the build frame by each row's number.
        var list = -1
        var frame_list = -1
        var has_unmatched = how == JOIN_LEFT or how == JOIN_FULL
        for i in range(right.width()):
            if step.coalesce and i in right_keys:
                continue
            ref column = right._columns[i]
            var t = table.output_index(column.name())
            if t < 0 and not has_unmatched:
                # Read by row number through the build frame (no bytes
                # copied, a string_t gather's cost; a column a plan
                # replaced after the build).
                if frame_list < 0:
                    frame_list = len(result.row_lists)
                    var numbers = List[UInt32](capacity=len(pointers))
                    for r in range(len(pointers)):
                        numbers.append(
                            UInt32(table.build_side_row(pointers[r]))
                        )
                    result.row_lists.append(numbers^)
                result.handles.append(self.frame_builds[s][i])
                result.starts.append(0)
                result.lists.append(frame_list)
            elif t >= 0 and table.use_dictionary:
                if list < 0:
                    list = len(result.row_lists)
                    result.row_lists.append(dictionary_rows.copy())
                result.handles.append(self.builds[s][t])
                result.starts.append(0)
                result.lists.append(list)
            else:
                result.handles.append(
                    self.arena.add(_build_series(table, column, pointers))
                )
                result.starts.append(0)
                result.lists.append(-1)
        if result.width() != output.width():
            raise Error("pipeline join: output columns differ from its schema")
        return result^


def _build_series(
    table: JoinHashTable, column: Series, pointers: List[UInt64]
) raises -> Series:
    """Build column `column` at the rows `pointers` (GatherRHS), or, for a
    column a plan replaced after the build, through each row's number
    from the build frame (`JoinHashTable.mark_frame_only`)."""
    var t = table.output_index(column.name())
    if t >= 0:
        return gather_rows_series(
            table.row_layout,
            Int(pointers.unsafe_ptr()),
            len(pointers),
            table.output_columns[t],
            column.name(),
            table.output_types[t],
        )
    var rows = List[Int](capacity=len(pointers))
    for r in range(len(pointers)):
        rows.append(table.build_side_row(pointers[r]))
    return take_parallel([column.copy()], rows^, 1, or_null=True)[0].copy()


def _sink_chunk[
    S: Sink
](mut sink: S, mut chunk: DataChunk, mut arena: ColumnArena) raises:
    """One chunk into a worker's local sink state."""
    sink.sink(chunk, arena)


def _combined[S: Sink](var locals: List[S]) raises -> List[S]:
    """Every worker's local sink state combined into the global state, in worker order."""
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
    var runner = _ChunkRunner(
        ArcPointer(steps.copy()), schemas, joins, indexes, False
    )
    runner.materializes = True
    runner.begin_sink(0)
    runner.begin_morsel()
    var source = _Columns.of_frame(runner.arena, frame)
    _ = runner.run(source^, 0)
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
    in stable heap storage, as the thread budget of `parallel` keeps its
    counters."""

    var address: Int

    @staticmethod
    def new() -> _Cursor:
        var pointer = alloc(Layout[Atomic[Int64]](length=1)).unsafe_leak()
        pointer.unsafe_write(Atomic[Int64](0))
        return _Cursor(Int(pointer))

    def take(self, rows: Int) -> Int:
        return Int(
            Pointer[Atomic[Int64], MutAnyOrigin](
                unsafe_from_address=self.address
            )[].fetch_add(Int64(rows))
        )

    def free(self):
        dealloc(
            Allocation[Atomic[Int64]](
                unsafe_owned_ptr=Pointer[Atomic[Int64], MutUntrackedOrigin](
                    unsafe_from_address=self.address
                ),
                layout=Layout[Atomic[Int64]](length=1),
            )
        )


struct _ReduceSinkLayout(Copyable, Movable):
    """What every worker's reduce sink shares (an ungrouped reduction the
    ungrouped sink does not take): the sink input's types, the reductions
    and the input columns read."""

    var schema: DataFrame
    var expressions: List[Expr]
    # None: a selector may read any column, so every one is read.
    var reads: Optional[List[String]]

    def __init__(out self, schema: DataFrame, expressions: List[Expr]):
        self.schema = schema.copy()
        self.expressions = expressions.copy()
        self.reads = _reads(expressions)


struct _ReduceSink(Sink):
    """One worker's reduce sink state, fed a chunk at a time: its reducer
    state, each chunk's state merged into it."""

    var shared: _ReduceSinkLayout
    var reduction: List[_StreamReduction]
    var sequence: Int

    def __init__(out self, shared: _ReduceSinkLayout):
        self.shared = shared.copy()
        self.reduction = List[_StreamReduction]()
        self.sequence = 0

    def begin(mut self, morsel: Int):
        self.sequence = morsel << 32

    def sink(mut self, mut chunk: DataChunk, mut arena: ColumnArena) raises:
        var n = chunk.count
        if n == 0:
            return
        ref schema = self.shared.schema
        var expressions = self.shared.expressions.copy()
        var input: DataFrame
        if self.shared.reads:
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
        else:
            input = _chunk_frame(chunk, arena, schema, schema.columns())
        var first = self.sequence
        self.sequence += n
        var state = _StreamReduction(input, expressions, List[String]())
        state.shift_firsts(first)
        if len(self.reduction) == 0:
            self.reduction.append(state^)
        else:
            self.reduction[0].merge(state)

    def release(mut self) raises:
        pass

    def combine(var self, mut combined: List[_ReduceSink]) raises:
        """This worker's state into the global state, every worker's: the
        finish merges them."""
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


struct _PipelineJob(Job):
    """One worker's run of the pipeline: morsels from the shared cursor
    through every step into this worker's own sink state."""

    var frame: ArcPointer[DataFrame]
    var cursor: Int
    # Morsel ranges as (offset, length) pairs, none crossing a chunk
    # boundary of the frame's columns (a slice within one Parquet row
    # group is a view; one across two is copied when a kernel reads it).
    var ranges: ArcPointer[List[Int]]
    var steps: ArcPointer[List[Step]]
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
    # Dynamic scan filter from the top-N boundary on the leading sort key,
    # when that key is an unchanged source column.
    var top_filter: _TopScanFilter
    # Rows fed, then each step's input and output rows, for the report.
    var counts: List[Int]
    # The join builds' filters on source columns (`_join_filters`): row
    # filters, with this worker's selectivity of each, and the statistics
    # filters with their answers per vector.
    var join_filters: List[JoinFilter]
    var join_stats: List[SelectivityStats]
    var join_scan: Optional[ArcPointer[ScanFilterInfo]]
    # The step the morsels enter at (0: the source's; past an outer join
    # step for the build rows it emits after its probe), and the number
    # of the frame's first morsel among all the morsels sunk.
    var first_step: Int
    var first_morsel: Int
    # Whether this run ends the worker's work (folds its top-k candidates
    # and derives its grouping sets): not before an outer join's
    # unmatched build rows have run.
    var finishes: Bool
    # While an outer join's unmatched build rows run (`begin_round`):
    # that join's step (-1 otherwise) and the probe side's columns at the
    # step, empty. The frame is then the build, and each worker finds its
    # morsel's unmatched rows itself.
    var unmatched_step: Int
    var unmatched_probe: ArcPointer[DataFrame]
    # The steps over data chunks, with this worker's vectors and arena.
    var runner: _ChunkRunner

    def __init__(
        out self,
        frame: ArcPointer[DataFrame],
        cursor: Int,
        ranges: ArcPointer[List[Int]],
        steps: ArcPointer[List[Step]],
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
        self.first_step = 0
        self.first_morsel = 0
        self.finishes = True
        self.unmatched_step = -1
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
        self.steps = steps
        self.joins = joins.copy()
        self.indexes = indexes.copy()
        self.reduces = reduces
        # Rows fed, then each step's rows in and out, then each step's
        # busy nanoseconds on this worker.
        self.counts = List[Int](length=3 * len(steps[]) + 1, fill=0)

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

    def run(mut self) raises:
        var count = len(self.ranges[]) // 2
        var cursor = _Cursor(self.cursor)
        while True:
            var taken = cursor.take(1)
            if taken >= count:
                self.counts = self.runner.counts.copy()
                if not self.finishes:
                    return
                self.runner.release_sink()
                self.runner.release_local()
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
                self._sink_from(tail^, k, offset)
                continue
            var morsel = self.frame[].slice(offset, length)
            if self.top_filter.row_ids:
                morsel = morsel.with_column(_row_ids(offset, length))
            if self.first_step > 0:
                # An outer join's unmatched build rows: the source's key
                # ranges and top-k bound do not apply to them.
                var source = _Columns.of_frame(self.runner.arena, morsel)
                self._sink_from(source^, k, offset)
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
                self._top_chunks(source^, rows, k, offset, 0)
                self.runner.scan_offset = -1
                continue
            _ = self.runner.run(source^, 0)
            self.runner.scan_offset = -1

    def _sink_from(mut self, var columns: _Columns, k: Int, offset: Int) raises:
        """An outer join's unmatched build rows (or a morsel of them past
        its step) through the steps after `first_step` into the sink."""
        if self._chunk_top():
            self._top_chunks(columns^, None, k, offset, self.first_step)
        else:
            _ = self.runner.run(columns^, self.first_step)

    def _chunk_top(self) -> Bool:
        """Whether the sink is a top-N, taking chunks one at a time."""
        return self.top > 0 and not self.reduces

    def _top_chunks(
        mut self,
        var source: _Columns,
        rows: Optional[List[Int]],
        k: Int,
        offset: Int,
        first_step: Int,
    ) raises:
        """Feed a morsel into the top-N sink one chunk at a time from
        first_step. Recheck the dynamic boundary before reading each vector:
        earlier chunks may have tightened it enough to skip later vectors."""
        var ran = self.runner.run_to_last(source^, first_step)
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
                if self.runner.scan and self.runner.scan.value()[].skip(
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

    def _unmatched(self, chunk: Int, length: Int) raises -> DataFrame:
        """The build rows of the hash table's row chunk `chunk` the join
        emits after the probe, as the step's output: a
        right-semi or right-anti join's build columns, or an outer join's
        with the probe columns null, coalesced keys taken from the rows."""
        ref step = self.steps[][self.unmatched_step]
        ref table = self.indexes[][step.join].value()
        ref right = self.joins[][step.join]
        var want = UInt8(1) if step.how == JOIN_RIGHT_SEMI else UInt8(0)
        var pointers = List[UInt64]()
        table.scan_chunk(chunk, want, pointers)
        ref schema = self.runner.schemas[][self.unmatched_step + 1]
        if len(pointers) == 0:
            return schema.clear()
        trace_path("pipeline.join_unmatched")
        var n = len(pointers)
        var columns = List[Series](capacity=schema.width())
        var right_semi = (
            step.how == JOIN_RIGHT_SEMI or step.how == JOIN_RIGHT_ANTI
        )
        if not right_semi:
            ref probe = self.unmatched_probe[]
            for c in range(probe.width()):
                ref column = probe._columns[c]
                var key = -1
                if step.coalesce:
                    for k in range(len(step.names)):
                        if step.names[k] == column.name():
                            key = k
                if key >= 0:
                    columns.append(
                        gather_rows_series(
                            table.row_layout,
                            Int(pointers.unsafe_ptr()),
                            n,
                            key,
                            column.name(),
                            column.dtype(),
                        )
                    )
                else:
                    columns.append(
                        Series.full_null(column.name(), column.dtype(), n)
                    )
        var right_keys = List[Int]()
        for key in step.right_keys:
            right_keys.append(right._index(key))
        for i in range(right.width()):
            if not right_semi and step.coalesce and i in right_keys:
                continue
            columns.append(_build_series(table, right._columns[i], pointers))
        for c in range(len(columns)):
            columns[c] = columns[c].renamed(schema._columns[c].name())
        return DataFrame(columns^, height=n)


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
    """Derive a scan bound from the leading top-N key: strict inequality for
    one key, or inclusive inequality when later keys must settle ties. Do
    not push nulls-first keys because statistics cannot retain null rows
    separately from rejected values. Categorical storage codes also cannot
    supply label-order bounds."""
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
    return _bitmap_word[DType.uint64](
        Span(
            unsafe_ptr=Pointer[UInt8, MutAnyOrigin](
                unsafe_from_address=address
            ),
            length=count,
        ),
        bit,
        n,
    )


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


def _morsel_ranges(frame: DataFrame, workers: Int) -> List[Int]:
    """(offset, length) pairs covering the frame, each aligned chunk of its
    columns cut in pieces of `morsel_size`."""
    var ends = List[Int]()
    if can_filter_aligned_chunks(frame._columns):
        ends = frame._columns[0]._chunked.value()[].ends.copy()
    else:
        ends.append(frame.height())
    return morsel_ranges(ends, max(1, workers))


def run_pipeline(
    frame: DataFrame,
    steps: List[Step],
    joins: ArcPointer[List[DataFrame]],
    indexes: ArcPointer[List[Optional[JoinHashTable]]],
    expressions: List[Expr],
    keys: List[String],
    ordered: Bool,
    workers: Int,
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
    finish_steps: List[Step] = List[Step](),
    pushed: List[PushedJoinFilter] = List[PushedJoinFilter](),
) raises -> DataFrame:
    """Run the steps over `frame` in morsels on `workers` threads, into a
    materialize sink, or a reduce sink when `expressions` are reductions
    over the steps' output, grouped by `keys` when given (`ordered`: groups
    in first-occurrence order; `table`: into the aggregate table, when it
    holds every state, else collected for the eager group-by; `sets`: grouping sets, each a
    list of indices into `keys`, one sink per set fed by every morsel,
    whose finished frames (set order) land in `set_frames`, the grand
    total's among them with its one row, and the result is the steps'
    empty output). `finish_steps` are row-local steps (no
    joins) over the reduce sink's output, run where it is made: on each
    worker finishing a part of the aggregate table, else as a pipeline
    over the output. `pushed` are join keys from above an aggregate these
    steps feed, for the source's join filters (`_join_filters`).
    `counts` receives the rows fed, then each step's input and output
    rows, then each finish step's."""
    var run = begin_pipeline(
        frame,
        steps,
        joins,
        indexes,
        expressions,
        keys,
        ordered,
        workers,
        batch_size,
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
        finish_steps,
        pushed,
    )
    try:
        run.execute_all()
    except e:
        run.release()
        raise e^
    return finish_pipeline(run, counts, set_frames)


def finish_pipeline(
    mut run: PipelineRun,
    mut counts: List[Int],
    mut set_frames: List[DataFrame],
) raises -> DataFrame:
    """`run`'s result once its execute tasks are done; `counts` and
    `set_frames` as in `run_pipeline`."""
    var finish_counts = List[Int]()
    var result = _finish_pipeline(run, counts, set_frames, finish_counts)
    if len(run.finish_steps) > 0 and len(finish_counts) == 0:
        # The sink finished the output whole: the steps run over it as
        # a pipeline of their own, morsels on every worker.
        var inner = List[Int]()
        var none = List[DataFrame]()
        var second = begin_pipeline(
            result,
            run.finish_steps,
            run.joins,
            run.indexes,
            List[Expr](),
            List[String](),
            False,
            run.workers,
            run.batch_size,
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
            List[Step](),
            List[PushedJoinFilter](),
        )
        try:
            second.execute_all()
        except e:
            second.release()
            raise e^
        result = _finish_pipeline(second, inner, none, finish_counts)
        for k in range(len(run.finish_steps)):
            finish_counts.append(inner[2 * k + 1])
            finish_counts.append(inner[2 * k + 2])
    counts.extend(Span(finish_counts))
    return result^


struct _FinishStepsJob(Job):
    """Run row-local steps above an aggregate on the worker finishing its
    groups. Apply filters before assembling the complete output, so rejected
    groups are neither materialized nor scanned again."""

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


struct PipelineRun(Movable):
    """A morsel pipeline between `begin_pipeline` and `finish_pipeline`:
    its workers' jobs (the execute tasks `Pipeline::Schedule` makes, one
    per thread) and what its finish reads. Lives on one thread at each
    stage but its execute, where each task has its own job."""

    var jobs: List[_PipelineJob]
    var cursor: _Cursor
    var boundary: TopNBoundary
    var frame: DataFrame
    var steps: List[Step]
    var planned: Int
    var joins: ArcPointer[List[DataFrame]]
    var indexes: ArcPointer[List[Optional[JoinHashTable]]]
    var expressions: List[Expr]
    var keys: List[String]
    var ordered: Bool
    var workers: Int
    var batch_size: Int
    var sets: List[List[Int]]
    var top: Int
    var top_names: List[String]
    var top_descending: List[Bool]
    var top_nulls_last: List[Bool]
    var top_maintain_order: Bool
    var layout: List[AggregateLayout]
    var template: List[_StreamReduction]
    var set_empty: List[DataFrame]
    var derived_from: List[Int]
    var schemas: ArcPointer[List[DataFrame]]
    var reduce_layout: List[ArcPointer[_ReduceSinkLayout]]
    var collects: Bool
    var ungrouped: List[ArcPointer[UngroupedLayout]]
    var finish_steps: List[Step]
    var released: Bool
    # A right or full join's unmatched build rows (`begin_round`): the
    # join steps that emit them, the next to run, the morsel number the
    # next round's start from, and the running round's cursor.
    var outer_steps: List[Int]
    var next_round: Int
    var next_morsel: Int
    var round_cursor: Optional[_Cursor]

    def __init__(
        out self,
        var jobs: List[_PipelineJob],
        cursor: _Cursor,
        boundary: TopNBoundary,
        var frame: DataFrame,
        var steps: List[Step],
        planned: Int,
        joins: ArcPointer[List[DataFrame]],
        indexes: ArcPointer[List[Optional[JoinHashTable]]],
        var expressions: List[Expr],
        var keys: List[String],
        ordered: Bool,
        workers: Int,
        batch_size: Int,
        var sets: List[List[Int]],
        top: Int,
        var top_names: List[String],
        var top_descending: List[Bool],
        var top_nulls_last: List[Bool],
        top_maintain_order: Bool,
        var layout: List[AggregateLayout],
        var template: List[_StreamReduction],
        var set_empty: List[DataFrame],
        var derived_from: List[Int],
        schemas: ArcPointer[List[DataFrame]],
        var reduce_layout: List[ArcPointer[_ReduceSinkLayout]],
        collects: Bool,
        var ungrouped: List[ArcPointer[UngroupedLayout]],
        var finish_steps: List[Step],
    ):
        self.jobs = jobs^
        self.cursor = cursor.copy()
        self.boundary = boundary
        self.frame = frame^
        self.steps = steps^
        self.planned = planned
        self.joins = joins
        self.indexes = indexes
        self.expressions = expressions^
        self.keys = keys^
        self.ordered = ordered
        self.workers = workers
        self.batch_size = batch_size
        self.sets = sets^
        self.top = top
        self.top_names = top_names^
        self.top_descending = top_descending^
        self.top_nulls_last = top_nulls_last^
        self.top_maintain_order = top_maintain_order
        self.layout = layout^
        self.template = template^
        self.set_empty = set_empty^
        self.derived_from = derived_from^
        self.schemas = schemas
        self.reduce_layout = reduce_layout^
        self.collects = collects
        self.ungrouped = ungrouped^
        self.finish_steps = finish_steps^
        self.released = False
        self.outer_steps = List[Int]()
        for j in range(len(self.steps)):
            if self.steps[j].kind == STEP_JOIN and (
                join_emits_build_rows(self.steps[j].how)
            ):
                self.outer_steps.append(j)
        self.next_round = 0
        self.next_morsel = len(self.jobs[0].ranges[]) // 2
        self.round_cursor = None

    def tasks(self) -> Int:
        return len(self.jobs)

    def execute(mut self, task: Int) raises:
        self.jobs[task].run()

    def execute_all(mut self) raises:
        run_jobs(self.jobs)

    def begin_round(mut self) raises -> Bool:
        """After probing, partition the build table's rows among workers and
        emit the required matched/unmatched rows. Fill absent probe columns
        with nulls and run the same downstream steps and sinks as ordinary
        join output. Scheduling places this scan after probe completion so
        match flags are final."""
        self.end_round()
        while self.next_round < len(self.outer_steps):
            var j = self.outer_steps[self.next_round]
            var last = self.outer_steps[len(self.outer_steps) - 1]
            self.next_round += 1
            ref step = self.steps[j]
            # One morsel per chunk of the table's rows (ScanFullOuter).
            var ranges = List[Int]()
            for c in range(self.indexes[][step.join].value().chunk_count()):
                ranges.append(c)
                ranges.append(1)
            if len(ranges) == 0 and j != last:
                continue
            var cursor = _Cursor.new()
            var shared_build = ArcPointer(self.joins[][step.join].copy())
            var morsels = len(ranges) // 2
            var shared_ranges = ArcPointer(ranges^)
            var shared_probe = ArcPointer(self.jobs[0].runner.schema(j))
            for w in range(len(self.jobs)):
                ref job = self.jobs[w]
                job.frame = shared_build
                job.ranges = shared_ranges
                job.cursor = cursor.address
                job.first_step = j + 1
                job.first_morsel = self.next_morsel
                job.finishes = j == last
                job.unmatched_step = j
                job.unmatched_probe = shared_probe
            self.next_morsel += morsels
            self.round_cursor = cursor^
            return True
        return False

    def end_round(mut self):
        if self.round_cursor:
            self.round_cursor.value().free()
            self.round_cursor = None

    def release(mut self):
        """Free the workers' shared cursors and top-N boundary; idempotent,
        for a run that failed before its finish."""
        if self.released:
            return
        self.released = True
        self.end_round()
        self.cursor.free()
        self.boundary.free()


def begin_pipeline(
    frame: DataFrame,
    given_steps: List[Step],
    joins: ArcPointer[List[DataFrame]],
    indexes: ArcPointer[List[Optional[JoinHashTable]]],
    given_expressions: List[Expr],
    keys: List[String],
    ordered: Bool,
    workers: Int,
    batch_size: Int,
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
    finish_steps: List[Step],
    pushed: List[PushedJoinFilter],
) raises -> PipelineRun:
    """`run_pipeline` up to its workers' jobs, not yet run: its execute
    stage is the jobs, as `Pipeline::Schedule` makes one task per thread;
    `finish_pipeline` is its finish, after the rounds of `begin_round`."""
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
            var made = aggregate_layout(
                empty_keys,
                table_inputs(empty, empty_keys, expressions),
                table_specs(reduction),
            )
            # Bounded source key statistics can select direct-address
            # aggregation. Use those bounds only when every grouping key has a
            # compatible estimate.
            var stats = _key_statistics(frame, steps, keys)
            if len(stats) == len(keys):
                made.use_perfect_hash(stats)
            layout.append(made^)
            template.append(reduction^)
    if len(layout) > 0:
        trace_path("pipeline.group_table")
    elif len(keys) > 0 and len(expressions) > 0:
        trace_path("pipeline.group_collect")
    var shared = ArcPointer(frame.copy())
    # Visit top-N row groups in statistics order, assigning one group at a
    # time to each available worker so later groups can use a tighter
    # boundary.
    var ranges = ArcPointer(
        top_ranges.value().copy() if top_ranges else _morsel_ranges(
            frame, workers
        )
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
        # A state by address links the states it merges into a chain
        # through their rows, which a derived set's merge would change in
        # its source set's rows.
        for slot in layout[0].slots:
            if slot.by_address():
                merges = False
        if merges:
            # A set holding another's keys has more of them.
            var sizes = List[Int](capacity=len(sets))
            for t in range(len(sets)):
                sizes.append(len(sets[t]))
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
            and join_emits_build_rows(step.how)
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
    var shared_steps = ArcPointer(steps.copy())
    for _ in range(job_count):
        jobs.append(
            _PipelineJob(
                shared,
                cursor.address,
                ranges,
                shared_steps,
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
    # last step's chunks.
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
                derived_from,
                derived_columns,
                derive_order,
            )
        )
        for w in range(len(jobs)):
            jobs[w].runner.sets_layout.append(sets_layout)
            jobs[w].runner.table_workers = workers
    # A grouping the aggregate table does not take: the rows are collected
    # (each worker's chunks copied into its collections), then grouped by
    # the eager group-by, which evaluates any expression over the groups
    # the table finds. Any other ungrouped reduction: each worker's reduce
    # sink takes the chunks.
    var collects = len(expressions) > 0 and len(keys) > 0 and len(layout) == 0
    var reduce_layout = List[ArcPointer[_ReduceSinkLayout]]()
    if (
        len(expressions) > 0
        and not collects
        and len(layout) == 0
        and len(ungrouped) == 0
    ):
        reduce_layout.append(
            ArcPointer(
                _ReduceSinkLayout(schemas[][len(steps)].clear(), expressions)
            )
        )
        for w in range(len(jobs)):
            jobs[w].runner.reduce_layout.append(reduce_layout[0])
    # No reduction and no top-N: the rows are the result, each worker's
    # chunks copied into its collections (PhysicalBatchCollector).
    if (len(expressions) == 0 and top <= 0) or collects:
        for w in range(len(jobs)):
            jobs[w].runner.materializes = True
    if scan:
        var shared_scan = ArcPointer(scan.take())
        for w in range(len(jobs)):
            jobs[w].runner.scan = shared_scan
    # An outer join's unmatched build rows run after every morsel of the
    # source (`begin_round`), and the workers finish after them.
    for step in steps:
        if step.kind == STEP_JOIN and join_emits_build_rows(step.how):
            for w in range(len(jobs)):
                jobs[w].finishes = False
    return PipelineRun(
        jobs^,
        cursor.copy(),
        boundary,
        frame.clear(),
        steps^,
        len(given_steps),
        joins,
        indexes,
        expressions^,
        keys.copy(),
        ordered,
        workers,
        batch_size,
        sets.copy(),
        top,
        top_names.copy(),
        top_descending.copy(),
        top_nulls_last.copy(),
        top_maintain_order,
        layout^,
        template^,
        set_empty^,
        derived_from^,
        schemas,
        reduce_layout^,
        collects,
        ungrouped^,
        finish_steps.copy(),
    )


def _finish_pipeline(
    mut run: PipelineRun,
    mut counts: List[Int],
    mut set_frames: List[DataFrame],
    mut finish_counts: List[Int],
) raises -> DataFrame:
    """The sink's result, after every execute task of `run`: the rows no
    probe matched (`begin_round`), then the workers' sinks combined and
    finished. `counts` and `set_frames` are `run_pipeline`'s; `finish_counts`
    receives each finish step's input and output rows when the aggregate
    table's finish ran them, and stays empty otherwise."""
    ref jobs = run.jobs
    ref steps = run.steps
    ref joins = run.joins
    ref indexes = run.indexes
    ref expressions = run.expressions
    ref keys = run.keys
    var ordered = run.ordered
    var batch_size = run.batch_size
    ref sets = run.sets
    var top = run.top
    ref top_names = run.top_names
    ref top_descending = run.top_descending
    ref top_nulls_last = run.top_nulls_last
    var top_maintain_order = run.top_maintain_order
    ref layout = run.layout
    ref template = run.template
    ref set_empty = run.set_empty
    ref derived_from = run.derived_from
    ref schemas = run.schemas
    ref reduce_layout = run.reduce_layout
    var collects = run.collects
    ref ungrouped = run.ungrouped
    ref finish_steps = run.finish_steps
    ref frame = run.frame
    var planned = run.planned
    # The rounds no schedule ran (`begin_round`).
    while run.begin_round():
        run.execute_all()
    run.release()
    counts = List[Int](length=3 * len(steps) + 1, fill=0)
    for j in range(len(jobs)):
        for i in range(len(counts)):
            counts[i] += jobs[j].counts[i]
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
            # Run an initial filter on reduction outputs during aggregate
            # finalization, before rejected groups enter output columns. Apply
            # remaining row steps only to surviving groups.
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
    if collects:
        var locals = List[MaterializeSink]()
        for j in range(len(jobs)):
            if len(jobs[j].runner.materialize) > 0:
                locals.append(jobs[j].runner.materialize.pop())
        var combined = _combined(locals^)
        var whole = materialize_finish(combined, schemas[][len(steps)].clear())
        return whole.group_by(keys, maintain_order=ordered).agg(expressions)
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
    # A top-N's workers' heaps, combined into one, sorted (Finalize).
    var locals = List[TopNSink]()
    for j in range(len(jobs)):
        if len(jobs[j].topn) > 0:
            locals.append(jobs[j].topn.pop())
    var heaps = _combined(locals^)
    if len(heaps) == 0 or heaps[0].heap.size == 0:
        var empty = schemas[][len(steps)].copy()
        return empty^
    var merged = heaps[0].finish()
    if top_maintain_order:
        merged = merged.drop([_ORDER_COLUMN])
    return merged^
