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
from .expr import (
    ARG_MAX,
    ARG_MIN,
    COL,
    FIRST,
    IMPLODE,
    LAST,
    SELECTOR,
    WHEN,
    Expr,
)
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
    AggregateFinish,
    AggregateResult,
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
    UngroupedFinish,
    UngroupedSink,
    ungrouped_layout,
)
from dataframe.sorting.top_n import TopNBoundary
from dataframe.sorting.top_n_sink import TopNSink
from dataframe.vector.column_data_collection import (
    LimitBound,
    MaterializeFinish,
    MaterializeSink,
    append_raw,
    materialize_finish,
    raw_series,
)
from dataframe.vector.sink import Sink
from dataframe.vector.selection_batches import SelectionBatch, SelectionBatches
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
from dataframe.io.scan_source import ScanSource
from dataframe.aggregation.state import (
    _StreamReduction,
    PartsFinish,
    chunks_frame,
    table_parts,
    table_inputs,
    table_specs,
    table_supported,
    _equality_words,
    _word_hash,
)
from dataframe.aggregation.keys import (
    _expand_struct_keys,
    _group_keys,
)
from dataframe.aggregation.selected_rows import _reduce_floats, _reduce_ints
from dataframe.joins.hash_table import (
    JoinBuildLayout,
    JoinBuildSink,
    JoinHashTable,
    ScanStructure,
    join_build_layout,
)
from dataframe.joins.join_filter import (
    BF_CHECK_N,
    BF_THRESHOLD,
    JoinFilter,
    PushedJoinFilter,
    SelectivityStats,
    finalize_filters,
    finalize_table_filters,
    keep_rows,
    supports_join_filters,
)
from dataframe.table_filter import TableFilterSet
from .join_type import (
    JOIN_ANTI,
    JOIN_CROSS,
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
from .parallel import Job, Rounds, morsel_ranges, run_jobs, source_tasks
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
from dataframe.aggregation.expressions import (
    _bind_all,
    _row_local,
    aggregate_filter,
    without_filters,
)
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
# An explode of list columns (`names`) or an unnest of a struct column
# (`names[0]`): rows multiply, or a column becomes its fields, so each
# morsel's rows go through the operator whole and the next steps cut
# chunks from its output, as they do from a join's.
comptime STEP_EXPLODE = 5
comptime STEP_UNNEST = 6
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


def _frame_step(kind: Int) -> Bool:
    """Whether a step reads a morsel's rows whole (a join, an explode or
    an unnest) rather than one chunk at a time."""
    return kind == STEP_JOIN or kind == STEP_EXPLODE or kind == STEP_UNNEST


def _checks_every_row(steps: List[Step]) -> Bool:
    """Whether a step raises over rows a limit would not keep: an explode of
    several list columns requires each row's lists to have the same element
    count, so every row reaches it, as the operator over the whole frame
    checks them all."""
    for step in steps:
        if step.kind == STEP_EXPLODE and len(step.names) > 1:
            return True
    return False


def _cross_step(step: Step) -> Bool:
    """Whether a step is a cross product: every probe row paired with
    every row of its collected build."""
    return step.kind == STEP_JOIN and step.how == JOIN_CROSS


def _order_dependent(expressions: List[Expr]) -> Bool:
    """Whether a reduction's answer depends on the order its rows come in:
    a first or last, the position of a minimum or maximum, a list of the
    values."""
    for expression in expressions:
        for node in expression._nodes:
            if node.op in [FIRST, LAST, ARG_MIN, ARG_MAX, IMPLODE]:
                return True
    return False


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
        elif step.kind == STEP_UNNEST:
            # A struct's fields may take the names of other columns.
            return List[Series]()
        elif (
            step.kind == STEP_DROP
            # An exploded column holds its lists' elements, which the
            # source's statistics do not describe.
            or step.kind == STEP_EXPLODE
            or (
                step.kind == STEP_JOIN
                and (step.how == JOIN_RIGHT or step.how == JOIN_FULL)
            )
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
            # A conditional aggregate input with no otherwise projects only
            # its condition, which every row evaluates. Its value stays in
            # the reduction: the sink computes it at the rows the condition
            # keeps, so no other row is evaluated or can raise.
            # An aggregate filter projects its values and its predicate;
            # the reduction keeps the filter over those columns.
            var targets = List[Int]()
            for child in node_children(nodes[i]):
                if nodes[child].op == WHEN and nodes[child].extra < 0:
                    targets.append(nodes[child].left)
                elif aggregate_filter(e, child):
                    for part in node_children(nodes[child]):
                        targets.append(part)
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
    probe_rows: Int,
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
                    var made = finalize_table_filters(
                        name,
                        frame.column(name).dtype(),
                        table,
                        k,
                        at,
                        len(step.names),
                        probe_rows,
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
        elif step.kind == STEP_EXPLODE:
            # Each row repeats once per element, the other columns
            # unchanged: only the exploded ones stop being the source's.
            for name in step.names:
                if name in source:
                    _ = source.pop(name)
        elif step.kind == STEP_UNNEST:
            # A struct's fields may take the names of source columns.
            source = Dict[String, String]()
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
            probe_rows,
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

    def through_rows(self, var rows: List[UInt32]) -> _Columns:
        """These columns (each read whole from row 0) read through one
        selection, `rows` (row numbers of the columns): each column a
        dictionary over its values sharing that one row list, so the rows a
        filter kept travel on with nothing copied."""
        var result = _Columns()
        for c in range(self.width()):
            result.handles.append(self.handles[c])
            result.starts.append(0)
            result.lists.append(0)
        result.height = len(rows)
        result.row_lists.append(rows^)
        return result^

    def width(self) -> Int:
        return len(self.handles)

    def through(self, rows: List[Int]) -> _Columns:
        """Every column read at its rows `rows` (positions among this one's
        `height`), the dictionaries merged: a row list over a row list is one
        row list over the column, so a read follows one indirection. Columns
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
            if v.vector_type == CONSTANT_VECTOR:
                # Every row of a constant column is its one value.
                pass
            elif self.lists[c] < 0:
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


struct _CrossAt(Copyable, Movable):
    """Where a cross product's pieces over one morsel have got to: the
    probe chunk and the build chunk starting at rows `probe` and `build`,
    and the row of the smaller of the two that the next piece pairs with
    the larger."""

    var probe: Int
    var build: Int
    var row: Int

    def __init__(out self):
        self.probe = 0
        self.build = 0
        self.row = 0


struct _Collector(Movable):
    """A morsel's chunks gathered into `_Columns` for a join step, which
    reads frames (the adapter the join port deletes): a
    column whose chunks read one arena column is kept as that column's
    range or row list, and only values a step computed are copied.

    Columns sliced by one filter are dictionaries through the same
    selection, so they keep the same rows: such a column follows the first
    column whose rows it shares (its leader) and lists nothing itself, and
    every column of the group reads the leader's one row list. A follower
    lists its own rows from the first chunk whose rows differ."""

    var schema: DataFrame
    var modes: List[Int]
    var handles: List[Int]
    var starts: List[Int]
    var rows: List[List[UInt32]]
    var parts: List[Series]
    var raw: List[List[UInt8]]
    var raw_bits: List[List[UInt8]]
    var raw_physical: List[Int]
    # Per column, the column whose rows it shares (-1: its own).
    var leaders: List[Int]
    var followers: Int
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
        self.leaders = List[Int](length=width, fill=-1)
        self.followers = 0
        self.height = 0

    def _listed(self, c: Int, v: Vector) -> Bool:
        """Whether column `c` keeps vector `v` as rows of its arena column."""
        var mode = self.modes[c]
        return (
            v.source >= 0
            and v.vector_type != SEQUENCE_VECTOR
            and (
                mode == _COLLECT_NONE
                or (mode != _COLLECT_VALUES and self.handles[c] == v.source)
            )
        )

    @staticmethod
    def _same_rows(a: Vector, b: Vector) -> Bool:
        """Whether two vectors read the same rows of their columns: the same
        window, through the same selection when they are dictionaries."""
        return (
            a.vector_type == b.vector_type
            and a.source_offset == b.source_offset
            and (a.vector_type != DICTIONARY_VECTOR or a.sel.data == b.sel.data)
        )

    def _pair(mut self, chunk: DataChunk):
        """On the first chunk, each listed column follows the first column
        before it that lists the same rows."""
        for c in range(len(self.modes)):
            if not self._listed(c, chunk.data[c]):
                continue
            for p in range(c):
                if (
                    self.leaders[p] < 0
                    and self._listed(p, chunk.data[p])
                    and Self._same_rows(chunk.data[p], chunk.data[c])
                ):
                    self.leaders[c] = p
                    self.followers += 1
                    break

    def _detach(mut self, c: Int):
        """Column `c` lists its own rows from here on: its leader's so far."""
        var p = self.leaders[c]
        self.leaders[c] = -1
        self.followers -= 1
        self.modes[c] = self.modes[p]
        self.starts[c] = self.starts[p]
        if self.modes[p] == _COLLECT_ROWS:
            # Before this chunk: the leader lists `height` rows.
            self.rows[c] = self.rows[p].copy()

    def append(mut self, mut chunk: DataChunk, mut arena: ColumnArena) raises:
        var count = chunk.count
        if self.height == 0 and self.followers == 0:
            self._pair(chunk)
        if self.followers > 0:
            for c in range(len(self.modes)):
                var p = self.leaders[c]
                if p >= 0 and not (
                    self._listed(c, chunk.data[c])
                    and self._listed(p, chunk.data[p])
                    and Self._same_rows(chunk.data[p], chunk.data[c])
                ):
                    self._detach(c)
        for c in range(len(self.modes)):
            var v = chunk.data[c]
            var mode = self.modes[c]
            if self.leaders[c] >= 0:
                # Its leader lists these rows.
                self.handles[c] = v.source
                continue
            if self._listed(c, v):
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
        if self.followers > 0:
            for c in range(len(self.modes)):
                var p = self.leaders[c]
                if p >= 0:
                    self.modes[c] = self.modes[p]
                    self.starts[c] = self.starts[p]
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
            var p = self.leaders[c]
            if p >= 0 and mode == _COLLECT_ROWS:
                # The leader's row list, made before this column's.
                result.handles.append(self.handles[c])
                result.starts.append(0)
                result.lists.append(result.lists[p])
                continue
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
        elif _cross_step(step):
            current = current.join(
                joins[step.join].clear(), how="cross", suffix=step.suffix
            )
        elif step.kind == STEP_EXPLODE:
            current = current.explode(step.names)
        elif step.kind == STEP_UNNEST:
            current = current.unnest(step.names[0])
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
        if _frame_step(steps[s].kind):
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
    # A materialize sink (`MaterializeSink`), when the pipeline's rows are
    # its result, or the input of a reduction no aggregate sink takes.
    var materialize: List[MaterializeSink]
    var materializes: Bool
    # A provider that reads row mappings receives the collector's owned
    # sources and selections at each morsel boundary.
    var retains: Bool
    var retained: List[_Collector]
    var retained_batches: List[SelectionBatch]
    var retained_morsel: Int
    # A limited materialize sink: the first rows in source order it keeps
    # (-1: every row), and the last morsel any worker still needs.
    var limit: Int
    var limit_bound: LimitBound
    # A join build sink (`JoinBuildSink`), when the pipeline's rows are a
    # hash join's build side: the chunks go into the worker's own rows,
    # which become the join's table, and nothing is collected.
    var build: List[JoinBuildSink]
    var build_layout: List[ArcPointer[JoinBuildLayout]]
    # Retain scan-filter state and its source offset while the morsel runs.
    # Statistics can skip a vector before any values are read; -1 denotes no
    # source offset.
    var scan: Optional[ArcPointer[ScanFilterInfo]]
    var scan_offset: Int
    # Whether the sink's answer depends on the order its rows come in (a
    # first or last, the position of an extreme, groups in first-appearance
    # order): a cross product then makes its pairs in the join's order, each
    # probe row with every build row in turn (`cross_piece`).
    var ordered_pairs: Bool

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
        self.materialize = List[MaterializeSink]()
        self.materializes = False
        self.retains = False
        self.retained = List[_Collector]()
        self.retained_batches = List[SelectionBatch]()
        self.retained_morsel = 0
        self.limit = -1
        self.limit_bound = LimitBound.none()
        self.build = List[JoinBuildSink]()
        self.build_layout = List[ArcPointer[JoinBuildLayout]]()
        self.scan = None
        self.scan_offset = -1
        self.ordered_pairs = False

    def begin_morsel(mut self) raises:
        """Drop the previous morsel's columns. At a worker's first morsel,
        hold the join builds' columns, on its own thread: a worker that
        claims no morsel never does."""
        if not self.bound:
            self.bound = True
            self.builds = List[List[Int]](capacity=len(self.steps[]))
            for step in self.steps[]:
                var handles = List[Int]()
                if _cross_step(step):
                    # The build's columns, which every piece of the cross
                    # product reads in place (`cross_piece`).
                    for column in self.joins[][step.join]._columns:
                        handles.append(self.arena.add(column))
                elif step.kind == STEP_JOIN:
                    if (
                        self.indexes[][step.join]
                        and self.indexes[][step.join].value().use_dictionary
                    ):
                        # The table's dictionary columns (perfect or
                        # dictionary emission), one per output column,
                        # decoded once by the table.
                        ref table = self.indexes[][step.join].value()
                        ref dictionary = table.data[].dictionary
                        for c in range(len(dictionary)):
                            handles.append(
                                self.arena.add_decoded(
                                    dictionary[c], table.dictionary_values(c)
                                )
                            )
                self.builds.append(handles^)
            self.held = self.arena.mark()
        self.arena.truncate(self.held)

    def sinks_chunks(self) -> Bool:
        """Whether the worker's sink state is made (`begin_sink`): the
        last step's chunks go into it, and nothing is collected."""
        return (
            len(self.ungrouped) > 0
            or len(self.table) > 0
            or len(self.sets) > 0
            or len(self.materialize) > 0
            or len(self.build) > 0
            or len(self.retained) > 0
        )

    def begin_sink(
        mut self, morsel: Int, offset: Int = 0, length: Int = 0
    ) raises:
        """Morsel `morsel` (source rows [offset, offset + length)) comes next:
        the worker's local sink state, made on its own thread at its first
        morsel (so its memory is local to that thread), numbers its rows from
        the morsel's."""
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
        elif len(self.build_layout) > 0:
            if len(self.build) == 0:
                self.build.append(JoinBuildSink(self.build_layout[0]))
            self.build[0].begin(morsel)
        elif self.retains:
            self._retain_morsel()
            self.retained_morsel = morsel
            self.retained.append(_Collector(self.output_schema(), length))
        elif self.materializes:
            if len(self.materialize) == 0:
                self.materialize.append(
                    MaterializeSink(
                        self.output_schema().clear(),
                        limit=self.limit,
                        bound=self.limit_bound,
                    )
                )
            self.materialize[0].begin(morsel)

    def sink_full(self) -> Bool:
        """Whether the worker's limited sink holds all the rows it can use:
        nothing it would sink from here on is wanted."""
        return len(self.materialize) > 0 and self.materialize[0].full()

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
        if len(self.materialize) > 0:
            self.materialize[0].release()
        if len(self.build) > 0:
            self.build[0].release()
        self._retain_morsel()

    def _retain_morsel(mut self) raises:
        """Own the last morsel before its arena or chunk caches are reused."""
        if len(self.retained) == 0:
            return
        var collector = self.retained.pop()
        if collector.height == 0:
            return
        var batch = SelectionBatch()
        batch.morsel = self.retained_morsel
        for mode in collector.modes:
            batch.copied_columns += Int(
                mode == _COLLECT_RAW or mode == _COLLECT_VALUES
            )
        var columns = collector.finish(self.arena)
        batch.count = columns.height
        for c in range(columns.width()):
            batch.columns.append(
                self.arena.columns[columns.handles[c]].renamed(
                    collector.schema._columns[c].name()
                )
            )
        swap(batch.starts, columns.starts)
        swap(batch.lists, columns.lists)
        swap(batch.rows, columns.row_lists)
        self.retained_batches.append(batch^)

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
        elif len(self.build) > 0:
            _sink_chunk(self.build[0], chunk, self.arena)
        elif len(self.retained) > 0:
            # A join can switch source columns within one morsel. Finish the
            # current references before the collector would gather them to
            # append values from a different source.
            var changes_source = False
            for c in range(len(chunk.data)):
                var mode = self.retained[0].modes[c]
                if mode == _COLLECT_RANGE or mode == _COLLECT_ROWS:
                    changes_source |= not self.retained[0]._listed(
                        c, chunk.data[c]
                    )
                elif mode == _COLLECT_RAW or mode == _COLLECT_VALUES:
                    changes_source |= chunk.data[c].source >= 0
            if changes_source:
                self._retain_morsel()
                self.retained.append(_Collector(self.output_schema()))
            self.retained[0].append(chunk, self.arena)
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
        count when the last step reads its rows whole). The segment then
        runs one chunk at a time through `chunk`, so a sink takes each chunk
        as it is made. When the last such step is a cross product, the
        columns are its probe rows and the step returned is the cross
        product's: its pieces (`cross_piece`) are the chunks the steps after
        it take."""
        var n = len(self.steps[])
        var last = n
        while last > first and not _frame_step(self.steps[][last - 1].kind):
            last -= 1
        var cross = last > first and _cross_step(self.steps[][last - 1])
        if cross:
            last -= 1
        if last > first:
            columns = self.run_until(columns^, first, last)
        for s in range(last + 1 if cross else last, n):
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
            var kind = self.steps[][at].kind
            if (
                _cross_step(self.steps[][at])
                and stop == n
                and self.sinks_chunks()
            ):
                # The product is never held: each piece runs through the
                # steps after it into the sink.
                self._cross_into_sink(columns, at)
                return _Columns()
            if _frame_step(kind):
                self.counts[2 * at + 1] += columns.height
                var began = Int(perf_counter_ns())
                if kind == STEP_JOIN:
                    columns = self._join(columns, at)
                else:
                    columns = self._reshape(columns, at)
                self.counts[2 * n + 1 + at] += Int(perf_counter_ns()) - began
                self.counts[2 * at + 2] += columns.height
                at += 1
                continue
            var end = at
            while end < stop and not _frame_step(self.steps[][end].kind):
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

    def _reshape(mut self, columns: _Columns, s: Int) raises -> _Columns:
        """An explode or unnest step over a morsel's collected columns: the
        operator over them as a frame, its output the columns the next
        steps read."""
        ref step = self.steps[][s]
        var frame = columns.whole(self.arena, self.schemas[][s])
        if step.kind == STEP_EXPLODE:
            frame = frame.explode(step.names)
        else:
            frame = frame.unnest(step.names[0])
        return _Columns.of_frame(self.arena, frame)

    def _cross_into_sink(mut self, columns: _Columns, s: Int) raises:
        """Cross product step `s` over a morsel's probe rows, a piece at a
        time through the steps after it into the sink."""
        var n = len(self.steps[])
        self.counts[2 * s + 1] += columns.height
        var at = _CrossAt()
        while True:
            var began = Int(perf_counter_ns())
            var piece = self.cross_piece(columns, s, at)
            self.counts[2 * n + 1 + s] += Int(perf_counter_ns()) - began
            if not piece:
                return
            self.counts[2 * s + 2] += piece.value().height
            _ = self.run_until(piece.take(), s + 1, n)
            if self.sink_full():
                return

    def cross_piece(
        self, columns: _Columns, s: Int, mut at: _CrossAt
    ) -> Optional[_Columns]:
        """The next piece of cross product step `s` over the probe rows
        `columns`, or None once every pair is made. The probe rows and the
        build rows are each taken a chunk (STANDARD_VECTOR_SIZE rows) at a
        time; for each pair of chunks, every row of the smaller one in turn
        is paired with the whole larger one, so a piece has as many rows as
        the larger chunk and reads both sides in place: the larger chunk as
        it is, the one row through a row list repeating it. The probe
        columns come first, then the build's. With `ordered_pairs` the
        pieces are the pairs in order instead (`_ordered_cross_piece`)."""
        var n = columns.height
        var m = self.joins[][self.steps[][s].join].height()
        if n == 0 or m == 0:
            return None
        if self.ordered_pairs:
            return self._ordered_cross_piece(columns, s, at, n, m)
        while True:
            if at.probe >= n:
                return None
            var probe_count = min(STANDARD_VECTOR_SIZE, n - at.probe)
            var build_count = min(STANDARD_VECTOR_SIZE, m - at.build)
            # The smaller chunk is the one read a row at a time.
            var probe_rows = probe_count < build_count
            if at.row >= (probe_count if probe_rows else build_count):
                at.row = 0
                at.build += STANDARD_VECTOR_SIZE
                if at.build >= m:
                    at.build = 0
                    at.probe += STANDARD_VECTOR_SIZE
                continue
            var piece = _Columns()
            piece.height = build_count if probe_rows else probe_count
            ref handles = self.builds[s]
            if probe_rows:
                var probe = at.probe + at.row
                # Probe columns reading the same at-rest row share a list.
                var made_rows = List[Int]()
                var made_lists = List[Int]()
                for c in range(columns.width()):
                    var row = columns.starts[c] + probe
                    if columns.lists[c] >= 0:
                        row = Int(columns.row_lists[columns.lists[c]][probe])
                    var list = -1
                    for k in range(len(made_rows)):
                        if made_rows[k] == row:
                            list = made_lists[k]
                    if list < 0:
                        list = len(piece.row_lists)
                        piece.row_lists.append(
                            List[UInt32](length=piece.height, fill=UInt32(row))
                        )
                        made_rows.append(row)
                        made_lists.append(list)
                    piece.handles.append(columns.handles[c])
                    piece.starts.append(0)
                    piece.lists.append(list)
                for c in range(len(handles)):
                    piece.handles.append(handles[c])
                    piece.starts.append(at.build)
                    piece.lists.append(-1)
            else:
                var build = at.build + at.row
                # Probe columns sharing a row list share its window.
                var from_lists = List[Int]()
                var to_lists = List[Int]()
                for c in range(columns.width()):
                    piece.handles.append(columns.handles[c])
                    var l = columns.lists[c]
                    if l < 0:
                        piece.starts.append(columns.starts[c] + at.probe)
                        piece.lists.append(-1)
                        continue
                    var list = -1
                    for k in range(len(from_lists)):
                        if from_lists[k] == l:
                            list = to_lists[k]
                    if list < 0:
                        ref rows = columns.row_lists[l]
                        var window = List[UInt32](capacity=piece.height)
                        for r in range(at.probe, at.probe + piece.height):
                            window.append(rows[r])
                        list = len(piece.row_lists)
                        piece.row_lists.append(window^)
                        from_lists.append(l)
                        to_lists.append(list)
                    piece.starts.append(0)
                    piece.lists.append(list)
                var repeated = len(piece.row_lists)
                piece.row_lists.append(
                    List[UInt32](length=piece.height, fill=UInt32(build))
                )
                for c in range(len(handles)):
                    piece.handles.append(handles[c])
                    piece.starts.append(0)
                    piece.lists.append(repeated)
            at.row += 1
            return piece^

    def _ordered_cross_piece(
        self, columns: _Columns, s: Int, mut at: _CrossAt, n: Int, m: Int
    ) -> Optional[_Columns]:
        """The next STANDARD_VECTOR_SIZE pairs of the cross product in its
        order (probe row by probe row, each with every build row in turn),
        pair `at.row` first: pair p is probe row p / m with build row p % m,
        both sides read through row lists."""
        if at.row >= n * m:
            return None
        var piece = _Columns()
        piece.height = min(STANDARD_VECTOR_SIZE, n * m - at.row)
        var first = at.row
        # Probe columns reading the same rows share a list: the same row
        # list, or the same start of an at-rest range.
        var made_lists = List[Int]()
        var made_starts = List[Int]()
        var made = List[Int]()
        for c in range(columns.width()):
            var l = columns.lists[c]
            var start = columns.starts[c] if l < 0 else 0
            var list = -1
            for k in range(len(made)):
                if made_lists[k] == l and made_starts[k] == start:
                    list = made[k]
            if list < 0:
                var rows = List[UInt32](capacity=piece.height)
                for p in range(first, first + piece.height):
                    var probe = p // m
                    if l >= 0:
                        rows.append(columns.row_lists[l][probe])
                    else:
                        rows.append(UInt32(start + probe))
                list = len(piece.row_lists)
                piece.row_lists.append(rows^)
                made_lists.append(l)
                made_starts.append(start)
                made.append(list)
            piece.handles.append(columns.handles[c])
            piece.starts.append(0)
            piece.lists.append(list)
        var builds = List[UInt32](capacity=piece.height)
        for p in range(first, first + piece.height):
            builds.append(UInt32(p % m))
        var list = len(piece.row_lists)
        piece.row_lists.append(builds^)
        ref handles = self.builds[s]
        for c in range(len(handles)):
            piece.handles.append(handles[c])
            piece.starts.append(0)
            piece.lists.append(list)
        at.row = first + piece.height
        return piece^

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
                if self.sink_full():
                    break
            else:
                collector[0].append(input[], self.arena)
        if sinking:
            return _Columns()
        return collector[0].finish(self.arena)

    def _join(mut self, columns: _Columns, s: Int) raises -> _Columns:
        """A join step over a morsel's collected columns: it probes the step's
        table (`ScanStructure`) with the key columns only and keeps the probe
        side a dictionary over where it is, with no copy. An inner join leaves
        the build side a dictionary over the build's columns when it can; a
        left, full or right join gathers it, and flags the build rows it matched
        for `begin_round`, as a right-semi or right-anti join does without
        emitting anything. A step with no table (nested keys) or whose keys need
        dictionaries unified joins the morsel's frame eagerly."""
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
        if _cross_step(step):
            # Where its output is collected for a later join: the morsel's
            # whole product.
            var frame = columns.whole(self.arena, input)
            var crossed = frame.join(right, how="cross", suffix=step.suffix)
            return _Columns.of_frame(self.arena, crossed)
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
        # Each key is probed where it is: its arena column read whole from
        # its start, or through its row list (rows an earlier filter or join
        # kept), so the keys are not gathered. A constant key, whose
        # column holds one value, is the exception: it is read as a column.
        var keys = List[Series](capacity=len(step.names))
        var key_starts = List[Int](capacity=len(step.names))
        var key_lists = List[Int](capacity=len(step.names))
        var in_place = True
        for name in step.names:
            var c = input._index(name)
            ref column = self.arena.columns[columns.handles[c]]
            in_place = in_place and not column.is_constant()
            keys.append(column.copy())
            key_starts.append(columns.starts[c])
            key_lists.append(columns.lists[c])
        if not in_place:
            keys = columns.frame(self.arena, input, step.names)._columns.copy()
            for k in range(len(keys)):
                if keys[k].is_chunked():
                    keys[k] = keys[k].rechunk()
        var chunk_starts = List[Int](length=len(keys), fill=0)
        var selections = List[Int](length=len(keys), fill=0)
        if in_place:
            for l in key_lists:
                if l >= 0:
                    trace_path("pipeline.join_probe_selection")
                    break
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
            if in_place:
                for k in range(len(keys)):
                    var l = key_lists[k]
                    if l >= 0:
                        selections[k] = (
                            Int(columns.row_lists[l].unsafe_ptr()) + 4 * low
                        )
                    else:
                        chunk_starts[k] = key_starts[k] + low
                scan.probe_selected(keys, chunk_starts, selections, count)
            else:
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
        # gathered from the matched rows, or the table's dictionaries (a
        # perfect join or dictionary emission).
        var list = -1
        for i in range(right.width()):
            if step.coalesce and i in right_keys:
                continue
            ref column = right._columns[i]
            var t = table.output_index(column.name())
            if t < 0:
                raise Error("pipeline join: a build column is not in the rows")
            if table.use_dictionary:
                if list < 0:
                    list = len(result.row_lists)
                    result.row_lists.append(dictionary_rows.copy())
                result.handles.append(self.builds[s][t])
                result.starts.append(0)
                result.lists.append(list)
            else:
                result.handles.append(
                    self.arena.add(
                        table.gather(
                            t,
                            Int(pointers.unsafe_ptr()),
                            len(pointers),
                            column.name(),
                        )
                    )
                )
                result.starts.append(0)
                result.lists.append(-1)
        if result.width() != output.width():
            raise Error("pipeline join: output columns differ from its schema")
        return result^


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
    """`steps` over all of `frame` on this thread, chunk by chunk, its
    rows copied out here too: it is one task's share of a parallel step.
    `counts` receives each step's input and output rows."""
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
        runner.materialize, runner.output_schema().clear(), serial=True
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
    # Top-N sink: this worker's heap over the output's sort keys (made at its
    # first chunk), the boundary every worker's full heap shares, and whether
    # ties keep the input order (the source place is then the last key).
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
    # The scan's row filters, run vector by vector in `scan_order`: per
    # filter the join filter it is (an index of `join_filters`), or -1 for
    # the source's own filter step when it runs among them (`predicate`; the
    # steps then start after it). Set at the first morsel (`_scan_filters`).
    var scan_entries: List[Int]
    var scan_order: List[Int]
    var scan_ready: Bool
    var predicate: Bool
    # The source's filter reads a vector through `scan_chunk`, its
    # selection rebased to the vector in `scan_local` (input, then output).
    var scan_chunk: DataChunk
    var scan_local: List[UInt32]
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
    # A file source's shared scan state: its batches are the morsels,
    # and `frame` and `ranges` are unused.
    var source: Optional[ArcPointer[ScanSource]]
    # Under a limit, the last morsel any worker still needs: a morsel
    # numbered past it is not read.
    var bound: LimitBound

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
        self.source = None
        self.bound = LimitBound.none()
        self.join_filters = join_filters.copy()
        self.join_stats = List[SelectivityStats](
            length=len(join_filters),
            fill=SelectivityStats.of(BF_CHECK_N, BF_THRESHOLD),
        )
        self.join_scan = join_scan
        self.scan_entries = List[Int]()
        self.scan_order = List[Int]()
        self.scan_ready = False
        self.predicate = False
        self.scan_chunk = DataChunk()
        self.scan_local = List[UInt32]()
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

    def _scan_filters(mut self) raises:
        """The scan's row filters and their first order: the join filters
        applied to rows, then the source's filter step when there is one
        and some join filter is applied to rows (alone it has nothing to be
        ordered against). The cheapest per row runs first, each costed by
        its work per row on the scale of `VectorExpression.cost`: comparing
        a key with one value or testing its bit in a key bitmap costs about a
        comparison, a bloom filter a hash and a probe; the source's filter is
        costed from its expression.

        The order is not adapted to measured time. Trying another order
        means running some vectors in a worse one until the measurement says
        so, and a worker's share of one scan is too few vectors for a better
        order found that way to win back what the trials cost."""
        self.scan_ready = True
        var costs = List[Int]()
        for k in range(len(self.join_filters)):
            ref f = self.join_filters[k]
            if f.equal or f.bitmap:
                self.scan_entries.append(k)
                # A key column read, then a comparison with a constant, or
                # a subtraction and one bit's test: about a comparison.
                costs.append(8 + 5 + 1)
            elif f.bloom:
                self.scan_entries.append(k)
                # A key column read, a multiplying hash and one probe of
                # the filter's bits.
                costs.append(8 + 10 + 5)
        if (
            len(self.scan_entries) > 0
            and len(self.steps[]) > 0
            and self.steps[][0].kind == STEP_FILTER
        ):
            self.runner._compiled(0)
            ref step = self.runner.compiled[0].value()
            if len(step.expressions) > 0:
                self.predicate = True
                self.scan_entries.append(-1)
                costs.append(step.expressions[0].cost())
                self.scan_local = List[UInt32](
                    unsafe_uninit_length=2 * STANDARD_VECTOR_SIZE
                )
        self.scan_order = _cheapest_first(costs)

    def _select_source(
        mut self,
        columns: _Columns,
        bases: List[Vector],
        mut rows: List[UInt32],
        used: Int,
        start: Int,
        end: Int,
        count: Int,
        dense: Bool,
    ) raises -> Int:
        """The source's filter step over one vector (morsel rows [start,
        end)): of its `count` selected rows (at `used` in `rows`, or the
        whole vector when `dense`), how many it keeps, written over them in
        order unless it keeps them all."""
        var n = len(self.steps[])
        var compiled = Pointer[Optional[_VectorStep], MutAnyOrigin](
            unsafe_from_address=Int(self.runner.compiled.unsafe_ptr())
        )
        var chunk = Pointer[DataChunk, MutAnyOrigin](
            unsafe_from_address=Int(Pointer(to=self.scan_chunk))
        )
        var began = Int(perf_counter_ns())
        columns.fill(bases, self.runner.arena, chunk[], start, end - start)
        var local_at = Int(self.scan_local.unsafe_ptr())
        var out_at = local_at + 4 * STANDARD_VECTOR_SIZE
        var local = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=local_at)
        var out = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=out_at)
        var base = Pointer[UInt32, MutAnyOrigin](
            unsafe_from_address=Int(rows.unsafe_ptr()) + 4 * used
        )
        var sel = SelectionVector()
        if not dense:
            for i in range(count):
                local[unsafe_offset=i] = base[unsafe_offset=i] - UInt32(start)
            sel = SelectionVector(local_at)
        var kept = (
            compiled[unsafe_offset=0]
            .value()
            .expressions[0]
            .select(
                chunk[],
                self.runner.arena,
                sel,
                count,
                SelectionVector(out_at),
            )
        )
        if kept < count:
            for i in range(kept):
                base[unsafe_offset=i] = out[unsafe_offset=i] + UInt32(start)
        self.runner.counts[1] += count
        self.runner.counts[2] += kept
        self.runner.counts[2 * n + 1] += Int(perf_counter_ns()) - began
        return kept

    def _join_rows(
        mut self, frame: DataFrame, offset: Int, length: Int, columns: _Columns
    ) raises -> Tuple[Bool, Optional[List[UInt32]]]:
        """The rows of a fresh morsel (`frame`, at source row `offset`) the
        scan's filters keep: False when none is, else their selection (None:
        every row). Vectors the statistics filters rule out are not read; the
        row filters run vector by vector, each narrowing the vector's
        selection in turn (a bloom filter or key bitmap only while its
        selectivity keeps it active), and the morsel's selection is the
        vectors' selections end to end.

        The row filters are the join filters and, when `predicate`, the
        source's filter step (over `columns`, the morsel's columns). They run
        cheapest first (`_scan_filters`), each on the rows the ones before it
        kept."""
        for f in self.join_filters:
            if not f.any:
                return (False, None)
        if not self.scan_ready:
            self._scan_filters()
        var vectors = Optional[List[Int]]()
        if self.join_scan:
            vectors = _vectors_kept(self.join_scan.value()[], offset, length)
            if vectors:
                trace_path("pipeline.join_filter_vector_skip")
                if len(vectors.value()) == 0:
                    trace_path("pipeline.join_filter_skip")
                    return (False, None)
        if self.predicate and self.runner.scan and not self.source:
            # The source filter's statistics: as its step runs here, its
            # vectors are ruled out here too.
            var ruled = _vectors_kept(
                self.runner.scan.value()[], offset, length
            )
            if ruled:
                trace_path("scan.vector_skip")
                if vectors:
                    ruled = _common_vectors(vectors.value(), ruled.value())
                vectors = ruled^
                if len(vectors.value()) == 0:
                    return (False, None)
        if len(self.scan_entries) == 0:
            if not vectors:
                return (True, None)
            return (True, _rows_in_vectors(None, vectors.value(), length))
        var keys = List[Series](capacity=len(self.join_filters))
        # Whether each key has no null, counted once for the morsel.
        var valid = List[Bool](capacity=len(self.join_filters))
        for f in self.join_filters:
            if f.equal or f.bloom or f.bitmap:
                keys.append(_unchunked(frame.column(f.column)))
                valid.append(keys[len(keys) - 1].null_count() == 0)
            else:
                keys.append(Series("", Column[Int64](List[Int64]())))
                valid.append(True)
        var bases = List[Vector]()
        if self.predicate:
            bases = columns.bases(self.runner.arena)
        var starts = List[Int]()
        if vectors:
            starts = vectors.value().copy()
        else:
            for at in range(0, length, STATISTICS_VECTOR_SIZE):
                starts.append(at)
        # Room for every row: each vector's selection is written after the
        # ones before it, then narrowed in place. A vector no row filter
        # narrows (all paused, or keeping every row) keeps every row; its
        # identity selection is written only once some vector is narrowed,
        # as until then the selection is the vectors themselves.
        var rows = List[UInt32](unsafe_uninit_length=length)
        var used = 0
        var listed = False
        for v in range(len(starts)):
            var start = starts[v]
            var end = min(length, start + STATISTICS_VECTOR_SIZE)
            var count = end - start
            var dense = True
            for i in range(len(self.scan_entries)):
                if count == 0:
                    break
                var k = self.scan_entries[self.scan_order[i]]
                if k < 0:
                    var kept = self._select_source(
                        columns, bases, rows, used, start, end, count, dense
                    )
                    if kept < count:
                        if not listed:
                            _identity_rows(rows, starts, v, length)
                            listed = True
                        dense = False
                        count = kept
                    continue
                ref f = self.join_filters[k]
                if not f.equal and not self.join_stats[k].active:
                    self.join_stats[k].update(0, 0)
                    continue
                if not listed:
                    _identity_rows(rows, starts, v, length)
                    listed = True
                var at = Int(rows.unsafe_ptr()) + 4 * used
                var kept = keep_rows(
                    f, keys[k], start, count, at, dense, valid[k]
                )
                if not f.equal:
                    self.join_stats[k].update(kept, count)
                dense = False
                count = kept
            if dense and listed:
                _fill_identity(rows, used, start, count)
            used += count
        if used == length:
            return (True, None)
        if not listed:
            _identity_rows(rows, starts, len(starts), length)
        rows.shrink(used)
        return (True, rows^)

    def run(mut self) raises:
        if self.source and self.first_step == 0 and self.unmatched_step < 0:
            self._run_source()
            return
        var count = len(self.ranges[]) // 2
        var cursor = _Cursor(self.cursor)
        while True:
            var taken = cursor.take(1)
            if taken >= count or self.first_morsel + taken > self.bound.last():
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
            self._morsel(morsel, k, offset, length)

    def _run_source(mut self) raises:
        """Batches from the file source until it is done: each is taken under
        the source's lock and decoded on this thread, then runs as a morsel
        numbered by its place in the file."""
        while True:
            var next = self.source.value()[].take(self.bound.last())
            if not next:
                break
            var batch = next.take()
            var k = batch.number
            var morsel = batch^.into_frame()
            var length = morsel.height()
            self.runner.begin_sink(k, 0, length)
            self.runner.begin_morsel()
            self._morsel(morsel, k, 0, length)
        self.counts = self.runner.counts.copy()
        if self.finishes:
            self.runner.release_sink()
            self.runner.release_local()

    def _morsel(
        mut self, morsel: DataFrame, k: Int, offset: Int, length: Int
    ) raises:
        """Morsel `k` (source rows [offset, offset + length)) from the
        source's step through every step into the sink."""
        self.runner.counts[0] += length
        var vectors = Optional[List[Int]]()
        # A top-N taking chunks checks each vector as it is read.
        if self.top_filter.info and not self._chunk_top():
            # The TopN's dynamic filter against the scan's vector statistics:
            # vectors it rules out are not read.
            vectors = _vectors_kept(
                self.top_filter.info.value()[], offset, length
            )
            if vectors and len(vectors.value()) == 0:
                trace_path("pipeline.top_dynamic_skip")
                return
        if not self.scan_ready:
            self._scan_filters()
        # When the source's filter step runs among the scan's filters, it
        # reads the morsel's columns there and the steps start after it.
        var columns = _Columns.of_frame(self.runner.arena, morsel)
        var first = 1 if self.predicate else 0
        var keyed = self._join_rows(morsel, offset, length, columns)
        if not keyed[0]:
            return
        var rows = keyed[1].copy()
        if vectors:
            rows = _rows_in_vectors(rows, vectors.value(), length)
        var source: _Columns
        if rows:
            if len(rows.value()) == 0:
                return
            trace_path("pipeline.join_filter_selection")
            # A top-N maps its chunks back to source rows through them.
            var kept = rows.value().copy() if self._chunk_top() else rows.take()
            source = columns.through_rows(kept^)
        else:
            source = columns^
            self.runner.scan_offset = offset
        if self._chunk_top():
            self._top_chunks(source^, rows, k, offset, first, True)
            self.runner.scan_offset = -1
            return
        _ = self.runner.run(source^, first)
        self.runner.scan_offset = -1

    def _sink_from(mut self, var columns: _Columns, k: Int, offset: Int) raises:
        """An outer join's unmatched build rows (or a morsel of them past
        its step) through the steps after `first_step` into the sink."""
        if self._chunk_top():
            self._top_chunks(columns^, None, k, offset, self.first_step, False)
        else:
            _ = self.runner.run(columns^, self.first_step)

    def _chunk_top(self) -> Bool:
        """Whether the sink is a top-N, taking chunks one at a time."""
        return self.top > 0 and not self.reduces

    def _top_chunks(
        mut self,
        var source: _Columns,
        rows: Optional[List[UInt32]],
        k: Int,
        offset: Int,
        first_step: Int,
        scanned: Bool,
    ) raises:
        """Feed a morsel into the top-N sink one chunk at a time from
        first_step. Recheck the dynamic boundary before reading each vector:
        earlier chunks may have tightened it enough to skip later vectors.
        `scanned`: the columns are the source's rows (through `rows`)."""
        var ran = self.runner.run_to_last(source^, first_step)
        var columns = ran[0].copy()
        var first = ran[1]
        var made = 0
        if len(self.topn) > 0:
            self.topn[0].begin(k)
        if first < len(self.runner.steps[]) and _cross_step(
            self.runner.steps[][first]
        ):
            # A cross product's pieces, each one chunk through the steps
            # after it.
            var at = _CrossAt()
            self.runner.counts[2 * first + 1] += columns.height
            while True:
                var piece = self.runner.cross_piece(columns, first, at)
                if not piece:
                    break
                self.runner.counts[2 * first + 2] += piece.value().height
                var bases = piece.value().bases(self.runner.arena)
                var address = self.runner.chunk(
                    piece.value(), bases, 0, piece.value().height, first + 1
                )
                made = self._take_top(address, k, offset, made)
            return
        # The vectors map to source rows only before any join.
        var direct = scanned and first == first_step
        var bases = columns.bases(self.runner.arena)
        var skipped = 0
        for low in range(0, columns.height, STANDARD_VECTOR_SIZE):
            var count = min(STANDARD_VECTOR_SIZE, columns.height - low)
            if direct and (self.top_filter.info or self.runner.scan):
                var start = low
                var end = low + count
                if rows:
                    start = Int(rows.value()[low])
                    end = Int(rows.value()[low + count - 1]) + 1
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
            made = self._take_top(address, k, offset, made)
        if skipped > 0:
            # Once a morsel: tracing reads the environment.
            trace_path("pipeline.top_dynamic_skip_vector")

    def _take_top(
        mut self, address: Int, k: Int, offset: Int, made: Int
    ) raises -> Int:
        """The chunk at `address` (0: none) into this worker's top-N heap,
        `made` rows of morsel `k` before it; the rows made with it."""
        if address == 0:
            return made
        var chunk = Pointer[DataChunk, MutAnyOrigin](
            unsafe_from_address=address
        )
        if len(self.topn) == 0:
            self._make_top()
        if self.top_maintain_order:
            var height = chunk[].count
            self._sink_ordered_top(chunk[], k, offset, made)
            return made + height
        _sink_chunk(self.topn[0], chunk[], self.runner.arena)
        return made

    def _make_top(mut self) raises:
        """This worker's top-N heap over the output schema; an order
        column (`_ORDER_COLUMN`) after it when ties keep the input
        order."""
        var schema = self.runner.output_schema()
        if self.top_maintain_order:
            schema = schema.with_column(
                Series(_ORDER_COLUMN, Column[Int64](List[Int64]()))
            )
        self.topn.append(
            TopNSink(
                schema,
                _top_keys(self.top_names, self.top_maintain_order),
                _top_flags(self.top_descending, self.top_maintain_order),
                _top_flags(self.top_nulls_last, self.top_maintain_order),
                self.top,
                self.boundary,
                self.top_filter.dynamic,
                self.top_filter.op,
            )
        )

    def _sink_ordered_top(
        mut self, mut chunk: DataChunk, k: Int, offset: Int, first_row: Int
    ) raises:
        """A chunk into this worker's top-N heap whose ties keep the input
        order: each row's place in the input is its last sort key, a
        sequence vector after the chunk's columns (morsel `k`'s rows
        numbered from `first_row`), so the heap reads it like any key and
        nothing is materialized for it."""
        var base = Int64(k) << 40
        if self.top_filter.reordered:
            # A row group's rows (at most ROW_GROUP_ROWS, no join) after
            # its source row.
            base = Int64(offset) << 20
        var ordered = DataChunk(len(chunk.data) + 1)
        for c in range(len(chunk.data)):
            ordered.data[c] = chunk.data[c]
        ordered.data[len(chunk.data)] = Vector.sequence(
            base + Int64(first_row), 1
        )
        ordered.count = chunk.count
        _sink_chunk(self.topn[0], ordered, self.runner.arena)

    def _unmatched(self, chunk: Int, length: Int) raises -> DataFrame:
        """The build rows of the hash table's row chunks [chunk, chunk +
        length) the join emits after the probe, as the step's output: a
        right-semi or right-anti join's build columns, or an outer join's
        with the probe columns null, coalesced keys taken from the rows."""
        ref step = self.steps[][self.unmatched_step]
        ref table = self.indexes[][step.join].value()
        ref right = self.joins[][step.join]
        var want = UInt8(1) if step.how == JOIN_RIGHT_SEMI else UInt8(0)
        var pointers = List[UInt64]()
        for c in range(chunk, chunk + length):
            table.scan_chunk(c, want, pointers)
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
                        table.gather_field(
                            key,
                            Int(pointers.unsafe_ptr()),
                            n,
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
            var t = table.output_index(right._columns[i].name())
            if t < 0:
                raise Error("pipeline join: a build column is not in the rows")
            columns.append(
                table.gather(
                    t, Int(pointers.unsafe_ptr()), n, right._columns[i].name()
                )
            )
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
    """A TopN's dynamic filter on its first sort column, pushed into the scan,
    so the scan skips rows (and whole vectors, by statistics) that cannot enter
    the heap: `info` checks it against the source's statistics, `dynamic` is the
    shared constant the sinks set, `op` its comparison."""

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


def _fill_identity(mut rows: List[UInt32], at: Int, start: Int, count: Int):
    """Rows [start, start + count) as a selection in `rows` from `at`, eight
    at a time."""
    comptime W = 8
    var out = rows.unsafe_ptr().unsafe_offset(at)
    var lanes = SIMD[DType.uint32, W](0, 1, 2, 3, 4, 5, 6, 7) + UInt32(start)
    var i = 0
    while i + W <= count:
        out.unsafe_store(i, lanes)
        lanes += UInt32(W)
        i += W
    while i < count:
        out[unsafe_offset=i] = UInt32(start + i)
        i += 1


def _identity_rows(
    mut rows: List[UInt32], starts: List[Int], vectors: Int, length: Int
):
    """Every row of the first `vectors` vectors (at `starts`, the last
    ending at `length`), written to the front of `rows`."""
    var used = 0
    for v in range(vectors):
        var start = starts[v]
        var count = min(length, start + STATISTICS_VECTOR_SIZE) - start
        _fill_identity(rows, used, start, count)
        used += count


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


def _cheapest_first(costs: List[Int]) -> List[Int]:
    """The positions of `costs` from the lowest cost up, ties in their given
    order."""
    var order = List[Int](capacity=len(costs))
    for i in range(len(costs)):
        # Insertion sort: stable, and a scan has a handful of filters.
        var j = len(order)
        order.append(i)
        while j > 0 and costs[order[j - 1]] > costs[i]:
            order[j] = order[j - 1]
            j -= 1
        order[j] = i
    return order^


def _common_vectors(a: List[Int], b: List[Int]) -> List[Int]:
    """The vector starts in both ascending lists."""
    var out = List[Int]()
    var j = 0
    for start in a:
        while j < len(b) and b[j] < start:
            j += 1
        if j < len(b) and b[j] == start:
            out.append(start)
    return out^


def _rows_in_vectors(
    rows: Optional[List[UInt32]], vectors: List[Int], length: Int
) -> List[UInt32]:
    """The rows of the selection `rows` (None: every row of the morsel)
    inside the kept vectors."""
    if not rows:
        var total = 0
        for start in vectors:
            total += min(length, start + STATISTICS_VECTOR_SIZE) - start
        var every = List[UInt32](unsafe_uninit_length=total)
        _identity_rows(every, vectors, len(vectors), length)
        return every^
    var out = List[UInt32]()
    ref mine = rows.value()
    var v = 0
    for r in mine:
        var row = Int(r)
        while v < len(vectors) and vectors[v] + STATISTICS_VECTOR_SIZE <= row:
            v += 1
        if v < len(vectors) and vectors[v] <= row:
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


def finish_pipeline(mut run: PipelineRun) raises -> PipelineFinish:
    """`run`'s finish once its execute tasks are done, as rounds of tasks
    (`PipelineFinish`). `run` must stay where it is until the finish's
    `result`."""
    return PipelineFinish(Int(Pointer(to=run)))


# The finish's stages (`PipelineFinish.stage`).
# The sink's finish.
comptime _FINISH_SINK = 0
# The finish steps run on the aggregate table's finished parts.
comptime _FINISH_PARTS = 1
# The finish steps' second pipeline: its execute tasks, then its sink's
# finish.
comptime _FINISH_SECOND = 2
comptime _FINISH_SECOND_SINK = 3
comptime _FINISH_DONE = 4

# The stages of a sink's finish (`PipelineFinish.sink`).
comptime _SINK_BEGIN = 0
# A grouping set's aggregate table finishing, then its parts finishing
# into its frame.
comptime _SINK_SET_TABLE = 1
comptime _SINK_SET_FRAME = 2
# The aggregate table finishing, then its parts finishing into the
# output.
comptime _SINK_TABLE = 3
comptime _SINK_FRAME = 4
# The workers' batches copied into the output.
comptime _SINK_BATCHES = 5
comptime _SINK_DONE = 6
# A round of a right or full join's unmatched build rows that no
# pipeline of the schedule scanned (`PipelineRun.begin_round`).
comptime _SINK_OUTER = 7
# The ungrouped reductions' finish (`UngroupedFinish`).
comptime _SINK_UNGROUPED = 8


struct PipelineFinish(Movable, Rounds):
    """A pipeline's finish once its execute tasks are done, as rounds of
    tasks (`next_round`, `round_task`), the result once they are done
    (`result`). The tasks of a round run at the same time; the serial
    work between rounds is small (combining the workers' sink states,
    setting up the next round).

    First the sink's finish: an aggregate table's phase 2
    (`AggregateFinish`) and its parts finished into frames
    (`PartsFinish`), for each grouping set in turn; or the workers'
    batches copied out (`MaterializeFinish`). Then the row-local steps
    above an aggregate: run on each finished part of the aggregate table
    by the task that takes the part (`parts`), or, when the sink
    finished its output whole, as a second pipeline over it (`second`),
    a task per worker, whose own sink then finishes. Either way each row
    runs through the steps on one thread, and no thread waits on another
    while a round runs."""

    # The address of the pipeline's `PipelineRun`.
    var run: Int
    var stage: Int
    var sink: Int
    # The sink's output, or the result once no work is left.
    var frame: DataFrame
    # The rows fed and each step's input and output rows, of the sink
    # finishing (`sink_counts`) and of the pipeline (`run_counts`).
    var sink_counts: List[Int]
    var run_counts: List[Int]
    # The finished frames of grouping sets (set order, the grand total's
    # with its one row).
    var set_frames: List[DataFrame]
    # The finish steps' input and output rows counted so far.
    var counts: List[Int]
    var parts: List[_FinishStepsJob]
    var second: List[PipelineRun]
    # The steps the parts or the second pipeline run.
    var steps: Int
    # Tasks for the parts: task t runs parts t, t + stride, ...
    var stride: Int
    # The aggregate table's sinks, whose heaps hold the long key strings
    # of its finished groups until they are read out.
    var held: List[AggregateSink]
    # The sink's finish under way: one entry at most in each.
    var table: List[AggregateFinish]
    var frames: List[PartsFinish]
    var batches: List[MaterializeFinish]
    var ungrouped: List[UngroupedFinish]
    # The run whose unmatched build rows the round under way scans.
    var outer: Int
    # The filter the aggregate table's finish runs (`GroupFilter` points
    # at it), and the finish steps left after it.
    var having: List[_HavingFilter]
    var rest: List[Step]
    # Grouping sets: the combined sinks, the sets in finish order, the
    # set finishing (`order[next]`), each set's place among the sets
    # still held, and the frames finished so far.
    var grouped: List[GroupingSetsSink]
    var order: List[Int]
    var next: Int
    var remaining: List[Int]
    var finished_sets: List[DataFrame]

    def __init__(out self, run: Int) raises:
        self.run = run
        self.stage = _FINISH_SINK
        self.sink = _SINK_BEGIN
        self.frame = DataFrame(List[Series](), height=0)
        self.sink_counts = List[Int]()
        self.run_counts = List[Int]()
        self.set_frames = List[DataFrame]()
        self.counts = List[Int]()
        self.parts = List[_FinishStepsJob]()
        self.second = List[PipelineRun]()
        self.steps = 0
        self.stride = 0
        self.held = List[AggregateSink]()
        self.table = List[AggregateFinish]()
        self.frames = List[PartsFinish]()
        self.batches = List[MaterializeFinish]()
        self.ungrouped = List[UngroupedFinish]()
        self.outer = 0
        self.having = List[_HavingFilter]()
        self.rest = List[Step]()
        self.grouped = List[GroupingSetsSink]()
        self.order = List[Int]()
        self.next = 0
        self.remaining = List[Int]()
        self.finished_sets = List[DataFrame]()

    def next_round(mut self) raises -> Int:
        """The tasks of the next round, 0 when the output is ready."""
        while True:
            if self.stage == _FINISH_SINK:
                var tasks = self._sink_round(self.run)
                if tasks > 0:
                    return tasks
                swap(self.run_counts, self.sink_counts)
                self._after_sink()
                if len(self.parts) > 0:
                    self.stage = _FINISH_PARTS
                    return self.stride
                if len(self.second) > 0:
                    self.stage = _FINISH_SECOND
                    if self.second[0].tasks() > 0:
                        return self.second[0].tasks()
                    continue
                self.stage = _FINISH_DONE
            elif self.stage == _FINISH_PARTS:
                self._end_parts()
                self.stage = _FINISH_DONE
            elif self.stage == _FINISH_SECOND:
                self.sink = _SINK_BEGIN
                self.stage = _FINISH_SECOND_SINK
            elif self.stage == _FINISH_SECOND_SINK:
                var tasks = self._sink_round(Int(Pointer(to=self.second[0])))
                if tasks > 0:
                    return tasks
                for k in range(self.steps):
                    self.counts.append(self.sink_counts[2 * k + 1])
                    self.counts.append(self.sink_counts[2 * k + 2])
                self.second = List[PipelineRun]()
                self.stage = _FINISH_DONE
            else:
                return 0

    def round_task(mut self, task: Int) raises:
        if self.sink == _SINK_OUTER:
            Pointer[PipelineRun, MutAnyOrigin](
                unsafe_from_address=self.outer
            )[].execute(task)
        elif self.stage == _FINISH_PARTS:
            var p = task
            while p < len(self.parts):
                self.parts[p].run()
                p += self.stride
        elif self.stage == _FINISH_SECOND:
            self.second[0].execute(task)
        elif len(self.table) > 0:
            self.table[0].round_task(task)
        elif len(self.frames) > 0:
            self.frames[0].round_task(task)
        elif len(self.ungrouped) > 0:
            self.ungrouped[0].round_task(task)
        else:
            self.batches[0].round_task(task)

    def result(
        mut self, mut counts: List[Int], mut set_frames: List[DataFrame]
    ) raises -> DataFrame:
        """The output, once `next_round` returned 0: the rows fed and each
        step's input and output rows in `counts`, then the finish steps'
        input and output rows; the grouping sets' frames in
        `set_frames`."""
        counts = self.run_counts.copy()
        counts.extend(Span(self.counts))
        set_frames = self.set_frames.copy()
        return self.frame.copy()

    def release(mut self):
        """Free the second pipeline's shared state, for a finish that
        failed before its result."""
        if len(self.second) > 0:
            self.second[0].release()

    def _end_parts(mut self) raises:
        """The parts' frames as the result's chunks."""
        var frames = List[DataFrame](capacity=len(self.parts))
        var rest = List[Int](length=2 * self.steps, fill=0)
        for j in range(len(self.parts)):
            frames.append(self.parts[j].frame.copy())
            for k in range(len(rest)):
                rest[k] += self.parts[j].counts[k]
        self.frame = chunks_frame(frames)
        self.counts.extend(Span(rest))
        self.parts = List[_FinishStepsJob]()
        self.stride = 0
        self.held = List[AggregateSink]()

    def _after_sink(mut self) raises:
        """Once the sink finished its output whole: the finish steps over
        it as a pipeline of their own, morsels on every worker."""
        ref run = Pointer[PipelineRun, MutAnyOrigin](
            unsafe_from_address=self.run
        )[]
        if (
            len(run.finish_steps) == 0
            or len(self.parts) > 0
            or len(self.counts) > 0
        ):
            return
        self.second.append(
            begin_pipeline(
                self.frame,
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
        )
        self.steps = len(run.finish_steps)

    def _sink_round(mut self, at: Int) raises -> Int:
        """The tasks of the next round of the sink's finish of the run at
        `at`, 0 once its output is in `frame` (and, for the aggregate
        table's finished parts that the finish steps run on, in
        `parts`)."""
        ref run = Pointer[PipelineRun, MutAnyOrigin](unsafe_from_address=at)[]
        while True:
            if self.sink == _SINK_BEGIN:
                if run.begin_round():
                    # A round of the unmatched build rows, a task per
                    # worker, before the sink combines the workers'
                    # states.
                    self.outer = at
                    self.sink = _SINK_OUTER
                    return run.tasks()
                self._begin_sink(run)
            elif self.sink == _SINK_OUTER:
                self.sink = _SINK_BEGIN
            elif self.sink == _SINK_UNGROUPED:
                var tasks = self.ungrouped[0].next_round()
                if tasks > 0:
                    return tasks
                self.frame = self.ungrouped.pop().frame.copy()
                self.sink = _SINK_DONE
            elif self.sink == _SINK_SET_TABLE:
                var tasks = self.table[0].next_round()
                if tasks > 0:
                    return tasks
                var done = self.table.pop()
                swap(self.held, done.sinks)
                var result = AggregateResult()
                swap(result, done.result)
                var s = self.order[self.next]
                self.frames.append(
                    PartsFinish.of_table(run.template[s], result^, False)
                )
                self.sink = _SINK_SET_FRAME
            elif self.sink == _SINK_SET_FRAME:
                var tasks = self.frames[0].next_round()
                if tasks > 0:
                    return tasks
                var finished = self.frames.pop().frame.copy()
                self.held = List[AggregateSink]()
                var s = self.order[self.next]
                if len(run.sets[s]) == 0:
                    if finished.height() == 0:
                        # No rows: the grand total is still one row.
                        finished = run.set_empty[0].select_exprs(
                            run.expressions
                        )
                    else:
                        finished = finished.drop([_TOTAL_KEY])
                self.finished_sets[s] = finished^
                self.next += 1
                if self.next < len(self.order):
                    self._begin_set(run)
                else:
                    for t in range(len(run.sets)):
                        self.set_frames.append(self.finished_sets[t].copy())
                    self.finished_sets = List[DataFrame]()
                    self.grouped = List[GroupingSetsSink]()
                    # An empty frame of the steps' output types the
                    # caller's null keys.
                    self.frame = run.set_empty[0].copy()
                    self.sink = _SINK_DONE
            elif self.sink == _SINK_TABLE:
                var tasks = self.table[0].next_round()
                if tasks > 0:
                    return tasks
                self._end_table(run)
            elif self.sink == _SINK_FRAME:
                var tasks = self.frames[0].next_round()
                if tasks > 0:
                    return tasks
                var finished = self.frames.pop().frame.copy()
                if len(run.finish_steps) > 0:
                    # The groups in first-occurrence order, then the steps
                    # over them.
                    finished = _steps_over_frame(
                        finished,
                        run.finish_steps,
                        run.joins,
                        run.indexes,
                        self.counts,
                    )
                self.held = List[AggregateSink]()
                self.frame = finished^
                self.sink = _SINK_DONE
            elif self.sink == _SINK_BATCHES:
                var tasks = self.batches[0].next_round()
                if tasks > 0:
                    return tasks
                var whole = self.batches.pop().frame.copy()
                if run.collects:
                    if len(run.keys) == 0:
                        whole = whole.select_exprs(run.expressions)
                    else:
                        whole = whole.group_by(
                            run.keys, maintain_order=run.ordered
                        ).agg(run.expressions)
                self.frame = whole^
                self.sink = _SINK_DONE
            else:
                return 0

    def _begin_sink(mut self, mut run: PipelineRun) raises:
        """The sink's serial start, once the unmatched build rows' rounds
        ran: the counts, the workers' sink states combined, and the first
        round of the sink's finish set up, or the output made here when
        the sink has no parallel finish."""
        ref jobs = run.jobs
        ref steps = run.steps
        ref layout = run.layout
        ref sets = run.sets
        var planned = run.planned
        run.release()
        var counts = List[Int](length=3 * len(steps) + 1, fill=0)
        for j in range(len(jobs)):
            for i in range(len(counts)):
                counts[i] += jobs[j].counts[i]
        if len(steps) != planned:
            # The aggregate inputs' projection has no plan node: the
            # counts are the planned steps'.
            var n = len(steps)
            var kept = List[Int](capacity=3 * planned + 1)
            for i in range(2 * planned + 1):
                kept.append(counts[i])
            for i in range(planned):
                kept.append(counts[2 * n + 1 + i])
            counts = kept^
        self.sink_counts = counts^
        self.sink = _SINK_DONE
        if run.selection_output:
            ref output = run.selection_output.value()[]
            # Each worker's morsels are ordered. Merge them in source order,
            # moving the owners and row lists without copying their contents.
            var next = List[Int](length=len(jobs), fill=0)
            while True:
                var best = -1
                var first = 0
                for j in range(len(jobs)):
                    ref batches = jobs[j].runner.retained_batches
                    if next[j] < len(batches):
                        var morsel = batches[next[j]].morsel
                        if best < 0 or morsel < first:
                            best = j
                            first = morsel
                if best < 0:
                    break
                var batch = SelectionBatch()
                swap(batch, jobs[best].runner.retained_batches[next[best]])
                output.append(batch^)
                next[best] += 1
            output.ready = True
            self.frame = output.schema.copy()
            return
        if len(layout) > 0 and len(sets) > 0:
            # Derived sets finish first: their rows point at key bytes the
            # source set's sinks hold. A worker's sinks are in set order,
            # so a set's is found by its place among those still held.
            self.order = List[Int]()
            for back in range(len(sets)):
                var t = len(sets) - 1 - back
                if run.derived_from[t] >= 0:
                    self.order.append(t)
            for back in range(len(sets)):
                var t = len(sets) - 1 - back
                if run.derived_from[t] < 0:
                    self.order.append(t)
            self.remaining = List[Int]()
            for t in range(len(sets)):
                self.remaining.append(t)
            self.finished_sets = List[DataFrame](
                length=len(sets), fill=run.frame.clear()
            )
            var locals = List[GroupingSetsSink]()
            for j in range(len(jobs)):
                if len(jobs[j].runner.sets) > 0:
                    locals.append(jobs[j].runner.sets.pop())
            self.grouped = _combined(locals^)
            self.next = 0
            self._begin_set(run)
            return
        if len(layout) > 0:
            self._begin_table(run)
            return
        if run.collects or (
            len(run.ungrouped) == 0 and not run.build_layout and run.top <= 0
        ):
            # The workers' batches in source order.
            var locals = List[MaterializeSink]()
            for j in range(len(jobs)):
                if len(jobs[j].runner.materialize) > 0:
                    locals.append(jobs[j].runner.materialize.pop())
            var combined = _combined(locals^)
            self.batches.append(
                MaterializeFinish(
                    combined^,
                    run.schemas[][len(steps)].clear(),
                    limit=-1 if run.collects else run.limit,
                )
            )
            self.sink = _SINK_BATCHES
            return
        if len(run.ungrouped) > 0:
            var sinks = List[UngroupedSink](capacity=len(jobs))
            for j in range(len(jobs)):
                if len(jobs[j].runner.ungrouped) > 0:
                    sinks.append(jobs[j].runner.ungrouped.pop())
            self.ungrouped.append(UngroupedFinish(run.ungrouped[0], sinks^))
            self.sink = _SINK_UNGROUPED
            return
        self.frame = _finish_here(run)

    def _begin_set(mut self, mut run: PipelineRun) raises:
        """The aggregate table of the next set in finish order."""
        var s = self.order[self.next]
        var place = self.remaining.index(s)
        var sinks = List[AggregateSink]()
        for j in range(len(self.grouped)):
            sinks.append(self.grouped[j].tables.pop(place))
        _ = self.remaining.pop(place)
        self.table.append(AggregateFinish(sinks^, run.layout[s], ordered=False))
        self.sink = _SINK_SET_TABLE

    def _begin_table(mut self, mut run: PipelineRun) raises:
        """The aggregate table's finish. Long key strings live in the
        sinks' heaps until the finish has read them out."""
        ref jobs = run.jobs
        var locals = List[TableSink]()
        for j in range(len(jobs)):
            if len(jobs[j].runner.table) > 0:
                locals.append(jobs[j].runner.table.pop())
        var combined = _combined(locals^)
        var sinks = List[AggregateSink](capacity=len(combined))
        while len(combined) > 0:
            sinks.append(combined.pop(0).take_table())
        self.sink = _SINK_TABLE
        if len(run.finish_steps) == 0 or run.ordered:
            # With steps, the groups in first-occurrence order (each
            # group row's first sequence number), then the steps over
            # them.
            self.table.append(
                AggregateFinish(sinks^, run.layout[0], ordered=run.ordered)
            )
            return
        # Run an initial filter on reduction outputs during aggregate
        # finalization, before rejected groups enter output columns. Apply
        # remaining row steps only to surviving groups.
        var having = _having(
            run.template[0], run.finish_steps[0], run.keys, run.batch_size
        )
        self.rest = run.finish_steps.copy()
        var filter = Optional[GroupFilter]()
        if having:
            trace_path("pipeline.aggregate_finish_filter")
            _ = self.rest.pop(0)
            self.having.append(having.take())
            filter = GroupFilter(
                self.having[0].slots.copy(),
                Int(Pointer(to=self.having[0])),
                _having_keep,
            )
        self.table.append(
            AggregateFinish(sinks^, run.layout[0], filter, ordered=False)
        )

    def _end_table(mut self, mut run: PipelineRun) raises:
        """The aggregate table finished: its parts finished into the
        output, or, with finish steps and no order, handed to the finish
        steps' tasks."""
        var done = self.table.pop()
        swap(self.held, done.sinks)
        var result = AggregateResult()
        swap(result, done.result)
        if len(run.finish_steps) == 0 or run.ordered:
            self.frames.append(
                PartsFinish.of_table(run.template[0], result^, run.ordered)
            )
            self.sink = _SINK_FRAME
            return
        var filtered = len(self.having) > 0
        self.having = List[_HavingFilter]()
        var merged = result.merged
        var kept = 0
        for p in range(len(result.firsts)):
            kept += len(result.firsts[p])
        var parts = table_parts(run.template[0], result^)
        if filtered:
            self.counts.append(merged)
            self.counts.append(kept)
        self.sink = _SINK_DONE
        if len(parts) == 0:
            # No group: the steps over the empty result, for its columns.
            var rest_counts = List[Int]()
            self.frame = _steps_over_frame(
                run.template[0].finish(),
                self.rest,
                run.joins,
                run.indexes,
                rest_counts,
            )
            self.counts.extend(Span(rest_counts))
            self.held = List[AggregateSink]()
            return
        # Each part finished and run through the steps by the task that
        # takes it, the kept rows of each a chunk of the result. The
        # parts' key strings still point into the sinks' heaps.
        trace_path("pipeline.aggregate_finish_steps")
        while len(parts) > 0:
            self.parts.append(
                _FinishStepsJob(
                    parts.pop(0),
                    self.rest,
                    run.joins,
                    run.indexes,
                    run.batch_size,
                )
            )
        self.steps = len(self.rest)
        self.stride = min(len(self.parts), max(1, run.workers))
        self.frame = run.frame.clear()


def _finish_here(mut run: PipelineRun) raises -> DataFrame:
    """The output of a sink whose finish has no parallel work: a join's
    build side, or a top-N."""
    ref jobs = run.jobs
    ref steps = run.steps
    ref schemas = run.schemas
    if run.build_layout:
        # A join's build side: the workers' rows combined into its table,
        # whose finalize tasks the caller's finish event runs. The result
        # is the build's columns, empty.
        var locals = List[JoinBuildSink]()
        for j in range(len(jobs)):
            if len(jobs[j].runner.build) > 0:
                locals.append(jobs[j].runner.build.pop())
        var combined = _combined(locals^)
        run.built = JoinHashTable(
            run.build_layout.value(), combined^, defer_finalize=True
        )
        return schemas[][len(steps)].clear()
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
    if run.top_maintain_order:
        merged = merged.drop([_ORDER_COLUMN])
    return merged^


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


struct PipelineRun(Movable):
    """A morsel pipeline between `begin_pipeline` and `finish_pipeline`: its
    workers' jobs (the execute tasks, one per thread) and what its finish reads.
    Lives on one thread at each stage but its execute, where each task has its
    own job."""

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
    # The file source the workers scan, when not `frame`.
    var source: Optional[ArcPointer[ScanSource]]
    # A join build's row layout, when the workers sink into its rows, and
    # the table the finish makes of them (its finalize tasks not yet run).
    var build_layout: Optional[ArcPointer[JoinBuildLayout]]
    var built: Optional[JoinHashTable]
    var selection_output: Optional[ArcPointer[SelectionBatches]]
    # A limited materialize sink's row count (-1: none) and the last
    # morsel its workers still need.
    var limit: Int
    var bound: LimitBound

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
        self.source = None
        self.build_layout = None
        self.built = None
        self.selection_output = None
        self.limit = -1
        self.bound = LimitBound.none()

    def tasks(self) -> Int:
        return len(self.jobs)

    def execute(mut self, task: Int) raises:
        self.jobs[task].run()

    def begin_round(mut self) raises -> Bool:
        """After probing, partition the build table's rows among workers and
        emit the required matched/unmatched rows. Fill absent probe columns
        with nulls and run the same downstream steps and sinks as ordinary
        join output. Scheduling places this scan after probe completion so
        match flags are final."""
        self.end_round()
        if self.source and self.next_round == 0:
            # The morsels the source's batches were numbered as.
            self.next_morsel = self.source.value()[].batches()
        while self.next_round < len(self.outer_steps):
            var j = self.outer_steps[self.next_round]
            var last = self.outer_steps[len(self.outer_steps) - 1]
            self.next_round += 1
            ref step = self.steps[j]
            # The table's row chunks split into one run per worker: a
            # table sunk by many threads holds many partly filled chunks
            # (one set per thread and partition), and a morsel per chunk
            # would run the steps after the join on a few rows at a time.
            var ranges = List[Int]()
            var chunks = self.indexes[][step.join].value().chunk_count()
            var per_worker = max(
                (chunks + len(self.jobs) - 1) // len(self.jobs), 1
            )
            for c in range(0, chunks, per_worker):
                ranges.append(c)
                ranges.append(min(per_worker, chunks - c))
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
        self.bound.free()
        if self.source:
            try:
                self.source.value()[].finish()
            except:
                pass


struct PipelineTypes(Movable):
    """What a pipeline's setup decides from its columns' types alone, not
    from any row: each step's input columns and the output, the
    row-local steps compiled once for every worker, and the aggregate
    table's layout (`pipeline_types`). A pipeline's first stage can make
    them while the builds its rows probe are still running."""

    var steps: List[Step]
    var expressions: List[Expr]
    var schemas: ArcPointer[List[DataFrame]]
    var templates: ArcPointer[List[Optional[_VectorStep]]]
    var layout: List[AggregateLayout]
    var template: List[_StreamReduction]
    var set_empty: List[DataFrame]

    def __init__(
        out self,
        var steps: List[Step],
        var expressions: List[Expr],
        schemas: ArcPointer[List[DataFrame]],
        templates: ArcPointer[List[Optional[_VectorStep]]],
        var layout: List[AggregateLayout],
        var template: List[_StreamReduction],
        var set_empty: List[DataFrame],
    ):
        self.steps = steps^
        self.expressions = expressions^
        self.schemas = schemas
        self.templates = templates
        self.layout = layout^
        self.template = template^
        self.set_empty = set_empty^


def pipeline_types(
    schema_frame: DataFrame,
    given_steps: List[Step],
    joins: List[DataFrame],
    given_expressions: List[Expr],
    keys: List[String],
    sets: List[List[Int]],
    table: Bool,
) raises -> PipelineTypes:
    """The steps over `schema_frame`'s columns typed and compiled, and,
    when the aggregate table takes the reduction (`table`), its layout:
    `joins` are the build sides (their columns; rows are not read). A
    layout reads no key statistics here: those are the rows'
    (`begin_pipeline`)."""
    # Every step's input columns and the output, typed once, and the
    # row-local steps compiled once for every worker.
    var steps = given_steps.copy()
    var expressions = given_expressions.copy()
    var typed = _step_schemas(schema_frame, steps, joins)
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
            # An aggregate filter drops rows, not types: the states are
            # the unfiltered reductions'.
            var reduction = _StreamReduction(
                empty, without_filters(expressions), keys
            )
            layout.append(
                aggregate_layout(
                    empty_keys,
                    table_inputs(empty, empty_keys, expressions),
                    table_specs(reduction),
                )
            )
            template.append(reduction^)
    return PipelineTypes(
        steps^,
        expressions^,
        schemas,
        templates,
        layout^,
        template^,
        set_empty^,
    )


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
    source: Optional[ArcPointer[ScanSource]] = None,
    source_rows: Int = -1,
    build_keys: List[String] = List[String](),
    build_how: Int = JOIN_INNER,
    build_coalesce: Bool = True,
    build_probe_estimate: Int = -1,
    build_filtered: Bool = False,
    var types: Optional[PipelineTypes] = None,
    limit: Int = -1,
    selection_output: Optional[ArcPointer[SelectionBatches]] = None,
) raises -> PipelineRun:
    """`run_pipeline` up to its workers' jobs, not yet run: its execute stage is
    the jobs, one task per thread; `finish_pipeline` is its finish, after the
    rounds of `begin_round`. With a file `source`, `frame` holds its columns but
    no row: the workers take the source's batches as their morsels, and
    `source_rows` is its estimated height. With `build_keys`, the rows
    are the build side of a hash join of type `build_how` on those keys
    (`build_coalesce`, `build_probe_estimate` and `build_filtered` as
    `finish_build` takes them): each worker sinks its chunks into its own rows of the join's
    hash table (`JoinBuildSink`), and the finish makes the table
    (`PipelineRun.built`) instead of a frame, unless a column's type
    keeps the build on a frame (`join_build_layout`). `types`: the steps
    typed and compiled and the table's layout (`pipeline_types`), when
    the caller made them already for `given_steps` over `frame`'s
    columns. `limit`: only the first that many rows in source order are
    the result (-1: every row), when the rows are materialized."""
    trace_path("pipeline.run")
    # The steps typed and compiled and the table's layout, unless the
    # caller made them already for these steps and columns.
    var made = Optional[PipelineTypes]()
    if not types:
        # A late-materialized top-N's scan adds its row id column to
        # every morsel.
        var schema_frame = frame.clear()
        if top_row_ids:
            schema_frame = schema_frame.with_column(
                Series(ROW_ID_COLUMN, Column[Int64](List[Int64]()))
            )
        made = pipeline_types(
            schema_frame,
            given_steps,
            joins[],
            given_expressions,
            keys,
            sets,
            table,
        )
    var chosen = types.take() if types else made.take()
    var steps = List[Step]()
    var expressions = List[Expr]()
    swap(steps, chosen.steps)
    swap(expressions, chosen.expressions)
    var schemas = chosen.schemas
    var templates = chosen.templates
    var layout = List[AggregateLayout]()
    var template = List[_StreamReduction]()
    var set_empty = List[DataFrame]()
    if table:
        swap(layout, chosen.layout)
        swap(template, chosen.template)
        swap(set_empty, chosen.set_empty)
    if len(layout) > 0 and len(sets) == 0 and not source:
        # Bounded source key statistics can select direct-address
        # aggregation. Use those bounds only when every grouping key has a
        # compatible estimate. A file source's statistics are not known
        # before its scan.
        var stats = _key_statistics(frame, steps, keys)
        if len(stats) == len(keys):
            layout[0].use_perfect_hash(stats)
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
        frame,
        source_rows if source else frame.height(),
        steps,
        joins[],
        indexes[],
        pushed,
        join_scan_filters,
    )
    if len(join_filters) > 0:
        trace_path("pipeline.join_filter")
    for f in join_filters:
        if f.bloom:
            trace_path("pipeline.join_bloom")
            break
    for f in join_filters:
        if f.bitmap:
            trace_path("pipeline.join_key_bitmap")
            break
    var join_scan = Optional[ArcPointer[ScanFilterInfo]]()
    if join_scan_filters.has_filters() and not source:
        for f in join_scan_filters.filters:
            _ensure_zones(frame.column(f.column))
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
    # One job a thread, up to the source's morsels (`ranges` holds an
    # (offset, length) pair each); a file's batches are not known ahead.
    var job_count = source_tasks(-1 if source else len(ranges[]) // 2, workers)
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
    if _order_dependent(expressions):
        for w in range(len(jobs)):
            jobs[w].runner.ordered_pairs = True
    # The aggregate table (one grouping): each worker's table takes the last
    # step's chunks, so workers never share a table while sinking.
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
    # A reduction neither the aggregate table nor the ungrouped sink
    # takes: the rows are collected (each worker's chunks copied into its
    # collections), then reduced by the eager evaluator, which takes any
    # expression (grouped, over the groups the table finds).
    var collects = (
        len(expressions) > 0 and len(layout) == 0 and len(ungrouped) == 0
    )
    # A hash join's build side: each worker's chunks into its own rows of
    # the join's table, so the rows are never copied into a frame first.
    var build_layout = Optional[ArcPointer[JoinBuildLayout]]()
    if (
        len(build_keys) > 0
        and len(expressions) == 0
        and top <= 0
        and len(sets) == 0
    ):
        var made = join_build_layout(
            schemas[][len(steps)]._columns,
            build_keys,
            build_how,
            build_coalesce,
            build_probe_estimate,
            True,
            build_filtered=build_filtered,
        )
        if made:
            trace_path("pipeline.join_build_sink")
            build_layout = ArcPointer(made.take())
            for w in range(len(jobs)):
                jobs[w].runner.build_layout.append(build_layout.value())
    # No reduction and no top-N: the rows are the result, each worker's chunks
    # copied into its collections, numbered by morsel so the finish restores
    # source order.
    if ((len(expressions) == 0 and top <= 0) or collects) and not build_layout:
        for w in range(len(jobs)):
            jobs[w].runner.materializes = True
    var retains = (
        Bool(selection_output)
        and len(expressions) == 0
        and top <= 0
        and not build_layout
        and limit < 0
        and len(finish_steps) == 0
    )
    if retains:
        trace_path("pipeline.selection_sink")
        for w in range(len(jobs)):
            jobs[w].runner.materializes = False
            jobs[w].runner.retains = True
    # A limit over the materialized rows: each worker keeps at most that
    # many of its own first rows, and the workers stop at the last morsel
    # the first rows can come from (`LimitBound`).
    var bound = LimitBound.none()
    if (
        limit >= 0
        and len(expressions) == 0
        and top <= 0
        and not build_layout
        and not _checks_every_row(steps)
    ):
        trace_path("pipeline.limit_sink")
        bound = LimitBound.new()
        if limit == 0:
            # No row is wanted: no morsel is read.
            bound.lower(-1)
        for w in range(len(jobs)):
            jobs[w].runner.limit = limit
            jobs[w].runner.limit_bound = bound
            jobs[w].bound = bound
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
    for w in range(len(jobs)):
        jobs[w].source = source
    var run = PipelineRun(
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
        collects,
        ungrouped^,
        finish_steps.copy(),
    )
    run.source = source
    run.build_layout = build_layout^
    if retains:
        run.selection_output = selection_output
    if bound.address != 0:
        run.limit = limit
        run.bound = bound
    return run^
