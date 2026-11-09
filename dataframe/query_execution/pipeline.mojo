"""Streaming pipeline lowering and execution."""
from dataframe.io.dataset_scan import _DatasetBatches
from dataframe.io.sinks import FileSink
from dataframe.io.csv_reader import _open_csv_batches
from dataframe.io.csv_types import _CsvScanOptions

from dataframe.planning.node_builders import (
    _frame_scan_node,
    _join_node,
)
from dataframe.planning.cost.aggregation import (
    select_aggregation_strategy,
)
from dataframe.kernels.distinct_statistics import distinct_estimate
from dataframe.planning.cost.eligibility import (
    pipeline_eligibility,
    consider_inline_table,
)
from dataframe.planning.cost.model import trace_decision

from dataframe.categorical import (
    decode,
)
from dataframe.storage.boolean import BoolColumn
from dataframe.storage.column import (
    Column,
)
from dataframe.dtype import (
    DataType,
)
from dataframe.expr import (
    COL,
    Expr,
    MAX,
    conjuncts,
    is_grouping,
    lit,
)
from dataframe.frame import (
    DataFrame,
    concat,
)
from dataframe.aggregation.expressions import (
    _row_local,
    _stream_reductions,
)
from dataframe.aggregation.state import (
    table_supported,
)
from dataframe.kernels.hashing import (
    encode_rows,
)
from dataframe.io.csv import (
    CsvSchema,
)
from dataframe.io.csv_reader import (
    _CsvBatches,
    _DecodeJob,
)
from dataframe.io.csv_types import (
    _map_file,
)
from dataframe.io.ipc import (
    _IpcBatches,
)
from dataframe.io.json_batches import (
    _NdjsonBatches,
)
from dataframe.io.json_parallel import (
    _DecodeJob as _JsonDecodeJob,
)
from dataframe.io.parquet import (
    _ParquetBatches,
    _pruned_row_groups,
    parquet_row_group_statistics,
)
from dataframe.joins.hash_table import JoinHashTable
from dataframe.join_type import (
    JOIN_ANTI,
    JOIN_CROSS,
    JOIN_FULL,
    JOIN_INNER,
    JOIN_LEFT,
    JOIN_RIGHT,
    JOIN_SEMI,
)
from dataframe.joins.join_filter import (
    PushedJoinFilter,
    PushedJoinFilters,
    supports_join_filters,
)
from dataframe.parallel import (
    Job,
    Pool,
    configured_workers,
    run_jobs,
)
from dataframe.sorting.top_n import TopNBoundary
from dataframe.sorting.top_n_sink import TopNSink
from dataframe.sorting.row_group_order import orderable, row_group_order
from dataframe.query_execution.late_materialization import (
    LATE_MATERIALIZATION_MAX_ROWS,
    late_fetch,
    late_materialization_plan,
)
from dataframe.pipeline import (
    _ORDER_COLUMN,
    _top_flags,
    _top_keys,
    _ensure_zones,
    _range_key,
    _window_bounds,
    run_pipeline,
)
from dataframe.planning.context import (
    FinishAbove,
    QueryContext,
    _ReadyIndex,
)
from dataframe.planning.inputs import (
    _SubplanPipeline,
    finish_build,
)
from dataframe.scheduler import (
    META_REGULAR,
    MetaPipeline,
    SchedulablePipeline,
    run_pipelines,
)
from dataframe.planning.graph import (
    plan_is_filtering,
    plan_columns_of,
    plan_filters_rows,
    plan_grouped_on,
    plan_input_name,
    plan_keeps_column,
    plan_key_source,
)
from dataframe.planning.model import (
    AGG,
    DROP,
    EXPLODE,
    FILTER,
    GROUPING_SETS,
    JOIN,
    PlanNode,
    SCAN_CSV,
    SCAN_FRAME,
    SCAN_IPC,
    SCAN_NDJSON,
    SCAN_PARQUET,
    SELECT,
    SLICE,
    SORT,
    UNNEST,
    WITH_COLUMNS,
)
from dataframe.planning.physical import (
    plan_build_probe_sides,
)
from dataframe.planning.properties import (
    _distinct_inputs,
    _is_scan,
    _same_names,
    _set_members,
    _stream_join,
    _stream_rows,
)
from dataframe.planning.cost.eligibility import (
    _counts_distinct,
    _filters,
)
from dataframe.planning.cost.sampling import (
    _keys_repeat,
)
from dataframe.planning.expression_inputs import (
    _references,
)
from dataframe.planning.scans import (
    _scan_options,
)
from dataframe.planning.cost.sampling import (
    _all_reduced,
    _frame_bytes,
    _judge_held,
    _stream_collect_bytes,
    _stream_split_groups,
)
from dataframe.query_execution.materialize import (
    plan_schedule_inputs,
)
from dataframe.query_execution.operators import (
    _set_rows,
)
from dataframe.query_execution.reporting import (
    plan_record,
)
from dataframe.query_execution.stream_state import (
    _StateAccumulator,
    _StreamJob,
    _apply_operation,
    _coalesced,
    _steps_of,
)
from dataframe.planning.cost.eligibility import (
    _pipeline_steps,
)
from dataframe.series import (
    Series,
)
from dataframe.table_scan import conjunction, scan_filtered
from dataframe.aggregation.state import (
    _StreamReduction,
    _finish_parts,
)
from dataframe.trace import (
    trace_path,
)
from std.collections import (
    Dict,
    Optional,
)
from std.memory import (
    ArcPointer,
    Pointer,
)
from std.time import (
    perf_counter_ns,
)


def plan_stream_execute(
    self: QueryContext, index: Int, batch_size: Int, sink: Int = 0
) raises -> Optional[DataFrame]:
    var cursor = index
    var expressions = List[Expr]()
    var keys = List[String]()
    var ordered = False
    var skip = 0
    var limit = -1
    ref terminal = self._nodes[index]
    var grouping = terminal.kind == GROUPING_SETS
    if grouping:
        # The aggregations, without the grouping columns (those are
        # constants of each set).
        for e in terminal.exprs:
            if not is_grouping(e):
                expressions.append(e.copy())
        if not _stream_reductions(expressions, True):
            return None
        keys = terminal.names.copy()
        cursor = terminal.left
    elif terminal.kind == AGG or terminal.kind == SELECT:
        # A grouped median or quantile streams only into the aggregate
        # table (checked below): its other routes keep no such state.
        if _stream_reductions(
            terminal.exprs, terminal.kind == AGG, terminal.kind == AGG
        ):
            expressions = terminal.exprs.copy()
            if terminal.kind == AGG:
                keys = terminal.names.copy()
                ordered = terminal.maintain_order
                if len(keys) == 0:
                    return None
            cursor = terminal.left
    if terminal.kind == SLICE:
        if terminal.slice_offset() < 0:
            return None
        skip = terminal.slice_offset()
        limit = terminal.slice_length()
        cursor = terminal.left
    # A sort limited to its first rows (#332) streams: each batch keeps
    # its own first rows, which contain the overall first rows, since a
    # row among those is among its own batch's. Batches arrive in input
    # order, so ties keep the stable sort's order.
    var top = -1
    var top_node = -1
    var top_maintain_order = False
    var top_names = List[String]()
    var top_descending = List[Bool]()
    var top_nulls_last = List[Bool]()
    if (
        terminal.kind == SLICE
        and cursor >= 0
        and self._nodes[cursor].kind == SORT
        and self._nodes[cursor].sort_limit() >= 0
    ):
        top = self._nodes[cursor].sort_limit()
        top_node = cursor
        top_maintain_order = self._nodes[cursor].maintain_order
        top_names = self._nodes[cursor].names.copy()
        var n = len(top_names)
        for k in range(n):
            top_descending.append(self._nodes[cursor].sort_descending(k))
            top_nulls_last.append(self._nodes[cursor].sort_nulls_last(k))
        cursor = self._nodes[cursor].left
    # Join filters through an aggregate the stream starts from: the
    # builds above it run first, and the plan with their results runs.
    var through = plan_push_filters_through_aggregate(self, cursor, batch_size)
    if through:
        return _stream_after_materialization(
            through.value(), index, batch_size, sink
        )
    # Every join's build side is chosen before anything runs or is
    # scheduled, from estimates (DuckDB's `BuildProbeSideOptimizer`):
    # the stream then builds each join's right input as it stands.
    var sided = plan_build_probe_sides(self, cursor, batch_size)
    if sided:
        return _stream_after_materialization(
            sided.take(), index, batch_size, sink
        )
    var head = _StreamHead(
        cursor,
        expressions^,
        keys^,
        ordered,
        skip,
        limit,
        grouping,
        top,
        top_node,
        top_maintain_order,
        top_names^,
        top_descending^,
        top_nulls_last^,
    )
    if not self._prefetched:
        return _stream_probe(self, index, batch_size, sink, head^)
    # Each computed input is a MetaPipeline under the stream's (DuckDB's
    # `MetaPipeline::Build`); the probe is the pipeline that depends on
    # them all (`Pipeline::AddDependency`), so builds overlap and the
    # caller only waits. An execution report records one task at a time.
    var inputs = plan_schedule_inputs(self, cursor, batch_size)
    if len(inputs.wanted) == 0:
        return _stream_probe(self, index, batch_size, sink, head^)
    trace_path("stream.scheduled_inputs")
    var count = len(inputs.wanted)
    var pipelines = List[_StreamPipeline](capacity=count + 1)
    var metas = List[MetaPipeline](capacity=count + 1)
    var dependencies = List[Int](capacity=count)
    for k in range(count):
        pipelines.append(_StreamPipeline(inputs.jobs.pop(0)))
        var meta = inputs.metas[k].copy()
        meta.parent = count
        metas.append(meta^)
        dependencies.append(k)
    metas.append(MetaPipeline(META_REGULAR, -1, dependencies))
    pipelines.append(
        _StreamPipeline(
            _StreamProbe(
                Int(Pointer(to=self)),
                index,
                batch_size,
                sink,
                head^,
                inputs.wanted.copy(),
                inputs.keys.copy(),
                inputs.hows.copy(),
            )
        )
    )
    pipelines[count].probe[0].siblings = Int(Pointer(to=pipelines))
    run_pipelines(pipelines, metas, parallel=not self._report)
    var result = Optional[DataFrame]()
    swap(result, pipelines[count].probe[0].result)
    return result^


@fieldwise_init
struct _StreamHead(Movable):
    """What a stream decided before scheduling its inputs, which its
    probe pipeline starts from."""

    var cursor: Int
    var expressions: List[Expr]
    var keys: List[String]
    var ordered: Bool
    var skip: Int
    var limit: Int
    var grouping: Bool
    var top: Int
    var top_node: Int
    var top_maintain_order: Bool
    var top_names: List[String]
    var top_descending: List[Bool]
    var top_nulls_last: List[Bool]


struct _StreamProbe(Movable):
    """A stream's probe pipeline: runs once every input it depends on
    has completed (`Pipeline::GetDependencies`)."""

    var plan: Int
    var index: Int
    var batch_size: Int
    var sink: Int
    var head: List[_StreamHead]
    var wanted: List[Int]
    var keys: List[List[String]]
    var hows: List[Int]
    # The caller's List[_StreamPipeline]; its first `len(wanted)` hold
    # the inputs, this probe after them.
    var siblings: Int
    var result: Optional[DataFrame]

    def __init__(
        out self,
        plan: Int,
        index: Int,
        batch_size: Int,
        sink: Int,
        var head: _StreamHead,
        var wanted: List[Int],
        var keys: List[List[String]],
        var hows: List[Int],
    ):
        self.plan = plan
        self.index = index
        self.batch_size = batch_size
        self.sink = sink
        self.head = List[_StreamHead]()
        self.head.append(head^)
        self.wanted = wanted^
        self.keys = keys^
        self.hows = hows^
        self.siblings = 0
        self.result = None

    def run(mut self) raises:
        ref plan = Pointer[QueryContext, MutAnyOrigin](
            unsafe_from_address=self.plan
        )[]
        ref pipelines = Pointer[List[_StreamPipeline], MutAnyOrigin](
            unsafe_from_address=self.siblings
        )[]
        # Every input has completed: the memo is written by one thread,
        # the plan's `_execute` calls below take from it.
        var ahead = plan._prefetched.value()
        for k in range(len(self.wanted)):
            ref job = pipelines[k].input[0]
            ahead[][self.wanted[k]] = job.result.copy()
            if job.built and plan._prefetched_indexes:
                plan._prefetched_indexes.value()[][
                    self.wanted[k]
                ] = _ReadyIndex(
                    self.keys[k], self.hows[k], job.built.value().copy()
                )
        self.result = _stream_probe(
            plan, self.index, self.batch_size, self.sink, self.head.pop(0)
        )


struct _StreamPipeline(SchedulablePipeline):
    """A stream's schedule holds its inputs and its probe."""

    var input: List[_SubplanPipeline]
    var probe: List[_StreamProbe]

    def __init__(out self, var input: _SubplanPipeline):
        self.input = List[_SubplanPipeline]()
        self.input.append(input^)
        self.probe = List[_StreamProbe]()

    def __init__(out self, var probe: _StreamProbe):
        self.input = List[_SubplanPipeline]()
        self.probe = List[_StreamProbe]()
        self.probe.append(probe^)

    def initialize(mut self) raises:
        pass

    def execute_tasks(self) -> Int:
        return 1

    def execute(mut self, task: Int) raises:
        if len(self.probe) > 0:
            self.probe[0].run()
        else:
            self.input[0].execute(task)

    def prepare_finalize(mut self) raises:
        pass

    def finalize(mut self) raises:
        if len(self.input) > 0:
            self.input[0].finalize()


def _stream_probe(
    self: QueryContext,
    index: Int,
    batch_size: Int,
    sink: Int,
    var head: _StreamHead,
) raises -> Optional[DataFrame]:
    """The stream's probe pipeline, once its inputs are done."""
    var cursor = head.cursor
    var expressions = head.expressions.copy()
    var keys = head.keys.copy()
    var ordered = head.ordered
    var skip = head.skip
    var limit = head.limit
    var grouping = head.grouping
    var top = head.top
    var top_node = head.top_node
    var top_maintain_order = head.top_maintain_order
    var top_names = head.top_names.copy()
    var top_descending = head.top_descending.copy()
    var top_nulls_last = head.top_nulls_last.copy()
    ref terminal = self._nodes[index]
    # A streamed top-N's sink, made at its first batch.
    var topn = List[TopNSink]()
    var top_batches = 0
    var operations = List[PlanNode]()
    # The plan node of each operation, for the execution report.
    var operation_nodes = List[Int]()
    var joins = List[DataFrame]()
    # The plan node of each join built so far.
    var join_nodes = List[Int]()
    var indexes = List[Optional[JoinHashTable]]()
    # A computed source (a union, an aggregate, a sort) is run in full
    # whatever the stream does with it; as an in-memory table its rows
    # run through the morsel pipelines and the aggregate table on every
    # worker, as DuckDB's union children sink into their parent's
    # aggregate. Batched, a grouped reduction over it ran on one thread
    # (TPC-DS q76's three channels). It is found and run before any
    # join below builds, so no build is made twice.
    var first = cursor
    while first >= 0:
        ref step = self._nodes[first]
        if _stream_rows(step):
            first = step.left
        elif _stream_join(step):
            first = step.left
        else:
            break
    if first >= 0 and first != index and not _is_scan(self._nodes[first].kind):
        # Row-local steps all the way up from an aggregate (a filter on
        # its outputs, as SQL's HAVING, and projections): the aggregate's
        # finish runs them on each worker's part of its groups, as
        # DuckDB's pipeline whose source is the aggregate table's scan
        # runs a filter over each scanned chunk, so the groups they drop
        # are never assembled and scanned again.
        var above = _row_steps_above(self, index, first, sink, len(expressions))
        if len(above) > 0 and top < 0 and limit < 0 and skip == 0:
            var plan = self.copy()
            var finish = ArcPointer(
                FinishAbove(first, self._nodes[first].names.copy(), above^)
            )
            plan._finish_above = Optional(finish)
            var aggregated = plan._execute(first, False, True, batch_size)
            if finish[].applied:
                return aggregated^
            plan._finish_above = None
            plan._frames.append(aggregated^)
            plan._schemas.append(Optional[CsvSchema]())
            plan._nodes[first] = _frame_scan_node(slot=len(plan._frames) - 1)
            return _stream_after_materialization(plan, index, batch_size, sink)
        var computed = self._execute(first, False, True, batch_size)
        var plan = self.copy()
        plan._frames.append(computed^)
        plan._schemas.append(Optional[CsvSchema]())
        plan._nodes[first] = _frame_scan_node(slot=len(plan._frames) - 1)
        return _stream_after_materialization(plan, index, batch_size, sink)
    while cursor >= 0:
        ref node = self._nodes[cursor]
        if _stream_rows(node):
            operations.append(node.copy())
            operation_nodes.append(cursor)
        elif _stream_join(node):
            var prepared = Optional[JoinHashTable]()
            var operation = node.copy()
            operation.set_join_build_slot(len(joins))
            var algorithm = String(
                "cross" if node.how == JOIN_CROSS else "eager_hash"
            )
            var built = self._execute(node.right, False, True, batch_size)
            if built.height() == 0 and (
                node.how == JOIN_INNER
                or node.how == JOIN_SEMI
                or node.how == JOIN_RIGHT
            ):
                # No output is possible below this join (DuckDB's
                # `EmptyResultIfRHSIsEmpty`: its sink finalizes as
                # NO_OUTPUT_POSSIBLE and the probe pipeline never runs):
                # the join's result is empty, and nothing under it runs.
                # The steps above it run on no rows, so an aggregate
                # without keys still yields its one row; the builds
                # already made are kept.
                var plan = self.copy()
                for k in range(len(joins)):
                    plan._frames.append(joins[k].copy())
                    plan._schemas.append(Optional[CsvSchema]())
                    plan._nodes.append(
                        _frame_scan_node(slot=len(plan._frames) - 1)
                    )
                    plan._nodes[join_nodes[k]].right = len(plan._nodes) - 1
                plan._frames.append(
                    self._execute(cursor, True, False, batch_size)
                )
                plan._schemas.append(Optional[CsvSchema]())
                plan._nodes[cursor] = _frame_scan_node(
                    slot=len(plan._frames) - 1
                )
                trace_path("stream.empty_build")
                return _stream_after_materialization(
                    plan, index, batch_size, sink
                )
            # The index a prefetch made with this build, if any.
            var ready = Optional[_ReadyIndex]()
            if self._prefetched_indexes:
                var held = self._prefetched_indexes.value()
                if node.right in held[]:
                    ready = held[].pop(node.right)
            # The build finished: one chunk per column, and its keys'
            # hash index, unless a prefetch made the same one already.
            var keys = List[String]()
            if node.how != JOIN_CROSS and len(node.names):
                keys = node.right_keys.copy()
            if (
                len(keys) > 0
                and ready
                and ready.value().keys == keys
                and ready.value().how == node.how
            ):
                prepared = ready.value().index.copy()
                algorithm = "hash_index"
                keys = List[String]()
            var finished = finish_build(built, keys, node.how)
            if finished:
                prepared = finished^
                algorithm = "hash_index"
            if prepared:
                var hash_index = prepared.take()
                hash_index.build_filtered = plan_is_filtering(self, node.right)
                prepared = hash_index^
            joins.append(built^)
            join_nodes.append(cursor)
            plan_record(
                self,
                cursor,
                "streaming",
                0,
                0,
                executions=0,
                build_rows=joins[len(joins) - 1].height(),
                builds=1,
                algorithm=algorithm,
                build_side="right",
            )
            indexes.append(prepared^)
            operations.append(operation^)
            operation_nodes.append(cursor)
        else:
            break
        cursor = node.left
    if cursor < 0 or cursor == index and not _is_scan(self._nodes[cursor].kind):
        return None
    if cursor == index and self._nodes[cursor].kind == SCAN_FRAME:
        # The frame itself, with nothing to apply: batches would only
        # copy it into chunks that the eager step above rechunks
        # again (ClickBench q16 over 10M in-memory rows, 275 -> 181
        # ms).
        return None
    if (
        len(expressions) == 0
        and top < 0
        and limit < 0
        and skip == 0
        and len(joins) == 0
        and self._nodes[cursor].kind == SCAN_FRAME
        and not _filters(operations)
    ):
        # Row-local steps that keep every row of an in-memory frame
        # (with_columns, select, drop) under an eager step: a stream
        # applies them to batches and copies the results back
        # together, while the eager steps share the columns they do
        # not touch (ClickBench q18 adds a minute column to 10M rows
        # before a 4.9M-group aggregation).
        return None
    operations.reverse()
    operation_nodes.reverse()
    var distinct = _counts_distinct(expressions)
    var eligibility = pipeline_eligibility(
        self._nodes[cursor].kind,
        top,
        limit,
        skip,
        operations,
        indexes,
        distinct,
        len(keys),
        len(joins),
        ordered,
        expressions,
    )
    var inline_table_eligible = False
    if consider_inline_table(
        eligibility, len(keys), len(expressions), ordered, distinct
    ):
        var empty = self._frames[self._nodes[cursor].source_slot()].clear()
        var empty_joins = ArcPointer(joins.copy())
        var empty_indexes = ArcPointer(indexes.copy())
        for operation in operations:
            _apply_operation(operation, empty, empty_joins, empty_indexes)
        inline_table_eligible = table_supported(empty, keys, expressions)
    var decision = select_aggregation_strategy(
        self,
        cursor,
        batch_size,
        keys,
        expressions,
        operations,
        joins,
        distinct,
        ordered,
        top,
        grouping,
        eligibility,
        inline_table_eligible,
    )
    if decision.is_eager():
        trace_decision(decision)
        return None
    if top >= 0:
        operations.append(self._nodes[top_node].copy())
        operation_nodes.append(top_node)
    var shared_joins = ArcPointer(joins^)
    var shared_indexes = ArcPointer(indexes^)
    ref source = self._nodes[cursor]
    var csv = List[_CsvBatches]()
    var dataset = List[_DatasetBatches]()
    var parquet = List[_ParquetBatches]()
    var ipc = List[_IpcBatches]()
    var ndjson = List[_NdjsonBatches]()
    var input = DataFrame(List[Series](), height=0)
    if source.has_dataset():
        var columns = Optional[List[String]]()
        if source.dataset_projected:
            columns = source.scan_projection().copy()
        var scan = _CsvScanOptions.defaults()
        if source.kind == SCAN_CSV:
            scan = _scan_options(source)
        var predicate = Optional[Expr]()
        # A source limit precedes these operations. Pruning a file here
        # would change which rows an earlier head selects.
        if (
            source.scan_row_limit() < 0
            and len(operations)
            and operations[0].kind == FILTER
        ):
            predicate = operations[0].exprs[0].copy()
        source._record_execution()
        dataset.append(
            _DatasetBatches(
                source.dataset_spec(),
                source.kind == SCAN_PARQUET,
                schema=self._schemas[source.source_slot()],
                scan=scan,
                columns=columns,
                limit=source.scan_row_limit(),
                batch_size=batch_size,
                predicate=predicate,
            )
        )
    elif source.kind == SCAN_CSV:
        var opened = _open_csv_batches(
            source.text,
            self._schemas[source.source_slot()],
            source.names,
            source.scan_row_limit(),
            batch_size,
            _scan_options(source),
        )
        if not opened:
            return None
        csv.append(opened.take())
    elif source.kind == SCAN_PARQUET:
        var groups = Optional[List[Int]]()
        if len(operations) and operations[0].kind == FILTER:
            groups = _pruned_row_groups(
                parquet_row_group_statistics(source.text),
                operations[0].exprs[0],
            )
        parquet.append(
            _ParquetBatches(source.text, source.names, groups, batch_size)
        )
    elif source.kind == SCAN_NDJSON:
        source._record_execution()
        ndjson.append(
            _NdjsonBatches(
                source.text,
                source.json_schema,
                Int(source.ndjson_infer_schema_length()),
                source.ndjson_ignore_errors(),
                source.names,
                source.scan_row_limit(),
                batch_size,
            )
        )
    elif source.kind == SCAN_IPC:
        source._record_execution()
        ipc.append(
            _IpcBatches(
                source.text,
                source.names,
                source.ipc_has_projection(),
                source.ipc_memory_map(),
                Int(source.ipc_row_offset()),
                source.scan_row_limit(),
                batch_size,
            )
        )
    elif source.kind == SCAN_FRAME:
        source._record_execution()
        input = self._frames[source.source_slot()].copy()
        if len(source.names):
            input = input.select(source.names)
        # A first filter over the frame is the scan's table filters
        # (DuckDB's filter pushdown into the table scan): vectors their
        # statistics rule out are never batched, and parts that keep
        # every row left are not evaluated.
        if (
            len(operations) > 0
            and operations[0].kind == FILTER
            and _row_local(operations[0].exprs)
            and input.height() > 0
        ):
            var scanned = scan_filtered(
                input, conjuncts(operations[0].exprs[0])
            )
            if scanned:
                trace_path("filter.table_filters")
                input = scanned.value().frame.copy()
                operations[0].exprs = [conjunction(scanned.value().remaining)]
    else:
        input = self._execute(cursor, False, True, batch_size)
    trace_decision(decision)
    if decision.is_eager():
        return None
    var workers = configured_workers()
    var pool = Pool(1)
    var pool_ready = False
    var offset = 0
    var emitted = False
    var ended = False
    var outputs = List[DataFrame]()
    # Execution report: rows fed from the source, and rows into and out
    # of each operation, summed over batches.
    var counting = Bool(self._report)
    var fed_rows = 0
    var operation_rows = List[Int](
        length=2 * len(operations) if counting else 0, fill=0
    )
    # Per operation, then the batches' reductions.
    var operation_times = List[Int](
        length=len(operations) + 1 if counting else 0, fill=0
    )
    var stream_began = Int(perf_counter_ns())
    # Aggregate state (#326). Batch states wait in `pending` and are
    # merged together once their groups reach the accumulated count (or
    # 64 batches), so each merge covers at least as much new work as old
    # and total merge work stays linear, with pending state about the
    # size of the accumulated state. When the first batch shows many
    # groups, every batch state is split into hash parts that are merged
    # on separate workers and interleaved by first occurrence at the end.
    var split_groups = _stream_split_groups()
    var acc = _StateAccumulator(workers, split_groups)
    # A grouped reduction whose first batch shows many groups collects
    # the batches (the columns it reads) and runs one eager group-by
    # at the end, where the hash-partitioned reduce encodes every key
    # once and never merges: a state per batch merged through a key
    # index cost TPC-DS q39 (1M groups) 761 ms against DuckDB's 32.
    var collecting = False
    var collected = List[DataFrame]()
    var collected_bytes = 0
    var collected_rows = 0
    var collect_budget = _stream_collect_bytes()
    # The decision waits for a batch with enough rows to judge: an
    # input sorted by date yields empty batches first where a filter
    # keeps a later year. Until then the batches' frames are kept
    # beside their states, to be collected or dropped.
    var undecided = len(expressions) > 0 and len(keys) > 0
    # The columns distinct counts read, judged with the keys: a batch's
    # state holds each group's set of values, so its size is the
    # distinct (key, value) pairs, not the groups. One that reads an
    # expression collects (its values are assumed many).
    var distinct_inputs = _distinct_inputs(expressions)
    if undecided and not distinct_inputs:
        collecting = True
        undecided = False
    var held = List[DataFrame]()
    # Each held batch's state, when its job built one (a batch under
    # 8 * split_groups rows: a few ms at most, in parallel, and used
    # if the stream keeps batch states); a longer batch whose first
    # rows show many groups keeps its frame only, since its state
    # would be the expensive one to drop (TPC-DS q39: 10 ms for 95K
    # rows), and is reduced in a replay round if needed after all.
    var held_states = List[List[_StreamReduction]]()
    var held_rows = 0
    var projection = List[String]()
    if len(expressions) and len(keys):
        var wanted = List[String]()
        var any = True
        for name in keys:
            wanted.append(name)
        for expression in expressions:
            var reads = _references(expression)
            if not reads:
                any = False
                break
            for name in reads.value():
                if name not in wanted:
                    wanted.append(name)
        if any:
            projection = wanted^
    # Batches of a large in-memory input that streams through joins grow
    # up to four times the default (not into the aggregate table, which
    # holds no state per batch: there a morsel's size is what spreads
    # rows a selective join keeps from a few stretches of sorted input
    # over the workers, as TPC-DS q39's date join keeps 180K of 11.7M
    # inventory rows from two of 45 grown batches): each batch probes every join, and
    # fewer, larger batches cost less per row (PDS-H q21's two joins of
    # 3.8M rows against 1.5M-row builds: 186 -> 148 ms). They grow only
    # while every worker still gets a batch: a 150K-row customer table
    # in one batch ran on one worker (q10, q13 and q22 were 7-23% slower).
    # A TopN pushes a dynamic filter into the scan when its first sort
    # column is read from the source as it is: no join, and no operation
    # that writes a column of that name.
    var top_scan_key = top > 0 and len(shared_joins[]) == 0
    if top_scan_key:
        for operation in operations:
            if operation.kind == WITH_COLUMNS or operation.kind == SELECT:
                for e in operation.exprs:
                    if e._name != top_names[0]:
                        continue
                    var r = _references(e)
                    if not (
                        r
                        and len(r.value()) == 1
                        and r.value()[0] == e._name
                        and len(e._nodes) == 1
                    ):
                        top_scan_key = False
    # RowGroupPruner: a table scan under ORDER BY ... LIMIT on a column it
    # reads unchanged is read in its row groups' statistics order, and the
    # row groups that cannot reach the limit are dropped; a filter between
    # them (or the scan's table filters) leaves the order alone but
    # prunes nothing. Dropped leading row groups shorten the offset.
    var top_ranges = Optional[List[Int]]()
    if (
        top_scan_key
        and source.kind == SCAN_FRAME
        and top - skip > 0
        and top_names[0] in input.columns()
        and orderable(input.column(top_names[0]).dtype())
    ):
        var prune = True
        for operation in operations:
            if operation.kind == FILTER:
                prune = False
        var order = row_group_order(
            input.column(top_names[0]),
            top_descending[0],
            top_nulls_last[0],
            top - skip,
            skip,
            prune,
        )
        if order:
            trace_path("pipeline.top_row_group_order")
            top = top - skip + order.value().remaining_offset
            skip = order.value().remaining_offset
            top_ranges = order.value().ranges.copy()
    # Row-local steps over an in-memory frame into a materialize or an
    # ungrouped reduce sink run as a thread-owned pipeline (#538): each
    # worker takes morsels from a shared cursor and runs every step on
    # them; nothing is held, judged or merged on the main thread.
    if (
        sink == 0
        and len(dataset) == 0
        and decision.uses_pipeline()
        and len(csv) == 0
        and len(parquet) == 0
        and len(ipc) == 0
        and len(ndjson) == 0
    ):
        var began = Int(perf_counter_ns())
        var piped_counts = List[Int]()
        # A top-k's TOP_K node was appended to the operations for the
        # stream's per-batch selection; the pipeline's sink keeps the
        # k best itself, so the steps stop before it.
        var step_count = len(operations) - 1 if top > 0 else len(operations)
        var step_nodes = List[PlanNode](capacity=step_count)
        for k in range(step_count):
            step_nodes.append(operations[k].copy())
        # LateMaterialization: a top-N of at most 50 rows over filters and
        # projections on a table scan runs on the columns its order and
        # filters read plus the row id; the other columns are fetched for
        # its rows afterwards.
        var pipe_input = input.copy()
        var late_operations = Optional[List[PlanNode]]()
        if (
            top > 0
            and limit >= 0
            and limit <= LATE_MATERIALIZATION_MAX_ROWS
            and source.kind == SCAN_FRAME
            and len(shared_joins[]) == 0
            and len(keys) == 0
            and len(expressions) == 0
            and not grouping
        ):
            var late = late_materialization_plan(
                input.columns(), step_nodes, top_names
            )
            if late:
                trace_path("pipeline.late_materialization")
                late_operations = step_nodes.copy()
                pipe_input = input.select(late.value().columns)
                step_nodes = late.value().operations.copy()
        var set_frames = List[DataFrame]()
        var set_indices = List[List[Int]]()
        if grouping:
            for members in _set_members(terminal):
                var indices = List[Int]()
                for name in members:
                    var at = keys.index(name) if name in keys else -1
                    if at not in indices:
                        indices.append(at)
                set_indices.append(indices^)
        # Nodes above this aggregate that its finish runs, when the stream
        # over it handed them here (`_row_steps_above`).
        var finish_nodes = List[Int]()
        var finish = Optional[ArcPointer[FinishAbove]]()
        if (
            self._finish_above
            and self._finish_above.value()[].node == index
            and not self._finish_above.value()[].applied
            and terminal.kind == AGG
            and self._finish_above.value()[].keys == terminal.names
            and not grouping
            and top < 0
        ):
            finish = self._finish_above
            finish_nodes = self._finish_above.value()[].above.copy()
        var finish_operations = List[PlanNode](capacity=len(finish_nodes))
        # Join keys pushed into this aggregate's input from above it
        # (`plan_push_filters_through_aggregate`).
        var pushed = List[PushedJoinFilter]()
        if (
            self._pushed_filters
            and self._pushed_filters.value()[].node == index
            and terminal.kind == AGG
        ):
            pushed = self._pushed_filters.value()[].filters.copy()
        for n in finish_nodes:
            finish_operations.append(self._nodes[n].copy())
        var piped_result = run_pipeline(
            pipe_input,
            _steps_of(step_nodes),
            shared_joins,
            shared_indexes,
            expressions,
            keys,
            ordered,
            workers,
            batch_size,
            piped_counts,
            set_frames,
            sets=set_indices,
            top=top,
            top_names=top_names,
            top_descending=top_descending,
            top_nulls_last=top_nulls_last,
            top_maintain_order=top_maintain_order,
            top_scan_key=top_scan_key,
            top_ranges=top_ranges,
            top_row_ids=Bool(late_operations),
            table=decision.uses_aggregate_table(),
            finish_steps=_steps_of(finish_operations),
            pushed=pushed,
        )
        if finish:
            finish.value()[].applied = True
        if grouping:
            if len(set_frames) != len(set_indices):
                return None
            # The empty frame through the steps types the keys a set
            # leaves null.
            var dtypes = List[DataType]()
            for name in terminal.names:
                if name in keys:
                    dtypes.append(piped_result.column(name).dtype())
                else:
                    dtypes.append(DataType.INT64)
            var parts = List[DataFrame](capacity=len(set_frames))
            var members = _set_members(terminal)
            for k in range(len(set_frames)):
                parts.append(
                    _set_rows(
                        terminal, set_frames[k].copy(), members[k], dtypes
                    )
                )
            piped_result = concat(parts)
        if top > 0:
            piped_result = piped_result.slice(skip, limit)
        if late_operations:
            piped_result = late_fetch(
                input, piped_result, late_operations.value()
            )
        if counting:
            plan_record(self, cursor, "pipeline", 0, piped_counts[0])
            var into_terminal = piped_counts[0]
            for k in range(step_count):
                plan_record(
                    self,
                    operation_nodes[k],
                    "pipeline",
                    piped_counts[2 * k + 1],
                    piped_counts[2 * k + 2],
                    busy_ns=piped_counts[2 * step_count + 1 + k],
                )
                into_terminal = piped_counts[2 * k + 2]
            if top > 0:
                plan_record(
                    self,
                    top_node,
                    "pipeline",
                    into_terminal,
                    piped_result.height(),
                )
            var wall = Int(perf_counter_ns()) - began
            if index != cursor and (index not in operation_nodes):
                plan_record(
                    self,
                    index,
                    "pipeline",
                    into_terminal,
                    piped_result.height(),
                    wall_ns=wall,
                )
            else:
                plan_record(
                    self, index, "pipeline", 0, 0, executions=0, wall_ns=wall
                )
            var base = 3 * step_count + 1
            for k in range(len(finish_nodes)):
                plan_record(
                    self,
                    finish_nodes[k],
                    "pipeline",
                    piped_counts[base + 2 * k],
                    piped_counts[base + 2 * k + 1],
                )
        return piped_result^
    var per_round = workers
    # Batches held while undecided, to reduce in the next round once
    # the stream decides not to collect.
    var replay = List[DataFrame]()
    while not ended or len(replay) > 0:
        var jobs = List[_StreamJob]()
        for frame in replay:
            var job = _StreamJob(
                frame.copy(),
                List[PlanNode](),
                shared_joins,
                shared_indexes,
                expressions,
                keys,
            )
            job.bits = acc.bits
            job.counting = counting
            job.replay = True
            jobs.append(job^)
        replay = List[DataFrame]()
        for _ in range(per_round if not ended else 0):
            var frame = DataFrame(List[Series](), height=0)
            var decode = List[_DecodeJob]()
            var json_decode = List[_JsonDecodeJob]()
            if len(dataset):
                var next = dataset[0].next()
                if not next:
                    ended = True
                    break
                frame = next.take()
            elif len(csv):
                var next = csv[0].next()
                if not next:
                    ended = True
                    break
                decode.append(next.take())
            elif len(parquet):
                var next = parquet[0].next()
                if not next:
                    ended = True
                    break
                frame = next.take()
            elif len(ndjson):
                var next = ndjson[0].next()
                if not next:
                    ended = True
                    break
                json_decode.append(next.take())
            elif len(ipc):
                var next = ipc[0].next()
                if not next:
                    ended = True
                    break
                frame = next.take()
            else:
                if emitted and offset >= input.height():
                    ended = True
                    break
                frame = input.slice(offset, batch_size)
                offset += frame.height()
                emitted = True
            var job = _StreamJob(
                frame^,
                operations,
                shared_joins,
                shared_indexes,
                expressions,
                keys,
            )
            job.decode = decode^
            job.json_decode = json_decode^
            job.bits = acc.bits
            job.split_groups = split_groups
            job.counting = counting
            job.collect_only = collecting
            job.keep_frame = undecided
            job.projection = projection.copy()
            jobs.append(job^)
        if len(jobs) == 0 and not (undecided and len(held) > 0):
            break
        # A round can find the input ended with batches still held: it
        # runs no jobs but still judges them below.
        if len(jobs) > 0:
            if not pool_ready:
                pool = Pool(min(len(jobs), workers))
                pool_ready = True
            pool.run(jobs, claim=True)
        if counting:
            for i in range(len(jobs)):
                ref counts = jobs[i].counts
                if jobs[i].replay:
                    continue
                if len(counts) > 0:
                    fed_rows += counts[0]
                elif len(expressions):
                    fed_rows += jobs[i].rows
                else:
                    fed_rows += jobs[i].frame.height()
                for k in range(len(counts)):
                    operation_rows[k] += counts[k]
                for k in range(len(jobs[i].times)):
                    if k < len(operation_times):
                        operation_times[k] += jobs[i].times[k]
        # Pool returns jobs in submission order, independently of worker
        # completion order. Merge states and assemble rows in that order.
        for i in range(len(jobs)):
            if collecting:
                collected_bytes += _frame_bytes(jobs[i].frame)
                collected_rows += jobs[i].rows
                collected.append(jobs[i].frame.copy())
            elif len(expressions) and undecided:
                held.append(jobs[i].frame.copy())
                held_states.append(jobs[i].take_reduced())
                held_rows += jobs[i].rows
                # Judged on a sample of the rows held so far, once
                # there are twice `split_groups` of them: a selective
                # join before the group-by leaves a few hundred rows a
                # batch (TPC-DS q39: 500 of 32K), and one batch alone
                # never decides. The sample stays small: encoding it
                # is on the main thread, 1 ms for 8K rows.
                if held_rows >= 2 * split_groups:
                    undecided = False
                    # Many groups, at least one row in eight
                    # starting one: then most rows would go
                    # through the index on merge. TPC-DS q39
                    # (three rows a group) and q65 gain 35-40%,
                    # PDS-H q16 (one in four) 35%; PDS-H q13 (one
                    # in ten) pays 7% either way.
                    if _judge_held(
                        held,
                        held_states,
                        keys,
                        held_rows,
                        split_groups,
                        distinct_inputs.value(),
                    ):
                        # Many groups: collect every batch held so
                        # far and from here on; the states built
                        # for them are dropped.
                        collecting = True
                        for frame in held:
                            var kept = frame.select(projection) if len(
                                projection
                            ) else frame.copy()
                            collected_bytes += _frame_bytes(kept)
                            collected_rows += kept.height()
                            collected.append(kept^)
                        held = List[DataFrame]()
                        held_states = List[List[_StreamReduction]]()
                        acc.rows_seen = collected_rows
                        for j in range(i + 1, len(jobs)):
                            var kept = (
                                jobs[j]
                                .frame.select(projection) if len(
                                    projection
                                ) else jobs[j]
                                .frame.copy()
                            )
                            collected_bytes += _frame_bytes(kept)
                            collected_rows += jobs[j].rows
                            collected.append(kept^)
                        break
                    # Few groups, or keys too long to collect: the
                    # states, from the held batches and those of
                    # this round still to come, all in input order.
                    for j in range(i + 1, len(jobs)):
                        held.append(jobs[j].frame.copy())
                        held_states.append(jobs[j].take_reduced())
                    if _all_reduced(held_states):
                        for h in range(len(held)):
                            acc.absorb(held_states.pop(0), held[h].height())
                    else:
                        replay = _coalesced(held, 2 * split_groups)
                    held = List[DataFrame]()
                    held_states = List[List[_StreamReduction]]()
                    break
            elif len(expressions):
                acc.absorb(jobs[i].take_reduced(), jobs[i].rows)
            elif top >= 0:
                # The job has already cut its batch to its first rows
                # (its own top-N); they combine into the stream's.
                var batch = jobs[i].frame.copy()
                if top_maintain_order:
                    # Batches arrive in input order: the batch number
                    # and row are the place that settles ties.
                    var places = List[Int64](capacity=batch.height())
                    var base = Int64(top_batches) << 40
                    for r in range(batch.height()):
                        places.append(base + Int64(r))
                    batch = batch.with_column(
                        Series(_ORDER_COLUMN, Column[Int64](places^))
                    )
                top_batches += 1
                if len(topn) == 0:
                    topn.append(
                        TopNSink(
                            batch,
                            _top_keys(top_names, top_maintain_order),
                            _top_flags(top_descending, top_maintain_order),
                            _top_flags(top_nulls_last, top_maintain_order),
                            top,
                            TopNBoundary.none(),
                        )
                    )
                topn[0].sink(batch)
            else:
                var part = jobs[i].frame.copy()
                var dropped = min(skip, part.height())
                skip -= dropped
                part = part.slice(dropped, limit)
                if limit >= 0:
                    limit -= part.height()
                if sink != 0:
                    Pointer[FileSink, MutAnyOrigin](
                        unsafe_from_address=sink
                    )[].append(part)
                else:
                    outputs.append(part^)
        if collecting and collected_bytes >= collect_budget:
            # The budget is reached: reduce what is collected into one
            # state (every key encoded once) and keep collecting. The
            # states merge at the end as batch states do.
            var part = _StreamReduction(concat(collected), expressions, keys)
            part.shift_firsts(acc.rows_seen)
            acc.rows_seen += collected_rows
            collected = List[DataFrame]()
            collected_bytes = 0
            collected_rows = 0
            if len(acc.states) == 0:
                acc.pending.append(List[_StreamReduction]())
                acc.pending_groups.append(0)
                acc.states.append(part^)
            else:
                acc.pending_groups[0] += part.group_count()
                acc.pending[0].append(part^)
        acc.merge(False)
        if len(csv):
            csv[0].discard()
        if limit == 0 and top < 0:
            ended = True
        if ended and undecided and len(held) > 0:
            # The input ran out first: the same judgement on what was
            # held. Many groups among few rows collect too (TPC-DS
            # q7: 5K groups in 7.7K rows), where splitting the states
            # into hash parts cost more than one eager group-by.
            undecided = False
            if _judge_held(
                held,
                held_states,
                keys,
                held_rows,
                split_groups,
                distinct_inputs.value(),
            ):
                collecting = True
                for frame in held:
                    var kept = frame.select(projection) if len(
                        projection
                    ) else frame.copy()
                    collected_bytes += _frame_bytes(kept)
                    collected_rows += kept.height()
                    collected.append(kept^)
                acc.rows_seen = collected_rows
            elif _all_reduced(held_states):
                for h in range(len(held)):
                    acc.absorb(held_states.pop(0), held[h].height())
            else:
                replay = _coalesced(held, 2 * split_groups)
            held = List[DataFrame]()
            held_states = List[List[_StreamReduction]]()
    pool.release()
    if len(csv):
        csv[0].finish()
    if len(dataset):
        dataset[0].finish()
    # NDJSON jobs borrow mapped addresses; retain the source through
    # the final worker join, even after an early head or an error.
    _ = ndjson^
    var result = Optional[DataFrame]()
    if collecting and len(acc.states) == 0:
        trace_path("lazy.collect_group_by")
        # Contiguous first: the bucket encode reads string keys in
        # view storage through a buffer table, slow across hundreds
        # of batches (TPC-DS q39: 10 ms of 40 on 358 batches).
        var whole = concat(collected).rechunk()
        # Groups come back in first-occurrence order only when asked,
        # as the eager group-by's do.
        result = whole.group_by(keys, maintain_order=ordered).agg(expressions)
    elif collecting:
        # Spilled at least once: the rest joins the states.
        trace_path("lazy.collect_group_by_bounded")
        if len(collected):
            var part = _StreamReduction(concat(collected), expressions, keys)
            part.shift_firsts(acc.rows_seen)
            acc.pending_groups[0] += part.group_count()
            acc.pending[0].append(part^)
        acc.merge(True)
        result = acc.states[0].finish()
    elif len(acc.states):
        acc.merge(True)
        if len(acc.states) == 1:
            result = acc.states[0].finish()
        else:
            var parts = List[_StreamReduction]()
            swap(parts, acc.states)
            result = _finish_parts(parts^)
    elif top >= 0:
        if len(topn) > 0:
            var best = topn[0].finish()
            if top_maintain_order:
                best = best.drop([_ORDER_COLUMN])
            result = best.slice(skip, limit)
    elif len(outputs):
        result = concat(outputs)
    if (
        sink != 0
        and not result
        and (
            len(expressions) == 0
            or (len(keys) > 0 and not grouping)
            or top >= 0
        )
    ):
        result = Pointer[FileSink, MutAnyOrigin](
            unsafe_from_address=sink
        )[].schema.copy()
    if counting:
        if _is_scan(source.kind):
            plan_record(self, cursor, "streaming", 0, fed_rows)
        var into_terminal = fed_rows
        for k in range(len(operations)):
            plan_record(
                self,
                operation_nodes[k],
                "streaming",
                operation_rows[2 * k],
                operation_rows[2 * k + 1],
                busy_ns=operation_times[k],
            )
            into_terminal = operation_rows[2 * k + 1]
        var stream_wall = Int(perf_counter_ns()) - stream_began
        if index != cursor and (index not in operation_nodes):
            plan_record(
                self,
                index,
                "streaming",
                into_terminal,
                result.value().height() if result else 0,
                busy_ns=operation_times[len(operations)],
                wall_ns=stream_wall,
            )
        else:
            # The stream's last node is one of its operations: add the
            # stream's wall time there, with nothing else.
            plan_record(
                self,
                index,
                "streaming",
                0,
                0,
                executions=0,
                wall_ns=stream_wall,
            )
    return result^


def _row_steps_above(
    self: QueryContext, index: Int, first: Int, sink: Int, reductions: Int
) -> List[Int]:
    """The nodes from the one reading `first` up to `index`, in that
    order, when `first` is an aggregate and every one of them is a
    row-local filter, projection or column drop with no reduction of its
    own; else none."""
    var nodes = List[Int]()
    if sink != 0 or reductions > 0 or self._nodes[first].kind != AGG:
        return nodes^
    var at = index
    while at != first:
        ref node = self._nodes[at]
        if not (
            _stream_rows(node)
            and (
                node.kind == FILTER
                or node.kind == SELECT
                or node.kind == WITH_COLUMNS
                or node.kind == DROP
            )
        ):
            return List[Int]()
        nodes.append(at)
        at = node.left
    nodes.reverse()
    return nodes^


def _stream_after_materialization(
    plan: QueryContext,
    index: Int,
    batch_size: Int,
    sink: Int,
) raises -> Optional[DataFrame]:
    if sink != 0:
        var streamed = plan_stream_execute(plan, index, batch_size, sink)
        if streamed:
            return streamed^
    return plan._execute(index, False, True, batch_size)


# How a join key fares on its way down to an aggregate
# (`_aggregate_group_key`).
comptime _KEY_DROPPED = 0
comptime _KEY_GROUPED = 1
comptime _KEY_BAIL = 2


def _aggregate_group_key(
    self: QueryContext, index: Int, aggregate: Int, key: String
) raises -> Tuple[Int, String]:
    """Where join key `key`, read from node `index`, goes on the way down
    to `aggregate`, by JoinFilterPushdownOptimizer::GetPushdownFilterTargets:
    through filters, a projection that passes it on (renamed or not), the
    left input of joins and explodes of other columns, to one of the
    aggregate's group keys (_KEY_GROUPED, with its name there). A
    projection that computes it drops it from the filters (_KEY_DROPPED);
    anything else, or a column that is not a group key, ends the join's
    pushdown (_KEY_BAIL)."""
    var at = index
    var name = key
    while at >= 0:
        ref node = self._nodes[at]
        if at == aggregate:
            if name in node.names:
                return (_KEY_GROUPED, name)
            return (_KEY_BAIL, String())
        if node.kind == JOIN:
            # The probe side's columns keep their names; a coalesced key
            # of a right or full join may take the build side's value.
            var passes = node.how in [
                JOIN_INNER,
                JOIN_LEFT,
                JOIN_SEMI,
                JOIN_ANTI,
                JOIN_CROSS,
            ] or (
                (node.how == JOIN_RIGHT or node.how == JOIN_FULL)
                and not node.coalesce
            )
            if not passes or name not in plan_columns_of(self, node.left):
                return (_KEY_BAIL, String())
            at = node.left
            continue
        if node.kind == EXPLODE or node.kind == UNNEST:
            # LOGICAL_UNNEST: not through the column it unnests.
            if name in node.names or name == node.text:
                return (_KEY_BAIL, String())
            if name not in plan_columns_of(self, node.left):
                return (_KEY_BAIL, String())
            at = node.left
            continue
        if not (_stream_rows(node) or node.kind == DROP):
            return (_KEY_BAIL, String())
        var below = plan_input_name(self, at, name)
        if not below:
            if node.kind == SELECT or node.kind == WITH_COLUMNS:
                return (_KEY_DROPPED, String())
            return (_KEY_BAIL, String())
        name = below.value()
        at = node.left
    return (_KEY_BAIL, String())


def _keyed_rows(frame: DataFrame, keys: List[String]) raises -> Int:
    """The rows of a build whose keys are all non-null: those its hash
    table holds (`JoinHashTable.inserted`)."""
    if len(keys) == 1:
        return frame.height() - frame.column(keys[0]).null_count()
    return frame.select(keys).drop_nulls().height()


def plan_push_filters_through_aggregate(
    self: QueryContext, top: Int, batch_size: Int
) raises -> Optional[QueryContext]:
    """DuckDB's join filters through an aggregate's group keys
    (`JoinFilterPushdownOptimizer::GenerateJoinFilters` and
    `GetPushdownFilterTargets`, the LOGICAL_AGGREGATE_AND_GROUP_BY case):
    when the stream below `top` starts from an aggregate, each inner, semi
    or right join above it whose probe keys reach the aggregate as group
    keys (all of them, but those a projection computes) has its build
    side run first, and the aggregate's input pipeline filters its source
    by the build's keys, as a hash join's dynamic filters reach the scan
    under the aggregate. A probe row whose keys no build row has joins
    nothing, and a filter on group keys removes whole groups, so the
    result is unchanged. DuckDB makes the probe side's pipelines wait for
    the build (`PhysicalJoin::BuildJoinPipelines`); here the builds run
    before the stream and replace their inputs, so nothing runs twice.
    None when no join qualifies."""
    if self._pushed_filters:
        return None
    var first = top
    var above = -1
    while first >= 0:
        ref step = self._nodes[first]
        if _stream_rows(step) or _stream_join(step):
            above = first
            first = step.left
        else:
            break
    if (
        first < 0
        or first == top
        or self._nodes[first].kind != AGG
        or len(self._nodes[first].names) == 0
    ):
        return None
    var plan = self.copy()
    var pushed = List[PushedJoinFilter]()
    var probe = top
    while probe >= 0 and probe != first:
        var j = probe
        var node = self._nodes[j].copy()
        probe = node.left
        if node.kind != JOIN or not supports_join_filters(node.how):
            continue
        if len(node.names) == 0 or len(node.names) != len(node.right_keys):
            continue
        var keys = List[Int]()
        var grouped = List[String]()
        var bail = False
        for k in range(len(node.names)):
            var found = _aggregate_group_key(
                self, node.left, first, node.names[k]
            )
            if found[0] == _KEY_BAIL:
                bail = True
                break
            if found[0] == _KEY_GROUPED:
                keys.append(k)
                grouped.append(found[1])
        if bail or len(keys) == 0:
            continue
        var filtered = plan_is_filtering(self, node.right)
        var built = plan._execute(node.right, False, True, batch_size)
        var count = _keyed_rows(built, node.right_keys)
        for i in range(len(keys)):
            pushed.append(
                PushedJoinFilter(
                    grouped[i],
                    built.column(node.right_keys[keys[i]]),
                    len(node.names),
                    count,
                    filtered,
                )
            )
        plan._frames.append(built^)
        plan._schemas.append(Optional[CsvSchema]())
        plan._nodes.append(_frame_scan_node(slot=len(plan._frames) - 1))
        plan._nodes[j].right = len(plan._nodes) - 1
    if len(pushed) == 0:
        return None
    trace_path("stream.join_filter_through_aggregate")
    # The filters belong to this stream's read of the aggregate: a copy of
    # its node takes them, so another reader of the node reads it whole.
    plan._nodes.append(self._nodes[first].copy())
    var aggregate = len(plan._nodes) - 1
    plan._nodes[above].left = aggregate
    plan._pushed_filters = ArcPointer(PushedJoinFilters(aggregate, pushed^))
    return plan^
