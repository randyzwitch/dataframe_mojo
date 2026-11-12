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
from dataframe.planning.cost.model import PipelineEligibility, trace_decision

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
from dataframe.io.csv_types import (
    _map_file,
)
from dataframe.io.ipc import (
    _IpcBatches,
)
from dataframe.io.json_batches import (
    _NdjsonBatches,
)
from dataframe.io.scan_source import (
    ScanBatch,
    ScanSource,
)
from dataframe.io.parquet import (
    _ParquetBatches,
    _pruned_row_groups,
    parquet_row_group_statistics,
)
from dataframe.planning.statistics import PlanStatistics, plan_cardinality
from dataframe.joins.hash_table import JoinHashTable
from dataframe.join_type import (
    JOIN_ANTI,
    JOIN_CROSS,
    join_emits_build_rows,
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
    PipelineRun,
    begin_pipeline,
    finish_pipeline,
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
    CONCAT,
    HANDOFF_OFF,
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
    _retain_frame,
    _stream_collect_bytes,
    _stream_split_groups,
)
from dataframe.query_execution.materialize import (
    _StreamInputs,
    plan_union,
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
from dataframe.storage.retained import RetainedBuffers
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
    var outcome = _stream_head(self, index, batch_size, sink)
    if len(outcome.plan) > 0:
        return _stream_after_materialization(
            outcome.plan[0], index, batch_size, sink
        )
    if len(outcome.head) == 0:
        return None
    var head = outcome.head.pop()
    if not self._prefetched:
        return _stream_probe(self, index, batch_size, sink, head^)
    # Schedule computed inputs as child pipelines. The probe depends on all
    # child completions, allowing independent builds to overlap. An input
    # that is a stream itself adds its own pipelines (its builds, probe and
    # outer scans) to this schedule rather than running a schedule of its
    # own: one queue then feeds every thread, where a nested schedule would
    # run on the one thread that waits for it. Execution reports
    # serialize these tasks to attribute their measurements.
    var inputs = plan_schedule_inputs(self, index, head.cursor, batch_size)
    if len(inputs.wanted) == 0:
        return _stream_probe(self, index, batch_size, sink, head^)
    trace_path("stream.scheduled_inputs")
    var schedule = _StreamSchedule()
    var top = _add_stream(
        schedule,
        Int(Pointer(to=self)),
        index,
        batch_size,
        sink,
        head^,
        inputs^,
        _BuildFor(),
        META_REGULAR,
        False,
    )
    schedule.bind()
    try:
        run_pipelines(
            schedule.pipelines, schedule.metas, parallel=not self._report
        )
    except e:
        # A failed schedule never reaches the probes' finish.
        schedule.release()
        raise e^
    var result = Optional[DataFrame]()
    swap(result, schedule.pipelines[top].probe[0].result)
    return result^


struct _HeadOutcome(Movable):
    """What `_stream_head` decided: no stream (both empty), a plan with
    some inputs materialized for the stream to restart from, or the head
    its probe pipeline starts from."""

    var plan: List[QueryContext]
    var head: List[_StreamHead]

    def __init__(out self):
        self.plan = List[QueryContext]()
        self.head = List[_StreamHead]()

    def __init__(out self, var plan: QueryContext):
        self.plan = List[QueryContext]()
        self.plan.append(plan^)
        self.head = List[_StreamHead]()

    def __init__(out self, var head: _StreamHead):
        self.plan = List[QueryContext]()
        self.head = List[_StreamHead]()
        self.head.append(head^)


def _stream_head(
    self: QueryContext, index: Int, batch_size: Int, sink: Int
) raises -> _HeadOutcome:
    """The stream's terminal (aggregate, top-N, slice) and its build sides
    decided before anything is scheduled."""
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
            return _HeadOutcome()
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
                    return _HeadOutcome()
            cursor = terminal.left
    if terminal.kind == SLICE:
        if terminal.slice_offset() < 0:
            return _HeadOutcome()
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
        return _HeadOutcome(through.take())
    # Every join's build side is chosen before anything runs or is
    # scheduled, from estimates:
    # the stream then builds each join's right input as it stands.
    var sided = plan_build_probe_sides(self, cursor, batch_size)
    if sided:
        return _HeadOutcome(sided.take())
    return _HeadOutcome(
        _StreamHead(
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
    )


@fieldwise_init
struct _BuildFor(Copyable, Movable):
    """The join a scheduled input is the build side of, for its finish
    (`finish_build`): no keys for an input that is not a hash join's."""

    var keys: List[String]
    var how: Int
    var coalesce: Bool
    var probe_estimate: Int

    def __init__(out self):
        self.keys = List[String]()
        self.how = JOIN_CROSS
        self.coalesce = True
        self.probe_estimate = -1


struct _StreamSchedule(Movable):
    """The pipelines of one schedule: every stream's inputs, probe and outer
    scans, those of streams that are inputs of others among them, and the
    plans those streams run on (each its own copy, with its own memo, at
    a fixed address while the schedule runs)."""

    var pipelines: List[_StreamPipeline]
    var metas: List[MetaPipeline]
    var plans: List[ArcPointer[QueryContext]]

    def __init__(out self):
        self.pipelines = List[_StreamPipeline]()
        self.metas = List[MetaPipeline]()
        self.plans = List[ArcPointer[QueryContext]]()

    def bind(mut self):
        """Point every probe and outer scan at the pipeline list, now that
        it no longer grows."""
        var address = Int(Pointer(to=self.pipelines))
        for p in range(len(self.pipelines)):
            if len(self.pipelines[p].probe) > 0:
                self.pipelines[p].probe[0].siblings = address
            if len(self.pipelines[p].outer) > 0:
                self.pipelines[p].outer[0].siblings = address

    def release(mut self):
        for p in range(len(self.pipelines)):
            if len(self.pipelines[p].probe) == 0:
                continue
            ref started = self.pipelines[p].probe[0].started
            if len(started) > 0 and len(started[0].piped) > 0:
                started[0].piped[0].run.release()


def _add_stream(
    mut schedule: _StreamSchedule,
    plan: Int,
    index: Int,
    batch_size: Int,
    sink: Int,
    var head: _StreamHead,
    var inputs: _StreamInputs,
    var build: _BuildFor,
    type: Int,
    is_input: Bool,
) raises -> Int:
    """Add a stream's pipelines to `schedule` and return its probe's: each
    input a pipeline (a stream's own pipelines when the input is one),
    then the probe depending on them, then an outer scan per right or full
    join in the probe's finish group. `build` is the join this stream's
    result is the build side of, `type` its kind there
    (`META_REGULAR` or `META_JOIN_BUILD`)."""
    ref self = Pointer[QueryContext, MutAnyOrigin](unsafe_from_address=plan)[]
    var count = len(inputs.wanted)
    var at = List[Int](capacity=count)
    for k in range(count):
        var builds = _BuildFor(
            inputs.keys[k].copy(),
            inputs.hows[k],
            inputs.coalesces[k],
            inputs.probe_estimates[k],
        )
        at.append(
            _add_input(
                schedule,
                self,
                inputs.wanted[k],
                batch_size,
                builds^,
                inputs.metas[k].type,
            )
        )
    var probe = len(schedule.pipelines)
    for k in range(count):
        schedule.metas[at[k]].parent = probe
    schedule.metas.append(MetaPipeline(type, -1, at))
    schedule.pipelines.append(
        _StreamPipeline(
            _StreamProbe(
                plan,
                index,
                batch_size,
                sink,
                head^,
                inputs.wanted.copy(),
                at.copy(),
                inputs.keys.copy(),
                inputs.hows.copy(),
                build^,
                is_input,
            )
        )
    )
    # Schedule unmatched build-row scans after the probe in operator order,
    # sharing its downstream sink so outer-join rows receive the same later
    # processing.
    var below = schedule.pipelines[probe].probe[0].head[0].cursor
    var previous = probe
    while below >= 0:
        ref step = self._nodes[below]
        if _stream_rows(step):
            below = step.left
            continue
        if not _stream_join(step):
            break
        if join_emits_build_rows(step.how):
            schedule.pipelines.append(_StreamPipeline(_OuterScan(0, probe)))
            schedule.metas.append(
                MetaPipeline(
                    META_REGULAR, -1, List[Int](), finish_group=previous
                )
            )
            previous = len(schedule.pipelines) - 1
        below = step.left
    return probe


def _add_input(
    mut schedule: _StreamSchedule,
    self: QueryContext,
    index: Int,
    batch_size: Int,
    var build: _BuildFor,
    type: Int,
) raises -> Int:
    """Add input `index` of a stream on `self` to `schedule` and return the
    pipeline that makes its result: the input's own stream when it is one
    (on a copy of the plan with a memo of its own), else a subplan
    pipeline that executes it."""
    if self._handoff == HANDOFF_OFF:
        var sub = self.copy()
        sub._prefetched = Optional(ArcPointer(Dict[Int, DataFrame]()))
        sub._prefetched_indexes = Optional(ArcPointer(Dict[Int, _ReadyIndex]()))
        var outcome = _stream_head(sub, index, batch_size, 0)
        if len(outcome.head) > 0:
            var head = outcome.head.pop()
            var owned = ArcPointer(sub^)
            schedule.plans.append(owned)
            var address = Int(Pointer(to=owned[]))
            ref copy = owned[]
            var inputs = plan_schedule_inputs(
                copy, index, head.cursor, batch_size
            )
            return _add_stream(
                schedule,
                address,
                index,
                batch_size,
                0,
                head^,
                inputs^,
                build^,
                type,
                True,
            )
        if len(outcome.plan) > 0:
            # Some inputs materialized for it: the input runs from there.
            schedule.metas.append(MetaPipeline(type))
            schedule.pipelines.append(
                _StreamPipeline(
                    _SubplanPipeline(
                        outcome.plan.pop(),
                        index,
                        True,
                        batch_size,
                        build.keys,
                        build.how,
                        build.coalesce,
                        build.probe_estimate,
                    )
                )
            )
            return len(schedule.pipelines) - 1
    schedule.metas.append(MetaPipeline(type))
    schedule.pipelines.append(
        _StreamPipeline(
            _SubplanPipeline(
                self.copy(),
                index,
                True,
                batch_size,
                build.keys,
                build.how,
                build.coalesce,
                build.probe_estimate,
            )
        )
    )
    return len(schedule.pipelines) - 1


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
    # The pipeline making each input's result.
    var at: List[Int]
    var keys: List[List[String]]
    var hows: List[Int]
    # The join this stream's result is the build side of, when it is an
    # input of another stream, and the hash table its finish makes.
    var build: _BuildFor
    var built: Optional[JoinHashTable]
    # Whether this stream is another's input (`_add_input`); the stream
    # at the schedule's top hands a declined run back to its caller.
    var is_input: Bool
    # The schedule's List[_StreamPipeline].
    var siblings: Int
    var started: List[_ProbeStart]
    var result: Optional[DataFrame]

    def __init__(
        out self,
        plan: Int,
        index: Int,
        batch_size: Int,
        sink: Int,
        var head: _StreamHead,
        var wanted: List[Int],
        var at: List[Int],
        var keys: List[List[String]],
        var hows: List[Int],
        var build: _BuildFor,
        is_input: Bool,
    ):
        self.plan = plan
        self.index = index
        self.batch_size = batch_size
        self.sink = sink
        self.head = List[_StreamHead]()
        self.head.append(head^)
        self.wanted = wanted^
        self.at = at^
        self.keys = keys^
        self.hows = hows^
        self.build = build^
        self.built = None
        self.is_input = is_input
        self.siblings = 0
        self.started = List[_ProbeStart]()
        self.result = None

    def tasks(mut self) raises -> Int:
        """`Pipeline::Schedule`, every input complete: its results in the
        memo (written by this one thread), the probe set up, and a task
        per morsel worker. An execution report runs them as one task (its
        schedule is on one thread), the workers within it as jobs."""
        ref plan = Pointer[QueryContext, MutAnyOrigin](
            unsafe_from_address=self.plan
        )[]
        ref pipelines = Pointer[List[_StreamPipeline], MutAnyOrigin](
            unsafe_from_address=self.siblings
        )[]
        var ahead = plan._prefetched.value()
        for k in range(len(self.wanted)):
            ref made = pipelines[self.at[k]]
            var built = Optional[JoinHashTable]()
            if len(made.input) > 0:
                ahead[][self.wanted[k]] = made.input[0].result.copy()
                built = made.input[0].built.copy()
            else:
                ref stream = made.probe[0]
                ahead[][self.wanted[k]] = stream.result.value().copy()
                built = stream.built.copy()
            if built and plan._prefetched_indexes:
                plan._prefetched_indexes.value()[][
                    self.wanted[k]
                ] = _ReadyIndex(self.keys[k], self.hows[k], built.take())
        self.started.append(
            _probe_start(
                plan, self.index, self.batch_size, self.sink, self.head.pop(0)
            )
        )
        if len(self.started[0].piped) == 0 or plan._report:
            return 1
        return self.started[0].piped[0].run.tasks()

    def execute(mut self, task: Int) raises:
        if len(self.started[0].piped) == 0:
            return
        if Pointer[QueryContext, MutAnyOrigin](
            unsafe_from_address=self.plan
        )[]._report:
            self.started[0].piped[0].run.execute_all()
        else:
            self.started[0].piped[0].run.execute(task)

    def finalize(mut self) raises:
        ref plan = Pointer[QueryContext, MutAnyOrigin](
            unsafe_from_address=self.plan
        )[]
        ref started = self.started[0]
        if len(started.piped) > 0:
            self.result = _probe_finish(plan, started.piped[0])
        else:
            swap(self.result, started.result)
        if not self.result and self.is_input:
            # The stream declined: the input's operator runs on its own,
            # as `plan_execute` runs a node no stream takes.
            plan._declined = self.index
            self.result = plan._execute(
                self.index, False, True, self.batch_size
            )
        if len(self.build.keys) > 0 and self.result:
            var frame = self.result.take()
            self.built = finish_build(
                frame,
                self.build.keys,
                self.build.how,
                self.build.coalesce,
                self.build.probe_estimate,
            )
            self.result = frame^


struct _OuterScan(Movable):
    """A right or full join's unmatched build rows as a pipeline of the
    probe's finish group: runs once the probe (and the round before) has
    executed, each of the probe's workers one task (`PipelineRun`)."""

    var siblings: Int
    var probe: Int
    var active: Bool

    def __init__(out self, siblings: Int, probe: Int):
        self.siblings = siblings
        self.probe = probe
        self.active = False

    def _run(self) -> Pointer[PipelineRun, MutAnyOrigin]:
        ref pipelines = Pointer[List[_StreamPipeline], MutAnyOrigin](
            unsafe_from_address=self.siblings
        )[]
        return Pointer[PipelineRun, MutAnyOrigin](
            to=pipelines[self.probe].probe[0].started[0].piped[0].run
        )

    def tasks(mut self) raises -> Int:
        ref pipelines = Pointer[List[_StreamPipeline], MutAnyOrigin](
            unsafe_from_address=self.siblings
        )[]
        ref probe = pipelines[self.probe].probe[0]
        ref plan = Pointer[QueryContext, MutAnyOrigin](
            unsafe_from_address=probe.plan
        )[]
        if len(probe.started[0].piped) == 0 or plan._report:
            return 1
        ref run = self._run()[]
        self.active = run.begin_round()
        return run.tasks() if self.active else 1

    def execute(mut self, task: Int) raises:
        if self.active:
            self._run()[].execute(task)


struct _StreamPipeline(SchedulablePipeline):
    """A stream's schedule holds its inputs, its probe and the outer scans
    of its right and full joins."""

    var input: List[_SubplanPipeline]
    var probe: List[_StreamProbe]
    var outer: List[_OuterScan]

    def __init__(out self, var input: _SubplanPipeline):
        self.input = List[_SubplanPipeline]()
        self.input.append(input^)
        self.probe = List[_StreamProbe]()
        self.outer = List[_OuterScan]()

    def __init__(out self, var probe: _StreamProbe):
        self.input = List[_SubplanPipeline]()
        self.probe = List[_StreamProbe]()
        self.probe.append(probe^)
        self.outer = List[_OuterScan]()

    def __init__(out self, var outer: _OuterScan):
        self.input = List[_SubplanPipeline]()
        self.probe = List[_StreamProbe]()
        self.outer = List[_OuterScan]()
        self.outer.append(outer^)

    def initialize(mut self) raises:
        pass

    def max_execute_tasks(self) -> Int:
        return configured_workers() if len(self.input) == 0 else 1

    def execute_tasks(mut self) raises -> Int:
        if len(self.probe) > 0:
            return self.probe[0].tasks()
        if len(self.outer) > 0:
            return self.outer[0].tasks()
        return 1

    def execute(mut self, task: Int) raises:
        if len(self.probe) > 0:
            self.probe[0].execute(task)
        elif len(self.outer) > 0:
            self.outer[0].execute(task)
        else:
            self.input[0].execute(task)

    def prepare_finalize(mut self) raises:
        pass

    def finalize(mut self) raises:
        if len(self.probe) > 0:
            self.probe[0].finalize()
        elif len(self.input) > 0:
            self.input[0].finalize()

    def max_finalize_tasks(self) -> Int:
        if len(self.input) > 0:
            return self.input[0].max_finalize_tasks()
        return 0

    def finalize_tasks(mut self) raises -> Int:
        if len(self.input) > 0:
            return self.input[0].finalize_tasks()
        return 0

    def finalize_task(mut self, task: Int) raises:
        self.input[0].finalize_task(task)

    def finish_finalize(mut self) raises:
        if len(self.input) > 0:
            self.input[0].finish_finalize()


def _probe_start(
    self: QueryContext,
    index: Int,
    batch_size: Int,
    sink: Int,
    var head: _StreamHead,
) raises -> _ProbeStart:
    """The stream's probe pipeline set up, once its inputs are done
    (`Pipeline::Schedule`): its morsel workers, not yet run, or its
    result when it ran here (a batch stream, or no pipeline at all)."""
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
    # Run union inputs through morsel pipelines into worker-local aggregate
    # tables. Discover this route before building joins below it, avoiding
    # duplicate build work.
    var first = cursor
    while first >= 0:
        ref step = self._nodes[first]
        if _stream_rows(step):
            first = step.left
        elif _stream_join(step):
            first = step.left
        else:
            break
    if first >= 0 and first != index and self._nodes[first].kind == CONCAT:
        # The union's children, run as pipelines of this stream's schedule
        # when it has one (`plan_schedule_inputs`): their results are the
        # union pipelines' sources, one chunk each, into this stream's
        # operators and sink.
        var plan = self.copy()
        plan._frames.append(plan_union(self, first, True, batch_size))
        plan._schemas.append(Optional[CsvSchema]())
        plan._nodes[first] = _frame_scan_node(slot=len(plan._frames) - 1)
        return _ProbeStart(
            _stream_after_materialization(plan, index, batch_size, sink)
        )
    if first >= 0 and first != index and not _is_scan(self._nodes[first].kind):
        # Run row-local steps during each worker's aggregate scan. Discard
        # rejected groups before assembling a frame, avoiding an additional
        # scan of materialized groups.
        var above = _row_steps_above(self, index, first, sink, len(expressions))
        if len(above) > 0 and top < 0 and limit < 0 and skip == 0:
            var plan = self.copy()
            var finish = ArcPointer(
                FinishAbove(first, self._nodes[first].names.copy(), above^)
            )
            plan._finish_above = Optional(finish)
            var aggregated = plan._execute(first, False, True, batch_size)
            if finish[].applied:
                return _ProbeStart(aggregated^)
            plan._finish_above = None
            plan._frames.append(aggregated^)
            plan._schemas.append(Optional[CsvSchema]())
            plan._nodes[first] = _frame_scan_node(slot=len(plan._frames) - 1)
            return _ProbeStart(
                _stream_after_materialization(plan, index, batch_size, sink)
            )
        var computed = self._execute(first, False, True, batch_size)
        var plan = self.copy()
        plan._frames.append(computed^)
        plan._schemas.append(Optional[CsvSchema]())
        plan._nodes[first] = _frame_scan_node(slot=len(plan._frames) - 1)
        return _ProbeStart(
            _stream_after_materialization(plan, index, batch_size, sink)
        )
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
                # Join semantics prove the result empty: finalize the sink and
                # skip both probe work and unnecessary execution beneath the
                # join.
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
                return _ProbeStart(
                    _stream_after_materialization(plan, index, batch_size, sink)
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
            var stats = PlanStatistics()
            var finished = finish_build(
                built,
                keys,
                node.how,
                node.coalesce,
                plan_cardinality(self, stats, node.left),
            )
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
        return _ProbeStart(None)
    if cursor == index and (
        self._nodes[cursor].kind == SCAN_FRAME or sink == 0
    ):
        # The source itself, with nothing to apply: batches would only
        # copy it into chunks that the eager step above rechunks
        # again (ClickBench q16 over 10M in-memory rows, 275 -> 181
        # ms); a file is read whole by its parallel reader. Only a file
        # sink streams it.
        return _ProbeStart(None)
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
        return _ProbeStart(None)
    operations.reverse()
    operation_nodes.reverse()
    ref source = self._nodes[cursor]
    # A file source's scan state (`ScanSource`), opened once, and its
    # columns, which the pipeline's steps are typed from.
    var file_scan = Optional[ArcPointer[ScanSource]]()
    var source_columns = Optional[DataFrame]()
    var distinct = _counts_distinct(expressions)
    var eligibility = pipeline_eligibility(
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
    if eligibility.allowed and source.kind != SCAN_FRAME:
        var opened = _open_scan_source(self, cursor, operations, batch_size)
        if not opened:
            return _ProbeStart(None)
        var state = opened.take()
        source_columns = state.schema()
        if not source_columns:
            # No batch at all: the eager scan makes the empty result.
            state.finish()
            return _ProbeStart(None)
        file_scan = ArcPointer(state^)
        for column in source_columns.value()._columns:
            if column.dtype().is_categorical():
                # Each batch may carry its own dictionary (Arrow
                # dictionary deltas), which steps compiled once against
                # the first batch's columns would not read: the batches
                # are evaluated one by one.
                eligibility = PipelineEligibility(False, "batch_dictionaries")
                source_columns = None
                break
    var inline_table_eligible = False
    if consider_inline_table(
        eligibility, len(keys), len(expressions), ordered, distinct
    ):
        var empty = source_columns.value().copy() if source_columns else (
            self._frames[source.source_slot()].clear()
        )
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
        return _ProbeStart(None)
    if top >= 0:
        operations.append(self._nodes[top_node].copy())
        operation_nodes.append(top_node)
    var shared_joins = ArcPointer(joins^)
    var shared_indexes = ArcPointer(indexes^)
    var input = DataFrame(List[Series](), height=0)
    if source.kind != SCAN_FRAME:
        if not file_scan:
            var opened = _open_scan_source(self, cursor, operations, batch_size)
            if not opened:
                return _ProbeStart(None)
            file_scan = ArcPointer(opened.take())
    else:
        source._record_execution()
        input = self._frames[source.source_slot()].copy()
        if len(source.names):
            input = input.select(source.names)
        # Apply the first row-local filter as a scan filter. Skip vectors
        # ruled out by statistics and omit predicate evaluation where every
        # remaining row is proven to pass.
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
    trace_decision(decision)
    if decision.is_eager():
        return _ProbeStart(None)
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
    # A grouping the pipeline does not run (its steps or order are the
    # batches'): the batches are collected for the eager group-by, which
    # is the aggregate table over them,
    # and reduced into one state each time they reach the budget.
    var collecting = len(expressions) > 0 and len(keys) > 0
    var collected = List[DataFrame]()
    var collected_memory = RetainedBuffers()
    var collected_rows = 0
    var collect_budget = _stream_collect_bytes()
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
    # A file's batches have no statistics before they are read.
    var top_scan_key = (
        top > 0 and len(shared_joins[]) == 0 and source.kind == SCAN_FRAME
    )
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
    if sink == 0 and decision.uses_pipeline():
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
        var pipe_input = source_columns.value().copy() if source_columns else (
            input.copy()
        )
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
        var run = begin_pipeline(
            pipe_input,
            _steps_of(step_nodes),
            shared_joins,
            shared_indexes,
            expressions,
            keys,
            ordered,
            workers,
            batch_size,
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
            source=file_scan,
            source_rows=_source_rows(self, cursor) if file_scan else 0,
        )
        return _ProbeStart(
            _PipedProbe(
                run^,
                piped_counts^,
                set_frames^,
                finish,
                grouping,
                set_indices^,
                index,
                keys.copy(),
                expressions.copy(),
                top,
                skip,
                limit,
                late_operations^,
                input^,
                counting,
                cursor,
                step_count,
                operation_nodes.copy(),
                top_node,
                began,
                finish_nodes^,
            )
        )
    var per_round = workers
    while not ended:
        var jobs = List[_StreamJob]()
        for _ in range(per_round):
            var frame = DataFrame(List[Series](), height=0)
            var batch = List[ScanBatch]()
            if file_scan:
                # The job decodes a byte range itself.
                var next = file_scan.value()[].take()
                if not next:
                    ended = True
                    break
                batch.append(next.take())
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
            job.scan = batch^
            job.bits = acc.bits
            job.counting = counting
            job.collect_only = collecting
            job.projection = projection.copy()
            jobs.append(job^)
        if len(jobs) == 0:
            break
        if len(jobs) > 0:
            if not pool_ready:
                pool = Pool(min(len(jobs), workers))
                pool_ready = True
            pool.run(jobs, claim=True)
        if counting:
            for i in range(len(jobs)):
                ref counts = jobs[i].counts
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
                _retain_frame(jobs[i].frame, collected_memory)
                collected_rows += jobs[i].rows
                collected.append(jobs[i].frame.copy())
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
        if collecting and collected_memory.bytes >= collect_budget:
            # The budget is reached: reduce what is collected into one
            # state (every key encoded once) and keep collecting. The
            # states merge at the end as batch states do.
            var part = _StreamReduction(concat(collected), expressions, keys)
            part.shift_firsts(acc.rows_seen)
            acc.rows_seen += collected_rows
            collected = List[DataFrame]()
            collected_memory = RetainedBuffers()
            collected_rows = 0
            if len(acc.states) == 0:
                acc.pending.append(List[_StreamReduction]())
                acc.pending_groups.append(0)
                acc.states.append(part^)
            else:
                acc.pending_groups[0] += part.group_count()
                acc.pending[0].append(part^)
        acc.merge(False)
        if file_scan:
            # Every job of the round has decoded its byte range.
            file_scan.value()[].discard()
        if limit == 0 and top < 0:
            ended = True
    pool.release()
    if file_scan:
        file_scan.value()[].finish()
    # Byte ranges borrow mapped addresses; retain the source through the
    # final worker join, even after an early head or an error.
    _ = file_scan^
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
    return _ProbeStart(result^)


def _source_rows(self: QueryContext, cursor: Int) raises -> Int:
    """The scan's estimated rows (its probe side's cardinality for join
    filters), as the planner's statistics give them."""
    var stats = PlanStatistics()
    return plan_cardinality(self, stats, cursor)


def _open_scan_source(
    self: QueryContext,
    cursor: Int,
    operations: List[PlanNode],
    batch_size: Int,
) raises -> Optional[ScanSource]:
    """The file scan at `cursor` opened as a pipeline source, its batches
    `batch_size` rows; `operations` are the steps over it, the first of
    which, a filter, prunes Parquet row groups and dataset files. None
    when a CSV file cannot be mapped (the eager read takes it)."""
    ref source = self._nodes[cursor]
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
        return ScanSource.of(
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
    if source.kind == SCAN_CSV:
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
        return ScanSource.of(opened.take())
    if source.kind == SCAN_PARQUET:
        var groups = Optional[List[Int]]()
        if len(operations) and operations[0].kind == FILTER:
            groups = _pruned_row_groups(
                parquet_row_group_statistics(source.text),
                operations[0].exprs[0],
            )
        return ScanSource.of(
            _ParquetBatches(source.text, source.names, groups, batch_size)
        )
    if source.kind == SCAN_NDJSON:
        source._record_execution()
        return ScanSource.of(
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
    source._record_execution()
    return ScanSource.of(
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


def _probe_finish(
    self: QueryContext, mut piped: _PipedProbe
) raises -> Optional[DataFrame]:
    """The probe pipeline's result, its execute tasks done (`Sink::Finalize`
    and the plan's bookkeeping after it)."""
    var index = piped.index
    ref terminal = self._nodes[index]
    ref piped_counts = piped.counts
    ref set_frames = piped.set_frames
    ref finish = piped.finish
    var grouping = piped.grouping
    ref set_indices = piped.set_indices
    ref keys = piped.keys
    ref expressions = piped.expressions
    var top = piped.top
    var skip = piped.skip
    var limit = piped.limit
    ref late_operations = piped.late_operations
    ref input = piped.input
    var counting = piped.counting
    var cursor = piped.cursor
    var step_count = piped.step_count
    ref operation_nodes = piped.operation_nodes
    var top_node = piped.top_node
    var began = piped.began
    ref finish_nodes = piped.finish_nodes
    var piped_result = finish_pipeline(piped.run, piped_counts, set_frames)
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
                _set_rows(terminal, set_frames[k].copy(), members[k], dtypes)
            )
        piped_result = concat(parts)
    if top > 0:
        piped_result = piped_result.slice(skip, limit)
    if late_operations:
        piped_result = late_fetch(input, piped_result, late_operations.value())
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


def _stream_probe(
    self: QueryContext,
    index: Int,
    batch_size: Int,
    sink: Int,
    var head: _StreamHead,
) raises -> Optional[DataFrame]:
    """The stream's probe pipeline, run on the calling thread's own
    schedule: no input ahead of it, so its workers run as `run_jobs`."""
    var start = _probe_start(self, index, batch_size, sink, head^)
    if len(start.piped) == 0:
        var result = Optional[DataFrame]()
        swap(result, start.result)
        return result^
    ref piped = start.piped[0]
    try:
        piped.run.execute_all()
    except e:
        piped.run.release()
        raise e^
    return _probe_finish(self, piped)


struct _PipedProbe(Movable):
    """A probe set up as a morsel pipeline (`begin_pipeline`): what its
    finish reads."""

    var run: PipelineRun
    var counts: List[Int]
    var set_frames: List[DataFrame]
    var finish: Optional[ArcPointer[FinishAbove]]
    var grouping: Bool
    var set_indices: List[List[Int]]
    var index: Int
    var keys: List[String]
    var expressions: List[Expr]
    var top: Int
    var skip: Int
    var limit: Int
    var late_operations: Optional[List[PlanNode]]
    var input: DataFrame
    var counting: Bool
    var cursor: Int
    var step_count: Int
    var operation_nodes: List[Int]
    var top_node: Int
    var began: Int
    var finish_nodes: List[Int]

    def __init__(
        out self,
        var run: PipelineRun,
        var counts: List[Int],
        var set_frames: List[DataFrame],
        var finish: Optional[ArcPointer[FinishAbove]],
        grouping: Bool,
        var set_indices: List[List[Int]],
        index: Int,
        var keys: List[String],
        var expressions: List[Expr],
        top: Int,
        skip: Int,
        limit: Int,
        var late_operations: Optional[List[PlanNode]],
        var input: DataFrame,
        counting: Bool,
        cursor: Int,
        step_count: Int,
        var operation_nodes: List[Int],
        top_node: Int,
        began: Int,
        var finish_nodes: List[Int],
    ):
        self.run = run^
        self.counts = counts^
        self.set_frames = set_frames^
        self.finish = finish^
        self.grouping = grouping
        self.set_indices = set_indices^
        self.index = index
        self.keys = keys^
        self.expressions = expressions^
        self.top = top
        self.skip = skip
        self.limit = limit
        self.late_operations = late_operations^
        self.input = input^
        self.counting = counting
        self.cursor = cursor
        self.step_count = step_count
        self.operation_nodes = operation_nodes^
        self.top_node = top_node
        self.began = began
        self.finish_nodes = finish_nodes^


struct _ProbeStart(Movable):
    """`_probe_start`'s answer: the result, or a probe set up to run."""

    var result: Optional[DataFrame]
    var piped: List[_PipedProbe]

    def __init__(out self, var result: Optional[DataFrame]):
        self.result = result^
        self.piped = List[_PipedProbe]()

    def __init__(out self, var piped: _PipedProbe):
        self.result = None
        self.piped = List[_PipedProbe]()
        self.piped.append(piped^)


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
    """Push safe join-key filters below an aggregate through its grouping keys.
    A filter on group keys removes whole groups, preserving every retained
    group's aggregate values. Prepare the build inputs first so the probe
    filters use complete key information. Return None when no join permits
    this transformation."""
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
