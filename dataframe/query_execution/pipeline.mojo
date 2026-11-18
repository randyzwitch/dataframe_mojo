"""Streaming pipeline lowering and execution."""
from dataframe.io.dataset_scan import _DatasetBatches
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
    JOIN_RIGHT_ANTI,
    JOIN_RIGHT_SEMI,
    JOIN_SEMI,
)
from dataframe.joins.join_filter import (
    PushedJoinFilter,
    PushedJoinFilters,
    PlannedJoinFilter,
    supports_join_filters,
)
from dataframe.parallel import (
    configured_workers,
)
from dataframe.sorting.row_group_order import orderable, row_group_order
from dataframe.query_execution.late_materialization import (
    LATE_MATERIALIZATION_MAX_ROWS,
    late_fetch,
    late_materialization_plan,
)
from dataframe.pipeline import (
    PipelineTypes,
    _step_schemas,
    pipeline_types,
    _ensure_zones,
    _range_key,
    _window_bounds,
    PipelineFinish,
    PipelineRun,
    begin_pipeline,
    finish_pipeline,
)
from dataframe.planning.context import (
    FinishAbove,
    QueryContext,
    SharedResults,
    _ReadyIndex,
    plan_shared_id,
    plan_slot_table,
)
from dataframe.planning.inputs import (
    BuildFinish,
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
    plan_remember_schemas,
    plan_columns_of,
    plan_filters_rows,
    plan_grouped_on,
    plan_input_name,
    plan_keeps_column,
    plan_key_source,
)
from dataframe.planning.model import (
    AGG,
    ASOF,
    CONCAT,
    GATHER,
    HCONCAT,
    INSERT_COLUMN,
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
    _column_projection,
    _is_scan,
    _same_names,
    _set_members,
    _stream_join,
    _stream_rows,
)
from dataframe.planning.cost.eligibility import (
    _counts_distinct,
)
from dataframe.planning.expression_inputs import (
    _references,
)
from dataframe.planning.scans import (
    _scan_options,
)
from dataframe.query_execution.materialize import (
    _StreamInputs,
    _stack_union,
    plan_take_prefetched,
    plan_union_children,
    plan_schedule_inputs,
)
from dataframe.query_execution.operators import (
    _set_rows,
)
from dataframe.query_execution.reporting import (
    plan_record,
)
from dataframe.query_execution.stream_state import (
    _steps_of,
)
from dataframe.series import (
    Series,
)
from dataframe.table_scan import conjunction, scan_filtered
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
    if not self._prefetched or not self._prefetched_indexes:
        # A stream reads what its input pipelines made from a memo: a
        # plan with none runs on a copy that has one.
        var plan = self.copy()
        plan._prefetched = Optional(ArcPointer(Dict[Int, DataFrame]()))
        plan._prefetched_indexes = Optional(
            ArcPointer(Dict[Int, _ReadyIndex]())
        )
        return plan_stream_execute(plan, index, batch_size, sink)
    var outcome = _stream_head(self, index, batch_size, sink)
    if len(outcome.plan) > 0:
        return _stream_after_materialization(
            outcome.plan[0], index, batch_size, sink
        )
    if len(outcome.head) == 0:
        return None
    var head = outcome.head.pop()
    # Schedule computed inputs as child pipelines. The probe depends on all
    # child completions, allowing independent builds to overlap. An input
    # that is a stream itself adds its own pipelines (its builds, probe and
    # outer scans) to this schedule rather than running a schedule of its
    # own: one queue then feeds every thread, where a nested schedule would
    # run on the one thread that waits for it. Execution reports
    # serialize these tasks to attribute their measurements.
    var inputs = _scheduled_inputs(self, index, head, batch_size, sink)
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
    return _run_schedule(self, schedule, top)


def _run_schedule(
    self: QueryContext, mut schedule: _StreamSchedule, top: Int
) raises -> Optional[DataFrame]:
    """Run `schedule`'s pipelines and return the result of its stream
    whose probe is pipeline `top`."""
    schedule.bind()
    try:
        run_pipelines(
            schedule.pipelines, schedule.metas, one_event=Bool(self._report)
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


def _remembering(self: QueryContext) -> Bool:
    """Keep each node's schema on `self` once derived, until
    `_forgetting`: deciding a stream's build sides and its inputs asks for
    the columns of the same subtrees at every join of its spine, and the
    nodes stay as they are meanwhile (a side flip rewrites a copy, under a
    selection that keeps every node's output). True when this call started
    remembering."""
    ref plan = Pointer[QueryContext, MutAnyOrigin](
        unsafe_from_address=Int(Pointer(to=self))
    )[]
    return plan_remember_schemas(plan)


def _forgetting(self: QueryContext, owned: Bool):
    """End what `_remembering` started, when it started it."""
    if not owned:
        return
    ref plan = Pointer[QueryContext, MutAnyOrigin](
        unsafe_from_address=Int(Pointer(to=self))
    )[]
    plan._schema_memo = None


def _stream_head(
    self: QueryContext, index: Int, batch_size: Int, sink: Int
) raises -> _HeadOutcome:
    """The stream's terminal (aggregate, top-N, slice) and its build sides
    decided before anything is scheduled."""
    var owned = _remembering(self)
    try:
        var outcome = _decide_head(self, index, batch_size, sink)
        _forgetting(self, owned)
        for k in range(len(outcome.plan)):
            # A rewritten plan's nodes differ: it starts with no schema
            # remembered.
            outcome.plan[k]._schema_memo = None
        return outcome^
    except e:
        _forgetting(self, owned)
        raise e^


def _scheduled_inputs(
    self: QueryContext,
    terminal: Int,
    head: _StreamHead,
    batch_size: Int,
    sink: Int,
) raises -> _StreamInputs:
    """`plan_schedule_inputs`, each node's schema derived once. When the
    computed source is an aggregate under only row-local steps, its input
    pipeline's finish runs those steps too (`FinishAbove`): its rows are
    then the stream's result."""
    var first = head.cursor
    while first >= 0 and (
        _stream_rows(self._nodes[first]) or _stream_join(self._nodes[first])
    ):
        first = self._nodes[first].left
    var above = List[Int]()
    if (
        first >= 0
        and first != terminal
        and head.top < 0
        and head.limit < 0
        and head.skip == 0
    ):
        above = _row_steps_above(self, terminal, first, len(head.expressions))
    var owned = _remembering(self)
    try:
        var inputs = plan_schedule_inputs(
            self, terminal, head.cursor, batch_size
        )
        _forgetting(self, owned)
        if len(above) > 0 and first in inputs.wanted:
            inputs.finish = ArcPointer(
                FinishAbove(first, self._nodes[first].names.copy(), above^)
            )
        return inputs^
    except e:
        _forgetting(self, owned)
        raise e^


def _decide_head(
    self: QueryContext, index: Int, batch_size: Int, sink: Int
) raises -> _HeadOutcome:
    """`_stream_head`'s decisions."""
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
        # So does a grouped reduction of an aggregate filter.
        if _stream_reductions(
            terminal.exprs,
            terminal.kind == AGG,
            terminal.kind == AGG,
            terminal.kind == AGG,
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
    # Every join's build side is chosen before anything runs or is
    # scheduled, from estimates:
    # the stream then builds each join's right input as it stands.
    var sided = plan_build_probe_sides(self, cursor, batch_size)
    if sided:
        return _HeadOutcome(sided.take())
    # Then join filters through an aggregate the stream starts from, on
    # the sides just chosen (a filter only flows from a build into its
    # probe side, so sides come first): the aggregate's pipeline waits
    # for those builds (`_add_stream`).
    var through = plan_push_filters_through_aggregate(self, cursor)
    if through:
        return _HeadOutcome(through.take())
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
            Dict[Int, Int](),
            Dict[Int, Bool](),
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
    var build_filtered: Bool

    def __init__(out self):
        self.keys = List[String]()
        self.how = JOIN_CROSS
        self.coalesce = True
        self.probe_estimate = -1
        self.build_filtered = False


struct _StreamSchedule(Movable):
    """The pipelines of one schedule: every stream's inputs, probe and outer
    scans, those of streams that are inputs of others among them, and the
    plans those streams run on (each its own copy, with its own memo, at
    a fixed address while the schedule runs)."""

    var pipelines: List[_StreamPipeline]
    var metas: List[MetaPipeline]
    var plans: List[ArcPointer[QueryContext]]
    # The pipeline writing each shared subplan's result, by its id, for
    # the shared subplans this schedule makes (`plan_run_shared`).
    var shared: Dict[Int, Int]

    def __init__(out self):
        self.pipelines = List[_StreamPipeline]()
        self.metas = List[MetaPipeline]()
        self.plans = List[ArcPointer[QueryContext]]()
        self.shared = Dict[Int, Int]()

    def bind(mut self):
        """Point every probe, outer scan, operator and shared result's sink
        at the pipeline list, now that it no longer grows."""
        var address = Int(Pointer(to=self.pipelines))
        for p in range(len(self.pipelines)):
            if len(self.pipelines[p].probe) > 0:
                self.pipelines[p].probe[0].siblings = address
            if len(self.pipelines[p].outer) > 0:
                self.pipelines[p].outer[0].siblings = address
            if len(self.pipelines[p].input) > 0:
                self.pipelines[p].input[0].siblings = address
            if len(self.pipelines[p].shared) > 0:
                self.pipelines[p].shared[0].siblings = address

    def reads(self, plan: QueryContext, index: Int) -> List[Int]:
        """The pipelines writing the shared subplan results that the
        subtree at `index` of `plan` scans, which a pipeline running that
        subtree depends on. A subtree runs in the pipeline that makes it
        or in the ones it depends on, so the pipeline at its top waits for
        every result scanned below it."""
        var found = List[Int]()
        if len(self.shared) == 0 or len(plan._shared_slots) == 0:
            return found^
        var stack: List[Int] = [index]
        while len(stack) > 0:
            ref node = plan._nodes[stack.pop()]
            if node.kind == SCAN_FRAME:
                var at = self.shared.get(
                    plan_shared_id(plan, node.source_slot())
                )
                if at and at.value() not in found:
                    found.append(at.value())
            if node.left >= 0:
                stack.append(node.left)
            if node.right >= 0:
                stack.append(node.right)
        return found^

    def release(mut self):
        for p in range(len(self.pipelines)):
            if len(self.pipelines[p].probe) == 0:
                continue
            ref probe = self.pipelines[p].probe[0]
            ref started = probe.started
            if len(started) > 0 and len(started[0].piped) > 0:
                started[0].piped[0].run.release()
            if len(probe.finish) > 0:
                probe.finish[0].release()


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
    ref self = Pointer[QueryContext, ImmutAnyOrigin](unsafe_from_address=plan)[]
    var count = len(inputs.wanted)
    for k in range(count):
        if inputs.joins[k] >= 0:
            head.probe_estimates[inputs.joins[k]] = inputs.probe_estimates[k]
            head.build_filtered[inputs.joins[k]] = inputs.build_filtered[k]
    var at = List[Int](capacity=count)
    for k in range(count):
        var builds = _BuildFor(
            inputs.keys[k].copy(),
            inputs.hows[k],
            inputs.coalesces[k],
            inputs.probe_estimates[k],
            inputs.build_filtered[k],
        )
        var finish = Optional[ArcPointer[FinishAbove]]()
        if inputs.finish and inputs.finish.value()[].node == inputs.wanted[k]:
            finish = inputs.finish
        at.append(
            _add_input(
                schedule,
                self,
                inputs.wanted[k],
                batch_size,
                builds^,
                inputs.metas[k].type,
                finish=finish,
            )
        )
    if self._pushed_filters:
        # The aggregate the probe reads waits for the builds whose keys
        # filter its input: it can start only once their keys are known,
        # and reading fewer rows into the aggregate is the point.
        ref pushed = self._pushed_filters.value()[]
        var source = -1
        for k in range(count):
            if inputs.wanted[k] == pushed.node:
                source = at[k]
        if source >= 0 and len(schedule.pipelines[source].probe) > 0:
            for planned in pushed.planned:
                var right = self._nodes[planned.join].right
                for k in range(count):
                    if inputs.wanted[k] == right and at[k] != source:
                        schedule.metas[source].dependencies.append(at[k])
                        schedule.pipelines[source].probe[0].through.append(
                            _ThroughBuild(at[k], planned.copy())
                        )
                        break
    var probe = len(schedule.pipelines)
    for k in range(count):
        schedule.metas[at[k]].parent = probe
    var waits = at.copy()
    waits.extend(schedule.reads(self, index))
    schedule.metas.append(MetaPipeline(type, -1, waits))
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
    schedule.pipelines[probe].probe[0].finish_above = inputs.finish
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
    streaming: Bool = True,
    sink: Int = 0,
    finish: Optional[ArcPointer[FinishAbove]] = None,
) raises -> Int:
    """Add input `index` of a stream on `self` to `schedule` and return the
    pipeline that makes its result: the input's own stream when it is one
    (on a copy of the plan with a memo of its own), else an operator
    pipeline that runs its node over whole inputs, those inputs added
    before it as pipelines of the same schedule. `sink` receives a
    stream's rows. `finish`: the row steps above this input, an
    aggregate, that its finish may run for the stream reading it."""
    if streaming and self._handoff == HANDOFF_OFF:
        var sub = self.copy()
        sub._finish_above = finish
        # Filters pushed into this input arrive when its builds finish
        # (`_StreamProbe.tasks`); the input plans its own below it.
        sub._pushed_filters = None
        sub._prefetched = Optional(ArcPointer(Dict[Int, DataFrame]()))
        sub._prefetched_indexes = Optional(ArcPointer(Dict[Int, _ReadyIndex]()))
        var outcome = _stream_head(sub, index, batch_size, sink)
        while len(outcome.plan) > 0:
            # Build sides chosen (or filters pushed): the input streams on
            # the plan that made, still in this schedule.
            sub = outcome.plan.pop()
            outcome = _stream_head(sub, index, batch_size, sink)
        if len(outcome.head) > 0 and not _operator_only(
            sub, index, outcome.head[0]
        ):
            var head = outcome.head.pop()
            var owned = ArcPointer(sub^)
            schedule.plans.append(owned)
            var address = Int(Pointer(to=owned[]))
            ref copy = owned[]
            var inputs = _scheduled_inputs(copy, index, head, batch_size, 0)
            return _add_stream(
                schedule,
                address,
                index,
                batch_size,
                sink,
                head^,
                inputs^,
                build^,
                type,
                True,
            )
    var wanted = List[Int]()
    var at = List[Int]()
    if streaming and self._handoff == HANDOFF_OFF:
        # The operator's own inputs run first, in this schedule: run inside
        # the operator's task, each would be a schedule of its own, whose
        # waiting caller could help only schedules started after it.
        wanted = _operator_inputs(self, index)
        for child in wanted:
            at.append(
                _add_input(
                    schedule, self, child, batch_size, _BuildFor(), META_REGULAR
                )
            )
    if len(at) > 0:
        trace_path("stream.operator_inputs")
    var operator = len(schedule.pipelines)
    for p in at:
        schedule.metas[p].parent = operator
    # It also waits for the shared results its subtree scans.
    var deps = at.copy()
    for p in schedule.reads(self, index):
        if p not in deps:
            deps.append(p)
    schedule.metas.append(MetaPipeline(type, -1, deps^))
    schedule.pipelines.append(
        _StreamPipeline(
            _OperatorPipeline(
                self.copy(),
                index,
                streaming,
                batch_size,
                wanted^,
                at^,
                build^,
            )
        )
    )
    return operator


def _operator_only(self: QueryContext, index: Int, head: _StreamHead) -> Bool:
    """Whether the stream at `index` would only hand its node back to run
    on its own (`_probe_start` finds no step to stream: the node is the
    stream's first and is neither a scan nor a row step or join)."""
    if head.cursor != index:
        return False
    ref node = self._nodes[index]
    return not (_is_scan(node.kind) or _stream_rows(node) or _stream_join(node))


def _operator_inputs(self: QueryContext, index: Int) -> List[Int]:
    """The computed nodes the operator at `index` reads as whole frames
    when it runs (`plan_execute_node`), each once. Scans are left out: the
    operator reads them where it runs, as its own source."""
    ref node = self._nodes[index]
    var children = List[Int]()
    if node.left < 0:
        return children^
    if node.kind == CONCAT:
        children = plan_union_children(self, index)
    elif node.kind == SELECT and self._nodes[node.left].kind == SORT:
        if _column_projection(node):
            # The projection takes the sort's input and orders it itself.
            children.append(self._nodes[node.left].left)
        else:
            children.append(node.left)
    elif (
        node.kind == JOIN
        or node.kind == ASOF
        or node.kind == HCONCAT
        or node.kind == GATHER
        or node.kind == INSERT_COLUMN
    ):
        children.append(node.left)
        children.append(node.right)
    else:
        children.append(node.left)
    var wanted = List[Int]()
    for child in children:
        if (
            child >= 0
            and not _is_scan(self._nodes[child].kind)
            and child not in wanted
        ):
            wanted.append(child)
    return wanted^


def plan_execute_inputs(
    self: QueryContext, nodes: List[Int], streaming: Bool, batch_size: Int
) raises -> List[DataFrame]:
    """Independent inputs `nodes`, computed as pipelines of one
    schedule: an input that is a stream adds its own pipelines
    (`_add_input`), any other runs as one subplan pipeline."""
    var schedule = _StreamSchedule()
    var at = List[Int](capacity=len(nodes))
    for node in nodes:
        at.append(
            _add_input(
                schedule,
                self,
                node,
                batch_size,
                _BuildFor(),
                META_REGULAR,
                streaming,
            )
        )
    schedule.bind()
    try:
        run_pipelines(
            schedule.pipelines, schedule.metas, one_event=Bool(self._report)
        )
    except e:
        schedule.release()
        raise e^
    var frames = List[DataFrame](capacity=len(at))
    for p in at:
        frames.append(schedule.pipelines[p].result())
    return frames^


def plan_union(
    self: QueryContext, index: Int, streaming: Bool, batch_size: Int
) raises -> DataFrame:
    """The union at node `index` with its children computed as pipelines of
    one schedule (every child is a pipeline of that schedule, none
    nested in another's, so all of them share the threads at once). A
    child a stream already ran (`plan_schedule_inputs`) is taken."""
    var children = plan_union_children(self, index)
    var got = Dict[Int, DataFrame]()
    var wanted = List[Int]()
    for child in children:
        if child in got or child in wanted:
            continue
        var ready = plan_take_prefetched(self, child)
        if ready:
            got[child] = ready.take()
        else:
            wanted.append(child)
    if len(wanted) > 0:
        var frames = plan_execute_inputs(self, wanted, streaming, batch_size)
        for k in range(len(wanted)):
            got[wanted[k]] = frames[k].copy()
    var frames = List[DataFrame](capacity=len(children))
    for child in children:
        frames.append(got[child].copy())
    return _stack_union(frames)


def plan_run_shared(
    self: QueryContext,
    shared: List[QueryContext],
    index: Int,
    streaming: Bool,
    batch_size: Int,
    sink: Int = 0,
) raises -> DataFrame:
    """Node `index` (the query) with the shared subplans it reads, as
    pipelines of one schedule. `shared[k]` is the plan of shared subplan
    `k`, whose result it reads from slots of its own when it scans
    another (always one with a smaller id). Each subplan's pipelines end
    in a sink that keeps its result for the scans of it, and every
    pipeline scanning one depends on that sink, so the query's other
    pipelines run while it is made. `sink` receives the query's rows
    when it streams."""
    # The subplans the query reads, directly or through others'.
    var needed = List[Bool](length=len(shared), fill=False)
    _mark_shared_reads(self, index, needed)
    for back in range(len(shared)):
        var id = len(shared) - 1 - back
        if needed[id]:
            _mark_shared_reads(shared[id], len(shared[id]._nodes) - 1, needed)
    var schedule = _StreamSchedule()
    for id in range(len(shared)):
        if not needed[id]:
            continue
        trace_path("stream.shared_pipeline")
        ref plan = shared[id]
        var made = _add_input(
            schedule,
            plan,
            len(plan._nodes) - 1,
            batch_size,
            _BuildFor(),
            META_REGULAR,
            streaming,
        )
        schedule.metas.append(MetaPipeline(META_REGULAR, -1, [made]))
        schedule.pipelines.append(
            _StreamPipeline(_SharedSink(made, id, self._shared.value()))
        )
        schedule.shared[id] = len(schedule.pipelines) - 1
    var top = _add_input(
        schedule,
        self,
        index,
        batch_size,
        _BuildFor(),
        META_REGULAR,
        streaming,
        sink,
    )
    schedule.bind()
    try:
        run_pipelines(
            schedule.pipelines, schedule.metas, one_event=Bool(self._report)
        )
    except e:
        schedule.release()
        raise e^
    return schedule.pipelines[top].result()


def _mark_shared_reads(self: QueryContext, index: Int, mut needed: List[Bool]):
    """Mark the shared subplans the subtree at `index` scans."""
    if len(self._shared_slots) == 0:
        return
    var stack: List[Int] = [index]
    while len(stack) > 0:
        ref node = self._nodes[stack.pop()]
        if node.kind == SCAN_FRAME:
            var id = plan_shared_id(self, node.source_slot())
            if id >= 0:
                needed[id] = True
        if node.left >= 0:
            stack.append(node.left)
        if node.right >= 0:
            stack.append(node.right)


@fieldwise_init
struct _StreamHead(Copyable, Movable):
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
    # Each join's probe-side estimate and whether its build side filters
    # its rows, by the join's node, as the plan gave them when its
    # inputs were scheduled (`plan_schedule_inputs`): the probe reads them
    # rather than estimating the same plan again once its builds finish.
    var probe_estimates: Dict[Int, Int]
    var build_filtered: Dict[Int, Bool]


@fieldwise_init
struct _ThroughBuild(Copyable, Movable):
    """A build of the schedule (`pipeline`) whose keys filter the input of
    the aggregate a probe computes (`plan_push_filters_through_aggregate`)."""

    var pipeline: Int
    var planned: PlannedJoinFilter


struct _StreamProbe(Movable):
    """A stream's probe pipeline: runs once every input it depends on has
    completed, since it reads their results."""

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
    # Whether the table was made by the probe pipeline's sink, from the
    # rows as they came: the result is then the build's columns, empty.
    var sank: Bool
    # Whether this stream is another's input (`_add_input`); the stream
    # at the schedule's top hands a declined run back to its caller.
    var is_input: Bool
    # The schedule's List[_StreamPipeline].
    var siblings: Int
    # The builds whose keys filter this stream's aggregate input; the
    # schedule runs them first.
    var through: List[_ThroughBuild]
    var started: List[_ProbeStart]
    # What the probe's setup reads from the plan alone, made in the
    # pipeline's first stage while its inputs still run (`_probe_types`).
    var types: Optional[_ProbeTypes]
    # The morsel pipeline's finish once its sink has finished, and the
    # tasks of the finish's round under way (the pipeline's or the batch
    # stream's).
    var finish: List[PipelineFinish]
    var round: Int
    # Whether the current round fills the built table's pointer table.
    var filling: Bool
    var result: Optional[DataFrame]
    # The row steps above the aggregate input this stream reads, which
    # that input's finish runs when it can (`FinishAbove`).
    var finish_above: Optional[ArcPointer[FinishAbove]]
    # When the node declined to stream at run time: the plan it declined
    # on, when not this one (its computed source is a frame there), and
    # the node below it whose rows this pipeline makes instead (-1 for
    # none), which the node then runs over (`_decline`).
    var declined: List[QueryContext]
    var below: Int
    # The build side's table made from its result frame, as rounds.
    var building: List[BuildFinish]

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
        self.sank = False
        self.is_input = is_input
        self.siblings = 0
        self.through = List[_ThroughBuild]()
        self.started = List[_ProbeStart]()
        self.types = None
        self.finish = List[PipelineFinish]()
        self.round = 0
        self.filling = False
        self.result = None
        self.finish_above = None
        self.declined = List[QueryContext]()
        self.below = -1
        self.building = List[BuildFinish]()

    def initialize(mut self):
        """The probe's setup that reads no input's rows, made as the
        schedule starts, at the same time as the inputs it waits for:
        each step's columns, the steps compiled, the aggregate table's
        layout (`_probe_types`). A probe set up before the schedule has
        none to make."""
        if len(self.started) > 0 or len(self.head) == 0:
            return
        ref plan = Pointer[QueryContext, MutAnyOrigin](
            unsafe_from_address=self.plan
        )[]
        try:
            self.types = _probe_types(
                plan, self.index, self.head[0], self.batch_size
            )
        except:
            # The setup is made again where the inputs' results are
            # read, which raises what is wrong.
            self.types = None

    def tasks(mut self) raises -> Int:
        """Called once every input is complete: its results in the memo (written
        by this one thread), the probe set up, and a task per morsel worker."""
        ref plan = Pointer[QueryContext, MutAnyOrigin](
            unsafe_from_address=self.plan
        )[]
        if len(self.started) > 0:
            # Set up before the schedule (no input to wait for).
            return self._execute_tasks(plan)
        ref pipelines = Pointer[List[_StreamPipeline], MutAnyOrigin](
            unsafe_from_address=self.siblings
        )[]
        for k in range(len(self.wanted)):
            var ahead = plan._prefetched.value()
            ref made = pipelines[self.at[k]]
            ahead[][self.wanted[k]] = made.result()
            var built = made.built()
            var sank = len(made.probe) > 0 and made.probe[0].sank
            if built and plan._prefetched_indexes:
                plan._prefetched_indexes.value()[][
                    self.wanted[k]
                ] = _ReadyIndex(self.keys[k], self.hows[k], built.take(), sank)
        if len(self.through) > 0:
            # The builds above finished: their keys filter this aggregate's
            # input, published at run time rather than planned from data.
            var filters = List[PushedJoinFilter]()
            for t in self.through:
                filters.extend(
                    _published_filters(t.planned, pipelines[t.pipeline])
                )
            plan._pushed_filters = ArcPointer(
                PushedJoinFilters(
                    self.index, List[PlannedJoinFilter](), filters^
                )
            )
        var types = Optional[_ProbeTypes]()
        swap(types, self.types)
        var head = self.head.pop(0)
        var below = _rows_below(head)
        var started = _probe_start(
            plan,
            self.index,
            self.batch_size,
            self.sink,
            head^,
            self.build,
            types^,
            self.finish_above,
        )
        if not started.runs() and not started.result:
            started = self._decline(started^, below^)
        self.started.append(started^)
        return self._execute_tasks(plan)

    def _on(self) -> Int:
        """The address of the plan the node runs on after declining."""
        if len(self.declined) > 0:
            return Int(Pointer(to=self.declined[0]))
        return self.plan

    def _decline(
        mut self, var started: _ProbeStart, var below: _StreamHead
    ) raises -> _ProbeStart:
        """The node declined to stream once its inputs were known (a
        strategy or column types its pipeline does not take): this
        pipeline makes the rows of the steps and joins below the node
        instead, and the node runs over them as it completes. The stream
        at the schedule's top, on its own plan, hands the node back to
        its caller."""
        if len(started.declined) > 0:
            self.declined.append(started.declined.pop())
            # Estimates on that plan differ: its source is a frame.
            below.probe_estimates = Dict[Int, Int]()
            below.build_filtered = Dict[Int, Bool]()
        if not self.is_input and len(self.declined) == 0:
            return started^
        ref on = Pointer[QueryContext, ImmutAnyOrigin](
            unsafe_from_address=self._on()
        )[]
        var cursor = below.cursor
        if (
            cursor < 0
            or cursor == self.index
            or _is_scan(on._nodes[cursor].kind)
        ):
            # Nothing streams below the node: it reads its source itself.
            return started^
        var rows = _probe_start(on, cursor, self.batch_size, 0, below^)
        if rows.runs() or rows.result:
            trace_path("stream.declined_rows_below")
            self.below = cursor
            return rows^
        return started^

    def _execute_tasks(self, plan: QueryContext) -> Int:
        """A task per morsel worker, or one for a probe with no morsel
        pipeline."""
        if len(self.started[0].piped) == 0:
            return 1
        return self.started[0].piped[0].run.tasks()

    def execute(mut self, task: Int) raises:
        if len(self.started[0].piped) == 0:
            return
        self.started[0].piped[0].run.execute(task)

    def finalize(mut self) raises:
        """The sink's finish: its parallel work as rounds of tasks
        (`finalize_tasks`), each round's end in `finish_finalize`, and
        the result after the last."""
        ref started = self.started[0]
        if len(started.piped) > 0:
            ref piped = started.piped[0]
            self.finish.append(finish_pipeline(piped.run))
            self.round = self.finish[0].next_round()
        else:
            swap(self.result, started.result)
            self.round = 0
            self._complete()
            return
        if self.round == 0:
            self._complete()

    def finalize_task(mut self, task: Int) raises:
        if len(self.building) > 0:
            self.building[0].round_task(task)
        elif self.filling:
            self.built.value().finalize_task(task)
        elif len(self.finish) > 0:
            self.finish[0].round_task(task)

    def finish_finalize(mut self) raises:
        """A round of finalize tasks has ended: the next round, or the
        result. The build table's fill is the last round."""
        if len(self.building) > 0:
            self._build_round()
            return
        if self.filling:
            self.built.value().finish_finalize()
            self.filling = False
            self.round = 0
            return
        if len(self.finish) > 0:
            self.round = self.finish[0].next_round()
        else:
            self.round = 0
        if self.round == 0:
            self._complete()

    def _complete(mut self) raises:
        """The result once no finalize task is left, and the work after
        it: a declined input's own run, a build side's hash table."""
        ref plan = Pointer[QueryContext, MutAnyOrigin](
            unsafe_from_address=self.plan
        )[]
        ref started = self.started[0]
        if len(started.piped) > 0:
            self.result = _probe_finish(plan, started.piped[0], self.finish[0])
            ref run = started.piped[0].run
            if run.built:
                # The pipeline sank its rows into the join's table.
                self.built = run.built.take()
                self.sank = True
                self._fill_table()
                return
        ref on = Pointer[QueryContext, MutAnyOrigin](
            unsafe_from_address=self._on()
        )[]
        if self.below >= 0 and self.result:
            # The node declined, and this pipeline made the rows below
            # it: the node runs over them, as an operator over a frame.
            on._prefetched.value()[][self.below] = self.result.take()
            on._declined = self.index
            self.result = on._execute(self.index, False, True, self.batch_size)
        elif not self.result and (self.is_input or len(self.declined) > 0):
            # The node declined over its source alone: it runs on its
            # own, as `plan_execute` runs a node no stream takes.
            on._declined = self.index
            self.result = on._execute(self.index, False, True, self.batch_size)
        if len(self.build.keys) > 0 and self.result:
            # A result made as a frame (the stream ran no pipeline, or its
            # columns' types keep the build on one): its table from it,
            # as rounds of this finish.
            self.building.append(
                BuildFinish(
                    self.result.take(),
                    self.build.keys,
                    self.build.how,
                    self.build.coalesce,
                    self.build.probe_estimate,
                    self.build.build_filtered,
                )
            )
            self._build_round()

    def _build_round(mut self) raises:
        """The next round of the build side's table, or, once it is
        made, its fill."""
        self.round = self.building[0].next_round()
        if self.round > 0:
            return
        var done = self.building.pop()
        self.result = done.frame.copy()
        swap(self.built, done.table)
        if self.built:
            self._fill_table()

    def _fill_table(mut self) raises:
        """The built table's pointer table filled as one more round of
        finalize tasks, each on its own partitions' regions."""
        self.round = self.built.value().finalize_tasks()
        if self.round > 0:
            self.filling = True
        else:
            self.built.value().finish_finalize()


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
        if len(probe.started[0].piped) == 0:
            return 1
        ref run = self._run()[]
        self.active = run.begin_round()
        return run.tasks() if self.active else 1

    def execute(mut self, task: Int) raises:
        if self.active:
            self._run()[].execute(task)


struct _SharedSink(Movable):
    """The sink of a shared subplan's pipelines: once the pipeline making
    its result completes, the result goes where every scan of it reads it
    (`SharedResults`). Each pipeline scanning it depends on this one."""

    var source: Int
    var id: Int
    var results: ArcPointer[SharedResults]
    # The schedule's List[_StreamPipeline].
    var siblings: Int

    def __init__(
        out self, source: Int, id: Int, results: ArcPointer[SharedResults]
    ):
        self.source = source
        self.id = id
        self.results = results
        self.siblings = 0

    def execute(mut self) raises:
        ref pipelines = Pointer[List[_StreamPipeline], MutAnyOrigin](
            unsafe_from_address=self.siblings
        )[]
        ref results = self.results[]
        results.frames[self.id] = pipelines[self.source].result()
        results.ready[self.id] = True


struct _OperatorPipeline(Movable):
    """An operator that reads whole inputs (a sort, a window, a pivot...)
    as a pipeline of the schedule: its computed inputs are pipelines it
    depends on, their results handed to its plan's memo as its one task
    starts, and its finish makes the hash table when its result is a
    join's build side."""

    var plan: QueryContext
    var index: Int
    var streaming: Bool
    var batch_size: Int
    # The nodes of its inputs and the pipelines computing them.
    var wanted: List[Int]
    var at: List[Int]
    # The join its result is the build side of (no keys for none).
    var build: _BuildFor
    # The schedule's List[_StreamPipeline].
    var siblings: Int
    var result: DataFrame
    var built: Optional[JoinHashTable]
    # The table made from the result, as rounds; then its fill.
    var building: List[BuildFinish]
    var filled: Bool
    var filling: Bool

    def __init__(
        out self,
        var plan: QueryContext,
        index: Int,
        streaming: Bool,
        batch_size: Int,
        var wanted: List[Int],
        var at: List[Int],
        var build: _BuildFor,
    ) raises:
        self.plan = plan^
        # Its own memo: the inputs' results go there, not into the memo
        # of the plan it was copied from, which other pipelines read.
        self.plan._prefetched = Optional(ArcPointer(Dict[Int, DataFrame]()))
        self.plan._prefetched_indexes = Optional(
            ArcPointer(Dict[Int, _ReadyIndex]())
        )
        self.index = index
        self.streaming = streaming
        self.batch_size = batch_size
        self.wanted = wanted^
        self.at = at^
        self.build = build^
        self.siblings = 0
        self.result = DataFrame(List[Series]())
        self.built = None
        self.building = List[BuildFinish]()
        self.filled = False
        self.filling = False

    def execute(mut self) raises:
        if len(self.wanted) > 0:
            ref pipelines = Pointer[List[_StreamPipeline], MutAnyOrigin](
                unsafe_from_address=self.siblings
            )[]
            var ahead = self.plan._prefetched.value()
            for k in range(len(self.wanted)):
                ahead[][self.wanted[k]] = pipelines[self.at[k]].take_result()
        self.result = self.plan._execute(
            self.index, False, self.streaming, self.batch_size
        )

    def finalize(mut self) raises:
        """A build side's table from the result, then its fill, as
        rounds of finalize tasks (`finalize_tasks`)."""
        if len(self.build.keys) > 0:
            self.building.append(
                BuildFinish(
                    self.result.copy(),
                    self.build.keys,
                    self.build.how,
                    self.build.coalesce,
                    self.build.probe_estimate,
                    self.build.build_filtered,
                )
            )

    def max_finalize_tasks(self) -> Int:
        return configured_workers() if len(self.build.keys) > 0 else 0

    def finalize_tasks(mut self) raises -> Int:
        if len(self.building) > 0:
            var tasks = self.building[0].next_round()
            if tasks > 0:
                return tasks
            var done = self.building.pop()
            swap(self.result, done.frame)
            swap(self.built, done.table)
        if not self.built or self.filled:
            return 0
        self.filled = True
        var tasks = self.built.value().finalize_tasks()
        self.filling = tasks > 0
        return tasks

    def finalize_task(mut self, task: Int) raises:
        if len(self.building) > 0:
            self.building[0].round_task(task)
        else:
            self.built.value().finalize_task(task)

    def finish_finalize(mut self) raises:
        if self.filling:
            self.built.value().finish_finalize()
            self.filling = False


struct _StreamPipeline(SchedulablePipeline):
    """A stream's schedule holds its inputs, its probe and the outer scans
    of its right and full joins, and the sinks of the shared subplans it
    makes."""

    var input: List[_OperatorPipeline]
    var probe: List[_StreamProbe]
    var outer: List[_OuterScan]
    var shared: List[_SharedSink]

    def __init__(out self, var input: _OperatorPipeline):
        self.input = List[_OperatorPipeline]()
        self.input.append(input^)
        self.probe = List[_StreamProbe]()
        self.outer = List[_OuterScan]()
        self.shared = List[_SharedSink]()

    def __init__(out self, var probe: _StreamProbe):
        self.input = List[_OperatorPipeline]()
        self.probe = List[_StreamProbe]()
        self.probe.append(probe^)
        self.outer = List[_OuterScan]()
        self.shared = List[_SharedSink]()

    def __init__(out self, var outer: _OuterScan):
        self.input = List[_OperatorPipeline]()
        self.probe = List[_StreamProbe]()
        self.outer = List[_OuterScan]()
        self.outer.append(outer^)
        self.shared = List[_SharedSink]()

    def __init__(out self, var shared: _SharedSink):
        self.input = List[_OperatorPipeline]()
        self.probe = List[_StreamProbe]()
        self.outer = List[_OuterScan]()
        self.shared = List[_SharedSink]()
        self.shared.append(shared^)

    def take_result(mut self) raises -> DataFrame:
        """The input's result, once this input's pipeline completed,
        moved out for its one reader."""
        var taken = DataFrame(List[Series]())
        if len(self.input) > 0:
            swap(taken, self.input[0].result)
        elif self.probe[0].result:
            taken = self.probe[0].result.take()
        return taken^

    def result(self) raises -> DataFrame:
        """The input's result, once this input's pipeline completed."""
        if len(self.input) > 0:
            return self.input[0].result.copy()
        return self.probe[0].result.value().copy()

    def built(self) -> Optional[JoinHashTable]:
        """The hash table the input's finish made, if any."""
        if len(self.input) > 0:
            return self.input[0].built.copy()
        return self.probe[0].built.copy()

    def initialize(mut self) raises:
        if len(self.probe) > 0:
            self.probe[0].initialize()

    def max_execute_tasks(self) -> Int:
        return (
            configured_workers() if len(self.probe) > 0
            or len(self.outer) > 0 else 1
        )

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
        elif len(self.shared) > 0:
            self.shared[0].execute()
        else:
            self.input[0].execute()

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
        if len(self.probe) > 0:
            # A round of the probe's finish or of its build table's fill:
            # a task per worker at most.
            return configured_workers()
        return 0

    def finalize_tasks(mut self) raises -> Int:
        if len(self.input) > 0:
            return self.input[0].finalize_tasks()
        if len(self.probe) > 0:
            return self.probe[0].round
        return 0

    def finalize_task(mut self, task: Int) raises:
        if len(self.input) > 0:
            self.input[0].finalize_task(task)
        elif len(self.probe) > 0:
            self.probe[0].finalize_task(task)

    def finish_finalize(mut self) raises:
        if len(self.input) > 0:
            self.input[0].finish_finalize()
        elif len(self.probe) > 0:
            self.probe[0].finish_finalize()


def _probe_start(
    self: QueryContext,
    index: Int,
    batch_size: Int,
    sink: Int,
    var head: _StreamHead,
    build: _BuildFor = _BuildFor(),
    var types: Optional[_ProbeTypes] = None,
    finish_above: Optional[ArcPointer[FinishAbove]] = None,
) raises -> _ProbeStart:
    """The stream's probe pipeline set up, once its inputs are done: its morsel
    workers, not yet run, or its result when no pipeline runs it. `build`: the join this stream's rows are the build side of; a
    pipeline whose rows are the result sinks them into that join's hash
    table. `types`: the setup made from the plan alone (`_probe_types`),
    used where the inputs' results have the columns it was made for.
    `finish_above`: the row steps above the aggregate source that its
    input pipeline's finish ran when it could."""
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
    var operations = List[PlanNode]()
    # The plan node of each operation, for the execution report.
    var operation_nodes = List[Int]()
    var joins = List[DataFrame]()
    # The plan node of each join built so far.
    var join_nodes = List[Int]()
    var indexes = List[Optional[JoinHashTable]]()
    # Set once an inner, semi or right join's build is empty: no row of the
    # source can reach the sink, so the pipeline is set up as usual but its
    # source yields nothing and the sink finalizes over no rows. A sort's
    # first no rows keep no row either.
    var no_output = top == 0
    # Estimates the schedule did not already give, from one memo: the
    # joins' probe sides nest, so each estimate reuses the one below it.
    var stats = PlanStatistics()
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
        return _probe_restart(plan^, index, batch_size, sink, head^, build)
    if first >= 0 and first != index and not _is_scan(self._nodes[first].kind):
        # The computed source, made by an input pipeline of this stream's
        # schedule (`plan_schedule_inputs`). An aggregate's finish that ran
        # the row-local steps above it, rejecting groups before any frame
        # of them was assembled, made this stream's result.
        var computed = self._execute(first, False, True, batch_size)
        if (
            finish_above
            and finish_above.value()[].node == first
            and finish_above.value()[].applied
        ):
            return _ProbeStart(computed^)
        var plan = self.copy()
        plan._frames.append(computed^)
        plan._schemas.append(Optional[CsvSchema]())
        plan._nodes[first] = _frame_scan_node(slot=len(plan._frames) - 1)
        return _probe_restart(plan^, index, batch_size, sink, head^, build)
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
            # The index a prefetch made with this build, if any.
            var ready = Optional[_ReadyIndex]()
            if self._prefetched_indexes:
                var held = self._prefetched_indexes.value()
                if node.right in held[]:
                    ready = held[].pop(node.right)
            var built: DataFrame
            var build_rows: Int
            if ready and ready.value().rows_only:
                # The build's rows are only in its table: its result is
                # the build's columns, empty, which type the join's output.
                built = self._prefetched.value()[].pop(node.right)
                build_rows = ready.value().index.count
            elif no_output:
                # Rows below an empty build never reach the sink: a build
                # not computed yet is only needed for its columns.
                built = self._execute(node.right, True, True, batch_size)
                build_rows = 0
            else:
                built = self._execute(node.right, False, True, batch_size)
                build_rows = built.height()
            if build_rows == 0 and (
                node.how == JOIN_INNER
                or node.how == JOIN_CROSS
                or node.how == JOIN_SEMI
                or node.how == JOIN_RIGHT
                or node.how == JOIN_RIGHT_SEMI
                or node.how == JOIN_RIGHT_ANTI
            ):
                # Each output row of these joins needs a matching build
                # row, so no probe row passes this join: the pipeline keeps
                # its operators and sink, and its source is emptied below.
                # Joins above that emit unmatched build rows still do,
                # from their own scans.
                if not no_output:
                    trace_path("stream.empty_build")
                no_output = True
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
            elif ready and ready.value().rows_only:
                # The table was made for another join of these rows: the
                # rows as a frame, for this join to build its own from.
                built = DataFrame(
                    ready.value().index.rows_frame(),
                    height=ready.value().index.rows_stored,
                )
                build_rows = built.height()
            if no_output and not prepared and build_rows > 0:
                # A join below an empty build never sees a row: an empty
                # table of the same columns serves it.
                built = built.clear()
                build_rows = 0
            var known = head.probe_estimates.get(cursor)
            var probe_estimate = known.value() if known else plan_cardinality(
                self, stats, node.left
            )
            var filtered = head.build_filtered.get(cursor)
            var build_filtered = filtered.value() if filtered else (
                plan_is_filtering(self, node.right)
            )
            var finished = finish_build(
                built,
                keys,
                node.how,
                node.coalesce,
                probe_estimate,
                build_filtered,
            )
            if finished:
                prepared = finished^
                algorithm = "hash_index"
            if prepared:
                var hash_index = prepared.take()
                hash_index.build_filtered = build_filtered
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
                build_rows=build_rows,
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
    if cursor == index and sink == 0:
        # A scan with nothing to apply is read as it is: a file whole by
        # its parallel reader, a frame as the frame. Only a file sink
        # streams it.
        return _ProbeStart(None)
    operations.reverse()
    operation_nodes.reverse()
    # The setup made from the plan alone holds when every build has the
    # columns it was made for (an empty build's are its own, cleared).
    var ahead = Bool(types) and _same_frames(types.value().builds, joins)
    ref source = self._nodes[cursor]
    # With no output possible, the source is its columns without rows,
    # read as an in-memory frame: nothing is scanned.
    var empty_source = Optional[DataFrame]()
    if no_output and ahead:
        empty_source = types.value().source.copy()
    elif no_output:
        empty_source = self._execute(cursor, True, False, batch_size)
    var frame_source = source.kind == SCAN_FRAME or Bool(empty_source)
    # A file source's scan state (`ScanSource`), opened once, and its
    # columns, which the pipeline's steps are typed from.
    var file_scan = Optional[ArcPointer[ScanSource]]()
    var source_columns = Optional[DataFrame]()
    var distinct = _counts_distinct(expressions)
    var eligibility = pipeline_eligibility(
        operations,
        indexes,
        distinct,
        len(keys),
        ordered,
        expressions,
    )
    if eligibility.allowed and not frame_source:
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
        if ahead and not _same_columns(
            types.value().source, source_columns.value()
        ):
            ahead = False
        # Each batch may carry its own dictionary (Arrow dictionary
        # deltas, a CSV byte range's values): the source moves every
        # batch's codes onto one dictionary per column over the scan, which
        # the steps and sinks are typed by, not a setup made ahead.
        if file_scan.value()[].share_dictionaries(source_columns.value()):
            ahead = False
    var inline_table_eligible = False
    if not frame_source and not source_columns:
        # A file read batch by batch: no setup is made for its steps.
        ahead = False
    if consider_inline_table(
        eligibility, len(keys), len(expressions), ordered, distinct
    ):
        if ahead:
            ref typed = types.value().typed
            inline_table_eligible = table_supported(
                typed[len(typed) - 1], keys, expressions
            )
        else:
            var empty: DataFrame
            if source_columns:
                empty = source_columns.value().copy()
            elif empty_source:
                empty = empty_source.value().copy()
            else:
                empty = plan_slot_table(self, source.source_slot()).clear()
                if len(source.names):
                    empty = empty.select(source.names)
            var typed = _step_schemas(empty, _steps_of(operations), joins)
            inline_table_eligible = table_supported(
                typed[len(typed) - 1], keys, expressions
            )
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
    if not frame_source:
        if not file_scan:
            var opened = _open_scan_source(self, cursor, operations, batch_size)
            if not opened:
                return _ProbeStart(None)
            file_scan = ArcPointer(opened.take())
    elif empty_source:
        input = empty_source.value().copy()
        if ahead and types.value().filter:
            # The steps as the setup compiled them: over no rows, the
            # conjuncts the table filters proved true change nothing.
            operations[0].exprs = types.value().filter.value().copy()
    elif ahead:
        # Read and narrowed by the table filters as the setup was made.
        source._record_execution()
        input = types.value().input.value().copy()
        if types.value().filter:
            trace_path("filter.table_filters")
            operations[0].exprs = types.value().filter.value().copy()
    else:
        source._record_execution()
        input = plan_slot_table(self, source.source_slot())
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
    var counting = Bool(self._report)
    # A TopN pushes a dynamic filter into the scan when its first sort column is read from the
    # source as it is: no join, and no operation that writes a column of that
    # name. A file's batches have no statistics before they are read.
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
    # A table scan under ORDER BY ... LIMIT on a column it reads unchanged is
    # read in its row groups' statistics order, and the row groups that cannot
    # reach the limit are dropped; a filter between them (or the scan's table
    # filters) leaves the order alone but prunes nothing. Dropped leading row
    # groups shorten the offset.
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
    # The steps run as a thread-owned pipeline (#538): each worker takes
    # morsels from a shared cursor and runs every step on them into its
    # own sink state; nothing is held, judged or merged on the main
    # thread. Into a file (`sink`), the result's rows go to it from the
    # workers as each morsel is done, when every row the pipeline keeps is
    # the result: no slice, top-N or reduction is left to apply.
    var began = Int(perf_counter_ns())
    var piped_counts = List[Int]()
    # A top-k's TOP_K node was appended to the operations for the
    # stream's per-batch selection; the pipeline's sink keeps the
    # k best itself, so the steps stop before it.
    var step_count = len(operations) - 1 if top >= 0 else len(operations)
    var step_nodes = List[PlanNode](capacity=step_count)
    for k in range(step_count):
        step_nodes.append(operations[k].copy())
    # A top-N of at most LATE_MATERIALIZATION_MAX_ROWS rows over filters and
    # projections on a table scan runs on the columns its order and filters
    # read plus the row id; the other columns are fetched for its rows
    # afterwards.
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
    var pipeline_ahead = Optional[PipelineTypes]()
    if ahead and not late_operations:
        swap(pipeline_ahead, types.value().types)
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
        # The scan's estimated rows: its probe side's cardinality for
        # join filters.
        source_rows=plan_cardinality(self, stats, cursor) if file_scan else 0,
        # The rows are the join's build side only when the pipeline's
        # rows are the stream's whole result.
        build_keys=build.keys.copy() if (
            limit < 0
            and skip == 0
            and not grouping
            and not finish
            and not late_operations
        ) else List[String](),
        build_how=build.how,
        build_coalesce=build.coalesce,
        build_probe_estimate=build.probe_estimate,
        build_filtered=build.build_filtered,
        # Made ahead for the same steps over the same columns, unless
        # late materialization changed them.
        types=pipeline_ahead^,
        # A slice's rows are the first `skip + limit` in source order,
        # less the first `skip`.
        limit=_slice_end(skip, limit) if top < 0 else -1,
        writer=sink if (
            top < 0 and skip == 0 and limit < 0 and not grouping
        ) else 0,
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


def _slice_end(skip: Int, limit: Int) -> Int:
    """The rows a slice of `limit` rows (-1: to the end) after the first
    `skip` reads from the start: -1 when it reads to the end."""
    if limit < 0 or skip > Int.MAX - limit:
        return -1
    return skip + limit


def _probe_finish(
    self: QueryContext, mut piped: _PipedProbe, mut done: PipelineFinish
) raises -> Optional[DataFrame]:
    """The probe pipeline's result, once its sink has finished
    (`finish_pipeline`) and the tasks of that finish, `done`, have run:
    the output and the plan's bookkeeping after it."""
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
    var piped_result = done.result(piped_counts, set_frames)
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
    if top > 0 or skip > 0 or limit >= 0:
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
    """A stream with no input to schedule ahead of its probe: the probe
    set up here, and, when that leaves work to run, scheduled with its
    outer scans so its execute and finish tasks share one queue."""
    var start = _probe_start(self, index, batch_size, sink, head.copy())
    if not start.runs() and len(start.declined) > 0 and not start.result:
        # Declined on a plan whose computed source is a frame: the node
        # runs there.
        ref plan = start.declined[0]
        plan._declined = index
        return plan._execute(index, False, True, batch_size)
    if not start.runs():
        var result = Optional[DataFrame]()
        swap(result, start.result)
        return result^
    var schedule = _StreamSchedule()
    var top = _add_stream(
        schedule,
        Int(Pointer(to=self)),
        index,
        batch_size,
        sink,
        head^,
        _StreamInputs(),
        _BuildFor(),
        META_REGULAR,
        False,
    )
    schedule.pipelines[top].probe[0].started.append(start^)
    return _run_schedule(self, schedule, top)


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


struct _ProbeTypes(Movable):
    """What a probe's setup reads from the plan alone, made while the
    inputs it waits for still run (`_probe_types`): each join build's
    columns, the source's columns and, for an in-memory source, its rows
    left by the first filter's table filters and that filter's remaining
    conjuncts; each operation's input columns and output; and the
    pipeline's steps typed and compiled with the aggregate table's
    layout."""

    var builds: List[DataFrame]
    var source: DataFrame
    var input: Optional[DataFrame]
    var filter: Optional[List[Expr]]
    var typed: List[DataFrame]
    var types: Optional[PipelineTypes]

    def __init__(
        out self,
        var builds: List[DataFrame],
        var source: DataFrame,
        var input: Optional[DataFrame],
        var filter: Optional[List[Expr]],
        var typed: List[DataFrame],
        var types: Optional[PipelineTypes],
    ):
        self.builds = builds^
        self.source = source^
        self.input = input^
        self.filter = filter^
        self.typed = typed^
        self.types = types^


def _probe_types(
    self: QueryContext, index: Int, head: _StreamHead, batch_size: Int
) raises -> Optional[_ProbeTypes]:
    """The setup `_probe_start` makes from the plan alone, for a probe
    that reads a scan through its operations and joins: what the plan's
    types decide (each build's and the source's columns, every step's
    columns, the steps compiled, the aggregate table's layout) and the
    in-memory source's table filters, none of which reads an input's
    result. None for a probe whose source an input computes (a union, an
    aggregate): it is set up on another plan."""
    var first = head.cursor
    while first >= 0:
        ref step = self._nodes[first]
        if _stream_rows(step) or _stream_join(step):
            first = step.left
        else:
            break
    if first < 0 or (first != index and not _is_scan(self._nodes[first].kind)):
        return None
    var cursor = head.cursor
    var operations = List[PlanNode]()
    var builds = List[DataFrame]()
    while cursor >= 0:
        ref node = self._nodes[cursor]
        if _stream_rows(node):
            operations.append(node.copy())
        elif _stream_join(node):
            var operation = node.copy()
            operation.set_join_build_slot(len(builds))
            builds.append(self._execute(node.right, True, True, batch_size))
            operations.append(operation^)
        else:
            break
        cursor = node.left
    if cursor < 0 or not _is_scan(self._nodes[cursor].kind):
        return None
    operations.reverse()
    ref source = self._nodes[cursor]
    if (
        source.kind == SCAN_FRAME
        and plan_shared_id(self, source.source_slot()) >= 0
    ):
        # A shared subplan's result, which another pipeline is still
        # making: the setup reads it once the probe's inputs are done.
        return None
    var empty: DataFrame
    var input = Optional[DataFrame]()
    var filter = Optional[List[Expr]]()
    if source.kind == SCAN_FRAME:
        var frame = self._frames[source.source_slot()].copy()
        if len(source.names):
            frame = frame.select(source.names)
        empty = frame.clear()
        # The first row-local filter as a scan filter: vectors ruled out
        # by statistics are skipped, and conjuncts every remaining row is
        # proven to pass are not evaluated.
        if (
            len(operations) > 0
            and operations[0].kind == FILTER
            and _row_local(operations[0].exprs)
            and frame.height() > 0
        ):
            var scanned = scan_filtered(
                frame, conjuncts(operations[0].exprs[0])
            )
            if scanned:
                frame = scanned.value().frame.copy()
                var remaining = List[Expr]()
                remaining.append(conjunction(scanned.value().remaining))
                operations[0].exprs = remaining.copy()
                filter = remaining^
        input = frame^
    else:
        empty = self._execute(cursor, True, False, batch_size)
    var steps = _steps_of(operations)
    var typed = _step_schemas(empty, steps, builds)
    var sets = List[List[Int]]()
    if head.grouping:
        for members in _set_members(self._nodes[index]):
            var indices = List[Int]()
            for name in members:
                var at = head.keys.index(name) if name in head.keys else -1
                if at not in indices:
                    indices.append(at)
            sets.append(indices^)
    var types = pipeline_types(
        empty,
        steps,
        builds,
        head.expressions,
        head.keys,
        sets,
        len(head.keys) > 0 and len(head.expressions) > 0,
    )
    return _ProbeTypes(
        builds^, empty^, input^, filter^, typed^, Optional(types^)
    )


def _same_columns(a: DataFrame, b: DataFrame) -> Bool:
    """Whether two frames have the same columns: names and types, in
    order."""
    if a.width() != b.width():
        return False
    for c in range(a.width()):
        if (
            a._columns[c].name() != b._columns[c].name()
            or a._columns[c].dtype() != b._columns[c].dtype()
        ):
            return False
    return True


def _same_frames(a: List[DataFrame], b: List[DataFrame]) -> Bool:
    """Whether two lists of frames have the same columns, frame by frame."""
    if len(a) != len(b):
        return False
    for k in range(len(a)):
        if not _same_columns(a[k], b[k]):
            return False
    return True


struct _ProbeStart(Movable):
    """`_probe_start`'s answer: the result, or a probe set up to run as a
    morsel pipeline."""

    var result: Optional[DataFrame]
    var piped: List[_PipedProbe]
    # The plan the node declined on, when the probe was set up again on
    # one whose input is a frame (`_probe_restart`).
    var declined: List[QueryContext]

    def __init__(out self, var result: Optional[DataFrame]):
        self.result = result^
        self.piped = List[_PipedProbe]()
        self.declined = List[QueryContext]()

    def __init__(out self, var piped: _PipedProbe):
        self.result = None
        self.piped = List[_PipedProbe]()
        self.piped.append(piped^)
        self.declined = List[QueryContext]()

    def runs(self) -> Bool:
        """Whether a probe is set up to run, rather than a result made."""
        return len(self.piped) > 0


def _row_steps_above(
    self: QueryContext, index: Int, first: Int, reductions: Int
) -> List[Int]:
    """The nodes from the one reading `first` up to `index`, in that
    order, when `first` is an aggregate and every one of them is a
    row-local filter, projection or column drop with no reduction of its
    own; else none."""
    var nodes = List[Int]()
    if reductions > 0 or self._nodes[first].kind != AGG:
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


def _probe_restart(
    var plan: QueryContext,
    index: Int,
    batch_size: Int,
    sink: Int,
    var head: _StreamHead,
    build: _BuildFor = _BuildFor(),
) raises -> _ProbeStart:
    """The probe set up again on `plan`, where an input it reads is now a
    frame (a union's children, the aggregate below it): the same head, its
    sides and filters already decided, the same join whose build side its
    rows are (`build`), and this pipeline of the same schedule, not a new
    stream with a schedule of its own. A declined stream hands `plan`
    back, where the node then runs (`_StreamProbe._decline`)."""
    # The plan's estimates for it changed with its inputs: they are
    # taken from `plan`.
    head.probe_estimates = Dict[Int, Int]()
    head.build_filtered = Dict[Int, Bool]()
    var started = _probe_start(plan, index, batch_size, sink, head^, build)
    if started.runs() or started.result:
        return started^
    var declined = _ProbeStart(None)
    declined.declined.append(plan^)
    return declined^


def _rows_below(head: _StreamHead) -> _StreamHead:
    """A head that makes the rows of `head`'s steps and joins as they
    are, with no terminal above them: what a stream whose node declined
    makes instead."""
    return _StreamHead(
        head.cursor,
        List[Expr](),
        List[String](),
        False,
        0,
        -1,
        False,
        -1,
        -1,
        False,
        List[String](),
        List[Bool](),
        List[Bool](),
        head.probe_estimates.copy(),
        head.build_filtered.copy(),
    )


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
    """Where join key `key`, read from node `index`, goes on the way down to
    `aggregate`, through the operators that keep a key's values unchanged and so
    let a filter on them pass: filters, a projection that passes it on (renamed
    or not), the left input of joins and explodes of other columns, to one of
    the aggregate's group keys (_KEY_GROUPED, with its name there). A projection
    that computes it drops it from the filters (_KEY_DROPPED); anything else, or
    a column that is not a group key, ends the join's pushdown (_KEY_BAIL)."""
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
    self: QueryContext, top: Int
) raises -> Optional[QueryContext]:
    """Plan join-key filters below an aggregate through its grouping keys
    (after build sides are chosen). A
    filter on group keys removes whole groups, preserving every retained
    group's aggregate values. Nothing runs here: each join's build is a
    pipeline of the stream's schedule, and the aggregate's pipeline
    depends on those builds and filters by their keys
    (`_StreamProbe.tasks`). Return None when no join permits this."""
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
    var planned = List[PlannedJoinFilter]()
    var probe = top
    while probe >= 0 and probe != first:
        var j = probe
        ref node = self._nodes[j]
        probe = node.left
        if node.kind != JOIN or not supports_join_filters(node.how):
            continue
        if len(node.names) == 0 or len(node.names) != len(node.right_keys):
            continue
        var positions = List[Int]()
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
                positions.append(k)
                grouped.append(found[1])
        if bail or len(positions) == 0:
            continue
        planned.append(
            PlannedJoinFilter(
                j,
                positions^,
                grouped^,
                node.right_keys.copy(),
                plan_is_filtering(self, node.right),
            )
        )
    if len(planned) == 0:
        return None
    trace_path("stream.join_filter_through_aggregate")
    # The filters belong to this stream's read of the aggregate: a copy of
    # its node takes them, so another reader of the node reads it whole.
    var plan = self.copy()
    plan._nodes.append(self._nodes[first].copy())
    var aggregate = len(plan._nodes) - 1
    plan._nodes[above].left = aggregate
    plan._pushed_filters = ArcPointer(
        PushedJoinFilters(aggregate, planned^, List[PushedJoinFilter]())
    )
    return plan^


def _published_filters(
    planned: PlannedJoinFilter, made: _StreamPipeline
) raises -> List[PushedJoinFilter]:
    """The filters a finished build publishes for an aggregate below its
    join's probe side: one per key that reaches a group key. A build whose
    pipeline sank its rows into the join's table has an empty result, so
    its keys are read from the table."""
    var filters = List[PushedJoinFilter]()
    if len(made.probe) > 0 and made.probe[0].sank:
        var hashed = made.built().value().copy()
        for i in range(len(planned.positions)):
            var key = planned.positions[i]
            filters.append(
                PushedJoinFilter(
                    planned.grouped[i],
                    hashed.key_column(key, planned.right_keys[key]),
                    len(planned.right_keys),
                    hashed.inserted,
                    planned.build_filtered,
                )
            )
        return filters^
    var built = made.result()
    var count = _keyed_rows(built, planned.right_keys)
    for i in range(len(planned.positions)):
        filters.append(
            PushedJoinFilter(
                planned.grouped[i],
                built.column(planned.right_keys[planned.positions[i]]),
                len(planned.right_keys),
                count,
                planned.build_filtered,
            )
        )
    return filters^
