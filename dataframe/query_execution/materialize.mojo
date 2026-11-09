"""Shared subplan materialization and independent input prefetch."""
from dataframe.planning.node_builders import (
    _frame_scan_node,
)
from dataframe.io.csv import (
    CsvSchema,
)
from dataframe.frame import (
    DataFrame,
)
from dataframe.join_type import (
    JOIN_CROSS,
)
from dataframe.scheduler import (
    META_JOIN_BUILD,
    META_REGULAR,
    MetaPipeline,
    SchedulablePipeline,
    run_pipelines,
)
from dataframe.planning.context import (
    QueryContext,
    _ReadyIndex,
)
from dataframe.planning.graph import (
    plan_aggregated,
    plan_class_refs,
    plan_compact_scan_slots,
    plan_cover,
    plan_ends_in_breaker,
    plan_filtered_scan,
    plan_height_bound,
    plan_materialized,
    plan_reorder,
    plan_spine_rows,
    plan_subplan,
    plan_subplan_classes,
)
from dataframe.planning.inputs import (
    _SubplanPipeline,
)
from dataframe.planning.statistics import (
    PlanStatistics,
    plan_cardinality,
    plan_distinct,
)
from dataframe.planning.properties import (
    _pinned,
    _stream_join,
    _stream_rows,
)
from dataframe.series import (
    Series,
)
from dataframe.trace import (
    trace_path,
)
from std.collections import (
    Optional,
)


def plan_share_subplans(
    mut self: QueryContext, streaming: Bool, batch_size: Int
) raises:
    """Execute once a subplan the query computes in several places, under
    the conditions DuckDB 2.0 keeps one materialized (`plan_materialized`,
    `_selective_multi_table_plan`). Our plans hold no CTE nodes: a lazy
    frame used twice is copied into each reader, so a CTE written twice
    and a subplan repeated are the same thing here, and the two passes
    that decide them in DuckDB (`CTEInlining`, `CommonSubplanOptimizer`)
    are one test. A subtree that passes is collected on its own and every
    node holding it becomes a scan of the result; largest first, then the
    next largest holding no copy of it (`plan_collect_shared`). A filter
    or a projection over one table does not pass: DuckDB has that filter
    in the scan, one operator, which is never worth materializing, and
    each reader then scans with the table's own statistics."""
    var computed = plan_subplan_classes(self)
    ref classes = computed[0]
    ref sizes = computed[1]
    # Nodes under a shared subtree are replaced with it.
    var covered = List[Bool](length=len(self._nodes), fill=False)
    var shared = False
    var done = False
    while not done:
        # Nodes change between rounds, so each has its own memo.
        var stats = PlanStatistics()
        # This round's shared subtrees, largest first, and the copies each
        # replaces. None holds another's copy, so they run at once.
        var picks = List[Int]()
        var copies = List[List[Int]]()
        var pending = List[Bool](length=len(self._nodes), fill=False)
        while True:
            # The largest subtree that two uncovered nodes hold.
            var best = -1
            for i in range(len(self._nodes)):
                # A reader's pinned filter stays outside what is shared
                # (`plan_filter_shared_subplans`): two readers with one filter
                # in common would otherwise share it, each pair with its
                # own copy of the subplan below.
                if covered[i] or _pinned(self._nodes[i]):
                    continue
                if best >= 0 and sizes[i] <= sizes[best]:
                    continue
                var refs = plan_class_refs(classes, covered, i)
                if refs >= 2 and (
                    plan_materialized(self, i, refs)
                    or _selective_multi_table_plan(self, stats, i)
                ):
                    best = i
            if best < 0:
                done = True
                break
            var held = List[Int]()
            var reads_pick = False
            for j in range(len(self._nodes)):
                if not covered[j] and classes[j] == classes[best]:
                    held.append(j)
                    reads_pick = reads_pick or _holds(self, j, pending)
            if reads_pick:
                # A subtree over a copy of this round's pick waits for
                # the next round, to read that pick's result.
                break
            for j in held:
                plan_cover(self, j, pending)
                plan_cover(self, j, covered)
            picks.append(best)
            copies.append(held^)
        if len(picks) == 0:
            break
        var origins = List[Int](capacity=len(picks))
        for best in picks:
            trace_path("lazy.shared_subplan")
            origins.append(plan_spine_rows(self, best))
        var counts = List[Dict[String, Int]]()
        var frames = plan_collect_shared(
            self, picks, origins, streaming, batch_size, counts
        )
        for k in range(len(picks)):
            self._frames.append(frames[k].copy())
            self._schemas.append(Optional[CsvSchema]())
            var slot = len(self._frames) - 1
            if len(counts[k]) > 0:
                self._materialized_distinct[slot] = counts[k].copy()
            var origin = origins[k]
            if origin > 0:
                self._origin_rows[slot] = origin
            for j in copies[k]:
                self._nodes[j] = _frame_scan_node(slot=slot)
        shared = True
    if shared:
        plan_reorder(self)
        plan_compact_scan_slots(self)


def _relation_stats(
    self: QueryContext, mut stats: PlanStatistics, index: Int
) raises -> Tuple[Int, Int]:
    """DuckDB's `GetSubplanStats`: how many base tables the subtree at
    `index` reads and the largest one's estimated rows. A table's filters
    belong to its scan there, so a scan under row-local steps counts as
    one table with the rows those steps keep."""
    ref node = self._nodes[index]
    if node.left < 0 or plan_filtered_scan(self, index):
        return (1, max(plan_cardinality(self, stats, index), 0))
    var left = _relation_stats(self, stats, node.left)
    if node.right < 0:
        return left
    var right = _relation_stats(self, stats, node.right)
    return (left[0] + right[0], max(left[1], right[1]))


def _selective_multi_table_plan(
    self: QueryContext, mut stats: PlanStatistics, index: Int
) raises -> Bool:
    """DuckDB's `IsSelectiveMultiTablePlan`, with its constants: a subplan
    over two or more tables is worth materializing when its estimate is
    at most 2,048 rows or under half its largest table."""
    comptime CARDINALITY_THRESHOLD = 2048
    comptime CARDINALITY_RATIO = 2
    var rows = plan_cardinality(self, stats, index)
    if rows < 0:
        return False
    var relations = _relation_stats(self, stats, index)
    if relations[0] < 2:
        return False
    if rows <= CARDINALITY_THRESHOLD:
        return True
    return rows < relations[1] // CARDINALITY_RATIO


def _holds(self: QueryContext, index: Int, marked: List[Bool]) -> Bool:
    """Whether the subtree at `index` holds a `marked` node."""
    var stack = List[Int]()
    stack.append(index)
    while len(stack) > 0:
        var i = stack.pop()
        if marked[i]:
            return True
        for child in [self._nodes[i].left, self._nodes[i].right]:
            if child >= 0:
                stack.append(child)
    return False


struct _SharedPipeline(SchedulablePipeline):
    """One shared subtree, collected from its own copy of the plan as a
    scheduled pipeline, as DuckDB materializes a CTE in a pipeline of its
    own."""

    var plan: QueryContext
    var aggregated: Bool
    var origin: Int
    var streaming: Bool
    var batch_size: Int
    var result: DataFrame
    var distinct_counts: Dict[String, Int]

    def __init__(
        out self,
        var plan: QueryContext,
        aggregated: Bool,
        origin: Int,
        streaming: Bool,
        batch_size: Int,
    ) raises:
        self.plan = plan^
        self.aggregated = aggregated
        self.origin = origin
        self.streaming = streaming
        self.batch_size = batch_size
        self.result = DataFrame(List[Series]())
        self.distinct_counts = Dict[String, Int]()

    def initialize(mut self) raises:
        pass

    def execute_tasks(self) -> Int:
        return 1

    def execute(mut self, task: Int) raises:
        self.result = self.plan._collect(
            optimize=True, streaming=self.streaming, batch_size=self.batch_size
        )

    def prepare_finalize(mut self) raises:
        pass

    def finalize(mut self) raises:
        # A materialized result keeps the distinct counts its plan estimated
        # for it, as DuckDB gives a CTE reference its definition's relation
        # statistics (`AddMaterializedCTEStats`). An aggregate's result
        # has its own columns counted (`ExtractAggregationStats`).
        if self.aggregated:
            self.result = self.result^._with_statistics()
            return
        var stats = PlanStatistics()
        var top = len(self.plan._nodes) - 1
        for name in self.result.columns():
            var count = plan_distinct(self.plan, stats, top, name)
            if count.known():
                self.distinct_counts[name] = count.count


def plan_collect_shared(
    self: QueryContext,
    picks: List[Int],
    origins: List[Int],
    streaming: Bool,
    batch_size: Int,
    mut distinct_counts: List[Dict[String, Int]],
) raises -> List[DataFrame]:
    """Each shared subtree in `picks`, collected on its own. The subtrees
    are independent, so outside an execution report they are scheduled
    as pipelines that run at once (`run_pipelines`), as DuckDB schedules
    the pipelines of independent materialized CTEs together: one subtree's
    serial steps (its planning, a small eager input, an aggregate's
    final frame) then overlap another's parallel work. The plan around
    them already optimized each, so only the passes after optimization
    run on it again (sharing, join order)."""
    var jobs = List[_SharedPipeline](capacity=len(picks))
    var metas = List[MetaPipeline](capacity=len(picks))
    for k in range(len(picks)):
        var sub = plan_subplan(self, picks[k])
        sub._optimized = True
        jobs.append(
            _SharedPipeline(
                sub^,
                plan_aggregated(self, picks[k]),
                origins[k],
                streaming,
                batch_size,
            )
        )
        metas.append(MetaPipeline(META_REGULAR))
    run_pipelines(jobs, metas, parallel=not self._report)
    var frames = List[DataFrame](capacity=len(jobs))
    for k in range(len(jobs)):
        frames.append(jobs[k].result.copy())
        distinct_counts.append(jobs[k].distinct_counts.copy())
    return frames^


struct _StreamInputs(Movable):
    """The pipelines of a stream's computed inputs, not yet run."""

    var jobs: List[_SubplanPipeline]
    var metas: List[MetaPipeline]
    var wanted: List[Int]
    var keys: List[List[String]]
    var hows: List[Int]

    def __init__(out self):
        self.jobs = List[_SubplanPipeline]()
        self.metas = List[MetaPipeline]()
        self.wanted = List[Int]()
        self.keys = List[List[String]]()
        self.hows = List[Int]()


def plan_schedule_inputs(
    self: QueryContext, top: Int, batch_size: Int
) raises -> _StreamInputs:
    """The inputs the stream below `top` will execute in full, as
    pipelines for its schedule: every join's build side, a table already
    in memory included (its build finishes into a hash index like any
    other), and its source where that ends in a pipeline breaker. Build
    sides are chosen before this (`plan_build_probe_sides`), so no build
    made here is one a swap would stream instead. Each is a MetaPipeline
    of the stream's pipeline: a build side is a JOIN_BUILD child, the
    source a REGULAR one, as DuckDB's `MetaPipeline::Build` makes them.
    The stream's probe depends on them all (`plan_stream_execute`).

    A build side's finish event also finishes it as a build
    (`finish_build`): its columns in one chunk and its keys' hash
    index, as DuckDB finalizes each hash join's table in its build
    pipeline's finish event. Sibling builds all execute before any
    finishes (`BuildPipelineSchedule`). The results wait in `_prefetched`
    (`_prefetched_indexes` for the index) for the probe."""
    var ahead = self._prefetched.value()
    var inputs = _StreamInputs()
    var coalesces = List[Bool]()
    var probe_estimates = List[Int]()
    var stats = PlanStatistics()
    var probe = top
    while probe >= 0:
        ref step = self._nodes[probe]
        if _stream_rows(step):
            probe = step.left
        elif _stream_join(step):
            if step.right not in ahead[]:
                inputs.wanted.append(step.right)
                if step.how != JOIN_CROSS and len(step.names) > 0:
                    inputs.keys.append(step.right_keys.copy())
                else:
                    inputs.keys.append(List[String]())
                inputs.hows.append(step.how)
                coalesces.append(step.coalesce)
                probe_estimates.append(plan_cardinality(self, stats, step.left))
                inputs.metas.append(MetaPipeline(META_JOIN_BUILD))
            probe = step.left
        else:
            break
    # With no build to overlap, the stream runs its source itself, where
    # an aggregate's finish can take the steps above it (`FinishAbove`).
    if (
        len(inputs.wanted) > 0
        and probe >= 0
        and probe != top
        and plan_ends_in_breaker(self, probe)
        and probe not in ahead[]
    ):
        inputs.wanted.append(probe)
        inputs.keys.append(List[String]())
        inputs.hows.append(JOIN_CROSS)
        coalesces.append(True)
        probe_estimates.append(-1)
        inputs.metas.append(MetaPipeline(META_REGULAR))
    for k in range(len(inputs.wanted)):
        inputs.jobs.append(
            _SubplanPipeline(
                self.copy(),
                inputs.wanted[k],
                True,
                batch_size,
                inputs.keys[k],
                inputs.hows[k],
                coalesces[k],
                probe_estimates[k],
            )
        )
    return inputs^


def plan_take_prefetched(
    self: QueryContext, index: Int
) raises -> Optional[DataFrame]:
    """The result `plan_schedule_inputs` computed for node `index`, taken
    from the plan's memo, if there is one: a subplan run on its own copy
    of the plan has a memo of its own and would compute it again."""
    if not self._prefetched:
        return None
    var ahead = self._prefetched.value()
    if index in ahead[]:
        return ahead[].pop(index)
    return None
