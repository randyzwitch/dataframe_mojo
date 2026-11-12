"""Shared subplan materialization and independent input prefetch."""
from dataframe.planning.node_builders import (
    _frame_scan_node,
)
from dataframe.io.csv import (
    CsvSchema,
)
from dataframe.frame import (
    DataFrame,
    concat,
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
from dataframe.planning.model import (
    CONCAT,
)
from std.collections import (
    Dict,
    Optional,
)


def plan_share_subplans(
    mut self: QueryContext, streaming: Bool, batch_size: Int
) raises:
    """Execute eligible repeated subplans once and replace their readers with
    scans of the retained result. Process the largest eligible subtree
    first, then disjoint repeated subtrees. Sharing avoids repeated blocking
    work or selective multi-table computation. A simple filter/projection
    over one table stays in each scan, retaining scan pruning without an
    extra intermediate result."""
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
    """Count base tables read by a subtree and report the largest estimated
    table cardinality. Row-local filters belong to the scan, so use the
    number of rows those filters retain."""
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
    """Whether a subplan over at least two tables is selective enough to
    retain: its estimate is at most 2,048 rows or below half the largest
    input table. This limits retained output relative to repeated upstream
    work."""
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
    """Collect a shared subtree from its own plan copy as an independently
    scheduled pipeline. Retain the result until all readers finish."""

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

    def max_execute_tasks(self) -> Int:
        return 1

    def execute_tasks(mut self) raises -> Int:
        return 1

    def execute(mut self, task: Int) raises:
        self.result = self.plan._collect(
            optimize=True, streaming=self.streaming, batch_size=self.batch_size
        )

    def prepare_finalize(mut self) raises:
        pass

    def finalize(mut self) raises:
        # Preserve a materialized result's estimated distinct counts. For
        # aggregate outputs, count their own result columns before downstream
        # planning.
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
    """Collect independent shared subtrees concurrently as scheduled pipelines.
    This lets one subtree's serial planning or final assembly overlap
    another's parallel work. Retain their results for the surrounding plan
    instead of recomputing them at every reference."""
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
    """A stream's computed inputs and the joins they are built for, not
    yet scheduled."""

    var metas: List[MetaPipeline]
    var wanted: List[Int]
    var keys: List[List[String]]
    var hows: List[Int]
    var coalesces: List[Bool]
    var probe_estimates: List[Int]

    def __init__(out self):
        self.coalesces = List[Bool]()
        self.probe_estimates = List[Int]()
        self.metas = List[MetaPipeline]()
        self.wanted = List[Int]()
        self.keys = List[List[String]]()
        self.hows = List[Int]()


def plan_schedule_inputs(
    self: QueryContext, terminal: Int, top: Int, batch_size: Int
) raises -> _StreamInputs:
    """Schedule computed join inputs as children of the probe pipeline. Each
    build finishes its columns and hash index in its finish event. The probe
    depends on all build completions. Sibling builds finish execution before
    finalization; retained frames and indexes are consumed from the
    prefetched slots. A union the probe reads (below `terminal`, the
    stream's last node) has its children scheduled the same way: the
    probe's pipeline is its union pipelines, sharing their operators and
    sink (`PhysicalUnion::BuildPipelines`)."""
    var ahead = self._prefetched.value()
    var inputs = _StreamInputs()
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
                inputs.coalesces.append(step.coalesce)
                inputs.probe_estimates.append(
                    plan_cardinality(self, stats, step.left)
                )
                inputs.metas.append(MetaPipeline(META_JOIN_BUILD))
            probe = step.left
        else:
            break
    if (
        probe >= 0
        and probe != terminal
        and self._nodes[probe].kind == CONCAT
        and probe not in ahead[]
    ):
        for child in plan_union_children(self, probe):
            if child in ahead[] or child in inputs.wanted:
                continue
            inputs.wanted.append(child)
            inputs.keys.append(List[String]())
            inputs.hows.append(JOIN_CROSS)
            inputs.coalesces.append(True)
            inputs.probe_estimates.append(-1)
            inputs.metas.append(MetaPipeline(META_REGULAR))
    # With no build to overlap, the stream runs its source itself, where
    # an aggregate's finish can take the steps above it (`FinishAbove`).
    elif (
        len(inputs.wanted) > 0
        and probe >= 0
        and probe != top
        and plan_ends_in_breaker(self, probe)
        and probe not in ahead[]
    ):
        inputs.wanted.append(probe)
        inputs.keys.append(List[String]())
        inputs.hows.append(JOIN_CROSS)
        inputs.coalesces.append(True)
        inputs.probe_estimates.append(-1)
        inputs.metas.append(MetaPipeline(META_REGULAR))
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


def plan_union_children(self: QueryContext, index: Int) -> List[Int]:
    """The inputs of the union at node `index`, in order, a union below it
    contributing its own (`PhysicalUnion` holds every child of a chain of
    unions, one union pipeline each)."""
    var children = List[Int]()
    var pending = List[Int]()
    pending.append(index)
    while len(pending) > 0:
        var at = pending.pop()
        ref node = self._nodes[at]
        if node.kind == CONCAT:
            # Right first: the stack pops the left next.
            pending.append(node.right)
            pending.append(node.left)
        else:
            children.append(at)
    return children^


def _stack_union(frames: List[DataFrame]) raises -> DataFrame:
    """The union's child results as one frame, each child's columns in the
    first's order: their column chunks appended, nothing copied (the
    union pipelines' sources, numbered in child order as
    `MetaPipeline::AssignNextBatchIndex` numbers them)."""
    var names = frames[0].columns()
    var parts = List[DataFrame](capacity=len(frames))
    for k in range(len(frames)):
        var other = frames[k].columns()
        if len(other) != len(names):
            raise Error(
                "concat: inputs have "
                + String(len(names))
                + " and "
                + String(len(other))
                + " columns"
            )
        for name in names:
            if name not in other:
                raise Error("concat: right input has no column " + name)
        parts.append(frames[k].select(names) if k > 0 else frames[k].copy())
    # `concat` checks the types.
    return concat(parts)
