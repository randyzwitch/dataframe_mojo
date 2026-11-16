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
)
from dataframe.planning.context import (
    QueryContext,
    SharedResults,
    _ReadyIndex,
)
from dataframe.planning.graph import (
    plan_class_refs,
    plan_compact_scan_slots,
    plan_cover,
    plan_ends_in_breaker,
    plan_filtered_scan,
    plan_height_bound,
    plan_is_filtering,
    plan_materialized,
    plan_reorder,
    plan_spine_rows,
    plan_subplan,
    plan_subplan_classes,
)
from dataframe.planning.physical import (
    plan_order_joins,
)
from dataframe.optimizer.partial_aggregates import (
    plan_push_partial_aggregates,
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
from std.memory import (
    ArcPointer,
)


def plan_share_subplans(
    mut self: QueryContext, streaming: Bool, batch_size: Int
) raises -> List[QueryContext]:
    """Plan eligible repeated subplans to run once, replacing their
    readers with scans of the retained result, and return the plans of
    those subplans by id (`plan_run_shared` runs them as pipelines of the
    query's schedule). Process the largest eligible subtree first, then
    disjoint repeated subtrees. Sharing avoids repeated blocking work or
    selective multi-table computation. A simple filter/projection over one
    table stays in each scan, retaining scan pruning without an extra
    intermediate result. Nothing runs: the consumers are planned from the
    estimates made for each subplan's definition, which hold for every
    scan of its result."""
    var shared = List[QueryContext]()
    var computed = plan_subplan_classes(self)
    ref classes = computed[0]
    ref sizes = computed[1]
    # Nodes under a shared subtree are replaced with it.
    var covered = List[Bool](length=len(self._nodes), fill=False)
    var done = False
    while not done:
        # Nodes change between rounds, so each has its own memo.
        var stats = PlanStatistics()
        # This round's shared subtrees, largest first, and the copies each
        # replaces. None holds another's copy.
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
        if not self._shared:
            self._shared = ArcPointer(SharedResults())
        var origins = List[Int](capacity=len(picks))
        for best in picks:
            trace_path("lazy.shared_subplan")
            origins.append(plan_spine_rows(self, best))
        var defined = List[_SharedDefinition](capacity=len(picks))
        for best in picks:
            defined.append(
                _define_shared(self, best, streaming, batch_size, shared)
            )
        for k in range(len(picks)):
            self._frames.append(defined[k].columns.copy())
            self._schemas.append(Optional[CsvSchema]())
            var slot = len(self._frames) - 1
            self._shared_slots[slot] = defined[k].id
            if len(defined[k].counts) > 0:
                self._materialized_distinct[slot] = defined[k].counts.copy()
            var origin = origins[k]
            if origin > 0:
                self._origin_rows[slot] = origin
            for j in copies[k]:
                self._nodes[j] = _frame_scan_node(slot=slot)
    if len(shared) > 0:
        plan_reorder(self)
        plan_compact_scan_slots(self)
    return shared^


struct _SharedDefinition(Movable):
    """A shared subplan as its scans see it: its id, its result's columns
    (no rows) and the distinct counts estimated for them."""

    var id: Int
    var columns: DataFrame
    var counts: Dict[String, Int]

    def __init__(
        out self, id: Int, var columns: DataFrame, var counts: Dict[String, Int]
    ):
        self.id = id
        self.columns = columns^
        self.counts = counts^


def _define_shared(
    self: QueryContext,
    index: Int,
    streaming: Bool,
    batch_size: Int,
    mut shared: List[QueryContext],
) raises -> _SharedDefinition:
    """Plan the subtree at `index` as a shared subplan of its own, the way
    the query is planned (its own repeated subtrees shared, its joins
    ordered, partial aggregates pushed), add its plan to `shared` after
    those of the subplans it reads, and estimate its result. The rows and
    distinct counts estimated for the definition hold for every scan of
    its result, since it runs once."""
    var sub = plan_subplan(self, index)
    sub._optimized = True
    shared.extend(plan_share_subplans(sub, streaming, batch_size))
    plan_order_joins(sub, streaming, batch_size)
    plan_push_partial_aggregates(sub)
    var top = len(sub._nodes) - 1
    var columns = sub._execute(top, True, streaming, batch_size)
    var stats = PlanStatistics()
    var rows = plan_cardinality(sub, stats, top)
    var counts = Dict[String, Int]()
    for name in columns.columns():
        var count = plan_distinct(sub, stats, top, name)
        if count.known():
            counts[name] = count.count
    ref results = self._shared.value()[]
    var id = len(results.frames)
    results.frames.append(columns.copy())
    results.rows.append(rows)
    results.ready.append(False)
    shared.append(sub^)
    return _SharedDefinition(id, columns^, counts^)


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


struct _StreamInputs(Movable):
    """A stream's computed inputs and the joins they are built for, not
    yet scheduled."""

    var metas: List[MetaPipeline]
    var wanted: List[Int]
    # The join each input is the build side of; -1 for an input that is
    # not a join's (a union child, the probe's own source).
    var joins: List[Int]
    var keys: List[List[String]]
    var hows: List[Int]
    var coalesces: List[Bool]
    var probe_estimates: List[Int]
    # Whether each build side filters its rows (the join filters' bloom
    # rule reads it).
    var build_filtered: List[Bool]

    def __init__(out self):
        self.coalesces = List[Bool]()
        self.probe_estimates = List[Int]()
        self.build_filtered = List[Bool]()
        self.metas = List[MetaPipeline]()
        self.wanted = List[Int]()
        self.joins = List[Int]()
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
    sink, so the union's rows never pass through a frame of their own."""
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
                inputs.joins.append(probe)
                if step.how != JOIN_CROSS and len(step.names) > 0:
                    inputs.keys.append(step.right_keys.copy())
                else:
                    inputs.keys.append(List[String]())
                inputs.hows.append(step.how)
                inputs.coalesces.append(step.coalesce)
                inputs.probe_estimates.append(
                    plan_cardinality(self, stats, step.left)
                )
                inputs.build_filtered.append(
                    plan_is_filtering(self, step.right)
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
            inputs.joins.append(-1)
            inputs.keys.append(List[String]())
            inputs.hows.append(JOIN_CROSS)
            inputs.coalesces.append(True)
            inputs.probe_estimates.append(-1)
            inputs.build_filtered.append(False)
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
        inputs.joins.append(-1)
        inputs.keys.append(List[String]())
        inputs.hows.append(JOIN_CROSS)
        inputs.coalesces.append(True)
        inputs.probe_estimates.append(-1)
        inputs.build_filtered.append(False)
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
    if index not in ahead[]:
        return None
    var frame = ahead[].pop(index)
    if self._prefetched_indexes:
        var held = self._prefetched_indexes.value()
        if index in held[] and held[][index].rows_only:
            # A build whose pipeline sank its rows into a join's table,
            # read by an operator that takes frames (the stream over the
            # join did not run it): the rows as a frame.
            trace_path("join.build_rows_frame")
            var ready = held[].pop(index)
            return DataFrame(
                ready.index.rows_frame(), height=ready.index.rows_stored
            )
    return frame^


def plan_union_children(self: QueryContext, index: Int) -> List[Int]:
    """The inputs of the union at node `index`, in order, a union below it
    contributing its own: a chain of unions is one union of all their
    children, each child one pipeline into the same operators."""
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
    union pipelines' sources, numbered in child order, which keeps the
    union's rows in its children's order)."""
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
