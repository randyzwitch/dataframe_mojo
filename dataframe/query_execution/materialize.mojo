"""Shared subplan materialization and independent input prefetch."""
from dataframe.planning.node_builders import (
    _frame_scan_node,
)
from dataframe.aggregation.expressions import (
    _stream_reductions,
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
from dataframe.parallel import (
    Job,
    run_jobs,
)
from dataframe.planning.context import (
    QueryContext,
    _ReadyIndex,
)
from dataframe.planning.graph import (
    plan_aggregated,
    plan_compact_scan_slots,
    plan_cover,
    plan_ends_in_breaker,
    plan_height_bound,
    plan_reorder,
    plan_spine_rows,
    plan_spine_source,
    plan_subplan,
    plan_subplan_classes,
)
from dataframe.planning.inputs import (
    _SubplanJob,
)
from dataframe.planning.model import (
    AGG,
    JOIN,
    SCAN_FRAME,
    SELECT,
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
    """Execute a subplan used in two places once. Two nodes whose
    subtrees read the same (`_signature`)
    would compute the same frame twice: PDS-H q21 filters lineitem
    for late shipments both as a join input and under a grouped
    n_unique, TPC-DS q39 joins and groups inventory once per month
    compared. The largest such subtree is collected on its own, and
    every node holding it becomes a scan of the result; the next
    largest that holds no copy of it is collected at the same time
    (`plan_collect_shared`). Runs after join key sets pass sideways: copies that
    receive different key sets are left apart, since one shared
    result could not take either set (TPC-DS q59's weekly sales per
    store are joined with a different year's weeks on each side, and
    shared whole cost 40 to 56 ms).

    A subtree holding an aggregation or a unique is shared: its result
    is small next to its input, and holding it whole costs little. A
    subtree is also shared when every copy is read whole
    (`_read_whole`): as a join's build side, materialized in any case
    (TPC-DS q83 builds one date subquery three times), or as the
    direct input of an aggregation when it holds joins (q27's rollup
    aggregates one five-way join three ways). The shared result is
    materialized once and scanned by each consumer.
    Other joins and filters are not shared, since their output can
    be as large as their input and a consumer's filter above one
    copy would have narrowed it (TPC-DS q39's two branches agree only
    on the 11.7M-row joins below their date filters, and holding
    those doubled the query; copies with different consumer filters
    therefore remain separate)."""
    var computed = plan_subplan_classes(self)
    ref classes = computed[0]
    ref sizes = computed[1]
    ref reducing = computed[2]
    # Which nodes hold a join, and each node's parent.
    var joining = List[Bool]()
    var parents = List[Int](length=len(self._nodes), fill=-1)
    for i in range(len(self._nodes)):
        ref node = self._nodes[i]
        var joins = node.kind == JOIN
        if node.left >= 0:
            joins = joins or joining[node.left]
            parents[node.left] = i
        if node.right >= 0:
            joins = joins or joining[node.right]
            parents[node.right] = i
        joining.append(joins)
    # Nodes under a shared subtree are replaced with it.
    var covered = List[Bool](length=len(self._nodes), fill=False)
    var shared = False
    var done = False
    while not done:
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
                # (`_filter_shared_subplans`): two readers with one filter
                # in common would otherwise share it, each pair with its
                # own copy of the subplan below.
                if covered[i] or _pinned(self._nodes[i]):
                    continue
                if not reducing[i] and not plan_read_whole(
                    self, i, classes, parents, joining, covered
                ):
                    continue
                if best >= 0 and sizes[i] <= sizes[best]:
                    continue
                for j in range(len(self._nodes)):
                    if j != i and not covered[j] and classes[j] == classes[i]:
                        best = i
                        break
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
        var tables = List[Optional[DataFrame]]()
        var names = List[Dict[String, String]]()
        var frames = plan_collect_shared(
            self, picks, origins, streaming, batch_size, tables, names
        )
        for k in range(len(picks)):
            self._frames.append(frames[k].copy())
            self._schemas.append(Optional[CsvSchema]())
            var slot = len(self._frames) - 1
            if tables[k]:
                self._origin_tables[slot] = tables[k].value().copy()
                self._origin_names[slot] = names[k].copy()
            var origin = origins[k]
            if origin > 0:
                self._origin_rows[slot] = origin
            for j in copies[k]:
                self._nodes[j] = _frame_scan_node(slot=slot)
        shared = True
    if shared:
        plan_reorder(self)
        plan_compact_scan_slots(self)


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


struct _SharedJob(Job):
    """One shared subtree, collected from its own copy of the plan."""

    var plan: QueryContext
    var aggregated: Bool
    var origin: Int
    var streaming: Bool
    var batch_size: Int
    var result: DataFrame
    var origin_table: Optional[DataFrame]
    var origin_names: Dict[String, String]

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
        self.origin_table = None
        self.origin_names = Dict[String, String]()

    def run(mut self) raises:
        var frame = self.plan._collect(
            optimize=True, streaming=self.streaming, batch_size=self.batch_size
        )
        # A result holding every row of its table reads its columns'
        # distinct counts from that table: they are the same columns, as
        # DuckDB carries a column's count through a projection into a
        # materialized CTE's statistics. TPC-DS q76's three channels share
        # one projection of date_dim, whose date keys then had no count,
        # and the join order took the 1,990 dates the sales hold for the
        # domain, putting the item join first on a date join estimated 37
        # times too large. A filtered result reads none: DuckDB keeps the
        # table's counts there too, but a filtered dimension's full key
        # count is the date-containment estimate of #1083 (a month of
        # dates joined to sales estimated 37 times too small, TPC-DS q60),
        # whose low estimates the run-time swaps then undo.
        if (
            not self.aggregated
            and self.origin > 0
            and frame.height() == self.origin
        ):
            var top = len(self.plan._nodes) - 1
            var origin_slot = -1
            for name in frame.columns():
                var source = plan_spine_source(self.plan, top, name)
                if not source:
                    continue
                var found = source.value()
                if not self.plan._frames[found[0]]._statistics:
                    continue
                if origin_slot >= 0 and found[0] != origin_slot:
                    continue
                origin_slot = found[0]
                self.origin_names[name] = found[1]
            if origin_slot >= 0:
                self.origin_table = self.plan._frames[origin_slot].copy()
        # An aggregate's result keeps distinct counts of its own
        # columns for the join order, as DuckDB keeps a materialized
        # CTE's statistics (`AddMaterializedCTEStats`) and gives an
        # aggregate's columns distinct counts (`ExtractAggregationStats`).
        if self.aggregated:
            self.result = frame^._with_statistics()
        else:
            self.result = frame^


def plan_collect_shared(
    self: QueryContext,
    picks: List[Int],
    origins: List[Int],
    streaming: Bool,
    batch_size: Int,
    mut origin_tables: List[Optional[DataFrame]],
    mut origin_names: List[Dict[String, String]],
) raises -> List[DataFrame]:
    """Each shared subtree in `picks`, collected on its own. The subtrees
    are independent, so outside an execution report they run as jobs at
    once, sharing the crew's thread budget, as DuckDB schedules the
    pipelines of independent materialized CTEs together: one subtree's
    serial steps (its planning, a small eager input, an aggregate's
    final frame) then overlap another's parallel work. The plan around
    them already optimized each, so only the passes after optimization
    run on it again (join key sets, sharing, join order)."""
    var jobs = List[_SharedJob](capacity=len(picks))
    for k in range(len(picks)):
        var sub = plan_subplan(self, picks[k])
        sub._optimized = True
        jobs.append(
            _SharedJob(
                sub^,
                plan_aggregated(self, picks[k]),
                origins[k],
                streaming,
                batch_size,
            )
        )
    if len(jobs) > 1 and not self._report:
        run_jobs(jobs)
    else:
        for k in range(len(jobs)):
            jobs[k].run()
    var frames = List[DataFrame](capacity=len(jobs))
    for k in range(len(jobs)):
        frames.append(jobs[k].result.copy())
        origin_tables.append(jobs[k].origin_table.copy())
        origin_names.append(jobs[k].origin_names.copy())
    return frames^


def plan_read_whole(
    self: QueryContext,
    index: Int,
    classes: List[Int],
    parents: List[Int],
    joining: List[Bool],
    covered: List[Bool],
) -> Bool:
    """Whether every uncovered copy of the subtree at `index` is read
    whole: as a join's build side, which the executor materializes in
    any case, or, when the subtree holds a join, as the direct input of
    a grouped aggregation or a selection of reductions. One result then
    serves every copy. A chain of row steps over a scan under aggregations reads
    its input once either way and is not worth holding."""
    if self._nodes[index].kind == SCAN_FRAME:
        return False
    var copies = 0
    for j in range(len(self._nodes)):
        if covered[j] or classes[j] != classes[index]:
            continue
        var parent = parents[j]
        if parent < 0:
            return False
        ref consumer = self._nodes[parent]
        if consumer.kind == JOIN and consumer.right == j:
            copies += 1
        elif consumer.left != j or not joining[index]:
            return False
        elif consumer.kind == AGG or (
            consumer.kind == SELECT and _stream_reductions(consumer.exprs)
        ):
            copies += 1
        else:
            return False
    return copies >= 2


def plan_prefetch_breakers(
    self: QueryContext, top: Int, batch_size: Int
) raises:
    """Compute at once, ahead of the stream below `top`, the inputs it
    will execute in full whatever it decides: its joins' build sides
    and its source, where each ends in a pipeline breaker. TPC-DS q78
    aggregated web and catalog sales to build on, then store sales to
    probe with, one after another on a crew that the aggregates'
    serial phases left partly idle. These independent inputs can run
    concurrently. The stream's decisions do not
    change: each result waits in `_prefetched` for the `_execute` call
    that would have computed it.

    A build side's job also finishes it as a build (`finish_build`): its
    columns in one chunk and its keys' hash index, made on that job's
    thread as soon as its rows are in, while the other inputs still run.
    DuckDB finalizes each hash join's table at the end of its own build
    pipeline, which depends on nothing but that pipeline
    (`BuildPipelineSchedule`); TPC-DS q97, its full join streamed,
    hashed its catalog side's distinct pairs only after the store side's
    aggregate had finished, with most of the crew idle. The index waits
    in `_prefetched_indexes` for the stream, which takes it where it
    would have made the same one."""
    var ahead = self._prefetched.value()
    var wanted = List[Int]()
    # A build side's keys, as the stream prepares its table (none for the
    # source or a cross join).
    var keys = List[List[String]]()
    var probe = top
    while probe >= 0:
        ref step = self._nodes[probe]
        if _stream_rows(step):
            probe = step.left
        elif _stream_join(step):
            if (
                plan_ends_in_breaker(self, step.right)
                and step.right not in ahead[]
            ):
                wanted.append(step.right)
                if step.how != JOIN_CROSS and len(step.names) > 0:
                    keys.append(step.right_keys.copy())
                else:
                    keys.append(List[String]())
            probe = step.left
        else:
            break
    if (
        probe >= 0
        and probe != top
        and plan_ends_in_breaker(self, probe)
        and probe not in ahead[]
    ):
        wanted.append(probe)
        keys.append(List[String]())
    if len(wanted) < 2:
        return
    trace_path("stream.prefetch_breakers")
    var jobs = List[_SubplanJob](capacity=len(wanted))
    for k in range(len(wanted)):
        jobs.append(
            _SubplanJob(
                self.copy(),
                wanted[k],
                True,
                batch_size,
                keys[k],
            )
        )
    run_jobs(jobs)
    for k in range(len(wanted)):
        ahead[][wanted[k]] = jobs[k].result.copy()
        if jobs[k].built and self._prefetched_indexes:
            self._prefetched_indexes.value()[][wanted[k]] = _ReadyIndex(
                keys[k], jobs[k].built.value().copy()
            )
