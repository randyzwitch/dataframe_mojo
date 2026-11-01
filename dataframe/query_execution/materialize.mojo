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
from dataframe.join_type import (
    JOIN_CROSS,
)
from dataframe.parallel import (
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
    compared. The largest such subtree is collected on its own, with
    every planner pass, and every node holding it becomes a scan of
    the result. Runs after join key sets pass sideways: copies that
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
            break
        trace_path("lazy.shared_subplan")
        var origin = plan_spine_rows(self, best)
        var aggregated = plan_aggregated(self, best)
        var sub = self.copy()
        plan_reorder(sub, best)
        var frame = sub._collect(
            optimize=True, streaming=streaming, batch_size=batch_size
        )
        # An aggregate's result keeps distinct counts of its own
        # columns for the join order, as DuckDB keeps a materialized
        # CTE's statistics (`AddMaterializedCTEStats`) and gives an
        # aggregate's columns distinct counts (`ExtractAggregationStats`).
        if aggregated:
            self._frames.append(frame^._with_statistics())
        else:
            self._frames.append(frame^)
        self._schemas.append(Optional[CsvSchema]())
        var slot = len(self._frames) - 1
        if origin > 0:
            self._origin_rows[slot] = origin
        for j in range(len(self._nodes)):
            if not covered[j] and classes[j] == classes[best]:
                plan_cover(self, j, covered)
                self._nodes[j] = _frame_scan_node(slot=slot)
        shared = True
    if shared:
        plan_reorder(self)
        plan_compact_scan_slots(self)


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
    # A build side's keys, and the bound on the rows that will probe it,
    # as the stream prepares its index (none for the source or a cross
    # join).
    var keys = List[List[String]]()
    var probe_rows = List[Int]()
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
                probe_rows.append(plan_height_bound(self, step.left))
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
        probe_rows.append(-1)
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
                probe_rows[k],
            )
        )
    run_jobs(jobs)
    for k in range(len(wanted)):
        ahead[][wanted[k]] = jobs[k].result.copy()
        if jobs[k].built and self._prefetched_indexes:
            self._prefetched_indexes.value()[][wanted[k]] = _ReadyIndex(
                keys[k], probe_rows[k], jobs[k].built.value().copy()
            )
