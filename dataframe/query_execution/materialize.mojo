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
    JOIN_ANTI,
    JOIN_CROSS,
    JOIN_INNER,
    JOIN_LEFT,
    JOIN_SEMI,
)
from dataframe.parallel import (
    run_jobs,
)
from dataframe.planning.context import (
    QueryContext,
)
from dataframe.planning.graph import (
    plan_aggregated,
    plan_compact_scan_slots,
    plan_cover,
    plan_ends_in_breaker,
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
    compared. The largest such subtree is collected on its own, and
    every node holding it becomes a scan of the result. The plan
    around it already optimized it, so only the passes after
    optimization run on it again (join key sets, sharing, join order). Runs after join key sets pass sideways: copies that
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
        var sub = plan_subplan(self, best)
        sub._optimized = True
        var frame = sub._collect(
            optimize=True, streaming=streaming, batch_size=batch_size
        )
        # An aggregate's result keeps distinct counts of its own
        # columns for the join order, as DuckDB keeps a materialized
        # CTE's statistics (`AddMaterializedCTEStats`) and gives an
        # aggregate's columns distinct counts (`ExtractAggregationStats`).
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
        var origin_slot = -1
        var origin_names = Dict[String, String]()
        if not aggregated and origin > 0 and frame.height() == origin:
            var top = len(sub._nodes) - 1
            for name in frame.columns():
                var source = plan_spine_source(sub, top, name)
                if not source:
                    continue
                var found = source.value()
                if not sub._frames[found[0]]._statistics:
                    continue
                if origin_slot >= 0 and found[0] != origin_slot:
                    continue
                origin_slot = found[0]
                origin_names[name] = found[1]
        if aggregated:
            self._frames.append(frame^._with_statistics())
        else:
            self._frames.append(frame^)
        self._schemas.append(Optional[CsvSchema]())
        var slot = len(self._frames) - 1
        if origin_slot >= 0:
            self._origin_tables[slot] = sub._frames[origin_slot].copy()
            self._origin_names[slot] = origin_names^
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
    that would have computed it."""
    var ahead = self._prefetched.value()
    var wanted = List[Int]()
    var probe = top
    while probe >= 0:
        ref step = self._nodes[probe]
        if _stream_rows(step):
            probe = step.left
        elif step.kind == JOIN and step.how in [
            JOIN_INNER,
            JOIN_LEFT,
            JOIN_SEMI,
            JOIN_ANTI,
            JOIN_CROSS,
        ]:
            if (
                plan_ends_in_breaker(self, step.right)
                and step.right not in ahead[]
            ):
                wanted.append(step.right)
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
    if len(wanted) < 2:
        return
    trace_path("stream.prefetch_breakers")
    var jobs = List[_SubplanJob](capacity=len(wanted))
    for i in wanted:
        jobs.append(_SubplanJob(self.copy(), i, True, batch_size))
    run_jobs(jobs)
    for k in range(len(wanted)):
        ahead[][wanted[k]] = jobs[k].result.copy()
