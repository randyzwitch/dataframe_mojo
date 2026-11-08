"""Join order and build sides, chosen from plan statistics before anything
runs."""
from dataframe.planning.node_builders import (
    _frame_scan_node,
    _join_node,
    _select_node,
)

from dataframe.expr import (
    Expr,
    col,
    conjuncts,
)


from dataframe.io.csv import (
    CsvSchema,
)
from dataframe.joins.eager import (
    build_row_width,
)
from dataframe.join_order import (
    FilterEdge,
    JoinEdge,
    JoinOperator,
    JoinTree,
    OPERATOR_ANTI,
    OPERATOR_INNER,
    OPERATOR_LEFT,
    OPERATOR_SEMI,
    build_conflicts,
    order_joins,
)
from dataframe.join_type import (
    JOIN_ANTI,
    JOIN_INNER,
    JOIN_LEFT,
    JOIN_RIGHT,
    JOIN_SEMI,
)
from dataframe.optimizer.passes import (
    plan_push_projections,
)
from dataframe.planning.context import (
    QueryContext,
)
from dataframe.planning.graph import (
    GraphJoin,
    plan_collect_graph,
    plan_columns_of,
    plan_compact_scan_slots,
    plan_graph_input,
    plan_graph_step,
    plan_is_filtering,
    plan_live_parents,
    plan_order_free_above,
    plan_provider_scan,
    plan_reorder,
)
from dataframe.planning.statistics import (
    COUNT_MIN_MAX,
    PlanStatistics,
    plan_cardinality,
    plan_distinct,
)
from dataframe.planning.model import (
    JOIN,
    PlanNode,
)
from dataframe.planning.properties import (
    _comparison_edge,
    _is_scan,
    _references,
    _stream_join,
    _stream_rows,
)


from dataframe.trace import (
    trace_path,
)
from std.collections import (
    Dict,
    Optional,
)


def plan_joins_below(self: QueryContext, index: Int) -> Bool:
    """Whether a join sits under `index` along its left inputs, through
    the row-local steps a stream applies."""
    var cursor = index
    while cursor >= 0:
        ref node = self._nodes[cursor]
        if node.kind == JOIN:
            return True
        if _is_scan(node.kind):
            return False
        cursor = node.left
    return False


# DuckDB 2.0's BuildProbeSideOptimizer (build_probe_side_optimizer.hpp):
# a build side whose other side is a join costs 15% more when it is no
# join itself, since the join's rows are already in flight.
comptime _PREFER_RIGHT_DEEP_PENALTY = 0.15
# JoinFilterBuildSideHeuristics (build_probe_side_optimizer.cpp): a
# build side that filters earns its runtime filter's bonus only against
# a probe scan of at least a million rows, 64 times its own rows; a
# semi join's filtering left builds for a right scan three times both
# sides.
comptime _MIN_FILTER_TARGET_CARDINALITY = 1000000
comptime _MAX_BUILD_TO_TARGET_RATIO = 64
comptime _SEMI_JOIN_FILTER_TARGET_RATIO = 3


def plan_side_estimate(
    self: QueryContext, mut stats: PlanStatistics, index: Int
) raises -> Int:
    """Rows node `index` yields, estimated from statistics before it runs,
    as the build side optimizer reads a child's estimated cardinality
    (`plan_cardinality`, DuckDB's `RelationStatisticsHelper`), with
    nothing sampled or run. -1 when an input has no estimate."""
    return plan_cardinality(self, stats, index)


def _filter_target_rows(
    self: QueryContext, side: Int, keys: List[String]
) raises -> Int:
    """The most rows of a table scan under `side` that a runtime filter on
    `keys` would reach (`MaxDynamicFilterTargetCardinality`): the scans
    supplying the keys (`GetPushdownFilterTargets`), whatever filters
    later reduce."""
    var most = 0
    for key in keys:
        var found = plan_provider_scan(self, side, key)
        if found[0] < 0:
            continue
        most = max(
            most, self._frames[self._nodes[found[0]].source_slot()].height()
        )
    return most


def _filter_build_bonus(
    self: QueryContext,
    probe: Int,
    probe_keys: List[String],
    build: Int,
    build_rows: Int,
) raises -> Float64:
    """How much cheaper building on `build` is for the runtime filter it
    publishes into the probe side's scans (`DynamicFilterBuildBonus`):
    the target scan's rows over the build's, when the build filters its
    tables and the scan is large next to it; else 1."""
    if not plan_is_filtering(self, build):
        return 1.0
    var target = _filter_target_rows(self, probe, probe_keys)
    if target < _MIN_FILTER_TARGET_CARDINALITY:
        return 1.0
    if build_rows > 0 and build_rows > target // _MAX_BUILD_TO_TARGET_RATIO:
        return 1.0
    return Float64(target) / Float64(max(build_rows, 1))


def _child_joins(self: QueryContext, index: Int) -> Int:
    """The joins in the subplan at `index` (`ChildHasJoins`): both inputs
    of a join, the first input of anything else."""
    if index < 0:
        return 0
    ref node = self._nodes[index]
    if node.kind == JOIN:
        return (
            1 + _child_joins(self, node.left) + _child_joins(self, node.right)
        )
    return _child_joins(self, node.left)


def _builds_on_left(
    self: QueryContext, mut stats: PlanStatistics, at: Int, batch_size: Int
) raises -> Bool:
    """Whether the join at `at` should build on its left input
    (`TryFlipJoinChildren`): each side's hash table bytes
    (`GetBuildSize` over its schema and estimated rows), the right's
    divided by its runtime filter bonus where the join type publishes
    one, the left's where the flipped type does (an anti join's RIGHT_ANTI
    does not), and the right's 15% dearer when it holds no join and the
    left does; the left builds when the right costs more, or for a semi
    join whose filtering left would filter a right scan three times both
    sides' rows. False when a side has no estimate."""
    ref node = self._nodes[at]
    if (
        node.sides_fixed
        or len(node.names) == 0
        or node.how not in [JOIN_INNER, JOIN_LEFT, JOIN_SEMI, JOIN_ANTI]
    ):
        return False
    var lhs = plan_side_estimate(self, stats, node.left)
    var rhs = plan_side_estimate(self, stats, node.right)
    if lhs < 0 or rhs < 0:
        return False
    var left_width = build_row_width(
        self._execute(node.left, True, False, batch_size)
    )
    var right_width = build_row_width(
        self._execute(node.right, True, False, batch_size)
    )
    var left_cost = max(Float64(left_width) * Float64(lhs), 8.0)
    var right_cost = max(Float64(right_width) * Float64(rhs), 8.0)
    var swap = False
    # JoinFilterPushdownUtil::JoinTypeIsSupported: inner and semi joins
    # filter their probe (a left join keeps every probe row, an anti
    # join's filter would need inverting); flipped, an inner, right and
    # right semi join do.
    if node.how == JOIN_INNER or node.how == JOIN_SEMI:
        right_cost /= _filter_build_bonus(
            self, node.left, node.names, node.right, rhs
        )
    if node.how != JOIN_ANTI:
        left_cost /= _filter_build_bonus(
            self, node.right, node.right_keys, node.left, lhs
        )
    if node.how == JOIN_SEMI and plan_is_filtering(self, node.left):
        var target = _filter_target_rows(self, node.right, node.right_keys)
        if (
            target >= _MIN_FILTER_TARGET_CARDINALITY
            and target // _SEMI_JOIN_FILTER_TARGET_RATIO > lhs
            and target // _SEMI_JOIN_FILTER_TARGET_RATIO > rhs
        ):
            swap = True
    if (
        _child_joins(self, node.right) == 0
        and _child_joins(self, node.left) > 0
    ):
        right_cost *= 1.0 + _PREFER_RIGHT_DEEP_PENALTY
    if right_cost > left_cost:
        swap = True
    return swap


def _flip_inner_join(
    mut plan: QueryContext, at: Int, batch_size: Int
) raises -> Bool:
    """Flip the inner join at `at` (`FlipChildren`): its right input
    probes and its left builds, under a selection that restores the
    join's columns, their order, names and key types. False, with
    nothing changed, when a name the sides share other than as a key
    would be renamed by the join, or a right key repeats."""
    var node = plan._nodes[at].copy()
    if not node.coalesce:
        return False
    for i in range(len(node.right_keys)):
        for j in range(i):
            if node.right_keys[i] == node.right_keys[j]:
                return False
    var left = plan._execute(node.left, True, False, batch_size)
    var right = plan._execute(node.right, True, False, batch_size)
    var left_names = left.columns()
    var right_names = right.columns()
    for name in right_names:
        if name in left_names and not (
            name in node.right_keys and name in node.names
        ):
            return False
    var exprs = List[Expr]()
    for name in left_names:
        var k = -1
        for i in range(len(node.names)):
            if node.names[i] == name:
                k = i
        if k < 0:
            exprs.append(col(name))
            continue
        var key = col(node.right_keys[k])
        var dtype = left.column(name).dtype()
        if right.column(node.right_keys[k]).dtype() != dtype:
            key = key.cast(dtype)
        exprs.append(key.alias(name))
    for name in right_names:
        if name not in node.right_keys:
            exprs.append(col(name))
    var join = node.copy()
    join.left = node.right
    join.right = node.left
    join.names = node.right_keys.copy()
    join.right_keys = node.names.copy()
    join.sides_fixed = True
    plan._nodes.append(join^)
    plan._nodes[at] = _select_node(left=len(plan._nodes) - 1, exprs=exprs)
    trace_path("lazy.swap_join_sides")
    return True


def _flip_left_join(
    mut plan: QueryContext, at: Int, batch_size: Int
) raises -> Bool:
    """Flip the left join at `at` into a right join (`FlipChildren` with
    `InverseJoinType`): its right input probes and its left builds, the
    build rows no probe row matched coming out last (`_emit_unmatched`,
    DuckDB's right-outer scan), under a selection that restores the
    join's columns: the left's, then the right's other than its keys.
    False, with nothing changed, when the sides share a column name, a
    key repeats, or the key types differ or need the general join
    (categorical, nested)."""
    var node = plan._nodes[at].copy()
    if not node.coalesce:
        return False
    for i in range(len(node.right_keys)):
        for j in range(i):
            if node.right_keys[i] == node.right_keys[j]:
                return False
    var left = plan._execute(node.left, True, False, batch_size)
    var right = plan._execute(node.right, True, False, batch_size)
    var left_names = left.columns()
    var right_names = right.columns()
    for name in right_names:
        if name in left_names:
            return False
    for i in range(len(node.names)):
        var ltype = left.column(node.names[i]).dtype()
        var rtype = right.column(node.right_keys[i]).dtype()
        if (
            ltype != rtype
            or ltype.is_categorical()
            or ltype.is_nested()
            or ltype.is_struct()
        ):
            return False
    var exprs = List[Expr]()
    for name in left_names:
        exprs.append(col(name))
    for name in right_names:
        if name not in node.right_keys:
            exprs.append(col(name))
    var join = node.copy()
    join.left = node.right
    join.right = node.left
    join.names = node.right_keys.copy()
    join.right_keys = node.names.copy()
    join.how = JOIN_RIGHT
    join.text = "right"
    join.coalesce = False
    join.sides_fixed = True
    plan._nodes.append(join^)
    plan._nodes[at] = _select_node(left=len(plan._nodes) - 1, exprs=exprs)
    trace_path("lazy.swap_left_join_sides")
    return True


def plan_build_probe_sides(
    self: QueryContext, top: Int, batch_size: Int
) raises -> Optional[QueryContext]:
    """Choose the build side of every join the stream below `top` runs,
    before anything runs or is scheduled: DuckDB 2.0's
    `BuildProbeSideOptimizer` on the joins' estimated inputs
    (`_builds_on_left`). A join that should build on its left is
    flipped: an inner join probes its right input against its left, a
    left join becomes a right join, and a semi or anti join DuckDB's
    RIGHT_SEMI or RIGHT_ANTI (`build_left`). A copy of the plan with the
    flips made, or None when every join keeps its sides. A flipped join
    is decided (`sides_fixed`), so the stream the copy starts keeps it."""
    var spine = List[Int]()
    var cursor = top
    while cursor >= 0:
        ref step = self._nodes[cursor]
        if _stream_rows(step):
            cursor = step.left
        elif _stream_join(step):
            spine.append(cursor)
            cursor = step.left
        else:
            break
    # Children first, as the optimizer visits them; a flip leaves every
    # other join's inputs, their estimates and schemas as they were.
    var flips = List[Int]()
    var stats = PlanStatistics()
    for k in range(len(spine) - 1, -1, -1):
        if _builds_on_left(self, stats, spine[k], batch_size):
            flips.append(spine[k])
    if len(flips) == 0:
        return None
    var plan = self.copy()
    var changed = False
    for at in flips:
        var how = plan._nodes[at].how
        if how == JOIN_INNER:
            if _flip_inner_join(plan, at, batch_size):
                changed = True
        elif how == JOIN_LEFT:
            if _flip_left_join(plan, at, batch_size):
                changed = True
        else:
            plan._nodes[at].build_left = True
            plan._nodes[at].sides_fixed = True
            trace_path("lazy.build_left_semi_join")
            changed = True
    if not changed:
        return None
    return Optional(plan^)


def plan_order_joins(
    mut self: QueryContext, streaming: Bool, batch_size: Int
) raises:
    """Order each join graph by estimated cost (`plan_cost_order_joins`,
    DuckDB 2.0's `JoinOrderOptimizer`), from statistics alone: nothing
    runs while planning. Run when the plan executes, not in `explain`.
    Projection pushdown runs again afterwards, as each reordered graph
    ends with one selection of its original output."""
    var root = len(self._nodes) - 1
    if plan_cost_order_joins(self, root, streaming, batch_size):
        plan_reorder(self, root)
        plan_compact_scan_slots(self)
        plan_push_projections(self)


def plan_cost_order_joins(
    mut self: QueryContext, root: Int, streaming: Bool, batch_size: Int
) raises -> Bool:
    """Order each join graph of two or more relations by estimated cost
    (`join_order.mojo`): inner joins flattened across nested inputs into
    relations and key pairs, each relation's rows estimated after its
    own filters from statistics (`plan_cardinality`), and the cheapest
    tree built. A semi, anti or left join is an
    operator of the graph whose right input is one relation, as in
    DuckDB 2.0 (`JoinOrderOperator`): its conflict rules say where it
    may go, so a semi join can run before inner joins its keys do not
    read, and it keeps its sides. A filter reading a left join's right
    columns stays above that join (`PinFilterAfterLeftJoin`). Choose each hash-build side
    using rows times columns, with a lone relation costing 15% more
    when the other
    side is a join). Filters between joins follow the lowest join
    holding their columns; a selection restores the graph's output,
    a key a join dropped read from the key it matched.

    Only where nothing above reads the rows' order
    (`_order_free_above`) and every column name is unique across the
    relations but a key pair of one name, so no join renames one. True when a graph changed."""
    var parents = plan_live_parents(self, root)
    var changed = False
    var count = len(self._nodes)
    for top in range(count):
        if parents[top] == -2 or not plan_graph_step(self, top):
            continue
        # The graph's top: no graph step above it that it belongs to.
        var up = parents[top]
        if up >= 0 and plan_graph_input(self, up, top):
            continue
        if not plan_order_free_above(self, top, parents):
            continue
        var relations = List[Int]()
        var pairs = List[Tuple[String, String]]()
        var filters = List[Int]()
        var joins = List[GraphJoin]()
        plan_collect_graph(self, top, relations, pairs, filters, joins)
        var n = len(relations)
        if n < 2 or n > 62:
            continue
        # The right relations of semi and anti joins: their columns end
        # at their join, so no other key or filter reads them.
        var hidden = List[Bool](length=n, fill=False)
        # The right relations of left joins: always a join's build side.
        var nullable = List[Bool](length=n, fill=False)
        for g in joins:
            ref how = self._nodes[g.node].how
            if how == JOIN_SEMI or how == JOIN_ANTI:
                hidden[g.right_from] = True
            elif how == JOIN_LEFT:
                nullable[g.right_from] = True
        # Which relation holds each column. A name several relations share
        # is one column only as inner join key pairs of that name joining
        # them all (coalesced into one, as DuckDB's bindings equate them);
        # `twins` holds its other relations, checked below.
        var provider = Dict[String, Int]()
        var twins = Dict[String, List[Int]]()
        var columns = List[List[String]]()
        for r in range(n):
            var names = plan_columns_of(self, relations[r])
            if not hidden[r]:
                for c in names:
                    if c not in provider:
                        provider[c] = r
                    elif c in twins:
                        twins[c].append(r)
                    else:
                        twins[c] = [r]
            columns.append(names^)
        # Each join's left and right relation sets.
        var join_masks = List[Tuple[Int, Int]]()
        for g in joins:
            join_masks.append(
                (
                    ((1 << g.right_from) - 1) & ~((1 << g.left_from) - 1),
                    ((1 << g.end) - 1) & ~((1 << g.right_from) - 1),
                )
            )
        # Every relation of a shared name joins another of it on that name.
        var unique = True
        for c in twins.keys():
            var holders = 1 << provider[c]
            for r in twins[c]:
                holders |= 1 << r
            var linked = 0
            for j in range(len(joins)):
                ref join = self._nodes[joins[j].node]
                if join.how != JOIN_INNER:
                    continue
                for i in range(len(join.names)):
                    if join.names[i] == c and join.right_keys[i] == c:
                        var l = join_masks[j][0] & holders
                        var r = join_masks[j][1] & holders
                        if l != 0 and r != 0:
                            linked |= l | r
            unique = unique and linked == holders
        if not unique:
            continue
        # Each join as an operator (`JoinOrderOperator`): its inputs'
        # relations and the relations its keys read.
        var ops = List[JoinOperator]()
        var known = True
        for j in range(len(joins)):
            ref g = joins[j]
            ref join = self._nodes[g.node]
            var left_mask = join_masks[j][0]
            var right_mask = join_masks[j][1]
            var syntactic = 0
            for i in range(len(join.names)):
                var a = _holder(provider, twins, join.names[i], left_mask)
                var b = g.right_from
                if join.how == JOIN_INNER:
                    b = _holder(provider, twins, join.right_keys[i], right_mask)
                if (
                    a < 0
                    or b < 0
                    or left_mask & (1 << a) == 0
                    or right_mask & (1 << b) == 0
                ):
                    known = False
                    break
                syntactic |= (1 << a) | (1 << b)
            if not known:
                break
            var kind = OPERATOR_INNER
            if join.how == JOIN_SEMI:
                kind = OPERATOR_SEMI
            elif join.how == JOIN_ANTI:
                kind = OPERATOR_ANTI
            elif join.how == JOIN_LEFT:
                kind = OPERATOR_LEFT
            var lefts = List[Int]()
            for c in range(g.joins_from, g.joins_mid):
                lefts.append(c)
            var rights = List[Int]()
            for c in range(g.joins_mid, j):
                rights.append(c)
            ops.append(
                JoinOperator(
                    kind, left_mask, right_mask, syntactic, lefts^, rights^
                )
            )
        if not known:
            continue
        build_conflicts(ops)
        # Edges: the key pairs between each two relations. An inner
        # join's pairs are free, unless a conflict rule ties them to their
        # join; a left join's always belong to it; a semi or anti join's
        # take no part in the estimate.
        var edges = List[JoinEdge]()
        for j in range(len(joins)):
            ref join = self._nodes[joins[j].node]
            if ops[j].kind == OPERATOR_SEMI or ops[j].kind == OPERATOR_ANTI:
                continue
            var owner = j if ops[j].must_apply else -1
            for i in range(len(join.names)):
                var left_key = join.names[i]
                var right_key = join.right_keys[i]
                var a = _holder(provider, twins, left_key, join_masks[j][0])
                var b = _holder(provider, twins, right_key, join_masks[j][1])
                var found = False
                for k in range(len(edges)):
                    if edges[k].owner != owner:
                        continue
                    if edges[k].a == a and edges[k].b == b:
                        edges[k].a_keys.append(left_key)
                        edges[k].b_keys.append(right_key)
                        found = True
                    elif edges[k].a == b and edges[k].b == a:
                        edges[k].a_keys.append(right_key)
                        edges[k].b_keys.append(left_key)
                        found = True
                if not found:
                    edges.append(JoinEdge(a, b, [left_key], [right_key], owner))
        # Each filter's relations; one reading a name no relation holds
        # (made along the way) leaves the graph alone.
        var filter_masks = List[Int]()
        for f in filters:
            var reads = _references(self._nodes[f].exprs[0])
            if not reads:
                known = False
                break
            var mask = 0
            for name in reads.value():
                var r = provider.get(name, -1)
                if r < 0:
                    known = False
                else:
                    mask |= 1 << r
                if name in twins:
                    for t in twins[name]:
                        mask |= 1 << t
            # One reading a left join's right columns runs after that
            # join (`PinFilterAfterLeftJoin`).
            for op in ops:
                if op.kind == OPERATOR_LEFT and mask & op.right_relations != 0:
                    mask |= op.syntactic
            filter_masks.append(mask)
        if not known:
            continue
        var output = plan_columns_of(self, top)
        # Each relation's statistics (`RelationStatisticsHelper`), with
        # nothing run: its estimated rows after its own filters, and the
        # distinct counts of the columns the graph compares. A semi or
        # anti join's right relation is in no estimate. A left join's
        # right relation of no estimate is left out of the estimates
        # (`uncounted`): the join keeps every left row. Any other
        # relation without one leaves the graph as written, as DuckDB
        # reorders only a graph whose relations all have statistics.
        var stats = PlanStatistics()
        var rows = List[Float64](capacity=n)
        var domains = List[Float64](capacity=n)
        var estimates = List[Int](capacity=n)
        var uncounted = 0
        for r in range(n):
            var estimate = 1
            if not hidden[r]:
                estimate = plan_cardinality(self, stats, relations[r])
                if estimate < 0 and nullable[r]:
                    uncounted |= 1 << r
                elif estimate < 0:
                    known = False
                    break
            estimates.append(estimate)
            rows.append(Float64(max(estimate, 1)))
            # No rows known: its keys bound no domain.
            domains.append(
                1.0e18 if uncounted & (1 << r) != 0 else Float64(estimate)
            )
        if not known:
            continue
        # An inner join with an empty input is empty (`EmptyResultPullup`).
        var every = List[Int](capacity=n)
        for r in range(n):
            if uncounted & (1 << r) == 0:
                every.append(r)
        if plan_empty_graph(
            self, top, every, hidden, nullable, estimates, batch_size
        ):
            changed = True
            continue
        # Filters comparing a column of one relation with a column of
        # another: edges of DuckDB's graph (`CreateHyperGraphEdges`),
        # which the estimate counts as join conditions.
        var compares = List[FilterEdge]()
        for f in filters:
            for part in conjuncts(self._nodes[f].exprs[0]):
                var compare = _comparison_edge(part, provider)
                if compare:
                    compares.append(compare.take())
        # Each compared column's distinct count by its source, for the
        # sets' total domains (`UpdateTotalDomains`): reliable counts,
        # counts from integer bounds, and fallback rows.
        var compared = List[Tuple[Int, String]]()
        for e in edges:
            for i in range(len(e.a_keys)):
                compared.append((e.a, e.a_keys[i]))
                compared.append((e.b, e.b_keys[i]))
        for f in compares:
            compared.append((f.a, f.a_column))
            compared.append((f.b, f.b_column))
        var distinct = Dict[String, Float64]()
        var bounded = Dict[String, Float64]()
        var fallback = Dict[String, Float64]()
        for pair in compared:
            var key = String(pair[0]) + "/" + pair[1]
            if key in distinct or key in bounded or key in fallback:
                continue
            if hidden[pair[0]] or uncounted & (1 << pair[0]) != 0:
                continue
            var count = plan_distinct(self, stats, relations[pair[0]], pair[1])
            if not count.known():
                continue
            if count.reliable():
                distinct[key] = Float64(count.count)
            elif count.source == COUNT_MIN_MAX:
                bounded[key] = Float64(count.count)
            else:
                fallback[key] = Float64(count.count)
        var chosen = order_joins(
            rows,
            domains,
            edges,
            distinct,
            compares,
            ops,
            uncounted,
            bounded,
            fallback,
        )
        if not chosen:
            continue
        var tree = chosen.take()
        var swaps = plan_tree_swaps(self, tree, relations, columns, edges, ops)
        # The written tree, each join's left and right relation sets: a
        # chosen tree that is the same joins of the same sides changes
        # nothing, and the graph stays as written.
        var written = List[Tuple[Int, Int]]()
        for g in joins:
            written.append(
                (
                    ((1 << g.right_from) - 1) & ~((1 << g.left_from) - 1),
                    ((1 << g.end) - 1) & ~((1 << g.right_from) - 1),
                )
            )
        var same = True
        for e in range(n, len(tree.mask)):
            var lm = tree.mask[tree.right[e] if swaps[e] else tree.left[e]]
            var rm = tree.mask[tree.left[e] if swaps[e] else tree.right[e]]
            var found = False
            for w in written:
                found = found or (w[0] == lm and w[1] == rm)
            same = same and found
        if same:
            continue
        # Build the chosen tree. `node_of` is each tree entry's plan
        # node; `renamed` maps each key a join dropped to the key it
        # matched.
        var entries = len(tree.mask)
        var node_of = List[Int](length=entries, fill=-1)
        for r in range(n):
            node_of[r] = relations[r]
        var renamed = Dict[String, String]()
        var placed = List[Bool](length=len(filters), fill=False)
        var added = List[PlanNode]()
        var base = len(self._nodes)
        for e in range(n, entries):
            var left = tree.left[e]
            var right = tree.right[e]
            var via = tree.operator[e]
            if via >= 0 and ops[via].kind != OPERATOR_INNER:
                # A semi, anti or left join keeps its sides: its left
                # input holds its left keys' relations.
                if swaps[e]:
                    var t = left
                    left = right
                    right = t
                var join = self._nodes[joins[via].node].copy()
                join.left = node_of[left]
                join.right = node_of[right]
                for i in range(len(join.names)):
                    while join.names[i] in renamed:
                        join.names[i] = renamed[join.names[i]]
                if ops[via].kind == OPERATOR_LEFT:
                    for i in range(len(join.names)):
                        if join.right_keys[i] != join.names[i]:
                            renamed[join.right_keys[i]] = join.names[i]
                added.append(join^)
                var operator_node = base + len(added) - 1
                operator_node = _place_graph_filters(
                    self,
                    filters,
                    filter_masks,
                    placed,
                    renamed,
                    tree.mask[e],
                    added,
                    base,
                    operator_node,
                )
                node_of[e] = operator_node
                continue
            if swaps[e]:
                var t = left
                left = right
                right = t
            var lmask = tree.mask[left]
            var rmask = tree.mask[right]
            var left_keys = List[String]()
            var right_keys = List[String]()
            for edge in edges:
                if edge.owner >= 0 and edge.owner != via:
                    continue
                var forward = (
                    lmask & (1 << edge.a) != 0 and rmask & (1 << edge.b) != 0
                )
                var backward = (
                    lmask & (1 << edge.b) != 0 and rmask & (1 << edge.a) != 0
                )
                if not (forward or backward):
                    continue
                for i in range(len(edge.a_keys)):
                    var lk = edge.a_keys[i] if forward else edge.b_keys[i]
                    var rk = edge.b_keys[i] if forward else edge.a_keys[i]
                    while lk in renamed:
                        lk = renamed[lk]
                    while rk in renamed:
                        rk = renamed[rk]
                    left_keys.append(lk)
                    right_keys.append(rk)
            for i in range(len(right_keys)):
                if right_keys[i] != left_keys[i]:
                    renamed[right_keys[i]] = left_keys[i]
            var join = _join_node(
                left=node_of[left],
                right=node_of[right],
                left_keys=left_keys,
                join_type="inner",
                type_code=JOIN_INNER,
                right_keys=right_keys,
                suffix="_right",
            )
            # Its sides are the tree's (`plan_tree_swaps`) until the build
            # side optimizer decides them at stream start, as DuckDB runs
            # `BuildProbeSideOptimizer` after the join order.
            added.append(join^)
            var current = _place_graph_filters(
                self,
                filters,
                filter_masks,
                placed,
                renamed,
                tree.mask[e],
                added,
                base,
                base + len(added) - 1,
            )
            node_of[e] = current
        var keep = List[Expr]()
        for name in output:
            var source = name
            while source in renamed:
                source = renamed[source]
            keep.append(
                col(source) if source == name else col(source).alias(name)
            )
        var restore = _select_node(left=node_of[entries - 1], exprs=keep)
        # Filters left: read names a join dropped; they run on the
        # restored columns.
        var last = restore^
        for f in range(len(filters)):
            if placed[f]:
                continue
            added.append(last^)
            last = self._nodes[filters[f]].copy()
            last.left = base + len(added) - 1
        for node in added:
            self._nodes.append(node.copy())
        self._nodes[top] = last^
        trace_path("lazy.cost_join_order")
        changed = True
    return changed


def _holder(
    provider: Dict[String, Int],
    twins: Dict[String, List[Int]],
    name: String,
    mask: Int,
) raises -> Int:
    """The first relation in `mask` holding column `name` (`provider`,
    then a shared name's `twins`); its first holder when none is in
    `mask`, -1 when none holds it."""
    var r = provider.get(name, -1)
    if r >= 0 and mask & (1 << r) != 0:
        return r
    if name in twins:
        for t in twins[name]:
            if mask & (1 << t) != 0:
                return t
    return r


def _place_graph_filters(
    self: QueryContext,
    filters: List[Int],
    filter_masks: List[Int],
    mut placed: List[Bool],
    renamed: Dict[String, String],
    mask: Int,
    mut added: List[PlanNode],
    base: Int,
    current: Int,
) raises -> Int:
    """Add after node `current` the graph's filters whose relations a
    join over relation set `mask` completes, while their names are all
    still held; the last node added (or `current`)."""
    var top = current
    for f in range(len(filters)):
        if placed[f] or filter_masks[f] & ~mask != 0:
            continue
        var reads = _references(self._nodes[filters[f]].exprs[0]).value().copy()
        var held = True
        for name in reads:
            held = held and name not in renamed
        if not held:
            continue
        var filter = self._nodes[filters[f]].copy()
        filter.left = top
        added.append(filter^)
        top = base + len(added) - 1
        placed[f] = True
    return top


def plan_tree_swaps(
    self: QueryContext,
    tree: JoinTree,
    relations: List[Int],
    columns: List[List[String]],
    edges: List[JoinEdge],
    ops: List[JoinOperator],
) raises -> List[Bool]:
    """For each entry of `tree` that joins two others, whether its left
    child builds rather than its right. DuckDB's build side: the side of
    fewer rows times columns builds, a lone relation 15% dearer against
    a join, and a lone relation joined on one key whose values are a
    progression costs nothing, since it builds no hash table (the other
    side looks its rows up by position, `join.progression`). A semi,
    anti or left join keeps its sides: its right input builds."""
    var n = len(relations)
    var entries = len(tree.mask)
    var swaps = List[Bool](length=entries, fill=False)
    var width = List[Int](length=entries, fill=0)
    var joined = List[Bool](length=entries, fill=False)
    var node_of = List[Int](length=entries, fill=-1)
    for r in range(n):
        node_of[r] = relations[r]
        width[r] = len(columns[r])
    for e in range(n, entries):
        var left = tree.left[e]
        var right = tree.right[e]
        var via = tree.operator[e]
        if via >= 0 and ops[via].kind != OPERATOR_INNER:
            # Its left input holds its left keys' relations.
            var total = ops[via].left_total
            swaps[e] = tree.mask[left] & total != total
            var kept = right if swaps[e] else left
            var other = left if swaps[e] else right
            if ops[via].kind == OPERATOR_LEFT:
                # It drops its right keys: the pairs it owns.
                var dropped = 0
                for edge in edges:
                    if edge.owner == via:
                        dropped += len(edge.a_keys)
                width[e] = width[kept] + width[other] - dropped
            else:
                width[e] = width[kept]
            joined[e] = True
            continue
        var left_cost = tree.rows[left] * Float64(width[left])
        var right_cost = tree.rows[right] * Float64(width[right])
        if not joined[right] and joined[left]:
            right_cost *= 1.15
        if not joined[left] and joined[right]:
            left_cost *= 1.15
        swaps[e] = right_cost > left_cost
        # The keys the join drops: one per key pair between its sides.
        var keys = 0
        for edge in edges:
            if edge.owner >= 0 and edge.owner != via:
                continue
            var lmask = tree.mask[left]
            var rmask = tree.mask[right]
            if (lmask & (1 << edge.a) != 0 and rmask & (1 << edge.b) != 0) or (
                lmask & (1 << edge.b) != 0 and rmask & (1 << edge.a) != 0
            ):
                keys += len(edge.a_keys)
        width[e] = width[left] + width[right] - keys
        joined[e] = True
    return swaps^


def plan_empty_graph(
    mut self: QueryContext,
    top: Int,
    counted: List[Int],
    hidden: List[Bool],
    nullable: List[Bool],
    estimates: List[Int],
    batch_size: Int,
) raises -> Bool:
    """Replace the graph at `top` by its empty result when one of the
    relations `counted` (run, so their counts are exact) has no rows and
    every output row needs one of its rows: any relation but the right
    input of an anti or left join (`nullable`, or `hidden` and so never
    run). DuckDB's `EmptyResultPullup`: an inner or semi join with an
    empty input is empty, and so are an anti or left join with an empty
    left. Nothing else of the graph then runs. True when replaced."""
    for r in counted:
        if hidden[r] or nullable[r] or estimates[r] != 0:
            continue
        var empty = self._execute(top, True, False, batch_size)
        self._frames.append(empty^)
        self._schemas.append(Optional[CsvSchema]())
        self._nodes[top] = _frame_scan_node(slot=len(self._frames) - 1)
        trace_path("lazy.empty_join_graph")
        return True
    return False
