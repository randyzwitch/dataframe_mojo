"""CPU strategy selection, join ordering, and measured input planning."""
from dataframe.planning.node_builders import (
    _filter_node,
    _frame_scan_node,
    _join_node,
    _select_node,
)
from dataframe.planning.cost.aggregation import (
    plan_sampled_groups,
    plan_batch_states_win,
    plan_row_number_shape,
)

from dataframe.expr import (
    COL,
    Expr,
    GE,
    GT,
    LE,
    LT,
    MAX,
    MIN,
    col,
    conjuncts,
)
from dataframe.frame import (
    DataFrame,
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
    plan_merge_filters,
    plan_push_predicates,
    plan_push_projections,
)
from dataframe.scheduler import (
    META_REGULAR,
    MetaPipeline,
    run_pipelines,
)
from dataframe.planning.context import (
    QueryContext,
)
from dataframe.planning.graph import (
    GraphJoin,
    plan_aggregate_input_rows,
    plan_aggregated,
    plan_available,
    plan_collect_graph,
    plan_columns_of,
    plan_compact_scan_slots,
    plan_filters_rows,
    plan_estimated_rows,
    plan_filtered_table,
    plan_graph_input,
    plan_graph_step,
    plan_has_join,
    plan_has_kind,
    plan_height_bound,
    plan_is_filtering,
    plan_known_height,
    plan_largest_scan,
    plan_live_parents,
    plan_measurable,
    plan_order_free_above,
    plan_provider_scan,
    plan_relation_estimate,
    plan_reorder,
    plan_spine_distinct,
    plan_spine_rows,
)
from dataframe.planning.inputs import (
    _SubplanPipeline,
)
from dataframe.planning.model import (
    AGG,
    CONCAT,
    DROP,
    FILTER,
    JOIN,
    PlanNode,
    SCAN_FRAME,
    SELECT,
    SORT,
    WITH_COLUMNS,
)
from dataframe.planning.properties import (
    _bound_filter,
    _chain_member,
    _comparison_edge,
    _is_scan,
    _references,
    _stream_join,
    _stream_rows,
)
from dataframe.aggregation.expressions import (
    _row_local,
)


from dataframe.trace import (
    trace_path,
)
from std.math import exp
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
# RelationStatisticsHelper::DEFAULT_SELECTIVITY: the rows a filter keeps
# when nothing measures it.
comptime _DEFAULT_SELECTIVITY = 0.2


def plan_side_estimate(self: QueryContext, index: Int) raises -> Int:
    """Rows node `index` yields, estimated from statistics before it runs,
    as the build side optimizer reads a child's estimated cardinality
    (`RelationStatisticsHelper`), with nothing sampled or run: a table's
    rows; through projections, windows, sorts (a top-N at most its
    limit) and row-local filters, where a filter keeps DuckDB's default
    fifth of its rows (`DEFAULT_SELECTIVITY`); a union's inputs' rows
    together; an aggregate's groups from its keys' distinct counts
    (`plan_aggregate_estimate`'s formula); a semi or anti join its left
    input's rows, a left join the larger input's, and an inner join
    DuckDB's estimate of joining two relations (`CardinalityEstimator`):
    their rows multiplied, divided by the key pairs' largest total
    domain, a key's distinct count where its table keeps statistics,
    else the fewest rows of a table holding the key. -1 when an input
    has no estimate."""
    ref node = self._nodes[index]
    if node.kind == SCAN_FRAME:
        return self._frames[node.source_slot()].height()
    if (
        node.kind == DROP
        or node.kind == WITH_COLUMNS
        or (node.kind == SELECT and _stream_rows(node))
    ):
        # Projections and windows keep their input's rows.
        return plan_side_estimate(self, node.left)
    if node.kind == SORT:
        var rows = plan_side_estimate(self, node.left)
        if rows < 0 or node.sort_limit() < 0:
            return rows
        return min(rows, node.sort_limit())
    if node.kind == FILTER and _row_local(node.exprs):
        var rows = plan_side_estimate(self, node.left)
        if rows < 0:
            return -1
        return Int(Float64(rows) * _DEFAULT_SELECTIVITY)
    if node.kind == CONCAT:
        var first = plan_side_estimate(self, node.left)
        var second = plan_side_estimate(self, node.right)
        if first < 0 or second < 0:
            return -1
        return first + second
    if node.kind == AGG:
        var rows = plan_side_estimate(self, node.left)
        if rows < 0:
            return -1
        if len(node.names) == 0:
            return 1
        var product = 1.0
        for key in node.names:
            var count = plan_spine_distinct(self, node.left, key)
            if count < 0:
                return -1
            product *= Float64(max(count, 1))
        product *= 0.95 ** Float64(len(node.names) - 1)
        var multiplier = 1.0 - exp(-Float64(rows) / product)
        var groups = (
            Float64(rows) if multiplier == 0.0 else product * multiplier
        )
        var result = Int(min(groups, Float64(rows)))
        return max(result, 1) if rows > 0 else 0
    if node.kind != JOIN or len(node.names) == 0:
        return -1
    var left = plan_side_estimate(self, node.left)
    if left < 0:
        return -1
    if node.how == JOIN_SEMI or node.how == JOIN_ANTI:
        return left
    var right = plan_side_estimate(self, node.right)
    if right < 0:
        return -1
    if node.how != JOIN_INNER:
        return max(left, right)
    if left == 0 or right == 0:
        return 0
    var domain = 1.0
    for i in range(len(node.names)):
        var count = max(
            plan_spine_distinct(self, node.left, node.names[i]),
            plan_spine_distinct(self, node.right, node.right_keys[i]),
        )
        if count <= 0:
            for side in range(2):
                var found = plan_provider_scan(
                    self,
                    node.left if side == 0 else node.right,
                    node.names[i] if side == 0 else node.right_keys[i],
                )
                if found[0] < 0:
                    continue
                var rows = self._frames[
                    self._nodes[found[0]].source_slot()
                ].height()
                if count <= 0 or rows < count:
                    count = rows
        domain = max(domain, Float64(count))
    return max(Int(Float64(left) * Float64(right) / domain), 1)


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
    self: QueryContext, at: Int, batch_size: Int
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
    var lhs = plan_side_estimate(self, node.left)
    var rhs = plan_side_estimate(self, node.right)
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
    for k in range(len(spine) - 1, -1, -1):
        if _builds_on_left(self, spine[k], batch_size):
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
    """Order each chain of inner joins by how much their inputs narrow.

    A chain is a run of inner joins, filters and plain column selections
    along left inputs, over one base input. Run when the plan executes,
    not in `explain`: a join's right input is executed first when it
    is a known table narrowed by filters or joins, and how many of the
    table's rows it kept is its selectivity, estimated from available
    statistics. Then:

    - A join whose keys come from one earlier join's right input, that
      keeps under half its table and holds at most a quarter as many
      rows as that input's table, joins that input instead (a bushy
      plan): PDS-H q7 joins customer to the two nations its
      filter allows, and orders to those customers, before any
      lineitem row is probed.
    - The remaining joins run most selective first, each once its keys
      are available, with every filter as soon as its columns are.

    Inputs run here replace their subtrees with their results, so
    nothing runs twice. Projection pushdown runs again afterwards, as
    the chain's column selections are dropped and the chain ends with
    one selecting its original output.
    """
    var original_frames = len(self._frames)
    var root = len(self._nodes) - 1
    # Join graphs whose row order cannot show are ordered by estimated
    # cost (`_cost_order_joins`); the passes below order the rest.
    if plan_cost_order_joins(self, root, streaming, batch_size):
        plan_reorder(self, root)
        plan_compact_scan_slots(self)
        plan_push_projections(self)
        return
    # First choose each chain's first input, then order what joins it.
    var changed = plan_probe_largest_inputs(self, root, streaming, batch_size)
    # The chains' top nodes, found before any is reordered: reordering
    # one appends nodes and leaves others unreachable.
    var parents = plan_live_parents(self, root)
    var tops = List[Int]()
    for top in range(len(parents)):
        if parents[top] == -2:
            continue
        if not _chain_member(self._nodes[top]):
            continue
        var parent = parents[top]
        if (
            parent >= 0
            and self._nodes[parent].left == top
            and _chain_member(self._nodes[parent])
        ):
            continue
        tops.append(top)
    for top in tops:
        if plan_order_chain(self, top, streaming, batch_size):
            changed = True
    if changed or len(self._frames) != original_frames:
        plan_reorder(self, root)
        plan_compact_scan_slots(self)
    if changed:
        plan_push_projections(self)


def plan_cost_order_joins(
    mut self: QueryContext, root: Int, streaming: Bool, batch_size: Int
) raises -> Bool:
    """Order each join graph of three or more relations by estimated
    cost (`join_order.mojo`):
    inner joins flattened across nested inputs into relations and key
    pairs, each relation's rows counted after its own filters (run
    when they cannot be counted otherwise, as `_order_chain` measures
    them), and the cheapest tree built. A semi, anti or left join is an
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
    relations, so no join renames one. True when a graph changed."""
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
        if n < 3 or n > 62:
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
        var provider = Dict[String, Int]()
        var unique = True
        var columns = List[List[String]]()
        for r in range(n):
            var names = plan_columns_of(self, relations[r])
            if not hidden[r]:
                for c in names:
                    if c in provider:
                        unique = False
                    provider[c] = r
            columns.append(names^)
        if not unique:
            continue
        # Each join as an operator (`JoinOrderOperator`): its inputs'
        # relations and the relations its keys read.
        var ops = List[JoinOperator]()
        var known = True
        for j in range(len(joins)):
            ref g = joins[j]
            ref join = self._nodes[g.node]
            var left_mask = ((1 << g.right_from) - 1) & ~(
                (1 << g.left_from) - 1
            )
            var right_mask = ((1 << g.end) - 1) & ~((1 << g.right_from) - 1)
            var syntactic = 0
            for i in range(len(join.names)):
                var a = provider.get(join.names[i], -1)
                var b = g.right_from
                if join.how == JOIN_INNER:
                    b = provider.get(join.right_keys[i], -1)
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
                var a = provider[left_key]
                var b = provider[right_key]
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
            # One reading a left join's right columns runs after that
            # join (`PinFilterAfterLeftJoin`).
            for op in ops:
                if op.kind == OPERATOR_LEFT and mask & op.right_relations != 0:
                    mask |= op.syntactic
            filter_masks.append(mask)
        if not known:
            continue
        var output = plan_columns_of(self, top)
        # Rows per relation after its own filters: counted, or
        # estimated from a sample of the table, else run.
        var rows = List[Float64](capacity=n)
        # Each relation's table rows before its filters: its columns'
        # distinct counts, as DuckDB takes them with no statistics.
        var domains = List[Float64](capacity=n)
        var estimates = List[Int](capacity=n)
        var tables = List[Int](capacity=n)
        var runs = List[Int]()
        var aggregated = List[Bool](capacity=n)
        for r in range(n):
            tables.append(plan_spine_rows(self, relations[r]))
            aggregated.append(plan_aggregated(self, relations[r]))
        # A relation that narrows its table becomes a build side, which
        # runs in full whatever the order, unless it is the largest input
        # and streams: run it now and count it, rather than sample its
        # filter and then run it anyway. Its result replaces it, so
        # nothing runs twice, as the measured chain order
        # (`_order_chain`) does. A relation that narrows nothing yields
        # exactly its table's rows, and a narrowing one at most its
        # table's: one whose table is no larger than some relation that
        # narrows nothing can never be the largest input. A narrowing
        # relation larger than all of those is estimated instead; when
        # every relation narrows, the one of the largest table is.
        # A semi or anti join's right relation is in no estimate: it is
        # neither counted nor run here. A left join's right relation is
        # estimated, and runs only beside relations that must run (below):
        # the join keeps every left row, so with no estimate for it the
        # join's estimate is its left input's.
        #
        # A relation joined to one of a smaller table may instead stream
        # through that one's index inside a build for a third, and is
        # then never held whole (PDS-H q3's orders through its
        # customers, inside the build its lines probe): running it ahead
        # would materialize every filtered row. Such a relation is
        # estimated (`deferred`), and runs only once the estimated tree
        # builds on it whole, the tree then chosen again with its count.
        var narrows = List[Bool](capacity=n)
        var deferred = List[Bool](length=n, fill=False)
        var widest_plain = -1
        var largest = 0
        while largest < n - 1 and hidden[largest]:
            largest += 1
        for r in range(n):
            narrows.append(
                not hidden[r]
                and self._nodes[relations[r]].kind != SCAN_FRAME
                and plan_filters_rows(self, relations[r])
            )
            if hidden[r]:
                continue
            if not narrows[r]:
                widest_plain = max(widest_plain, tables[r])
            if tables[r] > tables[largest]:
                largest = r
        var uncounted = 0
        for r in range(n):
            # An aggregate yields at most its input's rows: over a table
            # no larger than one that narrows nothing, it can never be
            # the largest input, so it is a build side that runs in
            # full whatever the order. It runs now, as a narrowing
            # relation does, rather than have its input's filter
            # sampled for an estimate and then run anyway. Over a larger
            # table it is estimated (`plan_aggregate_estimate`), and
            # runs only where the chosen order puts it: PDS-H q20's
            # lineitem quantities per part and supplier, against 800K
            # partsupp rows.
            var small_aggregate = False
            if aggregated[r] and not hidden[r]:
                var input_rows = plan_aggregate_input_rows(self, relations[r])
                small_aggregate = (
                    input_rows >= 0 and input_rows <= widest_plain
                ) or (widest_plain < 0 and r != largest)
            if hidden[r]:
                estimates.append(1)
            elif small_aggregate:
                if nullable[r]:
                    uncounted |= 1 << r
                    estimates.append(1)
                else:
                    estimates.append(-1)
            elif nullable[r]:
                var estimate = plan_relation_estimate(self, relations[r])
                if estimate < 0:
                    uncounted |= 1 << r
                estimates.append(max(estimate, 1))
            elif narrows[r] and (
                (tables[r] >= 0 and tables[r] <= widest_plain)
                or (widest_plain < 0 and r != largest)
            ):
                var smaller_neighbor = False
                for e in edges:
                    var other = e.b if e.a == r else (e.a if e.b == r else -1)
                    if (
                        other >= 0
                        and tables[other] >= 0
                        and tables[other] < tables[r]
                    ):
                        smaller_neighbor = True
                if smaller_neighbor:
                    estimates.append(plan_relation_estimate(self, relations[r]))
                    deferred[r] = estimates[r] >= 0
                else:
                    estimates.append(-1)
            else:
                estimates.append(plan_relation_estimate(self, relations[r]))
            if estimates[r] < 0:
                runs.append(r)
        # A left join's right relation is built whole in every order. With
        # no estimate, when other relations must run anyway, it runs with
        # them, at once, and is counted: independent pipelines run
        # together, as `_prefetch_breakers` runs them, not one after
        # another.
        if len(runs) > 0:
            for r in range(n):
                if uncounted & (1 << r) != 0:
                    runs.append(r)
                    uncounted &= ~(1 << r)
        # Filters comparing a column of one relation with a column of
        # another: edges of DuckDB's graph (`CreateHyperGraphEdges`),
        # which the estimate counts as join conditions.
        var compares = List[FilterEdge]()
        for f in filters:
            for part in conjuncts(self._nodes[f].exprs[0]):
                var compare = _comparison_edge(part, provider)
                if compare:
                    compares.append(compare.take())
        # Each compared column's distinct count, where its relation's
        # table keeps statistics: DuckDB's estimator then takes a set's
        # domain as the largest count among its columns. Read before any
        # relation runs, from the tables under them: DuckDB keeps a
        # column's base-table count whatever filters narrow it.
        var compared = List[Tuple[Int, String]]()
        for e in edges:
            for i in range(len(e.a_keys)):
                compared.append((e.a, e.a_keys[i]))
                compared.append((e.b, e.b_keys[i]))
        for f in compares:
            compared.append((f.a, f.a_column))
            compared.append((f.b, f.b_column))
        var distinct = Dict[String, Float64]()
        for pair in compared:
            var key = String(pair[0]) + "/" + pair[1]
            if key in distinct:
                continue
            # Through renames: a relation's key is often an alias of its
            # table's column (TPC-DS's d1_sk for d_date_sk).
            var count = plan_spine_distinct(self, relations[pair[0]], pair[1])
            if count > 0:
                distinct[key] = Float64(count)
        # Relations that must run to be counted.
        plan_run_relations(
            self, relations, runs, aggregated, estimates, streaming, batch_size
        )
        if plan_empty_graph(
            self, top, runs, hidden, nullable, estimates, batch_size
        ):
            changed = True
            continue
        for r in range(n):
            var table = tables[r]
            var estimated = estimates[r]
            rows.append(Float64(estimated))
            if uncounted & (1 << r) != 0 and table <= 0:
                # No rows known: its keys bound no domain.
                domains.append(1.0e18)
            else:
                domains.append(Float64(table if table > 0 else estimated))
        # An aggregate relation that ran is a new table with statistics
        # of its own (`ExtractAggregationStats`): its columns' counts are
        # read now.
        for pair in compared:
            var key = String(pair[0]) + "/" + pair[1]
            if key in distinct:
                continue
            # Through renames: a relation's key is often an alias of its
            # table's column (TPC-DS's d1_sk for d_date_sk).
            var count = plan_spine_distinct(self, relations[pair[0]], pair[1])
            if count > 0:
                distinct[key] = Float64(count)
        var chosen = order_joins(
            rows, domains, edges, distinct, compares, ops, uncounted
        )
        if not chosen:
            continue
        var tree = chosen.take()
        # Deferred relations the estimated tree builds on whole run now,
        # and the tree is chosen again with their counts.
        var swaps = plan_tree_swaps(self, tree, relations, columns, edges, ops)
        var later = List[Int]()
        for e in range(n, len(tree.mask)):
            var build = tree.left[e] if swaps[e] else tree.right[e]
            if build < n and deferred[build]:
                later.append(build)
        if len(later) > 0:
            plan_run_relations(
                self,
                relations,
                later,
                aggregated,
                estimates,
                streaming,
                batch_size,
            )
            for r in later:
                rows[r] = Float64(estimates[r])
            if plan_empty_graph(
                self, top, later, hidden, nullable, estimates, batch_size
            ):
                changed = True
                continue
            chosen = order_joins(
                rows, domains, edges, distinct, compares, ops, uncounted
            )
            if not chosen:
                continue
            tree = chosen.take()
            swaps = plan_tree_swaps(self, tree, relations, columns, edges, ops)
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
            # Its build side is the tree's (`plan_tree_swaps`), which
            # knows the estimates of the joins under it.
            join.sides_fixed = True
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


def plan_run_relations(
    mut self: QueryContext,
    relations: List[Int],
    runs: List[Int],
    aggregated: List[Bool],
    mut estimates: List[Int],
    streaming: Bool,
    batch_size: Int,
) raises:
    """Run join-graph relations `runs` for their counts (`estimates`),
    each replaced by its result so nothing runs twice. They are
    independent pipelines, scheduled to run at once (`run_pipelines`),
    not one after another on a crew their serial phases leave partly
    idle."""
    if len(runs) == 0:
        return
    var jobs = List[_SubplanPipeline](capacity=len(runs))
    var metas = List[MetaPipeline](capacity=len(runs))
    for r in runs:
        jobs.append(
            _SubplanPipeline(self.copy(), relations[r], streaming, batch_size)
        )
        metas.append(MetaPipeline(META_REGULAR))
    run_pipelines(jobs, metas)
    for k in range(len(runs)):
        var r = runs[k]
        estimates[r] = jobs[k].result.height()
        # An aggregate's result is a new table: it keeps distinct
        # counts of its own columns, as DuckDB's estimator gives an
        # aggregate relation's columns (`ExtractAggregationStats`).
        if aggregated[r]:
            self._frames.append(jobs[k].result.copy()._with_statistics())
        else:
            self._frames.append(jobs[k].result.copy())
        self._schemas.append(Optional[CsvSchema]())
        self._nodes[relations[r]] = _frame_scan_node(slot=len(self._frames) - 1)


def plan_probe_largest_inputs(
    mut self: QueryContext, root: Int, streaming: Bool, batch_size: Int
) raises -> Bool:
    """Start each chain of inner joins from its largest input.

    A chain streams its first input through hash tables built on every
    other input, so the first input is the only one never hashed.
    Re-root the chain at its largest input, and let the
    other inputs join outward from it along the same key pairs.

    Only where the rows' order cannot show (`_order_free_above`): an
    inner join's rows follow its left input, so a new first input
    changes their order.
    """
    var parents = plan_live_parents(self, root)
    var changed = False
    for top in range(len(parents)):
        if parents[top] == -2:
            continue
        ref node = self._nodes[top]
        if node.kind != JOIN or not _chain_member(node):
            continue
        var parent = parents[top]
        if (
            parent >= 0
            and self._nodes[parent].left == top
            and (
                self._nodes[parent].kind == SELECT
                or self._nodes[parent].kind == JOIN
            )
            and _chain_member(self._nodes[parent])
        ):
            continue
        if not plan_order_free_above(self, top, parents):
            continue
        if plan_probe_largest_input(self, top, streaming, batch_size):
            changed = True
    return changed


def plan_probe_largest_input(
    mut self: QueryContext, top: Int, streaming: Bool, batch_size: Int
) raises -> Bool:
    """Re-root the join chain ending at `top`; see
    `_probe_largest_inputs`. False when the chain stays as it is."""
    var joins = List[Int]()
    var cursor = top
    while cursor >= 0:
        ref node = self._nodes[cursor]
        if node.kind == JOIN and _chain_member(node):
            joins.append(cursor)
        elif not (node.kind == SELECT and _chain_member(node)):
            break
        cursor = node.left
    if cursor < 0 or len(joins) == 0:
        return False
    # The first input is the lowest join's whole left input, with any
    # column selection on it, so it keeps supplying only those columns.
    var base = self._nodes[joins[len(joins) - 1]].left
    joins.reverse()
    var n = len(joins)
    # Table 0 is the chain's first input; table p + 1 is join p's right.
    var tables: List[Int] = [base]
    for p in range(n):
        tables.append(self._nodes[joins[p]].right)
    # Which table supplies each column. A name two tables share is
    # allowed only as a join's key pair of one name; anything else
    # would be renamed by a suffix in one order and not in another.
    var provider = Dict[String, Int]()
    var shared = List[Tuple[String, Int, Int]]()
    for t in range(n + 1):
        for c in plan_columns_of(self, tables[t]):
            if c in provider:
                shared.append((c, provider[c], t))
            else:
                provider[c] = t
    # Join p links table p + 1 to the one table holding its left keys.
    var linked = List[Int]()
    for p in range(n):
        ref node = self._nodes[joins[p]]
        if len(node.names) != len(node.right_keys):
            return False
        var from_table = -1
        for name in node.names:
            if name not in provider:
                return False
            if from_table == -1:
                from_table = provider[name]
            elif from_table != provider[name]:
                return False
        if from_table > p:
            return False
        linked.append(from_table)
    for item in shared:
        var allowed = False
        for p in range(n):
            ref node = self._nodes[joins[p]]
            if not (
                (linked[p] == item[1] and p + 1 == item[2])
                or (linked[p] == item[2] and p + 1 == item[1])
            ):
                continue
            for i in range(len(node.names)):
                if node.names[i] == item[0] and node.right_keys[i] == item[0]:
                    allowed = True
        if not allowed:
            return False
    # The first input streams through the chain as slices, so the
    # columns it supplies cost nothing to carry; as a right input each
    # would be gathered for every joined row. Move it only when it
    # supplies nothing but its join keys.
    var top_columns = plan_columns_of(self, top)
    for name in top_columns:
        if provider[name] != 0:
            continue
        var key = False
        for p in range(n):
            if linked[p] == 0 and name in self._nodes[joins[p]].names:
                key = True
        if not key:
            return False
    # Rows per table, estimated only when needed (-2: not yet). A
    # table that cannot hold more rows than the first input is never
    # a candidate, so most chains estimate nothing beyond the first.
    var rows = List[Int](length=n + 1, fill=-2)
    rows[0] = plan_estimated_rows(self, tables[0])
    if rows[0] < 0:
        return False
    # Candidates by size. For each, compare the rows hashed along the
    # path between it and the first input, the only joins whose build
    # side changes: now each table on the path but the first is built;
    # re-rooted, each but the candidate is.
    var largest = 0
    var saved = 0
    for candidate in range(1, n + 1):
        var bound = plan_height_bound(self, tables[candidate])
        if bound < 0 or bound <= rows[0]:
            continue
        rows[candidate] = plan_estimated_rows(self, tables[candidate])
        if rows[candidate] <= rows[0]:
            continue
        var path = List[Int]()
        var t = candidate
        var known = True
        while t != 0:
            if rows[t] == -2:
                rows[t] = plan_estimated_rows(self, tables[t])
            known = known and rows[t] >= 0
            path.append(t)
            t = linked[t - 1]
        if not known:
            continue
        var before = 0
        var after = 0
        for k in range(len(path)):
            var table = path[k]
            # Built now, keyed by the join's right keys.
            before += rows[table]
            # Built after re-rooting: the table on the other end,
            # keyed by the join's left keys.
            var other = linked[table - 1]
            after += rows[other]
        if before - after > saved:
            saved = before - after
            largest = candidate
    if largest == 0:
        return False
    # Join outward from the largest table. A join met from its right
    # table's side swaps its key lists; the keys it then drops are the
    # other table's, whose values the surviving keys hold, so later
    # uses of those names read the surviving ones (`renamed`).
    var visited = List[Bool](length=n + 1, fill=False)
    visited[largest] = True
    var placed = List[Bool](length=n, fill=False)
    var renamed = Dict[String, String]()
    var added = List[PlanNode]()
    var current = tables[largest]
    # The columns the chain holds so far. Each join must find its left
    # keys there and add no name already present.
    var have = Dict[String, Bool]()
    for c in plan_columns_of(self, tables[largest]):
        have[c] = True
    for _ in range(n):
        var pick = -1
        for p in range(n):
            if not placed[p] and visited[linked[p]] != visited[p + 1]:
                pick = p
                break
        if pick < 0:
            return False
        var node = self._nodes[joins[pick]].copy()
        node.left = current
        var forward = visited[linked[pick]]
        var left_keys = node.names.copy() if forward else node.right_keys.copy()
        for i in range(len(left_keys)):
            while left_keys[i] in renamed:
                left_keys[i] = renamed[left_keys[i]]
        if forward:
            visited[pick + 1] = True
        else:
            var dropped = node.names.copy()
            node.right = tables[linked[pick]]
            node.right_keys = dropped.copy()
            for i in range(len(dropped)):
                if dropped[i] != left_keys[i]:
                    renamed[dropped[i]] = left_keys[i]
            visited[linked[pick]] = True
        for key in left_keys:
            if key not in have:
                return False
        for c in plan_columns_of(self, node.right):
            if c in node.right_keys:
                continue
            if c in have:
                return False
            have[c] = True
        node.names = left_keys^
        placed[pick] = True
        current = len(self._nodes) + len(added)
        added.append(node^)
    for node in added:
        self._nodes.append(node.copy())
    var keep = List[Expr]()
    for name in top_columns:
        var source = name
        while source in renamed:
            source = renamed[source]
        keep.append(col(source) if source == name else col(source).alias(name))
    self._nodes[top] = _select_node(left=current, exprs=keep)
    trace_path("lazy.probe_largest")
    return True


def plan_selectivity(
    mut self: QueryContext, index: Int, streaming: Bool, batch_size: Int
) raises -> Float64:
    """The share of its table's rows that node `index` keeps, executing
    it and replacing it with its result; 1 when it is a plain table or
    no table is known."""
    var table = plan_spine_rows(self, index)
    ref node = self._nodes[index]
    if node.kind == SCAN_FRAME and node.source_slot() in self._origin_rows:
        # A shared subplan's result: the share of its table it kept.
        return Float64(self._frames[node.source_slot()].height()) / Float64(
            table
        )
    if table <= 0 or node.kind == SCAN_FRAME:
        return 1.0
    var frame = self._execute(index, False, streaming, batch_size)
    var rows = frame.height()
    self._frames.append(frame^)
    self._schemas.append(Optional[CsvSchema]())
    self._nodes[index] = _frame_scan_node(slot=len(self._frames) - 1)
    return Float64(rows) / Float64(table)


def plan_order_chain(
    mut self: QueryContext, top: Int, streaming: Bool, batch_size: Int
) raises -> Bool:
    """Reorder the chain whose last node is `top`; see `_order_joins`.
    False when join topology stays unchanged. Measured inputs remain
    cached even when no step moves or a candidate is rejected."""
    var spine = List[Int]()
    var cursor = top
    while cursor >= 0 and _chain_member(self._nodes[cursor]):
        spine.append(cursor)
        cursor = self._nodes[cursor].left
    var base = cursor
    if base < 0:
        return False
    # Steps bottom to top, without the column selections.
    var steps = List[Int]()
    var join_count = 0
    for k in range(len(spine)):
        var i = spine[len(spine) - 1 - k]
        if self._nodes[i].kind == SELECT:
            continue
        steps.append(i)
        if self._nodes[i].kind == JOIN:
            join_count += 1
    if join_count < 2:
        return False
    var top_columns = plan_columns_of(self, top)
    # Which step supplies each column (-1 the base). Every name must be
    # unique across the chain, so no join renames a column in any order.
    var provider = Dict[String, Int]()
    for c in plan_columns_of(self, base):
        provider[c] = -1
    var outputs = List[List[String]]()
    for p in range(len(steps)):
        ref node = self._nodes[steps[p]]
        var produced = List[String]()
        if node.kind == JOIN:
            for c in plan_columns_of(self, node.right):
                if c in node.right_keys:
                    continue
                if c in provider:
                    return False
                provider[c] = p
                produced.append(c)
        outputs.append(produced^)
    var needs = List[List[String]]()
    for p in range(len(steps)):
        ref node = self._nodes[steps[p]]
        if node.kind == JOIN:
            needs.append(node.names.copy())
        else:
            var reads = _references(node.exprs[0])
            if not reads:
                return False
            needs.append(reads.value().copy())
        for name in needs[p]:
            if name not in provider:
                return False
    # owner[p]: the step whose output now carries step p's columns.
    var owner = List[Int](capacity=len(steps))
    for p in range(len(steps)):
        owner.append(p)
    var selectivity = List[Float64](length=len(steps), fill=1.0)
    var measured = List[Bool](length=len(steps), fill=False)
    var pushed = False
    # Keep tentative topology changes separate from input results.
    # Restoring a chain must preserve scans produced by measurement.
    var saved = self._nodes.copy()
    for reverse in range(len(steps)):
        var p = len(steps) - 1 - reverse
        if self._nodes[steps[p]].kind != JOIN:
            continue
        var keys = self._nodes[steps[p]].names.copy()
        var host = -2
        for name in keys:
            var from_step = provider[name]
            var o = owner[from_step] if from_step >= 0 else -1
            if host == -2:
                host = o
            elif host != o:
                host = -1
        if host < 0:
            continue
        # The host's right input must not hold a name this join adds.
        var host_columns = plan_columns_of(self, self._nodes[steps[host]].right)
        var clash = False
        for c in outputs[p]:
            clash = clash or c in host_columns
        if clash:
            continue
        # The pushed input must be small next to the host's table, as a
        # dimension of it: joining 1.5M lineitem rows into 57K orders
        # first made PDS-H q10 12% slower. Its table bounds its rows
        # before it runs, and its rows bound them after.
        var host_rows = plan_spine_rows(self, self._nodes[steps[host]].right)
        var table = plan_spine_rows(self, self._nodes[steps[p]].right)
        if host_rows < 0 or table < 0 or 4 * table > host_rows:
            continue
        var share = plan_selectivity(
            self, self._nodes[steps[p]].right, streaming, batch_size
        )
        selectivity[p] = share
        measured[p] = True
        if share >= 0.5:
            continue
        var rows = plan_known_height(self, self._nodes[steps[p]].right)
        if rows < 0 or 4 * rows > host_rows:
            continue
        var inner = self._nodes[steps[p]].copy()
        inner.left = self._nodes[steps[host]].right
        self._nodes.append(inner^)
        self._nodes[steps[host]].right = len(self._nodes) - 1
        for q in range(len(steps)):
            if owner[q] == p:
                owner[q] = host
        pushed = True
    # Selectivity of the joins that stay in the chain, measured only
    # where it can change the order: a narrowed input whose join could
    # run before one that precedes it, and that one. A plain table keeps
    # all its rows and never moves ahead.
    var need = List[Bool](length=len(steps), fill=False)
    var done = List[Bool](length=len(steps), fill=False)
    for p in range(len(steps)):
        if owner[p] != p:
            done[p] = True
    for p in range(len(steps)):
        if owner[p] != p:
            continue
        if self._nodes[steps[p]].kind == JOIN:
            for q in range(p + 1, len(steps)):
                if (
                    owner[q] != q
                    or self._nodes[steps[q]].kind != JOIN
                    or not plan_available(self, needs[q], provider, owner, done)
                ):
                    continue
                if plan_measurable(self, self._nodes[steps[q]].right):
                    need[q] = True
                    need[p] = need[p] or plan_measurable(
                        self, self._nodes[steps[p]].right
                    )
        done[p] = True
    for p in range(len(steps)):
        if not need[p] or measured[p]:
            continue
        selectivity[p] = plan_selectivity(
            self, self._nodes[steps[p]].right, streaming, batch_size
        )
    # Greedy order: filters as soon as their columns are available, then
    # the most selective available join (ties keep the original order).
    var placed = List[Bool](length=len(steps), fill=False)
    for p in range(len(steps)):
        if owner[p] != p:
            placed[p] = True
    var order = List[Int]()
    var remaining = 0
    for p in range(len(steps)):
        if not placed[p]:
            remaining += 1
    while len(order) < remaining:
        var progress = False
        for p in range(len(steps)):
            if placed[p] or self._nodes[steps[p]].kind != FILTER:
                continue
            if plan_available(self, needs[p], provider, owner, placed):
                placed[p] = True
                order.append(p)
                progress = True
        var best = -1
        for p in range(len(steps)):
            if placed[p] or self._nodes[steps[p]].kind != JOIN:
                continue
            if not plan_available(self, needs[p], provider, owner, placed):
                continue
            if best < 0 or selectivity[p] < selectivity[best]:
                best = p
        if best >= 0:
            placed[best] = True
            order.append(best)
            progress = True
        if not progress:
            plan_restore_measured_chain(self, saved^)
            return False
    var moved = pushed
    var expected = 0
    for p in range(len(steps)):
        if owner[p] != p:
            continue
        if order[expected] != p:
            moved = True
        expected += 1
    if not moved:
        plan_restore_measured_chain(self, saved^)
        return False
    var current = base
    for p in order:
        var node = self._nodes[steps[p]].copy()
        node.left = current
        self._nodes.append(node^)
        current = len(self._nodes) - 1
    var keep = List[Expr]()
    for name in top_columns:
        keep.append(col(name))
    self._nodes[top] = _select_node(left=current, exprs=keep)
    trace_path("lazy.join_order")
    return True


def plan_restore_measured_chain(
    mut self: QueryContext, var saved: List[PlanNode]
):
    """Discard speculative topology, retaining executed original inputs.

    Only original nodes can be restored as cached scans. Results of
    newly appended speculative joins have no equivalent original node;
    scan-slot compaction drops those once unreachable.
    """
    for i in range(len(saved)):
        if self._nodes[i].kind == SCAN_FRAME and saved[i].kind != SCAN_FRAME:
            saved[i] = self._nodes[i].copy()
    self._nodes = saved^
