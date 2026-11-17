"""Join order, chosen from plan statistics before anything runs, and build
sides, chosen from them as a stream starts, a filtered input counted first
where a guess would move it."""
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


from dataframe.frame import DataFrame, concat
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
    JOIN_RIGHT_ANTI,
    JOIN_RIGHT_SEMI,
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
    plan_shared_id,
    plan_slot_rows,
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
    DROP,
    FILTER,
    JOIN,
    PlanNode,
    SCAN_FRAME,
    SELECT,
    WITH_COLUMNS,
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


# Penalize building a standalone relation against an already-streaming join by
# 15% to account for the opportunity to keep those join rows in flight.
comptime _PREFER_RIGHT_DEEP_PENALTY = 0.15
# A build side that filters earns its runtime filter's bonus only against a
# probe scan of at least a million rows and 64 times its own rows: below that,
# the rows the filter can skip do not repay building it, and a build with many
# keys filters little. A semi join's filtering left builds for a right scan
# three times both sides. Empirical choices.
comptime _MIN_FILTER_TARGET_CARDINALITY = 1000000
comptime _MAX_BUILD_TO_TARGET_RATIO = 64
comptime _SEMI_JOIN_FILTER_TARGET_RATIO = 3


def plan_side_estimate(
    self: QueryContext, mut stats: PlanStatistics, index: Int
) raises -> Int:
    """Rows node `index` yields, estimated from statistics before it runs,
    as the build side optimizer reads a child's estimated cardinality, with
    nothing sampled or run. -1 when an input has no estimate."""
    return plan_cardinality(self, stats, index)


def _filter_target_rows(
    self: QueryContext, side: Int, keys: List[String]
) raises -> Int:
    """The most rows of a table scan under `side` that a runtime filter on
    `keys` would reach: the scans supplying the keys, whatever filters later
    reduce."""
    var most = 0
    for key in keys:
        var found = plan_provider_scan(self, side, key)
        if found[0] < 0:
            continue
        most = max(
            most, plan_slot_rows(self, self._nodes[found[0]].source_slot())
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
    publishes into the probe side's scans: the target scan's rows over the
    build's, when the build filters its tables and the scan is large next to it;
    else 1."""
    if not plan_is_filtering(self, build):
        return 1.0
    var target = _filter_target_rows(self, probe, probe_keys)
    if target < _MIN_FILTER_TARGET_CARDINALITY:
        return 1.0
    if build_rows > 0 and build_rows > target // _MAX_BUILD_TO_TARGET_RATIO:
        return 1.0
    return Float64(target) / Float64(max(build_rows, 1))


def _child_joins(self: QueryContext, index: Int) -> Int:
    """The joins in the subplan at `index`: both inputs of a join, the first
    input of anything else."""
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
    """Whether the join at `at` should build on its left input rather than its
    right: each side's hash table bytes (`build_size` over its schema and
    estimated rows), the right's divided by its runtime filter bonus where the
    join type publishes one, the left's where the flipped type does (an anti
    join's RIGHT_ANTI does not), and the right's 15% dearer when it holds no
    join and the left does; the left builds when the right costs more, or for a
    semi join whose filtering left would filter a right scan three times both
    sides' rows. False when a side has no estimate."""
    ref node = self._nodes[at]
    if (
        node.sides_fixed
        or len(node.names) == 0
        or node.how
        not in [JOIN_INNER, JOIN_LEFT, JOIN_RIGHT, JOIN_SEMI, JOIN_ANTI]
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
    # A filter built from the build side's keys may drop probe rows only
    # when an unmatched probe row produces no output: inner, semi and right
    # joins (a left join keeps every probe row, an anti join's filter would
    # need inverting). Flipped, an inner, right and right semi join can,
    # a left join (a flipped right join) cannot.
    if (
        node.how == JOIN_INNER
        or node.how == JOIN_SEMI
        or node.how == JOIN_RIGHT
    ):
        right_cost /= _filter_build_bonus(
            self, node.left, node.names, node.right, rhs
        )
    if node.how != JOIN_ANTI and node.how != JOIN_RIGHT:
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
    """Flip the inner join at `at`: its right input probes and its left builds,
    under a selection that restores the join's columns, their order, names and
    key types. False, with nothing changed, when a name the sides share other
    than as a key would be renamed by the join, or a right key repeats."""
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
    """Flip the left join at `at` into a right join: its right input probes and
    its left builds, the build rows no probe row matched coming out last, under
    a selection that restores the join's columns: the left's, then the right's
    other than its keys. False, with nothing changed, when the sides share a
    column name, a key repeats, or the key types differ or need the general join
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


def _flip_right_join(
    mut plan: QueryContext, at: Int, batch_size: Int
) raises -> Bool:
    """Flip the right join at `at` into a left join with the inputs
    swapped, which gives the same rows: its right input probes, keeping
    every row, and its left builds, under a selection that restores the
    join's columns and their order. False, with nothing changed, when the
    sides share a column name, a key repeats, or the key types differ or
    need the general join (categorical, nested)."""
    var node = plan._nodes[at].copy()
    for keys in [node.names.copy(), node.right_keys.copy()]:
        for i in range(len(keys)):
            for j in range(i):
                if keys[i] == keys[j]:
                    return False
    var left = plan._execute(node.left, True, False, batch_size)
    var right = plan._execute(node.right, True, False, batch_size)
    var left_names = left.columns()
    for name in right.columns():
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
    for name in plan._execute(at, True, False, batch_size).columns():
        exprs.append(col(name))
    var join = node.copy()
    join.left = node.right
    join.right = node.left
    join.names = node.right_keys.copy()
    join.right_keys = node.names.copy()
    join.how = JOIN_LEFT
    join.text = "left"
    join.sides_fixed = True
    plan._nodes.append(join^)
    plan._nodes[at] = _select_node(left=len(plan._nodes) - 1, exprs=exprs)
    trace_path("lazy.swap_right_join_sides")
    return True


def _flip_semi_join(
    mut plan: QueryContext, at: Int, batch_size: Int
) raises -> Bool:
    """Flip a semi/anti join so its left input builds and its right probes. The
    build-side scan emits matched/unmatched left rows respectively. Decline
    differing or nested key types that cannot share this hash table."""
    var node = plan._nodes[at].copy()
    var left = plan._execute(node.left, True, False, batch_size)
    var right = plan._execute(node.right, True, False, batch_size)
    for i in range(len(node.names)):
        var ltype = left.column(node.names[i]).dtype()
        var rtype = right.column(node.right_keys[i]).dtype()
        if ltype != rtype or ltype.is_nested():
            return False
    var join = node.copy()
    join.left = node.right
    join.right = node.left
    join.names = node.right_keys.copy()
    join.right_keys = node.names.copy()
    join.how = JOIN_RIGHT_SEMI if node.how == JOIN_SEMI else JOIN_RIGHT_ANTI
    join.sides_fixed = True
    plan._nodes[at] = join^
    trace_path("lazy.swap_semi_join_sides")
    return True


def _filtered_table(self: QueryContext, index: Int) -> Bool:
    """Whether the subplan at `index` is one in-memory table's rows under
    row-local filters and projections, at least one of them a filter: an
    input whose row count is a guess until it runs, and whose run is a
    single pass over its table."""
    var cursor = index
    var filtered = False
    while cursor >= 0:
        ref node = self._nodes[cursor]
        if node.kind == FILTER or node.kind == SELECT or node.kind == DROP:
            if not _stream_rows(node):
                return False
            filtered = filtered or node.kind == FILTER
            cursor = node.left
        elif node.kind == WITH_COLUMNS and _stream_rows(node):
            cursor = node.left
        else:
            break
    return (
        filtered
        and cursor >= 0
        and self._nodes[cursor].kind == SCAN_FRAME
        and plan_shared_id(self, self._nodes[cursor].source_slot()) < 0
    )


def _builds_on_left_rows(
    self: QueryContext,
    mut stats: PlanStatistics,
    at: Int,
    rows: Int,
    batch_size: Int,
) raises -> Bool:
    """Whether the join at `at` builds on its left input with that input
    taken to have `rows` rows. `stats` is one memo for every such question:
    the choice reads only the two inputs' estimates, the left's set here."""
    stats.rows[self._nodes[at].left] = rows
    return _builds_on_left(self, stats, at, batch_size)


def _run_filtered_source(
    self: QueryContext, at: Int, batch_size: Int
) raises -> Optional[DataFrame]:
    """The rows of the filtered table on the left of the join at `at`, run
    while the join could still build on them: None when it would not build
    on even an empty left, or once the rows kept, or those rows scaled from
    the part of the table read to all of it, are more than it would build
    on. The table runs in consecutive parts, each twice the last from one
    batch: the share a filter keeps in the rows read so far shows early
    whether its rows are many, so a large input stops after a little of its
    table, while one the join builds on reads its table once."""
    var stats = PlanStatistics()
    if not _builds_on_left_rows(self, stats, at, 0, batch_size):
        return None
    var index = self._nodes[at].left
    var cursor = index
    while self._nodes[cursor].kind != SCAN_FRAME:
        cursor = self._nodes[cursor].left
    var slot = self._nodes[cursor].source_slot()
    var table = self._frames[slot].copy()
    var rows = table.height()
    var part = self.copy()
    # A probe of the input, not an operator of the query: its runs stay out
    # of the query's execution report.
    part._report = None
    var parts = List[DataFrame]()
    var kept = 0
    var offset = 0
    var length = max(batch_size, 1)
    while offset < rows or len(parts) == 0:
        # A slot of its own: plan copies share their table slots.
        part._frames.append(table.slice(offset, length))
        part._nodes[cursor].set_source_slot(len(part._frames) - 1)
        # The first part is one batch, run as it is: a stream's workers pay
        # off only over the larger parts after it.
        var made = part._execute(index, False, offset > 0, batch_size)
        kept += made.height()
        offset = min(offset + length, rows)
        if not _builds_on_left_rows(self, stats, at, kept, batch_size):
            return None
        if offset < rows:
            var scaled = Int(Float64(kept) * Float64(rows) / Float64(offset))
            if not _builds_on_left_rows(self, stats, at, scaled, batch_size):
                return None
        parts.append(made^)
        length *= 2
    if len(parts) == 1:
        return Optional(parts.pop())
    return Optional(concat(parts))


def _flips(
    self: QueryContext,
    mut stats: PlanStatistics,
    spine: List[Int],
    batch_size: Int,
) raises -> List[Int]:
    """The joins of `spine` (top first) that build on their left input,
    visited children first; a flip leaves every other join's inputs, their
    estimates and schemas as they were."""
    var flips = List[Int]()
    for k in range(len(spine) - 1, -1, -1):
        if _builds_on_left(self, stats, spine[k], batch_size):
            flips.append(spine[k])
    return flips^


def plan_build_probe_sides(
    self: QueryContext, top: Int, batch_size: Int
) raises -> Optional[QueryContext]:
    """Choose build sides from input cardinalities before scheduling. Flip
    inner joins to probe the chosen build, exchange left/right outer kinds,
    and use build-side semi/anti kinds as needed. Mark each choice fixed so
    later cardinality observations cannot repeatedly reverse it. Return None
    when all joins keep their sides and nothing ran.

    Estimates choose, but the share of rows a filter keeps is a guess, often
    off by tens of times, and acting on it can hash the larger side. So an
    input that is a filtered table, and that a choice here would move, runs
    first and its exact count decides; its rows then serve the join from
    memory, as its build or as the stream's source. The joins of a stream
    that counted hold their sides whatever the counts chose."""
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
    if len(spine) == 0:
        return None
    var stats = PlanStatistics()
    var flips = _flips(self, stats, spine, batch_size)
    # Inputs whose counts are guesses (filtered tables) and that a choice
    # here would move. A join's right input is read whole whether it builds
    # or not, so a flipping join's is counted in full. The stream's source
    # is read whole only if it builds: it is counted while it stays under
    # the most rows at which the lowest join would build on it, and left to
    # stream, its partial count dropped, once it passes them. Inputs on a
    # join's left above the lowest are joins, made by the stream itself.
    var counted = List[Int]()
    var frames = List[DataFrame]()
    if self._prefetched:
        var bottom = spine[len(spine) - 1]
        var source = self._nodes[bottom].left
        for at in flips:
            var right = self._nodes[at].right
            if right in counted or right in self._prefetched.value()[]:
                continue
            if not _filtered_table(self, right):
                continue
            trace_path("lazy.count_filtered_build_input")
            frames.append(self._execute(right, False, True, batch_size))
            counted.append(right)
        if (
            source not in counted
            and source not in self._prefetched.value()[]
            and _filtered_table(self, source)
        ):
            var frame = _run_filtered_source(self, bottom, batch_size)
            if frame:
                trace_path("lazy.count_filtered_build_input")
                frames.append(frame.take())
                counted.append(source)
    if len(counted) > 0:
        stats = PlanStatistics()
        for i in range(len(counted)):
            stats.rows[counted[i]] = frames[i].height()
        flips = _flips(self, stats, spine, batch_size)
    if len(flips) == 0 and len(counted) == 0:
        return None
    var plan = self.copy()
    var changed = len(counted) > 0
    if changed:
        # Every join here was decided on what the counts showed.
        for at in spine:
            plan._nodes[at].sides_fixed = True
    for at in flips:
        var how = plan._nodes[at].how
        if how == JOIN_INNER:
            if _flip_inner_join(plan, at, batch_size):
                changed = True
        elif how == JOIN_LEFT:
            if _flip_left_join(plan, at, batch_size):
                changed = True
        elif how == JOIN_RIGHT:
            if _flip_right_join(plan, at, batch_size):
                changed = True
        elif _flip_semi_join(plan, at, batch_size):
            changed = True
    if not changed:
        return None
    if len(counted) > 0:
        # A counted input the stream reads through, its source after the
        # flips, becomes a scan of its rows; one that is a build waits in
        # the memo of computed inputs, where the build reads it.
        var through = List[Int]()
        cursor = top
        while cursor >= 0:
            ref step = plan._nodes[cursor]
            if not _stream_rows(step) and not _stream_join(step):
                break
            through.append(cursor)
            cursor = step.left
        if cursor >= 0:
            through.append(cursor)
        for i in range(len(counted)):
            if counted[i] in through:
                plan._frames.append(frames[i].copy())
                plan._schemas.append(Optional[CsvSchema]())
                plan._nodes[counted[i]] = _frame_scan_node(
                    slot=len(plan._frames) - 1
                )
            else:
                plan._prefetched.value()[][counted[i]] = frames[i].copy()
    return Optional(plan^)


def plan_order_joins(
    mut self: QueryContext, streaming: Bool, batch_size: Int
) raises:
    """Order each join graph by estimated cost, from statistics alone: nothing
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
    """Order a join graph subject to semantic conflict rules. Semi joins may
    move before inner joins whose keys they do not read. Predicates on an
    outer join's nullable side stay above it unless proven safe. Rebuild the
    graph with projected names preserved and choose hash-build sides after
    choosing join order."""
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
        # A repeated output name identifies one provider only when inner-join
        # key pairs coalesce all contributing columns. Record the other
        # relations in twins for validation.
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
        # Each join as an operator: its inputs' relations and the relations its
        # keys read.
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
            # One reading a left join's right columns runs after that join:
            # below it, the filter would see rows the left join has not yet
            # null-extended.
            for op in ops:
                if op.kind == OPERATOR_LEFT and mask & op.right_relations != 0:
                    mask |= op.syntactic
            filter_masks.append(mask)
        if not known:
            continue
        var output = plan_columns_of(self, top)
        # An uncounted preserved side does not reduce left-join cardinality.
        # For other missing relation estimates, retain the written graph:
        # choosing an order without comparable costs would be arbitrary.
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
        # An inner join with an empty input is empty.
        var every = List[Int](capacity=n)
        for r in range(n):
            if uncounted & (1 << r) == 0:
                every.append(r)
        if plan_empty_graph(
            self, top, every, hidden, nullable, estimates, batch_size
        ):
            changed = True
            continue
        # Cross-relation comparisons become graph edges and contribute join
        # conditions to the cardinality estimate.
        var compares = List[FilterEdge]()
        for f in filters:
            for part in conjuncts(self._nodes[f].exprs[0]):
                var compare = _comparison_edge(part, provider)
                if compare:
                    compares.append(compare.take())
        # Each compared column's distinct count by its source, for the sets'
        # total domains: reliable counts, counts from integer bounds, and
        # fallback rows.
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
            # Keep the tree's chosen sides until the build-side pass runs at
            # stream start, after join-order selection.
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
    """For each binary join tree entry, choose the build side using estimated
    rows times row width. Penalize a standalone input against an already-
    streaming join by 15%. A single-key integer progression needs no hash
    table and therefore has a lower build cost."""
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
    """Replace a join graph with an empty result when its semantics prove no
    output is possible. An inner/semi join with either input empty is empty;
    a left/anti join is empty only when its left input is empty. Preserve
    schema and skip unnecessary graph execution."""
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
