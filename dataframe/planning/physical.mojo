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

from dataframe.storage.column import (
    Column,
)

from dataframe.dtype import (
    DataType,
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
from dataframe.kernels.join_hash import (
    _PERFECT_SPAN,
)
from dataframe.join_order import (
    FilterEdge,
    JoinEdge,
    order_joins,
)
from dataframe.join_type import (
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
from dataframe.parallel import (
    run_jobs,
)
from dataframe.pipeline import (
    _int_bounds,
)
from dataframe.planning.context import (
    QueryContext,
)
from dataframe.planning.graph import (
    plan_aggregated,
    plan_available,
    plan_collect_graph,
    plan_columns_of,
    plan_compact_scan_slots,
    plan_estimated_rows,
    plan_filtered_table,
    plan_graph_step,
    plan_has_join,
    plan_has_kind,
    plan_height_bound,
    plan_indexed_relation,
    plan_known_height,
    plan_largest_scan,
    plan_live_parents,
    plan_measurable,
    plan_order_free_above,
    plan_progression_key,
    plan_provider_scan,
    plan_relation_estimate,
    plan_reorder,
    plan_spine_frame,
    plan_spine_rows,
)
from dataframe.planning.inputs import (
    _SubplanJob,
)
from dataframe.planning.model import (
    AGG,
    FILTER,
    JOIN,
    PlanNode,
    SCAN_FRAME,
    SELECT,
)
from dataframe.planning.properties import (
    _bound_filter,
    _chain_member,
    _comparison_edge,
    _is_scan,
    _key_set_filter,
    _references,
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


def plan_direct_key(
    self: QueryContext, index: Int, names: List[String]
) raises -> Bool:
    """Whether the one join key `names` of node `index` comes from an
    in-memory table whose whole key column is unique, never null and
    spans at most `_PERFECT_SPAN` values: any rows of it then index
    directly (`_perfect_slots`), so a stream probing them pays one load
    a row. Told from the table, before anything runs."""
    if len(names) != 1:
        return False
    var found = plan_provider_scan(self, index, names[0])
    if found[0] < 0:
        return False
    var key = self._frames[self._nodes[found[0]].source_slot()].column(found[1])
    if key.dtype().physical() != DataType.INT64 or key.null_count() > 0:
        return False
    # More rows than the span allows cannot all differ: no pass needed.
    if len(key) > _PERFECT_SPAN:
        return False
    var flat = key.rechunk() if key.is_chunked() else key.copy()
    ref column = flat._data[Column[Int64]]
    var n = len(column)
    if n == 0:
        return False
    var least = Int64.MAX
    var most = Int64.MIN
    for i in range(n):
        least = min(least, column._get(i))
        most = max(most, column._get(i))
    if most - least >= Int64(_PERFECT_SPAN):
        return False
    var seen = List[Bool](length=Int(most - least) + 1, fill=False)
    for i in range(n):
        var at = Int(column._get(i) - least)
        if seen[at]:
            return False
        seen[at] = True
    return True


def plan_swap_join_sides(
    self: QueryContext, at: Int, built: DataFrame, batch_size: Int
) raises -> Optional[QueryContext]:
    """A copy of the plan in which the inner join at `at` takes its
    executed right side, `built`, as its left and its left side, run
    now, as its right, under a selection that restores the join's
    columns: their order, names and key types. None, with nothing run,
    when a name the sides share other than as a key would be renamed
    by the join."""
    ref node = self._nodes[at]
    if not node.coalesce or node.how != JOIN_INNER:
        return None
    var left_names = plan_columns_of(self, node.left)
    var right_names = built.columns()
    for i in range(len(node.right_keys)):
        for j in range(i):
            if node.right_keys[i] == node.right_keys[j]:
                return None
    for name in right_names:
        if name in left_names and not (
            name in node.right_keys and name in node.names
        ):
            return None
    var left_frame = self._execute(node.left, False, True, batch_size)
    var exprs = List[Expr]()
    for name in left_frame.columns():
        var k = -1
        for i in range(len(node.names)):
            if node.names[i] == name:
                k = i
        if k < 0:
            exprs.append(col(name))
            continue
        var key = col(node.right_keys[k])
        var dtype = left_frame.column(name).dtype()
        if built.column(node.right_keys[k]).dtype() != dtype:
            key = key.cast(dtype)
        exprs.append(key.alias(name))
    for name in right_names:
        if name not in node.right_keys:
            exprs.append(col(name))
    var plan = self.copy()
    plan._frames.append(built.copy())
    plan._schemas.append(Optional[CsvSchema]())
    plan._nodes.append(_frame_scan_node(slot=len(plan._frames) - 1))
    var probe = len(plan._nodes) - 1
    plan._frames.append(left_frame^)
    plan._schemas.append(Optional[CsvSchema]())
    plan._nodes.append(_frame_scan_node(slot=len(plan._frames) - 1))
    var build = len(plan._nodes) - 1
    var join = node.copy()
    join.left = probe
    join.right = build
    join.names = node.right_keys.copy()
    join.right_keys = node.names.copy()
    plan._nodes.append(join^)
    plan._nodes[at] = _select_node(left=len(plan._nodes) - 1, exprs=exprs)
    trace_path("lazy.swap_join_sides")
    return Optional(plan^)


def plan_swap_left_join_sides(
    self: QueryContext, at: Int, built: DataFrame, batch_size: Int
) raises -> Optional[QueryContext]:
    """A copy of the plan in which the left join at `at` becomes a
    right join of its executed right side, `built`, as the probe and
    its left side, run now, as the build, under a selection that
    restores the join's columns: the left's, then the right's other
    than its keys. The probe streams through the pipeline against the
    build's index, and the build rows no probe row matched come out
    last (`_emit_unmatched`), as in Polars' `emit_unmatched_build` and
    DuckDB's right-outer scan of a swapped join. None, with nothing
    run, when the sides share a column name, a key repeats, or the
    left's key types differ from the right's or need the general join
    (categorical, nested)."""
    ref node = self._nodes[at]
    if not node.coalesce or node.how != JOIN_LEFT:
        return None
    var left_names = plan_columns_of(self, node.left)
    var right_names = built.columns()
    for i in range(len(node.right_keys)):
        for j in range(i):
            if node.right_keys[i] == node.right_keys[j]:
                return None
    for name in right_names:
        if name in left_names:
            return None
    var left_frame = self._execute(node.left, False, True, batch_size)
    for i in range(len(node.names)):
        var ltype = left_frame.column(node.names[i]).dtype()
        var rtype = built.column(node.right_keys[i]).dtype()
        if (
            ltype != rtype
            or ltype.is_categorical()
            or ltype.is_nested()
            or ltype.is_struct()
        ):
            return None
    var exprs = List[Expr]()
    for name in left_frame.columns():
        exprs.append(col(name))
    for name in right_names:
        if name not in node.right_keys:
            exprs.append(col(name))
    var plan = self.copy()
    plan._frames.append(built.copy())
    plan._schemas.append(Optional[CsvSchema]())
    plan._nodes.append(_frame_scan_node(slot=len(plan._frames) - 1))
    var probe = len(plan._nodes) - 1
    plan._frames.append(left_frame^)
    plan._schemas.append(Optional[CsvSchema]())
    plan._nodes.append(_frame_scan_node(slot=len(plan._frames) - 1))
    var build = len(plan._nodes) - 1
    var join = node.copy()
    join.left = probe
    join.right = build
    join.names = node.right_keys.copy()
    join.right_keys = node.names.copy()
    join.how = JOIN_RIGHT
    join.text = "right"
    join.coalesce = False
    plan._nodes.append(join^)
    plan._nodes[at] = _select_node(left=len(plan._nodes) - 1, exprs=exprs)
    trace_path("lazy.swap_left_join_sides")
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


def plan_push_join_key_sets(
    mut self: QueryContext, streaming: Bool, batch_size: Int
) raises:
    """An inner or semi join where one input is a filtered in-memory
    table or a join chain over tables at most a quarter the size of
    the other side's largest runs that input first, and each key's
    values become a filter on the other side's key when a sample of the table
    supplying it says the filter keeps at most one row in eight;
    predicate pushdown then moves it toward that table. Rows whose key no left
    row has join nothing, so the result is unchanged. Up to 50
    distinct keys filter as a list (an equality per value per row);
    larger sets use their min-max range. The left input is replaced by its
    result, so it does not run twice. Only for a right side with a
    join of its own, which the filter then reaches first. TPC-DS q37
    and q82 join four items to an 11.7M-row inventory that is joined
    to dates first; q72 joins 1999's sales to inventory by week, and
    the weeks' range narrows the dates the inventory joins; q59 joins
    a year's weeks (the small right side) to weekly sales, whose week
    filter passes the group_by down to the date join.
    """
    var root = len(self._nodes) - 1
    var changed = True
    while changed:
        changed = False
        var parents = plan_live_parents(self, root)
        for j in range(len(self._nodes)):
            if parents[j] == -2:
                continue
            ref node = self._nodes[j]
            if node.kind != JOIN or not (
                node.how == JOIN_INNER or node.how == JOIN_SEMI
            ):
                continue
            if len(node.names) != len(node.right_keys):
                continue
            if plan_key_set_side(self, j, True, streaming, batch_size):
                changed = True
            elif plan_key_set_side(self, j, False, streaming, batch_size):
                changed = True
            if changed:
                root = len(self._nodes) - 1
                break


def plan_key_set_side(
    mut self: QueryContext,
    j: Int,
    small_left: Bool,
    streaming: Bool,
    batch_size: Int,
) raises -> Bool:
    """One direction of `_push_join_key_sets` for the join at `j`: the
    small side (left when `small_left`) runs first and its keys filter
    the other side. True when the plan changed."""
    var small = self._nodes[j].left if small_left else self._nodes[j].right
    var large = self._nodes[j].right if small_left else self._nodes[j].left
    # A filtered table, or a join chain: inputs that keep few rows of
    # what they read. A bare table keeps all of them.
    if not (plan_filtered_table(self, small) or plan_has_join(self, small)):
        return False
    # Only a large side that does more than scan and filter before this
    # join (a join of its own) gains: against a bare table the join
    # already builds on its smaller side, and a filter there only
    # copies rows (TPC-DS q43, 2x).
    if not plan_has_join(self, large):
        return False
    # A small right side already filters the left one inside the
    # stream, where the join order puts the selective joins first;
    # its keys pay only when a group_by on the left would otherwise
    # aggregate rows the join then drops (TPC-DS q59's weekly sales).
    # Pushed into plain join chains they cost PDS-H q8 5x and q21 2x.
    if not small_left and not plan_has_kind(self, large, AGG):
        return False
    var large_rows = plan_largest_scan(self, large)
    # The tables under the small side, unfiltered, must be at most a
    # quarter of the other side's largest: then running it first and
    # keeping its result costs at most a quarter of reading that table
    # (TPC-DS q72's sales chain reads 1.9M demographics against 11.7M
    # inventory rows).
    var small_bound = plan_height_bound(self, small)
    if small_bound < 0:
        small_bound = plan_largest_scan(self, small)
    if large_rows <= 0 or small_bound <= 0:
        return False
    # Against a left side the join probes anyway, a right side is run
    # early only when it is tiny next to it (32x: TPC-DS q59's weeks
    # against store sales are 39x smaller); PDS-H q21's 1.5M orders
    # against 6M lineitem rows ran twice and filtered nothing.
    if (4 if small_left else 32) * small_bound > large_rows:
        return False
    var small_keys = (
        self._nodes[j]
        .names.copy() if small_left else self._nodes[j]
        .right_keys.copy()
    )
    var large_keys = (
        self._nodes[j]
        .right_keys.copy() if small_left else self._nodes[j]
        .names.copy()
    )
    # Each key's table on the large side and its name there (through
    # renames and group_by keys); computed keys get none.
    var providers = List[Tuple[Int, String]]()
    var any = False
    for k in range(len(large_keys)):
        var found = plan_provider_scan(self, large, large_keys[k])
        any = any or found[0] >= 0
        providers.append(found)
    if not any:
        return False
    # Read before the plan changes: the join's inequality conditions,
    # as (large side column, comparison, small side column).
    var comparisons = plan_side_comparisons(self, j, small_left)
    var result = self._execute(small, False, streaming, batch_size)
    if not small_left and not plan_any_selective(
        self, result, small_keys, providers
    ):
        # Nothing to pass: keep a right input as written, so the join
        # order still ranks it by its filters (a table already run looks
        # unfiltered, and PDS-H q21 then ran its selective supplier join
        # last). A left input is the stream's base, which nothing ranks:
        # its result is kept below, so it does not run twice.
        return False
    if small_left:
        trace_path("lazy.key_set_left")
    else:
        trace_path("lazy.key_set_right")
    self._frames.append(result.copy())
    self._schemas.append(Optional[CsvSchema]())
    self._nodes.append(_frame_scan_node(slot=len(self._frames) - 1))
    if small_left:
        self._nodes[j].left = len(self._nodes) - 1
    else:
        self._nodes[j].right = len(self._nodes) - 1
    var marker = len(self._frames) - 1
    var added = False
    for k in range(len(providers)):
        var scan = providers[k][0]
        if scan < 0:
            continue
        var base = providers[k][1]
        var dtype = (
            self._frames[self._nodes[scan].source_slot()].column(base).dtype()
        )
        var keys = result.column(small_keys[k])
        var sampled = _key_set_filter(keys, base, dtype)
        if not sampled or not plan_selective_on(self, scan, sampled.value()):
            continue
        var pushed = _key_set_filter(keys, large_keys[k], dtype)
        var target = self._nodes[j].right if small_left else self._nodes[j].left
        self._nodes.append(_filter_node(left=target, exprs=[pushed.take()]))
        if small_left:
            self._nodes[j].right = len(self._nodes) - 1
        else:
            self._nodes[j].left = len(self._nodes) - 1
        added = True
    # An inequality condition passes the small side's greatest value
    # (for < and <=) or least (for > and >=) to the large side as a
    # bound, as DuckDB's join filters do for these comparisons
    # (`JoinFilterPushdownInfo::FinalizeFilters`). TPC-DS q72's stock
    # on hand below the quantity ordered keeps the inventory rows
    # under the largest quantity.
    for comparison in comparisons:
        var name = comparison[0]
        var found = plan_provider_scan(self, large, name)
        if found[0] < 0:
            continue
        ref source = self._frames[self._nodes[found[0]].source_slot()]
        var dtype = source.column(found[1]).dtype()
        var other = result.column(comparison[2])
        if (
            not (dtype == DataType.INT64 or dtype == DataType.INT32)
            or other.dtype() != dtype
        ):
            continue
        var bounds = _int_bounds(other)
        if not bounds:
            continue
        var op = comparison[1]
        var bound = (
            bounds.value()[1] if op == LT or op == LE else bounds.value()[0]
        )
        var sampled = _bound_filter(found[1], op, bound, dtype)
        if not plan_selective_on(self, found[0], sampled):
            continue
        var target = self._nodes[j].right if small_left else self._nodes[j].left
        self._nodes.append(
            _filter_node(
                left=target, exprs=[_bound_filter(name, op, bound, dtype)]
            )
        )
        if small_left:
            self._nodes[j].right = len(self._nodes) - 1
        else:
            self._nodes[j].left = len(self._nodes) - 1
        added = True
    plan_reorder(self)
    if not added:
        return True
    # Not split again: splitting a merged filter and merging it back
    # reverses its parts, and the written order is the order they are
    # evaluated in (ClickBench q21 then ran its string search first).
    plan_push_predicates(self)
    plan_merge_filters(self)
    if small_left:
        # The right side is built next whatever happens: run it now,
        # filtered, so the join sees both real heights and builds on
        # the smaller side (TPC-DS q72 built its two-key index on 2.4M
        # stock rows for 8,689 sales).
        for at in range(len(self._nodes)):
            ref joined = self._nodes[at]
            if (
                joined.kind == JOIN
                and joined.left >= 0
                and self._nodes[joined.left].kind == SCAN_FRAME
                and self._nodes[joined.left].source_slot() == marker
            ):
                var right = self._execute(
                    joined.right, False, streaming, batch_size
                )
                self._frames.append(right^)
                self._schemas.append(Optional[CsvSchema]())
                self._nodes.append(_frame_scan_node(slot=len(self._frames) - 1))
                self._nodes[at].right = len(self._nodes) - 1
                break
        plan_reorder(self)
    return True


def plan_side_comparisons(
    self: QueryContext, j: Int, small_left: Bool
) raises -> List[Tuple[String, Int, String]]:
    """The inequalities between a column of each input of the inner
    join at `j` in filters directly above it, represented as join
    conditions: each as (large side column, its comparison with
    the small side column, small side column), the large side being
    the right input when `small_left`. Only columns that one input
    alone names."""
    var found = List[Tuple[String, Int, String]]()
    if self._nodes[j].how != JOIN_INNER:
        return found^
    var parents = plan_live_parents(self, len(self._nodes) - 1)
    var pairs = List[Tuple[String, Int, String]]()
    var at = parents[j]
    while at >= 0 and self._nodes[at].kind == FILTER:
        for part in conjuncts(self._nodes[at].exprs[0]):
            if len(part._nodes) != 3:
                continue
            ref root = part._nodes[2]
            if not (
                root.op == LT or root.op == LE or root.op == GT or root.op == GE
            ):
                continue
            if root.left < 0 or root.right < 0:
                continue
            ref a = part._nodes[root.left]
            ref b = part._nodes[root.right]
            if a.op == COL and b.op == COL:
                pairs.append((a.text, root.op, b.text))
        at = parents[at]
    if len(pairs) == 0:
        return found^
    var large_cols = plan_columns_of(
        self, self._nodes[j].right if small_left else self._nodes[j].left
    )
    var small_cols = plan_columns_of(
        self, self._nodes[j].left if small_left else self._nodes[j].right
    )
    for pair in pairs:
        var a = pair[0]
        var op = pair[1]
        var b = pair[2]
        var a_large = a in large_cols and a not in small_cols
        var b_large = b in large_cols and b not in small_cols
        var a_small = a in small_cols and a not in large_cols
        var b_small = b in small_cols and b not in large_cols
        if a_large and b_small:
            found.append((a, op, b))
        elif b_large and a_small:
            # `a op b` is `b flipped a`.
            var flipped = GT
            if op == LE:
                flipped = GE
            elif op == GT:
                flipped = LT
            elif op == GE:
                flipped = LE
            found.append((b, flipped, a))
    return found^


def plan_selective_on(
    mut self: QueryContext, scan: Int, keys: Expr
) raises -> Bool:
    """Whether `keys` keeps at most one row in eight of the table at
    node `scan`, estimated on evenly spaced runs of it
    (`_estimated_rows`). A key set spanning most of the right side's
    keys filters nothing and only moves the join off its better
    build side."""
    var height = self._frames[self._nodes[scan].source_slot()].height()
    self._nodes.append(_filter_node(left=scan, exprs=[keys.copy()]))
    var estimated = plan_estimated_rows(self, len(self._nodes) - 1)
    _ = self._nodes.pop()
    return estimated >= 0 and 8 * estimated <= height


def plan_any_selective(
    mut self: QueryContext,
    result: DataFrame,
    keys: List[String],
    providers: List[Tuple[Int, String]],
) raises -> Bool:
    """Whether any key of `result` would filter its provider table to at
    most one row in eight (`_selective_on`)."""
    for k in range(len(providers)):
        var scan = providers[k][0]
        if scan < 0:
            continue
        var base = providers[k][1]
        var dtype = (
            self._frames[self._nodes[scan].source_slot()].column(base).dtype()
        )
        var sampled = _key_set_filter(result.column(keys[k]), base, dtype)
        if sampled and plan_selective_on(self, scan, sampled.value()):
            return True
    return False


def plan_cost_order_joins(
    mut self: QueryContext, root: Int, streaming: Bool, batch_size: Int
) raises -> Bool:
    """Order each join graph of three or more relations by estimated
    cost (`join_order.mojo`):
    inner joins flattened across nested inputs into relations and key
    pairs, each relation's rows counted after its own filters (run
    when they cannot be counted otherwise, as `_order_chain` measures
    them), and the cheapest tree built. Choose each hash-build side
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
        # The graph's top: no graph step above it.
        var up = parents[top]
        if up >= 0 and plan_graph_step(self, up):
            continue
        if not plan_order_free_above(self, top, parents):
            continue
        var relations = List[Int]()
        var pairs = List[Tuple[String, String]]()
        var filters = List[Int]()
        plan_collect_graph(self, top, relations, pairs, filters)
        var n = len(relations)
        if n < 3 or n > 62:
            continue
        var provider = Dict[String, Int]()
        var unique = True
        var columns = List[List[String]]()
        for r in range(n):
            var names = plan_columns_of(self, relations[r])
            for c in names:
                if c in provider:
                    unique = False
                provider[c] = r
            columns.append(names^)
        if not unique:
            continue
        # Edges: the key pairs between each two relations.
        var edges = List[JoinEdge]()
        var known = True
        for pair in pairs:
            var a = provider.get(pair[0], -1)
            var b = provider.get(pair[1], -1)
            if a < 0 or b < 0 or a == b:
                known = False
                break
            var found = False
            for k in range(len(edges)):
                if edges[k].a == a and edges[k].b == b:
                    edges[k].a_keys.append(pair[0])
                    edges[k].b_keys.append(pair[1])
                    found = True
                elif edges[k].a == b and edges[k].b == a:
                    edges[k].a_keys.append(pair[1])
                    edges[k].b_keys.append(pair[0])
                    found = True
            if not found:
                edges.append(JoinEdge(a, b, [pair[0]], [pair[1]]))
        if not known:
            continue
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
            estimates.append(plan_relation_estimate(self, relations[r]))
            aggregated.append(plan_aggregated(self, relations[r]))
            if estimates[r] < 0:
                runs.append(r)
        # Relations that must run to be counted are independent
        # pipelines: run them at once, as Polars and DuckDB run
        # independent pipelines (`_prefetch_breakers`), not one after
        # another on a crew their serial phases leave partly idle.
        if len(runs) > 0:
            var jobs = List[_SubplanJob](capacity=len(runs))
            for r in runs:
                jobs.append(
                    _SubplanJob(
                        self.copy(), relations[r], streaming, batch_size
                    )
                )
            if len(jobs) == 1:
                jobs[0].run()
            else:
                run_jobs(jobs)
            for k in range(len(runs)):
                var r = runs[k]
                estimates[r] = jobs[k].result.height()
                # An aggregate's result is a new table: it keeps
                # distinct counts of its own columns, as DuckDB's
                # estimator gives an aggregate relation's columns
                # (`ExtractAggregationStats`).
                if aggregated[r]:
                    self._frames.append(
                        jobs[k].result.copy()._with_statistics()
                    )
                else:
                    self._frames.append(jobs[k].result.copy())
                self._schemas.append(Optional[CsvSchema]())
                self._nodes[relations[r]] = _frame_scan_node(
                    slot=len(self._frames) - 1
                )
        for r in range(n):
            var table = tables[r]
            var estimated = estimates[r]
            rows.append(Float64(estimated))
            domains.append(Float64(table if table > 0 else estimated))
            # Kept for the build side decisions (`plan_cardinality`), so
            # they read it rather than sample the relation again.
            if self._nodes[relations[r]].kind != SCAN_FRAME:
                self._nodes[relations[r]].estimated_rows = estimated
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
        # domain as the largest count among its columns.
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
            var table = plan_spine_frame(self, relations[pair[0]])
            if not table:
                continue
            var count = table.value()._distinct_count(pair[1])
            if count > 0:
                distinct[key] = Float64(count)
        var chosen = order_joins(rows, domains, edges, distinct, compares)
        if not chosen:
            continue
        var tree = chosen.take()
        # Build the chosen tree. `node_of` is each tree entry's plan
        # node; `renamed` maps each key a join dropped to the key it
        # matched; `joined` marks entries holding a join.
        var entries = len(tree.mask)
        var node_of = List[Int](length=entries, fill=-1)
        var width = List[Int](length=entries, fill=0)
        var joined = List[Bool](length=entries, fill=False)
        for r in range(n):
            node_of[r] = relations[r]
            width[r] = len(columns[r])
        var renamed = Dict[String, String]()
        var placed = List[Bool](length=len(filters), fill=False)
        var added = List[PlanNode]()
        var base = len(self._nodes)
        for e in range(n, entries):
            var left = tree.left[e]
            var right = tree.right[e]
            # DuckDB's build side: the side of fewer rows times
            # columns builds, a relation 15% dearer against a join.
            var left_cost = tree.rows[left] * Float64(width[left])
            var right_cost = tree.rows[right] * Float64(width[right])
            if not joined[right] and joined[left]:
                right_cost *= 1.15
            if not joined[left] and joined[right]:
                left_cost *= 1.15
            # A lone relation joined on one key whose values are a
            # progression builds no hash table: the other side looks
            # its rows up by position (`join.progression`), so it costs
            # nothing to build on.
            if plan_indexed_relation(
                self, left, right, joined, node_of, edges, tree.mask
            ):
                left_cost = 0.0
            if plan_indexed_relation(
                self, right, left, joined, node_of, edges, tree.mask
            ):
                right_cost = 0.0
            if right_cost > left_cost:
                var t = left
                left = right
                right = t
            var lmask = tree.mask[left]
            var rmask = tree.mask[right]
            var left_keys = List[String]()
            var right_keys = List[String]()
            for edge in edges:
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
            # The set's estimate, kept on the join and on the filters
            # placed over it (their comparisons are edges the estimate
            # counted), as DuckDB keeps `estimated_cardinality` on each
            # operator for its build side optimizer.
            var estimated = Int(min(tree.rows[e], Float64(Int.MAX // 2)))
            join.estimated_rows = estimated
            added.append(join^)
            var current = base + len(added) - 1
            # Filters whose relations this join completes, while their
            # names are all still held.
            for f in range(len(filters)):
                if placed[f] or filter_masks[f] & ~tree.mask[e] != 0:
                    continue
                var reads = (
                    _references(self._nodes[filters[f]].exprs[0]).value().copy()
                )
                var held = True
                for name in reads:
                    held = held and name not in renamed
                if not held:
                    continue
                var filter = self._nodes[filters[f]].copy()
                filter.left = current
                filter.estimated_rows = estimated
                added.append(filter^)
                current = base + len(added) - 1
                placed[f] = True
            node_of[e] = current
            width[e] = width[left] + width[right] - len(left_keys)
            joined[e] = True
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
            last.estimated_rows = -1
        for node in added:
            self._nodes.append(node.copy())
        self._nodes[top] = last^
        trace_path("lazy.cost_join_order")
        changed = True
    return changed


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
            ref edge = self._nodes[joins[table - 1]]
            # Built now, keyed by the join's right keys.
            if not plan_progression_key(self, tables[table], edge.right_keys):
                before += rows[table]
            # Built after re-rooting: the table on the other end,
            # keyed by the join's left keys.
            var other = linked[table - 1]
            if not plan_progression_key(self, tables[other], edge.names):
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
