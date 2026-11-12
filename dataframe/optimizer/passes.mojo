"""Logical optimizer passes; runtime evaluation enters through QueryContext ports."""
from dataframe.planning.node_builders import (
    _aggregate_node,
    _filter_node,
    _frame_scan_node,
    _select_node,
    _with_columns_node,
)
from dataframe.expr import (
    AND,
    COL,
    Expr,
    LIT_BOOL,
    MAX,
    Node,
    SELECTOR,
    col,
    subtree,
)
from dataframe.frame import (
    DataFrame,
)
from dataframe.series import (
    Series,
)
from dataframe.trace import (
    trace_path,
)
from dataframe.aggregation.expressions import (
    _row_local,
)
from dataframe.io.csv import (
    CsvSchema,
)
from dataframe.join_type import (
    JOIN_ANTI,
    JOIN_INNER,
    JOIN_LEFT,
    JOIN_RIGHT,
    JOIN_SEMI,
)
from dataframe.optimizer.expressions import (
    _cannot_raise,
    _emit_shared,
    _fold_casts,
    _fold_constants,
    _shared_text,
)
from dataframe.planning.context import (
    QueryContext,
)
from dataframe.planning.graph import (
    plan_class_refs,
    plan_columns_of,
    plan_cover,
    plan_forget_schema,
    plan_forget_schemas_above,
    plan_parents,
    plan_materialized,
    plan_projection_visits,
    plan_remember_schemas,
    plan_reorder,
    plan_share_copy_schemas,
    plan_subplan_classes,
)
from dataframe.planning.model import (
    PlanNode,
    AGG,
    ASOF,
    CONCAT,
    DROP,
    EXPLODE,
    FILTER,
    GATHER,
    HCONCAT,
    INSERT_COLUMN,
    JOIN,
    PIVOT,
    SCAN_CSV,
    SCAN_FRAME,
    SCAN_IPC,
    SCAN_NDJSON,
    SELECT,
    SLICE,
    SORT,
    UNIQUE,
    UNNEST,
    UNPIVOT,
    WITH_COLUMNS,
)
from dataframe.planning.properties import (
    _boolean_parts,
    _covers,
    _covers_exclusive,
    _expr_signature,
    _field_list,
    _implied_filters,
    _is_scan,
    _is_time_window,
    _merge_copy_needs,
    _pinned,
    _plain_or_constant,
    _produced_at,
    _reads_input,
    _row_order_step,
    _through_projection,
    _time_window_reads,
)
from dataframe.planning.expression_inputs import (
    _references,
)
from std.collections import (
    Dict,
    Optional,
)
from std.memory import (
    ArcPointer,
)


def plan_optimized(self: QueryContext) raises -> QueryContext:
    var plan = self.copy()
    plan_fold_casts(plan)
    plan_unique_as_aggregate(plan)
    plan_lower_right_joins(plan)
    plan_filter_shared_subplans(plan)
    # Memoize derived schemas across predicate and projection pushdown. These
    # passes retain output shape; invalidate memo entries for nodes that are
    # rewritten or renumbered.
    plan._schema_memo = Optional(ArcPointer(Dict[Int, DataFrame]()))
    try:
        plan_push_predicates(plan)
        plan_push_slices(plan)
        plan_fuse_top_k(plan)
        plan_eliminate_common_subexpressions(plan)
        plan_push_projections(plan)
    except e:
        plan._schema_memo = None
        raise e^
    plan._schema_memo = None
    return plan^


def plan_lower_right_joins(mut self: QueryContext) raises:
    """Normalize outer-join column naming so the join can stream with an
    explicit unmatched-build scan. Preserve the left/right column order and
    decline conflicting, repeated or missing key names so existing
    validation still reports them."""
    var changed = False
    for i in range(len(self._nodes)):
        if (
            self._nodes[i].kind != JOIN
            or self._nodes[i].how != JOIN_RIGHT
            or not self._nodes[i].coalesce
            or len(self._nodes[i].names) == 0
            or len(self._nodes[i].names) != len(self._nodes[i].right_keys)
        ):
            continue
        var node = self._nodes[i].copy()
        var left_cols = plan_columns_of(self, node.left)
        var right_cols = plan_columns_of(self, node.right)
        var suffix = node.join_suffix()
        var valid = True
        for k in range(len(node.names)):
            if (
                node.names[k] not in left_cols
                or node.right_keys[k] not in right_cols
            ):
                valid = False
            for j in range(k):
                if node.names[j] == node.names[k]:
                    valid = False
        if not valid:
            continue
        # Names as the kept-keys join gives them (`kept`) and as the
        # coalesced one does (`exprs`); both must be free of collisions.
        var kept = Dict[String, Bool]()
        var named = Dict[String, Bool]()
        var exprs = List[Expr]()
        for c in left_cols:
            kept[c] = True
            if c not in node.names:
                named[c] = True
                exprs.append(col(c))
        var others = named.copy()
        for c in right_cols:
            var held = c + suffix if c in left_cols else c
            var shown = c + suffix if c in others else c
            if held in kept or shown in named:
                valid = False
                break
            kept[held] = True
            named[shown] = True
            exprs.append(col(held).alias(shown) if held != shown else col(held))
        if not valid:
            continue
        node.coalesce = False
        self._nodes.append(node^)
        self._nodes[i] = _select_node(left=len(self._nodes) - 1, exprs=exprs)
        changed = True
    if changed:
        trace_path("lazy.lower_right_join")
        plan_reorder(self)


def plan_eliminate_common_subexpressions(mut self: QueryContext) raises:
    """Compute a repeated row-local subexpression once. An aggregation
    or projection that reads it multiple times uses a temporary column
    produced by a `with_columns` node below it. For example, TPC-DS
    q99's five delay buckets compare the same date difference nine
    times."""
    var count = len(self._nodes)
    var changed = False
    for i in range(count):
        var kind = self._nodes[i].kind
        if not (kind == AGG or kind == SELECT):
            continue
        if self._nodes[i].left < 0:
            continue
        ref exprs = self._nodes[i].exprs
        # How often each eligible subexpression appears, by its text.
        var seen = Dict[String, Int]()
        for e in exprs:
            for r in range(len(e._nodes)):
                var text = _shared_text(e, r)
                if text:
                    seen[text.value()] = seen.get(text.value(), 0) + 1
        var any = False
        for entry in seen.items():
            any = any or entry.value > 1
        if not any:
            continue
        # Hidden column names none of the node's inputs is read as: an
        # input column of the same name that it does not read is
        # dropped by the node anyway, so no schema is needed.
        var taken = List[String]()
        var readable = True
        for e in exprs:
            var reads = _references(e)
            if not reads:
                readable = False
                break
            for name in reads.value():
                taken.append(name)
        if not readable:
            continue
        for name in self._nodes[i].names:
            taken.append(name)
        var shared = List[Expr]()
        var names = Dict[String, String]()
        var rewritten = List[Expr](capacity=len(exprs))
        for e in exprs:
            var out = List[Node]()
            _ = _emit_shared(
                e, len(e._nodes) - 1, seen, names, shared, taken, out
            )
            rewritten.append(Expr(out^, e._name))
        if len(shared) == 0:
            continue
        self._nodes[i].exprs = rewritten^
        self._nodes.append(
            _with_columns_node(left=self._nodes[i].left, exprs=shared)
        )
        self._nodes[i].left = len(self._nodes) - 1
        plan_forget_schema(self, i)
        changed = True
    if changed:
        plan_reorder(self)


def plan_fold_casts(mut self: QueryContext) raises:
    """Evaluate a cast expression that reads no column once and
    replace it with a typed literal, rather than repeating the work for
    every batch or morsel. For example, a decimal written as a cast
    string literal is parsed only once."""
    for i in range(len(self._nodes)):
        var kind = self._nodes[i].kind
        if not (
            kind == FILTER
            or kind == SELECT
            or kind == WITH_COLUMNS
            or kind == AGG
        ):
            continue
        for k in range(len(self._nodes[i].exprs)):
            var folded = _fold_casts(self._nodes[i].exprs[k])
            if folded:
                self._nodes[i].exprs[k] = folded.take()


def plan_unique_as_aggregate(mut self: QueryContext) raises:
    """A unique over every column keeps one row per set of equal rows
    (or the first in input order). Implement it as a group_by on all
    columns and run it through the parallel aggregate table rather
    than deduplicating collected rows. The table needs one aggregate,
    so add and then drop a row count."""
    var count = len(self._nodes)
    var changed = False
    var owned = plan_remember_schemas(self)
    for i in range(count):
        ref node = self._nodes[i]
        if node.kind != UNIQUE or node.left < 0:
            continue
        if node.text != "any" and node.text != "first":
            continue
        var columns = plan_columns_of(self, node.left)
        if len(columns) == 0:
            continue
        if len(node.names) > 0:
            # A subset naming every column, in any order.
            if len(node.names) != len(columns):
                continue
            var every = True
            for c in columns:
                if c not in node.names:
                    every = False
            if not every:
                continue
        var counted = "__unique_rows"
        while counted in columns:
            counted += "_"
        self._nodes.append(
            _aggregate_node(
                left=node.left,
                exprs=[col(columns[0]).len().alias(counted)],
                keys=columns,
                maintain_order=node.maintain_order,
            )
        )
        var shown = List[Expr]()
        for c in columns:
            shown.append(col(c))
        self._nodes[i] = _select_node(left=len(self._nodes) - 1, exprs=shown)
        changed = True
    if owned:
        self._schema_memo = None
    if changed:
        plan_reorder(self)


def plan_filter_shared_subplans(mut self: QueryContext):
    """For a shared subplan read only through row-local filters, push the union
    of reader predicates into the shared computation. Keep each reader's own
    filter above it. The retained result then contains every row any reader
    needs while discarding rows none can observe."""
    # Copies need two aggregates or joins alike in their keys, expression
    # count and subtree size at least; most plans have none, and
    # subplan classes are not worth finding for them (ClickBench q29's
    # ninety sums, TPC-DS q41's two small distincts).
    var subtree = List[Int](capacity=len(self._nodes))
    var shapes = Dict[String, Int]()
    var alike = False
    for i in range(len(self._nodes)):
        ref node = self._nodes[i]
        var size = 1
        if node.left >= 0:
            size += subtree[node.left]
        if node.right >= 0:
            size += subtree[node.right]
        subtree.append(size)
        if node.kind != AGG and node.kind != UNIQUE and node.kind != JOIN:
            continue
        var shape = (
            String(node.kind)
            + "|"
            + _field_list(node.names)
            + "|"
            + String(len(node.exprs))
            + "|"
            + String(size)
        )
        if shape in shapes:
            alike = True
            break
        shapes[shape] = i
    if not alike:
        return
    var computed = plan_subplan_classes(self)
    ref classes = computed[0]
    ref sizes = computed[1]
    var parents = plan_parents(self)
    var covered = List[Bool](length=len(self._nodes), fill=False)
    var count = len(self._nodes)
    while True:
        # The largest materialized subtree written twice or more, under
        # its readers' filters (two readers may share one).
        var best = -1
        for i in range(count):
            if covered[i] or self._nodes[i].kind == FILTER:
                continue
            if best >= 0 and sizes[i] <= sizes[best]:
                continue
            var refs = plan_class_refs(classes, covered, i)
            if refs >= 2 and plan_materialized(self, i, refs):
                best = i
        if best < 0:
            break
        var copies = List[Int]()
        for j in range(count):
            if not covered[j] and classes[j] == classes[best]:
                copies.append(j)
                plan_cover(self, j, covered)
        # Each copy's readers: the run of row-local filters above it.
        var readers = List[List[Int]]()
        var kept = List[Expr]()
        var alike = True
        for copy in copies:
            var run = List[Int]()
            var above = parents[copy]
            while (
                above >= 0
                and self._nodes[above].kind == FILTER
                and _row_local(self._nodes[above].exprs)
                and _cannot_raise(self._nodes[above].exprs)
                and not _pinned(self._nodes[above])
            ):
                run.append(above)
                above = parents[above]
            if len(run) == 0:
                break
            var keeps = self._nodes[run[0]].exprs[0].copy()
            for k in range(1, len(run)):
                keeps = keeps & self._nodes[run[k]].exprs[0]
            if len(kept) > 0 and _expr_signature(keeps) != _expr_signature(
                kept[0]
            ):
                alike = False
            kept.append(keeps^)
            readers.append(run^)
        # A copy read whole needs every row; readers alike move into
        # their copies alike anyway.
        if len(readers) < len(copies) or alike:
            continue
        var any = kept[0].copy()
        for k in range(1, len(kept)):
            any = any | kept[k]
        for c in range(len(copies)):
            ref run = readers[c]
            for r in run:
                self._nodes[r].pin_filter()
            self._nodes.append(_filter_node(left=copies[c], exprs=[any.copy()]))
            self._nodes[run[0]].left = len(self._nodes) - 1
    if len(self._nodes) > count:
        plan_reorder(self)


def plan_push_predicates(mut self: QueryContext) raises:
    """DuckDB's FilterPushdown (optimizer/filter_pushdown.cpp and
    optimizer/pushdown/) in one top-down pass: each filter's conjuncts
    join the set carried down the plan, each operator passes on the
    ones that cannot change its result and the rest land above it as
    one filter. The plan is renumbered once at the end."""
    var owned = plan_remember_schemas(self)
    try:
        var root = plan_pushdown(self, len(self._nodes) - 1, List[Expr]())
        plan_reorder(self, root)
    except e:
        if owned:
            self._schema_memo = None
        raise e^
    if owned:
        self._schema_memo = None


def plan_movable(node: PlanNode) -> Bool:
    """Whether a filter's predicate may join the pushed set: row-local,
    unable to raise and not pinned above a shared subplan. Any other
    filter stays where it is and the set stops above it."""
    return (
        node.kind == FILTER
        and not _pinned(node)
        and _row_local(node.exprs)
        and _cannot_raise(node.exprs)
    )


def _literal(predicate: Expr) -> Int:
    """1 for a literal true, 0 for a literal false, else -1."""
    ref nodes = predicate._nodes
    if len(nodes) != 1 or nodes[0].op != LIT_BOOL:
        return -1
    return 1 if nodes[0].integer != 0 else 0


def plan_add_filter(mut filters: List[Expr], predicate: Expr) -> Bool:
    """FilterPushdown::AddFilter: split `predicate` into its conjuncts,
    with the single-input filters an OR of ANDs implies
    (`_implied_filters`), and add them to `filters`. A true conjunct
    keeps every row and goes. False when a conjunct is false: no row
    passes (FilterResult::UNSATISFIABLE)."""
    var unsatisfiable = False
    for part in _boolean_parts(predicate, AND):
        var value = _literal(part)
        if value == 1:
            continue
        if value == 0:
            unsatisfiable = True
        filters.append(part.copy())
        for implied in _implied_filters(part):
            filters.append(implied.copy())
    return not unsatisfiable


def plan_add_conjunct(mut filters: List[Expr], predicate: Expr) -> Bool:
    """Add one conjunct already split by `plan_add_filter` (rewritten
    through a projection, or carried to the other join input): a true
    one goes, a false one makes the set unsatisfiable."""
    var value = _literal(predicate)
    if value != 1:
        filters.append(predicate.copy())
    return value != 0


def plan_empty_result(mut self: QueryContext, index: Int) raises -> Bool:
    """LogicalEmptyResult: the subtree at `index` becomes an empty frame
    with its columns, when skipping it cannot suppress an evaluation
    error (`plan_can_discard`)."""
    if not plan_can_discard(self, index):
        return False
    var empty = self._execute(index, True)
    self._frames.append(empty^)
    self._schemas.append(Optional[CsvSchema]())
    self._nodes[index] = _frame_scan_node(slot=len(self._frames) - 1)
    return True


def plan_final_filters(
    mut self: QueryContext, index: Int, filters: List[Expr]
) -> Int:
    """FilterPushdown::PushFinalFilters: the filters that stop at
    `index`, as one filter above it."""
    if len(filters) == 0:
        return index
    var predicate = filters[0].copy()
    for k in range(1, len(filters)):
        predicate = predicate & filters[k]
    self._nodes.append(_filter_node(left=index, exprs=[predicate^]))
    return len(self._nodes) - 1


def plan_finish_pushdown(
    mut self: QueryContext, index: Int, filters: List[Expr]
) raises -> Int:
    """FilterPushdown::FinishPushdown: each input starts a pushdown of
    its own and the filters stop above `index`."""
    var left = self._nodes[index].left
    var right = self._nodes[index].right
    if left >= 0:
        var new_left = plan_pushdown(self, left, List[Expr]())
        self._nodes[index].left = new_left
    if right >= 0:
        var new_right = plan_pushdown(self, right, List[Expr]())
        self._nodes[index].right = new_right
    return plan_final_filters(self, index, filters)


def plan_pushdown(
    mut self: QueryContext, index: Int, var filters: List[Expr]
) raises -> Int:
    """FilterPushdown::Rewrite: push `filters`, read by the parent of
    `index`, into the subtree at `index`; returns the subtree's new
    root. Every filter in `filters` is row-local and cannot raise. One
    reading columns through a selector stays above `index`: below, the
    selector could match other columns."""
    var fixed = List[Expr]()
    for k in range(len(filters) - 1, -1, -1):
        if not _references(filters[k]):
            fixed.append(filters.pop(k))
    if len(fixed) > 0:
        var root = plan_pushdown(self, index, filters^)
        return plan_final_filters(self, root, fixed)
    var kind = self._nodes[index].kind
    if kind == FILTER:
        return plan_pushdown_filter(self, index, filters^)
    if len(filters) == 0:
        # Nothing to place: each operator only rewrites its inputs.
        return plan_finish_pushdown(self, index, filters)
    if kind == SELECT or kind == WITH_COLUMNS:
        return plan_pushdown_projection(self, index, filters^)
    if kind == DROP or kind == SORT:
        # A drop removes no row and a filter above it reads none of
        # the dropped columns; a stable sort keeps the surviving rows'
        # relative order (LOGICAL_ORDER_BY pushes straight through).
        var new_left = plan_pushdown(self, self._nodes[index].left, filters^)
        self._nodes[index].left = new_left
        return index
    if kind == CONCAT:
        return plan_pushdown_set_operation(self, index, filters^)
    if kind == AGG or kind == PIVOT:
        return plan_pushdown_aggregate(self, index, filters^)
    if kind == JOIN:
        var how = self._nodes[index].how
        if how == JOIN_INNER or how == JOIN_LEFT:
            return plan_pushdown_join(self, index, filters^)
        if how == JOIN_SEMI or how == JOIN_ANTI:
            return plan_pushdown_semi_anti_join(self, index, filters^)
    return plan_finish_pushdown(self, index, filters)


def plan_pushdown_filter(
    mut self: QueryContext, index: Int, var filters: List[Expr]
) raises -> Int:
    """FilterPushdown::PushdownFilter: a movable filter's conjuncts join
    the set and the filter goes. A filter that cannot move (pinned,
    not row-local, or able to raise on rows the set would remove) stays
    a boundary: the set stops above it. A row-local one there takes the
    set into its own predicate, after its own, so a scan below sees
    them together."""
    var left = self._nodes[index].left
    if plan_movable(self._nodes[index]):
        if not plan_add_filter(filters, self._nodes[index].exprs[0]):
            if plan_empty_result(self, left):
                return plan_final_filters(self, left, List[Expr]())
        return plan_pushdown(self, left, filters^)
    left = plan_pushdown(self, left, List[Expr]())
    self._nodes[index].left = left
    var pinned = _pinned(self._nodes[index])
    var local = _row_local(self._nodes[index].exprs)
    # A reader's pinned filters merge with each other, never with the
    # shared subplan's own (`plan_filter_shared_subplans`).
    if (
        pinned
        and local
        and _cannot_raise(self._nodes[index].exprs)
        and _pinned(self._nodes[left])
        and _row_local(self._nodes[left].exprs)
    ):
        var merged = self._nodes[left].exprs[0] & self._nodes[index].exprs[0]
        self._nodes[index].exprs = [merged^]
        self._nodes[index].left = self._nodes[left].left
        plan_forget_schema(self, index)
    if len(filters) > 0 and not pinned and local:
        var merged = self._nodes[index].exprs[0].copy()
        for f in filters:
            merged = merged & f
        self._nodes[index].exprs = [merged^]
        return index
    return plan_final_filters(self, index, filters)


def plan_pushdown_projection(
    mut self: QueryContext, index: Int, var filters: List[Expr]
) raises -> Int:
    """FilterPushdown::PushdownProjection: a filter reading only columns
    the projection passes through or sets to a constant is rewritten
    over its input and moves below it. One reading a computed column
    stays above (DuckDB's PRESERVE_COMPUTED_EXPRESSIONS, without the
    split projection). A window or aggregate over the input depends on
    which rows are there (TPC-DS q47 ranked months after `d_year = 1999`
    was pushed below the rank), and a select of literals alone sets
    the row count: nothing passes either."""
    var exprs = self._nodes[index].exprs.copy()
    var select = self._nodes[index].kind == SELECT
    if (
        not _cannot_raise(exprs)
        or not _row_local(exprs)
        or (select and not _reads_input(exprs))
    ):
        return plan_finish_pushdown(self, index, filters)
    var below = List[Expr]()
    var remain = List[Expr]()
    var unsatisfiable = False
    for f in filters:
        var reads = _references(f)
        var passes = True
        for name in reads.value():
            var at = _produced_at(exprs, name)
            if at < 0:
                if select:
                    passes = False
                continue
            if not _plain_or_constant(exprs[at]):
                passes = False
        if not passes:
            remain.append(f.copy())
            continue
        var rewritten = _fold_constants(_through_projection(f, exprs))
        if not plan_add_conjunct(below, rewritten):
            unsatisfiable = True
    if unsatisfiable and plan_empty_result(self, index):
        return plan_final_filters(self, index, remain)
    var new_left = plan_pushdown(self, self._nodes[index].left, below^)
    self._nodes[index].left = new_left
    return plan_final_filters(self, index, remain)


def plan_pushdown_set_operation(
    mut self: QueryContext, index: Int, var filters: List[Expr]
) raises -> Int:
    """FilterPushdown::PushdownSetOperation: both inputs of a vertical
    concat supply the same names and types, so every filter applies to
    each, as both engines push predicates into every branch of a UNION
    ALL."""
    var left = self._nodes[index].left
    var right = self._nodes[index].right
    var copies = filters.copy()
    var new_left = plan_pushdown(self, left, filters^)
    self._nodes[index].left = new_left
    var new_right = plan_pushdown(self, right, copies^)
    self._nodes[index].right = new_right
    return index


def plan_pushdown_aggregate(
    mut self: QueryContext, index: Int, var filters: List[Expr]
) raises -> Int:
    """FilterPushdown::PushdownAggregate: a filter reading only group
    keys removes whole groups, so it moves below the aggregate and the
    kept groups' values and first-occurrence order stay as they were.
    A pivot's index columns are its keys when it keeps one value per
    cell (min, max, first, last, count, len). Filters on aggregates,
    and every filter over an aggregate without keys, stay above."""
    var keys = List[String]()
    ref node = self._nodes[index]
    if node.kind == AGG:
        if _cannot_raise(node.exprs):
            keys = node.names.copy()
    elif node.text in ["min", "max", "first", "last", "count", "len"]:
        keys = List[String](node.pivot_columns()[: node.pivot_index_count()])
    var below = List[Expr]()
    var remain = List[Expr]()
    for f in filters:
        var reads = _references(f)
        if len(keys) > 0 and _covers(keys, reads.value()):
            below.append(f.copy())
        else:
            remain.append(f.copy())
    var new_left = plan_pushdown(self, self._nodes[index].left, below^)
    self._nodes[index].left = new_left
    return plan_final_filters(self, index, remain)


def plan_join_sides(
    self: QueryContext, index: Int
) raises -> Tuple[List[String], List[String]]:
    """The names a filter above the join can read from each input.
    Coalesced right keys are not in the join output, so no filter above
    can mean them."""
    ref node = self._nodes[index]
    var left_cols = plan_columns_of(self, node.left)
    var right_cols = List[String]()
    for c in plan_columns_of(self, node.right):
        if not (node.coalesce and c in node.right_keys):
            right_cols.append(c)
    return (left_cols^, right_cols^)


def plan_pushdown_join(
    mut self: QueryContext, index: Int, var filters: List[Expr]
) raises -> Int:
    """FilterPushdown::PushdownInnerJoin and PushdownLeftJoin: a filter
    on one input's columns moves into it (a left join's right input
    keeps its filters out, its unmatched rows are null there). A filter
    on a left join's right columns that rejects nulls rejects every
    unmatched row, so the join is inner (FilterRemovesNull). A filter
    on one input's join keys also holds for the other input's keys,
    as the FilterCombiner's equivalence sets derive. Filters on both
    inputs stay above."""
    var sides = plan_join_sides(self, index)
    ref left_cols = sides[0]
    ref right_cols = sides[1]
    if self._nodes[index].how == JOIN_LEFT:
        for f in filters:
            if plan_left_join_to_inner(self, f, index, left_cols, right_cols):
                break
    var inner = self._nodes[index].how == JOIN_INNER
    var left_below = List[Expr]()
    var right_below = List[Expr]()
    var remain = List[Expr]()
    for f in filters:
        var reads = _references(f)
        if _covers(left_cols, reads.value()):
            var copy = plan_transfer_key_filter(self, f, index, False)
            if copy:
                _ = plan_add_conjunct(right_below, copy.value())
            left_below.append(f.copy())
        elif inner and _covers_exclusive(right_cols, left_cols, reads.value()):
            var copy = plan_transfer_key_filter(self, f, index, True)
            if copy:
                _ = plan_add_conjunct(left_below, copy.value())
            right_below.append(f.copy())
        else:
            remain.append(f.copy())
    var left = self._nodes[index].left
    var right = self._nodes[index].right
    var new_left = plan_pushdown(self, left, left_below^)
    self._nodes[index].left = new_left
    var new_right = plan_pushdown(self, right, right_below^)
    self._nodes[index].right = new_right
    return plan_final_filters(self, index, remain)


def plan_pushdown_semi_anti_join(
    mut self: QueryContext, index: Int, var filters: List[Expr]
) raises -> Int:
    """FilterPushdown::PushdownSemiAntiJoin: a semi or anti join's output
    is its left input's rows, so every filter on them moves into the
    left input; the right input starts a pushdown of its own, with any
    filter on the left join keys carried over to the right keys."""
    var sides = plan_join_sides(self, index)
    ref left_cols = sides[0]
    var left_below = List[Expr]()
    var right_below = List[Expr]()
    var remain = List[Expr]()
    for f in filters:
        var reads = _references(f)
        if not _covers(left_cols, reads.value()):
            remain.append(f.copy())
            continue
        var copy = plan_transfer_key_filter(self, f, index, False)
        if copy:
            _ = plan_add_conjunct(right_below, copy.value())
        left_below.append(f.copy())
    var left = self._nodes[index].left
    var right = self._nodes[index].right
    var new_left = plan_pushdown(self, left, left_below^)
    self._nodes[index].left = new_left
    var new_right = plan_pushdown(self, right, right_below^)
    self._nodes[index].right = new_right
    return plan_final_filters(self, index, remain)


def plan_left_join_to_inner(
    mut self: QueryContext,
    predicate: Expr,
    join: Int,
    left_cols: List[String],
    right_cols: List[String],
) raises -> Bool:
    """FilterRemovesNull: convert a left join to an inner join when a
    filter above it reads only right-side columns and rejects a row
    whose right-side columns are all null. Those are exactly the extra
    unmatched rows supplied by the left join."""
    var reads = _references(predicate)
    if not reads:
        return False
    if not _covers_exclusive(right_cols, left_cols, reads.value()):
        return False
    var schema = self._execute(self._nodes[join].right, True)
    var nulls = List[Series]()
    var names = List[String]()
    for name in reads.value():
        if name in names:
            continue
        names.append(name)
        nulls.append(Series.full_null(name, schema.column(name).dtype(), 1))
    if DataFrame(nulls^).filter(predicate).height() > 0:
        return False
    self._nodes[join].how = JOIN_INNER
    self._nodes[join].text = "inner"
    trace_path("lazy.left_join_to_inner")
    return True


def plan_transfer_key_filter(
    self: QueryContext, predicate: Expr, join: Int, right_side: Bool
) -> Optional[Expr]:
    """A filter moving into one input of an equi-join that reads only
    that input's join keys also holds for the other input's keys:
    a row there whose key fails it matches no row the filter keeps.
    So the predicate in the other input's key names filters that input
    too. TPC-DS q78's `ss_sold_year = 2000`, above two left joins on
    the year, then also narrows the web and catalog totals."""
    ref node = self._nodes[join]
    if len(node.names) != len(node.right_keys) or len(node.names) == 0:
        return None
    var keys = node.right_keys.copy() if right_side else node.names.copy()
    var other = node.names.copy() if right_side else node.right_keys.copy()
    var copy = predicate.copy()
    for k in range(len(copy._nodes)):
        ref item = copy._nodes[k]
        if item.op == SELECTOR:
            return None
        if item.op != COL:
            continue
        var at = -1
        for j in range(len(keys)):
            if keys[j] == item.text:
                at = j
                break
        if at < 0:
            return None
        copy._nodes[k].text = other[at]
    return copy^


def plan_can_discard(self: QueryContext, index: Int) -> Bool:
    """Whether skipping this subtree preserves evaluation errors.

    File scans can fail while decoding rows even if their schema is valid.
    Only in-memory scans and operators with proven total expressions qualify.
    """
    var pending: List[Int] = [index]
    while pending:
        var at = pending.pop()
        if at < 0:
            continue
        ref node = self._nodes[at]
        if node.kind == SCAN_FRAME:
            continue
        if node.kind not in [SELECT, WITH_COLUMNS, FILTER, SORT, CONCAT, AGG]:
            return False
        if not _cannot_raise(node.exprs):
            return False
        pending.append(node.left)
        pending.append(node.right)
    return True


def plan_push_slices(mut self: QueryContext) raises:
    """Move leading slices below row-local projections; a leading slice
    over CSV becomes n_rows; IPC also pushes nonnegative row offsets."""
    var changed = True
    while changed:
        changed = False
        for i in range(len(self._nodes)):
            ref node = self._nodes[i]
            if (
                node.kind != SLICE
                or node.slice_offset() < 0
                or node.slice_length() < -1
            ):
                continue
            if (
                node.slice_offset() != 0 or node.slice_length() < 0
            ) and not plan_slice_over_ipc(self, node.left):
                continue
            ref below = self._nodes[node.left]
            var preserves_rows = _row_local(below.exprs) and _cannot_raise(
                below.exprs
            )
            if below.kind == SELECT and len(below.exprs):
                # Literal-only selections and selectors matching no
                # columns produce one row, even over an empty input.
                # Until selectors are bound, their shape is unknown.
                var has_column = False
                var has_selector = False
                for expression in below.exprs:
                    for item in expression._nodes:
                        has_column |= item.op == COL
                        has_selector |= item.op == SELECTOR
                preserves_rows &= has_column and not has_selector
            if (
                below.kind == SELECT or below.kind == WITH_COLUMNS
            ) and preserves_rows:
                plan_swap_down(self, i, node.left)
                changed = True
                break
    var pushed = False
    for i in range(len(self._nodes)):
        ref node = self._nodes[i]
        if (
            node.kind != SLICE
            or node.slice_offset() < 0
            or node.slice_length() < -1
        ):
            continue
        ref scan = self._nodes[node.left]
        if scan.kind == SCAN_IPC:
            var replacement = scan.copy()
            var start = Int(scan.ipc_row_offset())
            var advance = (
                min(
                    node.slice_offset(), scan.scan_row_limit()
                ) if scan.scan_row_limit()
                >= 0 else node.slice_offset()
            )
            var remaining = (
                max(0, scan.scan_row_limit() - advance) if scan.scan_row_limit()
                >= 0 else -1
            )
            if node.slice_length() >= 0:
                remaining = (
                    min(remaining, node.slice_length()) if remaining
                    >= 0 else node.slice_length()
                )
            # An offset beyond Int.MAX is necessarily beyond every file.
            if start > Int.MAX - advance:
                replacement.set_ipc_slice("0", 0)
            else:
                replacement.set_ipc_slice(String(start + advance), remaining)
            self._nodes[i] = replacement^
            pushed = True
        elif (
            scan.kind in [SCAN_CSV, SCAN_NDJSON]
            and node.slice_offset() == 0
            and node.slice_length() >= 0
            and (
                scan.scan_row_limit() < 0
                or scan.scan_row_limit() > node.slice_length()
            )
        ):
            self._nodes[node.left].set_scan_row_limit(node.slice_length())
    if pushed:
        plan_reorder(self)


def plan_swap_down(mut self: QueryContext, upper: Int, child: Int):
    """Move the single-input node `upper` from above `child` to directly
    above `child`'s input, then renumber."""
    var parents = plan_parents(self)
    var parent = parents[upper]
    self._nodes[upper].left = self._nodes[child].left
    self._nodes[child].left = upper
    if parent >= 0:
        if self._nodes[parent].left == upper:
            self._nodes[parent].left = child
        else:
            self._nodes[parent].right = child
    plan_forget_schema(self, upper)
    plan_reorder(self)


def plan_slice_over_ipc(self: QueryContext, index: Int) -> Bool:
    var cursor = index
    while self._nodes[cursor].kind in [SELECT, WITH_COLUMNS, SLICE]:
        cursor = self._nodes[cursor].left
    return self._nodes[cursor].kind == SCAN_IPC


def plan_fuse_top_k(mut self: QueryContext):
    """A slice directly above a sort needs only the sort's first
    offset + length rows (SQL's ORDER BY ... LIMIT), so the sort records
    that limit and selects those rows instead of sorting them all
    (#332). The slice stays to apply its offset; a sort holds its limit
    in `length`, which it otherwise leaves at -1."""
    for i in range(len(self._nodes)):
        ref node = self._nodes[i]
        if (
            node.kind != SLICE
            or node.slice_offset() < 0
            or node.slice_length() < 0
        ):
            continue
        ref below = self._nodes[node.left]
        if below.kind != SORT:
            continue
        var limit = node.slice_offset() + node.slice_length()
        if below.sort_limit() < 0 or limit < below.sort_limit():
            self._nodes[node.left].set_sort_limit(limit)


def plan_push_projections(
    mut self: QueryContext, *, prune_outputs: Bool = True
) raises:
    """Scans read only columns that some operator above them uses.

    A join passes each input only its keys and the columns the plan
    above reads from that side, so unused payload columns are never
    read or gathered. A join input that is not a scan gets a plain
    column selection above it.
    """
    var needed = List[Optional[List[String]]](
        length=len(self._nodes), fill=Optional[List[String]]()
    )
    var all_needed = List[Bool](length=len(self._nodes), fill=False)
    all_needed[len(self._nodes) - 1] = True
    var owned = plan_remember_schemas(self)
    # (join index, right side?, columns) for join inputs to narrow.
    var narrow_joins = List[Int]()
    var narrow_sides = List[Bool]()
    var narrow_columns = List[List[String]]()
    # Unions whose inputs must each supply exactly the wanted columns.
    var unions = List[Int]()
    var union_columns = List[List[String]]()
    # The copies of a repeated subplan are each pruned to what all
    # their readers need (`_projection_visits`).
    var visits = plan_projection_visits(self)
    ref copy_of = visits[1]
    var merged_copies = Dict[Int, Bool]()
    for i in visits[0]:
        if copy_of[i] >= 0 and copy_of[i] not in merged_copies:
            merged_copies[copy_of[i]] = True
            plan_share_copy_schemas(self, copy_of, copy_of[i])
            _merge_copy_needs(copy_of, copy_of[i], needed, all_needed)
        if prune_outputs and not all_needed[i] and needed[i]:
            plan_prune_outputs(self, i, needed[i].value())
        ref node = self._nodes[i]
        if node.kind == JOIN and not all_needed[i] and needed[i]:
            var sides = plan_join_input_columns(self, i, needed[i].value())
            if sides:
                for side in range(2):
                    var child = node.right if side == 1 else node.left
                    ref keep = sides.value()[side]
                    needed[child] = keep.copy()
                    narrow_joins.append(i)
                    narrow_sides.append(side == 1)
                    narrow_columns.append(keep.copy())
                continue
        if node.kind == CONCAT and not all_needed[i] and needed[i]:
            unions.append(i)
            union_columns.append(needed[i].value().copy())
        if _is_scan(node.kind):
            if not all_needed[i] and needed[i]:
                var columns = plan_columns_of(self, i)
                var keep = List[String]()
                for c in columns:
                    for want in needed[i].value():
                        if want == c:
                            keep.append(c)
                            break
                if (
                    len(keep) > 0 or node.kind == SCAN_IPC or node.has_dataset()
                ) and len(keep) < len(columns):
                    self._nodes[i].set_scan_projection(keep^)
                    # Only this scan's schema changes among the nodes
                    # still to visit: they come before it, and none
                    # reads it (children precede their parents).
                    try:
                        _ = self._schema_memo.value()[].pop(i)
                    except:
                        pass
            continue
        # What this node reads from its input(s).
        var reads = List[String]()
        var reads_all = all_needed[i] and (
            node.kind == FILTER
            or node.kind == SORT
            or node.kind == SLICE
            or node.kind == UNIQUE
            or node.kind == WITH_COLUMNS
            or node.kind == DROP
            or node.kind == JOIN
            or node.kind == EXPLODE
            or node.kind == UNNEST
            or node.kind == CONCAT
            or _row_order_step(node.kind)
            or node.kind == HCONCAT
            or node.kind == GATHER
        )
        if (
            not prune_outputs
            and node.kind == CONCAT
            and not plan_mask_only_union(self, i)
        ):
            # Without inserting projections, every branch must keep its
            # original output shape. Expressions on an intermediate branch
            # can retain fields that another branch would otherwise drop.
            reads_all = True
        if node.kind == UNNEST:
            reads.append(node.text)
        if node.kind == UNIQUE and len(node.names) == 0:
            reads_all = True
        if node.kind == UNPIVOT:
            # Its output is the index columns and the `on` columns'
            # values, whatever is read above it; with no `on`, every
            # column is a value column.
            reads_all = len(node.names) == 0
            reads = node.unpivot_columns() + node.unpivot_index().copy()
        elif _is_time_window(node.kind):
            # Only what the windows read, whatever is wanted above:
            # their outputs are new columns.
            var window_reads = _time_window_reads(node)
            reads_all = not window_reads
            if window_reads:
                reads = window_reads.take()
        elif not reads_all:
            if needed[i]:
                for n in needed[i].value():
                    reads.append(n)
            for e in node.exprs:
                var r = _references(e)
                if not r:
                    reads_all = True
                else:
                    for n in r.value():
                        reads.append(n)
            for n in node.names:
                reads.append(n)
        for child in [node.left, node.right]:
            if child < 0:
                continue
            if (
                reads_all
                or node.kind == JOIN
                or node.kind == ASOF
                or node.kind == INSERT_COLUMN
                or (node.kind == GATHER and child == node.right)
            ):
                all_needed[child] = True
            elif needed[child]:
                var merged = needed[child].value().copy()
                for n in reads:
                    merged.append(n)
                needed[child] = merged^
            else:
                needed[child] = reads.copy()
    if prune_outputs:
        var aligned = plan_align_union_inputs(self, unions, union_columns)
        plan_narrow_join_inputs(
            self, narrow_joins, narrow_sides, narrow_columns
        )
        if aligned:
            plan_reorder(self)
    if owned:
        self._schema_memo = None


def plan_align_union_inputs(
    mut self: QueryContext, unions: List[Int], columns: List[List[String]]
) raises -> Bool:
    """Give the inputs of each union whose inputs disagree the same
    columns, in its first input's order, as a vertical union needs. Pruning an input's projection
    can leave it wider than another: a select of only typed nulls
    (diagonal concat's padding for a column the input lacks) is not
    pruned, as it would lose its row count. A plain column selection
    above such an input keeps its rows and drops the rest, as the
    join inputs get one (`_narrow_join_inputs`). Inner unions are
    aligned before the unions over them, so each union's first input
    has a schema to read. True when a node was added; the caller
    renumbers the plan."""
    var added = False
    var parents = plan_parents(self)
    for r in range(len(unions)):
        var k = len(unions) - 1 - r
        var union = unions[k]
        # Inputs that already agree are left as they are.
        if plan_columns_of(self, self._nodes[union].left) == plan_columns_of(
            self, self._nodes[union].right
        ):
            continue
        # In its first input's order: the union's own schema would
        # stack the inputs as they are, which is what fails.
        var keep = List[String]()
        for c in plan_columns_of(self, self._nodes[union].left):
            if c in columns[k] and c not in keep:
                keep.append(c)
        for side in range(2):
            var child = self._nodes[union].right if side == 1 else (
                self._nodes[union].left
            )
            if plan_columns_of(self, child) == keep:
                continue
            var exprs = List[Expr]()
            for name in keep:
                exprs.append(col(name))
            self._nodes.append(_select_node(left=child, exprs=exprs))
            if side == 1:
                self._nodes[union].right = len(self._nodes) - 1
            else:
                self._nodes[union].left = len(self._nodes) - 1
            plan_forget_schemas_above(self, union, parents)
            added = True
    return added


def plan_prune_outputs(
    mut self: QueryContext, index: Int, wanted: List[String]
):
    """Drop the expressions of a select or with_columns whose outputs
    nothing above reads, so the columns only they read are not
    scanned (a rename's select, a cast's with_columns). A select keeps
    its row count: it is pruned only when every expression is
    row-local and one it keeps reads a column."""
    ref node = self._nodes[index]
    if node.kind != SELECT and node.kind != WITH_COLUMNS:
        return
    for e in node.exprs:
        if not _references(e):
            return
    if node.kind == SELECT and not _row_local(node.exprs):
        return
    var kept = List[Expr]()
    for k in range(len(node.exprs)):
        var name = node.exprs[k]._name
        if (
            name in wanted and _produced_at(node.exprs, name) == k
        ) or not _cannot_raise(node.exprs[k]):
            kept.append(node.exprs[k].copy())
    if len(kept) == len(node.exprs):
        return
    if node.kind == SELECT and not _reads_input(kept):
        return
    node.exprs = kept^
    if self._schema_memo:
        try:
            _ = self._schema_memo.value()[].pop(index)
        except:
            pass


def plan_join_input_columns(
    self: QueryContext, index: Int, wanted: List[String]
) raises -> Optional[List[List[String]]]:
    """Columns each join input must supply for the wanted output names.

    Keys are always kept. A right column whose output name carries the
    suffix keeps its colliding left column too, so no output is renamed.
    Returns None when the unpruned join would reject its output names,
    so execution still raises that error.
    """
    ref node = self._nodes[index]
    var left_cols = plan_columns_of(self, node.left)
    var right_cols = plan_columns_of(self, node.right)
    var suffix = node.join_suffix()
    var membership = node.how == JOIN_SEMI or node.how == JOIN_ANTI
    # A coalesced right join outputs the right keys and drops the left
    # ones (`join_frames`), so only the left's other columns take names.
    var right_join = node.how == JOIN_RIGHT and node.coalesce
    # The right keys are outputs of a right join and of any join that
    # keeps both sides' keys, named as the right's other columns are.
    var right_keys_out = right_join or not node.coalesce
    var outputs = Dict[String, Bool]()
    for c in left_cols:
        if not (right_join and c in node.names):
            outputs[c] = True
    var left_names = outputs.copy()
    var want = Dict[String, Bool]()
    for w in wanted:
        want[w] = True
    var keep_left = Dict[String, Bool]()
    var keep_right = Dict[String, Bool]()
    for k in node.names:
        keep_left[k] = True
    for k in node.right_keys:
        keep_right[k] = True
    for c in left_cols:
        if c in want:
            keep_left[c] = True
    if not membership:
        for c in right_cols:
            if c in keep_right and not right_keys_out:
                continue
            var name = c + suffix if c in left_names else c
            if name in outputs:
                return None
            outputs[name] = True
            if name in want:
                keep_right[c] = True
                if c in left_names:
                    keep_left[c] = True
    var left_keep = List[String]()
    for c in left_cols:
        if c in keep_left:
            left_keep.append(c)
    var right_keep = List[String]()
    for c in right_cols:
        if c in keep_right:
            right_keep.append(c)
    var sides = List[List[String]]()
    sides.append(left_keep^)
    sides.append(right_keep^)
    return sides^


def plan_narrow_join_inputs(
    mut self: QueryContext,
    joins: List[Int],
    right_sides: List[Bool],
    columns: List[List[String]],
) raises:
    """Select only the kept columns above join inputs that are not scans.

    Scans already read only those columns. Other inputs (a filter, a
    computed column) would otherwise hand the join every column they
    produce. The selection is only added when it drops a column.
    """
    var added = False
    var parents = plan_parents(self)
    for k in range(len(joins)):
        var join = joins[k]
        var child = self._nodes[join].left
        if right_sides[k]:
            child = self._nodes[join].right
        if _is_scan(self._nodes[child].kind):
            continue
        var produced = plan_columns_of(self, child)
        if len(columns[k]) >= len(produced):
            continue
        var exprs = List[Expr]()
        for name in columns[k]:
            exprs.append(col(name))
        self._nodes.append(_select_node(left=child, exprs=exprs))
        # The join and the nodes above it may now produce fewer
        # columns.
        plan_forget_schemas_above(self, join, parents)
        if right_sides[k]:
            self._nodes[join].right = len(self._nodes) - 1
        else:
            self._nodes[join].left = len(self._nodes) - 1
        added = True
    if added:
        plan_reorder(self)


def plan_mask_only_union(self: QueryContext, index: Int) -> Bool:
    """Whether a union tree can narrow every branch without inserted steps."""
    ref node = self._nodes[index]
    if _is_scan(node.kind):
        # Only dataset masks are transferred back to the original plan.
        # A frame or ordinary scan sibling would retain its original width.
        return node.has_dataset()
    if node.kind != CONCAT:
        return False
    return plan_mask_only_union(self, node.left) and plan_mask_only_union(
        self, node.right
    )


def plan_prepare_dataset_projections(mut self: QueryContext) raises:
    """Resolve required dataset fields without rewriting relational steps.

    Schema reconciliation must not reject an unobserved field merely because
    general optimizer rewrites were disabled. Keep the original expressions
    and operators, and copy only the scan's required physical field mask.
    """
    var present = False
    for node in self._nodes:
        present = present or node.has_dataset()
    if not present:
        return
    var projected = self.copy()
    projected._schema_memo = None
    plan_push_projections(projected, prune_outputs=False)
    for i in range(len(self._nodes)):
        if not self._nodes[i].has_dataset():
            continue
        for candidate in projected._nodes:
            if (
                candidate.kind == self._nodes[i].kind
                and candidate.source_slot() == self._nodes[i].source_slot()
                and candidate.has_dataset()
                and candidate.dataset_projected
                and candidate.dataset_spec().signature()
                == self._nodes[i].dataset_spec().signature()
            ):
                self._nodes[i].set_scan_projection(candidate.names.copy())
                break
    self._schema_memo = None
