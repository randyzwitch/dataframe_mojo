"""Logical optimizer passes; runtime evaluation enters through QueryContext ports."""
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
    _row_local,
)
from dataframe.io.csv import (
    CsvSchema,
)
from dataframe.join_type import (
    JOIN_ANTI,
    JOIN_INNER,
    JOIN_LEFT,
    JOIN_SEMI,
)
from dataframe.optimizer.expressions import (
    _emit_shared,
    _fold_casts,
    _fold_constants,
    _shared_text,
)
from dataframe.planning.context import (
    QueryContext,
)
from dataframe.planning.graph import (
    plan_columns_of,
    plan_cover,
    plan_forget_schema,
    plan_forget_schemas_above,
    plan_parents,
    plan_projection_visits,
    plan_remember_schemas,
    plan_reorder,
    plan_share_copy_schemas,
    plan_subplan_classes,
    plan_topological,
)
from dataframe.planning.model import (
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
    _plan_node,
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
    _references,
    _row_order_step,
    _through_projection,
    _time_window_reads,
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
    plan_split_filters(plan)
    plan_filter_shared_subplans(plan)
    # From predicate to projection pushdown, a node keeps the schema it
    # derived across renumbering and the passes between, as Polars' IR
    # nodes keep theirs: those passes change no node's output, and
    # forget the nodes they rewrite.
    plan._schema_memo = Optional(ArcPointer(Dict[Int, DataFrame]()))
    try:
        plan_push_predicates(plan)
        plan_merge_filters(plan)
        plan_push_slices(plan)
        plan_fuse_top_k(plan)
        plan_eliminate_common_subexpressions(plan)
        plan_push_projections(plan)
    except e:
        plan._schema_memo = None
        raise e^
    plan._schema_memo = None
    return plan^


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
            _plan_node(WITH_COLUMNS, self._nodes[i].left, exprs=shared)
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
            _plan_node(
                AGG,
                node.left,
                exprs=[col(columns[0]).len().alias(counted)],
                names=columns,
                maintain_order=node.maintain_order,
            )
        )
        var shown = List[Expr]()
        for c in columns:
            shown.append(col(c))
        self._nodes[i] = _plan_node(SELECT, len(self._nodes) - 1, exprs=shown)
        changed = True
    if owned:
        self._schema_memo = None
    if changed:
        plan_reorder(self)


def plan_filter_shared_subplans(mut self: QueryContext):
    """A subplan written in several places that `_share_subplans` will
    compute once (it aggregates), read by each copy only through
    row-local filters, needs only the rows some reader keeps: the OR
    of the readers' filters goes above every copy and moves into it,
    while each reader's own filters are pinned where they are, so the
    copies stay alike and are still computed once. TPC-DS
    q4, q11 and q74 read one yearly total per channel and year from a
    table of every channel and year."""
    # Copies need two aggregations alike in their kind, keys,
    # aggregate count and subtree size at least; most plans have none,
    # and subplan classes are not worth finding for them (ClickBench
    # q29's ninety sums, TPC-DS q41's two small distincts).
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
        if node.kind != AGG and node.kind != UNIQUE:
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
    ref reducing = computed[2]
    var parents = plan_parents(self)
    var covered = List[Bool](length=len(self._nodes), fill=False)
    var count = len(self._nodes)
    while True:
        # The largest aggregating subtree written twice or more, under
        # its readers' filters (two readers may share one).
        var best = -1
        for i in range(count):
            if covered[i] or not reducing[i]:
                continue
            if self._nodes[i].kind == FILTER:
                continue
            if best >= 0 and sizes[i] <= sizes[best]:
                continue
            for j in range(count):
                if j != i and not covered[j] and classes[j] == classes[i]:
                    best = i
                    break
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
                self._nodes[r].flags = [True]
            self._nodes.append(
                _plan_node(FILTER, copies[c], exprs=[any.copy()])
            )
            self._nodes[run[0]].left = len(self._nodes) - 1
    if len(self._nodes) > count:
        plan_split_filters(self, count)
        plan_reorder(self)


def plan_split_filters(mut self: QueryContext, first: Int = 0):
    """A row-local filter on `a & b` becomes a filter on b over one on
    a, so predicate pushdown moves each part as far as it can go; a
    filter on an OR of ANDs also gets the single-input filters it
    implies (`_implied_filters`). `_merge_filters` rejoins the parts
    that end up together. Only nodes from `first` on are split."""
    var count = len(self._nodes)
    var added = False
    for i in range(first, count):
        if self._nodes[i].kind != FILTER:
            continue
        if not _row_local(self._nodes[i].exprs):
            continue
        var parts = _boolean_parts(self._nodes[i].exprs[0], AND)
        var extra = List[Expr]()
        for part in parts:
            for implied in _implied_filters(part):
                extra.append(implied.copy())
        if len(parts) + len(extra) < 2:
            continue
        var below = self._nodes[i].left
        for implied in extra:
            self._nodes.append(
                _plan_node(FILTER, below, exprs=[implied.copy()])
            )
            below = len(self._nodes) - 1
        for k in range(len(parts) - 1, 0, -1):
            self._nodes.append(
                _plan_node(FILTER, below, exprs=[parts[k].copy()])
            )
            below = len(self._nodes) - 1
        self._nodes[i].left = below
        self._nodes[i].exprs = [parts[0].copy()]
        added = True
    if added:
        plan_reorder(self)


def plan_merge_filters(mut self: QueryContext):
    """Join a row-local filter directly over another into one filter
    on both predicates, lower first, so a scan below sees them all (CSV
    range filters, Parquet row-group pruning)."""
    var changed = True
    while changed:
        changed = False
        for i in range(len(self._nodes)):
            ref node = self._nodes[i]
            if node.kind != FILTER or node.left < 0:
                continue
            ref below = self._nodes[node.left]
            if below.kind != FILTER:
                continue
            if not (_row_local(node.exprs) and _row_local(below.exprs)):
                continue
            # A reader's pinned filters merge with each other, never
            # with the shared subplan's own.
            if _pinned(node) != _pinned(below):
                continue
            var merged = below.exprs[0] & node.exprs[0]
            var child = below.left
            self._nodes[i].exprs = [merged^]
            self._nodes[i].left = child
            plan_forget_schema(self, i)
            plan_reorder(self)
            changed = True
            break


def plan_push_predicates(mut self: QueryContext) raises:
    """Swap each filter below operators that cannot change its result."""
    var owned = plan_remember_schemas(self)
    try:
        plan_push_predicates_now(self)
        plan_drop_constant_filters(self)
    except e:
        if owned:
            self._schema_memo = None
        raise e^
    if owned:
        self._schema_memo = None


def plan_drop_constant_filters(mut self: QueryContext) raises:
    """A filter folded to true (`_fold_constants`) keeps every row
    and goes; one folded to false keeps none, and it and its input
    become an empty frame with the same columns. A UNION ALL branch
    whose constant fails a filter is never run."""
    var parents = plan_parents(self)
    var changed = False
    for i in range(len(self._nodes)):
        ref node = self._nodes[i]
        if node.kind != FILTER or _pinned(node):
            continue
        ref nodes = node.exprs[0]._nodes
        if len(nodes) != 1 or nodes[0].op != LIT_BOOL:
            continue
        if nodes[0].integer != 0:
            var parent = parents[i]
            if parent < 0:
                continue
            var input = node.left
            if self._nodes[parent].left == i:
                self._nodes[parent].left = input
            if self._nodes[parent].right == i:
                self._nodes[parent].right = input
            parents[input] = parent
        else:
            var empty = self._execute(i, True)
            self._frames.append(empty^)
            self._schemas.append(Optional[CsvSchema]())
            self._nodes[i] = _plan_node(
                SCAN_FRAME, offset=len(self._frames) - 1
            )
        changed = True
    if changed:
        plan_reorder(self)


def plan_push_predicates_now(mut self: QueryContext) raises:
    # Filters are visited in children-before-parents order and moved
    # in place: a filter keeps its input's columns, so no other node's
    # schema changes and only the moved filters' are forgotten. The
    # plan is renumbered once at the end.
    var root = len(self._nodes) - 1
    var moved = False
    var changed = True
    # Filters found unable to move. A filter's verdict depends on its
    # input and the run of filters under it, never on its parent, and
    # a move changes no node's columns: after one, only the filters
    # around the old and new place are asked again (`_unstick`).
    var stuck = List[Bool](length=len(self._nodes), fill=False)
    while changed:
        changed = False
        for i in plan_topological(self, root):
            if self._nodes[i].kind != FILTER:
                continue
            if i < len(stuck) and stuck[i]:
                continue
            var child = self._nodes[i].left
            var side = plan_passes(self, i, child)
            if side >= 0:
                # The filter moves as far as it goes before the scan
                # restarts: a moved filter stays first in the order it
                # was found in, and no move below a filter changes
                # where an earlier one can go, so the moves are the
                # ones a restart after each would make, without a
                # scan of the plan per move.
                while side >= 0:
                    if i == root:
                        root = child
                    ref below = self._nodes[child]
                    if below.kind == SELECT or below.kind == WITH_COLUMNS:
                        self._nodes[i].exprs = [
                            _fold_constants(
                                _through_projection(
                                    self._nodes[i].exprs[0], below.exprs
                                )
                            )
                        ]
                    elif below.kind == JOIN:
                        plan_transfer_key_filter(self, i, child, side == 1)
                    elif below.kind == CONCAT:
                        # A copy filters the second input; the filter
                        # itself moves above the first.
                        self._nodes.append(
                            _plan_node(
                                FILTER,
                                below.right,
                                exprs=self._nodes[i].exprs.copy(),
                            )
                        )
                        self._nodes[child].right = len(self._nodes) - 1
                    plan_unstick(self, i, child, stuck)
                    plan_swap_in_place(self, i, child, side == 1)
                    child = self._nodes[i].left
                    side = plan_passes(self, i, child)
                changed = True
                moved = True
                break
            # Over another row-local filter that is stuck where this one
            # could go further (a filter on an AND split into parts,
            # one on the keys of a group_by, one on its aggregates):
            # trade places, so this one moves on next time round.
            if not _row_local(self._nodes[i].exprs):
                continue
            # The operator under the run of row-local filters below.
            var under = child
            while (
                under >= 0
                and self._nodes[under].kind == FILTER
                and _row_local(self._nodes[under].exprs)
            ):
                under = self._nodes[under].left
            # Not over a join: the join order and stream batches are
            # tuned to where filters sit, and moving an OR-implied range
            # below the joins cost TPC-DS q13 10% (an OR of decimal
            # ranges over 2.9M rows, and joins on batches a tenth the
            # size) and q69 14%. Over a group_by (TPC-DS q39) or a
            # with_columns (PDS-H q22's phone prefix, 8%) it pays.
            if (
                under != child
                and under >= 0
                and self._nodes[under].kind != JOIN
                and plan_passes(self, child, under) < 0
                and plan_passes(self, i, under) >= 0
            ):
                if i == root:
                    root = child
                plan_unstick(self, i, child, stuck)
                plan_swap_in_place(self, i, child, False)
                changed = True
                moved = True
                break
            if i < len(stuck):
                stuck[i] = True
    if moved:
        plan_reorder(self, root)


def plan_unstick(
    self: QueryContext, filter: Int, child: Int, mut stuck: List[Bool]
):
    """Before `filter` moves below `child`: forget the verdicts it can
    change, those of `filter` and `child` and of the run of filters
    ending at `filter`'s parent, whose input or run below changes."""
    var parents = plan_parents(self)
    for node in [filter, child]:
        if node >= 0 and node < len(stuck):
            stuck[node] = False
    var above = parents[filter]
    while above >= 0 and self._nodes[above].kind == FILTER:
        if above < len(stuck):
            stuck[above] = False
        above = parents[above]


def plan_passes(self: QueryContext, filter: Int, child: Int) raises -> Int:
    """Whether the filter at `filter` can move below the node at
    `child` without changing its result: -1 when not, else the side of
    `child` it moves into (0 left or only input, 1 right)."""
    var reads = _references(self._nodes[filter].exprs[0])
    if not reads or child < 0 or _pinned(self._nodes[filter]):
        return -1
    ref below = self._nodes[child]
    if below.kind == WITH_COLUMNS or below.kind == SELECT:
        # A window or aggregate over the input (a rank, a sum over a
        # partition) depends on which rows are there: filtering first
        # changes its values even for rows the filter keeps (TPC-DS
        # q47 ranked months after `d_year = 1999` was pushed below the
        # rank, so every rank shifted by one). A select of those, or of
        # literals alone, also changes the row count.
        if not _row_local(below.exprs) or not _row_local(
            self._nodes[filter].exprs
        ):
            return -1
        if below.kind == SELECT and not _reads_input(below.exprs):
            return -1
        # Each column the filter reads must be an input column the
        # projection passes on, renamed or not, or a constant: the
        # filter is then rewritten in the input's terms
        # (`_through_projection`), as Polars renames a predicate's
        # columns through an alias projection and DuckDB substitutes a
        # projection's expressions (`PushdownProjection`).
        for name in reads.value():
            var at = _produced_at(below.exprs, name)
            if at < 0:
                if below.kind == SELECT:
                    return -1
                continue
            if not _plain_or_constant(below.exprs[at]):
                return -1
        return 0
    if below.kind == CONCAT and _row_local(self._nodes[filter].exprs):
        # Both inputs supply the same names and types, so the filter
        # applies to each (`_push_into_concat`), as both engines push
        # predicates into every branch of a UNION ALL.
        return 0
    if (
        below.kind == AGG
        and len(below.names) > 0
        and _row_local(self._nodes[filter].exprs)
        and _covers(below.names, reads.value())
    ):
        # A predicate on the grouping keys alone is the same for every
        # row of a group, so filtering rows before the aggregation keeps
        # exactly the groups it would keep after, with the same values
        # and first-occurrence order (Polars and DuckDB push it down
        # too; TPC-DS q39 filters d_moy after grouping 2.3M rows).
        return 0
    if (
        below.kind == PIVOT
        and _row_local(self._nodes[filter].exprs)
        and _covers(List[String](below.names[: below.offset]), reads.value())
    ):
        # As for a group_by's keys: each output row is one index key,
        # so a filter on index columns keeps the same rows before it.
        return 0
    if below.kind == SORT and _row_local(self._nodes[filter].exprs):
        # Stable sorting and a row-local filter commute: the surviving
        # rows retain the same relative sort order.
        return 0
    if below.kind == JOIN and (
        below.how == JOIN_INNER
        or below.how == JOIN_LEFT
        or below.how == JOIN_SEMI
        or below.how == JOIN_ANTI
    ):
        var left_cols = plan_columns_of(self, below.left)
        var right_cols = List[String]()
        for c in plan_columns_of(self, below.right):
            # Coalesced right keys are not in the join output, so no
            # filter above can mean them.
            if not (below.coalesce and c in below.right_keys):
                right_cols.append(c)
        if _covers(left_cols, reads.value()):
            return 0
        if below.how == JOIN_INNER and _covers_exclusive(
            right_cols, left_cols, reads.value()
        ):
            return 1
    return -1


def plan_transfer_key_filter(
    mut self: QueryContext, filter: Int, join: Int, right_side: Bool
):
    """A filter moving into one input of an equi-join that reads only
    that input's join keys also holds for the other input's keys:
    a row there whose key fails it matches no row the filter keeps.
    So a copy in the other input's key names filters that input too:
    join keys in both inputs belong to the same equivalence set.
    TPC-DS q78's `ss_sold_year = 2000`, above two left joins
    on the year, then also narrows the web and catalog totals."""
    ref node = self._nodes[join]
    if len(node.names) != len(node.right_keys) or len(node.names) == 0:
        return
    if not _row_local(self._nodes[filter].exprs):
        return
    var keys = node.right_keys.copy() if right_side else node.names.copy()
    var other = node.names.copy() if right_side else node.right_keys.copy()
    var predicate = self._nodes[filter].exprs[0].copy()
    for k in range(len(predicate._nodes)):
        ref item = predicate._nodes[k]
        if item.op == SELECTOR:
            return
        if item.op != COL:
            continue
        var at = -1
        for j in range(len(keys)):
            if keys[j] == item.text:
                at = j
                break
        if at < 0:
            return
        predicate._nodes[k].text = other[at]
    var target = node.left if right_side else node.right
    self._nodes.append(_plan_node(FILTER, target, exprs=[predicate^]))
    if right_side:
        self._nodes[join].left = len(self._nodes) - 1
    else:
        self._nodes[join].right = len(self._nodes) - 1


def plan_swap_down(
    mut self: QueryContext, filter: Int, child: Int, right_side: Bool
):
    """Move filter from above child to directly above child's input."""
    plan_swap_in_place(self, filter, child, right_side)
    plan_reorder(self)


def plan_swap_in_place(
    mut self: QueryContext, filter: Int, child: Int, right_side: Bool
):
    """`_swap_down` without renumbering the plan: the nodes keep their
    indices, so it is out of children-before-parents order until the
    next `_reorder`. Only the moved filter's and its old child's
    schemas are forgotten (a filter keeps its input's columns)."""
    var parents = plan_parents(self)
    var parent = parents[filter]
    var target = (
        self._nodes[child].right if right_side else self._nodes[child].left
    )
    self._nodes[filter].left = target
    if right_side:
        self._nodes[child].right = filter
    else:
        self._nodes[child].left = filter
    if parent >= 0:
        if self._nodes[parent].left == filter:
            self._nodes[parent].left = child
        else:
            self._nodes[parent].right = child
    if self._schema_memo:
        ref memo = self._schema_memo.value()[]
        for moved in [filter, child]:
            try:
                _ = memo.pop(moved)
            except:
                pass


def plan_push_slices(mut self: QueryContext) raises:
    """Move leading slices below row-local projections; a leading slice
    over CSV becomes n_rows; IPC also pushes nonnegative row offsets."""
    var changed = True
    while changed:
        changed = False
        for i in range(len(self._nodes)):
            ref node = self._nodes[i]
            if node.kind != SLICE or node.offset < 0 or node.length < -1:
                continue
            if (
                node.offset != 0 or node.length < 0
            ) and not plan_slice_over_ipc(self, node.left):
                continue
            ref below = self._nodes[node.left]
            var preserves_rows = _row_local(below.exprs)
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
                plan_swap_down(self, i, node.left, False)
                changed = True
                break
    var pushed = False
    for i in range(len(self._nodes)):
        ref node = self._nodes[i]
        if node.kind != SLICE or node.offset < 0 or node.length < -1:
            continue
        ref scan = self._nodes[node.left]
        if scan.kind == SCAN_IPC:
            var replacement = scan.copy()
            var start = Int(scan.names2[0])
            var advance = (
                min(node.offset, scan.length) if scan.length
                >= 0 else node.offset
            )
            var remaining = (
                max(0, scan.length - advance) if scan.length >= 0 else -1
            )
            if node.length >= 0:
                remaining = (
                    min(remaining, node.length) if remaining
                    >= 0 else node.length
                )
            # An offset beyond Int.MAX is necessarily beyond every file.
            if start > Int.MAX - advance:
                replacement.names2 = ["0"]
                replacement.length = 0
            else:
                replacement.names2 = [String(start + advance)]
                replacement.length = remaining
            self._nodes[i] = replacement^
            pushed = True
        elif (
            scan.kind in [SCAN_CSV, SCAN_NDJSON]
            and node.offset == 0
            and node.length >= 0
            and (scan.length < 0 or scan.length > node.length)
        ):
            self._nodes[node.left].length = node.length
    if pushed:
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
        if node.kind != SLICE or node.offset < 0 or node.length < 0:
            continue
        ref below = self._nodes[node.left]
        if below.kind != SORT:
            continue
        var limit = node.offset + node.length
        if below.length < 0 or limit < below.length:
            self._nodes[node.left].length = limit


def plan_push_projections(mut self: QueryContext) raises:
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
        if not all_needed[i] and needed[i]:
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
                if (len(keep) > 0 or node.kind == SCAN_IPC) and len(keep) < len(
                    columns
                ):
                    self._nodes[i].names = keep^
                    if node.kind == SCAN_IPC:
                        self._nodes[i].flags[1] = True
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
        if node.kind == UNNEST:
            reads.append(node.text)
        if node.kind == UNIQUE and len(node.names) == 0:
            reads_all = True
        if node.kind == UNPIVOT:
            # Its output is the index columns and the `on` columns'
            # values, whatever is read above it; with no `on`, every
            # column is a value column.
            reads_all = len(node.names) == 0
            reads = node.names + node.names2.copy()
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
    var aligned = plan_align_union_inputs(self, unions, union_columns)
    plan_narrow_join_inputs(self, narrow_joins, narrow_sides, narrow_columns)
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
            self._nodes.append(_plan_node(SELECT, child, exprs=exprs))
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
        if name in wanted and _produced_at(node.exprs, name) == k:
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
    var suffix = node.names2[0]
    var membership = node.how == JOIN_SEMI or node.how == JOIN_ANTI
    var outputs = Dict[String, Bool]()
    for c in left_cols:
        outputs[c] = True
    var left_names = outputs.copy()
    var want = Dict[String, Bool]()
    for w in wanted:
        want[w] = True
    if not node.coalesce:
        # Right keys then appear in the output as well; do not prune.
        return None
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
            if c in keep_right:
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
        self._nodes.append(_plan_node(SELECT, child, exprs=exprs))
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
