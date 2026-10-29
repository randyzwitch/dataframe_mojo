"""Plan traversal, identity, schemas, and cardinality properties."""
from dataframe.planning.node_builders import (
    _frame_scan_node,
)
from dataframe.storage.boolean import (
    true_count,
)
from dataframe.dtype import (
    DataType,
)
from dataframe.expr import (
    COL,
)
from dataframe.frame import (
    DataFrame,
)
from dataframe.aggregation.expressions import (
    _row_local,
    _stream_reductions,
)
from dataframe.io.csv import (
    CsvSchema,
)
from dataframe.kernels.join_hash import (
    int64_progression,
)
from dataframe.join_order import (
    JoinEdge,
)
from dataframe.join_type import (
    JOIN_ANTI,
    JOIN_CROSS,
    JOIN_INNER,
    JOIN_LEFT,
    JOIN_SEMI,
)
from dataframe.planning.context import (
    QueryContext,
)
from dataframe.planning.model import (
    AGG,
    CONCAT,
    DROP,
    FILTER,
    GROUPING_SETS,
    JOIN,
    PlanNode,
    SCAN_CSV,
    SCAN_FRAME,
    SCAN_IPC,
    SCAN_NDJSON,
    SCAN_PARQUET,
    SELECT,
    SLICE,
    SORT,
    UNIQUE,
    UNNEST,
    WITH_COLUMNS,
)
from dataframe.planning.properties import (
    _chain_member,
    _is_scan,
    _is_time_window,
    _mix,
    _mix_expr,
    _mix_list,
    _mix_text,
    _order_insensitive,
    _partitioned_order_free,
    _produced_at,
    _row_order_step,
    _same_expr,
    _stream_rows,
)
from dataframe.planning.expression_inputs import (
    _references,
)
from dataframe.series import (
    Series,
)
from std.collections import (
    Dict,
    Optional,
)
from std.memory import (
    ArcPointer,
)


def plan_height_bound(self: QueryContext, index: Int) -> Int:
    """An upper bound on the rows a node yields without executing it:
    the scanned frame's height through steps that only drop or keep
    rows; -1 when unknown (joins, aggregations, file scans)."""
    ref node = self._nodes[index]
    if node.kind == SCAN_FRAME:
        return self._frames[node.source_slot()].height()
    if (
        node.kind == FILTER
        or node.kind == DROP
        or (
            (node.kind == SELECT or node.kind == WITH_COLUMNS)
            and _stream_rows(node)
        )
    ):
        return plan_height_bound(self, node.left)
    if node.kind == SLICE:
        var height = plan_height_bound(self, node.left)
        if height >= 0 and node.slice_length() >= 0:
            return min(height, node.slice_length())
        return height
    return -1


def plan_keeps_column(self: QueryContext, index: Int, name: String) -> Bool:
    """Whether row-local node `index` passes column `name` on
    unchanged: a filter, a drop that keeps it, a with_columns that
    does not overwrite it, or a select that keeps it as itself."""
    ref node = self._nodes[index]
    if node.kind == FILTER:
        return True
    if node.kind == DROP:
        return name not in node.names
    if node.kind == WITH_COLUMNS:
        for e in node.exprs:
            if e._name == name:
                return False
        return True
    if node.kind == SELECT:
        for e in node.exprs:
            if (
                len(e._nodes) == 1
                and e._nodes[0].op == COL
                and e._nodes[0].text == name
                and (not e._name.byte_length() or e._name == name)
            ):
                return True
        return False
    return False


def plan_input_name(
    self: QueryContext, index: Int, name: String
) -> Optional[String]:
    """The name column `name` has in the input of row-local node
    `index` when the node passes it on unchanged, under that name or
    renamed (`col(a).alias(name)`, a projection of a column reference);
    None when the node
    computes or drops it."""
    ref node = self._nodes[index]
    if node.kind == FILTER:
        return name
    if node.kind == DROP:
        if name in node.names:
            return None
        return name
    if node.kind != SELECT and node.kind != WITH_COLUMNS:
        return None
    var at = _produced_at(node.exprs, name)
    if at < 0:
        if node.kind == WITH_COLUMNS:
            return name
        return None
    ref made = node.exprs[at]
    if len(made._nodes) == 1 and made._nodes[0].op == COL:
        return made._nodes[0].text
    return None


def plan_grouped_on(
    self: QueryContext, index: Int, key: String
) -> Tuple[Int, String]:
    """The aggregate under node `index`, through row-local steps that
    pass column `key` on (renamed or not), when the column is one of
    its group keys, and the key's name there; (-1, "") otherwise."""
    var at = index
    var name = key
    while at >= 0:
        ref node = self._nodes[at]
        if node.kind == AGG:
            if name in node.names:
                return (at, name)
            return (-1, String())
        if not (_stream_rows(node) or node.kind == DROP):
            return (-1, String())
        var below = plan_input_name(self, at, name)
        if not below:
            return (-1, String())
        name = below.value()
        at = node.left
    return (-1, String())


def plan_key_source(
    self: QueryContext, index: Int, key: String
) raises -> Optional[Series]:
    """The table column that supplies column `key` at node `index`,
    through row-local steps that keep it, aggregates grouped on it and
    the probe side of joins; None when it is made along the way."""
    var at = index
    while at >= 0:
        ref node = self._nodes[at]
        if node.kind == SCAN_FRAME:
            ref frame = self._frames[node.source_slot()]
            if key in frame.columns():
                return frame.column(key)
            return None
        if node.kind == AGG:
            if key not in node.names:
                return None
        elif node.kind == JOIN:
            if node.how == JOIN_CROSS:
                return None
        elif not (_stream_rows(node) or node.kind == DROP):
            return None
        elif not plan_keeps_column(self, at, key):
            return None
        at = node.left
    return None


def plan_ends_in_breaker(self: QueryContext, index: Int) -> Bool:
    """Whether node `index` is a pipeline breaker (an aggregate, a
    unique, a sort or a row-order operator such as tail), or row-local
    steps over one: it is computed in full before any of its rows stream
    on. A selection that reads its whole input (aggregates over all
    rows, or windows) is one too:
    TPC-DS q28's six one-row bands, cross joined, ran one after
    another."""
    var at = index
    while at >= 0:
        ref node = self._nodes[at]
        if (
            node.kind == AGG
            or node.kind == UNIQUE
            or node.kind == SORT
            or _is_time_window(node.kind)
            or _row_order_step(node.kind)
        ):
            return True
        if (
            node.kind == SELECT or node.kind == WITH_COLUMNS
        ) and not _row_local(node.exprs):
            return True
        if node.kind == DROP or _stream_rows(node):
            at = node.left
            continue
        return False
    return False


def plan_known_height(self: QueryContext, index: Int) -> Int:
    """Exact cheap cardinalities only; -1 means execution is required."""
    ref node = self._nodes[index]
    if node.kind == SCAN_FRAME:
        return self._frames[node.source_slot()].height()
    if (
        node.kind == DROP
        or node.kind == UNNEST
        or (
            (node.kind == SELECT or node.kind == WITH_COLUMNS)
            and _stream_rows(node)
        )
    ):
        return plan_known_height(self, node.left)
    if node.kind == SLICE and node.slice_offset() >= 0:
        var height = plan_known_height(self, node.left)
        if height >= 0:
            height = max(0, height - node.slice_offset())
            return (
                min(height, node.slice_length()) if node.slice_length()
                >= 0 else height
            )
    return -1


def plan_cover(self: QueryContext, index: Int, mut covered: List[Bool]):
    var stack = List[Int]()
    stack.append(index)
    while len(stack) > 0:
        var i = stack.pop()
        if covered[i]:
            continue
        covered[i] = True
        for child in [self._nodes[i].left, self._nodes[i].right]:
            if child >= 0:
                stack.append(child)


def plan_node_hash(
    self: QueryContext, index: Int, classes: List[Int], canonical: List[Int]
) -> UInt64:
    """A hash of what node `index` computes: its kind and every setting,
    its expressions node by node, and its inputs' classes (already in
    `classes`, since children come before parents). Nodes that
    `_same_node` calls equal hash alike; a collision only costs a
    comparison."""
    ref node = self._nodes[index]
    var h = _mix(UInt64(0xCBF29CE484222325), UInt64(node.kind))
    h = _mix_list(h, node.names)
    h = _mix_list(h, node.names2)
    for flag in node.flags:
        h = _mix(h, UInt64(2) if flag else UInt64(1))
    h = _mix_text(h, node.text)
    h = _mix(h, UInt64(plan_canonical_slot(self, index, canonical)))
    h = _mix(h, UInt64(node.length))
    h = _mix(h, UInt64(1) if node.maintain_order else UInt64(0))
    h = _mix(h, UInt64(node.how))
    h = _mix_list(h, node.right_keys)
    h = _mix(h, UInt64(1) if node.coalesce else UInt64(0))
    h = _mix_text(h, node.source_failure)
    if node.json_schema:
        h = _mix_text(h, node.json_schema.value().name())
    if node.dataset:
        h = _mix_text(h, node.dataset.value().signature())
    h = _mix(h, UInt64(node.dataset_projected))
    for e in node.exprs:
        h = _mix_expr(h, e)
    h = _mix(h, UInt64(classes[node.left] + 1) if node.left >= 0 else 0)
    h = _mix(h, UInt64(classes[node.right] + 1) if node.right >= 0 else 0)
    return h


def plan_canonical_slot(
    self: QueryContext, index: Int, canonical: List[Int]
) -> Int:
    """A node's `offset`, except that a frame scan names the first slot
    holding the same buffers (`canonical`)."""
    ref node = self._nodes[index]
    var slot = node.offset
    if node.kind == SCAN_FRAME and slot >= 0 and slot < len(canonical):
        slot = canonical[slot]
    return slot


def plan_same_node(
    self: QueryContext, a: Int, b: Int, classes: List[Int], canonical: List[Int]
) -> Bool:
    """Whether nodes `a` and `b` compute the same thing: every setting
    and expression is equal, and their inputs have the same classes."""
    ref x = self._nodes[a]
    ref y = self._nodes[b]
    if (
        x.kind != y.kind
        or x.names != y.names
        or x.names2 != y.names2
        or x.flags != y.flags
        or x.text != y.text
        or plan_canonical_slot(self, a, canonical)
        != plan_canonical_slot(self, b, canonical)
        or x.length != y.length
        or x.maintain_order != y.maintain_order
        or x.how != y.how
        or x.right_keys != y.right_keys
        or x.coalesce != y.coalesce
        or x.source_failure != y.source_failure
        or Bool(x.asof_tolerance) != Bool(y.asof_tolerance)
        or Bool(x.json_schema) != Bool(y.json_schema)
        or Bool(x.dataset) != Bool(y.dataset)
        or x.dataset_projected != y.dataset_projected
        or len(x.exprs) != len(y.exprs)
    ):
        return False
    for child in [(x.left, y.left), (x.right, y.right)]:
        var p = child[0]
        var q = child[1]
        if (p < 0) != (q < 0) or (p >= 0 and classes[p] != classes[q]):
            return False
    if x.asof_tolerance and String(x.asof_tolerance.value()) != String(
        y.asof_tolerance.value()
    ):
        return False
    if x.json_schema and x.json_schema.value() != y.json_schema.value():
        return False
    if (
        x.dataset
        and x.dataset.value().signature() != y.dataset.value().signature()
    ):
        return False
    for k in range(len(x.exprs)):
        if not _same_expr(x.exprs[k], y.exprs[k]):
            return False
    return True


def plan_parents(self: QueryContext) -> List[Int]:
    var parents = List[Int](length=len(self._nodes), fill=-1)
    for i in range(len(self._nodes)):
        if self._nodes[i].left >= 0:
            parents[self._nodes[i].left] = i
        if self._nodes[i].right >= 0:
            parents[self._nodes[i].right] = i
    return parents^


def plan_columns_of(self: QueryContext, index: Int) raises -> List[String]:
    return self._execute(index, True).columns()


def plan_repeats_a_scan(self: QueryContext) -> Bool:
    """Whether two scan nodes read the same source: the same file, or
    frames holding the same buffers (plans joined carry their own copies
    of a frame they both scan, in separate slots). Without one no subtree
    can be written twice."""
    var seen = List[Int]()
    var files = List[String]()
    for node in self._nodes:
        if node.kind == SCAN_FRAME:
            for f in seen:
                if f == node.source_slot() or self._frames[
                    f
                ]._shares_buffers_with(self._frames[node.source_slot()]):
                    return True
            seen.append(node.source_slot())
        elif node.kind in [SCAN_CSV, SCAN_PARQUET, SCAN_IPC, SCAN_NDJSON]:
            if node.text in files:
                return True
            files.append(node.text)
    return False


def plan_projection_visits(self: QueryContext) -> Tuple[List[Int], List[Int]]:
    """The order projection pushdown visits nodes in, and for each node
    the first node of the repeated subplan it is a copy of (-1 if
    none). A subtree written in several places (a CTE read twice) is
    one cached subplan, and projection pushdown merges what every
    reader of the cache needs before
    pushing down into it once. Here each copy is pruned to that union,
    so the copies stay alike and `_share_subplans` still runs them
    once: a node comes after every node above it, by height, and the
    copies of one subplan, all of one height, come after all their
    readers. Pruned per reader, TPC-DS q4's year totals ran six
    times."""
    var count = len(self._nodes)
    var order = List[Int](capacity=count)
    var copy_of = List[Int](length=count, fill=-1)
    if not plan_repeats_a_scan(self):
        for reverse in range(count):
            order.append(count - 1 - reverse)
        return (order^, copy_of^)
    var classes = plan_subplan_classes(self)[0].copy()
    var hits = Dict[Int, Int]()
    for c in classes:
        hits[c] = hits.get(c, 0) + 1
    var parents = plan_parents(self)
    # The topmost repeated nodes: more copies than the node above
    # them, and under no other such node (Polars caches the topmost
    # and leaves the subplan below it whole).
    var under = List[Bool](length=count, fill=False)
    var first = Dict[Int, Int]()
    var members = List[Int](length=count, fill=0)
    for reverse in range(count):
        var i = count - 1 - reverse
        var parent = parents[i]
        if parent >= 0:
            under[i] = under[parent] or copy_of[parent] >= 0
        var repeats = hits.get(classes[i], 0)
        # A source read in two places is not run twice: each read
        # keeps only its own columns.
        if under[i] or repeats < 2 or _is_scan(self._nodes[i].kind):
            continue
        if parent >= 0 and hits.get(classes[parent], 0) >= repeats:
            continue
        var head = first.get(classes[i], i)
        first[classes[i]] = head
        copy_of[i] = head
        members[head] += 1
    for i in range(count):
        if copy_of[i] >= 0 and members[copy_of[i]] < 2:
            copy_of[i] = -1
    # By height, highest first; within a height, by descending index
    # as children precede their parents.
    var heights = List[Int](capacity=count)
    var tallest = 0
    for i in range(count):
        ref node = self._nodes[i]
        var height = 0
        if node.left >= 0:
            height = heights[node.left] + 1
        if node.right >= 0:
            height = max(height, heights[node.right] + 1)
        heights.append(height)
        tallest = max(tallest, height)
    var levels = List[List[Int]](length=tallest + 1, fill=List[Int]())
    for reverse in range(count):
        var i = count - 1 - reverse
        levels[heights[i]].append(i)
    for reverse in range(tallest + 1):
        for i in levels[tallest - reverse]:
            order.append(i)
    return (order^, copy_of^)


def plan_subplan_classes(
    self: QueryContext,
) -> Tuple[List[Int], List[Int], List[Bool]]:
    """Each node's class, its subtree's node count, and whether the
    subtree holds an aggregation or a unique. Two nodes share a class
    exactly when their subtrees compute the same thing: hash a node
    with its inputs'
    classes, and compared only with the earlier nodes of its hash,
    setting by setting (`_same_node`). A class is named by its first
    node."""
    # Two plans joined carry their own copies of a frame they both
    # scan, in separate slots: a scan names the first slot holding the
    # same buffers.
    var canonical = List[Int](capacity=len(self._frames))
    for f in range(len(self._frames)):
        var first = f
        for g in range(f):
            if canonical[g] == g and self._frames[g]._shares_buffers_with(
                self._frames[f]
            ):
                first = g
                break
        canonical.append(first)
    var classes = List[Int](capacity=len(self._nodes))
    var sizes = List[Int](capacity=len(self._nodes))
    var reducing = List[Bool](capacity=len(self._nodes))
    # The first node of each class, by hash.
    var firsts = Dict[UInt64, List[Int]]()
    for i in range(len(self._nodes)):
        var h = plan_node_hash(self, i, classes, canonical)
        var found = i
        var known = firsts.get(h)
        if known:
            for r in known.value():
                if plan_same_node(self, i, r, classes, canonical):
                    found = r
                    break
        if found == i:
            if known:
                var heads = known.take()
                heads.append(i)
                firsts[h] = heads^
            else:
                firsts[h] = [i]
        classes.append(found)
        ref node = self._nodes[i]
        var size = 1
        var reduces = (
            node.kind == AGG
            or node.kind == UNIQUE
            or node.kind == GROUPING_SETS
        )
        if node.left >= 0:
            size += sizes[node.left]
            reduces = reduces or reducing[node.left]
        if node.right >= 0:
            size += sizes[node.right]
            reduces = reduces or reducing[node.right]
        sizes.append(size)
        reducing.append(reduces)
    return (classes^, sizes^, reducing^)


def plan_filtered_table(self: QueryContext, index: Int) -> Bool:
    """Whether node `index` is an in-memory table under row-local
    steps, at least one of them a filter."""
    var cursor = index
    var filtered = False
    while cursor >= 0:
        ref node = self._nodes[cursor]
        if node.kind == SCAN_FRAME:
            return filtered
        if node.kind == FILTER and _row_local(node.exprs):
            filtered = True
        elif not (
            node.kind == DROP
            or (
                (node.kind == SELECT or node.kind == WITH_COLUMNS)
                and _stream_rows(node)
            )
        ):
            return False
        cursor = node.left
    return False


def plan_provider_scan(
    self: QueryContext, index: Int, name: String
) raises -> Tuple[Int, String]:
    """The in-memory table node under node `index` that supplies
    column `name`, and the column's name there, following renames
    (`col(a).alias(b)`); (-1, "") when none does or the column is
    computed."""
    var stack = List[Tuple[Int, String]]()
    stack.append((index, name))
    while len(stack) > 0:
        var top = stack.pop()
        var i = top[0]
        var wanted = top[1]
        ref node = self._nodes[i]
        if node.kind == SCAN_FRAME:
            if wanted in self._frames[node.source_slot()].columns():
                return (i, wanted)
            continue
        if node.kind == AGG and wanted not in node.names:
            # An aggregate, not a key: computed here.
            continue
        if node.kind == SELECT or node.kind == WITH_COLUMNS:
            var renamed = False
            var computed = False
            for e in node.exprs:
                if e._name != wanted:
                    continue
                var reads = _references(e)
                if reads and len(reads.value()) == 1 and len(e._nodes) == 1:
                    wanted = reads.value()[0]
                    renamed = True
                else:
                    computed = True
            if computed and not renamed:
                continue
        if node.left >= 0:
            stack.append((node.left, wanted))
        if node.right >= 0:
            stack.append((node.right, wanted))
    return (-1, String())


def plan_has_kind(self: QueryContext, index: Int, kind: Int) -> Bool:
    """Whether a node of `kind` sits anywhere under node `index`."""
    var stack: List[Int] = [index]
    while len(stack) > 0:
        var i = stack.pop()
        ref node = self._nodes[i]
        if node.kind == kind:
            return True
        if node.left >= 0:
            stack.append(node.left)
        if node.right >= 0:
            stack.append(node.right)
    return False


def plan_filters_rows(self: QueryContext, index: Int) -> Bool:
    """Whether a filter, or an inner or semi join, sits anywhere under
    node `index`: a step that can drop rows."""
    var stack: List[Int] = [index]
    while len(stack) > 0:
        var i = stack.pop()
        ref node = self._nodes[i]
        if node.kind == FILTER or (
            node.kind == JOIN
            and (node.how == JOIN_INNER or node.how == JOIN_SEMI)
        ):
            return True
        if node.left >= 0:
            stack.append(node.left)
        if node.right >= 0:
            stack.append(node.right)
    return False


def plan_has_join(self: QueryContext, index: Int) -> Bool:
    """Whether a join sits anywhere under node `index`."""
    var stack: List[Int] = [index]
    while len(stack) > 0:
        var i = stack.pop()
        ref node = self._nodes[i]
        if node.kind == JOIN:
            return True
        if node.left >= 0:
            stack.append(node.left)
        if node.right >= 0:
            stack.append(node.right)
    return False


def plan_largest_scan(self: QueryContext, index: Int) -> Int:
    """The most rows any in-memory table under node `index` holds."""
    var largest = 0
    var stack: List[Int] = [index]
    while len(stack) > 0:
        var i = stack.pop()
        ref node = self._nodes[i]
        if node.kind == SCAN_FRAME:
            largest = max(largest, self._frames[node.source_slot()].height())
        if node.left >= 0:
            stack.append(node.left)
        if node.right >= 0:
            stack.append(node.right)
    return largest


def plan_live_parents(self: QueryContext, root: Int) -> List[Int]:
    """Each node's parent in the plan under `root`: -1 for the root and
    -2 for a node no longer reachable (an earlier rewrite's leftovers)."""
    var parents = List[Int](length=len(self._nodes), fill=-2)
    parents[root] = -1
    var stack: List[Int] = [root]
    while len(stack) > 0:
        var i = stack.pop()
        for child in [self._nodes[i].left, self._nodes[i].right]:
            if child >= 0 and parents[child] == -2:
                parents[child] = i
                stack.append(child)
    return parents^


def _order_from_join(self: QueryContext, index: Int) -> Bool:
    """Whether node `index`'s rows come in a join's order, through steps
    that keep their input's order: a join promises none."""
    var cursor = index
    while cursor >= 0:
        ref node = self._nodes[cursor]
        if node.kind == JOIN:
            return True
        if not (
            node.kind == FILTER
            or node.kind == DROP
            or (
                (node.kind == SELECT or node.kind == WITH_COLUMNS)
                and _row_local(node.exprs)
            )
        ):
            return False
        cursor = node.left
    return False


def plan_order_free_above(
    self: QueryContext,
    top: Int,
    parents: List[Int],
    through_joins: Bool = False,
) -> Bool:
    """Whether the plan above `top` gives the same result whatever
    order `top`'s rows arrive in: row-local steps up to an aggregation
    that reads no row position, or up to a join. A join promises no row
    order (as DuckDB's and Polars' joins by default), so neither the
    order of its inputs nor its own output's order can show; `top` made
    by a join through row-local steps is free for the same reason.
    `through_joins` is kept for callers and no longer changes the
    answer."""
    _ = through_joins
    if _order_from_join(self, top):
        return True
    var cursor = parents[top]
    while cursor >= 0:
        ref node = self._nodes[cursor]
        if node.kind == JOIN:
            return True
        if node.kind == AGG:
            return not node.maintain_order and _order_insensitive(node.exprs)
        if node.kind == SORT:
            # The sort sets the order; equal keys owe none (unless
            # `maintain_order`), so how rows arrived does not show.
            return not node.maintain_order
        if node.kind == SELECT and not _row_local(node.exprs):
            if _stream_reductions(node.exprs):
                return _order_insensitive(node.exprs)
            if not _partitioned_order_free(node.exprs):
                return False
            cursor = parents[cursor]
            continue
        if (
            node.kind == FILTER
            or node.kind == SELECT
            or node.kind == WITH_COLUMNS
        ):
            if not _row_local(node.exprs) and not _partitioned_order_free(
                node.exprs
            ):
                return False
        elif node.kind != DROP:
            return False
        cursor = parents[cursor]
    return False


def plan_estimated_rows(self: QueryContext, index: Int) raises -> Int:
    """Rows node `index` yields, without running it: exact for a table,
    estimated for one row-local filter over a table by applying the
    filter to 64 evenly spaced runs of 1,024 rows (every row of a table
    that small). -1 for anything else. Evenly spaced runs, not a
    prefix: tables are often stored in date or key order, which a
    prefix would misjudge."""
    ref node = self._nodes[index]
    if node.kind == SCAN_FRAME:
        return self._frames[node.source_slot()].height()
    if node.kind == DROP or (
        (node.kind == SELECT or node.kind == WITH_COLUMNS)
        and _stream_rows(node)
    ):
        return plan_estimated_rows(self, node.left)
    if node.kind != FILTER or not _row_local(node.exprs):
        return -1
    ref source = self._nodes[node.left]
    if source.kind != SCAN_FRAME:
        return -1
    var height = self._frames[source.source_slot()].height()
    var runs = 64
    var run = 1024
    # Only the columns the filter reads.
    var reads = _references(node.exprs[0])
    if not reads:
        return -1
    var names = List[String]()
    for name in reads.value():
        if name not in names:
            names.append(name)
    # A large table's runs are copied end to end (zero-copy slices
    # merged into one array), not gathered row by row, once for the
    # table: its statistics keep them.
    var sample = self._frames[source.source_slot()].select(names)
    if height > runs * run:
        sample = self._frames[source.source_slot()]._sampled(names, runs, run)
    # Only the matches are counted: the predicate's mask, as DuckDB
    # counts a selection vector, with no filtered frame built.
    var kept = true_count(sample._selection(node.exprs[0], 8192))
    if height <= runs * run:
        return kept
    return Int(Float64(kept) / Float64(runs * run) * Float64(height))


def plan_progression_key(
    self: QueryContext, index: Int, keys: List[String]
) raises -> Bool:
    """Whether a join building on node `index` with these keys needs no
    hash table: one integer key of a whole table whose values are an
    arithmetic progression, which is looked up by position."""
    if len(keys) != 1:
        return False
    var column = keys[0]
    var cursor = index
    while cursor >= 0:
        ref node = self._nodes[cursor]
        if node.kind == SCAN_FRAME:
            ref frame = self._frames[node.source_slot()]
            if column not in frame.columns():
                return False
            var key = frame[column]
            if key.dtype().physical() != DataType.INT64:
                return False
            return int64_progression(key)[0]
        if node.kind != DROP and not (
            (node.kind == SELECT or node.kind == WITH_COLUMNS)
            and _stream_rows(node)
        ):
            return False
        if node.kind == SELECT or node.kind == WITH_COLUMNS:
            # A renamed column is its source column; a computed one is
            # no table's key.
            for expr in node.exprs:
                if expr._name == column:
                    if len(expr._nodes) != 1 or expr._nodes[0].op != COL:
                        return False
                    column = expr._nodes[0].text
        cursor = node.left
    return False


def plan_graph_step(self: QueryContext, index: Int) -> Bool:
    """Whether node `index` is part of a join graph: an inner join with
    coalesced keys, or a row-local filter or plain column selection
    over one (through others of its kind)."""
    var cursor = index
    while cursor >= 0:
        ref node = self._nodes[cursor]
        if node.kind == JOIN:
            return _chain_member(node)
        if not _chain_member(node):
            return False
        cursor = node.left
    return False


def plan_collect_graph(
    self: QueryContext,
    index: Int,
    mut relations: List[Int],
    mut pairs: List[Tuple[String, String]],
    mut filters: List[Int],
):
    """The relations, key pairs and filters of the join graph under
    node `index`: both inputs of each
    inner join, through filters (kept to place again) and plain column
    selections; anything else is a relation."""
    ref node = self._nodes[index]
    if not plan_graph_step(self, index):
        relations.append(index)
        return
    if node.kind == JOIN:
        plan_collect_graph(self, node.left, relations, pairs, filters)
        plan_collect_graph(self, node.right, relations, pairs, filters)
        for i in range(len(node.names)):
            pairs.append((node.names[i], node.right_keys[i]))
        return
    if node.kind == FILTER:
        filters.append(index)
    plan_collect_graph(self, node.left, relations, pairs, filters)


def plan_indexed_relation(
    self: QueryContext,
    entry: Int,
    other: Int,
    joined: List[Bool],
    node_of: List[Int],
    edges: List[JoinEdge],
    masks: List[Int],
) raises -> Bool:
    """Whether tree entry `entry` is one relation joined to `other` on
    a single key that is a progression in its table (`_progression_key`),
    so building on it needs no hash table."""
    if joined[entry]:
        return False
    var mine = masks[entry]
    var theirs = masks[other]
    var keys = List[String]()
    for edge in edges:
        if mine & (1 << edge.a) != 0 and theirs & (1 << edge.b) != 0:
            for k in edge.a_keys:
                keys.append(k)
        elif mine & (1 << edge.b) != 0 and theirs & (1 << edge.a) != 0:
            for k in edge.b_keys:
                keys.append(k)
    return len(keys) == 1 and plan_progression_key(self, node_of[entry], keys)


def plan_relation_estimate(self: QueryContext, index: Int) raises -> Int:
    """Rows a join-graph relation yields, estimated without running it:
    `_estimated_rows`, else, for a join the order cannot reorder,
    combine its inputs' estimates: a semi or anti join keeps its left
    input's rows; other joins use the larger input's estimate. A UNION
    ALL yields its inputs' rows together, as DuckDB's estimate of a set
    operation (`RelationStatisticsHelper`). -1 when an input has no
    estimate."""
    var known = plan_estimated_rows(self, index)
    if known >= 0:
        return known
    ref node = self._nodes[index]
    if node.kind == CONCAT:
        var left = plan_relation_estimate(self, node.left)
        var right = plan_relation_estimate(self, node.right)
        if left < 0 or right < 0:
            return -1
        return left + right
    if node.kind == DROP or (
        (node.kind == SELECT or node.kind == WITH_COLUMNS)
        and _stream_rows(node)
    ):
        return plan_relation_estimate(self, node.left)
    if node.kind != JOIN or node.how == JOIN_INNER or node.how == JOIN_CROSS:
        return -1
    var left = plan_relation_estimate(self, node.left)
    if left < 0:
        return -1
    if node.how == JOIN_SEMI or node.how == JOIN_ANTI:
        return left
    var right = plan_relation_estimate(self, node.right)
    if right < 0:
        return -1
    return max(left, right)


def plan_spine_frame(self: QueryContext, index: Int) -> Optional[DataFrame]:
    """The in-memory table under `index`, along the steps
    `_spine_rows` follows and a left, semi or anti join's left input,
    whose columns keep their table's statistics; None through anything
    else."""
    var cursor = index
    while cursor >= 0:
        ref node = self._nodes[cursor]
        if node.kind == SCAN_FRAME:
            return Optional(self._frames[node.source_slot()].copy())
        if (
            node.kind == FILTER
            or node.kind == DROP
            or (
                node.kind == JOIN
                and (
                    node.how == JOIN_INNER
                    or node.how == JOIN_LEFT
                    or node.how == JOIN_SEMI
                    or node.how == JOIN_ANTI
                )
            )
            or (
                (node.kind == SELECT or node.kind == WITH_COLUMNS)
                and _stream_rows(node)
            )
        ):
            cursor = node.left
            continue
        return None
    return None


def plan_aggregated(self: QueryContext, index: Int) -> Bool:
    """Whether node `index` is an aggregation, under steps that keep,
    drop or rename its rows' columns (a filter, a projection)."""
    var cursor = index
    while cursor >= 0:
        ref node = self._nodes[cursor]
        if node.kind == AGG:
            return True
        if not (
            node.kind == FILTER
            or node.kind == DROP
            or (
                (node.kind == SELECT or node.kind == WITH_COLUMNS)
                and _stream_rows(node)
            )
        ):
            return False
        cursor = node.left
    return False


def plan_spine_rows(self: QueryContext, index: Int) -> Int:
    """Rows of the table under `index` along steps that keep, drop or
    join rows (a filter, a projection, an inner join's left input);
    -1 through anything else (an aggregation) or a file scan."""
    var cursor = index
    while cursor >= 0:
        ref node = self._nodes[cursor]
        if node.kind == SCAN_FRAME:
            var origin = self._origin_rows.get(node.source_slot(), -1)
            if origin > 0:
                return origin
            return self._frames[node.source_slot()].height()
        if (
            node.kind == FILTER
            or node.kind == DROP
            or (node.kind == JOIN and node.how == JOIN_INNER)
            or (
                (node.kind == SELECT or node.kind == WITH_COLUMNS)
                and _stream_rows(node)
            )
        ):
            cursor = node.left
            continue
        return -1
    return -1


def plan_measurable(self: QueryContext, index: Int) -> Bool:
    """Whether `_selectivity` measures node `index`: a known table
    narrowed by steps above it, or a shared subplan's result, which
    knows the table it narrowed."""
    ref node = self._nodes[index]
    if node.kind == SCAN_FRAME:
        return node.source_slot() in self._origin_rows
    return plan_spine_rows(self, index) > 0


def plan_compact_scan_slots(mut self: QueryContext):
    """Release scan inputs/materializations no reachable node owns."""
    var slots = List[Int](length=len(self._frames), fill=-1)
    var frames = List[DataFrame]()
    var schemas = List[Optional[CsvSchema]]()
    for i in range(len(self._nodes)):
        if not _is_scan(self._nodes[i].kind):
            continue
        var old = self._nodes[i].source_slot()
        if slots[old] < 0:
            slots[old] = len(frames)
            frames.append(self._frames[old].copy())
            schemas.append(self._schemas[old].copy())
        self._nodes[i].set_source_slot(slots[old])
    var origins = Dict[Int, Int]()
    for entry in self._origin_rows.items():
        if slots[entry.key] >= 0:
            origins[slots[entry.key]] = entry.value
    self._frames = frames^
    self._schemas = schemas^
    self._origin_rows = origins^


def plan_available(
    self: QueryContext,
    names: List[String],
    provider: Dict[String, Int],
    owner: List[Int],
    placed: List[Bool],
) raises -> Bool:
    """Whether every name comes from the base or a placed step."""
    for name in names:
        var p = provider[name]
        if p >= 0 and not placed[owner[p]]:
            return False
    return True


def plan_remember_schemas(mut self: QueryContext) -> Bool:
    """Keep each node's schema once derived, until the pass ends. True
    when this pass started remembering, and so ends it; False inside
    passes that already remember (`_optimized`)."""
    if self._schema_memo:
        return False
    self._schema_memo = Optional(ArcPointer(Dict[Int, DataFrame]()))
    return True


def plan_forget_schema(mut self: QueryContext, index: Int):
    """Forget one node's schema, after a rewrite of that node that
    leaves its output as it was (the nodes above keep theirs)."""
    if self._schema_memo:
        try:
            _ = self._schema_memo.value()[].pop(index)
        except:
            pass


def plan_reorder(mut self: QueryContext, root: Int = -1):
    """Restore children-before-parents order after rewiring. Nodes
    move to their new places rather than being copied, and remembered
    schemas move with them, so renumbering changes no node's output.
    Unreachable nodes and their schemas are dropped."""
    var order = plan_topological(self, root)
    var position = List[Int](length=len(self._nodes), fill=-1)
    for k in range(len(order)):
        position[order[k]] = k
    var nodes = List[PlanNode](capacity=len(order))
    for k in range(len(order)):
        # Each node is reached once, so it is taken, not copied.
        var node = _frame_scan_node()
        swap(node, self._nodes[order[k]])
        if node.left >= 0:
            node.left = position[node.left]
        if node.right >= 0:
            node.right = position[node.right]
        nodes.append(node^)
    self._nodes = nodes^
    if self._schema_memo:
        ref memo = self._schema_memo.value()[]
        var moved = Dict[Int, DataFrame]()
        for k in range(len(order)):
            if order[k] in memo:
                try:
                    moved[k] = memo.pop(order[k])
                except:
                    pass
        memo = moved^


def plan_topological(self: QueryContext, root: Int = -1) -> List[Int]:
    """The nodes reachable from `root` (by default the last node no
    other reads), children before parents: the order `_reorder`
    numbers them in."""
    var selected = root
    if selected < 0:
        var parents = plan_parents(self)
        for i in range(len(self._nodes)):
            if parents[i] < 0:
                selected = i
    var order = List[Int]()
    var stack = List[Int]()
    var visited = List[Bool](length=len(self._nodes), fill=False)
    stack.append(selected)
    while len(stack) > 0:
        var top = stack[len(stack) - 1]
        var pending = False
        for child in [self._nodes[top].left, self._nodes[top].right]:
            if child >= 0 and not visited[child]:
                stack.append(child)
                pending = True
        if not pending:
            _ = stack.pop()
            if not visited[top]:
                visited[top] = True
                order.append(top)
    return order^


def plan_forget_schemas_above(
    mut self: QueryContext, index: Int, parents: List[Int]
):
    """Forget the schemas of a node whose output changed and of the
    nodes above it, derived from it. Only these: forgetting by node
    number would leave the schemas of one copy of a repeated subplan
    and drop another's, and the copies would then be narrowed
    differently and no longer shared."""
    if not self._schema_memo:
        return
    ref memo = self._schema_memo.value()[]
    var at = index
    while at >= 0:
        try:
            _ = memo.pop(at)
        except:
            pass
        at = parents[at]


def plan_share_copy_schemas(
    mut self: QueryContext, copy_of: List[Int], head: Int
):
    """Give every copy of the repeated subplan `head` the schemas any
    copy already derived, node for node. The copies are alike and none
    of their nodes is pruned yet, so the schemas agree; a schema kept
    above a node pruned later goes stale alike in every copy, and the
    selections added after pruning (`_narrow_join_inputs`) agree, so
    the copies stay alike and are still shared."""
    if not self._schema_memo:
        return
    ref memo = self._schema_memo.value()[]
    var stack = List[List[Int]]()
    var copies = List[Int]()
    for i in range(len(copy_of)):
        if copy_of[i] == head:
            copies.append(i)
    stack.append(copies^)
    while len(stack) > 0:
        var nodes = stack.pop()
        var known = -1
        for n in nodes:
            if n in memo:
                known = n
                break
        if known >= 0:
            for n in nodes:
                if n not in memo:
                    try:
                        memo[n] = memo[known].copy()
                    except:
                        pass
        var left = List[Int]()
        var right = List[Int]()
        for n in nodes:
            left.append(self._nodes[n].left)
            right.append(self._nodes[n].right)
        if left[0] >= 0:
            stack.append(left^)
        if right[0] >= 0:
            stack.append(right^)
