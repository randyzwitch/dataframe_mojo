"""Metadata execution for shared accelerator DAGs without repeated subtrees."""
from dataframe.frame import DataFrame
from .code_domains import TextField
from dataframe.expr import Expr, col
from dataframe.lazy import (
    LazyFrame,
    SCAN_FRAME,
    SELECT,
    FILTER,
    SORT,
    SLICE,
    UNIQUE,
    WITH_COLUMNS,
    DROP,
    JOIN,
    ASOF,
    EXPLODE,
    UNNEST,
    CONCAT,
    UNPIVOT,
    _is_scan,
    _is_time_window,
    _time_window_reads,
    _plan_node,
    _references,
    _row_order_step,
)


def schema_frame(query: LazyFrame, index: Int) raises -> DataFrame:
    """Match CPU empty execution, evaluating each reachable node once.

    Sources execute in empty mode. Intermediate results retain their rows:
    an empty input's scalar reduction can produce one row, which subsequent
    joins and expressions must observe. Original ancestors remain in place
    until evaluated, preserving the CPU join-order decision. Cached inputs
    use fresh frame scans without source counters or execution reports.
    """
    var live = List[Bool](length=len(query._nodes), fill=False)
    var pending = List[Int]([index])
    var order = List[Int]()
    while len(pending):
        var current = pending.pop()
        if current < 0:
            order.append(-current - 1)
            continue
        if live[current]:
            continue
        live[current] = True
        pending.append(-current - 1)
        # Preserve the CPU left-before-right error observation order.
        for child in [query._nodes[current].right, query._nodes[current].left]:
            if child >= 0:
                pending.append(child)
    var staged = query.copy()
    staged._report = None
    for current in order:
        # All children already name cached metadata frames. Non-scan nodes
        # execute normally so those frames are not cleared a second time.
        var frame = staged._execute(
            current, _is_scan(query._nodes[current].kind)
        )
        staged._frames.append(frame^)
        staged._nodes[current] = _plan_node(
            SCAN_FRAME, offset=len(staged._frames) - 1
        )
    return staged._frames[len(staged._frames) - 1].copy()


def _projection_join_inputs(
    query: LazyFrame, index: Int, wanted: List[String]
) raises -> Optional[List[List[String]]]:
    # Reuse the CPU's key/coalescing/suffix rules on cached child schemas.
    # Its _columns_of probes then touch only these fresh metadata scans.
    var staged = query.copy()
    for child in [query._nodes[index].left, query._nodes[index].right]:
        staged._frames.append(schema_frame(query, child))
        staged._nodes[child] = _plan_node(
            SCAN_FRAME, offset=len(staged._frames) - 1
        )
    return staged._join_input_columns(index, wanted)


def prune_projections(mut query: LazyFrame) raises:
    """Scans read only columns that some operator above them uses.

    A join passes each input only its keys and the columns the plan
    above reads from that side, so unused payload columns are never
    read or gathered. A join input that is not a scan gets a plain
    column selection above it.
    """
    var needed = List[Optional[List[String]]](
        length=len(query._nodes), fill=Optional[List[String]]()
    )
    var all_needed = List[Bool](length=len(query._nodes), fill=False)
    all_needed[len(query._nodes) - 1] = True
    # (join index, right side?, columns) for join inputs to narrow.
    var narrow_joins = List[Int]()
    var narrow_sides = List[Bool]()
    var narrow_columns = List[List[String]]()
    for reverse in range(len(query._nodes)):
        var i = len(query._nodes) - 1 - reverse
        ref node = query._nodes[i]
        if not all_needed[i] and not needed[i]:
            continue
        if node.kind == JOIN and not all_needed[i] and needed[i]:
            var sides = _projection_join_inputs(query, i, needed[i].value())
            if sides:
                for side in range(2):
                    var child = node.right if side == 1 else node.left
                    ref keep = sides.value()[side]
                    needed[child] = keep.copy()
                    narrow_joins.append(i)
                    narrow_sides.append(side == 1)
                    narrow_columns.append(keep.copy())
                continue
        if _is_scan(node.kind):
            if not all_needed[i] and needed[i]:
                var columns = schema_frame(query, i).columns()
                var keep = List[String]()
                for c in columns:
                    for want in needed[i].value():
                        if want == c:
                            keep.append(c)
                            break
                if len(keep) > 0 and len(keep) < len(columns):
                    query._nodes[i].names = keep^
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
        )
        if node.kind == UNNEST:
            reads.append(node.text)
        if node.kind == UNIQUE and len(node.names) == 0:
            reads_all = True
        if node.kind == UNPIVOT:
            # Accelerators do not run unpivot; read every input column.
            reads_all = True
        if _is_time_window(node.kind):
            # Only what the windows read, whatever is wanted above: their
            # outputs are new columns.
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
            if reads_all or node.kind == JOIN or node.kind == ASOF:
                all_needed[child] = True
            elif needed[child]:
                var merged = needed[child].value().copy()
                for n in reads:
                    merged.append(n)
                needed[child] = merged^
            else:
                needed[child] = reads.copy()
    _narrow_projection_inputs(query, narrow_joins, narrow_sides, narrow_columns)


def _narrow_projection_inputs(
    mut query: LazyFrame,
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
    for k in range(len(joins)):
        var join = joins[k]
        var child = query._nodes[join].left
        if right_sides[k]:
            child = query._nodes[join].right
        if _is_scan(query._nodes[child].kind):
            continue
        var produced = schema_frame(query, child).columns()
        if len(columns[k]) >= len(produced):
            continue
        var exprs = List[Expr]()
        for name in columns[k]:
            exprs.append(col(name))
        # This projection may be inserted AFTER dictionary recoding. Preserve
        # logical text lineage for selected fields; UInt32 alone cannot recover
        # which columns use a dynamic dictionary or its original hash family.
        var fields = List[TextField]()
        for field in query._nodes[child].text_fields:
            if field.name in columns[k]:
                fields.append(field.copy())
        query._nodes.append(
            _plan_node(
                SELECT,
                child,
                exprs=exprs,
                code_domain=query._nodes[child].code_domain,
                text_fields=fields^,
            )
        )
        if right_sides[k]:
            query._nodes[join].right = len(query._nodes) - 1
        else:
            query._nodes[join].left = len(query._nodes) - 1
        added = True
    if added:
        query._reorder()
