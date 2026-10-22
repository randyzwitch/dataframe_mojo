"""Resolve CPU selectors against metadata before accelerator normalization."""
from dataframe.expr import SELECTOR, Expr, col, lit
from dataframe.lazy import (
    LazyFrame,
    PlanNode,
    SELECT,
    WITH_COLUMNS,
    AGG,
    FILTER,
    DROP,
    _plan_node,
)
from dataframe.selectors import expand_all
from .aggregates import _in_memory_sources
from .metadata import schema_frame


def _live_nodes(query: LazyFrame) -> List[Bool]:
    var live = List[Bool](length=len(query._nodes), fill=False)
    var pending = List[Int]([len(query._nodes) - 1])
    while len(pending):
        var index = pending.pop()
        if index < 0 or index >= len(live) or live[index]:
            continue
        live[index] = True
        ref operation = query._nodes[index]
        if operation.left >= 0:
            pending.append(operation.left)
        if operation.right >= 0:
            pending.append(operation.right)
    return live^


def has_selectors(query: LazyFrame) -> Bool:
    var live = _live_nodes(query)
    for index in range(len(live)):
        if not live[index]:
            continue
        ref operation = query._nodes[index]
        for expression in operation.exprs:
            for node in expression._nodes:
                if node.op == SELECTOR:
                    return True
    return False


def lower_selectors(query: LazyFrame) raises -> LazyFrame:
    """Expand names, dtypes, exclusions and positions using the CPU API.

    Expansion reads only schemas. Later lowering binds every expanded producer
    and retains its normal observation, precision and capability checks.
    """
    _in_memory_sources(query)
    var result = query.copy()
    result._nodes = List[PlanNode]()
    var roots = List[Int]()
    var live = _live_nodes(query)
    for index in range(len(query._nodes)):
        var operation = query._nodes[index].copy()
        if operation.left >= 0:
            operation.left = roots[operation.left]
        if operation.right >= 0:
            operation.right = roots[operation.right]
        if not live[index]:
            result._nodes.append(operation^)
            roots.append(len(result._nodes) - 1)
            continue
        var selected = False
        for expression in operation.exprs:
            for node in expression._nodes:
                selected |= node.op == SELECTOR
        if not selected:
            result._nodes.append(operation^)
            roots.append(len(result._nodes) - 1)
            continue
        if (
            operation.kind != SELECT
            and operation.kind != WITH_COLUMNS
            and operation.kind != AGG
            and operation.kind != FILTER
        ):
            raise Error("Metal unsupported [selector]: expression context")
        var schema = schema_frame(result, operation.left)
        var expanded = expand_all(operation.exprs, schema._columns)
        if operation.kind == FILTER and len(expanded) != 1:
            raise Error("A filter selector must match exactly one column")
        if operation.kind == SELECT and not len(expanded):
            # CPU select_exprs distinguishes a nonempty expression list whose
            # selectors match nothing (one row) from explicit select([]), which
            # preserves the input height. Count every input then drop the
            # counters to retain the scalar shape and all producer references.
            if not schema.width():
                # A zero-column frame still has rows. Broadcast a private
                # counter input so the reduction produces one row even when
                # that input has zero rows, then remove the counter as usual.
                var input = "__metal_empty_selection_input"
                result._nodes.append(
                    _plan_node(
                        WITH_COLUMNS,
                        operation.left,
                        exprs=[lit(Int8(0)).alias(input)],
                    )
                )
                operation.left = len(result._nodes) - 1
                schema = schema_frame(result, operation.left)
            var names = List[String]()
            for column in schema._columns:
                var name = "__metal_empty_selection_" + String(len(names))
                names.append(name)
                expanded.append(col(column.name()).count().alias(name))
            operation.exprs = expanded^
            result._nodes.append(operation^)
            result._nodes.append(
                _plan_node(DROP, len(result._nodes) - 1, names=names)
            )
            roots.append(len(result._nodes) - 1)
            continue
        operation.exprs = expanded^
        result._nodes.append(operation^)
        roots.append(len(result._nodes) - 1)
    return result^
