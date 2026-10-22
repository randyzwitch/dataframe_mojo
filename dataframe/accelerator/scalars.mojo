"""Materialize scalar-only projections as one-row resident expressions."""
from std.collections import Dict
from dataframe.expr import (
    COL,
    SELECTOR,
    OVER,
    Expr,
    col,
    lit,
    is_reduction,
    is_window,
)
from dataframe.lazy import LazyFrame, PlanNode, SELECT, WITH_COLUMNS, _plan_node
from .aggregates import _in_memory_sources
from .selectors import _live_nodes
from .metadata import schema_frame


def _scalar_projection(operation: PlanNode) -> Bool:
    if operation.kind != SELECT or not len(operation.exprs):
        return False
    for expression in operation.exprs:
        if not len(expression._nodes):
            return False
        for node in expression._nodes:
            if (
                node.op == COL
                or node.op == SELECTOR
                or node.op == OVER
                or is_reduction(node.op)
                or is_window(node.op)
            ):
                return False
    return True


def has_scalar_projections(query: LazyFrame) -> Bool:
    var live = _live_nodes(query)
    for index in range(len(live)):
        if live[index] and _scalar_projection(query._nodes[index]):
            return True
    return False


def lower_scalar_projections(query: LazyFrame) raises -> LazyFrame:
    """Count the resident prefix, then broadcast constants over its one row.

    Count every source column to retain producer observation checks. Counts
    remain private and are removed by the final projection. No source value
    is read by normalization, including when the source has no rows or columns.
    """
    _in_memory_sources(query)
    var result = query.copy()
    result._nodes = List[PlanNode]()
    var roots = List[Int]()
    var live = _live_nodes(query)
    var names = Dict[String, Bool]()
    for frame in query._frames:
        for name in frame.columns():
            names[name] = True
    for operation in query._nodes:
        for name in operation.names:
            names[name] = True
        for expression in operation.exprs:
            names[expression._name] = True
    var serial = 0
    for index in range(len(query._nodes)):
        var operation = query._nodes[index].copy()
        if operation.left >= 0:
            operation.left = roots[operation.left]
        if operation.right >= 0:
            operation.right = roots[operation.right]
        if not live[index] or not _scalar_projection(operation):
            result._nodes.append(operation^)
            roots.append(len(result._nodes) - 1)
            continue
        # Validate original aliases, dtypes and literal semantics first.
        var expected = schema_frame(query, index)
        var schema = schema_frame(result, operation.left)
        if not schema.width():
            var input = "__metal_scalar_input"
            while input in names:
                input += "_"
            names[input] = True
            result._nodes.append(
                _plan_node(
                    WITH_COLUMNS,
                    operation.left,
                    exprs=[lit(Int8(0)).alias(input)],
                )
            )
            operation.left = len(result._nodes) - 1
            schema = schema_frame(result, operation.left)
        var counts = List[Expr]()
        for column in schema._columns:
            var name = "__metal_scalar_count_" + String(serial)
            serial += 1
            while name in names:
                name += "_"
            names[name] = True
            counts.append(col(column.name()).count().alias(name))
        result._nodes.append(_plan_node(SELECT, operation.left, exprs=counts))
        result._nodes.append(
            _plan_node(
                WITH_COLUMNS, len(result._nodes) - 1, exprs=operation.exprs
            )
        )
        var projected = List[Expr]()
        for name in expected.columns():
            projected.append(col(name))
        result._nodes.append(
            _plan_node(SELECT, len(result._nodes) - 1, exprs=projected)
        )
        roots.append(len(result._nodes) - 1)
    return result^
