"""Partition aggregates materialized and broadcast entirely on the GPU."""
from std.collections import Dict
from dataframe.binding import bind, ROWS
from dataframe.expr import (
    Expr,
    Node,
    COL,
    OVER,
    SELECTOR,
    col,
    is_reduction,
    is_window,
    subtree,
    split_names,
    _node,
)
from dataframe.lazy import (
    LazyFrame,
    PlanNode,
    SELECT,
    FILTER,
    WITH_COLUMNS,
    AGG,
    JOIN,
    DROP,
    _plan_node,
)
from .aggregates import _in_memory_sources


def _aggregate_over(expression: Expr, index: Int) -> Bool:
    if expression._nodes[index].op != OVER:
        return False
    var inner = subtree(expression, expression._nodes[index].left)
    for node in inner._nodes:
        if is_reduction(node.op):
            return True
    return False


def has_partition_aggregates(query: LazyFrame) -> Bool:
    for operation in query._nodes:
        for expression in operation.exprs:
            for i in range(len(expression._nodes)):
                if _aggregate_over(expression, i):
                    return True
    return False


def lower_partition_aggregates(query: LazyFrame) raises -> LazyFrame:
    """Group a shared resident prefix, then align aggregates to its row order.

    CPU evaluate() prepares every OVER state before evaluating its consumer.
    Only these internal producers are marked as observable materializations;
    checked expressions elsewhere retain the normal child-region restrictions.
    """
    _in_memory_sources(query)
    var expected = query._execute(len(query._nodes) - 1, True)
    var names = Dict[String, Bool]()
    for frame in query._frames:
        for name in frame.columns():
            names[name] = True
    for operation in query._nodes:
        for name in operation.names:
            names[name] = True
        for expression in operation.exprs:
            names[expression._name] = True
            for node in expression._nodes:
                if node.op == COL:
                    names[node.text] = True
    var result = query.copy()
    result._nodes = List[PlanNode]()
    var roots = List[Int]()
    var serial = 0
    for index in range(len(query._nodes)):
        var operation = query._nodes[index].copy()
        if operation.left >= 0:
            operation.left = roots[operation.left]
        if operation.right >= 0:
            operation.right = roots[operation.right]
        var changed = False
        for expression in operation.exprs:
            for i in range(len(expression._nodes)):
                changed |= _aggregate_over(expression, i)
        if not changed:
            result._nodes.append(operation^)
            roots.append(len(result._nodes) - 1)
            continue
        if (
            operation.kind != SELECT
            and operation.kind != FILTER
            and operation.kind != WITH_COLUMNS
        ):
            raise Error(
                "Metal unsupported [window]: partition aggregates require a row consumer"
            )
        var schema = query._execute(query._nodes[index].left, True)
        var columns = List[Expr]()
        for name in schema.columns():
            columns.append(col(name))
        result._nodes.append(_plan_node(SELECT, operation.left, exprs=columns))
        var prefix = len(result._nodes) - 1
        var aligned = prefix
        var temporary = List[String]()
        var projected = List[Expr]()
        for expression in operation.exprs:
            var bound = bind(expression, schema._columns)
            var positions = List[Int](length=len(bound.expr._nodes), fill=-1)
            var rewritten = List[Node]()
            for n in range(len(bound.expr._nodes)):
                var node = bound.expr._nodes[n].copy()
                positions[n] = len(rewritten)
                if _aggregate_over(bound.expr, n):
                    var inner = subtree(bound.expr, node.left)
                    for child in inner._nodes:
                        if (
                            child.op == OVER
                            or is_window(child.op)
                            or child.op == SELECTOR
                        ):
                            raise Error(
                                "Metal unsupported [window]: nested windows and selectors"
                            )
                    var inner_bound = bind(inner, schema._columns)
                    if inner_bound.shape() == ROWS:
                        raise Error(
                            "Metal unsupported [window]: row-valued partition expression"
                        )
                    var name = "__dataframe_metal_window_" + String(serial)
                    serial += 1
                    while name in names:
                        name += "_"
                    names[name] = True
                    temporary.append(name)
                    var keys = split_names(node.text2)
                    result._nodes.append(
                        _plan_node(
                            AGG,
                            prefix,
                            exprs=[inner.alias(name)],
                            names=keys,
                            maintain_order=True,
                            text="metal_window_materialization",
                        )
                    )
                    var grouped = len(result._nodes) - 1
                    result._nodes.append(
                        _plan_node(
                            JOIN,
                            aligned,
                            grouped,
                            names=keys,
                            right_keys=keys,
                            names2=["_right"],
                            how=1,
                            text="left",
                            flags=[True],
                        )
                    )
                    aligned = len(result._nodes) - 1
                    rewritten.append(_node(COL, text=name))
                else:
                    if node.left >= 0:
                        node.left = positions[node.left]
                    if node.right >= 0:
                        node.right = positions[node.right]
                    if node.extra >= 0:
                        node.extra = positions[node.extra]
                    rewritten.append(node^)
            var rewritten_expression = Expr(rewritten^, expression._name)
            projected.append(
                subtree(
                    rewritten_expression, len(rewritten_expression._nodes) - 1
                )
            )
        operation.left = aligned
        operation.exprs = projected^
        var kind = operation.kind
        result._nodes.append(operation^)
        if kind == FILTER or kind == WITH_COLUMNS:
            result._nodes.append(
                _plan_node(DROP, len(result._nodes) - 1, names=temporary)
            )
        roots.append(len(result._nodes) - 1)
    var actual = result._execute(len(result._nodes) - 1, True)
    if actual.columns() != expected.columns():
        raise Error("Metal unsupported [window]: output names differ")
    for i in range(actual.width()):
        if actual._columns[i].dtype() != expected._columns[i].dtype():
            raise Error("Metal unsupported [window]: output dtypes differ")
    return result^
