"""Distinct counts through resident deduplication and row counting."""
from .metadata import schema_frame
from std.collections import Dict
from dataframe.binding import bind
from dataframe.expr import Expr, COL, N_UNIQUE, col, subtree
from dataframe.lazy import (
    LazyFrame,
    PlanNode,
    SELECT,
    AGG,
    UNIQUE,
    JOIN,
    _plan_node,
)
from .aggregates import _in_memory_sources


def _eligible(operation: PlanNode) -> Bool:
    if operation.kind != SELECT and operation.kind != AGG:
        return False
    for expression in operation.exprs:
        if (
            len(expression._nodes) > 0
            and expression._nodes[len(expression._nodes) - 1].op == N_UNIQUE
        ):
            return True
    return False


def has_distinct_counts(query: LazyFrame) -> Bool:
    for operation in query._nodes:
        if _eligible(operation):
            return True
    return False


def lower_distinct_counts(query: LazyFrame) raises -> LazyFrame:
    """Count distinct expressions, including their one null category.

    Input expressions keep the ordinary nonterminal checked-error restrictions.
    Aggregate siblings use independent branches of a shared resident prefix.
    No source rows are evaluated while constructing these native GPU operators.
    """
    _in_memory_sources(query)
    var expected = schema_frame(query, len(query._nodes) - 1)
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
    var split = False
    var serial = 0
    for index in range(len(query._nodes)):
        var operation = query._nodes[index].copy()
        if operation.left >= 0:
            operation.left = roots[operation.left]
        if operation.right >= 0:
            operation.right = roots[operation.right]
        if not _eligible(operation):
            result._nodes.append(operation^)
            roots.append(len(result._nodes) - 1)
            continue
        var schema = schema_frame(query, query._nodes[index].left)
        if len(operation.exprs) > 1:
            split = True
            var required = Dict[String, Bool]()
            for key in operation.names:
                required[key] = True
            for expression in operation.exprs:
                var bound = bind(expression, schema._columns)
                for node in bound.expr._nodes:
                    if node.op == COL:
                        required[node.text] = True
            var columns = List[Expr]()
            for name in schema.columns():
                if name in required:
                    columns.append(col(name))
            result._nodes.append(
                _plan_node(SELECT, operation.left, exprs=columns)
            )
            var prefix = len(result._nodes) - 1
            var aligned = -1
            var projected = List[Expr]()
            for key in operation.names:
                projected.append(col(key))
            for expression in operation.exprs:
                var name = "__dataframe_metal_distinct_sibling_" + String(
                    serial
                )
                serial += 1
                while name in names:
                    name += "_"
                names[name] = True
                var leaf = operation.copy()
                leaf.left = prefix
                leaf.exprs = [expression.alias(name)]
                result._nodes.append(leaf^)
                var branch = len(result._nodes) - 1
                if aligned < 0:
                    aligned = branch
                elif operation.kind == AGG:
                    result._nodes.append(
                        _plan_node(
                            JOIN,
                            aligned,
                            branch,
                            names=operation.names.copy(),
                            right_keys=operation.names.copy(),
                            names2=["_right"],
                            text="left",
                            how=1,
                            flags=[True],
                        )
                    )
                    aligned = len(result._nodes) - 1
                else:
                    result._nodes.append(
                        _plan_node(
                            JOIN,
                            aligned,
                            branch,
                            names2=["_right"],
                            text="cross",
                            how=6,
                        )
                    )
                    aligned = len(result._nodes) - 1
                projected.append(col(name).alias(expression._name))
            result._nodes.append(
                _plan_node(
                    SELECT, aligned, exprs=projected, text=operation.text
                )
            )
            roots.append(len(result._nodes) - 1)
            continue
        var bound = bind(operation.exprs[0], schema._columns)
        var name = "__dataframe_metal_distinct_value"
        while name in names:
            name += "_"
        names[name] = True
        var projected = List[Expr]()
        var keys = operation.names.copy()
        for key in keys:
            projected.append(col(key))
        ref top = bound.expr._nodes[len(bound.expr._nodes) - 1]
        projected.append(subtree(bound.expr, top.left).alias(name))
        result._nodes.append(
            _plan_node(SELECT, operation.left, exprs=projected)
        )
        keys.append(name)
        result._nodes.append(
            _plan_node(
                UNIQUE,
                len(result._nodes) - 1,
                names=keys,
                text="first",
                maintain_order=True,
            )
        )
        operation.left = len(result._nodes) - 1
        operation.exprs = [col(name).len().alias(operation.exprs[0]._name)]
        result._nodes.append(operation^)
        roots.append(len(result._nodes) - 1)
    var actual = schema_frame(result, len(result._nodes) - 1)
    if actual.columns() != expected.columns():
        raise Error("Metal unsupported [distinct count]: output names differ")
    for i in range(actual.width()):
        if actual._columns[i].dtype() != expected._columns[i].dtype():
            raise Error(
                "Metal unsupported [distinct count]: output dtypes differ"
            )
    if split:
        return lower_distinct_counts(result)
    return result^
