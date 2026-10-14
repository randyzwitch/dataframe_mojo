"""Scalar mode through resident grouping, maximum count, filter and sort.

Grouped and nested distributions require typed list owners and are not
rewritten by this scalar normalization.
"""
from std.collections import Dict
from dataframe.binding import bind, ROWS, SCALAR
from dataframe.dtype import DataType
from dataframe.expr import COL, MODE, VALUE_COUNTS, Expr, col, null, subtree
from dataframe.lazy import (
    LazyFrame,
    PlanNode,
    SELECT,
    WITH_COLUMNS,
    AGG,
    JOIN,
    FILTER,
    SORT,
    _plan_node,
)
from .aggregates import _in_memory_sources
from .metadata import schema_frame
from .selectors import _live_nodes


def _mode(expression: Expr) -> Bool:
    return (
        len(expression._nodes) > 0
        and expression._nodes[len(expression._nodes) - 1].op == MODE
    )


def has_scalar_modes(query: LazyFrame) -> Bool:
    var live = _live_nodes(query)
    for index in range(len(query._nodes)):
        if not live[index]:
            continue
        ref operation = query._nodes[index]
        if operation.kind == SELECT:
            for expression in operation.exprs:
                if _mode(expression):
                    return True
    return False


def _private(
    mut names: Dict[String, Bool], label: String, serial: Int
) -> String:
    var name = "__dataframe_mode_" + label + "_" + String(serial)
    while name in names:
        name += "_"
    names[name] = True
    return name^


def lower_scalar_modes(query: LazyFrame) raises -> LazyFrame:
    """Count nulls as a value, keep first equal representations, sort winners.

    Every value decision and the variable output height are computed by GPU
    operators. Host normalization binds only empty schemas. Scalar siblings
    join their one-row results onto the winners, including zero empty winners.
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
    var live = _live_nodes(query)
    result._nodes = List[PlanNode]()
    var roots = List[Int]()
    var serial = 0
    for index in range(len(query._nodes)):
        var operation = query._nodes[index].copy()
        if operation.left >= 0:
            operation.left = roots[operation.left]
        if operation.right >= 0:
            operation.right = roots[operation.right]
        var chosen = -1
        if live[index] and operation.kind == SELECT:
            for e in range(len(operation.exprs)):
                if _mode(operation.exprs[e]):
                    if chosen >= 0:
                        raise Error(
                            "Metal unsupported [mode]: multiple variable-height"
                            " distributions require ordinal alignment"
                        )
                    chosen = e
        if chosen < 0:
            result._nodes.append(operation^)
            roots.append(len(result._nodes) - 1)
            continue
        for expression in operation.exprs:
            if expression._nodes[len(expression._nodes) - 1].op == VALUE_COUNTS:
                raise Error(
                    "Metal unsupported [mode]: multiple variable-height"
                    " distributions require ordinal alignment"
                )
        var schema = schema_frame(query, query._nodes[index].left)
        var bound = bind(operation.exprs[chosen], schema._columns)
        ref top = bound.expr._nodes[len(bound.expr._nodes) - 1]
        if bound.shapes[top.left] != ROWS and bound.shapes[top.left] != SCALAR:
            raise Error("Metal unsupported [mode]: nested reduction operand")
        var value_name = _private(names, "value", serial)
        var count_name = _private(names, "count", serial)
        var best_name = _private(names, "best", serial)
        serial += 1
        var input_columns = List[Expr]()
        for name in schema.columns():
            input_columns.append(col(name))
        result._nodes.append(
            _plan_node(
                SELECT,
                operation.left,
                exprs=input_columns,
                text="metal_shared_prefix",
            )
        )
        var prefix = len(result._nodes) - 1
        var operand = subtree(bound.expr, top.left)
        if bound.dtypes[top.left] == DataType.FLOAT32:
            # CPU histogram states widen then narrow Float32. This quiets a
            # valid signaling NaN while retaining its first payload and sign.
            operand = operand.cast(DataType.FLOAT64).cast(DataType.FLOAT32)
        if len(operand._nodes) == 1 and operand._nodes[0].op == COL:
            for column in schema._columns:
                if (
                    column.name() == operand._nodes[0].text
                    and column.dtype().is_categorical()
                ):
                    # CPU expressions decode categorical values. This typed
                    # null identity keeps String output metadata while ordered
                    # dictionary codes remain the device representation.
                    operand = operand.fill_null(null(DataType.STRING))
        result._nodes.append(
            _plan_node(
                WITH_COLUMNS,
                prefix,
                exprs=[operand.alias(value_name)],
            )
        )
        result._nodes.append(
            _plan_node(SELECT, len(result._nodes) - 1, exprs=[col(value_name)])
        )
        result._nodes.append(
            _plan_node(
                AGG,
                len(result._nodes) - 1,
                names=[value_name],
                exprs=[col(value_name).len().alias(count_name)],
            )
        )
        var counts = len(result._nodes) - 1
        result._nodes.append(
            _plan_node(
                SELECT, counts, exprs=[col(count_name).max().alias(best_name)]
            )
        )
        result._nodes.append(
            _plan_node(
                JOIN,
                counts,
                len(result._nodes) - 1,
                names2=["_right"],
                text="cross",
                how=6,
            )
        )
        result._nodes.append(
            _plan_node(
                FILTER,
                len(result._nodes) - 1,
                exprs=[col(count_name) == col(best_name)],
            )
        )
        result._nodes.append(
            _plan_node(
                SORT,
                len(result._nodes) - 1,
                names=[value_name],
                flags=[False, True],
            )
        )
        var winners = len(result._nodes) - 1
        var projected = List[Expr]()
        var siblings = List[Expr]()
        for e in range(len(operation.exprs)):
            if e == chosen:
                projected.append(
                    col(value_name).alias(operation.exprs[e]._name)
                )
            else:
                var name = _private(names, "sibling", serial)
                serial += 1
                siblings.append(operation.exprs[e].alias(name))
                projected.append(col(name).alias(operation.exprs[e]._name))
        if len(siblings):
            result._nodes.append(_plan_node(SELECT, prefix, exprs=siblings))
            result._nodes.append(
                _plan_node(
                    JOIN,
                    winners,
                    len(result._nodes) - 1,
                    names2=["_right"],
                    text="cross",
                    how=6,
                )
            )
            winners = len(result._nodes) - 1
        result._nodes.append(
            _plan_node(SELECT, winners, exprs=projected, text=operation.text)
        )
        roots.append(len(result._nodes) - 1)
    var actual = schema_frame(result, len(result._nodes) - 1)
    if actual.columns() != expected.columns():
        raise Error("Metal unsupported [mode]: output names differ")
    for i in range(actual.width()):
        if actual._columns[i].dtype() != expected._columns[i].dtype():
            raise Error("Metal unsupported [mode]: output dtypes differ")
    return result^
