"""Scalar distributions through resident grouping, filtering and typed sorting.

Grouped and nested distributions require typed list owners and are not
rewritten by this scalar normalization.
"""
from std.collections import Dict
from dataframe.binding import bind, ROWS
from dataframe.dtype import DataType
from dataframe.frame import DataFrame
from dataframe.expr import (
    COL,
    MODE,
    VALUE_COUNTS,
    as_struct,
    split_names,
    Expr,
    col,
    lit,
    null,
    subtree,
    when,
)
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


def _distribution(expression: Expr) -> Bool:
    return len(expression._nodes) > 0 and (
        expression._nodes[len(expression._nodes) - 1].op == MODE
        or expression._nodes[len(expression._nodes) - 1].op == VALUE_COUNTS
    )


def has_scalar_modes(query: LazyFrame) -> Bool:
    """Detect top-level scalar MODE or VALUE_COUNTS projections."""
    var live = _live_nodes(query)
    for index in range(len(query._nodes)):
        if not live[index]:
            continue
        ref operation = query._nodes[index]
        if operation.kind == SELECT:
            for expression in operation.exprs:
                if _distribution(expression):
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


def _distribution_branch(
    mut result: LazyFrame,
    prefix: Int,
    schema: DataFrame,
    expression: Expr,
    mut names: Dict[String, Bool],
    serial: Int,
    ordinal: Bool,
) raises -> Tuple[Int, String, String, String]:
    var bound = bind(expression, schema._columns)
    ref top = bound.expr._nodes[len(bound.expr._nodes) - 1]
    if bound.shapes[top.left] != ROWS:
        raise Error("Metal unsupported [mode]: nested reduction operand")
    var value_name = _private(names, "value", serial)
    var count_name = _private(names, "count", serial)
    var best_name = _private(names, "best", serial)
    var operand = subtree(bound.expr, top.left)
    if len(operand._nodes) == 1 and operand._nodes[0].op == COL:
        for column in schema._columns:
            if (
                column.name() == operand._nodes[0].text
                and column.dtype().is_categorical()
            ):
                # A typed null identity records CPU categorical decoding while
                # the device continues to transport ordered dictionary codes.
                operand = operand.fill_null(null(DataType.STRING))
    if bound.dtypes[top.left] == DataType.FLOAT32:
        # CPU canonical histograms widen Float32 and narrow their selected
        # representatives, quieting signaling NaNs while retaining payloads.
        operand = operand.cast(DataType.FLOAT64).cast(DataType.FLOAT32)
    result._nodes.append(
        _plan_node(
            WITH_COLUMNS,
            prefix,
            exprs=[operand.alias(value_name)],
            text="metal_distribution_materialization",
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
    var count_output = String()
    if top.op == MODE:
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
    else:
        count_output = _private(names, "frequency", serial)
        var normalize = (top.integer & 2) != 0
        if normalize:
            # Row counts and every partial total fit Int32.MAX, hence Float64
            # sums are exact and need no checked integer aggregate boundary.
            var total_name = _private(names, "total", serial)
            result._nodes.append(
                _plan_node(
                    SELECT,
                    counts,
                    exprs=[
                        col(count_name)
                        .cast(DataType.FLOAT64)
                        .sum()
                        .alias(total_name)
                    ],
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
                    WITH_COLUMNS,
                    len(result._nodes) - 1,
                    exprs=[
                        (
                            col(count_name).cast(DataType.FLOAT64)
                            / col(total_name)
                        ).alias(count_output)
                    ],
                )
            )
        else:
            # Counts are nonnegative and bounded by the provider's Int32 row
            # limit. An unchecked transport cast avoids introducing a checked
            # user expression at this internal nonterminal boundary.
            result._nodes.append(
                _plan_node(
                    WITH_COLUMNS,
                    counts,
                    exprs=[
                        col(count_name)
                        .cast(DataType.UINT32, strict=False)
                        .alias(count_output)
                    ],
                )
            )
    if top.op == VALUE_COUNTS and (top.integer & 1) != 0:
        result._nodes.append(
            _plan_node(
                SORT,
                len(result._nodes) - 1,
                names=[count_name, value_name],
                flags=[True, False, False, True],
            )
        )
    else:
        result._nodes.append(
            _plan_node(
                SORT,
                len(result._nodes) - 1,
                names=[value_name],
                flags=[False, True],
            )
        )
    var ordinal_name = String()
    if ordinal:
        ordinal_name = _private(names, "ordinal", serial)
        # ROWS-shaped valid ones count every winner, including a null winner.
        # cum_count(value) would give that null the preceding ordinal, while
        # cum_sum(literal) is rejected because window inputs must be row-valued.
        result._nodes.append(
            _plan_node(
                WITH_COLUMNS,
                len(result._nodes) - 1,
                exprs=[
                    when(col(value_name).is_null())
                    .then(lit(Int64(1)))
                    .otherwise(lit(Int64(1)))
                    .cum_sum()
                    .alias(ordinal_name)
                ],
            )
        )
    var fields = List[Expr]([col(value_name)])
    if ordinal_name:
        fields.append(col(ordinal_name))
    if count_output:
        fields.append(col(count_output))
    result._nodes.append(
        _plan_node(
            SELECT,
            len(result._nodes) - 1,
            exprs=fields,
            text="metal_distribution_materialization",
        )
    )
    return (len(result._nodes) - 1, value_name^, ordinal_name^, count_output^)


def lower_scalar_modes(query: LazyFrame) raises -> LazyFrame:
    """Count values on device, sort winners, and zip equal-length distributions.

    Host normalization binds only empty schemas. Resident ZIP tasks observe
    each child before comparing actual cardinalities, including zero results.
    Ordinary scalar siblings cross-broadcast over the aligned winners.
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
        var choices = List[Int]()
        if live[index] and operation.kind == SELECT:
            for e in range(len(operation.exprs)):
                if _distribution(operation.exprs[e]):
                    choices.append(e)
        if not len(choices):
            result._nodes.append(operation^)
            roots.append(len(result._nodes) - 1)
            continue
        for choice in choices:
            if (
                operation.exprs[choice]
                ._nodes[len(operation.exprs[choice]._nodes) - 1]
                .op
                == VALUE_COUNTS
                and index != len(query._nodes) - 1
            ):
                raise Error(
                    "Metal unsupported [distribution]: intermediate value_counts struct consumers"
                )
        var schema = schema_frame(query, query._nodes[index].left)
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
        var producers = List[Int](length=len(operation.exprs), fill=-1)
        var mode_names = List[String](length=len(operation.exprs), fill="")
        var count_names = List[String](length=len(operation.exprs), fill="")
        var ordinal_names = List[String](length=len(operation.exprs), fill="")
        var scalar_names = List[String](length=len(operation.exprs), fill="")
        # Each original expression completes before the next one is observed.
        # Scalar identities are wrapped so scalar-projection normalization
        # cannot erase this producer's explicit materialization boundary.
        for e in range(len(operation.exprs)):
            if _distribution(operation.exprs[e]):
                var branch = _distribution_branch(
                    result,
                    prefix,
                    schema,
                    operation.exprs[e],
                    names,
                    serial,
                    len(choices) > 1,
                )
                producers[e] = branch[0]
                mode_names[e] = branch[1]
                ordinal_names[e] = branch[2]
                count_names[e] = branch[3]
            else:
                var name = _private(names, "sibling", serial)
                scalar_names[e] = name
                result._nodes.append(
                    _plan_node(
                        SELECT,
                        prefix,
                        exprs=[operation.exprs[e].alias(name)],
                        text="metal_distribution_materialization",
                    )
                )
                result._nodes.append(
                    _plan_node(
                        SELECT,
                        len(result._nodes) - 1,
                        exprs=[col(name)],
                        text="metal_distribution_materialization",
                    )
                )
                producers[e] = len(result._nodes) - 1
            serial += 1
        var control = -1
        var control_name = String()
        for e in range(len(producers)):
            var field = mode_names[e] if mode_names[e] else scalar_names[e]
            var next_name = _private(names, "observe", serial)
            serial += 1
            result._nodes.append(
                _plan_node(
                    SELECT,
                    producers[e],
                    exprs=[col(field).len().alias(next_name)],
                )
            )
            var next_control = len(result._nodes) - 1
            if control >= 0:
                result._nodes.append(
                    _plan_node(
                        JOIN,
                        control,
                        next_control,
                        names2=["_right"],
                        text="cross",
                        how=6,
                    )
                )
                # One-row controls never fan out. Compact the control schema
                # to one field so many siblings cannot inflate join width.
                result._nodes.append(
                    _plan_node(
                        SELECT,
                        len(result._nodes) - 1,
                        exprs=[col(control_name).len().alias(next_name)],
                    )
                )
                next_control = len(result._nodes) - 1
            control = next_control
            control_name = next_name
        var first = choices[0]
        result._nodes.append(
            _plan_node(
                JOIN,
                control,
                producers[first],
                names2=["_right"],
                text="cross",
                how=6,
            )
        )
        var fields = List[Expr]([col(mode_names[first])])
        if ordinal_names[first]:
            fields.append(col(ordinal_names[first]))
        if count_names[first]:
            fields.append(col(count_names[first]))
        result._nodes.append(
            _plan_node(SELECT, len(result._nodes) - 1, exprs=fields)
        )
        var winners = len(result._nodes) - 1
        for k in range(1, len(choices)):
            var choice = choices[k]
            result._nodes.append(
                _plan_node(
                    JOIN,
                    winners,
                    producers[choice],
                    names=[ordinal_names[first]],
                    right_keys=[ordinal_names[choice]],
                    names2=["_right"],
                    text="inner",
                    how=0,
                    flags=[False, True],
                )
            )
            winners = len(result._nodes) - 1
        for e in range(len(operation.exprs)):
            if scalar_names[e]:
                result._nodes.append(
                    _plan_node(
                        JOIN,
                        winners,
                        producers[e],
                        names2=["_right"],
                        text="cross",
                        how=6,
                    )
                )
                winners = len(result._nodes) - 1
        var projected = List[Expr]()
        for e in range(len(operation.exprs)):
            if count_names[e]:
                ref top = operation.exprs[e]._nodes[
                    len(operation.exprs[e]._nodes) - 1
                ]
                var fields = split_names(top.text)
                projected.append(
                    as_struct(
                        [
                            col(mode_names[e]).alias(fields[0]),
                            col(count_names[e]).alias(fields[1]),
                        ],
                        operation.exprs[e]._name,
                    )
                )
            elif mode_names[e]:
                projected.append(
                    col(mode_names[e]).alias(operation.exprs[e]._name)
                )
            else:
                projected.append(
                    col(scalar_names[e]).alias(operation.exprs[e]._name)
                )
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
