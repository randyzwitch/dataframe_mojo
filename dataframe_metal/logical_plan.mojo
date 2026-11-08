"""Lower Boolean/count reductions to shared native GPU aggregate state.

Only result-sized finishing runs on the host. True count, valid count and row
count retain enough information for both ignore-null and Kleene reductions.
"""
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Expr,
    LazyFrame,
    Series,
    col,
    lit,
    null,
    when,
)
from dataframe.binding import bind
from dataframe.expr import ANY, ALL, LEN, NULL_COUNT, COUNT, COL
from dataframe.lazy import SCAN_FRAME, FILTER, SELECT, AGG


@fieldwise_init
struct LogicalOutput(Copyable, Movable):
    var name: String
    var op: Int
    var total: String
    var valid: String
    var ignore_nulls: Bool


@fieldwise_init
struct LogicalPlan(Copyable, Movable):
    var region: LazyFrame
    var outputs: List[LogicalOutput]
    var keys: List[String]
    var rows: String


def boolean_value(value: Expr) -> Expr:
    # Testing E and NOT E separately preserves UNKNOWN for ordinary nullable
    # comparisons, while total missing comparisons remain valid.
    return (
        when(value)
        .then(lit(Int64(1)))
        .when(~value)
        .then(lit(Int64(0)))
        .otherwise(null(DataType.INT64))
    )


def plan_logical(region: LazyFrame) raises -> Optional[LogicalPlan]:
    var count = len(region._plan._nodes)
    if count < 2 or region._plan._nodes[0].kind != SCAN_FRAME:
        return None
    ref root = region._plan._nodes[count - 1]
    if root.kind != SELECT and root.kind != AGG:
        return None
    ref frame = region._plan._frames[region._plan._nodes[0].source_slot()]
    var wanted = False
    for expression in root.exprs:
        if expression._nodes:
            wanted = wanted or expression._nodes[
                len(expression._nodes) - 1
            ].op in [
                ANY,
                ALL,
                NULL_COUNT,
            ]
            wanted = wanted or (
                root.kind == SELECT
                and expression._nodes[len(expression._nodes) - 1].op == LEN
            )
            if (
                expression._nodes[len(expression._nodes) - 1].op == COUNT
                and len(expression._nodes) > 1
            ):
                var typed = bind(expression, frame._columns, fuse=False)
                wanted = (
                    wanted
                    or typed.dtypes[len(expression._nodes) - 2] == DataType.BOOL
                )
    if not wanted:
        return None
    for index in range(1, count):
        ref node = region._plan._nodes[index]
        if node.left != index - 1 or (
            index < count - 1 and node.kind != FILTER
        ):
            return None
    var seed = String()
    for name in frame.columns():
        var dtype = frame.column(name).dtype()
        if (
            dtype
            in [DataType.INT32, DataType.INT64, DataType.FLOAT32, DataType.BOOL]
            and not seed
        ):
            seed = name.copy()
    if not seed:
        return None
    var prefix = String("__metal_logical_")
    var names = frame.columns()
    for expression in root.exprs:
        names.append(expression._name.copy())
    while True:
        var collision = False
        for name in names:
            collision = collision or name.startswith(prefix)
        if not collision:
            break
        prefix += "_"
    var row_name = prefix + "rows"
    var expressions = List[Expr]()
    var outputs = List[LogicalOutput]()
    var row_value = (
        when(col(seed).is_null()).then(lit(Int64(1))).otherwise(lit(Int64(1)))
    )
    expressions.append(row_value.sum().alias(row_name))
    var seen = List[String]()
    for index in range(len(root.exprs)):
        ref expression = root.exprs[index]
        var size = len(expression._nodes)
        if not size or expression._name in seen:
            return None
        seen.append(expression._name.copy())
        ref aggregate = expression._nodes[size - 1]
        if aggregate.op not in [ANY, ALL, LEN, NULL_COUNT, COUNT]:
            expressions.append(expression.copy())
            outputs.append(
                LogicalOutput(
                    expression._name.copy(),
                    0,
                    expression._name.copy(),
                    "",
                    True,
                )
            )
            continue
        if size < 2 or aggregate.left != size - 2:
            return None
        var typed = bind(expression, frame._columns, fuse=False)
        var input_type = typed.dtypes[size - 2]
        if input_type not in [
            DataType.INT32,
            DataType.INT64,
            DataType.FLOAT32,
            DataType.BOOL,
        ]:
            return None
        if aggregate.op == COUNT and input_type != DataType.BOOL:
            expressions.append(expression.copy())
            outputs.append(
                LogicalOutput(
                    expression._name.copy(),
                    0,
                    expression._name.copy(),
                    "",
                    True,
                )
            )
            continue
        var value = expression.copy()
        _ = value._nodes.pop()
        var total_name = prefix + "true" + String(index)
        var valid_name = prefix + "valid" + String(index)
        if input_type == DataType.BOOL:
            value = boolean_value(value)
        elif aggregate.op == ANY or aggregate.op == ALL:
            return None
        # Keeping COUNT(value) also preserves eager checked work for LEN.
        expressions.append(value.count().alias(valid_name))
        if aggregate.op == ANY or aggregate.op == ALL:
            expressions.append(value.sum().alias(total_name))
        outputs.append(
            LogicalOutput(
                expression._name.copy(),
                aggregate.op,
                total_name,
                valid_name,
                Bool(aggregate.integer),
            )
        )
    var lowered = region.copy()
    lowered._plan._nodes[count - 1].exprs = expressions^
    var keys = root.group_keys().copy() if root.kind == AGG else List[String]()
    for key in keys:
        if key in seen:
            return None
    return LogicalPlan(lowered^, outputs^, keys^, row_name^)


def logical_key(plan: LogicalPlan) -> String:
    var result = String("logical_reductions:v1:")
    for output in plan.outputs:
        result += (
            String(output.op) + ":" + String(Int(output.ignore_nulls)) + ";"
        )
    return result^


def finish_logical(plan: LogicalPlan, result: DataFrame) raises -> DataFrame:
    var columns = List[Series]()
    for key in plan.keys:
        columns.append(result.column(key))
    for output in plan.outputs:
        if output.op == 0:
            columns.append(
                result.column(output.total).renamed(output.name.copy())
            )
            continue
        var values = List[Int64]()
        var booleans = List[Bool]()
        var validity = List[Bool]()
        for row in range(result.height()):
            var rows = result.item(row, plan.rows).int64()
            var valid = result.item(row, output.valid).int64()
            if output.op == LEN:
                values.append(rows)
            elif output.op == NULL_COUNT:
                values.append(rows - valid)
            elif output.op == COUNT:
                values.append(valid)
            else:
                var total = result.item(row, output.total).int64()
                var answer = total > 0 if output.op == ANY else total == valid
                var decisive = answer if output.op == ANY else not answer
                booleans.append(answer)
                validity.append(
                    output.ignore_nulls or decisive or valid == rows
                )
        if output.op == ANY or output.op == ALL:
            columns.append(
                Series(output.name.copy(), Column[Bool](booleans^, validity^))
            )
        else:
            columns.append(Series(output.name.copy(), Column[Int64](values^)))
    return DataFrame(columns^)
