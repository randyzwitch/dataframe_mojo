"""Match a single native binary expression reduced by SUM."""
from dataframe import DataFrame, DataType, Expr, LazyFrame, col
from dataframe.expr import COL, SUM, ADD, SUB, MUL, LIT_INT, LIT_FLOAT, UNTYPED
from dataframe.lazy import SCAN_FRAME, SELECT, FILTER
from .filter_plan import FilterSum, plan_region, supported_dtype


@fieldwise_init
struct ArithmeticSum(Copyable, Movable):
    var predicate: FilterSum
    var left: String
    var right: String
    var op: Int
    var literal_side: Int
    var integer: Int64
    var floating: Float32


def plan_arithmetic(region: LazyFrame) raises -> Optional[ArithmeticSum]:
    var n = len(region._plan._nodes)
    if n < 2 or region._plan._nodes[0].kind != SCAN_FRAME:
        return None
    ref root = region._plan._nodes[n - 1]
    if root.kind != SELECT or len(root.exprs) != 1:
        return None
    ref expression = root.exprs[0]
    if len(expression._nodes) != 4:
        return None
    ref a = expression._nodes[0]
    ref b = expression._nodes[1]
    ref binary = expression._nodes[2]
    ref total = expression._nodes[3]
    if (
        total.op != SUM
        or total.left != 2
        or binary.left != 0
        or binary.right != 1
    ):
        return None
    if binary.op != ADD and binary.op != SUB and binary.op != MUL:
        return None
    if a.op != COL and b.op != COL:
        return None
    var frame = region._plan._frames[
        region._plan._nodes[0].source_slot()
    ].copy()
    var name = a.text.copy() if a.op == COL else b.text.copy()
    if name not in frame.columns():
        return None
    var dtype = frame.column(name).dtype()
    if not supported_dtype(dtype):
        return None
    var right_name = name.copy()
    var side = 0
    var integer = Int64(0)
    var floating = Float32(0)
    if a.op == COL and b.op == COL:
        if (
            b.text not in frame.columns()
            or frame.column(b.text).dtype() != dtype
        ):
            return None
        right_name = b.text.copy()
    else:
        var literal = b.copy() if a.op == COL else a.copy()
        side = 2 if a.op == COL else 1
        if literal.op != LIT_INT and literal.op != LIT_FLOAT:
            return None
        # Typed literals must match the column. Untyped syntax binds to it.
        if literal.text != UNTYPED:
            var literal_dtype = (
                DataType.INT64 if literal.op == LIT_INT else DataType.FLOAT64
            )
            if len(literal.dtypes) == 1 and literal.dtypes[0]:
                literal_dtype = literal.dtypes[0].value()
            if literal_dtype != dtype:
                return None
        if dtype == DataType.FLOAT32:
            var value = (
                Float64(literal.integer) if literal.op
                == LIT_INT else literal.floating
            )
            floating = Float32(value)
            # Decline literals requiring rounding or outside the native range.
            if Float64(floating) != value:
                return None
        else:
            if literal.op != LIT_INT:
                return None
            integer = literal.integer
            if dtype == DataType.INT32 and (
                integer < Int64(Int32.MIN) or integer > Int64(Int32.MAX)
            ):
                return None
    var offered = frame.lazy()
    for i in range(1, n - 1):
        ref node = region._plan._nodes[i]
        if node.kind != FILTER or len(node.exprs) != 1:
            return None
        offered = offered.filter(node.exprs[0])
    var matched = plan_region(offered.select(col(name).sum()))
    if not matched:
        return None
    var predicate = matched.take()
    predicate.output = expression._name.copy()
    predicate.min_count = total.min_count
    return ArithmeticSum(
        predicate^, name^, right_name^, binary.op, side, integer, floating
    )


def arithmetic_key(plan: ArithmeticSum) -> String:
    var shape = ":literal" if plan.literal_side else (
        ":same_column" if plan.left == plan.right else ":two_columns"
    )
    var predicate = (
        ":external" if plan.predicate.filter_column != plan.left
        and plan.predicate.filter_column != plan.right else ":shared"
    )
    return (
        "arithmetic_sum:"
        + plan.predicate.dtype.name()
        + ":"
        + plan.predicate.filter_dtype.name()
        + ":"
        + String(plan.op)
        + shape
        + predicate
        + (":filtered" if plan.predicate.filtered else ":plain")
    )
