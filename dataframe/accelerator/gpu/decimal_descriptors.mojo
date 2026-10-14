"""Logical decimal metadata for direct Mojo row and reduction programs."""
from dataframe import DataType
from dataframe.expr import (
    ADD,
    SUB,
    MUL,
    DIV,
    NEG,
    ABS,
    CAST,
    GT,
    LT,
    GE,
    LE,
    EQ,
    NE,
    SUM,
    MEAN,
)
from dataframe.accelerator.rows import RowPlan, ROW_GROUP_OUTPUT
from dataframe.parallel import configured_workers
from std.memory import bitcast
from . import decimal, floating

comptime DECIMAL_FLOAT = 213
comptime DECIMAL_SUM = 215
comptime DECIMAL_MEAN = 216
comptime DECIMAL_CAST = 217
comptime DECIMAL_COMPARE = 218
comptime DECIMAL_BINARY = 219
comptime DECIMAL_UNARY = 220


def reduction_op(op: Int64, dtype: DataType) -> Int64:
    if dtype.is_decimal():
        if op == SUM:
            return DECIMAL_SUM
        if op == MEAN:
            return DECIMAL_MEAN
    return op


def reduction_parameter(dtype: DataType, minimum: Int) raises -> Int64:
    var precision = dtype.precision() if dtype.decimal_width() == 128 else 38
    return (
        Int64(min(minimum, Int(UInt32.MAX)))
        | Int64(precision) << 32
        | Int64(dtype.scale()) << 38
        | Int64(min(65535, max(1, configured_workers()))) << 44
    )


def prepare(
    plan: RowPlan, mut code: List[Int64], mut literals: List[UInt64]
) raises:
    for step in plan.steps:
        for node in range(step.start, step.start + step.nodes):
            var at = 4 * node
            var op = code[at]
            var target = plan.node_dtypes[node]
            if op == ROW_GROUP_OUTPUT:
                var source = plan.slot_dtypes[Int(code[at + 1])]
                code[at + 3] = reduction_op(code[at + 3], source)
                if source.is_decimal() and (
                    code[at + 3] == DECIMAL_SUM or code[at + 3] == DECIMAL_MEAN
                ):
                    literals[node] = UInt64(
                        reduction_parameter(
                            source, Int(literals[node].cast[DType.int64]())
                        )
                    )
                continue
            if (
                op != CAST
                and op != NEG
                and op != ABS
                and op != ADD
                and op != SUB
                and op != MUL
                and op != DIV
                and op != GT
                and op != LT
                and op != GE
                and op != LE
                and op != EQ
                and op != NE
            ):
                continue
            if code[at + 1] < 0:
                continue
            var source = plan.node_dtypes[step.start + Int(code[at + 1])]
            if op == CAST and (source.is_decimal() or target.is_decimal()):
                var from_scale = source.scale() if source.is_decimal() else 0
                var to_scale = target.scale() if target.is_decimal() else 0
                code[at] = DECIMAL_CAST
                if source.is_decimal() and target == DataType.FLOAT64:
                    code[at] = DECIMAL_FLOAT
                code[at + 2] = -1
                code[at + 3] = (
                    Int64(from_scale)
                    | Int64(to_scale) << 6
                    | Int64(target.precision() if target.is_decimal() else 0)
                    << 12
                    | code[at + 3] << 18
                )
                if target.is_float():
                    var unit = decimal.wide_pow10(UInt32(from_scale))
                    literals[node] = floating.from_wide(
                        unit.low, unit.high, True
                    )
            elif target.is_decimal() and (op == NEG or op == ABS):
                code[at] = DECIMAL_UNARY
                code[at + 3] = op
            elif (
                op == GT
                or op == LT
                or op == GE
                or op == LE
                or op == EQ
                or op == NE
                or op == ADD
                or op == SUB
                or op == MUL
                or op == DIV
            ) and source.is_decimal():
                var other = plan.node_dtypes[step.start + Int(code[at + 2])]
                if other.is_decimal():
                    code[at + 3] = (
                        Int64(source.scale())
                        | Int64(other.scale()) << 6
                        | op << 12
                    )
                    code[
                        at
                    ] = (
                        DECIMAL_BINARY if target.is_decimal() else DECIMAL_COMPARE
                    )
