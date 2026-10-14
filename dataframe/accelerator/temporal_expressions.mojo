"""Context-free admission checks for bounded temporal expressions.

Reject checked producers, predicates and aliases before lowering; category64
is the only possible new fault within an admitted expression. No data is read.
"""

from dataframe.binding import BoundExpr, ROWS
from dataframe.dtype import DataType
from dataframe.expr import (
    COL,
    LIT_INT,
    LIT_BOOL,
    LIT_NULL,
    CAST,
    WHEN,
    ADD,
    SUB,
    MUL,
    FLOORDIV,
    GT,
    LT,
    GE,
    LE,
    EQ,
    NE,
    IS_NULL,
    IS_NOT_NULL,
    NOT,
    AND,
    OR,
    XOR,
)
from .gpu.temporal_descriptors import binary_literal, cast_literal


def binary(bound: BoundExpr, index: Int) -> Bool:
    ref node = bound.expr._nodes[index]
    return (
        (
            node.op == ADD
            or node.op == SUB
            or node.op == MUL
            or node.op == FLOORDIV
        )
        and node.left >= 0
        and node.right >= 0
        and (
            bound.dtypes[node.left].is_temporal()
            or bound.dtypes[node.right].is_temporal()
        )
    )


def cast(bound: BoundExpr, index: Int) -> Bool:
    ref node = bound.expr._nodes[index]
    return (
        node.op == CAST
        and node.left >= 0
        and (
            bound.dtypes[node.left].is_temporal()
            or bound.dtypes[index].is_temporal()
        )
    )


def contains(bound: BoundExpr) -> Bool:
    for index in range(len(bound.expr._nodes)):
        if binary(bound, index) or cast(bound, index):
            return True
    return False


def _word(dtype: DataType) -> Bool:
    return (
        dtype == DataType.INT64 or dtype == DataType.BOOL or dtype.is_temporal()
    )


def validate(bound: BoundExpr) raises:
    if bound.shape() != ROWS:
        raise Error(
            "Metal unsupported [temporal]: prototype requires row-valued projections"
        )
    var safe = List[Bool](length=len(bound.expr._nodes), fill=False)
    for index in range(len(bound.expr._nodes)):
        ref node = bound.expr._nodes[index]
        var dtype = bound.dtypes[index]
        if not _word(dtype):
            raise Error(
                "Metal unsupported [temporal]: only temporal, Int64 and Bool nodes are admitted"
            )
        if (
            node.op == COL
            or node.op == LIT_INT
            or node.op == LIT_BOOL
            or node.op == LIT_NULL
        ):
            safe[index] = True
        elif binary(bound, index):
            if not safe[node.left] or not safe[node.right]:
                raise Error(
                    "Metal unsupported [temporal]: binary operands must be infallible"
                )
            _ = binary_literal(
                Int64(node.op),
                bound.dtypes[node.left],
                bound.dtypes[node.right],
            )
        elif cast(bound, index):
            if not safe[node.left]:
                raise Error(
                    "Metal unsupported [temporal]: cast input must be infallible"
                )
            var parameter = cast_literal(
                bound.dtypes[node.left], dtype, node.integer != 0
            )
            var mode = parameter & 15
            safe[index] = (
                node.integer == 0
                or mode == 0
                or mode == 2
                or mode == 3
                or (
                    mode == 4
                    and ((parameter >> 8) & 15) <= ((parameter >> 4) & 15)
                )
            )
        elif node.op == WHEN:
            if node.extra < 0:
                raise Error(
                    "Metal unsupported [temporal]: prototype requires explicit otherwise"
                )
            if bound.dtypes[node.left] != DataType.BOOL or not safe[node.left]:
                raise Error(
                    "Metal unsupported [temporal]: predicate must be infallible"
                )
            safe[index] = safe[node.right] and safe[node.extra]
        elif (
            node.op == GT
            or node.op == LT
            or node.op == GE
            or node.op == LE
            or node.op == EQ
            or node.op == NE
        ):
            if (
                not safe[node.left]
                or not safe[node.right]
                or bound.dtypes[node.left] != bound.dtypes[node.right]
                or (
                    bound.dtypes[node.left] != DataType.INT64
                    and bound.dtypes[node.left] != DataType.BOOL
                )
            ):
                raise Error(
                    "Metal unsupported [temporal]: predicate comparison must be infallible Int64 or Bool"
                )
            safe[index] = True
        elif node.op == IS_NULL or node.op == IS_NOT_NULL or node.op == NOT:
            if not safe[node.left]:
                raise Error(
                    "Metal unsupported [temporal]: predicate input must be infallible"
                )
            safe[index] = True
        elif node.op == AND or node.op == OR or node.op == XOR:
            if (
                not safe[node.left]
                or not safe[node.right]
                or bound.dtypes[node.left] != DataType.BOOL
                or bound.dtypes[node.right] != DataType.BOOL
            ):
                raise Error(
                    "Metal unsupported [temporal]: boolean inputs must be infallible"
                )
            safe[index] = True
        else:
            raise Error(
                "Metal unsupported [temporal]: mixed checked or unsupported expression"
            )
