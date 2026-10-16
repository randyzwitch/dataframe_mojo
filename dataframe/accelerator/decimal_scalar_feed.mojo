"""Reuse the narrow precision-only operand proof for whole scalar SUMs.

Float64 SUM state arithmetic cannot raise (IEEE overflow produces infinity).
Decimal sibling SUMs additionally need the existing physical-cardinality bound.
Original operand faults must still be observed before state consumers; this
metadata helper alone does not issue a marker or authorize runtime execution.
"""
from dataframe.binding import BoundExpr, bind
from dataframe.dtype import DataType
from dataframe.series import Series
from dataframe.expr import SUM, WHEN, subtree
from . import decimal_group_feed

comptime MATERIALIZATION = "metal_narrow_decimal_scalar_feed"
comptime ORDERED_MODE = decimal_group_feed.ORDERED_MODE


def eligible(expressions: List[BoundExpr], schema: List[Series]) raises -> Bool:
    var checked = False
    var floating = False
    if not len(expressions):
        return False
    for bound in expressions:
        var last = len(bound.expr._nodes) - 1
        if last < 0:
            return False
        ref root = bound.expr._nodes[last]
        if root.op != SUM or root.left < 0:
            return False
        var input = bind(subtree(bound.expr, root.left), schema)
        var proof = decimal_group_feed.operand(input)
        if not proof[0]:
            return False
        # The ordinary WHEN kernel selects equal logical types. Implicit
        # Decimal common casts require their separate occurrence certificate.
        for at in range(len(input.expr._nodes)):
            ref node = input.expr._nodes[at]
            if node.op == WHEN and (
                input.dtypes[node.left] != DataType.BOOL
                or input.dtypes[node.right] != input.dtypes[at]
                or (
                    node.extra >= 0
                    and input.dtypes[node.extra] != input.dtypes[at]
                )
            ):
                return False
        var dtype = input.dtypes[len(input.dtypes) - 1]
        if dtype == DataType.FLOAT64:
            if bound.dtypes[last] != DataType.FLOAT64:
                return False
            floating = True
        elif not decimal_group_feed.infallible_sum(dtype, Int(UInt32.MAX)):
            return False
        checked |= proof[1]
    return checked and floating
