"""Metadata proof for eager scalar Decimal SUM operand-feed replay."""
from dataframe.binding import BoundExpr, bind
from dataframe.series import Series
from dataframe.expr import (
    SUM,
    CAST,
    ADD,
    SUB,
    MUL,
    DIV,
    NEG,
    ABS,
    FLOORDIV,
    POW,
    subtree,
    is_comparison,
)
from .integer_ranges import proven_integer_nodes
from . import ordered_rows

comptime MATERIALIZATION = "metal_decimal_feed_materialization"
comptime BATCH_ROWS = 8192
comptime DESCRIPTOR_FLAG = UInt64(1) << 60


def eligible(bound: BoundExpr, schema: List[Series]) raises -> Bool:
    var last = len(bound.expr._nodes) - 1
    if last < 0 or bound.expr._nodes[last].op != SUM:
        return False
    ref root = bound.expr._nodes[last]
    if root.left < 0 or not bound.dtypes[root.left].is_decimal():
        return False
    var operand = bind(subtree(bound.expr, root.left), schema)
    if not ordered_rows.eligible(operand):
        return False
    var proven = proven_integer_nodes(operand)
    var checked = 0
    for at in range(len(operand.expr._nodes)):
        ref node = operand.expr._nodes[at]
        var dtype = operand.dtypes[at]
        # One fallible arithmetic occurrence gives the legacy operand plane
        # one unambiguous fault per row. Other fallible phases need ranked
        # AST observations, and are not certified by this first increment.
        if node.op == CAST and node.integer and not proven[at]:
            return False
        if (
            is_comparison(node.op)
            and node.left >= 0
            and operand.dtypes[node.left].is_decimal()
            and not proven[at]
        ):
            return False
        if dtype.is_temporal():
            return False
        if node.op in [ADD, SUB, MUL, DIV, NEG, ABS, FLOORDIV, POW]:
            if dtype.is_decimal() and not proven[at]:
                checked += 1
            elif dtype.is_integer() and not proven[at]:
                return False
    return checked == 1
