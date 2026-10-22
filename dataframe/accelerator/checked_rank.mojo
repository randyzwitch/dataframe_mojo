"""Bound proof for one eager integer rank-input observation phase."""
from dataframe.binding import bind
from dataframe.execution import _WINDOW_BLOCK
from dataframe.expr import (
    node_children,
    Expr,
    RANK,
    OVER,
    CAST,
    ADD,
    SUB,
    MUL,
    DIV,
    FLOORDIV,
    MOD,
    POW,
    NEG,
    ABS,
    is_reduction,
    is_window,
)
from dataframe.series import Series
from dataframe.dtype import DataType
from .integer_ranges import proven_integer_nodes

comptime ORDERED_RANK_INPUT = 10
comptime BATCH_ROWS = 8192


def row_order_over(rows: Int, workers: Int) -> Bool:
    # CPU _over attempts contiguous partition jobs only when lanes > 1.
    # This is a sufficient certificate for its original global-row route;
    # partition-cut fallbacks with more lanes need a separate geometry proof.
    return min(workers, rows // _WINDOW_BLOCK) <= 1


def isolated(expressions: List[Expr], schema: List[Series]) raises -> Bool:
    """Other projection/state work cannot compete with the rank operand fault."""
    var ranks = 0
    for expression in expressions:
        var bound = bind(expression, schema)
        var allowed = List[Bool](length=len(bound.expr._nodes), fill=False)
        for node in bound.expr._nodes:
            if node.op == RANK:
                ranks += 1
                var pending = List[Int]([node.left])
                while len(pending):
                    var at = pending.pop()
                    if at < 0 or allowed[at]:
                        continue
                    allowed[at] = True
                    ref child = bound.expr._nodes[at]
                    for index in node_children(child):
                        pending.append(index)
        var proven = proven_integer_nodes(bound)
        for at in range(len(bound.expr._nodes)):
            ref node = bound.expr._nodes[at]
            if allowed[at]:
                if (
                    is_reduction(node.op)
                    or is_window(node.op)
                    or node.op == OVER
                ):
                    return False
                if bound.dtypes[at].is_decimal():
                    return False
                continue
            if is_window(node.op) and node.op != RANK:
                return False
            if is_reduction(node.op) and not bound.dtypes[node.left].is_float():
                return False
            if node.op == CAST and node.integer and not proven[at]:
                if not (
                    bound.dtypes[at] == DataType.FLOAT64
                    and bound.dtypes[node.left].is_numeric()
                ):
                    return False
            if (
                (bound.dtypes[at].is_integer() or bound.dtypes[at].is_decimal())
                and node.op
                in [ADD, SUB, MUL, DIV, FLOORDIV, MOD, POW, NEG, ABS]
                and not proven[at]
            ):
                return False
    return ranks == 1
