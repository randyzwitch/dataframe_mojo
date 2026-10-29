"""Row selection within an expression's whole column or aggregation group."""
from dataframe.expr import (
    Node,
    EXPR_FILTER,
    EXPR_SLICE,
    EXPR_GATHER,
    EXPR_GET,
    split_names,
)
from dataframe.series import Series
from dataframe.storage.boolean import BoolColumn


def _position(index: Int, length: Int) raises -> Int:
    var at = index + length if index < 0 else index
    if at < 0 or at >= length:
        raise Error("Expression index out of bounds")
    return at


def select_sequence(
    node: Node, input: Series, predicate: Series
) raises -> Series:
    """Selection preserves source dtype, metadata, validity, and requested order."""
    var n = len(input)
    if node.op == EXPR_SLICE:
        var start = Int(node.integer)
        if start < 0:
            start = max(0, n + start)
        start = min(start, n)
        var length = n - start if node.min_count == -1 else min(
            node.min_count, n - start
        )
        return input.slice(start, length)
    var picks = List[Int]()
    if node.op == EXPR_FILTER:
        var mask = (
            predicate._broadcast(n) if len(predicate) == 1 else predicate.copy()
        )
        if len(mask) != n:
            raise Error("Expression filter operands have different lengths")
        if mask.is_chunked():
            mask = mask.rechunk()
        ref values = mask._data[BoolColumn]
        for row in range(n):
            if values._valid(row) and values._get(row):
                picks.append(row)
    elif node.op == EXPR_GET:
        picks.append(_position(Int(node.integer), n))
    elif node.op == EXPR_GATHER and node.min_count > 0:
        for index in split_names(node.text):
            picks.append(_position(Int(index), n))
    return input.take(picks)
