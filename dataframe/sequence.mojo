"""Whole-expression row operations, restarted within each aggregation group."""
from dataframe.expr import (
    Node,
    EXPR_FILTER,
    EXPR_SLICE,
    EXPR_GATHER,
    EXPR_GET,
    EXPR_SORT,
    EXPR_SORT_BY,
    EXPR_UNIQUE,
    EXPR_UNIQUE_COUNTS,
    EXPR_TOP_K,
    EXPR_BOTTOM_K,
    EXPR_TOP_K_BY,
    EXPR_BOTTOM_K_BY,
    EXPR_MAX_BY,
    EXPR_MIN_BY,
    EXPR_ARG_SORT,
    EXPR_REVERSE,
    split_names,
)
from dataframe.series import Series, sort_indices
from dataframe.categorical import sort_ranks as categorical_sort_ranks
from dataframe.kernels.binary import validity
from dataframe.storage.boolean import BoolColumn
from dataframe.storage.column import Column


def _position(index: Int, length: Int) raises -> Int:
    var at = index + length if index < 0 else index
    if at < 0 or at >= length:
        raise Error("Expression index out of bounds")
    return at


def _aligned(input: Series, n: Int) raises -> Series:
    var out = input._broadcast(n) if len(input) == 1 else input.copy()
    if len(out) != n:
        raise Error("Expression sequence operands have different lengths")
    return out^


def _order(
    keys: List[Series], descending: Bool, nulls_last: Bool
) raises -> List[Int]:
    var ranks = List[List[Int]](capacity=len(keys))
    for key in keys:
        var normalized = categorical_sort_ranks(
            key
        ) if key.dtype().is_categorical() else key.copy()
        ranks.append(normalized._sort_ranks(descending, nulls_last))
    return sort_indices(ranks)


def _by_keys(node: Node, by: Series) raises -> List[Series]:
    if node.text != "keys":
        return [by.copy()]
    var fields = by.struct_column()
    var keys = List[Series](capacity=fields.field_count())
    for k in range(fields.field_count()):
        keys.append(fields.field(k))
    return keys^


def select_sequence(node: Node, input: Series, other: Series) raises -> Series:
    """Keep source metadata and nulls while selecting at this sequence boundary."""
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
    if node.op == EXPR_REVERSE:
        return input.reverse()
    if node.op == EXPR_UNIQUE:
        return input.unique(maintain_order=node.integer != 0)
    if node.op == EXPR_UNIQUE_COUNTS:
        # Renaming avoids collisions with the count output. This uses the
        # existing exact grouping equality, including null/NaN equivalence.
        return (
            input.renamed("__value")
            .value_counts(sort=False, name="__count")
            .column("__count")
        )
    if node.op == EXPR_SORT or node.op == EXPR_ARG_SORT:
        var order = _order(
            [input.copy()], node.min_count != 0, node.integer != 0
        )
        if node.op == EXPR_ARG_SORT:
            var values = List[Int64](capacity=len(order))
            for row in order:
                values.append(Int64(row))
            return Series("", Column[Int64](values^))
        return input.take(order)
    if node.op in [
        EXPR_SORT_BY,
        EXPR_TOP_K_BY,
        EXPR_BOTTOM_K_BY,
        EXPR_MAX_BY,
        EXPR_MIN_BY,
    ]:
        var by = _aligned(other, n)
        var keys = _by_keys(node, by)
        var descending = (
            node.min_count
            != 0 if node.op
            == EXPR_SORT_BY else node.op
            in [EXPR_TOP_K_BY, EXPR_MAX_BY]
        )
        var nulls_last = node.integer != 0 if node.op == EXPR_SORT_BY else True
        var order = _order(keys, descending, nulls_last)
        if node.op == EXPR_SORT_BY:
            return input.take(order)
        if node.op == EXPR_MAX_BY or node.op == EXPR_MIN_BY:
            var valid = validity(by)
            for row in order:
                if valid[row]:
                    return input.take([row])
            return input.take_or_null([-1])
        var picks = List[Int](capacity=min(n, Int(node.integer)))
        for k in range(min(n, Int(node.integer))):
            picks.append(order[k])
        return input.take(picks)
    if node.op == EXPR_TOP_K or node.op == EXPR_BOTTOM_K:
        var order = _order([input.copy()], node.op == EXPR_TOP_K, True)
        var picks = List[Int](capacity=min(n, Int(node.integer)))
        for k in range(min(n, Int(node.integer))):
            picks.append(order[k])
        return input.take(picks)
    var picks = List[Int]()
    if node.op == EXPR_FILTER:
        var mask = _aligned(other, n)
        if mask.is_chunked():
            mask = mask.rechunk()
        ref values = mask._data[BoolColumn]
        for row in range(n):
            if values._valid(row) and values._get(row):
                picks.append(row)
    elif node.op == EXPR_GET:
        picks.append(_position(Int(node.integer), n))
    elif node.op == EXPR_GATHER:
        if node.min_count > 0:
            for index in split_names(node.text):
                picks.append(_position(Int(index), n))
    else:
        raise Error("Unknown expression sequence operation")
    return input.take(picks)
