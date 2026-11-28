"""`list().eval`, `filter` and `agg`: an expression over each list's
elements, which `element()` names.

The expression is bound once against the whole child (the elements of
every valid row, in row order), never per row, and runs one of three ways:

- When each element's result depends only on that element, it runs once
  over the child, and the result keeps the input's offsets.
- When it gives one value per element but looks at a whole list (a window
  such as rank or cum_sum, or `element() - element().mean()`), it runs as
  a window over the row each element belongs to, again keeping offsets.
- Otherwise (an aggregation, or a result of another length) the elements
  are grouped by row and it runs as a group aggregation, one group per
  list.

A filter keeps the elements whose predicate holds. `agg` of a plain
reduction of `element()` is rewritten to the list reduction itself when
the expression is built.
"""
from dataframe.binding import BoundExpr, ROWS, bind
from dataframe.dtype import DataType
from dataframe.expr import (
    ELEMENT,
    LIST_EVAL_AGG,
    LIST_EVAL_FILTER,
    OVER,
    Expr,
    Node,
    col,
    changes_length,
    decode_nodes,
    is_window,
)
from dataframe.kernels.list_transforms import (
    _IDENTITY,
    _choose,
    _elements,
    _row_bits,
)
from dataframe.series import Series
from dataframe.storage.boolean import BoolColumn
from dataframe.storage.column import Column
from dataframe.storage.nested import ListColumn

# The row each element belongs to, while grouping.
comptime _ROW = "__list_row"


def _per_element(body: BoundExpr) -> Bool:
    """Whether each element's result depends only on that element."""
    for i in range(len(body.expr._nodes)):
        var op = body.expr._nodes[i].op
        if body.aggregated[i] or is_window(op) or op == OVER:
            return False
    return True


def list_eval(left: Series, node: Node, dtype: DataType) raises -> Series:
    """Evaluate a LIST_EVAL node over the list column `left`."""
    from dataframe.frame import DataFrame

    var lists = left.list_column()
    var n = len(lists)
    var every = _choose(lists, _IDENTITY)
    var m = len(every.positions)
    var elements = _elements(lists)
    if len(elements) != m:
        # Null rows hide stored elements; leave them out.
        elements = elements.take(every.positions)
    elements = elements.renamed(ELEMENT)
    var body_expr = Expr(decode_nodes(node.text), ELEMENT)
    var body = bind(body_expr, [elements.slice(0, 0)])
    var root = len(body.dtypes) - 1
    var mode = Int(node.integer)
    var owners = List[Int64](unsafe_uninit_length=m)
    for i in range(n):
        for j in range(Int(every.offsets[i]), Int(every.offsets[i + 1])):
            owners[j] = Int64(i)
    var aligned = _per_element(body)
    if (
        not aligned
        and body.shapes[root] == ROWS
        and not changes_length(body_expr)
    ):
        # One result per element, computed within each list: a window over
        # the row each element belongs to.
        aligned = True
        body_expr = body_expr.over(_ROW)
    if aligned:
        var frame = DataFrame([elements.copy()])
        if not _per_element(body):
            frame = DataFrame(
                [Series(_ROW, Column[Int64](owners.copy())), elements.copy()]
            )
        var values = frame._select_exprs([body_expr.alias("v")]).column("v")
        if len(values) != m:
            # A constant (no element in it) holds for every element.
            values = values.take(List[Int](length=m, fill=0))
        if mode == LIST_EVAL_FILTER:
            if values.is_chunked():
                values = values.rechunk()
            ref flags = values._data[BoolColumn]
            var offsets = List[Int64](length=n + 1, fill=0)
            var kept = List[Int]()
            for i in range(n):
                for j in range(
                    Int(every.offsets[i]), Int(every.offsets[i + 1])
                ):
                    if flags._valid(j) and flags._get(j):
                        kept.append(j)
                offsets[i + 1] = Int64(len(kept))
            return Series(
                "",
                ListColumn(
                    offsets^,
                    elements.take(kept).renamed("item"),
                    _row_bits(lists),
                ),
            )
        return Series(
            "",
            ListColumn(
                every.offsets.copy(), values.renamed("item"), _row_bits(lists)
            ),
        )
    # Otherwise group the elements by row (rows without elements form no
    # group) and aggregate each group.
    var inner = body_expr.copy()
    if mode == LIST_EVAL_FILTER:
        inner = col(ELEMENT).filter(body_expr)
    var scalar = body.shapes[root] != ROWS and mode != LIST_EVAL_FILTER
    if not scalar:
        # One list per group.
        inner = inner.implode()
    var frame = DataFrame(
        [Series(_ROW, Column[Int64](owners^)), elements.copy()]
    )
    var grouped = frame.group_by([_ROW], maintain_order=True).agg(
        [inner.alias("v")]
    )
    var keys = grouped.column(_ROW)
    if keys.is_chunked():
        keys = keys.rechunk()
    ref rows = keys._data[Column[Int64]]
    var values = grouped.column("v")
    var groups = len(values)
    var valid = _row_bits(lists)
    var index = List[Int](length=n, fill=-1)
    for k in range(groups):
        index[Int(rows._get(k))] = k
    if scalar:
        # An aggregation over a list with no elements is its value over
        # nothing (sum 0, max null), as for an empty group.
        var empty = (
            DataFrame([elements.slice(0, 0)])
            ._select_exprs([inner.alias("v")])
            .column("v")
        )
        var combined = Series._from_chunks(
            [values^, empty.cast(values.dtype())]
        )
        combined = combined.rechunk()
        for i in range(n):
            if index[i] < 0 and lists._valid(i):
                index[i] = groups
        if mode == LIST_EVAL_AGG:
            return combined.take_or_null(index).renamed("")
        # eval keeps an aggregation as a one-element list.
        var offsets = List[Int64](length=n + 1, fill=0)
        var chosen = List[Int]()
        for i in range(n):
            if index[i] >= 0:
                chosen.append(index[i])
            offsets[i + 1] = Int64(len(chosen))
        return Series(
            "",
            ListColumn(offsets^, combined.take(chosen).renamed("item"), valid^),
        )
    # A list per group; a valid row with no elements gives an empty list.
    var picked = values.take_or_null(index).list_column()
    var offsets = List[Int64](unsafe_uninit_length=n + 1)
    var base = picked._start(0)
    for i in range(n + 1):
        offsets[i] = Int64(picked._start(i) - base)
    return Series("", ListColumn(offsets^, _elements(picked), valid^))
