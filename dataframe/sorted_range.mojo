"""Filter parts a sorted column answers by binary search.

A filter part comparing a column with a constant (`col == v`, `<`, `<=`,
`>`, `>=`, and so a `between`) keeps one contiguous run of rows when the
column is in ascending order with no nulls, so the run's bounds come from
two binary searches and the rows are a zero-copy slice. Order is checked,
not assumed: one pass over the column's whole buffer the first time, stopping
at the first descent, kept in the buffer's order cell (`Column._facts`)
for every later filter on any window onto it. Data stored in key order is
common (ClickBench's hits table is ordered by CounterID and EventDate);
after one column narrows the window, the next is checked on that window,
where a secondary sort key is in order again.
"""
from .expr import node_children
from .binding import bind
from dataframe.storage.column import Column
from .dtype import DataType, NUMERIC_DTYPES
from dataframe.execution import evaluate
from .expr import COL, EQ, GE, GT, LE, LT, Expr, subtree
from .series import Series


from dataframe.kernels.sorted_search import (
    _AscendingJob,
    _Bound,
    _ascending,
    _ascending_cached,
    _ascending_parallel,
    _first_at_least,
    _narrow,
)


@fieldwise_init
struct SortedWindow(Copyable, Movable):
    """Rows [low, high) of the frame hold every row the parts marked in
    `used` keep; the parts not marked still have to be applied."""

    var low: Int
    var high: Int
    var used: List[Bool]


def _is_column(expr: Expr, index: Int) -> Bool:
    return expr._nodes[index].op == COL


def _is_constant(expr: Expr, index: Int) -> Bool:
    """Whether the subtree at `index` reads no column."""
    var stack: List[Int] = [index]
    while len(stack) > 0:
        var i = stack.pop()
        ref node = expr._nodes[i]
        if node.op == COL:
            return False
        for child in node_children(node):
            stack.append(child)
    return True


def _constant_integer(expr: Expr) raises -> Optional[Tuple[Int64, DataType]]:
    """The value of a column-free expression stored as an integer (an
    integer or a temporal value), with its dtype; None otherwise."""
    var columns = List[Series]()
    var value = evaluate(bind(expr, columns), columns, 1)
    if len(value) != 1:
        return None
    if value.is_chunked():
        value = value.rechunk()
    comptime for d in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[d]
        comptime if D.is_integral() and D != DType.uint64:
            if value._data.isa[Column[Scalar[D]]]():
                ref column = value._data[Column[Scalar[D]]]
                if not column._valid(0):
                    return None
                return Optional((Int64(column._get(0)), value.dtype()))
    return None


def _flip(op: Int) -> Int:
    if op == LT:
        return GT
    if op == LE:
        return GE
    if op == GT:
        return LT
    if op == GE:
        return LE
    return op


def _part_bound(part: Expr) raises -> Optional[Tuple[_Bound, DataType]]:
    """The inclusive range of values a `column <op> constant` part keeps,
    with the constant's dtype; None for any other part."""
    var root = len(part._nodes) - 1
    ref node = part._nodes[root]
    var op = node.op
    if not (op == EQ or op == LT or op == LE or op == GT or op == GE):
        return None
    if node.left < 0 or node.right < 0:
        return None
    var column: Int
    var constant: Int
    if _is_column(part, node.left) and _is_constant(part, node.right):
        column = node.left
        constant = node.right
    elif _is_column(part, node.right) and _is_constant(part, node.left):
        column = node.right
        constant = node.left
        op = _flip(op)
    else:
        return None
    var found = _constant_integer(subtree(part, constant))
    if not found:
        return None
    var value = found.value()[0]
    var name = part._nodes[column].text
    var low = Int64.MIN
    var high = Int64.MAX
    if op == EQ:
        low = value
        high = value
    elif op == LT:
        if value == Int64.MIN:
            return Optional((_Bound(name, 1, 0), found.value()[1]))
        high = value - 1
    elif op == LE:
        high = value
    elif op == GT:
        if value == Int64.MAX:
            return Optional((_Bound(name, 1, 0), found.value()[1]))
        low = value + 1
    else:
        low = value
    return Optional((_Bound(name, low, high), found.value()[1]))


def sorted_window(
    columns: List[Series], height: Int, parts: List[Expr]
) raises -> SortedWindow:
    """The rows the `column <op> constant` parts on ascending columns keep,
    as one window, and which parts it answers. Columns are taken in the
    order the parts name them; each one in order on the window so far
    narrows it with every part on that column."""
    var used = List[Bool](length=len(parts), fill=False)
    var bounds = List[Optional[_Bound]]()
    var names = List[String]()
    for part in parts:
        var found = _part_bound(part)
        if not found:
            bounds.append(None)
            continue
        var bound = found.value()[0].copy()
        var dtype = found.value()[1]
        # The constant's type must be the column's, or an integer against
        # an integer column (the comparison adopted it).
        var matched = False
        for column in columns:
            if column.name() == bound.name:
                matched = column.dtype() == dtype or (
                    column.dtype().is_integer() and dtype.is_integer()
                )
        if not matched:
            bounds.append(None)
            continue
        if bound.name not in names:
            names.append(bound.name)
        bounds.append(Optional(bound^))
    var low = 0
    var high = height
    for name in names:
        var combined = _Bound(name, Int64.MIN, Int64.MAX)
        for k in range(len(parts)):
            if bounds[k] and bounds[k].value().name == name:
                combined.low = max(combined.low, bounds[k].value().low)
                combined.high = min(combined.high, bounds[k].value().high)
        var index = -1
        for i in range(len(columns)):
            if columns[i].name() == name:
                index = i
        if index < 0 or low >= high:
            continue
        if combined.low > combined.high:
            # Contradictory bounds keep no row, sorted or not (a null
            # compares as null, which a filter drops too).
            high = low
        else:
            var narrowed = _narrow(columns[index], low, high, combined)
            if not narrowed:
                continue
            low = narrowed.value()[0]
            high = narrowed.value()[1]
        for k in range(len(parts)):
            if bounds[k] and bounds[k].value().name == name:
                used[k] = True
    return SortedWindow(low, high, used^)
