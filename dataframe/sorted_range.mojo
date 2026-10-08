"""Filter parts a sorted column answers by binary search.

A filter part comparing a column with a constant (`col == v`, `<`, `<=`,
`>`, `>=`, and so a `between`) keeps one contiguous run of rows when the
column is in ascending order with no nulls, so the run's bounds come from
two binary searches and the rows are a zero-copy slice, as DuckDB's zone
maps and ClickHouse's primary key skip whole blocks. Order is checked, not
assumed: one pass over the column's whole buffer the first time, stopping
at the first descent, kept in the buffer's order cell (`Column._order`)
for every later filter on any window onto it, as DuckDB keeps zone maps
with its row groups. Data stored in key order is
common (ClickBench's hits table is ordered by CounterID and EventDate);
after one column narrows the window, the next is checked on that window,
where a secondary sort key is in order again.
"""
from .binding import bind
from .column import Column
from .dtype import DataType, NUMERIC_DTYPES
from .execution import evaluate
from .expr import COL, EQ, GE, GT, LE, LT, Expr, subtree
from .parallel import Job, partitions, run_jobs, worker_count
from .series import Series


@fieldwise_init
struct SortedWindow(Copyable, Movable):
    """Rows [low, high) of the frame hold every row the parts marked in
    `used` keep; the parts not marked still have to be applied."""

    var low: Int
    var high: Int
    var used: List[Bool]


@fieldwise_init
struct _Bound(Copyable, Movable):
    var name: String
    var low: Int64
    var high: Int64


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
        if node.left >= 0:
            stack.append(node.left)
        if node.right >= 0:
            stack.append(node.right)
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


def _ascending[
    D: DType
](column: Column[Scalar[D]], first: Int, last: Int) -> Bool:
    """Whether rows [first, last) are in ascending order."""
    var values = column.unsafe_values()
    var i = first
    while i + 9 <= last:
        var a = values.unsafe_offset(i).unsafe_load[width=8]()
        var b = values.unsafe_offset(i + 1).unsafe_load[width=8]()
        if a.gt(b).reduce_or():
            return False
        i += 8
    while i + 1 < last:
        if values.unsafe_offset(i)[] > values.unsafe_offset(i + 1)[]:
            return False
        i += 1
    return True


struct _AscendingJob[D: DType](Job):
    var column: Column[Scalar[Self.D]]
    var first: Int
    var last: Int
    var ascending: Bool

    def __init__(
        out self, column: Column[Scalar[Self.D]], first: Int, last: Int
    ):
        self.column = column.copy()
        self.first = first
        self.last = last
        self.ascending = True

    def run(mut self) raises:
        self.ascending = _ascending[Self.D](self.column, self.first, self.last)


def _ascending_parallel[D: DType](column: Column[Scalar[D]]) raises -> Bool:
    """`_ascending` over the whole column, in ranges on every worker that
    overlap by one row so each boundary pair is compared."""
    var n = len(column)
    var workers = worker_count(n)
    if workers <= 1:
        return _ascending[D](column, 0, n)
    var bounds = partitions(n, workers, 8)
    var jobs = List[_AscendingJob[D]](capacity=workers)
    for w in range(workers):
        jobs.append(
            _AscendingJob[D](column, bounds[w], min(n, bounds[w + 1] + 1))
        )
    run_jobs(jobs)
    for w in range(workers):
        if not jobs[w].ascending:
            return False
    return True


def _ascending_cached[D: DType](column: Column[Scalar[D]]) raises -> Bool:
    """Whether this window is ascending (its rows hold no null; the caller
    checks). The buffer's shared order cell says whether every slot of the
    whole buffer is valid and ascending, found by one pass the first time
    it is asked; a window onto any other buffer is checked on its own
    rows."""
    var cell = Pointer[Int, MutAnyOrigin](
        unsafe_from_address=Int(column._order.ptr())
    )
    if cell[] == 0:
        var whole = column.copy()
        whole._offset = 0
        whole._length = len(column._data[])
        # Two threads may race to fill it; both write the same answer.
        cell[] = 1 if (
            whole.null_count() == 0 and _ascending_parallel[D](whole)
        ) else 2
    if cell[] == 1:
        return True
    # A window onto an unordered buffer: a secondary sort key is ordered
    # again within a run of the first (EventDate within one CounterID).
    # One thread, stopping at the first descent: an unordered column stops
    # within a few rows, and a dispatch to every worker cost more.
    return _ascending[D](column, 0, len(column))


def _first_at_least[
    D: DType
](column: Column[Scalar[D]], first: Int, last: Int, value: Int64) -> Int:
    """The first row in [first, last) whose value is at least `value`."""
    var values = column.unsafe_values()
    var lo = first
    var hi = last
    while lo < hi:
        var mid = (lo + hi) // 2
        if Int64(values.unsafe_offset(mid)[]) < value:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _narrow(
    series: Series, low: Int, high: Int, bound: _Bound
) raises -> Optional[Tuple[Int, Int]]:
    """Rows [low, high) narrowed to those whose value is within `bound`,
    when the column is an integer-backed one with no nulls in the window
    and ascending there; None otherwise. Chunks are searched in place."""
    var pieces = List[Series]()
    var window = series.slice(low, high - low)
    if window.is_chunked():
        pieces = window.chunks()
    else:
        pieces.append(window^)
    var found = Optional[Tuple[Int, Int]]()
    comptime for d in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[d]
        comptime if D.is_integral() and D != DType.uint64:
            if len(pieces) > 0 and pieces[0]._data.isa[Column[Scalar[D]]]():
                # Ascending within each chunk and across their boundaries.
                var previous = Optional[Int64]()
                for piece in pieces:
                    if not piece._data.isa[Column[Scalar[D]]]():
                        return None
                    ref column = piece._data[Column[Scalar[D]]]
                    if column.null_count() > 0:
                        return None
                    var n = len(column)
                    if n == 0:
                        continue
                    if previous and Int64(column._get(0)) < previous.value():
                        return None
                    if not _ascending_cached[D](column):
                        return None
                    previous = Int64(column._get(n - 1))
                var first = high
                var stop = high
                var at = low
                var found_first = False
                for piece in pieces:
                    ref column = piece._data[Column[Scalar[D]]]
                    var n = len(column)
                    if n == 0:
                        continue
                    if not found_first:
                        if Int64(column._get(n - 1)) >= bound.low:
                            first = at + _first_at_least[D](
                                column, 0, n, bound.low
                            )
                            found_first = True
                    if (
                        bound.high < Int64.MAX
                        and Int64(column._get(n - 1)) > bound.high
                    ):
                        stop = at + _first_at_least[D](
                            column, 0, n, bound.high + 1
                        )
                        break
                    at += n
                if not found_first:
                    first = high
                found = Optional((first, max(first, stop)))
    return found


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
