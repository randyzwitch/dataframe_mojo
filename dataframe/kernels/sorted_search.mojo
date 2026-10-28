"""Ascending-order validation and binary search over typed column windows."""
from dataframe.storage.column import Column
from dataframe.dtype import NUMERIC_DTYPES
from dataframe.parallel import Job, partitions, run_jobs, worker_count
from dataframe.series import Series


@fieldwise_init
struct _Bound(Copyable, Movable):
    var name: String
    var low: Int64
    var high: Int64


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
    var known = column._facts[].order_state()
    if known == 0:
        var whole = column.copy()
        whole._offset = column._facts[].first()
        whole._length = (
            column._facts[].last(len(column._data[])) - whole._offset
        )
        var ascending = whole.null_count() == 0 and _ascending_parallel[D](
            whole
        )
        column._facts[].publish_order(ascending)
        known = 1 if ascending else 2
    if known == 1:
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
