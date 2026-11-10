"""A table scan with filters.

The scan checks each table filter against the statistics of every row
group first, then of every vector of a row group no filter ruled out: a
vector some filter keeps no row of is not read. A filter that keeps every
row of every vector left is not evaluated (`SetFilterAlwaysTrue`). The rows
left are the table's own buffers, sliced, never copied; the filters that
remain are evaluated over them, reading the filter columns first and
gathering the rest through the rows they keep.
"""
from std.collections import Optional

from dataframe.expr import COL, Expr, lit
from dataframe.frame import DataFrame
from dataframe.kernels.segment_statistics import (
    WindowStatistics,
    window_statistics,
)
from dataframe.series import Series
from dataframe.storage.statistics import (
    FILTER_ALWAYS_FALSE,
    FILTER_ALWAYS_TRUE,
    FILTER_FALSE_OR_NULL,
    NO_PRUNING_POSSIBLE,
    ROW_GROUP_VECTORS,
    STATISTICS_VECTOR_SIZE,
)
from dataframe.table_filter import (
    TableFilter,
    TableFilterSet,
    check_statistics,
    table_filters,
)


def _rules_out(answer: Int) -> Bool:
    return answer == FILTER_ALWAYS_FALSE or answer == FILTER_FALSE_OR_NULL


@fieldwise_init
struct _Segments(Copyable, Movable):
    """A filter's answer over consecutive table rows: rows up to `ends[i]`
    (from the previous end) answer `answers[i]`."""

    var ends: List[Int]
    var answers: List[Int]
    var known: Bool

    def add(mut self, end: Int, answer: Int):
        if end <= (self.ends[len(self.ends) - 1] if len(self.ends) else 0):
            return
        if (
            len(self.answers) > 0
            and self.answers[len(self.answers) - 1] == answer
        ):
            self.ends[len(self.ends) - 1] = end
            return
        self.ends.append(end)
        self.answers.append(answer)


def _group_segments(
    filter: TableFilter,
    found: WindowStatistics,
    start: Int,
    length: Int,
    mut segments: _Segments,
):
    """A window's answers per row group (`RowGroup::CheckZonemap`), table
    rows from `start`."""
    ref record = found.record[]
    var v = found.first
    while v < found.last:
        var group = v // ROW_GROUP_VECTORS
        var group_last = min((group + 1) * ROW_GROUP_VECTORS, found.last)
        var answer = NO_PRUNING_POSSIBLE
        if group < len(record.groups):
            answer = check_statistics(filter, record.groups[group])
        var end = min(group_last * STATISTICS_VECTOR_SIZE - found.shift, length)
        segments.add(start + end, answer)
        v = group_last


def _filter_segments(filter: TableFilter, column: Series) -> _Segments:
    var segments = _Segments(List[Int](), List[Int](), False)
    var start = 0
    for part in column.chunks():
        var length = len(part)
        if length == 0:
            continue
        var found = window_statistics(part)
        if found:
            segments.known = True
            _group_segments(filter, found.value(), start, length, segments)
        else:
            segments.add(start + length, NO_PRUNING_POSSIBLE)
        start += length
    return segments^


@fieldwise_init
struct ScanPruning(Copyable, Movable):
    """What the statistics leave of a scan: the table rows to read, as
    [start, end) pairs in order, and for each filter whether it keeps every
    one of them."""

    var ranges: List[Int]
    var always_true: List[Bool]

    def rows(self) -> Int:
        var total = 0
        for i in range(0, len(self.ranges), 2):
            total += self.ranges[i + 1] - self.ranges[i]
        return total


def _sweep(
    all: List[_Segments],
    height: Int,
    ranges: List[Int],
    mut kept: List[Int],
    mut always: List[Bool],
):
    """Over the rows of `ranges` ([start, end) pairs; all rows when empty
    and `kept` is to be found): the stretches no filter rules out, into
    `kept`, and per filter whether it keeps every row there."""
    var count = len(all)
    var at = List[Int](length=count, fill=0)
    var r = 0
    var row = 0 if len(ranges) == 0 else ranges[0]
    var stop = height if len(ranges) == 0 else ranges[1]
    while True:
        if row >= stop:
            r += 2
            if len(ranges) == 0 or r >= len(ranges):
                return
            row = ranges[r]
            stop = ranges[r + 1]
            continue
        var end = stop
        var keep = True
        for f in range(count):
            ref segments = all[f]
            while at[f] < len(segments.ends) and segments.ends[at[f]] <= row:
                at[f] += 1
            if at[f] >= len(segments.ends):
                continue
            end = min(end, segments.ends[at[f]])
            if _rules_out(segments.answers[at[f]]):
                keep = False
        if keep or len(ranges) > 0:
            for f in range(count):
                ref segments = all[f]
                if (
                    at[f] >= len(segments.ends)
                    or segments.answers[at[f]] != FILTER_ALWAYS_TRUE
                ):
                    always[f] = False
        if keep and len(ranges) == 0:
            if len(kept) > 0 and kept[len(kept) - 1] == row:
                kept[len(kept) - 1] = end
            else:
                kept.append(row)
                kept.append(end)
        row = end


def prune_scan(
    filters: TableFilterSet, columns: List[Series], height: Int
) raises -> Optional[ScanPruning]:
    """The row groups of a table the filters' statistics do not rule out
    (`RowGroup::CheckZonemap`): a row group is read when any of its
    vectors may hold a row every filter keeps. None when no filtered
    column has statistics. Skipping single vectors inside a row group
    (`CheckZonemapSegments`) is the chunk loop's (`zonemap_rows`)."""
    var all = List[_Segments]()
    var any = False
    for filter in filters.filters:
        var column = Optional[Series]()
        for c in columns:
            if c.name() == filter.column:
                column = c.copy()
                break
        if not column:
            all.append(_Segments([height], [NO_PRUNING_POSSIBLE], False))
            continue
        var segments = _filter_segments(filter, column.value())
        any = any or segments.known
        all.append(segments^)
    if not any:
        return None
    var count = len(all)
    var vectors = List[Int]()
    var unused = List[Bool](length=count, fill=True)
    _sweep(all, height, List[Int](), vectors, unused)
    # Whole row groups (DEFAULT_ROW_GROUP_SIZE rows) holding a kept vector.
    var group = ROW_GROUP_VECTORS * STATISTICS_VECTOR_SIZE
    var ranges = List[Int]()
    for i in range(0, len(vectors), 2):
        var first = (vectors[i] // group) * group
        var last = min(((vectors[i + 1] + group - 1) // group) * group, height)
        if len(ranges) > 0 and ranges[len(ranges) - 1] >= first:
            ranges[len(ranges) - 1] = max(ranges[len(ranges) - 1], last)
        else:
            ranges.append(first)
            ranges.append(last)
    var always = List[Bool](length=count, fill=True)
    var none = List[Int]()
    if len(ranges) > 0:
        _sweep(all, height, ranges, none, always)
    for f in range(count):
        if filters.filters[f].optional:
            always[f] = True
    return ScanPruning(ranges^, always^)


@fieldwise_init
struct _ColumnWindows(Copyable, Movable):
    """A filter column's chunks: the table row each starts at, its length,
    and the statistics its window reads."""

    var starts: List[Int]
    var lengths: List[Int]
    var stats: List[Optional[WindowStatistics]]


struct ScanFilterInfo(Copyable, Movable):
    """A scan's table filters and where their columns' statistics are
    (`ScanFilterInfo`): the chunk loop asks `skip` for each vector it is
    about to read (`CheckZonemapSegments`), which checks the vector's
    statistics then. Dynamic filters are read as they are at that moment,
    as the operator that sets them tightens them while the scan runs (a
    TopN boundary)."""

    var filters: TableFilterSet
    var windows: List[_ColumnWindows]

    def __init__(
        out self, filters: TableFilterSet, columns: List[Series]
    ) raises:
        self.filters = filters.copy()
        self.windows = List[_ColumnWindows]()
        for filter in filters.filters:
            var found = _ColumnWindows(
                List[Int](), List[Int](), List[Optional[WindowStatistics]]()
            )
            for c in columns:
                if c.name() != filter.column:
                    continue
                var start = 0
                for part in c.chunks():
                    found.starts.append(start)
                    found.lengths.append(len(part))
                    found.stats.append(window_statistics(part))
                    start += len(part)
                break
            self.windows.append(found^)

    def skip(self, start: Int, end: Int) -> Bool:
        """Whether some filter's statistics rule out every row of table
        rows [start, end)."""
        for f in range(len(self.filters.filters)):
            ref w = self.windows[f]
            var lo = 0
            var hi = len(w.starts)
            while lo < hi:
                var mid = (lo + hi) // 2
                if w.starts[mid] + w.lengths[mid] <= start:
                    lo = mid + 1
                else:
                    hi = mid
            if lo >= len(w.starts) or not w.stats[lo]:
                continue
            if end > w.starts[lo] + w.lengths[lo]:
                # Rows of two chunks: left to the filter.
                continue
            ref found = w.stats[lo].value()
            ref record = found.record[]
            var first = (start - w.starts[lo] + found.shift) // (
                STATISTICS_VECTOR_SIZE
            )
            var last = (end - 1 - w.starts[lo] + found.shift) // (
                STATISTICS_VECTOR_SIZE
            )
            var out = True
            for v in range(first, last + 1):
                if not _rules_out(
                    check_statistics(self.filters.filters[f], record.vectors[v])
                ):
                    out = False
                    break
            if out:
                return True
        return False


def zonemap_rows(
    filters: TableFilterSet, columns: List[Series], start: Int, end: Int
) raises -> Int:
    """FILTER_ALWAYS_FALSE when the filters' statistics rule out every row
    of table rows [start, end), else NO_PRUNING_POSSIBLE."""
    if ScanFilterInfo(filters, columns).skip(start, end):
        return FILTER_ALWAYS_FALSE
    return NO_PRUNING_POSSIBLE


def rows_of(frame: DataFrame, ranges: List[Int]) raises -> DataFrame:
    """The frame's rows in `ranges`, as slices of its own buffers."""
    if len(ranges) == 2 and ranges[0] == 0 and ranges[1] == frame.height():
        return frame.copy()
    if len(ranges) == 0:
        return frame.clear()
    if len(ranges) == 2:
        return frame.slice(ranges[0], ranges[1] - ranges[0])
    var columns = List[Series](capacity=frame.width())
    for column in frame._columns:
        var parts = List[Series](capacity=len(ranges) // 2)
        for i in range(0, len(ranges), 2):
            parts.append(column.slice(ranges[i], ranges[i + 1] - ranges[i]))
        columns.append(Series._from_chunks(parts))
    var height = 0
    for i in range(0, len(ranges), 2):
        height += ranges[i + 1] - ranges[i]
    var result = DataFrame(columns^, height=height)
    result._statistics = frame._statistics
    return result^


@fieldwise_init
struct FilteredScan(Copyable, Movable):
    """A scan's rows after its statistics, and the predicate parts still
    to evaluate over them."""

    var frame: DataFrame
    var remaining: List[Expr]
    # The rows of the scanned frame `frame` holds, as [start, end) pairs.
    var ranges: List[Int]


def scan_filtered(
    frame: DataFrame, parts: List[Expr]
) raises -> Optional[FilteredScan]:
    """The rows of `frame` the table filters among `parts` can keep, and the
    parts left to evaluate; None when statistics change nothing."""
    if not _reads_statistics(frame, parts):
        return None
    var filters = table_filters(parts, frame._columns)
    if not filters.has_filters():
        return None
    var pruned = prune_scan(filters, frame._columns, frame.height())
    if not pruned:
        return None
    ref found = pruned.value()
    var dropped = List[Bool](length=len(parts), fill=False)
    var any_dropped = False
    for f in range(len(filters.filters)):
        if found.always_true[f] and filters.filters[f].source >= 0:
            dropped[filters.filters[f].source] = True
            any_dropped = True
    var kept = found.rows()
    if kept == frame.height() and not any_dropped:
        return None
    var remaining = List[Expr]()
    if kept > 0:
        for p in range(len(parts)):
            if not dropped[p]:
                remaining.append(parts[p].copy())
    return FilteredScan(
        rows_of(frame, found.ranges), remaining^, found.ranges.copy()
    )


def _reads_statistics(frame: DataFrame, parts: List[Expr]) -> Bool:
    """Whether a column the parts read has statistics: constants are only
    evaluated then, as most frames filtered are not stored tables."""
    for part in parts:
        for node in part._nodes:
            if node.op != COL:
                continue
            for column in frame._columns:
                if column.name() == node.text and _has_statistics(column):
                    return True
    return False


def scan_filter_info(
    frame: DataFrame, parts: List[Expr]
) raises -> Optional[ScanFilterInfo]:
    """The table filters among a scan's first filter's `parts`, with their
    answers per vector of `frame`, or None when no filtered column has
    statistics."""
    if frame.height() == 0 or not _reads_statistics(frame, parts):
        return None
    var filters = table_filters(parts, frame._columns)
    if not filters.has_filters():
        return None
    return ScanFilterInfo(filters, frame._columns)


def _has_statistics(column: Series) -> Bool:
    for part in column.chunks():
        if len(part) > 0:
            return Bool(window_statistics(part))
    return False


def scan_ranges(
    filters: TableFilterSet, frame: DataFrame
) raises -> Optional[DataFrame]:
    """The frame's rows the optional filters' statistics can keep (a join
    filter's key range), or None when they rule nothing out."""
    var pruned = prune_scan(filters, frame._columns, frame.height())
    if not pruned or pruned.value().rows() == frame.height():
        return None
    return rows_of(frame, pruned.value().ranges)


def conjunction(parts: List[Expr]) -> Expr:
    """`parts` joined with `&`; a literal true for none."""
    if len(parts) == 0:
        return lit(True)
    var result = parts[0].copy()
    for k in range(1, len(parts)):
        result = result & parts[k]
    return result^
