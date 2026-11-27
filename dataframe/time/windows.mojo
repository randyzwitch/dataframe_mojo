"""Calendar-aware bounds shared by rolling expressions and time grouping."""
from dataframe.storage.column import Column
from dataframe.dtype import DataType
from dataframe.series import Series
from dataframe.time.temporal import (
    parse_interval,
    zone_of,
    to_local,
    relocalize,
)
from dataframe.kernels.temporal import Offset, _offset, _offset_in, _truncate
from dataframe.time.timezone import TimeZone


def positive_period(period: String) raises:
    var parts = parse_interval(period)
    if (
        parts[0] < 0
        or parts[1] < 0
        or parts[2] < 0
        or (parts[0] == 0 and parts[1] == 0 and parts[2] == 0)
    ):
        raise Error("window duration must be positive: " + period)


def closed_sides(closed: String) raises -> Tuple[Bool, Bool]:
    if closed not in ["left", "right", "both", "none"]:
        raise Error("closed must be left, right, both or none")
    return (
        closed == "left" or closed == "both",
        closed == "right" or closed == "both",
    )


def negate_period(period: String) -> String:
    return (
        String(period[byte=1:]) if (
            period.byte_length() > 0 and period.as_bytes()[0] == 45
        ) else "-"
        + period
    )


def shift_time(
    value: Int64, dtype: DataType, period: String, zone: TimeZone
) raises -> Int64:
    if dtype.is_datetime() and dtype.time_zone() != "":
        return _offset_in(value, dtype, period, zone)
    return _offset(value, dtype, period)


def truncate_time(
    value: Int64, dtype: DataType, every: String, zone: TimeZone
) raises -> Int64:
    if dtype.is_datetime() and dtype.time_zone() != "":
        return relocalize(
            _truncate(to_local(value, dtype, zone), dtype, every),
            value,
            dtype,
            zone,
        )
    return _truncate(value, dtype, every)


def check_time_index(index: Series, groups: List[List[Int]]) raises -> Bool:
    """Raise on a non-temporal or null index; return whether each group's
    rows already appear in index order."""
    if not (index.dtype().is_date() or index.dtype().is_datetime()):
        raise Error(
            "time windows require a Date or Datetime index: " + index.name()
        )
    ref values = index._data[Column[Int64]]
    var ordered = True
    for rows in groups:
        for j in range(len(rows)):
            var row = rows[j]
            if not values._valid(row):
                raise Error("time window index contains nulls: " + index.name())
            if j > 0 and values._get(row) < values._get(rows[j - 1]):
                ordered = False
    return ordered


def sort_time_groups(index: Series, mut groups: List[List[Int]]) raises:
    """Put each group's rows in stable index order (equal times keep row
    order), so unsorted input forms the same windows as sorted input."""
    ref values = index._data[Column[Int64]]
    for g in range(len(groups)):
        ref rows = groups[g]
        var ordered = True
        for j in range(1, len(rows)):
            if values._get(rows[j]) < values._get(rows[j - 1]):
                ordered = False
                break
        if ordered:
            continue
        var order = index.take(rows).argsort()
        var sorted = List[Int](capacity=len(rows))
        for k in order:
            sorted.append(rows[k])
        groups[g] = sorted^


def search_time(
    values: Column[Int64], rows: List[Int], value: Int64, after_equal: Bool
) -> Int:
    var lo = 0
    var hi = len(rows)
    while lo < hi:
        var mid = lo + (hi - lo) // 2
        var x = values._get(rows[mid])
        if x < value or (after_equal and x == value):
            lo = mid + 1
        else:
            hi = mid
    return lo


def rolling_bounds(
    index: Series,
    mut groups: List[List[Int]],
    period: String,
    offset: String,
    closed: String,
) raises -> Tuple[List[Int], List[Int]]:
    """Each row's window as [lower, upper) positions in its group's rows.

    Groups not already in index order are reordered in place first, and the
    positions refer to that order.
    """
    positive_period(period)
    _ = parse_interval(offset)
    var sides = closed_sides(closed)
    if not check_time_index(index, groups):
        sort_time_groups(index, groups)
    var lower = List[Int](length=len(index), fill=0)
    var upper = List[Int](length=len(index), fill=0)
    var dtype = index.dtype()
    var zone = zone_of(dtype)
    ref values = index._data[Column[Int64]]
    var times = values._ptr()
    var default_offset = offset == negate_period(period)
    # A naive index shifts by an interval parsed once; one in a time zone
    # shifts in local time, row by row.
    var naive = (
        not (dtype.is_datetime() and dtype.time_zone() != "") and len(index) > 0
    )
    var shift_start = Offset(dtype, offset) if naive else Offset.unused(dtype)
    var shift_end = Offset.unused(dtype)
    if naive and not default_offset:
        shift_end = Offset(dtype, period)
    for rows in groups:
        # Rows are in time order and shifting a time never reverses that
        # order, so both bounds only move forward: a sweep finds them, one
        # step per row and bound, where a search per row would take
        # O(log n) reads.
        var lo = 0
        var hi = 0
        var m = len(rows)
        for row in rows:
            var time = times[unsafe_offset=row]
            var start = shift_start.apply(time) if naive else shift_time(
                time, dtype, offset, zone
            )
            # Calendar month subtraction then addition need not round-trip.
            var end = time
            if not default_offset:
                end = shift_end.apply(start) if naive else shift_time(
                    start, dtype, period, zone
                )
            while lo < m and (
                times[unsafe_offset=rows[lo]] < start
                or (not sides[0] and times[unsafe_offset=rows[lo]] == start)
            ):
                lo += 1
            hi = max(hi, lo)
            while hi < m and (
                times[unsafe_offset=rows[hi]] < end
                or (sides[1] and times[unsafe_offset=rows[hi]] == end)
            ):
                hi += 1
            lower[row] = lo
            upper[row] = hi
    return (lower^, upper^)
