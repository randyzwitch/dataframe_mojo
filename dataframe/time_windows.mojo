"""Calendar-aware bounds shared by rolling expressions and time grouping."""
from .column import Column
from .dtype import DataType
from .series import Series
from .temporal import parse_interval, zone_of, to_local, relocalize
from .temporal_kernels import _offset, _offset_in, _truncate
from .timezone import TimeZone


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


def check_time_index(index: Series, groups: List[List[Int]]) raises:
    if not (index.dtype().is_date() or index.dtype().is_datetime()):
        raise Error(
            "time windows require a Date or Datetime index: " + index.name()
        )
    ref values = index._data[Column[Int64]]
    for rows in groups:
        for j in range(len(rows)):
            var row = rows[j]
            if not values._valid(row):
                raise Error("time window index contains nulls: " + index.name())
            if j > 0 and values._get(row) < values._get(rows[j - 1]):
                raise Error(
                    "time window index '"
                    + index.name()
                    + "' is not sorted within its group (row "
                    + String(row)
                    + ")"
                )


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
    groups: List[List[Int]],
    period: String,
    offset: String,
    closed: String,
) raises -> Tuple[List[Int], List[Int]]:
    positive_period(period)
    _ = parse_interval(offset)
    var sides = closed_sides(closed)
    check_time_index(index, groups)
    var lower = List[Int](length=len(index), fill=0)
    var upper = List[Int](length=len(index), fill=0)
    var dtype = index.dtype()
    var zone = zone_of(dtype)
    ref values = index._data[Column[Int64]]
    var default_offset = offset == negate_period(period)
    for rows in groups:
        for row in rows:
            var time = values._get(row)
            var start = shift_time(time, dtype, offset, zone)
            # Calendar month subtraction then addition need not round-trip.
            var end = time if default_offset else shift_time(
                start, dtype, period, zone
            )
            var lo = search_time(values, rows, start, not sides[0])
            var hi = search_time(values, rows, end, sides[1])
            lower[row] = lo
            upper[row] = max(lo, hi)
    return (lower^, upper^)
