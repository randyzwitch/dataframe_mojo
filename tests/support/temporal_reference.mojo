"""Gregorian and interval membership reference rules, independent of dataframe."""


def civil_days(year: Int, month: Int, day: Int) -> Int64:
    """Count whole civil days from 0001-01-01, then subtract the epoch ordinal."""
    var previous = year - 1
    var days = (
        365 * previous + previous // 4 - previous // 100 + previous // 400
    )
    var lengths: List[Int] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    for index in range(month - 1):
        days += lengths[index]
    if month > 2 and year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
        days += 1
    return Int64(days + day - 1 - 719162)


def civil_ns(
    year: Int,
    month: Int,
    day: Int,
    hour: Int = 0,
    minute: Int = 0,
    second: Int = 0,
    nanosecond: Int64 = 0,
) -> Int64:
    return (
        civil_days(year, month, day) * 86400000000000
        + Int64(hour * 3600 + minute * 60 + second) * 1000000000
        + nanosecond
    )


def members(
    times: List[Int64],
    start: Int64,
    end: Int64,
    closed: String,
    partition: List[Int] = List[Int](),
    key: Int = 0,
) -> List[Int]:
    """Scan every row; duplicate timestamps retain their full multiplicity."""
    var result = List[Int]()
    for row in range(len(times)):
        if len(partition) and partition[row] != key:
            continue
        var left = times[row] > start or (
            times[row] == start and closed in ["left", "both"]
        )
        var right = times[row] < end or (
            times[row] == end and closed in ["right", "both"]
        )
        if left and right:
            result.append(row)
    return result^


def sum_members(
    values: List[Int64], valid: List[Bool], rows: List[Int]
) -> Int64:
    var total: Int64 = 0
    for row in rows:
        if valid[row]:
            total += values[row]
    return total
