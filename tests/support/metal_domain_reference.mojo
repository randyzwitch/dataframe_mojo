"""Scalar fixture rules independent of dataframe and native lowering."""


def value(row: Int) -> Int:
    return row % 17 + 1


def valid(row: Int) -> Bool:
    return row % 7 != 3


def group_key(row: Int) -> Int:
    return -1 if row % 11 == 5 else row % 4


def groups(rows: List[Int]) -> List[List[Int]]:
    var keys = List[Int]()
    var result = List[List[Int]]()
    for row in rows:
        var key = group_key(row)
        var found = -1
        for at in range(len(keys)):
            if keys[at] == key:
                found = at
                break
        if found < 0:
            keys.append(key)
            result.append(List[Int]())
            found = len(keys) - 1
        result[found].append(row)
    return result^


def selected(n: Int, filtered: Bool = False) -> List[Int]:
    var result = List[Int]()
    for row in range(5, 5 + n):
        if not filtered or group_key(row) >= 2:
            result.append(row)
    return result^


def slice_positions(rows: List[Int], offset: Int, length: Int) -> List[Int]:
    var start = min(max(offset, 0), len(rows))
    if offset < 0:
        # Avoid negating Int.MIN or adding it to an unsigned quantity.
        start = 0 if offset < -len(rows) else len(rows) + offset
    var count = len(rows) - start if length == -1 else min(
        length, len(rows) - start
    )
    var result = List[Int]()
    for at in range(count):
        result.append(rows[start + at])
    return result^


def distinct_positions(keys: List[Int], keep: String) -> List[Int]:
    var out = List[Int]()
    for at in range(len(keys)):
        var before = False
        var after = False
        for other in range(len(keys)):
            if keys[at] == keys[other]:
                before = before or other < at
                after = after or other > at
        if (
            (keep == "last" and not after)
            or (keep == "none" and not before and not after)
            or (keep != "last" and keep != "none" and not before)
        ):
            out.append(at)
    return out^


def floor128(value: Int128, divisor: Int128) -> Int128:
    if divisor < 0:
        return floor128(-value, -divisor)
    return value // divisor if value >= 0 else -1 - ((-value - 1) // divisor)


def calendar(days: Int64) -> List[Int64]:
    # Count years and months inside a Gregorian 400-year cycle. Int128 keeps
    # the epoch shift defined even at the full signed day-storage endpoints.
    var ordinal = Int128(days) + 719162
    var era = floor128(ordinal, 146097)
    var remaining = ordinal - era * 146097
    var year = Int64(era * 400 + 1)
    while True:
        var leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
        var length = Int128(366 if leap else 365)
        if remaining < length:
            break
        remaining -= length
        year += 1
    var day_of_year = Int64(remaining + 1)
    var lengths: List[Int] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
        lengths[1] = 29
    var month = 0
    while remaining >= Int128(lengths[month]):
        remaining -= Int128(lengths[month])
        month += 1
    var weekday = Int64(
        Int128(days) + 3 - floor128(Int128(days) + 3, 7) * 7 + 1
    )
    return [year, Int64(month + 1), Int64(remaining + 1), weekday, day_of_year]
