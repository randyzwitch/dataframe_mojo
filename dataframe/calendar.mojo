"""Proleptic Gregorian calendar arithmetic on days since 1970-01-01.

Shared by temporal.mojo and timezone.mojo; it imports nothing from the
package so both can use it without an import cycle.
"""


def floor_div(a: Int64, b: Int64) -> Int64:
    return a // b


def floor_mod(a: Int64, b: Int64) -> Int64:
    return a % b


def days_from_civil(year: Int64, month: Int64, day: Int64) -> Int64:
    """Days since 1970-01-01 for a proleptic Gregorian date."""
    var y = year - (Int64(1) if month <= 2 else Int64(0))
    var era = floor_div(y, 400)
    var yoe = y - era * 400
    var mp = (month + 9) % 12
    var doy = (153 * mp + 2) // 5 + day - 1
    var doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def civil_from_days(days: Int64) -> Tuple[Int64, Int64, Int64]:
    """(year, month, day) for days since 1970-01-01."""
    var z = days + 719468
    var era = floor_div(z, 146097)
    var doe = z - era * 146097
    var yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    var doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    var mp = (5 * doy + 2) // 153
    var day = doy - (153 * mp + 2) // 5 + 1
    var month = mp + 3 if mp < 10 else mp - 9
    var year = yoe + era * 400 + (Int64(1) if month <= 2 else Int64(0))
    return (year, month, day)


def is_leap(year: Int64) -> Bool:
    return (year % 4 == 0 and year % 100 != 0) or year % 400 == 0


def days_in_month(year: Int64, month: Int64) -> Int64:
    if month == 2:
        return 29 if is_leap(year) else 28
    if month == 4 or month == 6 or month == 9 or month == 11:
        return 30
    return 31
