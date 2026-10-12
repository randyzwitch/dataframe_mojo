"""Literal civil transition records and a piecewise offset model, no dataframe."""
from tests.support.temporal_reference import civil_days, civil_ns


@fieldwise_init
struct Transition(Copyable):
    var zone: String
    var utc: Int64
    var before: Int64
    var after: Int64

    def offset(self, utc: Int64) -> Int64:
        return self.before if utc < self.utc else self.after

    def candidates(self, wall: Int64) -> List[Int64]:
        var result = List[Int64]()
        for offset in [self.before, self.after]:
            var utc = wall - offset
            if self.offset(utc) == offset and utc not in result:
                result.append(utc)
        if len(result) == 2 and result[0] > result[1]:
            var first = result[0]
            result[0] = result[1]
            result[1] = first
        return result^


def transition(
    zone: String,
    year: Int,
    month: Int,
    day: Int,
    hour: Int,
    minute: Int,
    before: Int64,
    after: Int64,
) -> Transition:
    return Transition(
        zone,
        civil_ns(year, month, day, hour, minute) // 1000000000,
        before,
        after,
    )


def transitions() -> List[Transition]:
    return [
        transition("America/New_York", 2024, 3, 10, 7, 0, -18000, -14400),
        transition("America/New_York", 2024, 11, 3, 6, 0, -14400, -18000),
        transition("America/New_York", 2045, 3, 12, 7, 0, -18000, -14400),
        transition("America/New_York", 1918, 3, 31, 7, 0, -18000, -14400),
        transition("Europe/London", 2024, 3, 31, 1, 0, 0, 3600),
        transition("Europe/London", 2024, 10, 27, 1, 0, 3600, 0),
        transition("Europe/London", 1968, 2, 18, 2, 0, 0, 3600),
        transition("Australia/Lord_Howe", 2024, 10, 5, 15, 30, 37800, 39600),
        transition("Australia/Lord_Howe", 2024, 4, 6, 15, 0, 39600, 37800),
        transition("Pacific/Apia", 2011, 12, 30, 10, 0, -36000, 50400),
        transition("Europe/Moscow", 2014, 10, 25, 22, 0, 14400, 10800),
        transition("America/Caracas", 2016, 5, 1, 7, 0, -16200, -14400),
    ]


def floor_div(value: Int64, divisor: Int64) -> Int64:
    return value // divisor if value >= 0 else -1 - ((-value - 1) // divisor)


def fields(wall: Int64) -> List[Int64]:
    var days = floor_div(wall, 86400)
    var year = 1970
    while civil_days(year, 1, 1) > days:
        year -= 1
    while civil_days(year + 1, 1, 1) <= days:
        year += 1
    var month = 1
    while month < 12 and civil_days(year, month + 1, 1) <= days:
        month += 1
    var within = wall - days * 86400
    return [
        Int64(year),
        Int64(month),
        days - civil_days(year, month, 1) + 1,
        within // 3600,
        (within // 60) % 60,
        within % 60,
        days + 3 - floor_div(days + 3, 7) * 7 + 1,
        days - civil_days(year, 1, 1) + 1,
        days,
        within * 1000000000,
    ]


def truncate_wall(wall: Int64, period: String) -> Int64:
    var date = fields(wall)
    if period == "1mo":
        return civil_days(Int(date[0]), Int(date[1]), 1) * 86400
    var days = date[8]
    if period == "1w":
        days -= date[6] - 1
    return days * 86400


def two(value: Int64) -> String:
    return (String("0") if value < 10 else String("")) + String(value)


def formatted_offset(offset: Int64) -> String:
    var magnitude = abs(offset)
    var sign = String("+") if offset >= 0 else String("-")
    var hour = two(magnitude // 3600)
    var minute = two((magnitude // 60) % 60)
    return sign + hour + minute + " " + sign + hour + ":" + minute
