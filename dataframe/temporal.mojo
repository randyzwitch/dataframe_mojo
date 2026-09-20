"""Calendar arithmetic, parsing, and formatting for temporal types.

Dates are days since 1970-01-01 in the proleptic Gregorian calendar;
datetimes are time-zone-naive ticks (ns, us, or ms) since that midnight; times
are nanoseconds since midnight. Negative values count backwards, using floor
division so fields of pre-1970 instants are correct. Conversions between
units are overflow-checked. There is no time zone or DST handling.

Parsing and formatting support these directives: %Y (year, optional sign),
%m, %d, %H, %M, %S (two digits when formatting, one or two when parsing), %f
(fractional seconds: up to nine digits when parsing, the unit's precision
when formatting), %j (day of year), and %%. Any other character must match
literally.
"""
from .dtype import DataType

comptime SECONDS_PER_DAY = Int64(86400)
comptime NANOS_PER_DAY = Int64(86400000000000)
comptime INT64_MAX = Int64(9223372036854775807)
comptime INT64_MIN = Int64(-9223372036854775807) - 1


def floor_div(a: Int64, b: Int64) -> Int64:
    return a // b


def floor_mod(a: Int64, b: Int64) -> Int64:
    return a % b


def checked_mul(a: Int64, b: Int64) raises -> Int64:
    if a != 0 and b != 0:
        var limit = INT64_MAX // (b if b > 0 else -b)
        if (a if a > 0 else -a) > limit or a == INT64_MIN:
            raise Error("temporal value overflows Int64")
    return a * b


def checked_add(a: Int64, b: Int64) raises -> Int64:
    if (b > 0 and a > INT64_MAX - b) or (b < 0 and a < INT64_MIN - b):
        raise Error("temporal value overflows Int64")
    return a + b


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


def ticks_per_day(dtype: DataType) -> Int64:
    return SECONDS_PER_DAY * dtype.per_second()


def convert_units(
    value: Int64, source: DataType, target: DataType
) raises -> Int64:
    """Rescale ticks between units: finer is exact (checked), coarser floors."""
    var a = source.per_second()
    var b = target.per_second()
    if a == b:
        return value
    if b > a:
        return checked_mul(value, b // a)
    return floor_div(value, a // b)


@fieldwise_init
struct Parts(Copyable):
    """Broken-down fields of a date, datetime, or time."""

    var year: Int64
    var month: Int64
    var day: Int64
    var hour: Int64
    var minute: Int64
    var second: Int64
    var nanos: Int64


def split(value: Int64, dtype: DataType) -> Parts:
    """Fields of a stored temporal value (dates have zero time fields)."""
    if dtype.is_date():
        var ymd = civil_from_days(value)
        return Parts(ymd[0], ymd[1], ymd[2], 0, 0, 0, 0)
    var per_day = NANOS_PER_DAY if dtype.is_time() else ticks_per_day(dtype)
    var days = Int64(0) if dtype.is_time() else floor_div(value, per_day)
    var within = floor_mod(value, per_day)
    var per_second = dtype.per_second()
    var seconds = within // per_second
    var nanos = (within % per_second) * (1000000000 // per_second)
    var ymd = civil_from_days(days)
    return Parts(
        ymd[0] if not dtype.is_time() else 1970,
        ymd[1] if not dtype.is_time() else 1,
        ymd[2] if not dtype.is_time() else 1,
        seconds // 3600,
        (seconds // 60) % 60,
        seconds % 60,
        nanos,
    )


def join(parts: Parts, dtype: DataType) raises -> Int64:
    """Stored value for validated fields."""
    if parts.month < 1 or parts.month > 12:
        raise Error("month out of range")
    if parts.day < 1 or parts.day > days_in_month(parts.year, parts.month):
        raise Error("day out of range")
    if parts.hour > 23 or parts.minute > 59 or parts.second > 59:
        raise Error("time of day out of range")
    var days = days_from_civil(parts.year, parts.month, parts.day)
    if dtype.is_date():
        return days
    var seconds = (parts.hour * 60 + parts.minute) * 60 + parts.second
    if dtype.is_time():
        return seconds * 1000000000 + parts.nanos
    var per_second = dtype.per_second()
    var fraction = parts.nanos // (1000000000 // per_second)
    var total = checked_add(checked_mul(days, SECONDS_PER_DAY), seconds)
    return checked_add(checked_mul(total, per_second), fraction)


def _digits(text: String, mut pos: Int, least: Int, most: Int) raises -> Int64:
    var bytes = text.as_bytes()
    var value = Int64(0)
    var count = 0
    while (
        count < most
        and pos < len(bytes)
        and bytes[pos] >= 48
        and bytes[pos] <= 57
    ):
        value = value * 10 + Int64(bytes[pos] - 48)
        pos += 1
        count += 1
    if count < least:
        raise Error("expected a number")
    return value


def default_format(dtype: DataType) -> String:
    if dtype.is_date():
        return "%Y-%m-%d"
    if dtype.is_time():
        return "%H:%M:%S%f"
    return "%Y-%m-%d %H:%M:%S%f"


def parse(text: String, dtype: DataType, format: String = "") raises -> Int64:
    """Parse text as a date, datetime, or time.

    With no format, dates accept YYYY-MM-DD; datetimes accept a date with an
    optional time "T" or " " HH:MM[:SS[.fraction]]; times accept
    HH:MM[:SS[.fraction]]. With a format, every directive must match.
    """
    try:
        if format.byte_length() > 0:
            return _parse_format(text, dtype, format)
        return _parse_iso(text, dtype)
    except e:
        raise Error(
            "cannot parse '" + text + "' as " + dtype.name() + ": " + String(e)
        )


def _parse_iso(text: String, dtype: DataType) raises -> Int64:
    var bytes = text.as_bytes()
    var pos = 0
    var parts = Parts(1970, 1, 1, 0, 0, 0, 0)
    if not dtype.is_time():
        var negative = pos < len(bytes) and bytes[pos] == 45
        if negative or (pos < len(bytes) and bytes[pos] == 43):
            pos += 1
        parts.year = _digits(text, pos, 4, 6) * (
            Int64(-1) if negative else Int64(1)
        )
        _expect(text, pos, 45)
        parts.month = _digits(text, pos, 2, 2)
        _expect(text, pos, 45)
        parts.day = _digits(text, pos, 2, 2)
        if dtype.is_date():
            _end(text, pos)
            return join(parts, dtype)
        if pos == len(bytes):
            return join(parts, dtype)
        if bytes[pos] != 84 and bytes[pos] != 32:
            raise Error("expected 'T' or ' ' before the time")
        pos += 1
    parts.hour = _digits(text, pos, 2, 2)
    _expect(text, pos, 58)
    parts.minute = _digits(text, pos, 2, 2)
    if pos < len(bytes) and bytes[pos] == 58:
        pos += 1
        parts.second = _digits(text, pos, 2, 2)
        parts.nanos = _fraction(text, pos)
    var offset = _zone(text, pos) if dtype.is_datetime() else Int64(0)
    _end(text, pos)
    var value = join(parts, dtype)
    if offset != 0:
        # Datetimes are time-zone-naive and hold UTC, so an offset is
        # applied here rather than remembered: 12:00+01:00 is 11:00 UTC.
        value = checked_add(
            value, checked_mul(-offset * 60, dtype.per_second())
        )
    return value


def _zone(text: String, mut pos: Int) raises -> Int64:
    """A trailing ISO 8601 zone designator, as minutes east of UTC.

    Accepts "Z" (and lowercase "z") for UTC, and "+HH:MM", "-HH:MM",
    "+HHMM", "+HH". Returns 0 when there is no designator, which is the
    naive case. The caller applies the shift; nothing here remembers the
    zone, because datetimes in this library are naive UTC.
    """
    var bytes = text.as_bytes()
    if pos >= len(bytes):
        return 0
    if bytes[pos] == 90 or bytes[pos] == 122:  # Z or z
        pos += 1
        return 0
    if bytes[pos] != 43 and bytes[pos] != 45:
        return 0
    var negative = bytes[pos] == 45
    pos += 1
    var hours = _digits(text, pos, 2, 2)
    var minutes = Int64(0)
    if pos < len(bytes) and bytes[pos] == 58:
        pos += 1
        minutes = _digits(text, pos, 2, 2)
    elif pos < len(bytes) and bytes[pos] >= 48 and bytes[pos] <= 57:
        minutes = _digits(text, pos, 2, 2)
    if hours > 23 or minutes > 59:
        raise Error("time zone offset out of range")
    var total = hours * 60 + minutes
    return -total if negative else total


def _fraction(text: String, mut pos: Int) raises -> Int64:
    var bytes = text.as_bytes()
    if pos >= len(bytes) or bytes[pos] != 46:
        return 0
    pos += 1
    var start = pos
    var value = _digits(text, pos, 1, 9)
    var scale = 9 - (pos - start)
    for _ in range(scale):
        value *= 10
    return value


def _expect(text: String, mut pos: Int, byte: UInt8) raises:
    var bytes = text.as_bytes()
    if pos >= len(bytes) or bytes[pos] != byte:
        raise Error("expected '" + chr(Int(byte)) + "'")
    pos += 1


def _end(text: String, pos: Int) raises:
    if pos != text.byte_length():
        raise Error("unexpected trailing text")


def _parse_format(
    text: String, dtype: DataType, format: String
) raises -> Int64:
    var f = format.as_bytes()
    var t = text.as_bytes()
    var pos = 0
    var i = 0
    var parts = Parts(1970, 1, 1, 0, 0, 0, 0)
    var ordinal = Int64(-1)
    while i < len(f):
        if f[i] == 37 and i + 1 < len(f):
            var d = f[i + 1]
            i += 2
            # With no literal between this directive and the next, the field
            # has nothing to delimit it, so it must take exactly its width:
            # "%Y%m%d" over "20240228" is 4 then 2 then 2, not a greedy year.
            var packed = i + 1 < len(f) and f[i] == 37
            if d == 89:  # Y
                var negative = pos < len(t) and t[pos] == 45
                if negative or (pos < len(t) and t[pos] == 43):
                    pos += 1
                parts.year = _digits(
                    text, pos, 4 if packed else 1, 4 if packed else 6
                ) * (Int64(-1) if negative else Int64(1))
            elif d == 109:  # m
                parts.month = _digits(text, pos, 2 if packed else 1, 2)
            elif d == 100:  # d
                parts.day = _digits(text, pos, 2 if packed else 1, 2)
            elif d == 72:  # H
                parts.hour = _digits(text, pos, 2 if packed else 1, 2)
            elif d == 77:  # M
                parts.minute = _digits(text, pos, 2 if packed else 1, 2)
            elif d == 83:  # S
                parts.second = _digits(text, pos, 2 if packed else 1, 2)
            elif d == 102:  # f
                if pos < len(t) and t[pos] == 46:
                    parts.nanos = _fraction(text, pos)
            elif d == 106:  # j
                ordinal = _digits(text, pos, 3 if packed else 1, 3)
            elif d == 37:
                _expect(text, pos, 37)
            else:
                raise Error("unsupported format directive %" + chr(Int(d)))
        else:
            _expect(text, pos, f[i])
            i += 1
    _end(text, pos)
    if ordinal >= 0:
        var days_in_year = Int64(366 if is_leap(parts.year) else 365)
        if ordinal < 1 or ordinal > days_in_year:
            raise Error("day of year out of range")
        var ymd = civil_from_days(
            days_from_civil(parts.year, 1, 1) + ordinal - 1
        )
        parts.month = ymd[1]
        parts.day = ymd[2]
    return join(parts, dtype)


def _pad(value: Int64, width: Int) -> String:
    var text = String(value if value >= 0 else -value)
    var out = String("-") if value < 0 else String()
    for _ in range(width - text.byte_length()):
        out += "0"
    return out + text


def format(value: Int64, dtype: DataType, pattern: String = "") -> String:
    """Render a stored temporal value (see the module docstring)."""
    if dtype.is_duration():
        return _format_duration(value, dtype)
    var p = split(value, dtype)
    var f = (
        pattern if pattern.byte_length() > 0 else default_format(dtype)
    ).as_bytes()
    var out = String()
    var i = 0
    while i < len(f):
        if f[i] == 37 and i + 1 < len(f):
            var d = f[i + 1]
            i += 2
            if d == 89:
                out += _pad(p.year, 4)
            elif d == 109:
                out += _pad(p.month, 2)
            elif d == 100:
                out += _pad(p.day, 2)
            elif d == 72:
                out += _pad(p.hour, 2)
            elif d == 77:
                out += _pad(p.minute, 2)
            elif d == 83:
                out += _pad(p.second, 2)
            elif d == 102:
                out += _format_fraction(p.nanos, dtype)
            elif d == 106:
                out += _pad(
                    days_from_civil(p.year, p.month, p.day)
                    - days_from_civil(p.year, 1, 1)
                    + 1,
                    3,
                )
            else:
                out += "%" + chr(Int(d))
        else:
            out += chr(Int(f[i]))
            i += 1
    return out^


def _format_fraction(nanos: Int64, dtype: DataType) -> String:
    """ ".fff", ".ffffff", or ".fffffffff" per unit; empty when zero."""
    if nanos == 0:
        return ""
    var digits = 9
    if dtype.per_second() == 1000000:
        digits = 6
    elif dtype.per_second() == 1000:
        digits = 3
    var scaled = nanos // (Int64(10) ** Int64(9 - digits))
    return "." + _pad(scaled, digits)


def _format_duration(value: Int64, dtype: DataType) -> String:
    """Compact form such as 1d 2h 3m 4.5s, or 0s; negative values lead
    with a minus sign."""
    if value == 0:
        return "0s"
    var negative = value < 0
    var ticks = -value if negative else value
    var per_second = dtype.per_second()
    var seconds = ticks // per_second
    var fraction = ticks % per_second
    var out = String("-") if negative else String()
    var days = seconds // 86400
    var hours = (seconds // 3600) % 24
    var minutes = (seconds // 60) % 60
    var secs = seconds % 60
    var pieces = List[String]()
    if days > 0:
        pieces.append(String(days) + "d")
    if hours > 0:
        pieces.append(String(hours) + "h")
    if minutes > 0:
        pieces.append(String(minutes) + "m")
    if secs > 0 or fraction > 0:
        var text = String(secs)
        if fraction > 0:
            var digits = 9 if per_second == 1000000000 else (
                6 if per_second == 1000000 else 3
            )
            var f = _pad(fraction, digits)
            var end = f.byte_length()
            while end > 0 and f.as_bytes()[end - 1] == 48:
                end -= 1
            var trimmed = String(f[byte=0:end])
            f = trimmed^
            text += "." + f
        pieces.append(text + "s")
    for i in range(len(pieces)):
        if i > 0:
            out += " "
        out += pieces[i]
    return out^


def parse_every(every: String) raises -> Tuple[Int64, Int64]:
    """Parse an interval like "3d", "1h30m", "2mo", or "1y" into
    (months, nanoseconds). Units: ns, us, ms, s, m, h, d, w, mo, y."""
    var bytes = every.as_bytes()
    if len(bytes) == 0:
        raise Error("empty interval")
    var pos = 0
    var negative = bytes[0] == 45
    if negative:
        pos = 1
    var months = Int64(0)
    var nanos = Int64(0)
    while pos < len(bytes):
        var amount = _digits(every, pos, 1, 18)
        var start = pos
        while pos < len(bytes) and (bytes[pos] < 48 or bytes[pos] > 57):
            pos += 1
        var unit = String(every[byte=start:pos])
        if unit == "y":
            months = checked_add(months, checked_mul(amount, 12))
        elif unit == "mo":
            months = checked_add(months, amount)
        else:
            var scale: Int64
            if unit == "ns":
                scale = 1
            elif unit == "us":
                scale = 1000
            elif unit == "ms":
                scale = 1000000
            elif unit == "s":
                scale = 1000000000
            elif unit == "m":
                scale = 60000000000
            elif unit == "h":
                scale = 3600000000000
            elif unit == "d":
                scale = NANOS_PER_DAY
            elif unit == "w":
                scale = 7 * NANOS_PER_DAY
            else:
                raise Error("unknown interval unit '" + unit + "' in " + every)
            nanos = checked_add(nanos, checked_mul(amount, scale))
    if negative:
        return (-months, -nanos)
    return (months, nanos)


def add_months(parts: Parts, months: Int64) -> Parts:
    """Shift by calendar months, clamping the day to the target month."""
    var index = parts.year * 12 + (parts.month - 1) + months
    var year = floor_div(index, 12)
    var month = floor_mod(index, 12) + 1
    var day = min(parts.day, days_in_month(year, month))
    return Parts(
        year, month, day, parts.hour, parts.minute, parts.second, parts.nanos
    )
