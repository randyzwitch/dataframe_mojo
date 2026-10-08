"""Calendar arithmetic, parsing, and formatting for temporal types.

Dates are days since 1970-01-01 in the proleptic Gregorian calendar;
datetimes are ticks (ns, us, or ms) since that midnight; times are
nanoseconds since midnight. Negative values count backwards, using floor
division so fields of pre-1970 instants are correct. Conversions between
units are overflow-checked.

A naive datetime is a wall-clock reading. A zone-aware one holds UTC ticks
and is shown, parsed and split into fields at its zone's local time (see
timezone.mojo); `to_local` and `localize` convert between the two.

Parsing and formatting support these directives: %Y (year, optional sign),
%m, %d, %H, %M, %S (two digits when formatting, one or two when parsing), %f
(fractional seconds: up to nine digits when parsing, the unit's precision
when formatting), %j (day of year), %z (UTC offset: +HHMM when formatting;
Z, +HH, +HHMM or +HH:MM when parsing), %:z (+HH:MM), %Z (zone
abbreviation, formatting only), and %%. Any other character must match
literally. A parsed offset makes the value UTC.
"""
from .calendar import (
    civil_from_days,
    days_from_civil,
    days_in_month,
    floor_div,
    floor_mod,
    is_leap,
)
from .dtype import DataType
from .timezone import TimeZone

comptime SECONDS_PER_DAY = Int64(86400)
comptime NANOS_PER_DAY = Int64(86400000000000)
comptime INT64_MAX = Int64(9223372036854775807)
comptime INT64_MIN = Int64(-9223372036854775807) - 1


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
    if dtype.time_zone().byte_length() > 0:
        return "%Y-%m-%d %H:%M:%S%f%:z"
    return "%Y-%m-%d %H:%M:%S%f"


# How localize treats a local time that occurs twice (clocks fall back) or
# never (clocks spring forward), as in Polars' `ambiguous` and
# `non_existent` options.
comptime AMBIGUOUS_RAISE = 0
comptime AMBIGUOUS_EARLIEST = 1
comptime AMBIGUOUS_LATEST = 2
comptime AMBIGUOUS_NULL = 3
comptime NON_EXISTENT_RAISE = 0
comptime NON_EXISTENT_NULL = 1


def ambiguous_code(name: String) raises -> Int:
    if name == "raise":
        return AMBIGUOUS_RAISE
    if name == "earliest":
        return AMBIGUOUS_EARLIEST
    if name == "latest":
        return AMBIGUOUS_LATEST
    if name == "null":
        return AMBIGUOUS_NULL
    raise Error(
        "ambiguous must be 'raise', 'earliest', 'latest' or 'null', found '"
        + name
        + "'"
    )


def non_existent_code(name: String) raises -> Int:
    if name == "raise":
        return NON_EXISTENT_RAISE
    if name == "null":
        return NON_EXISTENT_NULL
    raise Error("non_existent must be 'raise' or 'null', found '" + name + "'")


def zone_of(dtype: DataType) raises -> TimeZone:
    """The zone of a zone-aware datetime; UTC for other types."""
    var name = dtype.time_zone()
    if name.byte_length() == 0:
        return TimeZone.utc()
    return TimeZone.load(name)


def to_local(value: Int64, dtype: DataType, zone: TimeZone) -> Int64:
    """The wall-clock ticks in `zone` of UTC ticks `value`."""
    var per_second = dtype.per_second()
    return value + zone.offset_at(floor_div(value, per_second)) * per_second


def localize(
    local: Int64,
    dtype: DataType,
    zone: TimeZone,
    ambiguous: Int,
    non_existent: Int,
) raises -> Optional[Int64]:
    """The UTC ticks of wall-clock ticks `local` in `zone`; None when the
    options ask for null. Raises, as Polars does, for a non-existent or
    ambiguous time under the "raise" options."""
    var per_second = dtype.per_second()
    var seconds = floor_div(local, per_second)
    var fraction = local - seconds * per_second
    var found = zone.resolve(seconds)
    var instant = found[1]
    if found[0] == 0:
        if non_existent == NON_EXISTENT_NULL:
            return None
        raise Error(
            "datetime '"
            + _naive_text(local, dtype)
            + "' is non-existent in time zone '"
            + zone.name
            + "'; use non_existent='null' to return null instead"
        )
    if found[0] > 1:
        if ambiguous == AMBIGUOUS_NULL:
            return None
        if ambiguous == AMBIGUOUS_RAISE:
            raise Error(
                "datetime '"
                + _naive_text(local, dtype)
                + "' is ambiguous in time zone '"
                + zone.name
                + "'; use ambiguous='earliest', 'latest' or 'null'"
            )
        if ambiguous == AMBIGUOUS_LATEST:
            instant = found[2]
    return checked_add(checked_mul(instant, per_second), fraction)


def relocalize(
    local: Int64, original: Int64, dtype: DataType, zone: TimeZone
) raises -> Int64:
    """UTC ticks for wall-clock ticks `local` produced by shifting the value
    at UTC ticks `original` in local time (see TimeZone.relocalize)."""
    var per_second = dtype.per_second()
    var seconds = floor_div(local, per_second)
    var instant = zone.relocalize(seconds, floor_div(original, per_second))
    return checked_add(
        checked_mul(instant, per_second), local - seconds * per_second
    )


def _naive_text(local: Int64, dtype: DataType) -> String:
    return _render(local, dtype, "%Y-%m-%d %H:%M:%S%f", 0, "", False)


def strptime_target(target: DataType, format: String) raises -> DataType:
    """The dtype strptime produces: a naive datetime format with %z gives
    UTC-aware values, as in Polars."""
    if (
        target.is_datetime()
        and target.time_zone().byte_length() == 0
        and (format.find("%z") >= 0 or format.find("%:z") >= 0)
    ):
        return target.with_time_zone("UTC")
    return target


def parse(text: String, dtype: DataType, format: String = "") raises -> Int64:
    """Parse text as a date, datetime, or time.

    With no format, dates accept YYYY-MM-DD; datetimes accept a date with an
    optional time "T" or " " HH:MM[:SS[.fraction]] and an optional UTC
    offset; times accept HH:MM[:SS[.fraction]]. With a format, every
    directive must match. Loads a zone-aware dtype's zone on every call;
    use `parse_in` for many values.
    """
    return parse_in(text, dtype, format, zone_of(dtype))


def parse_in(
    text: String, dtype: DataType, format: String, zone: TimeZone
) raises -> Int64:
    """Parse as `parse` does. Text with an offset is converted to UTC; for a
    zone-aware dtype, text without one is a wall-clock time in `zone`, and
    raises when that time is non-existent or ambiguous there."""
    var seen = False
    var value: Int64
    try:
        if format.byte_length() > 0:
            value = _parse_format(text, dtype, format, seen)
        else:
            value = _parse_iso(text, dtype, seen)
    except e:
        raise Error(
            "cannot parse '" + text + "' as " + dtype.name() + ": " + String(e)
        )
    if seen or dtype.time_zone().byte_length() == 0:
        return value
    return localize(
        value, dtype, zone, AMBIGUOUS_RAISE, NON_EXISTENT_RAISE
    ).value()


def has_offset(text: String) -> Bool:
    """Whether ISO text for a datetime parses with a UTC offset or Z."""
    var seen = False
    try:
        _ = _parse_iso(text, DataType.datetime("us"), seen)
    except:
        return False
    return seen


def _parse_iso(text: String, dtype: DataType, mut seen: Bool) raises -> Int64:
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
    var offset = _zone(text, pos, seen) if dtype.is_datetime() else Int64(0)
    _end(text, pos)
    var value = join(parts, dtype)
    if offset != 0:
        # An offset makes the value UTC: 12:00+01:00 is 11:00 UTC.
        value = checked_add(
            value, checked_mul(-offset * 60, dtype.per_second())
        )
    return value


def _zone(text: String, mut pos: Int, mut seen: Bool) raises -> Int64:
    """A trailing ISO 8601 zone designator, as minutes east of UTC.

    Accepts "Z" (and lowercase "z") for UTC, and "+HH:MM", "-HH:MM",
    "+HHMM", "+HH". Returns 0 and leaves `seen` alone when there is no
    designator; sets `seen` when there is one.
    """
    var bytes = text.as_bytes()
    if pos >= len(bytes):
        return 0
    if bytes[pos] == 90 or bytes[pos] == 122:  # Z or z
        pos += 1
        seen = True
        return 0
    if bytes[pos] != 43 and bytes[pos] != 45:
        return 0
    seen = True
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
    text: String, dtype: DataType, format: String, mut seen: Bool
) raises -> Int64:
    var f = format.as_bytes()
    var t = text.as_bytes()
    var pos = 0
    var i = 0
    var parts = Parts(1970, 1, 1, 0, 0, 0, 0)
    var ordinal = Int64(-1)
    var offset = Int64(0)
    while i < len(f):
        if f[i] == 37 and i + 1 < len(f):
            var d = f[i + 1]
            i += 2
            if d == 58 and i < len(f) and f[i] == 122:  # %:z
                d = 122
                i += 1
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
            elif d == 122:  # z
                var before = seen
                seen = False
                offset = _zone(text, pos, seen)
                if not seen:
                    raise Error("expected a UTC offset")
                seen = seen or before
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
    var value = join(parts, dtype)
    if offset != 0 and dtype.is_datetime():
        value = checked_add(
            value, checked_mul(-offset * 60, dtype.per_second())
        )
    return value


def _pad(value: Int64, width: Int) -> String:
    var text = String(value if value >= 0 else -value)
    var out = String("-") if value < 0 else String()
    for _ in range(width - text.byte_length()):
        out += "0"
    return out + text


def format(value: Int64, dtype: DataType, pattern: String = "") -> String:
    """Render a stored temporal value (see the module docstring). Loads a
    zone-aware dtype's zone on every call; use `format_in` for many."""
    var zone: TimeZone
    try:
        zone = zone_of(dtype)
    except:
        # The zone was valid when the dtype was made; if its file has
        # since gone, show UTC rather than fail to print.
        zone = TimeZone.utc()
    return format_in(value, dtype, pattern, zone)


def format_in(
    value: Int64, dtype: DataType, pattern: String, zone: TimeZone
) -> String:
    """Render as `format` does; a zone-aware value at `zone`'s local time."""
    if dtype.is_duration():
        return _format_duration(value, dtype)
    if not dtype.is_datetime() or dtype.time_zone().byte_length() == 0:
        return _render(value, dtype, pattern, 0, "", False)
    var per_second = dtype.per_second()
    var type = zone.type_at(floor_div(value, per_second))
    var offset = zone.offset_of(type)
    return _render(
        value + offset * per_second,
        dtype,
        pattern,
        offset,
        zone.abbreviation(type),
        True,
    )


def _offset_text(offset: Int64, colon: Bool) -> String:
    var minutes = (offset if offset >= 0 else -offset) // 60
    return (
        ("+" if offset >= 0 else "-")
        + _pad(minutes // 60, 2)
        + (":" if colon else "")
        + _pad(minutes % 60, 2)
    )


def _render(
    value: Int64,
    dtype: DataType,
    pattern: String,
    offset: Int64,
    abbreviation: String,
    aware: Bool,
) -> String:
    """Render wall-clock `value`; %z, %:z and %Z are empty unless aware."""
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
            elif d == 122:  # z
                if aware:
                    out += _offset_text(offset, False)
            elif d == 58 and i < len(f) and f[i] == 122:  # %:z
                i += 1
                if aware:
                    out += _offset_text(offset, True)
            elif d == 90:  # Z
                if aware:
                    out += abbreviation
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
    var interval = parse_interval(every)
    return (
        interval[0],
        checked_add(checked_mul(interval[1], NANOS_PER_DAY), interval[2]),
    )


def parse_interval(every: String) raises -> Tuple[Int64, Int64, Int64]:
    """Parse an interval into (months, days, nanoseconds), keeping calendar
    days (d, w) apart from fixed units: in a time zone a day is a local
    calendar day, which is 23 or 25 hours across a DST change."""
    var bytes = every.as_bytes()
    if len(bytes) == 0:
        raise Error("empty interval")
    var pos = 0
    var negative = bytes[0] == 45
    if negative:
        pos = 1
    var months = Int64(0)
    var days = Int64(0)
    var nanos = Int64(0)
    while pos < len(bytes):
        var amount = _digits(every, pos, 1, 18)
        var start = pos
        while pos < len(bytes) and (bytes[pos] < 48 or bytes[pos] > 57):
            pos += 1
        var unit = String(every[byte=start:pos])
        if unit == "y":
            months = checked_add(months, checked_mul(amount, 12))
        elif unit == "q":
            months = checked_add(months, checked_mul(amount, 3))
        elif unit == "mo":
            months = checked_add(months, amount)
        elif unit == "d":
            days = checked_add(days, amount)
        elif unit == "w":
            days = checked_add(days, checked_mul(amount, 7))
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
            else:
                raise Error("unknown interval unit '" + unit + "' in " + every)
            nanos = checked_add(nanos, checked_mul(amount, scale))
    if negative:
        return (-months, -days, -nanos)
    return (months, days, nanos)


def add_months(parts: Parts, months: Int64) -> Parts:
    """Shift by calendar months, clamping the day to the target month."""
    var index = parts.year * 12 + (parts.month - 1) + months
    var year = floor_div(index, 12)
    var month = floor_mod(index, 12) + 1
    var day = min(parts.day, days_in_month(year, month))
    return Parts(
        year, month, day, parts.hour, parts.minute, parts.second, parts.nanos
    )
