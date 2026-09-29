"""Time zones for zone-aware datetime columns (#222).

A zone-aware datetime stores UTC ticks since the epoch, as Arrow's timestamp
type does when its time zone is set; the zone only changes how values are
shown, parsed and broken into fields. Zone names follow Arrow: an IANA name
such as "America/New_York", "UTC", or a fixed offset "+HH:MM" / "-HH:MM".

IANA zones are read from the system time-zone database, the compiled TZif
files (RFC 8536) that tzdata installs. The first directory that exists wins:
`$TZDIR`, then `$CONDA_PREFIX/share/zoneinfo` (the conda `tzdata` package),
then `/usr/share/zoneinfo`. Reading the system copy keeps the package small
and picks up tzdata updates without a release, at the cost of results that
follow the machine's tzdata version; containers without tzdata need the
conda package or `TZDIR`. Instants after a file's last transition follow its
POSIX TZ footer rule, as the RFC requires. Leap-second ("right/") files are
not supported.
"""
from std.os import getenv

from .calendar import (
    civil_from_days,
    days_from_civil,
    days_in_month,
    floor_div,
    floor_mod,
    is_leap,
)

comptime _DEFAULT_TIME = Int64(7200)  # POSIX rules switch at 02:00


@fieldwise_init
struct _RuleDate(Copyable, ImplicitlyCopyable, Movable):
    """One switch of a POSIX TZ rule: Jn (kind 0, day 1-365 skipping Feb
    29), n (kind 1, day 0-365), or Mm.w.d (kind 2), at `seconds` local."""

    var kind: Int
    var month: Int64
    var week: Int64
    var day: Int64
    var seconds: Int64

    def day_in(self, year: Int64) -> Int64:
        """Days since the epoch of this switch in `year`."""
        var first = days_from_civil(year, 1, 1)
        if self.kind == 0:
            var extra = Int64(1) if is_leap(year) and self.day >= 60 else 0
            return first + self.day - 1 + extra
        if self.kind == 1:
            return first + self.day
        var start = days_from_civil(year, self.month, 1)
        # 1970-01-01 was a Thursday; POSIX counts Sunday as day 0.
        var weekday = floor_mod(start + 4, 7)
        var day = start + floor_mod(self.day - weekday, 7) + (self.week - 1) * 7
        var end = start + days_in_month(year, self.month)
        while day >= end:
            day -= 7
        return day


struct TimeZone(Copyable, Movable):
    """A zone's UTC offsets over time, loaded once per column operation."""

    var name: String
    # Transition instants (UTC seconds) and the type in force from each.
    var _times: List[Int64]
    var _types: List[Int]
    # Per local-time type: seconds east of UTC, abbreviation, and the
    # daylight-saving part of the offset (0 for standard time).
    var _offsets: List[Int64]
    var _names: List[String]
    var _saves: List[Int64]
    # Types of the footer rule's standard and daylight time (-1 if none),
    # used after the last transition.
    var _std: Int
    var _dst: Int
    var _start: _RuleDate
    var _end: _RuleDate
    # Each distinct offset once; local-to-UTC tries each of them.
    var _candidates: List[Int64]

    def __init__(out self, var name: String, offset: Int64, var label: String):
        """A fixed-offset zone (UTC, or an Arrow "+HH:MM" zone)."""
        self.name = name^
        self._times = List[Int64]()
        self._types = List[Int]()
        self._offsets = [offset]
        self._names = [label^]
        self._saves = [Int64(0)]
        self._std = -1
        self._dst = -1
        self._start = _RuleDate(0, 0, 0, 0, 0)
        self._end = _RuleDate(0, 0, 0, 0, 0)
        self._candidates = [offset]

    @staticmethod
    def utc() -> TimeZone:
        return TimeZone("UTC", 0, "UTC")

    @staticmethod
    def load(name: String) raises -> TimeZone:
        """The zone called `name`; raises for an unknown or invalid name."""
        var canonical = canonical_zone_syntax(name)
        if canonical == "UTC":
            return TimeZone.utc()
        var fixed = _fixed_offset(canonical)
        if fixed:
            return TimeZone(canonical, fixed.value(), canonical)
        var data = _read_zone_file(canonical)
        return _parse_tzif(canonical, data)

    def type_at(self, seconds: Int64) -> Int:
        """The local-time type in force at UTC `seconds`."""
        var n = len(self._times)
        if n > 0 and seconds < self._times[0]:
            return 0
        if n == 0 or seconds >= self._times[n - 1]:
            if self._std >= 0:
                return self._rule_type(seconds)
            return self._types[n - 1] if n > 0 else 0
        var low = 0
        var high = n - 1
        # Largest i with _times[i] <= seconds; _times[0] <= seconds holds.
        while low < high:
            var middle = (low + high + 1) // 2
            if self._times[middle] <= seconds:
                low = middle
            else:
                high = middle - 1
        return self._types[low]

    def offset_at(self, seconds: Int64) -> Int64:
        """Seconds east of UTC at UTC `seconds`."""
        return self._offsets[self.type_at(seconds)]

    def offset_of(self, type: Int) -> Int64:
        return self._offsets[type]

    def abbreviation(self, type: Int) -> String:
        return self._names[type]

    def save_at(self, seconds: Int64) -> Int64:
        """The daylight-saving part of the offset at UTC `seconds`, as
        chrono's `dst_offset` reports it (0 in standard time)."""
        return self._saves[self.type_at(seconds)]

    def _rule_type(self, seconds: Int64) -> Int:
        if self._dst < 0:
            return self._std
        var std = self._offsets[self._std]
        var dst = self._offsets[self._dst]
        var year = civil_from_days(floor_div(seconds + std, 86400))[0]
        var start = self._start.day_in(year) * 86400 + self._start.seconds - std
        var end = self._end.day_in(year) * 86400 + self._end.seconds - dst
        var daylight: Bool
        if start < end:
            daylight = seconds >= start and seconds < end
        else:
            daylight = not (seconds >= end and seconds < start)
        return self._dst if daylight else self._std

    def resolve(self, local: Int64) -> Tuple[Int, Int64, Int64]:
        """The UTC instants whose local time is `local` (both in seconds):
        (count, earliest, latest). Count is 0 in a gap (a non-existent
        local time), 1 normally, and 2 when clocks fall back (ambiguous)."""
        var count = 0
        var earliest = Int64(0)
        var latest = Int64(0)
        for offset in self._candidates:
            var instant = local - offset
            if self.offset_at(instant) != offset:
                continue
            if count == 0 or instant < earliest:
                earliest = instant
            if count == 0 or instant > latest:
                latest = instant
            count += 1
        return (count, earliest, latest)

    def relocalize(self, local: Int64, original: Int64) raises -> Int64:
        """The UTC instant (seconds) for local time `local`, reached by
        shifting the value at UTC `original` in local time (truncate and
        calendar offset_by). Ports Polars' `localize_result_rfc_5545`: an
        ambiguous result keeps the original's daylight-saving state, and a
        result in a gap moves forward by the daylight-saving amount (the
        original's, or the one an hour later when the original is in
        standard time). Raises when neither rule gives one instant."""
        var found = self.resolve(local)
        if found[0] == 1:
            return found[1]
        var save = self.save_at(original)
        if found[0] > 1:
            if self.save_at(found[1]) == save:
                return found[1]
            if self.save_at(found[2]) == save:
                return found[2]
            raise Error(
                "could not localize a shifted datetime to time zone '"
                + self.name
                + "'"
            )
        var shifted: Int64
        if save != 0:
            shifted = local + save
        else:
            var later = self.resolve(local + 3600)
            if later[0] == 0:
                raise Error(
                    "could not localize a shifted datetime to time zone '"
                    + self.name
                    + "'"
                )
            shifted = local - self.save_at(later[1])
        var result = self.resolve(shifted)
        if result[0] == 1:
            return result[1]
        raise Error(
            "a shifted datetime is "
            + ("non-existent" if result[0] == 0 else "ambiguous")
            + " in time zone '"
            + self.name
            + "'"
        )


def canonical_zone_syntax(name: String) raises -> String:
    """`name` checked for Arrow zone syntax, with "utc" spelled "UTC"; the
    zone database is not consulted."""
    if name.byte_length() == 0:
        raise Error("time zone name is empty")
    if name.lower() == "utc":
        return "UTC"
    if _fixed_offset(name):
        return name
    var bytes = name.as_bytes()
    if bytes[0] == 47 or name.find("..") >= 0:  # leading '/'
        raise Error("invalid time zone name '" + name + "'")
    for byte in bytes:
        var ok = (
            (byte >= 65 and byte <= 90)
            or (byte >= 97 and byte <= 122)
            or (byte >= 48 and byte <= 57)
            or byte == 95  # _
            or byte == 45  # -
            or byte == 43  # +
            or byte == 47  # /
        )
        if not ok:
            raise Error("invalid time zone name '" + name + "'")
    return name


def canonical_zone(name: String) raises -> String:
    """`name` checked against the zone database (see TimeZone.load)."""
    return TimeZone.load(name).name


def _fixed_offset(name: String) -> Optional[Int64]:
    """Seconds east of UTC for an Arrow fixed offset "+HH:MM"/"-HH:MM"."""
    var bytes = name.as_bytes()
    if len(bytes) != 6 or (bytes[0] != 43 and bytes[0] != 45):
        return None
    if bytes[3] != 58:
        return None
    for i in [1, 2, 4, 5]:
        if bytes[i] < 48 or bytes[i] > 57:
            return None
    var hours = Int64(bytes[1] - 48) * 10 + Int64(bytes[2] - 48)
    var minutes = Int64(bytes[4] - 48) * 10 + Int64(bytes[5] - 48)
    if hours > 23 or minutes > 59:
        return None
    var seconds = (hours * 60 + minutes) * 60
    return -seconds if bytes[0] == 45 else seconds


def zone_directories() -> List[String]:
    """Where IANA zone files are looked up, in order (see the module
    docstring)."""
    var dirs = List[String]()
    var tzdir = getenv("TZDIR")
    if tzdir:
        dirs.append(tzdir)
    var prefix = getenv("CONDA_PREFIX")
    if prefix:
        dirs.append(prefix + "/share/zoneinfo")
    dirs.append("/usr/share/zoneinfo")
    return dirs^


def _read_zone_file(name: String) raises -> List[UInt8]:
    var dirs = zone_directories()
    for dir in dirs:
        try:
            with open(dir + "/" + name, "r") as file:
                var data = file.read_bytes()
                if (
                    len(data) >= 4
                    and String(unsafe_from_utf8=data[0:4]) == "TZif"
                ):
                    return data^
        except:
            pass
    var looked = String()
    for i in range(len(dirs)):
        looked += (", " if i > 0 else "") + dirs[i]
    raise Error(
        "unknown time zone '"
        + name
        + "': no TZif file in "
        + looked
        + " (set TZDIR or install tzdata)"
    )


def _be(data: List[UInt8], at: Int, width: Int) raises -> Int64:
    """A signed big-endian integer of 4 or 8 bytes."""
    if at + width > len(data):
        raise Error("truncated time zone file")
    var value = UInt64(0)
    for i in range(width):
        value = (value << 8) | UInt64(data[at + i])
    if width == 4:
        return Int64(Int32(UInt32(value)))
    return Int64(value)


def _parse_tzif(name: String, data: List[UInt8]) raises -> TimeZone:
    if len(data) < 44:
        raise Error("truncated time zone file for " + name)
    var version = data[4]
    var at = 0
    var width = 4
    if version >= 50:  # '2' or later: skip the 32-bit block
        var counts = _counts(data, 0)
        at = (
            44
            + counts[3] * 4
            + counts[3]
            + counts[4] * 6
            + counts[5]
            + counts[2] * 8
            + counts[1]
            + counts[0]
        )
        width = 8
    var counts = _counts(data, at)
    var leaps = counts[2]
    var times = counts[3]
    var types = counts[4]
    var chars = counts[5]
    if leaps != 0:
        raise Error("leap-second time zone files are not supported: " + name)
    if types == 0:
        raise Error("time zone file for " + name + " has no types")
    var p = at + 44
    var zone = TimeZone(name, 0, "")
    zone._offsets = List[Int64]()
    zone._names = List[String]()
    zone._saves = List[Int64]()
    var daylight = List[Bool]()
    zone._candidates = List[Int64]()
    for i in range(times):
        zone._times.append(_be(data, p + i * width, width))
    p += times * width
    for i in range(times):
        var type = Int(data[p + i])
        if type >= types:
            raise Error("corrupt time zone file for " + name)
        zone._types.append(type)
    p += times
    var chars_at = p + types * 6
    if chars_at + chars > len(data):
        raise Error("truncated time zone file for " + name)
    for i in range(types):
        zone._offsets.append(_be(data, p + i * 6, 4))
        daylight.append(data[p + i * 6 + 4] != 0)
        zone._saves.append(0)
        var index = Int(data[p + i * 6 + 5])
        var end = index
        while end < chars and data[chars_at + end] != 0:
            end += 1
        zone._names.append(
            String(unsafe_from_utf8=data[chars_at + index : chars_at + end])
        )
    # A daylight type saves its offset less the standard offset it returns
    # to (zic's SAVE): the next standard type in the transitions, else the
    # previous one, else the footer rule's, else an hour. Measuring against
    # the standard time before it would be wrong across an offset change
    # (Apia's +14 daylight time follows -11 standard time).
    for i in range(times):
        var type = zone._types[i]
        if not daylight[type] or zone._saves[type] != 0:
            continue
        var standard = Optional[Int64](None)
        for j in range(i + 1, times):
            if not daylight[zone._types[j]]:
                standard = zone._offsets[zone._types[j]]
                break
        if not standard:
            var j = i - 1
            while j >= 0:
                if not daylight[zone._types[j]]:
                    standard = zone._offsets[zone._types[j]]
                    break
                j -= 1
        if standard:
            zone._saves[type] = zone._offsets[type] - standard.value()
    var pending = List[Int]()
    for type in range(types):
        if daylight[type] and zone._saves[type] == 0:
            pending.append(type)
    p = chars_at + chars + counts[1] + counts[0]
    if width == 8 and p < len(data) and data[p] == 10:
        var end = p + 1
        while end < len(data) and data[end] != 10:
            end += 1
        var footer = String(unsafe_from_utf8=data[p + 1 : end])
        if footer.byte_length() > 0:
            _apply_rule(zone, footer)
    for type in pending:
        var standard = (
            zone._offsets[zone._std] if zone._std
            >= 0 else zone._offsets[type] - 3600
        )
        zone._saves[type] = zone._offsets[type] - standard
    for offset in zone._offsets:
        if offset not in zone._candidates:
            zone._candidates.append(offset)
    return zone^


def _counts(data: List[UInt8], at: Int) raises -> List[Int]:
    """isutcnt, isstdcnt, leapcnt, timecnt, typecnt, charcnt."""
    if (
        at + 44 > len(data)
        or String(unsafe_from_utf8=data[at : at + 4]) != "TZif"
    ):
        raise Error("not a time zone file")
    var counts = List[Int]()
    for i in range(6):
        counts.append(Int(_be(data, at + 20 + i * 4, 4)))
    return counts^


def _apply_rule(mut zone: TimeZone, rule: String) raises:
    """Add the POSIX TZ footer's types and switch rule to `zone`."""
    var pos = 0
    var std_name = _rule_name(rule, pos)
    var std = -_rule_time(rule, pos)
    zone._std = len(zone._offsets)
    zone._offsets.append(std)
    zone._names.append(std_name^)
    zone._saves.append(0)
    if pos >= rule.byte_length():
        return
    var dst_name = _rule_name(rule, pos)
    var dst = std + 3600
    var bytes = rule.as_bytes()
    if pos < len(bytes) and bytes[pos] != 44:  # ','
        dst = -_rule_time(rule, pos)
    zone._dst = len(zone._offsets)
    zone._offsets.append(dst)
    zone._names.append(dst_name^)
    zone._saves.append(dst - std)
    if pos >= len(bytes):
        # No switch rule: the POSIX default (US rules since 2007).
        zone._start = _RuleDate(2, 3, 2, 0, _DEFAULT_TIME)
        zone._end = _RuleDate(2, 11, 1, 0, _DEFAULT_TIME)
        return
    _rule_expect(rule, pos, 44)
    zone._start = _rule_date(rule, pos)
    _rule_expect(rule, pos, 44)
    zone._end = _rule_date(rule, pos)
    if pos != len(bytes):
        raise Error("unsupported time zone rule '" + rule + "'")


def _rule_expect(rule: String, mut pos: Int, byte: UInt8) raises:
    var bytes = rule.as_bytes()
    if pos >= len(bytes) or bytes[pos] != byte:
        raise Error("unsupported time zone rule '" + rule + "'")
    pos += 1


def _rule_name(rule: String, mut pos: Int) raises -> String:
    var bytes = rule.as_bytes()
    var start = pos
    if pos < len(bytes) and bytes[pos] == 60:  # '<'
        while pos < len(bytes) and bytes[pos] != 62:
            pos += 1
        if pos >= len(bytes):
            raise Error("unsupported time zone rule '" + rule + "'")
        pos += 1
        return String(rule[byte = start + 1 : pos - 1])
    while pos < len(bytes) and (
        (bytes[pos] >= 65 and bytes[pos] <= 90)
        or (bytes[pos] >= 97 and bytes[pos] <= 122)
    ):
        pos += 1
    if pos - start < 3:
        raise Error("unsupported time zone rule '" + rule + "'")
    return String(rule[byte=start:pos])


def _rule_number(rule: String, mut pos: Int) raises -> Int64:
    var bytes = rule.as_bytes()
    var value = Int64(0)
    var digits = 0
    while pos < len(bytes) and bytes[pos] >= 48 and bytes[pos] <= 57:
        value = value * 10 + Int64(bytes[pos] - 48)
        pos += 1
        digits += 1
    if digits == 0:
        raise Error("unsupported time zone rule '" + rule + "'")
    return value


def _rule_time(rule: String, mut pos: Int) raises -> Int64:
    """[+-]hh[:mm[:ss]] in seconds (hours up to 167, as RFC 8536 allows)."""
    var bytes = rule.as_bytes()
    var negative = False
    if pos < len(bytes) and (bytes[pos] == 43 or bytes[pos] == 45):
        negative = bytes[pos] == 45
        pos += 1
    var seconds = _rule_number(rule, pos) * 3600
    if pos < len(bytes) and bytes[pos] == 58:
        pos += 1
        seconds += _rule_number(rule, pos) * 60
        if pos < len(bytes) and bytes[pos] == 58:
            pos += 1
            seconds += _rule_number(rule, pos)
    return -seconds if negative else seconds


def _rule_date(rule: String, mut pos: Int) raises -> _RuleDate:
    var bytes = rule.as_bytes()
    var date: _RuleDate
    if pos < len(bytes) and bytes[pos] == 74:  # 'J'
        pos += 1
        date = _RuleDate(0, 0, 0, _rule_number(rule, pos), _DEFAULT_TIME)
    elif pos < len(bytes) and bytes[pos] == 77:  # 'M'
        pos += 1
        var month = _rule_number(rule, pos)
        _rule_expect(rule, pos, 46)
        var week = _rule_number(rule, pos)
        _rule_expect(rule, pos, 46)
        var day = _rule_number(rule, pos)
        date = _RuleDate(2, month, week, day, _DEFAULT_TIME)
    else:
        date = _RuleDate(1, 0, 0, _rule_number(rule, pos), _DEFAULT_TIME)
    if pos < len(bytes) and bytes[pos] == 47:  # '/'
        pos += 1
        date.seconds = _rule_time(rule, pos)
    return date
