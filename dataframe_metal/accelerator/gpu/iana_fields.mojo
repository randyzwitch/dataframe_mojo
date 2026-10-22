"""Exact CPU TimeZone table lookup on Mojo GPU words.

Host serialization validates header/type bounds. Input seconds derive from
valid Datetime ms/us/ns ticks; this module never resolves local wall times.
"""
from std.memory import Pointer
from dataframe.calendar import (
    civil_from_days,
    days_from_civil,
    days_in_month,
    floor_div,
    floor_mod,
    is_leap,
)
from dataframe_metal.accelerator.gpu.timezone_fields import (
    local_ticks as shift_ticks,
)

comptime HEADER_WORDS = 16
comptime FORMAT_VERSION = 1


def _offset(
    table: Pointer[UInt64, ImmutAnyOrigin], base: Int, kind: Int
) -> Int64:
    var transitions = Int(table[unsafe_offset=base])
    return table[
        unsafe_offset=base + HEADER_WORDS + 2 * transitions + kind
    ].cast[DType.int64]()


def _rule_day(
    table: Pointer[UInt64, ImmutAnyOrigin], base: Int, year: Int64
) -> Int64:
    var kind = table[unsafe_offset=base].cast[DType.int64]()
    var month = table[unsafe_offset=base + 1].cast[DType.int64]()
    var week = table[unsafe_offset=base + 2].cast[DType.int64]()
    var day = table[unsafe_offset=base + 3].cast[DType.int64]()
    var first = days_from_civil(year, 1, 1)
    if kind == 0:
        var extra = Int64(1) if is_leap(year) and day >= 60 else Int64(0)
        return first + day - 1 + extra
    if kind == 1:
        return first + day
    var start = days_from_civil(year, month, 1)
    var weekday = floor_mod(start + 4, 7)
    var result = start + floor_mod(day - weekday, 7) + (week - 1) * 7
    var end = start + days_in_month(year, month)
    while result >= end:
        result -= 7
    return result


def _rule_type(
    table: Pointer[UInt64, ImmutAnyOrigin], base: Int, seconds: Int64
) -> Int:
    var standard_type = Int(table[unsafe_offset=base + 2].cast[DType.int64]())
    var daylight_type = Int(table[unsafe_offset=base + 3].cast[DType.int64]())
    if daylight_type < 0:
        return standard_type
    var standard = _offset(table, base, standard_type)
    var daylight = _offset(table, base, daylight_type)
    var year = civil_from_days(floor_div(seconds + standard, 86400))[0]
    var start = (
        _rule_day(table, base + 4, year) * 86400
        + table[unsafe_offset=base + 8].cast[DType.int64]()
        - standard
    )
    var end = (
        _rule_day(table, base + 9, year) * 86400
        + table[unsafe_offset=base + 13].cast[DType.int64]()
        - daylight
    )
    var active = seconds >= start and seconds < end if start < end else not (
        seconds >= end and seconds < start
    )
    return daylight_type if active else standard_type


def offset_at(
    table: Pointer[UInt64, ImmutAnyOrigin], base: Int, seconds: Int64
) -> Int64:
    """Match TimeZone.type_at, including the inclusive final-footer boundary."""
    var count = Int(table[unsafe_offset=base])
    var start = base + HEADER_WORDS
    if count > 0 and seconds < table[unsafe_offset=start].cast[DType.int64]():
        return _offset(table, base, 0)
    if (
        count == 0
        or seconds
        >= table[unsafe_offset=start + 2 * (count - 1)].cast[DType.int64]()
    ):
        if table[unsafe_offset=base + 2].cast[DType.int64]() >= 0:
            return _offset(table, base, _rule_type(table, base, seconds))
        return _offset(
            table, base, Int(table[unsafe_offset=start + 2 * (count - 1) + 1])
        ) if count else _offset(table, base, 0)
    var low = 0
    var high = count - 1
    while low < high:
        var middle = (low + high + 1) // 2
        if (
            table[unsafe_offset=start + 2 * middle].cast[DType.int64]()
            <= seconds
        ):
            low = middle
        else:
            high = middle - 1
    return _offset(table, base, Int(table[unsafe_offset=start + 2 * low + 1]))


def local_ticks(
    table: Pointer[UInt64, ImmutAnyOrigin],
    base: Int,
    value: UInt64,
    ticks_per_second: Int64,
) -> UInt64:
    var seconds = floor_div(value.cast[DType.int64](), ticks_per_second)
    # The offset is chosen at the original UTC instant, before local wrapping.
    return shift_ticks(value, offset_at(table, base, seconds), ticks_per_second)
