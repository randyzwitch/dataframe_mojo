"""Fixed-width temporal row operations using the library's Mojo calendar.

The shared row plan encodes Date as unit 0, datetime ms/us/ns as 1/2/3,
and Time as 4. Parameters and literals are exact integer storage words.
Input validity and deferred errors remain the row interpreter's responsibility.
"""

from dataframe.calendar import (
    civil_from_days,
    days_from_civil,
    days_in_month,
    floor_div,
    floor_mod,
    is_leap,
)
from dataframe.expr import (
    DT_YEAR,
    DT_MONTH,
    DT_DAY,
    DT_HOUR,
    DT_MINUTE,
    DT_SECOND,
    DT_NANOSECOND,
    DT_WEEKDAY,
    DT_ORDINAL_DAY,
    DT_DATE,
    DT_TIME,
    DT_TRUNCATE,
    DT_OFFSET_BY,
    DT_TOTAL,
    DT_REPLACE_TZ,
    DT_CONVERT_TZ,
)
from dataframe.accelerator.rows import (
    ROW_CALENDAR_OFFSET,
    ROW_CALENDAR_TRUNCATE,
    ROW_FIXED_ZONE_FIELDS,
)
from . import floating, timezone_fields

comptime NANOS_PER_DAY = Int64(86400000000000)
comptime UTC_FLAG = Int64(256)


def supported(op: Int64) -> Bool:
    return (
        (op >= DT_YEAR and op <= DT_TOTAL)
        or op == ROW_CALENDAR_OFFSET
        or op == ROW_CALENDAR_TRUNCATE
        or op == DT_REPLACE_TZ
        or op == DT_CONVERT_TZ
    )


def per_second(unit: Int64) -> Int64:
    return (
        Int64(1000) if unit
        == 1 else Int64(1000000) if unit
        == 2 else Int64(1000000000)
    )


def checked_add(a: Int64, b: Int64) -> Tuple[Int64, UInt32]:
    var left = a.cast[DType.uint64]()
    var right = b.cast[DType.uint64]()
    var value = left + right
    return (
        value.cast[DType.int64](),
        UInt32(((left ^ value) & (right ^ value)) >> 63),
    )


def checked_mul(a: Int64, b: Int64) -> Tuple[Int64, UInt32]:
    var left = a.cast[DType.uint64]()
    var right = b.cast[DType.uint64]()
    var negative = (left ^ right) >> 63 != 0
    left = UInt64(0) - left if a < 0 else left
    right = UInt64(0) - right if b < 0 else right
    # Reuse the already validated paired-word product instead of Int128.
    var pair = floating.product(left, right)
    var maximum = UInt64(1) << 63
    maximum -= UInt64(not negative)
    var error = UInt32(pair[0] != 0 or pair[1] > maximum)
    var value = UInt64(0) - pair[1] if negative else pair[1]
    return (value.cast[DType.int64](), error)


def _join(days: Int64, within: Int64, unit: Int64) -> Tuple[Int64, UInt32]:
    if unit == 0:
        return (days, 0)
    var per = per_second(unit)
    # Match temporal.join's checked association, including boundary errors.
    var seconds = checked_mul(days, 86400)
    var whole = checked_add(seconds[0], within // per)
    var ticks = checked_mul(whole[0], per)
    var value = checked_add(ticks[0], within % per)
    return (value[0], seconds[1] | whole[1] | ticks[1] | value[1])


def _utc_localize(value: Int64, unit: Int64) -> Tuple[Int64, UInt32]:
    # Preserve CPU localize/relocalize's checked whole-second association,
    # including the lower-bound multiplication before a fractional tick.
    var per = per_second(unit)
    var whole = checked_mul(floor_div(value, per), per)
    var result = checked_add(whole[0], floor_mod(value, per))
    return (result[0], whole[1] | result[1])


def _utc_offset(
    value: Int64, ticks: Int64, unit: Int64, days: Int64
) -> Tuple[Int64, UInt32]:
    var output = value
    var error = UInt32(0)
    var day_ticks = days * per_second(unit) * 86400
    if days != 0:
        var local = checked_add(output, day_ticks)
        var instant = _utc_localize(local[0], unit)
        output = instant[0]
        error |= local[1] | instant[1]
    var fixed = checked_add(output, ticks - day_ticks)
    return (fixed[0], error | fixed[1])


def evaluate(
    value: UInt64, op: Int64, parameter: Int64, unit: Int64, literal: UInt64 = 0
) -> Tuple[UInt64, UInt32]:
    var input = value.cast[DType.int64]()
    var storage_unit = unit & 7
    if (unit & ROW_FIXED_ZONE_FIELDS) != 0 and op >= DT_YEAR and op <= DT_TIME:
        input = timezone_fields.local_ticks(
            value, parameter, per_second(storage_unit)
        ).cast[DType.int64]()
    var utc = (unit & UTC_FLAG) != 0
    # parse_every bounds fixed nanoseconds to Int64, so calendar days fit
    # a signed 18-bit payload; bits 9..26 retain the CPU's distinct day phase.
    var calendar_days = (unit >> 9) & 0x3FFFF
    if calendar_days & 0x20000 != 0:
        calendar_days -= 0x40000
    if op == DT_CONVERT_TZ or (op == DT_REPLACE_TZ and parameter == 0):
        return (value, 0)
    if op == DT_REPLACE_TZ:
        var instant = _utc_localize(input, storage_unit)
        return (instant[0].cast[DType.uint64](), instant[1])
    if op == DT_OFFSET_BY:
        if utc:
            var shifted = _utc_offset(
                input, parameter, storage_unit, calendar_days
            )
            return (shifted[0].cast[DType.uint64](), shifted[1])
        var shifted = checked_add(input, parameter)
        var output = shifted[0]
        if storage_unit == 4 and parameter != 0:
            output = floor_mod(output, NANOS_PER_DAY)
        return (output.cast[DType.uint64](), shifted[1])
    if op == DT_TOTAL:
        if storage_unit == 0:
            # Finer duration totals intentionally retain storage wrapping.
            return (value * parameter.cast[DType.uint64](), 0)
        var quotient = floor_div(input, parameter)
        if input < 0 and floor_mod(input, parameter) != 0:
            quotient += 1
        return (quotient.cast[DType.uint64](), 0)
    var per = per_second(storage_unit)
    var per_day = Int64(1) if storage_unit == 0 else per * 86400
    var days = input if storage_unit == 0 else floor_div(input, per_day)
    if op == DT_TRUNCATE:
        var origin = 4 * per_day if parameter % (7 * per_day) == 0 else Int64(0)
        # Fixed truncation uses the CPU's fixed-width storage arithmetic.
        var shifted = (value - origin.cast[DType.uint64]()).cast[DType.int64]()
        var quotient = floor_div(shifted, parameter)
        var output = (
            quotient.cast[DType.uint64]() * parameter.cast[DType.uint64]()
            + origin.cast[DType.uint64]()
        )
        if utc:
            var instant = _utc_localize(
                output.cast[DType.int64](), storage_unit
            )
            return (instant[0].cast[DType.uint64](), instant[1])
        return (output, 0)
    if op == DT_DATE:
        return (days.cast[DType.uint64](), 0)
    if op == DT_WEEKDAY:
        return (UInt64((floor_mod(days, 7) + 3) % 7 + 1), 0)
    if (op >= DT_HOUR and op <= DT_NANOSECOND) or op == DT_TIME:
        var within = floor_mod(input, per_day)
        var seconds = within // per
        var output: Int64
        if op == DT_HOUR:
            output = seconds // 3600
        elif op == DT_MINUTE:
            output = (seconds // 60) % 60
        elif op == DT_SECOND:
            output = seconds % 60
        elif op == DT_TIME:
            output = within * (1000000000 // per)
        else:
            output = (within % per) * (1000000000 // per)
        return (output.cast[DType.uint64](), 0)
    if op == DT_ORDINAL_DAY and storage_unit == 4:
        return ((days + 1).cast[DType.uint64](), 0)
    var civil = civil_from_days(days)
    var year = civil[0]
    var month = civil[1]
    var day = civil[2]
    if op == ROW_CALENDAR_OFFSET or op == ROW_CALENDAR_TRUNCATE:
        var index = year * 12 + month - 1
        if op == ROW_CALENDAR_OFFSET:
            index = (
                index.cast[DType.uint64]() + parameter.cast[DType.uint64]()
            ).cast[DType.int64]()
        else:
            index = floor_div(index, parameter) * parameter
        year = floor_div(index, 12)
        month = floor_mod(index, 12) + 1
        day = min(
            day, days_in_month(year, month)
        ) if op == ROW_CALENDAR_OFFSET else Int64(1)
        var target_days = days_from_civil(year, month, day)
        var within = floor_mod(
            input, per_day
        ) if storage_unit != 0 and op == ROW_CALENDAR_OFFSET else Int64(0)
        var joined = _join(target_days, within, storage_unit)
        if op == ROW_CALENDAR_OFFSET:
            if utc:
                var shifted = _utc_offset(
                    joined[0],
                    literal.cast[DType.int64](),
                    storage_unit,
                    calendar_days,
                )
                return (shifted[0].cast[DType.uint64](), joined[1] | shifted[1])
            var shifted = checked_add(joined[0], literal.cast[DType.int64]())
            return (shifted[0].cast[DType.uint64](), joined[1] | shifted[1])
        if utc:
            var instant = _utc_localize(joined[0], storage_unit)
            return (instant[0].cast[DType.uint64](), joined[1] | instant[1])
        return (joined[0].cast[DType.uint64](), joined[1])
    if op == DT_YEAR:
        return (year.cast[DType.uint64](), 0)
    if op == DT_MONTH:
        return (month.cast[DType.uint64](), 0)
    if op == DT_DAY:
        return (day.cast[DType.uint64](), 0)
    var ordinal = (367 * month - 362) // 12 + day
    if month > 2:
        ordinal -= Int64(1) if is_leap(year) else Int64(2)
    return (ordinal.cast[DType.uint64](), 0)
