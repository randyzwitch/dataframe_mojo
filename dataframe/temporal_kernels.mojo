"""Kernels for temporal arithmetic, dt operations, and temporal casts.

Temporal values are Int64 storage tagged with a DataType; these kernels take
the logical types from the binder, so their inputs may carry either tag.
"""
from .column import Column
from .string_column import StringColumn, StringBuilder
from .dtype import DataType
from .expr import (
    Node,
    ADD,
    SUB,
    MUL,
    FLOORDIV,
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
    DT_STRFTIME,
    DT_STRPTIME,
    DT_REPLACE_TZ,
    DT_CONVERT_TZ,
)
from .series import Series
from .temporal import (
    NANOS_PER_DAY,
    Parts,
    add_months,
    checked_add,
    checked_mul,
    days_from_civil,
    floor_div,
    floor_mod,
    format,
    format_in,
    join,
    localize,
    parse,
    parse_in,
    parse_every,
    parse_interval,
    relocalize,
    split,
    strptime_target,
    ticks_per_day,
    to_local,
    zone_of,
    ambiguous_code,
    non_existent_code,
)
from .timezone import TimeZone


def temporal_binary(
    op: Int,
    left: Series,
    right: Series,
    left_type: DataType,
    right_type: DataType,
    result_type: DataType,
) raises -> Series:
    if left.is_chunked() or right.is_chunked():
        return temporal_binary(
            op,
            left.rechunk(),
            right.rechunk(),
            left_type,
            right_type,
            result_type,
        )
    ref a = left._data[Column[Int64]]
    ref b = right._data[Column[Int64]]
    if len(a) != len(b) and len(a) != 1 and len(b) != 1:
        raise Error("Incompatible expression lengths")
    var n = 0 if len(a) == 0 or len(b) == 0 else max(len(a), len(b))
    var values = List[Int64](length=n, fill=0)
    var valid = List[Bool](length=n, fill=False)
    for i in range(n):
        var p = 0 if len(a) == 1 else i
        var q = 0 if len(b) == 1 else i
        if not (a._valid(p) and b._valid(q)):
            continue
        var x = a._get(p)
        var y = b._get(q)
        valid[i] = True
        if op == MUL:
            values[i] = checked_mul(x, y)
        elif op == FLOORDIV:
            if y == 0:
                valid[i] = False
            else:
                values[i] = floor_div(x, y)
        else:
            if left_type.is_date() and right_type.is_duration():
                x = checked_mul(x, ticks_per_day(result_type))
            var raw = checked_add(x, -y) if op == SUB else checked_add(x, y)
            if left_type.is_date() and right_type.is_date():
                raw = checked_mul(raw, 86400000)
            values[i] = raw
    return Series("", Column[Int64](values^, valid)).with_dtype(result_type)


def _weekday(days: Int64) -> Int64:
    # 1970-01-01 was a Thursday (ISO weekday 4).
    return floor_mod(days + 3, 7) + 1


def _days(value: Int64, dtype: DataType) -> Int64:
    if dtype.is_date():
        return value
    return floor_div(value, ticks_per_day(dtype))


def _truncate(value: Int64, dtype: DataType, every: String) raises -> Int64:
    var interval = parse_every(every)
    var months = interval[0]
    var nanos = interval[1]
    if (months != 0) == (nanos != 0) or months < 0 or nanos < 0:
        raise Error(
            "truncate needs one positive interval in months/years or in"
            " fixed units, found " + every
        )
    if months > 0:
        if dtype.is_time():
            raise Error("time values cannot be truncated by months")
        var p = split(value, dtype)
        var index = floor_div(p.year * 12 + p.month - 1, months) * months
        return join(
            Parts(
                floor_div(index, 12), floor_mod(index, 12) + 1, 1, 0, 0, 0, 0
            ),
            dtype,
        )
    var per_tick = (
        Int64(1000000000)
        // dtype.per_second() if not dtype.is_date() else NANOS_PER_DAY
    )
    if nanos % per_tick != 0:
        raise Error("interval " + every + " is finer than " + dtype.name())
    var step = nanos // per_tick
    # Whole weeks align to Monday (1970-01-05); other steps to the epoch.
    var origin = Int64(0)
    if nanos % (7 * NANOS_PER_DAY) == 0:
        origin = 4 * (NANOS_PER_DAY // per_tick)
    return floor_div(value - origin, step) * step + origin


def _offset_in(
    value: Int64, dtype: DataType, by: String, zone: TimeZone
) raises -> Int64:
    """offset_by for a zone-aware datetime: months and days move the local
    wall-clock time, then fixed units move the UTC instant (as Polars)."""
    var interval = parse_interval(by)
    var result = value
    # Months, then days, each localized from the step before, as Polars
    # does.
    if interval[0] != 0:
        var local = to_local(result, dtype, zone)
        local = join(add_months(split(local, dtype), interval[0]), dtype)
        result = relocalize(local, result, dtype, zone)
    if interval[1] != 0:
        var local = checked_add(
            to_local(result, dtype, zone),
            checked_mul(interval[1], ticks_per_day(dtype)),
        )
        result = relocalize(local, result, dtype, zone)
    if interval[2] != 0:
        var per_tick = Int64(1000000000) // dtype.per_second()
        if interval[2] % per_tick != 0:
            raise Error("offset " + by + " is finer than " + dtype.name())
        result = checked_add(result, interval[2] // per_tick)
    return result


def _offset(value: Int64, dtype: DataType, by: String) raises -> Int64:
    var interval = parse_every(by)
    var result = value
    if interval[0] != 0:
        if dtype.is_time():
            raise Error("time values cannot be offset by months")
        result = join(add_months(split(value, dtype), interval[0]), dtype)
    if interval[1] != 0:
        var per_tick = (
            Int64(1000000000)
            // dtype.per_second() if not dtype.is_date() else NANOS_PER_DAY
        )
        if interval[1] % per_tick != 0:
            raise Error("offset " + by + " is finer than " + dtype.name())
        result = checked_add(result, interval[1] // per_tick)
        if dtype.is_time():
            result = floor_mod(result, NANOS_PER_DAY)
    return result


def _total(value: Int64, dtype: DataType, unit: String) -> Int64:
    var nanos_per_tick = Int64(1000000000) // dtype.per_second()
    var nanos_per_unit: Int64
    if unit == "days":
        nanos_per_unit = NANOS_PER_DAY
    elif unit == "hours":
        nanos_per_unit = 3600000000000
    elif unit == "minutes":
        nanos_per_unit = 60000000000
    elif unit == "seconds":
        nanos_per_unit = 1000000000
    elif unit == "milliseconds":
        nanos_per_unit = 1000000
    elif unit == "microseconds":
        nanos_per_unit = 1000
    else:
        nanos_per_unit = 1
    if nanos_per_unit >= nanos_per_tick:
        var ticks = nanos_per_unit // nanos_per_tick
        # Truncate toward zero, like a duration's whole-unit count.
        return value // ticks if value >= 0 else -((-value) // ticks)
    return value * (nanos_per_tick // nanos_per_unit)


def dt_op(node: Node, input: Series, dtype: DataType) raises -> Series:
    if input.is_chunked():
        return dt_op(node, input.rechunk(), dtype)
    var op = node.op
    if op == DT_STRPTIME:
        return _strptime(
            input, node.dtypes[0].value(), node.text, node.integer == 1
        )
    if op == DT_REPLACE_TZ:
        return _replace_time_zone(input, dtype, node)
    if op == DT_CONVERT_TZ:
        return input.with_dtype(dtype.with_time_zone(node.text))
    ref column = input._data[Column[Int64]]
    var n = len(column)
    var valid = List[Bool](length=n, fill=False)
    var aware = dtype.is_datetime() and dtype.time_zone().byte_length() > 0
    var zone = zone_of(dtype)
    if op == DT_STRFTIME:
        var texts = List[String](length=n, fill="")
        for i in range(n):
            valid[i] = column._valid(i)
            if valid[i]:
                texts[i] = format_in(column._get(i), dtype, node.text, zone)
        return Series("", StringColumn(texts, valid))
    var values = List[Int64](length=n, fill=0)
    for i in range(n):
        valid[i] = column._valid(i)
        if not valid[i]:
            continue
        var v = column._get(i)
        if aware and op == DT_TRUNCATE:
            # Truncate the local time, then map back keeping the offset.
            var per_second = dtype.per_second()
            var offset = zone.offset_at(floor_div(v, per_second))
            values[i] = relocalize(
                _truncate(v + offset * per_second, dtype, node.text),
                v,
                dtype,
                zone,
            )
            continue
        if aware and op == DT_OFFSET_BY:
            values[i] = _offset_in(v, dtype, node.text, zone)
            continue
        if aware:
            # Every other operation reads local fields.
            v = to_local(v, dtype, zone)
        if op == DT_TRUNCATE:
            values[i] = _truncate(v, dtype, node.text)
        elif op == DT_OFFSET_BY:
            values[i] = _offset(v, dtype, node.text)
        elif op == DT_TOTAL:
            values[i] = _total(v, dtype, node.text)
        elif op == DT_DATE:
            values[i] = _days(v, dtype)
        elif op == DT_TIME:
            values[i] = floor_mod(v, ticks_per_day(dtype)) * (
                Int64(1000000000) // dtype.per_second()
            )
        elif op == DT_WEEKDAY:
            values[i] = _weekday(_days(v, dtype))
        elif op == DT_ORDINAL_DAY:
            var p = split(v, dtype)
            values[i] = _days(v, dtype) - days_from_civil(p.year, 1, 1) + 1
        elif op == DT_YEAR or op == DT_MONTH or op == DT_DAY:
            var p = split(v, dtype)
            if op == DT_YEAR:
                values[i] = p.year
            elif op == DT_MONTH:
                values[i] = p.month
            else:
                values[i] = p.day
        elif dtype.is_date():
            values[i] = 0
        else:
            # Time-of-day fields need no calendar: the ticks within the
            # day (ClickBench q18 takes the minute of 10M timestamps).
            var within = floor_mod(
                v, NANOS_PER_DAY if dtype.is_time() else ticks_per_day(dtype)
            )
            var per_second = dtype.per_second()
            var seconds = within // per_second
            if op == DT_HOUR:
                values[i] = seconds // 3600
            elif op == DT_MINUTE:
                values[i] = (seconds // 60) % 60
            elif op == DT_SECOND:
                values[i] = seconds % 60
            else:
                values[i] = (within % per_second) * (1000000000 // per_second)
    var result = Series("", Column[Int64](values^, valid))
    if op == DT_TRUNCATE or op == DT_OFFSET_BY:
        return result.with_dtype(dtype)
    if op == DT_DATE:
        return result.with_dtype(DataType.DATE)
    if op == DT_TIME:
        return result.with_dtype(DataType.TIME)
    return result^


def _replace_time_zone(
    input: Series, dtype: DataType, node: Node
) raises -> Series:
    """Keep each wall-clock time and read it in the new zone."""
    var target = dtype.with_time_zone(node.text)
    var options = node.text2.split(",")
    var ambiguous = ambiguous_code(String(options[0]))
    var non_existent = non_existent_code(String(options[1]))
    var source_zone = zone_of(dtype)
    var target_zone = zone_of(target)
    var aware = dtype.time_zone().byte_length() > 0
    ref column = input._data[Column[Int64]]
    var n = len(column)
    var values = List[Int64](length=n, fill=0)
    var valid = List[Bool](length=n, fill=False)
    for i in range(n):
        if not column._valid(i):
            continue
        var wall = column._get(i)
        if aware:
            wall = to_local(wall, dtype, source_zone)
        if target.time_zone().byte_length() == 0:
            values[i] = wall
            valid[i] = True
            continue
        var instant = localize(
            wall, target, target_zone, ambiguous, non_existent
        )
        if instant:
            values[i] = instant.value()
            valid[i] = True
    return Series("", Column[Int64](values^, valid)).with_dtype(target)


def _strptime(
    input: Series, target: DataType, format: String, strict: Bool
) raises -> Series:
    ref column = input._data[StringColumn]
    var n = len(column)
    var values = List[Int64](length=n, fill=0)
    var valid = List[Bool](length=n, fill=False)
    var result_type = strptime_target(target, format)
    var zone = zone_of(result_type)
    for i in range(n):
        if not column._valid(i):
            continue
        try:
            values[i] = parse_in(
                String(column._get(i)), result_type, format, zone
            )
            valid[i] = True
        except e:
            if strict:
                raise e^
    return Series("", Column[Int64](values^, valid)).with_dtype(result_type)


def cast_temporal(
    input: Series, source: DataType, target: DataType, strict: Bool
) raises -> Series:
    """Casts involving a temporal type: to and from Int64 (the stored
    value), to and from String (ISO 8601), between date and datetime,
    datetime to time, and between units."""
    if input.is_chunked():
        return cast_temporal(input.rechunk(), source, target, strict)
    var n = len(input)
    # A zone-aware source casts to date, time and string at local time;
    # casts between datetimes keep the instant, as in Polars.
    var zone = zone_of(source)
    if target == DataType.STRING:
        var texts = List[String](length=n, fill="")
        var valid = List[Bool](length=n, fill=False)
        ref column = input._data[Column[Int64]]
        for i in range(n):
            valid[i] = column._valid(i)
            if valid[i]:
                texts[i] = format_in(column._get(i), source, "", zone)
        return Series(input.name(), StringColumn(texts, valid))
    if source == DataType.STRING:
        return _strptime(input, target, "", strict).renamed(input.name())
    if (
        source.physical() != DataType.INT64
        or target.physical() != DataType.INT64
    ):
        raise Error("cannot cast " + source.name() + " to " + target.name())
    ref column = input._data[Column[Int64]]
    var values = List[Int64](length=n, fill=0)
    var valid = List[Bool](length=n, fill=False)
    for i in range(n):
        if not column._valid(i):
            continue
        var v = column._get(i)
        try:
            if (
                target == DataType.INT64
                or source == DataType.INT64
                or source == target
            ):
                values[i] = v
            elif source.is_date() and target.is_datetime():
                values[i] = checked_mul(v, ticks_per_day(target))
            elif source.is_datetime() and target.is_date():
                values[i] = floor_div(
                    to_local(v, source, zone), ticks_per_day(source)
                )
            elif source.is_datetime() and target.is_time():
                v = to_local(v, source, zone)
                values[i] = floor_mod(v, ticks_per_day(source)) * (
                    Int64(1000000000) // source.per_second()
                )
            elif (source.is_datetime() and target.is_datetime()) or (
                source.is_duration() and target.is_duration()
            ):
                var a = source.per_second()
                var b = target.per_second()
                values[i] = checked_mul(v, b // a) if b >= a else floor_div(
                    v, a // b
                )
            else:
                raise Error(
                    "cannot cast " + source.name() + " to " + target.name()
                )
            valid[i] = True
        except e:
            if strict or String(e).startswith("cannot cast"):
                raise e^
    return Series(input.name(), Column[Int64](values^, valid)).with_dtype(
        target
    )


def _zoned_range(
    start: String, end: String, interval: String, dtype: DataType, name: String
) raises -> Series:
    """A range in a time zone: calendar steps (d, w, mo, y) move the local
    time from start, fixed steps move the instant, as in Polars."""
    var zone = zone_of(dtype)
    var first = parse_in(start, dtype, "", zone)
    var last = parse_in(end, dtype, "", zone)
    var step = parse_interval(interval)
    if (
        step[0] < 0
        or step[1] < 0
        or step[2] < 0
        or (step[0] == 0 and step[1] == 0 and step[2] == 0)
    ):
        raise Error("range interval must be positive, found " + interval)
    var fixed = _ticks(step[2], dtype, interval)
    var per_second = dtype.per_second()
    var offset = zone.offset_at(floor_div(first, per_second))
    var local = checked_add(first, offset * per_second)
    var values = List[Int64]()
    var current = first
    var count = Int64(0)
    while current <= last:
        values.append(current)
        count += 1
        current = first
        if step[0] != 0 or step[1] != 0:
            var shifted = local
            if step[0] != 0:
                shifted = join(
                    add_months(split(local, dtype), step[0] * count), dtype
                )
            shifted = checked_add(
                shifted, checked_mul(step[1] * count, ticks_per_day(dtype))
            )
            current = relocalize(shifted, first, dtype, zone)
        current = checked_add(current, checked_mul(fixed, count))
    return Series(name, Column[Int64](values^)).with_dtype(dtype)


def _range(
    start: String, end: String, interval: String, dtype: DataType, name: String
) raises -> Series:
    if dtype.time_zone().byte_length() > 0:
        return _zoned_range(start, end, interval, dtype, name)
    var first = parse(start, dtype)
    var last = parse(end, dtype)
    var step = parse_every(interval)
    if (step[0] == 0 and step[1] == 0) or step[0] < 0 or step[1] < 0:
        raise Error("range interval must be positive, found " + interval)
    var values = List[Int64]()
    var current = first
    var count = 0
    while current <= last:
        values.append(current)
        count += 1
        if step[0] != 0:
            # Month steps restart from the first value so days do not drift.
            var shifted = add_months(
                split(first, dtype), step[0] * Int64(count)
            )
            current = join(shifted, dtype)
            if step[1] != 0:
                current = checked_add(current, _ticks(step[1], dtype, interval))
        else:
            current = checked_add(current, _ticks(step[1], dtype, interval))
    return Series(name, Column[Int64](values^)).with_dtype(dtype)


def _ticks(nanos: Int64, dtype: DataType, interval: String) raises -> Int64:
    var per_tick = (
        NANOS_PER_DAY if dtype.is_date() else Int64(1000000000)
        // dtype.per_second()
    )
    if nanos % per_tick != 0:
        raise Error("interval " + interval + " is finer than " + dtype.name())
    return nanos // per_tick


def date_range(
    start: String, end: String, interval: String = "1d", name: String = "date"
) raises -> Series:
    """Dates from start through end (inclusive, ISO 8601 text) every
    interval, such as "1d", "1w", or "1mo"."""
    return _range(start, end, interval, DataType.DATE, name)


def datetime_range(
    start: String,
    end: String,
    interval: String,
    unit: String = "us",
    name: String = "datetime",
    time_zone: String = "",
) raises -> Series:
    """Datetimes from start through end (inclusive) every interval. With a
    time zone, start and end are local times there."""
    return _range(
        start, end, interval, DataType.datetime(unit, time_zone), name
    )
