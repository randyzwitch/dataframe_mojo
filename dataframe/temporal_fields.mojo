"""Typed temporal field projections with packed validity and no row dispatch."""
from .calendar import civil_from_days
from .column import Column, _copy_validity
from .dtype import DataType
from .expr import (
    DT_YEAR,
    DT_MONTH,
    DT_DAY,
    DT_HOUR,
    DT_MINUTE,
    DT_SECOND,
    DT_DATE,
    DT_WEEKDAY,
    DT_QUARTER,
)
from .series import Series


def _fields[
    op: Int
](input: Series, dtype: DataType, mask: List[Bool]) raises -> Series:
    ref column = input._data[Column[Int64]]
    var n = len(column)
    var bits = _copy_validity(column._bits[], column._offset, n)
    if len(mask):
        if len(bits) == 0:
            bits = List[UInt8](length=(n + 7) // 8, fill=255)
        for i in range(n):
            if not mask[i]:
                bits[i // 8] &= ~(UInt8(1) << UInt8(i % 8))
    var nullable = len(bits) > 0
    var values = List[Int64](length=n, fill=0)
    var out = values.unsafe_ptr()
    var source = column._ptr()
    var is_date = dtype.is_date()
    var is_time = dtype.is_time()
    var per_second = dtype.per_second()
    var per_day = per_second * 86400
    for i in range(n):
        if nullable and (bits[i // 8] & (UInt8(1) << UInt8(i % 8))) == 0:
            continue
        var v = source.unsafe_offset(i)[]
        comptime if op == DT_HOUR or op == DT_MINUTE or op == DT_SECOND:
            if is_date:
                continue
            var seconds = (v % per_day) // per_second
            comptime if op == DT_HOUR:
                out.unsafe_offset(i)[] = seconds // 3600
            elif op == DT_MINUTE:
                out.unsafe_offset(i)[] = (seconds // 60) % 60
            else:
                out.unsafe_offset(i)[] = seconds % 60
        else:
            var days = v if is_date else v // per_day
            comptime if op == DT_DATE:
                out.unsafe_offset(i)[] = days
            elif op == DT_WEEKDAY:
                out.unsafe_offset(i)[] = ((days % 7 + 3) % 7) + 1
            else:
                # Time's calendar fields refer to 1970-01-01; extracting a
                # field never needs the other time-of-day components.
                var ymd = civil_from_days(0 if is_time else days)
                comptime if op == DT_YEAR:
                    out.unsafe_offset(i)[] = ymd[0]
                elif op == DT_MONTH:
                    out.unsafe_offset(i)[] = ymd[1]
                elif op == DT_DAY:
                    out.unsafe_offset(i)[] = ymd[2]
                else:
                    out.unsafe_offset(i)[] = (ymd[1] - 1) // 3 + 1
    var result = Series("", Column[Int64](values=values^, bits=bits^))
    comptime if op == DT_DATE:
        return result.with_dtype(DataType.DATE)
    return result^


def field_projection(
    op: Int, input: Series, dtype: DataType, mask: List[Bool]
) raises -> Optional[Series]:
    if dtype.is_datetime() and dtype.time_zone().byte_length() > 0:
        return None
    if op == DT_YEAR:
        return _fields[DT_YEAR](input, dtype, mask)
    if op == DT_MONTH:
        return _fields[DT_MONTH](input, dtype, mask)
    if op == DT_DAY:
        return _fields[DT_DAY](input, dtype, mask)
    if op == DT_HOUR:
        return _fields[DT_HOUR](input, dtype, mask)
    if op == DT_MINUTE:
        return _fields[DT_MINUTE](input, dtype, mask)
    if op == DT_SECOND:
        return _fields[DT_SECOND](input, dtype, mask)
    if op == DT_DATE:
        return _fields[DT_DATE](input, dtype, mask)
    if op == DT_WEEKDAY:
        return _fields[DT_WEEKDAY](input, dtype, mask)
    if op == DT_QUARTER:
        return _fields[DT_QUARTER](input, dtype, mask)
    return None
