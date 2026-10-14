"""GPU temporal words against existing CPU temporal/calendar operations."""

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.memory import Pointer
from std.testing import assert_equal
from dataframe import DataType
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
)
from dataframe.accelerator.gpu import temporal as gpu
from dataframe.temporal import (
    split,
    days_from_civil,
    floor_div,
    floor_mod,
    zone_of,
    localize,
    relocalize,
)
from dataframe.temporal_kernels import (
    Offset,
    ZonedOffset,
    Truncation,
    TotalUnits,
)


def dtype_of(unit: Int64) raises -> DataType:
    if unit == 0:
        return DataType.DATE
    if unit == 4:
        return DataType.TIME
    return DataType.datetime("ms" if unit == 1 else "us" if unit == 2 else "ns")


def reference(
    value: Int64, op: Int64, parameter: Int64, unit: Int64, literal: Int64
) raises -> Int64:
    var storage_unit = unit & 7
    var utc = (unit & 256) != 0
    var dtype = dtype_of(storage_unit)
    if utc:
        dtype = dtype.with_time_zone("UTC")
    if op == DT_CONVERT_TZ or (op == DT_REPLACE_TZ and parameter == 0):
        return value
    if op == DT_REPLACE_TZ:
        var target = dtype.with_time_zone("UTC")
        return localize(value, target, zone_of(target), 0, 0).value()
    if op == DT_OFFSET_BY or op == ROW_CALENDAR_OFFSET:
        if utc:
            var offset = ZonedOffset.unused(dtype)
            offset.months = parameter if op == ROW_CALENDAR_OFFSET else Int64(0)
            offset.days = (unit >> 9) & 0x3FFFF
            if offset.days & 0x20000 != 0:
                offset.days -= 0x40000
            var total = literal if op == ROW_CALENDAR_OFFSET else parameter
            offset.ticks = total - offset.days * dtype.per_second() * 86400
            return offset.apply(value, zone_of(dtype))
        var offset = Offset.unused(dtype)
        offset.months = parameter if op == ROW_CALENDAR_OFFSET else Int64(0)
        offset.ticks = literal if op == ROW_CALENDAR_OFFSET else parameter
        return offset.apply(value)
    if op == DT_TRUNCATE or op == ROW_CALENDAR_TRUNCATE:
        var truncation = Truncation.unused(dtype)
        if op == ROW_CALENDAR_TRUNCATE:
            truncation.months = parameter
        else:
            truncation.step = parameter
            var per_day = Int64(1) if unit == 0 else dtype.per_second() * 86400
            truncation.origin = 4 * per_day if parameter % (
                7 * per_day
            ) == 0 else Int64(0)
        var result = truncation.apply(value)
        return relocalize(
            result, value, dtype, zone_of(dtype)
        ) if utc else result
    if op == DT_TOTAL:
        var total = TotalUnits(DataType.duration("ns"), "nanoseconds")
        total.divide = parameter if unit != 0 else Int64(0)
        total.multiply = Int64(1) if unit != 0 else parameter
        return total.apply(value)
    var parts = split(value, dtype)
    var days = value if unit == 0 else floor_div(
        value, dtype.per_second() * 86400
    )
    if op == DT_YEAR:
        return parts.year
    if op == DT_MONTH:
        return parts.month
    if op == DT_DAY:
        return parts.day
    if op == DT_HOUR:
        return parts.hour
    if op == DT_MINUTE:
        return parts.minute
    if op == DT_SECOND:
        return parts.second
    if op == DT_NANOSECOND:
        return parts.nanos
    if op == DT_WEEKDAY:
        return floor_mod(floor_mod(days, 7) + 3, 7) + 1
    if op == DT_ORDINAL_DAY:
        return days - days_from_civil(parts.year, 1, 1) + 1
    if op == DT_DATE:
        return days
    return floor_mod(value, dtype.per_second() * 86400) * (
        1000000000 // dtype.per_second()
    )


def add(
    mut inputs: List[UInt64],
    mut expected: List[UInt64],
    value: Int64,
    op: Int64,
    parameter: Int64,
    unit: Int64,
    literal: Int64 = 0,
):
    inputs.extend(
        [
            value.cast[DType.uint64](),
            op.cast[DType.uint64](),
            parameter.cast[DType.uint64](),
            unit.cast[DType.uint64](),
            literal.cast[DType.uint64](),
        ]
    )
    try:
        expected.append(
            reference(value, op, parameter, unit, literal).cast[DType.uint64]()
        )
        expected.append(0)
    except:
        expected.append(0)
        expected.append(1)


def evaluate(
    inputs: Pointer[UInt64, ImmutAnyOrigin],
    outputs: Pointer[UInt64, MutAnyOrigin],
    count: UInt32,
):
    var row = global_idx.x
    if row >= Int(count):
        return
    var at = row * 5
    var value = inputs[unsafe_offset=at]
    var op = inputs[unsafe_offset=at + 1].cast[DType.int64]()
    var parameter = inputs[unsafe_offset=at + 2].cast[DType.int64]()
    var unit = inputs[unsafe_offset=at + 3].cast[DType.int64]()
    var literal = inputs[unsafe_offset=at + 4]
    var result = gpu.evaluate(value, op, parameter, unit, literal)
    outputs[unsafe_offset=row * 2] = result[0]
    outputs[unsafe_offset=row * 2 + 1] = UInt64(result[1])


def main() raises:
    var values: List[Int64] = [
        Int64.MIN,
        Int64.MIN + 1,
        Int64.MIN + 719468,
        -719469,
        -719468,
        -719467,
        -25509,
        -86400000000001,
        -86400000000000,
        -86401,
        -1,
        0,
        1,
        59,
        60,
        10957,
        11016,
        19782,
        86400,
        86400000000000,
        Int64.MAX - 719468,
        Int64.MAX - 1,
        Int64.MAX,
    ]
    var fields: List[Int64] = [
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
    ]
    var inputs = List[UInt64]()
    var expected = List[UInt64]()
    for unit in range(5):
        var per_day = (
            Int64(1) if unit == 0 else gpu.per_second(Int64(unit)) * 86400
        )
        for value in values:
            for op in fields:
                if unit == 0 and op >= DT_HOUR and op <= DT_NANOSECOND:
                    continue
                if unit == 4 and op <= DT_DAY:
                    continue
                if (op == DT_DATE or op == DT_TIME) and (
                    unit == 0 or unit == 4
                ):
                    continue
                add(inputs, expected, value, op, 0, Int64(unit))
            var steps: List[Int64] = [1, 3, per_day, per_day * 7]
            for step in steps:
                add(inputs, expected, value, DT_TRUNCATE, step, Int64(unit))
            var offsets: List[Int64] = [-1, 0, 1, per_day]
            for offset in offsets:
                add(inputs, expected, value, DT_OFFSET_BY, offset, Int64(unit))
            if unit != 4:
                for months in List[Int64]([Int64(-13), -1, 1, 12]):
                    add(
                        inputs,
                        expected,
                        value,
                        ROW_CALENDAR_OFFSET,
                        months,
                        Int64(unit),
                    )
                    add(
                        inputs,
                        expected,
                        value,
                        ROW_CALENDAR_OFFSET,
                        months,
                        Int64(unit),
                        per_day,
                    )
                for months in List[Int64]([Int64(1), 3, 12]):
                    add(
                        inputs,
                        expected,
                        value,
                        ROW_CALENDAR_TRUNCATE,
                        months,
                        Int64(unit),
                    )
    for divide in range(2):
        for factor in List[Int64]([Int64(1), 1000, 1000000, 1000000000, 86400]):
            for value in values:
                add(inputs, expected, value, DT_TOTAL, factor, Int64(divide))
    var utc_units: List[Int64] = [1, 2, 3]
    var calendar_days: List[Int64] = [-1, 0, 1]
    for storage_unit in utc_units:
        var per_day = gpu.per_second(storage_unit) * 86400
        for value in values:
            add(inputs, expected, value, DT_CONVERT_TZ, 1, storage_unit)
            add(inputs, expected, value, DT_REPLACE_TZ, 0, storage_unit)
            add(inputs, expected, value, DT_REPLACE_TZ, 1, storage_unit)
            for op in fields:
                add(inputs, expected, value, op, 0, storage_unit | 256)
            add(
                inputs,
                expected,
                value,
                DT_TRUNCATE,
                per_day,
                storage_unit | 256,
            )
            add(
                inputs,
                expected,
                value,
                ROW_CALENDAR_TRUNCATE,
                1,
                storage_unit | 256,
            )
            for days in calendar_days:
                var encoded = storage_unit | 256 | ((days & 0x3FFFF) << 9)
                var ticks = days * per_day + 1
                add(inputs, expected, value, DT_OFFSET_BY, ticks, encoded)
                add(
                    inputs,
                    expected,
                    value,
                    ROW_CALENDAR_OFFSET,
                    1,
                    encoded,
                    ticks,
                )
    var count = len(expected) // 2
    var actual = List[UInt64](length=len(expected), fill=0)
    var ctx = DeviceContext()
    var di = ctx.enqueue_create_buffer[DType.uint64](len(inputs))
    var dout = ctx.enqueue_create_buffer[DType.uint64](len(actual))
    ctx.enqueue_copy(di, inputs.unsafe_ptr())
    ctx.enqueue_function[evaluate](
        di.unsafe_ptr(),
        dout.unsafe_ptr(),
        UInt32(count),
        grid_dim=(count + 127) // 128,
        block_dim=128,
    )
    ctx.enqueue_copy(actual.unsafe_ptr(), dout)
    ctx.synchronize()
    for row in range(count):
        if actual[row * 2 + 1] != expected[row * 2 + 1] or (
            expected[row * 2 + 1] == 0 and actual[row * 2] != expected[row * 2]
        ):
            print(
                "TEMPORAL MISMATCH row/value/op/parameter/unit/literal/expected/error/actual/error",
                row,
                inputs[row * 5],
                inputs[row * 5 + 1],
                inputs[row * 5 + 2],
                inputs[row * 5 + 3],
                inputs[row * 5 + 4],
                expected[row * 2],
                expected[row * 2 + 1],
                actual[row * 2],
                actual[row * 2 + 1],
            )
        assert_equal(actual[row * 2 + 1], expected[row * 2 + 1])
        if expected[row * 2 + 1] == 0:
            assert_equal(actual[row * 2], expected[row * 2])
    print("PASS Mojo temporal/calendar CPU/device words:", count)
