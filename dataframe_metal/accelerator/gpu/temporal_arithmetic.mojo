"""Fixed-width temporal binary/cast helpers for the GPU row interpreter.

Only operations admitted by accelerator.temporal_expressions may call these.
Inputs are valid Int64 storage words; the row interpreter owns null/error
propagation and conditional masking. No DeviceContext or device UInt128.
"""

from dataframe.expr import ADD, SUB, MUL, FLOORDIV, MOD
from . import temporal, integers

comptime TEMPORAL_OVERFLOW = UInt32(64)
comptime RAW = UInt64(0)
comptime DATE_TO_DATETIME = UInt64(1)
comptime DATETIME_TO_DATE = UInt64(2)
comptime DATETIME_TO_TIME = UInt64(3)
comptime RESCALE = UInt64(4)
comptime DATE_KIND = UInt64(1)
comptime DURATION_KIND = UInt64(3)


def checked_sub(a: Int64, b: Int64) -> Tuple[Int64, UInt32]:
    var left = a.cast[DType.uint64]()
    var right = b.cast[DType.uint64]()
    var value = left - right
    return (
        value.cast[DType.int64](),
        UInt32(((left ^ right) & (left ^ value)) >> 63),
    )


def binary_parameter(
    left_kind: UInt64, right_kind: UInt64, unit: UInt64
) -> UInt64:
    """Logical kinds: raw=0/date=1/datetime=2/duration=3/time=4.

    Unit: ms=1/us=2/ns=3. For Date-Date the result unit is ms.
    """
    return left_kind | (right_kind << 4) | (unit << 8)


def binary(
    op: Int64, a: UInt64, b: UInt64, parameter: UInt64
) -> Tuple[UInt64, Bool, UInt32]:
    var x = a.cast[DType.int64]()
    var y = b.cast[DType.int64]()
    if op == FLOORDIV:
        var result = integers.evaluate(op, a, b, UInt32(64), False)
        return (result[0], result[1], result[2] * TEMPORAL_OVERFLOW)
    if op == MUL:
        var result = temporal.checked_mul(x, y)
        return (
            result[0].cast[DType.uint64](),
            True,
            result[1] * TEMPORAL_OVERFLOW,
        )
    var left_kind = parameter & 15
    var right_kind = (parameter >> 4) & 15
    var error = UInt32(0)
    if left_kind == DATE_KIND and right_kind == DURATION_KIND:
        var scale = Int64(86400) * temporal.per_second(
            Int64((parameter >> 8) & 15)
        )
        var scaled = temporal.checked_mul(x, scale)
        x = scaled[0]
        error |= scaled[1]
    var result = checked_sub(x, y) if op == SUB else temporal.checked_add(x, y)
    error |= result[1]
    x = result[0]
    if left_kind == DATE_KIND and right_kind == DATE_KIND:
        var scaled = temporal.checked_mul(x, Int64(86400000))
        x = scaled[0]
        error |= scaled[1]
    return (x.cast[DType.uint64](), True, error * TEMPORAL_OVERFLOW)


def cast_parameter(
    mode: UInt64, source_unit: UInt64 = 1, target_unit: UInt64 = 1
) -> UInt64:
    return mode | (source_unit << 4) | (target_unit << 8)


def cast(
    value: UInt64, parameter: UInt64, strict: Bool
) -> Tuple[UInt64, Bool, UInt32]:
    """Naive/UTC extraction only; host rejects unsupported mode/type pairs.

    Datetime->Datetime unit conversion is valid for any timezone metadata:
    it preserves the instant. Zoned Date/Time extraction needs zone lookup.
    """
    var mode = parameter & 15
    var input = value.cast[DType.int64]()
    var source_per = temporal.per_second(Int64((parameter >> 4) & 15))
    var target_per = temporal.per_second(Int64((parameter >> 8) & 15))
    var output = input
    var error = UInt32(0)
    if mode == DATE_TO_DATETIME:
        var result = temporal.checked_mul(input, Int64(86400) * target_per)
        output = result[0]
        error = result[1]
    elif mode == DATETIME_TO_DATE:
        var result = integers.evaluate(
            FLOORDIV, value, UInt64(86400 * source_per), UInt32(64), False
        )
        output = result[0].cast[DType.int64]()
        error = result[2]
    elif mode == DATETIME_TO_TIME:
        var day = Int64(86400) * source_per
        # Positive divisor, floor remainder; result is strictly within a day.
        var remainder = integers.evaluate(
            MOD, value, UInt64(day), UInt32(64), False
        )
        var within = remainder[0].cast[DType.int64]()
        output = within * (Int64(1000000000) // source_per)
    elif mode == RESCALE:
        if target_per > source_per:
            var result = temporal.checked_mul(input, target_per // source_per)
            output = result[0]
            error = result[1]
        elif target_per < source_per:
            var result = integers.evaluate(
                FLOORDIV,
                value,
                UInt64(source_per // target_per),
                UInt32(64),
                False,
            )
            output = result[0].cast[DType.int64]()
            error = result[2]
    if error != 0 and not strict:
        return (UInt64(0), False, UInt32(0))
    return (output.cast[DType.uint64](), True, error * TEMPORAL_OVERFLOW)
