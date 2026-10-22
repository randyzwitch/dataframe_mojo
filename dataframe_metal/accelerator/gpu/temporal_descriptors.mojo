"""Canonical descriptors for checked temporal arithmetic and physical casts.

The shared lowerer validates logical types and operand safety before these
private opcodes reach the GPU interpreter. This module imports no device APIs.
"""

from dataframe.binding import temporal_result
from dataframe.dtype import DataType
from dataframe.expr import COL, LIT_INT, LIT_BOOL, LIT_NULL

comptime ROW_TEMPORAL_BINARY = Int64(1021)
comptime ROW_TEMPORAL_CAST = Int64(1022)


def _infallible(producer: Int64) raises:
    if (
        producer != COL
        and producer != LIT_INT
        and producer != LIT_BOOL
        and producer != LIT_NULL
    ):
        raise Error(
            "temporal prototype requires infallible operands and predicate"
        )


def _kind(dtype: DataType) raises -> UInt64:
    if dtype == DataType.INT64:
        return 0
    if dtype.is_date():
        return 1
    if dtype.is_datetime():
        return 2
    if dtype.is_duration():
        return 3
    if dtype.is_time():
        return 4
    raise Error("temporal prototype requires Int64 or temporal logical types")


def _unit(dtype: DataType) -> UInt64:
    if dtype.is_time():
        return 3
    if dtype.is_datetime() or dtype.is_duration():
        return (
            UInt64(1) if dtype.unit()
            == "ms" else UInt64(2) if dtype.unit()
            == "us" else UInt64(3)
        )
    return 1


def binary_literal(
    op: Int64,
    left: DataType,
    right: DataType,
    left_producer: Int64 = COL,
    right_producer: Int64 = COL,
    predicate_producer: Int64 = LIT_BOOL,
) raises -> UInt64:
    _infallible(left_producer)
    _infallible(right_producer)
    _infallible(predicate_producer)
    var result = temporal_result(Int(op), left, right)
    return (
        _kind(left)
        | (_kind(right) << 4)
        | (_unit(result) << 8)
        | (UInt64(op) << 12)
    )


def cast_literal(
    source: DataType,
    target: DataType,
    strict: Bool,
    source_producer: Int64 = COL,
    predicate_producer: Int64 = LIT_BOOL,
) raises -> UInt64:
    _infallible(source_producer)
    _infallible(predicate_producer)
    # Decimal64 shares Int64 storage but is not an unscaled temporal word.
    _ = _kind(source)
    _ = _kind(target)
    if (
        source.physical() != DataType.INT64
        or target.physical() != DataType.INT64
    ):
        raise Error("cannot cast " + source.name() + " to " + target.name())
    var mode = UInt64(0)
    if source == DataType.INT64 or target == DataType.INT64 or source == target:
        pass
    elif source.is_date() and target.is_datetime():
        mode = 1
    elif source.is_datetime() and (target.is_date() or target.is_time()):
        if source.time_zone() != "" and source.time_zone() != "UTC":
            raise Error(
                "temporal prototype local extraction requires naive or UTC"
            )
        mode = UInt64(2) if target.is_date() else UInt64(3)
    elif (source.is_datetime() and target.is_datetime()) or (
        source.is_duration() and target.is_duration()
    ):
        mode = 4
    else:
        raise Error("cannot cast " + source.name() + " to " + target.name())
    return (
        mode
        | (_unit(source) << 4)
        | (_unit(target) << 8)
        | (UInt64(strict) << 12)
    )
