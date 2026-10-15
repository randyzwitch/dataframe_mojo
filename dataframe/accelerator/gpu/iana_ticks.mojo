"""Zone-aware row primitives with CPU tick arithmetic and calendar phase order.

The caller owns the immutable resolution planes and incoming row validity.
Statuses remain separate from values so a row interpreter can rank the original
AST fault and construct the selected public diagnostic from bounded controls.
"""
from std.memory import Pointer
from dataframe.calendar import floor_div, floor_mod
from dataframe.expr import DT_TRUNCATE
from dataframe.accelerator.rows import (
    ROW_CALENDAR_OFFSET,
    ROW_CALENDAR_TRUNCATE,
)
from . import iana_fields, iana_resolution, temporal, timezone_fields

comptime OK = iana_resolution.OK
comptime TICK_OVERFLOW = 4
comptime NONEXISTENT = 5
comptime AMBIGUOUS = 6


def to_local(
    table: Pointer[UInt64, ImmutAnyOrigin],
    base: Int,
    value: UInt64,
    unit: Int64,
) -> UInt64:
    var per = temporal.per_second(unit)
    var offset = iana_fields.offset_at(
        table, base, floor_div(value.cast[DType.int64](), per)
    )
    return timezone_fields.local_ticks(value, offset, per)


def _join_instant(
    instant: Int64, fraction: Int64, per: Int64
) -> Tuple[UInt64, Int]:
    var whole = temporal.checked_mul(instant, per)
    if whole[1]:
        return (whole[0].cast[DType.uint64](), TICK_OVERFLOW)
    var output = temporal.checked_add(whole[0], fraction)
    return (output[0].cast[DType.uint64](), TICK_OVERFLOW if output[1] else OK)


def localize(
    table: Pointer[UInt64, ImmutAnyOrigin],
    base: Int,
    candidates: Int,
    candidate_count: Int,
    value: UInt64,
    unit: Int64,
    ambiguous: Int,
    non_existent: Int,
) -> Tuple[UInt64, Bool, Int]:
    var per = temporal.per_second(unit)
    var local = value.cast[DType.int64]()
    var found = iana_resolution.resolve(
        table, base, candidates, candidate_count, floor_div(local, per)
    )
    if not found[0]:
        return (UInt64(0), False, OK if non_existent == 1 else NONEXISTENT)
    var instant = found[1]
    if found[0] > 1:
        if ambiguous == 3:
            return (UInt64(0), False, OK)
        if ambiguous == 0:
            return (UInt64(0), False, AMBIGUOUS)
        if ambiguous == 2:
            instant = found[2]
    var output = _join_instant(instant, floor_mod(local, per), per)
    return (output[0], output[1] == OK, output[1])


def relocalize(
    table: Pointer[UInt64, ImmutAnyOrigin],
    base: Int,
    saves: Int,
    candidates: Int,
    candidate_count: Int,
    value: UInt64,
    original: UInt64,
    unit: Int64,
) -> Tuple[UInt64, Int]:
    var per = temporal.per_second(unit)
    var local = value.cast[DType.int64]()
    var shifted = iana_resolution.relocalize(
        table,
        base,
        saves,
        candidates,
        candidate_count,
        floor_div(local, per),
        floor_div(original.cast[DType.int64](), per),
    )
    if shifted[1] != OK:
        return (UInt64(0), shifted[1])
    return _join_instant(shifted[0], floor_mod(local, per), per)


def truncate(
    table: Pointer[UInt64, ImmutAnyOrigin],
    base: Int,
    saves: Int,
    candidates: Int,
    candidate_count: Int,
    value: UInt64,
    unit: Int64,
    parameter: Int64,
    months: Bool,
) -> Tuple[UInt64, Int]:
    var local = to_local(table, base, value, unit)
    var truncated = temporal.evaluate(
        local,
        Int64(ROW_CALENDAR_TRUNCATE if months else DT_TRUNCATE),
        parameter,
        unit,
    )
    if truncated[1]:
        return (truncated[0], TICK_OVERFLOW)
    return relocalize(
        table,
        base,
        saves,
        candidates,
        candidate_count,
        truncated[0],
        value,
        unit,
    )


def offset_by(
    table: Pointer[UInt64, ImmutAnyOrigin],
    base: Int,
    saves: Int,
    candidates: Int,
    candidate_count: Int,
    value: UInt64,
    unit: Int64,
    months: Int64,
    days: Int64,
    ticks: Int64,
) -> Tuple[UInt64, Int]:
    var result = value
    if months:
        var local = to_local(table, base, result, unit)
        var shifted = temporal.evaluate(
            local, ROW_CALENDAR_OFFSET, months, unit
        )
        if shifted[1]:
            return (shifted[0], TICK_OVERFLOW)
        var localized = relocalize(
            table,
            base,
            saves,
            candidates,
            candidate_count,
            shifted[0],
            result,
            unit,
        )
        if localized[1] != OK:
            return localized
        result = localized[0]
    if days:
        var local = to_local(table, base, result, unit)
        var day_delta = temporal.checked_mul(
            days, temporal.per_second(unit) * 86400
        )
        if day_delta[1]:
            return (day_delta[0].cast[DType.uint64](), TICK_OVERFLOW)
        var shifted = temporal.checked_add(
            local.cast[DType.int64](), day_delta[0]
        )
        if shifted[1]:
            return (shifted[0].cast[DType.uint64](), TICK_OVERFLOW)
        var localized = relocalize(
            table,
            base,
            saves,
            candidates,
            candidate_count,
            shifted[0].cast[DType.uint64](),
            result,
            unit,
        )
        if localized[1] != OK:
            return localized
        result = localized[0]
    if ticks:
        var shifted = temporal.checked_add(result.cast[DType.int64](), ticks)
        if shifted[1]:
            return (shifted[0].cast[DType.uint64](), TICK_OVERFLOW)
        result = shifted[0].cast[DType.uint64]()
    return (result, OK)
