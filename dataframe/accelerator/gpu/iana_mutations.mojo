"""Private zone mutation adapter. Statuses must never reach legacy OR flags."""
from std.memory import Pointer
from dataframe.expr import DT_REPLACE_TZ, DT_TRUNCATE, DT_OFFSET_BY
from dataframe.accelerator.rows import (
    ROW_CALENDAR_OFFSET,
    ROW_CALENDAR_TRUNCATE,
)
from . import iana_ticks, temporal


def evaluate(
    words: Pointer[UInt64, ImmutAnyOrigin], base: Int, value: UInt64
) -> Tuple[UInt64, Bool, Int, UInt64]:
    """Value, validity, exact status, and diagnostic wall ticks.

    The fourth word is returned only to a bounded failure replay, never as an
    N-row diagnostic plane. The caller decides mask/phase observation.
    """
    var op = Int(words[unsafe_offset=base + 1])
    var source = Int(words[unsafe_offset=base + 2])
    var target = Int(words[unsafe_offset=base + 3])
    var saves = Int(words[unsafe_offset=base + 4])
    var candidates = Int(words[unsafe_offset=base + 5])
    var count = Int(words[unsafe_offset=base + 6])
    var unit = Int64(words[unsafe_offset=base + 7])
    if words[unsafe_offset=base + 15]:
        return (UInt64(0), False, 7, value)
    if op == DT_REPLACE_TZ:
        var wall = iana_ticks.to_local(words, source, value, unit) if words[
            unsafe_offset=base + 10
        ] else value
        if not words[unsafe_offset=base + 11]:
            return (wall, True, 0, wall)
        var localized = iana_ticks.localize(
            words,
            target,
            candidates,
            count,
            wall,
            unit,
            Int(words[unsafe_offset=base + 8]),
            Int(words[unsafe_offset=base + 9]),
        )
        return (localized[0], localized[1], localized[2], wall)
    var parameter = words[unsafe_offset=base + 12].cast[DType.int64]()
    if not words[unsafe_offset=base + 10]:
        var months = (
            words[unsafe_offset=base + 11]
            != 0 if op
            == DT_TRUNCATE else parameter
            != 0
        )
        var operation = Int64(
            ROW_CALENDAR_TRUNCATE if months else DT_TRUNCATE
        ) if op == DT_TRUNCATE else Int64(
            ROW_CALENDAR_OFFSET if months else DT_OFFSET_BY
        )
        var ticks = words[unsafe_offset=base + 14]
        var argument = (
            parameter if op == DT_TRUNCATE
            or months else ticks.cast[DType.int64]()
        )
        var result = temporal.evaluate(value, operation, argument, unit, ticks)
        return (result[0], result[1] == 0, 4 if result[1] else 0, value)
    if op == DT_TRUNCATE:
        var result = iana_ticks.truncate(
            words,
            source,
            saves,
            candidates,
            count,
            value,
            unit,
            parameter,
            words[unsafe_offset=base + 11] != 0,
        )
        return (result[0], result[1] == 0, result[1], value)
    var result = iana_ticks.offset_by(
        words,
        source,
        saves,
        candidates,
        count,
        value,
        unit,
        parameter,
        words[unsafe_offset=base + 13].cast[DType.int64](),
        words[unsafe_offset=base + 14].cast[DType.int64](),
    )
    return (result[0], result[1] == 0, result[1], value)
