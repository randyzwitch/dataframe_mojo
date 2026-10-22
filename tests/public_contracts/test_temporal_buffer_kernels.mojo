"""Temporal buffer kernels preserve values across layouts and calendar indexes."""
# contract-group: expr.temporal
from std.testing import assert_equal
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Series,
    col,
    duration,
    time,
    business_day_count,
)
from dataframe.temporal import format
from tests.public_contracts.test_temporal import tagged, assert_ticks
from tests.public_contracts.test_temporal_reporting import reporting_select
from tests.support.coverage_generators import offset_layout, chunk_layout


def reference_business_day(
    day: Int64, week: List[Bool], holidays: List[Int64]
) -> Bool:
    if not week[Int((day % 7 + 3) % 7)]:
        return False
    for holiday in holidays:
        if day == holiday:
            return False
    return True


def reference_count(
    start: Int64, end: Int64, week: List[Bool], holidays: List[Int64]
) -> Int64:
    var result = Int64(0)
    var day = start
    var step = Int64(1 if end > start else -1)
    while day != end:
        if reference_business_day(day, week, holidays):
            result += step
        day += step
    return result


def reference_add(
    day: Int64, n: Int64, week: List[Bool], holidays: List[Int64]
) -> Int64:
    var current = day
    while not reference_business_day(current, week, holidays):
        current += 1
    var step = Int64(1 if n >= 0 else -1)
    var remaining = n if n >= 0 else -n
    while remaining:
        current += step
        if reference_business_day(current, week, holidays):
            remaining -= 1
    return current


def test_business_calendar_index_matches_daily_model_for_unsorted_duplicate_holidays() raises:
    var holidays = List[Int64]()
    var text = List[String]()
    # Include weekends, duplicates, dates before the epoch, and dates far
    # outside the query intervals. Enumeration order is deliberately reversed.
    for i in range(127, -1, -1):
        var day = Int64(-80 + i * 3)
        holidays.append(day)
        text.append(format(day, DataType.DATE))
        if i % 7 == 0:
            text.append(format(day, DataType.DATE))
    var starts: List[Int64] = [-100, -80, -1, 0, 1, 50, 90, 300, Int64.MAX]
    var ends: List[Int64] = [300, -100, 90, 0, -80, 1, -1, -100, Int64.MIN]
    var offsets: List[Int64] = [21, -21, 7, 0, -1, 1, -7, 14, Int64.MIN]
    var valid: List[Bool] = [
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
    ]
    var frame = DataFrame(
        [
            tagged("d", DataType.DATE, starts, valid),
            tagged("end", DataType.DATE, ends, valid),
            Series("n", Column[Int64](offsets.copy(), valid)),
        ]
    )
    for calendar in range(2):
        var week: List[Bool] = [
            True,
            True,
            True,
            True,
            True,
            False,
            False,
        ] if calendar == 0 else [False, True, False, True, False, True, False]
        var counts = List[Int64](length=9, fill=0)
        var added = List[Int64](length=9, fill=0)
        for row in range(8):
            counts[row] = reference_count(
                starts[row], ends[row], week, holidays
            )
            added[row] = reference_add(
                starts[row], offsets[row], week, holidays
            )
        for layout in range(3):
            var source = (
                frame.copy() if layout
                == 0 else offset_layout(frame) if layout
                == 1 else chunk_layout(frame)
            )
            for mode in range(3):
                var result = reporting_select(
                    source,
                    [
                        business_day_count(
                            col("d"), col("end"), week_mask=week, holidays=text
                        ).alias("count"),
                        col("d")
                        .dt()
                        .add_business_days(
                            col("n"),
                            week_mask=week,
                            holidays=text,
                            roll="forward",
                        )
                        .alias("add"),
                        col("d")
                        .dt()
                        .is_business_day(week_mask=week, holidays=text)
                        .alias("is"),
                    ],
                    mode,
                )
                assert_ticks(result, "count", DataType.INT64, counts, valid)
                assert_ticks(result, "add", DataType.DATE, added, valid)
                for row in range(8):
                    assert_equal(
                        result.item(row, "is").bool(),
                        reference_business_day(starts[row], week, holidays),
                    )
                assert_equal(result.item(8, "is").is_null(), True)


def test_temporal_typed_windows_keep_nulls_units_and_zone_across_chunks() raises:
    var dtype = DataType.datetime("us", "America/New_York")
    var valid: List[Bool] = [True, True, False, True]
    var frame = DataFrame(
        [
            tagged(
                "t",
                dtype,
                [
                    1710055800000000,
                    1730615400000000,
                    Int64.MAX,
                    1705311000000000,
                ],
                valid,
            ),
            Series("n", Column[Int64]([1001, -1001, Int64.MAX, -999], valid)),
        ]
    )
    for layout in range(3):
        var source = (
            frame.copy() if layout
            == 0 else offset_layout(frame) if layout
            == 1 else chunk_layout(frame)
        )
        for mode in range(3):
            var result = reporting_select(
                source,
                [
                    duration(
                        milliseconds=1, nanoseconds=col("n"), time_unit="us"
                    ).alias("duration"),
                    col("t").dt().combine(time(12, 30)).alias("combined"),
                    col("t").dt().replace(hour=12, minute=30).alias("replaced"),
                ],
                mode,
            )
            assert_ticks(
                result,
                "duration",
                DataType.duration("us"),
                [1001, 999, 0, 1000],
                valid,
            )
            assert_ticks(
                result,
                "combined",
                dtype,
                [1710088200000000, 1730655000000000, 0, 1705339800000000],
                valid,
            )
            assert_ticks(
                result,
                "replaced",
                dtype,
                [1710088200000000, 1730655000000000, 0, 1705339800000000],
                valid,
            )
