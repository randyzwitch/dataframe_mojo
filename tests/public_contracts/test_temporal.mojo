"""Temporal contracts in civil time, UTC instants, and independent row memberships."""
from std.math import sqrt
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Expr,
    lit,
    when,
    Series,
    col,
    date_range,
    datetime_range,
)
from tests.support.coverage_compare import assert_float, assert_value
from tests.support.reduction_contract import assert_statistic
from tests.support.coverage_generators import chunk_layout, offset_layout
from tests.support.temporal_reference import (
    civil_days,
    civil_ns,
    members,
    sum_members,
)


def tagged(
    name: String,
    dtype: DataType,
    values: List[Int64],
    valid: List[Bool] = List[Bool](),
) raises -> Series:
    if len(valid):
        return Series(
            name, Column[Int64](values.copy(), valid.copy())
        ).with_dtype(dtype)
    return Series(name, Column[Int64](values.copy())).with_dtype(dtype)


def assert_ticks(
    frame: DataFrame,
    name: String,
    dtype: DataType,
    expected: List[Int64],
    valid: List[Bool] = List[Bool](),
) raises:
    assert_true(frame.column(name).dtype() == dtype)
    assert_equal(frame.height(), len(expected))
    for row in range(len(expected)):
        var present = not len(valid) or valid[row]
        assert_equal(frame.item(row, name).is_null(), not present)
        if present:
            if dtype == DataType.INT64:
                assert_equal(frame.item(row, name).int64(), expected[row])
            else:
                assert_equal(frame.item(row, name).to_physical(), expected[row])


def test_temporal_units_calendar_fields_and_negative_epochs() raises:
    assert_equal(civil_days(1970, 1, 1), Int64(0))
    assert_equal(civil_days(1969, 12, 31), Int64(-1))
    assert_equal(civil_days(2000, 2, 29), Int64(11016))
    assert_equal(civil_days(1900, 3, 1), Int64(-25508))
    assert_equal(civil_days(2024, 2, 29), Int64(19782))
    var years: List[Int64] = [1970, 1969, 2000, 1900, 2024, 0]
    var months: List[Int64] = [1, 12, 2, 3, 2, 0]
    var days: List[Int64] = [1, 31, 29, 1, 29, 0]
    var weekdays: List[Int64] = [4, 3, 2, 4, 4, 0]
    var ordinal: List[Int64] = [1, 365, 60, 60, 60, 0]
    var hours: List[Int64] = [0, 23, 12, 1, 5, 0]
    var minutes: List[Int64] = [0, 59, 34, 2, 6, 0]
    var seconds: List[Int64] = [0, 59, 56, 3, 7, 0]
    var fractions: List[Int64] = [0, 123456789, 987654321, 1, 999999999, 0]
    var valid: List[Bool] = [True, True, True, True, True, False]
    for unit in [String("ms"), String("us"), String("ns")]:
        var divisor = Int64(
            1000000 if unit == "ms" else 1000 if unit == "us" else 1
        )
        var dates = List[Int64]()
        var ticks = List[Int64]()
        var times = List[Int64]()
        var nanos = List[Int64]()
        for row in range(5):
            dates.append(
                civil_days(Int(years[row]), Int(months[row]), Int(days[row]))
            )
            var time = (
                hours[row] * 3600 + minutes[row] * 60 + seconds[row]
            ) * 1000000000 + fractions[row] // divisor * divisor
            times.append(time)
            nanos.append(fractions[row] // divisor * divisor)
            ticks.append(
                dates[row] * (86400000000000 // divisor) + time // divisor
            )
        dates.append(Int64.MAX)
        ticks.append(Int64.MIN)
        times.append(Int64.MAX)
        nanos.append(0)
        var original = DataFrame(
            [
                tagged("t", DataType.datetime(unit), ticks, valid),
                tagged("d", DataType.DATE, dates, valid),
                tagged("time", DataType.TIME, times, valid),
            ]
        )
        for layout in range(3):
            var frame = (
                original.copy() if layout
                == 0 else offset_layout(original) if layout
                == 1 else chunk_layout(original)
            )
            var expressions: List[Expr] = [
                col("t").dt().year().alias("year"),
                col("t").dt().month().alias("month"),
                col("t").dt().day().alias("day"),
                col("t").dt().weekday().alias("weekday"),
                col("t").dt().ordinal_day().alias("ordinal"),
                col("t").dt().hour().alias("hour"),
                col("t").dt().minute().alias("minute"),
                col("t").dt().second().alias("second"),
                col("t").dt().nanosecond().alias("nano"),
                col("t").dt().date().alias("date"),
                col("t").dt().time().alias("clock"),
            ]
            for lazy in range(2):
                var actual = (
                    frame.select_exprs(expressions) if lazy
                    == 0 else frame.lazy().select_exprs(expressions).collect()
                )
                var expected: List[List[Int64]] = [
                    years.copy(),
                    months.copy(),
                    days.copy(),
                    weekdays.copy(),
                    ordinal.copy(),
                    hours.copy(),
                    minutes.copy(),
                    seconds.copy(),
                    nanos.copy(),
                ]
                var names: List[String] = [
                    "year",
                    "month",
                    "day",
                    "weekday",
                    "ordinal",
                    "hour",
                    "minute",
                    "second",
                    "nano",
                ]
                for index in range(len(names)):
                    assert_ticks(
                        actual,
                        names[index],
                        DataType.INT64,
                        expected[index],
                        valid,
                    )
                assert_ticks(actual, "date", DataType.DATE, dates, valid)
                assert_ticks(actual, "clock", DataType.TIME, times, valid)
            var date_fields = frame.select_exprs(
                [
                    col("d").dt().year().alias("year"),
                    col("d").dt().ordinal_day().alias("ordinal"),
                    col("d").dt().weekday().alias("weekday"),
                ]
            )
            assert_ticks(date_fields, "year", DataType.INT64, years, valid)
            assert_ticks(date_fields, "ordinal", DataType.INT64, ordinal, valid)
            assert_ticks(
                date_fields, "weekday", DataType.INT64, weekdays, valid
            )
            var clock = frame.select_exprs(
                [
                    col("time").dt().hour().alias("hour"),
                    col("time").dt().nanosecond().alias("nano"),
                ]
            )
            assert_ticks(clock, "hour", DataType.INT64, hours, valid)
            assert_ticks(clock, "nano", DataType.INT64, nanos, valid)
            var empty = frame.clear().select_exprs(expressions)
            assert_equal(empty.height(), 0)
            assert_true(empty.column("date").dtype() == DataType.DATE)
            assert_true(empty.column("clock").dtype() == DataType.TIME)


def test_temporal_truncation_calendar_offsets_and_duration_totals() raises:
    var dates: List[Int64] = [
        civil_days(2024, 1, 31),
        civil_days(2023, 1, 31),
        civil_days(2024, 2, 29),
        -1,
        Int64.MAX,
    ]
    var valid: List[Bool] = [True, True, True, True, False]
    var source = DataFrame([tagged("d", DataType.DATE, dates, valid)])
    var offset = source.select_exprs(
        [
            col("d").dt().offset_by("1mo").alias("month"),
            col("d").dt().offset_by("1y").alias("year"),
            col("d").dt().offset_by("-2mo").alias("back"),
        ]
    )
    assert_ticks(
        offset,
        "month",
        DataType.DATE,
        [
            civil_days(2024, 2, 29),
            civil_days(2023, 2, 28),
            civil_days(2024, 3, 29),
            civil_days(1970, 1, 31),
            0,
        ],
        valid,
    )
    assert_ticks(
        offset,
        "year",
        DataType.DATE,
        [
            civil_days(2025, 1, 31),
            civil_days(2024, 1, 31),
            civil_days(2025, 2, 28),
            civil_days(1970, 12, 31),
            0,
        ],
        valid,
    )
    assert_ticks(
        offset,
        "back",
        DataType.DATE,
        [
            civil_days(2023, 11, 30),
            civil_days(2022, 11, 30),
            civil_days(2023, 12, 29),
            civil_days(1969, 10, 31),
            0,
        ],
        valid,
    )
    var truncated = source.select_exprs(
        [
            col("d").dt().truncate("1mo").alias("month"),
            col("d").dt().truncate("1y").alias("year"),
            col("d").dt().truncate("1w").alias("week"),
        ]
    )
    assert_ticks(
        truncated,
        "month",
        DataType.DATE,
        [
            civil_days(2024, 1, 1),
            civil_days(2023, 1, 1),
            civil_days(2024, 2, 1),
            civil_days(1969, 12, 1),
            0,
        ],
        valid,
    )
    assert_ticks(
        truncated,
        "year",
        DataType.DATE,
        [
            civil_days(2024, 1, 1),
            civil_days(2023, 1, 1),
            civil_days(2024, 1, 1),
            civil_days(1969, 1, 1),
            0,
        ],
        valid,
    )
    assert_ticks(
        truncated,
        "week",
        DataType.DATE,
        [
            civil_days(2024, 1, 29),
            civil_days(2023, 1, 30),
            civil_days(2024, 2, 26),
            civil_days(1969, 12, 29),
            0,
        ],
        valid,
    )
    for unit in [String("ms"), String("us"), String("ns")]:
        var per_second = Int64(
            1000 if unit == "ms" else 1000000 if unit == "us" else 1000000000
        )
        var temporal = DataFrame(
            [
                tagged(
                    "t",
                    DataType.datetime(unit),
                    [-1, 0, 901 * per_second, Int64.MAX],
                    [True, True, True, False],
                ),
                tagged(
                    "duration",
                    DataType.duration(unit),
                    [
                        -90061 * per_second,
                        90061 * per_second,
                        -per_second + 1,
                        Int64.MIN,
                    ],
                    [True, True, True, False],
                ),
            ]
        )
        var result = temporal.select_exprs(
            [
                col("t").dt().truncate("15m").alias("floor"),
                col("t").dt().offset_by("1h30m").alias("shift"),
                col("duration").dt().total_days().alias("days"),
                col("duration").dt().total_hours().alias("hours"),
                col("duration").dt().total_minutes().alias("minutes"),
                col("duration").dt().total_seconds().alias("seconds"),
                col("duration").dt().total_milliseconds().alias("millis"),
            ]
        )
        var present: List[Bool] = [True, True, True, False]
        assert_ticks(
            result,
            "floor",
            DataType.datetime(unit),
            [-900 * per_second, 0, 900 * per_second, 0],
            present,
        )
        assert_ticks(
            result,
            "shift",
            DataType.datetime(unit),
            [5400 * per_second - 1, 5400 * per_second, 6301 * per_second, 0],
            present,
        )
        assert_ticks(result, "days", DataType.INT64, [-1, 1, 0, 0], present)
        assert_ticks(result, "hours", DataType.INT64, [-25, 25, 0, 0], present)
        assert_ticks(
            result, "minutes", DataType.INT64, [-1501, 1501, 0, 0], present
        )
        assert_ticks(
            result, "seconds", DataType.INT64, [-90061, 90061, 0, 0], present
        )
        assert_ticks(
            result,
            "millis",
            DataType.INT64,
            [-90061000, 90061000, -999, 0],
            present,
        )


def test_temporal_arithmetic_retains_types_and_checks_overflow() raises:
    for unit in [String("ms"), String("us"), String("ns")]:
        var frame = DataFrame(
            [
                tagged(
                    "t",
                    DataType.datetime(unit),
                    [-5, 7, Int64.MAX],
                    [True, True, False],
                ),
                tagged(
                    "u",
                    DataType.datetime(unit),
                    [-8, 9, Int64.MIN],
                    [True, True, False],
                ),
                tagged(
                    "delta",
                    DataType.duration(unit),
                    [3, -2, Int64.MIN],
                    [True, True, False],
                ),
                Series("n", Column[Int64]([2, 0, -1])),
            ]
        )
        var result = frame.select_exprs(
            [
                (col("t") - col("u")).alias("difference"),
                (col("t") + col("delta")).alias("plus"),
                (col("delta") + col("t")).alias("commute"),
                (col("t") - col("delta")).alias("minus"),
                (-col("delta")).alias("neg"),
                col("delta").abs().alias("abs"),
                (col("delta") * col("n")).alias("mul"),
                (col("n") * col("delta")).alias("commute_mul"),
                (col("delta") // col("n")).alias("div"),
            ]
        )
        var valid: List[Bool] = [True, True, False]
        assert_ticks(
            result, "difference", DataType.duration(unit), [3, -2, 0], valid
        )
        assert_ticks(result, "plus", DataType.datetime(unit), [-2, 5, 0], valid)
        assert_ticks(
            result, "commute", DataType.datetime(unit), [-2, 5, 0], valid
        )
        assert_ticks(
            result, "minus", DataType.datetime(unit), [-8, 9, 0], valid
        )
        assert_ticks(result, "neg", DataType.duration(unit), [-3, 2, 0], valid)
        assert_ticks(result, "abs", DataType.duration(unit), [3, 2, 0], valid)
        assert_ticks(result, "mul", DataType.duration(unit), [6, 0, 0], valid)
        assert_ticks(
            result, "commute_mul", DataType.duration(unit), [6, 0, 0], valid
        )
        assert_ticks(
            result,
            "div",
            DataType.duration(unit),
            [1, 0, 0],
            [True, False, False],
        )
        var total = frame.select(col("delta").sum())
        assert_ticks(total, "delta", DataType.duration(unit), [1])
        var overflow = DataFrame(
            [
                tagged("d", DataType.duration(unit), [Int64.MAX]),
                tagged("one", DataType.duration(unit), [1]),
                Series("two", Column[Int64]([2])),
            ]
        )
        with assert_raises():
            _ = overflow.select(col("d") + col("one"))
        with assert_raises():
            _ = overflow.select(col("d") * col("two"))
        var minimum = DataFrame(
            [tagged("d", DataType.duration(unit), [Int64.MIN])]
        )
        with assert_raises():
            _ = minimum.select(-col("d"))
        with assert_raises():
            _ = minimum.select(col("d").abs())
    var frame = DataFrame(
        [
            tagged("a", DataType.DATE, [-1, 1]),
            tagged("b", DataType.DATE, [1, -1]),
            tagged("d", DataType.duration("ms"), [1000, -1000]),
            tagged("clock", DataType.TIME, [0, 1000000001]),
            tagged("other", DataType.TIME, [1, 1]),
        ]
    )
    var result = frame.select_exprs(
        [
            (col("a") - col("b")).alias("days"),
            (col("a") + col("d")).alias("added"),
            (col("clock") - col("other")).alias("elapsed"),
        ]
    )
    assert_ticks(
        result, "days", DataType.duration("ms"), [-172800000, 172800000]
    )
    assert_ticks(
        result, "added", DataType.datetime("ms"), [-86399000, 86399000]
    )
    assert_ticks(result, "elapsed", DataType.duration("ns"), [-1, 1000000000])


def test_temporal_dst_localization_conversion_and_calendar_shifts() raises:
    var ny = String("America/New_York")
    var overlap = DataFrame(
        [
            tagged(
                "t",
                DataType.datetime("us"),
                [civil_ns(2024, 11, 3, 1, 30) // 1000, Int64.MIN],
                [True, False],
            )
        ]
    )
    var earliest = overlap.select(
        col("t").dt().replace_time_zone(ny, ambiguous="earliest").alias("t")
    )
    var latest = overlap.select(
        col("t").dt().replace_time_zone(ny, ambiguous="latest").alias("t")
    )
    assert_ticks(
        earliest,
        "t",
        DataType.datetime("us", ny),
        [civil_ns(2024, 11, 3, 5, 30) // 1000, 0],
        [True, False],
    )
    assert_ticks(
        latest,
        "t",
        DataType.datetime("us", ny),
        [civil_ns(2024, 11, 3, 6, 30) // 1000, 0],
        [True, False],
    )
    var null_overlap = overlap.select(
        col("t").dt().replace_time_zone(ny, ambiguous="null")
    )
    assert_true(null_overlap.item(0, "t").is_null())
    with assert_raises():
        _ = overlap.select(col("t").dt().replace_time_zone(ny))
    var gap = DataFrame(
        [
            tagged(
                "t",
                DataType.datetime("us"),
                [civil_ns(2024, 3, 10, 2, 30) // 1000],
            )
        ]
    )
    with assert_raises():
        _ = gap.select(col("t").dt().replace_time_zone(ny))
    assert_true(
        gap.select(col("t").dt().replace_time_zone(ny, non_existent="null"))
        .item(0, "t")
        .is_null()
    )
    var converted = earliest.select(col("t").dt().convert_time_zone("+05:30"))
    assert_ticks(
        converted,
        "t",
        DataType.datetime("us", "+05:30"),
        [civil_ns(2024, 11, 3, 5, 30) // 1000, 0],
        [True, False],
    )
    var local = converted.select_exprs(
        [
            col("t").dt().hour().alias("hour"),
            col("t").dt().minute().alias("minute"),
        ]
    )
    assert_ticks(local, "hour", DataType.INT64, [11, 0], [True, False])
    assert_ticks(local, "minute", DataType.INT64, [0, 0], [True, False])
    var naive = earliest.select(col("t").dt().replace_time_zone(""))
    assert_ticks(
        naive,
        "t",
        DataType.datetime("us"),
        [civil_ns(2024, 11, 3, 1, 30) // 1000, 0],
        [True, False],
    )
    var source = DataFrame(
        [
            tagged(
                "t",
                DataType.datetime("us", ny),
                [
                    civil_ns(2024, 3, 9, 17) // 1000,
                    civil_ns(2024, 11, 2, 16) // 1000,
                ],
            )
        ]
    )
    var shifts = source.select_exprs(
        [
            col("t").dt().offset_by("1d").alias("calendar"),
            col("t").dt().offset_by("24h").alias("elapsed"),
        ]
    )
    assert_ticks(
        shifts,
        "calendar",
        DataType.datetime("us", ny),
        [civil_ns(2024, 3, 10, 16) // 1000, civil_ns(2024, 11, 3, 17) // 1000],
    )
    assert_ticks(
        shifts,
        "elapsed",
        DataType.datetime("us", ny),
        [civil_ns(2024, 3, 10, 17) // 1000, civil_ns(2024, 11, 3, 16) // 1000],
    )
    var repeated = DataFrame(
        [
            tagged(
                "t",
                DataType.datetime("us", ny),
                [
                    civil_ns(2024, 11, 3, 5, 45) // 1000,
                    civil_ns(2024, 11, 3, 6, 45) // 1000,
                ],
            )
        ]
    )
    assert_ticks(
        repeated.select(col("t").dt().truncate("1h")),
        "t",
        DataType.datetime("us", ny),
        [civil_ns(2024, 11, 3, 5) // 1000, civil_ns(2024, 11, 3, 6) // 1000],
    )
    var text = repeated.select(
        col("t").dt().strftime("%Y-%m-%d %H:%M %z %:z %Z")
    )
    assert_equal(
        text.item(0, "t").string(), "2024-11-03 01:45 -0400 -04:00 EDT"
    )
    assert_equal(
        text.item(1, "t").string(), "2024-11-03 01:45 -0500 -05:00 EST"
    )


def test_temporal_parsing_formatting_and_invalid_text() raises:
    var dates = DataFrame(
        [
            Series(
                "s",
                Column[String](
                    [
                        "2000-02-29",
                        "1900-02-29",
                        "2023-02-29",
                        "2024-02-29",
                        "2024-13-01",
                        "2024-01-00",
                        "garbage",
                        "1900-02-29",
                    ],
                    [True, True, True, True, True, True, True, False],
                ),
            )
        ]
    )
    var parsed = dates.select(col("s").str().strptime("date", strict=False))
    assert_ticks(
        parsed,
        "s",
        DataType.DATE,
        [11016, 0, 0, 19782, 0, 0, 0, 0],
        [True, False, False, True, False, False, False, False],
    )
    with assert_raises():
        _ = dates.select(col("s").str().to_date())
    var valid_date = dates.take([0, 3, 7]).select(col("s").str().to_date())
    assert_ticks(
        valid_date, "s", DataType.DATE, [11016, 19782, 0], [True, True, False]
    )
    var strings = DataFrame(
        [
            Series(
                "s",
                Column[String](
                    [
                        "1969-12-31T23:59:59.123456789",
                        "2024-02-29 05:06:07.987654321",
                        "not a timestamp",
                    ],
                    [True, True, False],
                ),
            )
        ]
    )
    for unit in [String("ms"), String("us"), String("ns")]:
        var divisor = Int64(
            1000000 if unit == "ms" else 1000 if unit == "us" else 1
        )
        var expected: List[Int64] = [
            -1000000000 // divisor + 123456789 // divisor,
            civil_ns(2024, 2, 29, 5, 6, 7) // divisor + 987654321 // divisor,
            0,
        ]
        var values = strings.select(col("s").str().to_datetime(unit=unit))
        assert_ticks(
            values, "s", DataType.datetime(unit), expected, [True, True, False]
        )
        var rendered = values.select(
            col("s").dt().strftime("%Y/%m/%d %H:%M:%S%f %j")
        )
        var fractional = (
            String("123") if unit
            == "ms" else String("123456") if unit
            == "us" else String("123456789")
        )
        assert_equal(
            rendered.item(0, "s").string(),
            "1969/12/31 23:59:59." + fractional + " 365",
        )
    var custom = DataFrame(
        [
            Series(
                "s",
                Column[String](
                    ["29/02/2024 05:06:07.123456", "01/01/1970 00:00:00.000001"]
                ),
            )
        ]
    )
    assert_ticks(
        custom.select(col("s").str().to_datetime("%d/%m/%Y %H:%M:%S%f")),
        "s",
        DataType.datetime("us"),
        [civil_ns(2024, 2, 29, 5, 6, 7, 123456000) // 1000, 1],
    )
    var times = DataFrame(
        [
            Series(
                "s",
                Column[String](
                    [
                        "00:00",
                        "23:59:59.999999999",
                        "24:00",
                        "12:60",
                        "12:30:60",
                    ]
                ),
            )
        ]
    )
    assert_ticks(
        times.select(col("s").str().strptime("time", strict=False)),
        "s",
        DataType.TIME,
        [0, 86399999999999, 0, 0, 0],
        [True, True, False, False, False],
    )
    var offset = DataFrame(
        [
            Series(
                "s",
                Column[String](
                    ["1970-01-01T05:30:00+0530", "1969-12-31T19:00:00-0500"]
                ),
            )
        ]
    )
    assert_ticks(
        offset.select(col("s").str().to_datetime("%Y-%m-%dT%H:%M:%S%z")),
        "s",
        DataType.datetime("us", "UTC"),
        [0, 0],
    )
    var local = DataFrame(
        [Series("s", Column[String](["2024-03-09 12:00", "2024-03-10 12:00"]))]
    )
    assert_ticks(
        local.select(col("s").str().to_datetime(time_zone="America/New_York")),
        "s",
        DataType.datetime("us", "America/New_York"),
        [civil_ns(2024, 3, 9, 17) // 1000, civil_ns(2024, 3, 10, 16) // 1000],
    )


def test_temporal_binding_intervals_and_null_payload_errors() raises:
    var frame = DataFrame(
        [
            tagged("d", DataType.DATE, [0]),
            tagged("t", DataType.datetime("us"), [0]),
            tagged("other", DataType.datetime("ms"), [0]),
            tagged("zoned", DataType.datetime("us", "UTC"), [0]),
            tagged("duration", DataType.duration("us"), [0]),
            Series("number", Column[Int64]([0])),
            Series("s", Column[String](["1970-01-01"])),
        ]
    )
    var invalid: List[Expr] = [
        col("number").dt().year(),
        col("d").dt().hour(),
        col("duration").dt().year(),
        col("t").dt().total_days(),
        col("d").dt().offset_by("1h"),
        col("d").dt().truncate("1s"),
        col("t") - col("other"),
        col("t") - col("zoned"),
        col("t") + col("other"),
        col("t").dt().convert_time_zone("Invalid/Zone").dt().hour(),
        col("t").dt().replace_time_zone("UTC", ambiguous="bad"),
        col("t").dt().replace_time_zone("UTC", non_existent="bad"),
        col("s").str().strptime("date", "%Q"),
    ]
    var invalid_names: List[String] = [
        "integer.year",
        "date.hour",
        "duration.year",
        "datetime.total",
        "date.offset_by subday",
        "date.truncate subday",
        "datetime unit mismatch",
        "datetime zone mismatch",
        "datetime + datetime",
        "unknown zone",
        "unknown ambiguity policy",
        "unknown gap policy",
        "unknown parse directive",
    ]
    for index in range(len(invalid)):
        for lazy in range(2):
            var rejected = False
            try:
                if lazy:
                    _ = frame.lazy().select(invalid[index]).collect()
                else:
                    _ = frame.select(invalid[index])
            except:
                rejected = True
            assert_true(
                rejected,
                "Temporal binding must reject "
                + invalid_names[index]
                + " in mode "
                + String(lazy),
            )
    for interval in [
        String(""),
        String("0d"),
        String("-1d"),
        String("1nope"),
    ]:
        with assert_raises():
            _ = frame.select(col("t").dt().truncate(interval))
    var overflow = DataFrame(
        [tagged("t", DataType.datetime("ns"), [Int64.MAX])]
    )
    with assert_raises():
        _ = overflow.select(col("t").dt().offset_by("1ns"))
    var shielded = DataFrame(
        [tagged("t", DataType.datetime("ns"), [Int64.MAX], [False])]
    )
    assert_true(
        shielded.select(col("t").dt().offset_by("1ns")).item(0, "t").is_null()
    )


def assert_time_aggregate_row(
    actual: DataFrame,
    row: Int,
    rows: List[Int],
    values: List[Int64],
    valid: List[Bool],
) raises:
    var count = 0
    var minimum: Int64 = 0
    var maximum: Int64 = 0
    for member in rows:
        if valid[member]:
            if count == 0:
                minimum = values[member]
                maximum = values[member]
            else:
                minimum = min(minimum, values[member])
                maximum = max(maximum, values[member])
            count += 1
    assert_equal(
        actual.item(row, "sum").int64(), sum_members(values, valid, rows)
    )
    assert_equal(actual.item(row, "count").int64(), Int64(count))
    assert_equal(actual.item(row, "len").int64(), Int64(len(rows)))
    for name in [String("min"), String("max"), String("mean")]:
        assert_equal(actual.item(row, name).is_null(), count == 0)
    if count:
        assert_equal(actual.item(row, "min").int64(), minimum)
        assert_equal(actual.item(row, "max").int64(), maximum)
        assert_float(
            actual.item(row, "mean").float64(),
            Float64(sum_members(values, valid, rows)) / Float64(count),
        )
    for terminal in range(2):
        var name = String("first") if terminal == 0 else String("last")
        var index = (
            rows[0] if len(rows)
            and terminal == 0 else rows[len(rows) - 1] if len(rows) else 0
        )
        var present = len(rows) > 0 and valid[index]
        assert_equal(actual.item(row, name).is_null(), not present)
        if present:
            assert_equal(actual.item(row, name).int64(), values[index])


def time_aggregates() -> List[Expr]:
    return [
        col("x").sum().alias("sum"),
        col("x").count().alias("count"),
        col("x").len().alias("len"),
        col("x").min().alias("min"),
        col("x").max().alias("max"),
        col("x").mean().alias("mean"),
        col("x").first().alias("first"),
        col("x").last().alias("last"),
    ]


def assert_time_aggregate_schema(actual: DataFrame, grouped: Bool) raises:
    var names: List[String] = [
        "t",
        "sum",
        "count",
        "len",
        "min",
        "max",
        "mean",
        "first",
        "last",
    ]
    if grouped:
        names.insert(0, "g")
    assert_equal(actual.columns(), names)
    for name in [
        String("sum"),
        String("count"),
        String("len"),
        String("min"),
        String("max"),
        String("first"),
        String("last"),
    ]:
        assert_true(actual.column(name).dtype() == DataType.INT64)
    assert_true(actual.column("mean").dtype() == DataType.FLOAT64)
    assert_true(actual.column("t").dtype() == DataType.datetime("ms"))


def test_dynamic_time_groups_match_scalar_boundaries_closed_and_labels() raises:
    var times: List[Int64] = [0, 0, 3600000, 7200000, 10800000, 18000000]
    var values: List[Int64] = [4, 9999999, 1, 7, -9999999, -2]
    var valid: List[Bool] = [True, False, True, True, False, True]
    var source = DataFrame(
        [
            tagged("t", DataType.datetime("ms"), times),
            Series("x", Column[Int64](values.copy(), valid.copy())),
        ]
    )
    for layout in range(3):
        var frame = (
            source.copy() if layout
            == 0 else offset_layout(source) if layout
            == 1 else chunk_layout(source)
        )
        for closed in [
            String("left"),
            String("right"),
            String("both"),
            String("none"),
        ]:
            for label in [String("left"), String("right"), String("datapoint")]:
                for hours in [1, 2, 3]:
                    for shifted in range(2):
                        var offset = String("30m") if shifted else String("0h")
                        # The 2h lattice with a +30m offset starts at -90m.
                        # Excluding a zero-offset left edge needs its preceding slot.
                        var first_start = Int64(
                            -5400000 if shifted else -7200000 if closed
                            in ["right", "none"] else 0
                        )
                        var expected_rows = List[List[Int]]()
                        var labels = List[Int64]()
                        var start = first_start
                        while start <= 18000000:
                            var end = start + Int64(hours) * 3600000
                            var rows = members(times, start, end, closed)
                            if len(rows):
                                labels.append(
                                    start if label
                                    == "left" else end if label
                                    == "right" else times[rows[0]]
                                )
                                expected_rows.append(rows^)
                            start += 7200000
                        for batch in [1, 64]:
                            var actual = frame.group_by_dynamic(
                                "t",
                                "2h",
                                period=String(hours) + "h",
                                offset=offset,
                                closed=closed,
                                label=label,
                            ).agg(time_aggregates(), batch_size=batch)
                            assert_time_aggregate_schema(actual, False)
                            assert_ticks(
                                actual, "t", DataType.datetime("ms"), labels
                            )
                            for row in range(len(labels)):
                                assert_time_aggregate_row(
                                    actual,
                                    row,
                                    expected_rows[row],
                                    values,
                                    valid,
                                )
    var empty = (
        source.clear().group_by_dynamic("t", "1h").agg(time_aggregates())
    )
    assert_equal(empty.height(), 0)
    assert_time_aggregate_schema(empty, False)


def test_rolling_time_groups_and_expressions_preserve_duplicate_memberships() raises:
    var times: List[Int64] = [0, 0, 3600000, 7200000, 10800000, 18000000]
    var values: List[Int64] = [4, 9999999, 1, 7, -9999999, -2]
    var valid: List[Bool] = [True, False, True, True, False, True]
    for grouped in range(2):
        var keys: List[Int] = [0, 1, 0, 1, 0, 1] if grouped else [
            0,
            0,
            0,
            0,
            0,
            0,
        ]
        var numeric_keys = List[Int64]()
        for key in keys:
            numeric_keys.append(Int64(key))
        var source = DataFrame(
            [
                tagged("t", DataType.datetime("ms"), times),
                Series("g", Column[Int64](numeric_keys^)),
                Series("x", Column[Int64](values.copy(), valid.copy())),
            ]
        )
        var group_names: List[String] = ["g"] if grouped else []
        for layout in range(3):
            var frame = (
                source.copy() if layout
                == 0 else offset_layout(source) if layout
                == 1 else chunk_layout(source)
            )
            for closed in [
                String("left"),
                String("right"),
                String("both"),
                String("none"),
            ]:
                for center in range(2):
                    var offset = String("-1h") if center else String("-2h")
                    var actual = frame.rolling(
                        "t",
                        "2h",
                        offset=offset,
                        closed=closed,
                        group_by=group_names,
                    ).agg(time_aggregates(), batch_size=1)
                    assert_time_aggregate_schema(actual, Bool(grouped))
                    assert_equal(actual.height(), len(times))
                    var output_row = 0
                    for key in range(2 if grouped else 1):
                        for row in range(len(times)):
                            if keys[row] != key:
                                continue
                            var start = times[row] - Int64(
                                3600000 if center else 7200000
                            )
                            var rows = members(
                                times, start, start + 7200000, closed, keys, key
                            )
                            assert_equal(
                                actual.item(output_row, "t").to_physical(),
                                times[row],
                            )
                            if grouped:
                                assert_equal(
                                    actual.item(output_row, "g").int64(),
                                    Int64(key),
                                )
                            assert_time_aggregate_row(
                                actual, output_row, rows, values, valid
                            )
                            output_row += 1
                var expressions: List[Expr] = [
                    col("x")
                    .rolling_sum_by("t", "2h", closed=closed)
                    .alias("sum"),
                    col("x")
                    .rolling_mean_by(col("t"), "2h", closed=closed)
                    .alias("mean"),
                    col("x")
                    .rolling_min_by("t", "2h", closed=closed)
                    .alias("min"),
                    col("x")
                    .rolling_max_by("t", "2h", closed=closed)
                    .alias("max"),
                    col("x")
                    .rolling_var_by("t", "2h", closed=closed, ddof=0)
                    .alias("var"),
                    col("x")
                    .rolling_std_by("t", "2h", closed=closed, ddof=1)
                    .alias("std"),
                    col("x")
                    .rolling_sum_by("t", "2h", closed=closed, min_samples=2)
                    .alias("two"),
                ]
                if grouped:
                    for index in range(len(expressions)):
                        expressions[index] = expressions[index].over("g")
                for lazy in range(2):
                    var rolling = (
                        frame.select_exprs(expressions) if lazy
                        == 0 else frame.lazy()
                        .select_exprs(expressions)
                        .collect()
                    )
                    assert_equal(
                        rolling.columns(),
                        List[String](
                            ["sum", "mean", "min", "max", "var", "std", "two"]
                        ),
                    )
                    for row in range(len(times)):
                        var rows = members(
                            times,
                            times[row] - 7200000,
                            times[row],
                            closed,
                            keys,
                            keys[row],
                        )
                        var sample = List[Int64]()
                        for member in rows:
                            if valid[member]:
                                sample.append(values[member])
                        var count = len(sample)
                        for name in [
                            String("sum"),
                            String("mean"),
                            String("min"),
                            String("max"),
                            String("var"),
                        ]:
                            assert_equal(
                                rolling.item(row, name).is_null(), count == 0
                            )
                        assert_equal(
                            rolling.item(row, "std").is_null(), count <= 1
                        )
                        assert_equal(
                            rolling.item(row, "two").is_null(), count < 2
                        )
                        if count:
                            var sum = sum_members(values, valid, rows)
                            var mean = Float64(sum) / Float64(count)
                            var minimum = sample[0]
                            var maximum = sample[0]
                            var squares: Float64 = 0
                            for value in sample:
                                minimum = min(minimum, value)
                                maximum = max(maximum, value)
                                squares += (Float64(value) - mean) * (
                                    Float64(value) - mean
                                )
                            assert_equal(rolling.item(row, "sum").int64(), sum)
                            assert_float(
                                rolling.item(row, "mean").float64(), mean
                            )
                            assert_equal(
                                rolling.item(row, "min").int64(), minimum
                            )
                            assert_equal(
                                rolling.item(row, "max").int64(), maximum
                            )
                            assert_statistic(
                                rolling.item(row, "var").float64(),
                                squares / Float64(count),
                            )
                            if count > 1:
                                assert_statistic(
                                    rolling.item(row, "std").float64(),
                                    sqrt(squares / Float64(count - 1)),
                                )
                                assert_equal(
                                    rolling.item(row, "two").int64(), sum
                                )


def test_time_windows_calendar_leap_days_and_dst_elapsed_difference() raises:
    var dates: List[Int64] = [
        civil_days(2024, 1, 31),
        civil_days(2024, 2, 1),
        civil_days(2024, 2, 29),
        civil_days(2024, 3, 1),
    ]
    var frame = DataFrame(
        [
            tagged("t", DataType.DATE, dates),
            Series("x", Column[Int64]([1, 2, 4, 8])),
        ]
    )
    var monthly = frame.group_by_dynamic("t", "1mo").agg(
        [col("x").sum().alias("sum"), col("x").len().alias("len")]
    )
    assert_ticks(
        monthly,
        "t",
        DataType.DATE,
        [
            civil_days(2024, 1, 1),
            civil_days(2024, 2, 1),
            civil_days(2024, 3, 1),
        ],
    )
    assert_ticks(monthly, "sum", DataType.INT64, [1, 6, 8])
    assert_ticks(monthly, "len", DataType.INT64, [1, 2, 1])
    var rolling = frame.rolling("t", "1mo", closed="both").agg(col("x").sum())
    assert_ticks(rolling, "x", DataType.INT64, [1, 3, 7, 14])
    var ny = String("America/New_York")
    var times: List[Int64] = [
        civil_ns(2024, 3, 9, 16, 30) // 1000,
        civil_ns(2024, 3, 9, 17) // 1000,
        civil_ns(2024, 3, 10, 16) // 1000,
    ]
    var zoned = DataFrame(
        [
            tagged("t", DataType.datetime("us", ny), times),
            Series("x", Column[Int64]([1, 2, 4])),
        ]
    )
    var result = zoned.select_exprs(
        [
            col("x").rolling_sum_by("t", "1d", closed="both").alias("calendar"),
            col("x").rolling_sum_by("t", "24h", closed="both").alias("elapsed"),
        ]
    )
    assert_ticks(result, "calendar", DataType.INT64, [1, 3, 6])
    assert_ticks(result, "elapsed", DataType.INT64, [1, 3, 7])
    var grouped = zoned.group_by_dynamic("t", "1d").agg(col("x").sum())
    assert_ticks(
        grouped,
        "t",
        DataType.datetime("us", ny),
        [civil_ns(2024, 3, 9, 5) // 1000, civil_ns(2024, 3, 10, 5) // 1000],
    )
    assert_ticks(grouped, "x", DataType.INT64, [3, 4])


def test_time_window_requests_reject_invalid_indices_and_bindings() raises:
    var source = DataFrame(
        [
            tagged("t", DataType.datetime("ms"), [0, 1000]),
            Series("x", Column[Int64]([1, 2])),
        ]
    )
    for period in [
        String("0s"),
        String("-1s"),
        String("nonsense"),
        String("1us"),
    ]:
        with assert_raises():
            _ = source.rolling("t", period).agg(col("x").sum())
        with assert_raises():
            _ = source.select(col("x").rolling_sum_by("t", period))
    with assert_raises():
        _ = source.group_by_dynamic("t", "0s").agg(col("x").sum())
    with assert_raises():
        _ = source.group_by_dynamic("t", "1s", closed="bad").agg(col("x").sum())
    with assert_raises():
        _ = source.group_by_dynamic("t", "1s", label="bad").agg(col("x").sum())
    with assert_raises():
        _ = source.group_by_dynamic("t", "1s", group_by="t").agg(col("x").sum())
    with assert_raises():
        _ = source.group_by_dynamic("t", "1s").agg(col("x"))
    with assert_raises():
        _ = source.group_by_dynamic("t", "1s").agg(col("x").sum().alias("t"))
    with assert_raises():
        _ = source.rolling("t", "1s").agg(col("x").sum(), batch_size=0)
    var invalid = List[DataFrame]()
    invalid.append(source.take([1, 0]))
    invalid.append(
        DataFrame(
            [
                tagged("t", DataType.datetime("ms"), [0, 1000], [True, False]),
                Series("x", Column[Int64]([1, 2])),
            ]
        )
    )
    invalid.append(
        DataFrame(
            [
                Series("t", Column[Int64]([0, 1000])),
                Series("x", Column[Int64]([1, 2])),
            ]
        )
    )
    for frame in invalid:
        with assert_raises():
            _ = frame.rolling("t", "1s").agg(col("x").sum())
        with assert_raises():
            _ = frame.group_by_dynamic("t", "1s").agg(col("x").sum())
        with assert_raises():
            _ = (
                frame.lazy()
                .select(col("x").rolling_sum_by("t", "1s"))
                .collect()
            )


def test_temporal_ranges_are_inclusive_and_months_keep_the_start_day() raises:
    var monthly = DataFrame(
        [date_range("2024-01-31", "2024-04-30", "1mo", name="d")]
    )
    assert_ticks(
        monthly,
        "d",
        DataType.DATE,
        [
            civil_days(2024, 1, 31),
            civil_days(2024, 2, 29),
            civil_days(2024, 3, 31),
            civil_days(2024, 4, 30),
        ],
    )
    var days = DataFrame([date_range("1969-12-31", "1970-01-02", name="d")])
    assert_ticks(days, "d", DataType.DATE, [-1, 0, 1])
    var descending = date_range("2024-03-01", "2024-02-29")
    assert_equal(len(descending), 0)
    assert_true(descending.dtype() == DataType.DATE)
    for unit in [String("ms"), String("us"), String("ns")]:
        var scale = Int64(
            1 if unit == "ms" else 1000 if unit == "us" else 1000000
        )
        var clock = DataFrame(
            [
                datetime_range(
                    "1969-12-31 23:59:59.5",
                    "1970-01-01 00:00:00.5",
                    "500ms",
                    unit=unit,
                    name="t",
                )
            ]
        )
        assert_ticks(
            clock, "t", DataType.datetime(unit), [-500 * scale, 0, 500 * scale]
        )
    var ny = String("America/New_York")
    var calendar = DataFrame(
        [
            datetime_range(
                "2024-03-09 12:00",
                "2024-03-11 12:00",
                "1d",
                time_zone=ny,
                name="t",
            )
        ]
    )
    var elapsed = DataFrame(
        [
            datetime_range(
                "2024-03-09 12:00",
                "2024-03-11 12:00",
                "24h",
                time_zone=ny,
                name="t",
            )
        ]
    )
    assert_ticks(
        calendar,
        "t",
        DataType.datetime("us", ny),
        [
            civil_ns(2024, 3, 9, 17) // 1000,
            civil_ns(2024, 3, 10, 16) // 1000,
            civil_ns(2024, 3, 11, 16) // 1000,
        ],
    )
    assert_ticks(
        elapsed,
        "t",
        DataType.datetime("us", ny),
        [civil_ns(2024, 3, 9, 17) // 1000, civil_ns(2024, 3, 10, 17) // 1000],
    )
    for interval in [String("0d"), String("-1d"), String("1h"), String("bad")]:
        with assert_raises():
            _ = date_range("2024-01-01", "2024-01-02", interval)
    with assert_raises():
        _ = datetime_range(
            "1970-01-01 00:00", "1970-01-01 00:01", "1us", unit="ms"
        )


def test_temporal_reference_calendar_and_membership_literal_anchors() raises:
    assert_equal(civil_days(1, 1, 1), Int64(-719162))
    assert_equal(civil_days(1970, 1, 1), Int64(0))
    assert_equal(civil_days(2000, 3, 1) - civil_days(2000, 2, 28), Int64(2))
    assert_equal(civil_days(1900, 3, 1) - civil_days(1900, 2, 28), Int64(1))
    assert_equal(civil_ns(1969, 12, 31, 23, 59, 59, 999999999), Int64(-1))
    var times: List[Int64] = [-1, 0, 0, 1, 2, 3]
    assert_equal(members(times, 0, 2, "left"), List[Int]([1, 2, 3]))
    assert_equal(members(times, 0, 2, "right"), List[Int]([3, 4]))
    assert_equal(members(times, 0, 2, "both"), List[Int]([1, 2, 3, 4]))
    assert_equal(members(times, 0, 2, "none"), List[Int]([3]))
    assert_equal(
        members(times, 0, 2, "both", [0, 1, 0, 1, 0, 1], 0), List[Int]([2, 4])
    )
    assert_equal(
        sum_members([1, 999, 3, 4], [True, False, True, True], [0, 1, 2]),
        Int64(4),
    )


def test_dynamic_time_groups_check_sortedness_within_nullable_partitions() raises:
    var times: List[Int64] = [0, 1000, 5000, 2000]
    var source = DataFrame(
        [
            tagged("t", DataType.datetime("ms"), times),
            Series(
                "g",
                Column[String](
                    ["a", "poison", "a", "poison"], [True, False, True, False]
                ),
            ),
            Series("x", Column[Int64]([1, 2, 3, 4])),
        ]
    )
    for layout in range(3):
        var frame = (
            source.copy() if layout
            == 0 else offset_layout(source) if layout
            == 1 else chunk_layout(source)
        )
        var grouped = frame.group_by_dynamic("t", "2s", group_by="g").agg(
            col("x").sum()
        )
        assert_equal(grouped.columns(), List[String](["g", "t", "x"]))
        assert_ticks(grouped, "t", DataType.datetime("ms"), [0, 4000, 0, 2000])
        assert_ticks(grouped, "x", DataType.INT64, [1, 3, 2, 4])
        assert_true(grouped.column("g").dtype() == DataType.STRING)
        for row in range(4):
            assert_equal(grouped.item(row, "g").is_null(), row >= 2)
            if row < 2:
                assert_equal(grouped.item(row, "g").string(), "a")
        with assert_raises():
            _ = frame.group_by_dynamic("t", "2s").agg(col("x").sum())
        var malformed = frame.take([2, 1, 0, 3])
        with assert_raises():
            _ = malformed.group_by_dynamic("t", "2s", group_by="g").agg(
                col("x").sum()
            )


def test_temporal_duration_totals_and_numeric_operators_include_negative_rounding() raises:
    for unit in [String("ms"), String("us"), String("ns")]:
        var per_second = Int64(
            1000 if unit == "ms" else 1000000 if unit == "us" else 1000000000
        )
        var frame = DataFrame(
            [
                tagged(
                    "a",
                    DataType.duration(unit),
                    [-3, 3, -per_second + 1, Int64.MIN],
                    [True, True, True, False],
                ),
                tagged(
                    "b",
                    DataType.duration(unit),
                    [2, -2, 1, Int64.MAX],
                    [True, True, True, False],
                ),
                Series("n", Column[Int64]([2, -2, 2, -1])),
            ]
        )
        var result = frame.select_exprs(
            [
                (col("a") + col("b")).alias("sum"),
                (col("a") - col("b")).alias("difference"),
                (col("a") // col("n")).alias("floor"),
                col("a").dt().total("microseconds").alias("micros"),
                col("a").dt().total("nanoseconds").alias("nanos"),
            ]
        )
        var valid: List[Bool] = [True, True, True, False]
        assert_ticks(
            result,
            "sum",
            DataType.duration(unit),
            [-1, 1, -per_second + 2, 0],
            valid,
        )
        assert_ticks(
            result,
            "difference",
            DataType.duration(unit),
            [-5, 5, -per_second, 0],
            valid,
        )
        assert_ticks(
            result,
            "floor",
            DataType.duration(unit),
            [-2, -2, -per_second // 2, 0],
            valid,
        )
        var to_ns = Int64(
            1000000 if unit == "ms" else 1000 if unit == "us" else 1
        )
        assert_ticks(
            result,
            "nanos",
            DataType.INT64,
            [-3 * to_ns, 3 * to_ns, (-per_second + 1) * to_ns, 0],
            valid,
        )
        var micros: List[Int64] = [-3000, 3000, -999000, 0]
        if unit == "us":
            micros = [-3, 3, -999999, 0]
        elif unit == "ns":
            micros = [0, 0, -999999, 0]
        assert_ticks(result, "micros", DataType.INT64, micros, valid)


def test_temporal_subtraction_equal_minimum_ticks_cancel_exactly() raises:
    for unit in [String("ms"), String("us"), String("ns")]:
        for kind in range(2):
            var dtype = DataType.datetime(unit) if kind else DataType.duration(
                unit
            )
            var frame = DataFrame(
                [
                    tagged("a", dtype, [Int64.MIN]),
                    tagged("b", dtype, [Int64.MIN]),
                ]
            )
            var actual = frame.select((col("a") - col("b")).alias("result"))
            assert_ticks(actual, "result", DataType.duration(unit), [0])


def test_temporal_subtraction_rejects_unrepresentable_minimum_rhs() raises:
    for unit in [String("ms"), String("us"), String("ns")]:
        for kind in range(2):
            var dtype = DataType.datetime(unit) if kind else DataType.duration(
                unit
            )
            var frame = DataFrame(
                [tagged("a", dtype, [0]), tagged("b", dtype, [Int64.MIN])]
            )
            var rejected = False
            try:
                _ = frame.select(col("a") - col("b"))
            except error:
                if String(error).find("temporal value overflows Int64") < 0:
                    raise error^
                rejected = True
            assert_true(
                rejected,
                "minimum temporal RHS must reject an out-of-range difference",
            )


def test_temporal_strftime_escaped_percent_formats_one_literal_percent() raises:
    var source = DataFrame([tagged("d", DataType.DATE, [0])])
    var actual = source.select(col("d").dt().strftime("%Y %% %j"))
    assert_true(
        actual.item(0, "d").string() == "1970 % 001",
        "strftime escaped percent must emit one literal percent",
    )


def test_duration_minimum_tick_totals_keep_the_negative_sign() raises:
    var source = DataFrame([tagged("d", DataType.duration("ns"), [Int64.MIN])])
    var actual = source.select(col("d").dt().total_seconds())
    assert_true(
        actual.item(0, "d").int64() == Int64(-9223372036),
        "minimum duration whole seconds must truncate toward zero without sign overflow",
    )


def test_temporal_offsets_skip_unobserved_overflow_in_conditional_branches() raises:
    var types: List[DataType] = [
        DataType.DATE,
        DataType.datetime("ms"),
        DataType.datetime("us"),
        DataType.datetime("ns"),
        DataType.TIME,
    ]
    for dtype in types:
        var tick = (
            "1d" if dtype.is_date() else "1ms" if dtype.per_second()
            == 1000 else "1us" if dtype.per_second()
            == 1000000 else "1ns"
        )
        var frame = DataFrame(
            [
                Series(
                    "t", Column[Int64]([Int64.MAX, Int64.MIN, 0])
                ).with_dtype(dtype)
            ]
        )
        var expression = (
            when(col("t") < lit(Int64.MAX).cast(dtype))
            .then(col("t").dt().offset_by(tick))
            .otherwise(lit(Int64(0)).cast(dtype))
            .alias("v")
        )
        var low = Int64.MIN + 1
        if dtype.is_time():
            low %= Int64(86400000000000)
        var expected: List[Int64] = [0, low, 1]
        for streaming in [False, True]:
            var actual = (
                frame.lazy()
                .select(expression)
                .collect(engine="cpu", streaming=streaming)
            )
            assert_equal(
                [
                    actual.column("v").int64()._get(0),
                    actual.column("v").int64()._get(1),
                    actual.column("v").int64()._get(2),
                ],
                expected,
            )
        var inactive = (
            when(lit(False))
            .then(lit(Int64.MAX).cast(dtype).dt().offset_by(tick))
            .otherwise(lit(Int64(0)).cast(dtype))
            .alias("v")
        )
        var actual = frame.select(inactive)
        assert_equal(actual.column("v").dtype(), dtype)
        assert_equal(actual.column("v").int64()._get(0), Int64(0))


def test_calendar_truncation_skips_unobserved_datetime_underflow() raises:
    for dtype in List[DataType](
        [
            DataType.datetime("ms"),
            DataType.datetime("us"),
            DataType.datetime("ns"),
        ]
    ):
        var frame = DataFrame(
            [Series("t", Column[Int64]([Int64.MIN, 0])).with_dtype(dtype)]
        )
        var expression = (
            when(col("t") > lit(Int64.MIN).cast(dtype))
            .then(col("t").dt().truncate("1mo"))
            .otherwise(lit(Int64(0)).cast(dtype))
            .alias("v")
        )
        for streaming in [False, True]:
            var actual = (
                frame.lazy()
                .select(expression)
                .collect(engine="cpu", streaming=streaming)
            )
            assert_equal(actual.column("v").dtype(), dtype)
            assert_equal(actual.column("v").int64()._get(0), Int64(0))
            assert_equal(actual.column("v").int64()._get(1), Int64(0))
