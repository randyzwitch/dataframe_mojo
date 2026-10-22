"""Literal calendar anchors for temporal constructors and reporting helpers."""
# contract-group: expr.temporal
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    AnyValue,
    Column,
    DataFrame,
    DataType,
    Expr,
    Series,
    col,
    lit,
    null,
    when,
    date,
    datetime,
    time,
    duration,
    business_day_count,
)
from dataframe.temporal import parse, format
from tests.public_contracts.test_temporal import tagged, assert_ticks
from tests.support.coverage_generators import offset_layout, chunk_layout
from tests.support.temporal_reference import civil_days


def reporting_select(
    frame: DataFrame, expressions: List[Expr], mode: Int
) raises -> DataFrame:
    if mode == 0:
        return frame.select_exprs(expressions)
    return (
        frame.lazy()
        .select_exprs(expressions)
        .collect(engine="cpu", streaming=mode == 2)
    )


def test_reporting_calendar_fields_iso_boundaries_and_nulls() raises:
    var source = DataFrame(
        [
            tagged(
                "d",
                DataType.DATE,
                [
                    civil_days(2019, 12, 30),
                    civil_days(2021, 1, 1),
                    civil_days(2024, 2, 29),
                    civil_days(1900, 3, 1),
                    Int64.MAX,
                ],
                [True, True, True, True, False],
            )
        ]
    )
    for layout in range(3):
        var frame = (
            source.copy() if layout
            == 0 else offset_layout(source) if layout
            == 1 else chunk_layout(source)
        )
        for mode in range(3):
            var result = reporting_select(
                frame,
                [
                    col("d").dt().quarter().alias("q"),
                    col("d").dt().week().alias("w"),
                    col("d").dt().iso_year().alias("y"),
                    col("d").dt().century().alias("c"),
                    col("d").dt().millennium().alias("m"),
                    col("d").dt().days_in_month().alias("days"),
                    col("d").dt().is_leap_year().alias("leap"),
                    col("d").dt().month_start().alias("start"),
                    col("d").dt().month_end().alias("end"),
                ],
                mode,
            )
            var valid: List[Bool] = [True, True, True, True, False]
            assert_ticks(result, "q", DataType.INT64, [4, 1, 1, 1, 0], valid)
            assert_ticks(result, "w", DataType.INT64, [1, 53, 9, 9, 0], valid)
            assert_ticks(
                result, "y", DataType.INT64, [2020, 2020, 2024, 1900, 0], valid
            )
            assert_ticks(
                result, "c", DataType.INT64, [21, 21, 21, 19, 0], valid
            )
            assert_ticks(result, "m", DataType.INT64, [3, 3, 3, 2, 0], valid)
            assert_ticks(
                result, "days", DataType.INT64, [31, 31, 29, 31, 0], valid
            )
            assert_true(result.column("leap").dtype() == DataType.BOOL)
            assert_equal(result.item(2, "leap").bool(), True)
            assert_equal(result.item(3, "leap").bool(), False)
            assert_true(result.item(4, "leap").is_null())
            assert_ticks(
                result,
                "start",
                DataType.DATE,
                [
                    civil_days(2019, 12, 1),
                    civil_days(2021, 1, 1),
                    civil_days(2024, 2, 1),
                    civil_days(1900, 3, 1),
                    0,
                ],
                valid,
            )
            assert_ticks(
                result,
                "end",
                DataType.DATE,
                [
                    civil_days(2019, 12, 31),
                    civil_days(2021, 1, 31),
                    civil_days(2024, 2, 29),
                    civil_days(1900, 3, 31),
                    0,
                ],
                valid,
            )


def test_reporting_constructors_typed_literals_and_composition() raises:
    var frame = DataFrame(
        [
            Series(
                "y", Column[Int64]([2024, 2023, Int64.MAX], [True, True, False])
            )
        ]
    )
    for mode in range(3):
        var result = reporting_select(
            frame,
            [
                date(col("y"), 2, 28).alias("date"),
                datetime(col("y"), 2, 28, 12, 34, 56, 123456).alias("dt"),
                time(12, 34, 56, 123456).alias("time"),
                duration(
                    days=1, hours=-2, nanoseconds=-1, time_unit="ns"
                ).alias("duration"),
                (col("y") == lit(Int64(2024))).alias("year_match"),
                (date(col("y"), 2, 28) >= date(2024, 1, 1)).alias("date_match"),
                lit(AnyValue.temporal(DataType.DATE, 19782)).alias("literal"),
                lit(AnyValue.null(DataType.DATE)).alias("null"),
                lit(Int64(-1), DataType.datetime("ns")).alias("negative"),
                date(col("y"), 1, 1)
                .dt()
                .combine(time(1, 2, 3))
                .alias("combined"),
                duration(seconds=col("y")).dt().total_seconds().alias("total"),
            ],
            mode,
        )
        assert_ticks(
            result,
            "date",
            DataType.DATE,
            [19781, 19416, 0],
            [True, True, False],
        )
        assert_ticks(
            result,
            "dt",
            DataType.datetime("us"),
            [1709123696123456, 1677587696123456, 0],
            [True, True, False],
        )
        assert_ticks(
            result,
            "time",
            DataType.TIME,
            [45296123456000, 45296123456000, 45296123456000],
        )
        assert_ticks(
            result,
            "duration",
            DataType.duration("ns"),
            [79199999999999, 79199999999999, 79199999999999],
        )
        assert_equal(result.item(0, "date_match").bool(), True)
        assert_equal(result.item(1, "date_match").bool(), False)
        assert_true(result.item(2, "date_match").is_null())
        assert_ticks(result, "literal", DataType.DATE, [19782, 19782, 19782])
        assert_true(result.item(0, "null").is_null())
        assert_ticks(result, "negative", DataType.datetime("ns"), [-1, -1, -1])
        assert_ticks(
            result,
            "combined",
            DataType.datetime("us"),
            [1704070923000000, 1672534923000000, 0],
            [True, True, False],
        )
        assert_ticks(
            result,
            "total",
            DataType.INT64,
            [2024, 2023, 0],
            [True, True, False],
        )


def test_reporting_epoch_subseconds_and_unit_retagging() raises:
    for unit in [String("ns"), String("us"), String("ms")]:
        var dtype = DataType.datetime(unit, "+05:30")
        var frame = DataFrame(
            [
                tagged(
                    "t", dtype, [-1, 0, 1, Int64.MAX], [True, True, True, False]
                )
            ]
        )
        for mode in range(3):
            var result = reporting_select(
                frame,
                [
                    col("t").dt().epoch("s").alias("s"),
                    col("t").dt().epoch("d").alias("d"),
                    col("t").dt().timestamp("ms").alias("ms"),
                    col("t").dt().microsecond().alias("micro"),
                    col("t").dt().millisecond().alias("milli"),
                    col("t").dt().with_time_unit("ms").alias("retag"),
                ],
                mode,
            )
            assert_ticks(
                result,
                "s",
                DataType.INT64,
                [-1, 0, 0, 0],
                [True, True, True, False],
            )
            assert_ticks(
                result,
                "d",
                DataType.INT64,
                [-1, 0, 0, 0],
                [True, True, True, False],
            )
            assert_ticks(
                result,
                "ms",
                DataType.INT64,
                [-1, 0, Int64(1 if unit == "ms" else 0), 0],
                [True, True, True, False],
            )
            assert_ticks(
                result,
                "micro",
                DataType.INT64,
                [
                    Int64(999000 if unit == "ms" else 999999),
                    0,
                    Int64(1000 if unit == "ms" else 1 if unit == "us" else 0),
                    0,
                ],
                [True, True, True, False],
            )
            assert_ticks(
                result,
                "milli",
                DataType.INT64,
                [999, 0, Int64(1 if unit == "ms" else 0), 0],
                [True, True, True, False],
            )
            assert_ticks(
                result,
                "retag",
                DataType.datetime("ms", "+05:30"),
                [-1, 0, 1, 0],
                [True, True, True, False],
            )


def test_reporting_local_month_edges_round_replace_and_dst() raises:
    var dtype = DataType.datetime("us", "America/New_York")
    var source = parse("2024-03-10 03:30:00", dtype)
    var frame = DataFrame([tagged("t", dtype, [source])])
    for mode in range(3):
        var result = reporting_select(
            frame,
            [
                col("t").dt().month_start().alias("start"),
                col("t").dt().month_end().alias("end"),
                col("t").dt().round("1h").alias("rounded"),
                col("t").dt().replace(day=1, hour=1).alias("replace"),
                col("t").dt().combine(time(0, 30)).alias("combine"),
                col("t").dt().combine(time(12, 30), "ns").alias("combine_ns"),
            ],
            mode,
        )
        assert_ticks(
            result, "start", dtype, [parse("2024-03-01 03:30:00", dtype)]
        )
        assert_ticks(
            result, "end", dtype, [parse("2024-03-31 03:30:00", dtype)]
        )
        assert_ticks(
            result, "rounded", dtype, [parse("2024-03-10 04:00:00", dtype)]
        )
        assert_ticks(
            result, "replace", dtype, [parse("2024-03-01 01:30:00", dtype)]
        )
        assert_ticks(result, "combine", dtype, [1710048600000000])
        assert_ticks(
            result,
            "combine_ns",
            DataType.datetime("ns", "America/New_York"),
            [1710088200000000000],
        )
    var dates = DataFrame(
        [tagged("d", DataType.DATE, [19737, 19739, 19740, -1])]
    )
    assert_ticks(
        dates.select_exprs([col("d").dt().round("1mo").alias("r")]),
        "r",
        DataType.DATE,
        [19723, 19723, 19754, 0],
    )
    var negative = DataFrame(
        [tagged("t", DataType.datetime("us"), [-1500001, -1500000, -1499999])]
    )
    assert_ticks(
        negative.select_exprs([col("t").dt().round("1s").alias("r")]),
        "r",
        DataType.datetime("us"),
        [-2000000, -1000000, -1000000],
    )
    with assert_raises(contains="non-existent"):
        _ = frame.select_exprs([col("t").dt().replace(hour=2)])


def test_reporting_business_calendars_count_roll_holidays_and_nulls() raises:
    var frame = DataFrame(
        [
            tagged(
                "d",
                DataType.DATE,
                [19727, 19728, 19730, Int64.MAX],
                [True, True, True, False],
            ),
            tagged("end", DataType.DATE, [19734, 19727, 19727, 0]),
            Series("n", Column[Int64]([1, -1, 7, 1])),
        ]
    )
    for mode in range(3):
        var result = reporting_select(
            frame,
            [
                col("d")
                .dt()
                .is_business_day(holidays=["2024-01-08"])
                .alias("is"),
                col("d")
                .dt()
                .add_business_days(
                    col("n"),
                    holidays=["2024-01-08", "2024-01-08"],
                    roll="forward",
                )
                .alias("add"),
                business_day_count(
                    col("d"), col("end"), holidays=["2024-01-08"]
                ).alias("count"),
                col("d")
                .dt()
                .add_business_days(0, roll="backward")
                .alias("back"),
            ],
            mode,
        )
        assert_equal(result.item(0, "is").bool(), True)
        assert_equal(result.item(1, "is").bool(), False)
        assert_equal(result.item(2, "is").bool(), False)
        assert_ticks(
            result,
            "add",
            DataType.DATE,
            [19731, 19727, 19740, 0],
            [True, True, True, False],
        )
        assert_ticks(
            result,
            "count",
            DataType.INT64,
            [4, 0, 0, 0],
            [True, True, True, False],
        )
        assert_ticks(
            result,
            "back",
            DataType.DATE,
            [19727, 19727, 19730, 0],
            [True, True, True, False],
        )
    with assert_raises(contains="not a business day"):
        _ = frame.select_exprs([col("d").dt().add_business_days(0)])
    var custom = frame.select_exprs(
        [
            col("d")
            .dt()
            .is_business_day(
                week_mask=[False, False, False, False, False, True, True]
            )
            .alias("is")
        ]
    )
    assert_equal(custom.item(1, "is").bool(), True)


def test_reporting_extended_format_directives_and_validation() raises:
    var dtype = DataType.datetime("us")
    var ticks = Int64(1709211906123456)  # 2024-02-29 13:05:06.123456
    var pattern = String("%A %a %B %b %y %I %p %e %U %W %u")
    assert_equal(
        format(ticks, dtype, pattern),
        "Thursday Thu February Feb 24 01 PM 29 08 09 4",
    )
    assert_equal(
        parse(
            "Thu February 29 24 01:05:06 PM", dtype, "%a %B %e %y %I:%M:%S %p"
        ),
        Int64(1709211906000000),
    )
    assert_equal(parse("2024 09 4", DataType.DATE, "%Y %W %u"), Int64(19782))
    assert_equal(parse("2024 08 Thu", DataType.DATE, "%Y %U %a"), Int64(19782))
    assert_equal(
        parse("69-01-01", DataType.DATE, "%y-%m-%d"), civil_days(2069, 1, 1)
    )
    assert_equal(
        parse("70-01-01", DataType.DATE, "%y-%m-%d"), civil_days(1970, 1, 1)
    )
    for text in [
        String("Tue February 29 24 01:05:06 PM"),
        String("Thu February 30 24 01:05:06 PM"),
        String("Thu February 29 24 00:05:06 PM"),
    ]:
        with assert_raises():
            _ = parse(text, dtype, "%a %B %e %y %I:%M:%S %p")
    var frame = DataFrame(
        [
            Series(
                "s", Column[Int64]([ticks, Int64.MAX], [True, False])
            ).with_dtype(dtype)
        ]
    )
    for mode in range(3):
        var result = reporting_select(
            frame,
            [
                col("s")
                .dt()
                .strftime("%d %b %Y %I:%M:%S %p")
                .str()
                .strptime("datetime[us]", "%d %b %Y %I:%M:%S %p")
                .alias("roundtrip")
            ],
            mode,
        )
        assert_ticks(
            result, "roundtrip", dtype, [1709211906000000, 0], [True, False]
        )


def test_reporting_errors_and_unobserved_invalid_constructors() raises:
    var frame = DataFrame(
        [
            Series("day", Column[Int64]([29, 30])),
            Series("active", Column[Int64]([1, 0])),
        ]
    )
    for mode in range(3):
        var result = reporting_select(
            frame,
            [
                when(col("active") == 1)
                .then(date(2024, 2, col("day")))
                .otherwise(null("date"))
                .alias("d")
            ],
            mode,
        )
        assert_ticks(result, "d", DataType.DATE, [19782, 0], [True, False])
        with assert_raises(contains="day out of range"):
            _ = reporting_select(frame, [date(2024, 2, col("day"))], mode)
        with assert_raises(contains="time of day out of range"):
            _ = reporting_select(frame, [time(-1)], mode)
        with assert_raises(contains="overflows"):
            _ = reporting_select(frame, [duration(weeks=lit(Int64.MAX))], mode)
        with assert_raises(contains="subsecond fields"):
            _ = reporting_select(
                frame, [date(2024, 1, 1).dt().microsecond()], mode
            )
        with assert_raises(contains="invalid epoch"):
            _ = reporting_select(
                frame, [date(2024, 1, 1).dt().epoch("bad")], mode
            )


def test_reporting_duration_subunit_truncation_and_minimum_datetime() raises:
    var frame = DataFrame(
        [Series("ns", Column[Int64]([-1001, -999, 999, 1001]))]
    )
    for mode in range(3):
        var result = reporting_select(
            frame,
            [
                duration(
                    milliseconds=1, nanoseconds=col("ns"), time_unit="us"
                ).alias("d")
            ],
            mode,
        )
        assert_ticks(
            result, "d", DataType.duration("us"), [999, 1000, 1000, 1001]
        )
    var dtype = DataType.datetime("ns")
    assert_equal(parse("1677-09-21 00:12:43.145224192", dtype), Int64.MIN)
    var invalid = DataFrame([Series("year", Column[Int64]([Int64.MAX]))])
    with assert_raises(contains="overflows"):
        _ = invalid.select_exprs([date(col("year"), 1, 1)])


def test_reporting_rounding_checks_boundaries_without_wrapping() raises:
    var dtype = DataType.datetime("ns")
    for value in [Int64.MIN, Int64.MAX]:
        var frame = DataFrame([tagged("t", dtype, [value])])
        for mode in range(3):
            assert_ticks(
                reporting_select(
                    frame, [col("t").dt().round("1ns").alias("r")], mode
                ),
                "r",
                dtype,
                [value],
            )
            with assert_raises(contains="overflows"):
                _ = reporting_select(frame, [col("t").dt().round("1s")], mode)


def test_reporting_filters_read_every_constructor_field() raises:
    # Only the later packed fields read columns: these are row predicates,
    # never constants that may be bound without the frame's columns.
    var frame = DataFrame(
        [
            tagged(
                "d", DataType.DATE, [19723, 19754, 19782, 19783, 19814, 19844]
            ),
            Series(
                "m",
                Column[Int64](
                    [1, 3, 2, 3, 5, Int64.MAX],
                    [True, True, True, True, True, False],
                ),
            ),
            Series(
                "day",
                Column[Int64](
                    [1, 1, 29, 1, 1, Int64.MAX],
                    [True, True, True, True, True, False],
                ),
            ),
        ]
    )
    for layout in range(3):
        var source = (
            frame.copy() if layout
            == 0 else offset_layout(frame) if layout
            == 1 else chunk_layout(frame)
        )
        for predicate in [
            col("d") >= date(2024, col("m"), 1),
            col("d") >= date(2024, col("m"), col("day")),
        ]:
            for mode in range(3):
                var actual = source.filter(
                    predicate
                ) if mode == 0 else source.lazy().filter(predicate).collect(
                    engine="cpu", streaming=mode == 2
                )
                assert_ticks(actual, "d", DataType.DATE, [19723, 19782, 19783])
    # An extra comparison exercises the fused conjunction path, while d
    # itself is sorted and non-null to exercise sorted-range admission.
    var actual = frame.filter(
        (col("d") >= date(2024, col("m"), 1)) & (col("d") < date(2024, 5, 1))
    )
    assert_ticks(actual, "d", DataType.DATE, [19723, 19782, 19783])
