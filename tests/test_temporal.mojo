"""Date, Datetime, Duration, and Time: calendar math, parsing, arithmetic,
dt operations, casts, CSV, and interaction with frame operations."""
from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
    assert_false,
    assert_raises,
)
from dataframe import (
    AnyValue,
    Column,
    CsvField,
    CsvSchema,
    DataFrame,
    DataType,
    Expr,
    Series,
    col,
    date_range,
    datetime_range,
    lit,
    read_csv,
    write_csv,
)
from dataframe.temporal import (
    civil_from_days,
    days_from_civil,
    format,
    parse,
)

comptime PATH = "/tmp/dataframe_mojo_temporal_test.csv"


def texts(series: Series) raises -> List[String]:
    var out = List[String]()
    for i in range(len(series)):
        out.append(String(series.get(i)))
    return out^


def one(frame: DataFrame, expr: Expr) raises -> List[String]:
    return texts(frame.select(expr.alias("r")).column("r"))


def dates() raises -> DataFrame:
    return DataFrame(
        [
            Series(
                "s",
                Column[String](
                    [
                        "2024-02-28",
                        "2024-02-29",
                        "1969-12-31",
                        "2000-01-01",
                        "1900-03-01",
                        "x",
                    ],
                    [True, True, True, True, True, False],
                ),
            )
        ]
    ).with_columns(col("s").str().to_date().alias("d"))


def test_calendar_round_trips() raises:
    # Every day across four centuries survives days <-> civil conversion.
    var previous = days_from_civil(1899, 12, 31)
    for days in range(
        Int(days_from_civil(1900, 1, 1)), Int(days_from_civil(2300, 1, 1)), 17
    ):
        var ymd = civil_from_days(Int64(days))
        assert_equal(days_from_civil(ymd[0], ymd[1], ymd[2]), Int64(days))
        assert_true(Int64(days) > previous)
        previous = Int64(days)
    assert_equal(days_from_civil(1970, 1, 1), Int64(0))
    assert_equal(
        days_from_civil(2000, 3, 1) - days_from_civil(2000, 2, 28), Int64(2)
    )
    assert_equal(
        days_from_civil(1900, 3, 1) - days_from_civil(1900, 2, 28), Int64(1)
    )
    assert_equal(format(Int64(-1), DataType.DATE), "1969-12-31")
    assert_equal(format(Int64(-719162), DataType.DATE), "0001-01-01")


def test_parsing_and_formatting() raises:
    var us = DataType.datetime("us")
    var ns = DataType.datetime("ns")
    assert_equal(
        format(parse("2024-02-29T13:45:07.25", us), us),
        "2024-02-29 13:45:07.250000",
    )
    assert_equal(
        format(parse("2024-02-29 13:45", us), us), "2024-02-29 13:45:00"
    )
    assert_equal(format(parse("2024-02-29", ns), ns), "2024-02-29 00:00:00")
    assert_equal(
        format(parse("1969-12-31 23:59:59.999999999", ns), ns),
        "1969-12-31 23:59:59.999999999",
    )
    assert_equal(
        format(parse("07:08:09.5", DataType.TIME), DataType.TIME),
        "07:08:09.500000000",
    )
    assert_equal(
        format(parse("31/12/1999", DataType.DATE, "%d/%m/%Y"), DataType.DATE),
        "1999-12-31",
    )
    assert_equal(
        format(parse("2024-060", DataType.DATE, "%Y-%j"), DataType.DATE),
        "2024-02-29",
    )
    assert_equal(
        format(
            parse("2024-02-29", DataType.DATE), DataType.DATE, "%d.%m.%Y (%j)"
        ),
        "29.02.2024 (060)",
    )
    var bad: List[String] = [
        "2023-02-29",
        "2024-13-01",
        "2024-1-01",
        "2024-01-01x",
        "2024-01-01 24:00",
        "",
    ]
    for text in bad:
        with assert_raises(contains="cannot parse"):
            _ = parse(text, us)
    with assert_raises(contains="unsupported format directive"):
        _ = parse("2024", DataType.DATE, "%Q")


def test_dtypes_display_and_values() raises:
    var df = dates()
    assert_true(df.column("d").dtype() == DataType.DATE)
    assert_equal(
        texts(df.column("d")),
        [
            String("2024-02-28"),
            "2024-02-29",
            "1969-12-31",
            "2000-01-01",
            "1900-03-01",
            "null",
        ],
    )
    assert_true(String(df).find("┆ date") >= 0)
    var cell = df.item(0, "d")
    assert_true(cell.dtype() == DataType.DATE)
    assert_equal(cell.to_physical(), days_from_civil(2024, 2, 28))
    assert_true(
        cell == AnyValue.temporal(DataType.DATE, days_from_civil(2024, 2, 28))
    )
    var bad = DataFrame([Series("s", Column[String](["2024-02-30"]))])
    with assert_raises(contains="cannot parse '2024-02-30' as date"):
        _ = bad.select(col("s").str().to_date())


def test_arithmetic() raises:
    var df = dates().drop_nulls()
    var ms = DataType.duration("ms")
    var gap = df.select((col("d") - col("d").min()).alias("gap"))
    assert_true(gap.dtypes()[0] == ms)
    assert_equal(
        one(gap, col("gap").dt().total_days()),
        [String("45289"), "45290", "25507", "36465", "0"],
    )
    var stamps = df.select(col("d").cast("datetime[ms]").alias("t"))
    var later = stamps.select(
        (col("t") + lit(Int64(90061000)).cast("duration[ms]")).alias("x")
    )
    assert_equal(one(later, col("x"))[0], "2024-02-29 01:01:01")
    var span = stamps.select((col("t") - col("t").shift()).alias("x"))
    assert_equal(one(span, col("x"))[1], "1d")
    assert_equal(one(span, col("x"))[2], "-19783d")
    var doubled = span.select((col("x") * lit(Int64(2))).alias("y"))
    assert_equal(one(doubled, col("y"))[1], "2d")
    assert_equal(one(span, col("x").abs())[2], "19783d")
    var total = span.select(col("x").sum().alias("t"))
    assert_true(total.dtypes()[0] == ms)
    with assert_raises(contains="is not defined for date and datetime[ms]"):
        _ = df.with_columns(col("d").cast("datetime[ms]").alias("t")).select(
            col("d") - col("t")
        )
    with assert_raises(
        contains="is not defined for datetime[ms] and duration[us]"
    ):
        _ = stamps.select(col("t") + lit(Int64(1)).cast("duration[us]"))


def test_dt_fields_and_transforms() raises:
    var df = dates().drop_nulls()
    assert_equal(
        one(df, col("d").dt().year()),
        [String("2024"), "2024", "1969", "2000", "1900"],
    )
    assert_equal(
        one(df, col("d").dt().month()), [String("2"), "2", "12", "1", "3"]
    )
    assert_equal(
        one(df, col("d").dt().weekday()), [String("3"), "4", "3", "6", "4"]
    )
    assert_equal(
        one(df, col("d").dt().ordinal_day()),
        [String("59"), "60", "365", "1", "60"],
    )
    assert_equal(one(df, col("d").dt().truncate("1mo"))[1], "2024-02-01")
    assert_equal(one(df, col("d").dt().truncate("3mo"))[2], "1969-10-01")
    assert_equal(
        one(df, col("d").dt().truncate("1w")),
        [
            String("2024-02-26"),
            "2024-02-26",
            "1969-12-29",
            "1999-12-27",
            "1900-02-26",
        ],
    )
    assert_equal(
        one(df, col("d").dt().offset_by("1y")),
        [
            String("2025-02-28"),
            "2025-02-28",
            "1970-12-31",
            "2001-01-01",
            "1901-03-01",
        ],
    )
    assert_equal(one(df, col("d").dt().offset_by("-1mo"))[3], "1999-12-01")
    assert_equal(one(df, col("d").dt().strftime("%Y/%m"))[0], "2024/02")
    var t = df.select(
        col("d").cast("datetime[us]").dt().offset_by("25h30m15s").alias("t")
    )
    assert_equal(one(t, col("t").dt().hour())[0], "1")
    assert_equal(one(t, col("t").dt().minute())[0], "30")
    assert_equal(one(t, col("t").dt().second())[0], "15")
    assert_equal(one(t, col("t").dt().date())[0], "2024-02-29")
    assert_equal(one(t, col("t").dt().time())[0], "01:30:15")
    assert_equal(one(t, col("t").dt().truncate("1h"))[2], "1970-01-01 01:00:00")
    with assert_raises(contains="date values have no time-of-day fields"):
        _ = df.select(col("d").dt().hour())
    with assert_raises(contains="finer than date"):
        _ = df.select(col("d").dt().offset_by("3h"))
    with assert_raises(contains="dt operations require"):
        _ = df.select(col("s").dt().year())


def test_casts() raises:
    var df = dates().drop_nulls()
    assert_equal(one(df, col("d").cast("int64"))[2], "-1")
    assert_equal(one(df, col("d").cast("string"))[0], "2024-02-28")
    assert_equal(
        one(df, col("d").cast("datetime[ns]"))[2], "1969-12-31 00:00:00"
    )
    var t = df.select(
        col("d").cast("datetime[ms]").dt().offset_by("1500ms").alias("t")
    )
    assert_equal(
        one(t, col("t").cast("datetime[us]"))[0], "2024-02-28 00:00:01.500000"
    )
    assert_equal(
        one(t, col("t").cast("datetime[ms]").cast("date"))[2], "1969-12-31"
    )
    assert_equal(one(t, col("t").cast("time"))[0], "00:00:01.500000000")
    var raw = DataFrame([Series("i", Column[Int64]([0, 86400]))])
    assert_equal(
        one(raw, col("i").cast("date")), [String("1970-01-01"), "2206-07-23"]
    )
    var text = DataFrame([Series("s", Column[String](["2024-01-02", "nope"]))])
    assert_equal(
        one(text, col("s").cast("date", strict=False)),
        [String("2024-01-02"), "null"],
    )
    with assert_raises(contains="cannot parse 'nope' as date"):
        _ = text.select(col("s").cast("date"))
    with assert_raises(contains="cannot cast"):
        _ = df.select(col("d").cast("float64"))


def test_frame_operations_keep_types() raises:
    var df = dates().drop_nulls().with_row_index()
    var sorted = df.sort(["d"])
    assert_equal(texts(sorted.column("d"))[0], "1900-03-01")
    assert_true(sorted.column("d").dtype() == DataType.DATE)
    var grouped = (
        df.with_columns(col("d").dt().year().alias("y"))
        .group_by("y", maintain_order=True)
        .agg([col("d").min().alias("first"), col("d").max().alias("last")])
    )
    assert_true(grouped.column("first").dtype() == DataType.DATE)
    assert_equal(texts(grouped.column("last"))[0], "2024-02-29")
    var joined = df.join(
        df.select(["d", "index"]).rename({"index": "other"}), "d"
    )
    assert_equal(joined.height(), 5)
    assert_true(joined.column("d").dtype() == DataType.DATE)
    var windows = df.with_columns(
        [col("d").shift().alias("prev"), col("d").cum_max().alias("peak")]
    )
    assert_true(windows.column("prev").dtype() == DataType.DATE)
    assert_equal(texts(windows.column("peak"))[2], "2024-02-29")
    assert_true(df.unique(["d"]).column("d").dtype() == DataType.DATE)
    var stacked = df.vstack(df)
    assert_true(stacked.column("d").dtype() == DataType.DATE)
    with assert_raises(contains="date"):
        _ = df.vstack(df.with_columns(col("d").cast("int64")))
    assert_true(
        df.filter(col("d") > col("d").min()).column("d").dtype()
        == DataType.DATE
    )


def test_ranges() raises:
    var days = date_range("2024-02-27", "2024-03-02")
    assert_equal(
        texts(days),
        [
            String("2024-02-27"),
            "2024-02-28",
            "2024-02-29",
            "2024-03-01",
            "2024-03-02",
        ],
    )
    assert_equal(
        texts(date_range("2024-01-31", "2024-05-31", "1mo")),
        [
            String("2024-01-31"),
            "2024-02-29",
            "2024-03-31",
            "2024-04-30",
            "2024-05-31",
        ],
    )
    assert_equal(len(date_range("2024-01-01", "2024-12-31", "1w")), 53)
    var stamps = datetime_range(
        "2024-01-01", "2024-01-01 02:00", "45m", unit="ms"
    )
    assert_equal(
        texts(stamps),
        [
            String("2024-01-01 00:00:00"),
            "2024-01-01 00:45:00",
            "2024-01-01 01:30:00",
        ],
    )
    assert_true(stamps.dtype() == DataType.datetime("ms"))
    assert_equal(len(date_range("2024-01-02", "2024-01-01")), 0)
    with assert_raises(contains="must be positive"):
        _ = date_range("2024-01-01", "2024-01-02", "-1d")


def test_csv_round_trip_and_inference() raises:
    var df = (
        dates()
        .drop_nulls()
        .with_columns(
            [
                col("d")
                .cast("datetime[us]")
                .dt()
                .offset_by("3h25m7s")
                .alias("t"),
                col("d").cast("datetime[ms]").dt().time().alias("clock"),
            ]
        )
        .drop(["s"])
    )
    write_csv(df, PATH)
    var schema = CsvSchema.of(df)
    assert_true(schema.field(1).dtype == DataType.datetime("us"))
    assert_true(read_csv(PATH, schema).equals(df))
    var inferred = read_csv(PATH, try_parse_dates=True)
    assert_true(inferred.dtypes()[0] == DataType.DATE)
    assert_true(inferred.dtypes()[1] == DataType.datetime("us"))
    assert_true(inferred.dtypes()[2] == DataType.TIME)
    with open(PATH, "w") as file:
        file.write("when,day\n31/12/1999 23:59,2024-060\n")
    var custom = read_csv(
        PATH,
        CsvSchema(
            [
                CsvField.datetime("when", "ms", "%d/%m/%Y %H:%M"),
                CsvField.date("day", "%Y-%j"),
            ]
        ),
    )
    assert_equal(String(custom.item(0, "when")), "1999-12-31 23:59:00")
    assert_equal(String(custom.item(0, "day")), "2024-02-29")
    with open(PATH, "w") as file:
        file.write("d\n2024-02-30\n")
    with assert_raises(
        contains="CSV record 2, field 'd': cannot parse '2024-02-30' as date"
    ):
        _ = read_csv(PATH, CsvSchema([CsvField.date("d")]))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
