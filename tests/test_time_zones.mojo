"""Zone-aware datetimes (#222): the dtype, the zone database reader, DST
handling in replace_time_zone and in local-time operations, parsing,
casts, CSV and the Arrow format string. Expected values are Polars 1.44
results (the oracle's `tz` kind checks many more)."""
from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from dataframe import (
    ArrowArray,
    ArrowSchema,
    Column,
    CsvField,
    CsvSchema,
    DataFrame,
    DataType,
    Series,
    StringColumn,
    col,
    datetime_range,
    export_arrow_series,
    import_arrow_series,
    read_csv,
    to_csv_string,
)
from dataframe.arrow import _at, _leak, _read_c_string, _reclaim
from dataframe.temporal import format, parse
from dataframe.timezone import TimeZone

comptime NY = "America/New_York"
comptime PATH = "/tmp/dataframe_mojo_time_zones.csv"


def naive(texts: List[String]) raises -> DataFrame:
    var valid = List[Bool]()
    for text in texts:
        valid.append(text != "")
    return DataFrame([Series("t", StringColumn(texts, valid))]).select(
        col("t").str().to_datetime()
    )


def texts(frame: DataFrame, name: String) raises -> List[String]:
    var out = List[String]()
    var series = frame.column(name)
    for i in range(len(series)):
        out.append(String(series.get(i)))
    return out^


def test_dtype_carries_the_zone() raises:
    var ny = DataType.datetime("us", NY)
    assert_equal(ny.name(), "datetime[us, America/New_York]")
    assert_equal(ny.time_zone(), NY)
    assert_true(DataType.parse(ny.name()) == ny)
    assert_true(
        DataType.parse("datetime[ms,UTC]") == DataType.datetime("ms", "UTC")
    )
    assert_true(ny != DataType.datetime("us"))
    assert_true(ny != DataType.datetime("us", "UTC"))
    assert_true(ny != DataType.datetime("ms", NY))
    assert_equal(DataType.datetime("us", "utc").time_zone(), "UTC")
    # Arrow's fixed offsets are zones too.
    assert_equal(
        DataType.datetime("us", "+05:30").name(), "datetime[us, +05:30]"
    )
    assert_equal(DataType.datetime("us").time_zone(), "")
    # The zone survives inside nested types.
    assert_true(DataType.parse("list[" + ny.name() + "]") == DataType.list(ny))
    var nested = DataType.list(DataType.struct(["at"], [ny]))
    assert_true(nested.inner().field_dtypes()[0] == ny)
    with assert_raises(contains="unknown time zone 'Mars/Base'"):
        _ = DataType.datetime("us", "Mars/Base")
    with assert_raises(contains="invalid time zone name"):
        _ = DataType.datetime("us", "../etc/passwd")


def test_zone_offsets() raises:
    var ny = TimeZone.load(NY)
    assert_equal(ny.offset_at(1717243200), -4 * 3600)  # 2024-06-01 EDT
    assert_equal(ny.offset_at(1704067200), -5 * 3600)  # 2024-01-01 EST
    # 2045 is past the file's transitions: the footer rule applies.
    assert_equal(ny.offset_at(2383257600), -4 * 3600)  # 2045-07-10
    assert_equal(ny.offset_at(2366668800), -5 * 3600)  # 2044-12-31
    # Before 1970: 1918's first US summer time.
    assert_equal(ny.offset_at(-1627862400), -4 * 3600)  # 1918-06-01
    var lord_howe = TimeZone.load("Australia/Lord_Howe")
    assert_equal(lord_howe.offset_at(1704067200), 11 * 3600)
    assert_equal(lord_howe.offset_at(1717243200), 10 * 3600 + 1800)
    assert_equal(lord_howe.abbreviation(lord_howe.type_at(1704067200)), "+11")
    assert_equal(TimeZone.load("+05:30").offset_at(0), 19800)
    assert_equal(TimeZone.load("-03:00").offset_at(0), -10800)


def test_replace_time_zone_options() raises:
    # 02:30 does not exist on 2024-03-10 in New York; 01:30 on 2024-11-03
    # happens twice.
    var frame = naive(
        [
            "2024-03-10 01:30:00",
            "2024-03-10 02:30:00",
            "2024-11-03 01:30:00",
            "2024-06-01 12:00:00",
        ]
    )
    var earliest = frame.select(
        col("t").dt().replace_time_zone(NY, "earliest", "null")
    )
    assert_true(earliest.column("t").dtype() == DataType.datetime("us", NY))
    assert_equal(
        texts(earliest, "t"),
        [
            "2024-03-10 01:30:00-05:00",
            "null",
            "2024-11-03 01:30:00-04:00",
            "2024-06-01 12:00:00-04:00",
        ],
    )
    var latest = frame.select(
        col("t").dt().replace_time_zone(NY, "latest", "null")
    )
    assert_equal(texts(latest, "t")[2], "2024-11-03 01:30:00-05:00")
    var nulls = frame.select(
        col("t").dt().replace_time_zone(NY, "null", "null")
    )
    assert_equal(texts(nulls, "t")[2], "null")
    with assert_raises(contains="is non-existent in time zone"):
        _ = frame.select(col("t").dt().replace_time_zone(NY, "earliest"))
    with assert_raises(contains="is ambiguous in time zone"):
        _ = frame.slice(2, 2).select(col("t").dt().replace_time_zone(NY))
    with assert_raises(contains="ambiguous must be"):
        _ = frame.select(col("t").dt().replace_time_zone(NY, "first"))
    # Aware to aware keeps the wall time; to naive keeps it too.
    var london = frame.slice(3, 1).select(
        col("t")
        .dt()
        .replace_time_zone(NY)
        .dt()
        .replace_time_zone("Europe/London")
    )
    assert_equal(texts(london, "t"), ["2024-06-01 12:00:00+01:00"])
    var back = london.select(col("t").dt().replace_time_zone(""))
    assert_true(back.column("t").dtype() == DataType.datetime("us"))
    assert_equal(texts(back, "t"), ["2024-06-01 12:00:00"])


def test_local_fields_and_conversion() raises:
    var utc = naive(["2024-11-03 05:30:00", "2024-11-03 06:30:00", ""]).select(
        col("t").dt().replace_time_zone("UTC").dt().convert_time_zone(NY)
    )
    var out = utc.select_exprs(
        [
            col("t").dt().hour().alias("hour"),
            col("t").dt().day().alias("day"),
            col("t").dt().date().alias("date"),
            col("t").dt().strftime("%H:%M %z %Z").alias("text"),
            col("t").cast("string").alias("cast"),
            col("t").cast("date").alias("cast_date"),
        ]
    )
    assert_equal(texts(out, "hour"), ["1", "1", "null"])
    assert_equal(texts(out, "day"), ["3", "3", "null"])
    assert_equal(texts(out, "date"), ["2024-11-03", "2024-11-03", "null"])
    assert_equal(
        texts(out, "text"), ["01:30 -0400 EDT", "01:30 -0500 EST", "null"]
    )
    assert_equal(
        texts(out, "cast"),
        ["2024-11-03 01:30:00-04:00", "2024-11-03 01:30:00-05:00", "null"],
    )
    assert_equal(texts(out, "cast_date"), texts(out, "date"))
    # Converting keeps the instant; casting to naive keeps the UTC value.
    var stored = utc.column("t")._data[Column[Int64]]._get(0)
    var utc_again = utc.select(col("t").dt().convert_time_zone("UTC"))
    assert_equal(utc_again.column("t")._data[Column[Int64]]._get(0), stored)
    var plain = utc.select(col("t").cast("datetime[us]"))
    assert_equal(texts(plain, "t")[0], "2024-11-03 05:30:00")
    with assert_raises(contains="convert_time_zone needs a time zone"):
        _ = utc.select(col("t").dt().convert_time_zone(""))


def test_truncate_and_offset_by_follow_polars() raises:
    # Values and results from Polars 1.44.
    var t = naive(
        [
            "2024-03-09 07:30:00",  # 02:30 EST the day before the gap
            "2024-11-04 06:30:00",  # 01:30 EST the day after the overlap
            "2024-11-03 06:45:00",  # 01:45 EST, the second 01:45
            "2024-03-10 17:00:00",  # 13:00 EDT
        ]
    ).select(col("t").dt().replace_time_zone("UTC").dt().convert_time_zone(NY))
    var out = t.select_exprs(
        [
            col("t").dt().offset_by("1d").alias("plus_day"),
            col("t").dt().offset_by("-1d").alias("minus_day"),
            col("t").dt().offset_by("24h").alias("plus_24h"),
            col("t").dt().truncate("1h").alias("hour"),
            col("t").dt().truncate("1d").alias("day"),
        ]
    )
    assert_equal(
        texts(out, "plus_day"),
        [
            # Into the gap: moved by the DST amount an hour later.
            "2024-03-10 01:30:00-05:00",
            "2024-11-05 01:30:00-05:00",
            "2024-11-04 01:45:00-05:00",
            "2024-03-11 13:00:00-04:00",
        ],
    )
    assert_equal(
        texts(out, "minus_day"),
        [
            "2024-03-08 02:30:00-05:00",
            # Into the overlap: keeps standard time, like the original.
            "2024-11-03 01:30:00-05:00",
            "2024-11-02 01:45:00-04:00",
            "2024-03-09 13:00:00-05:00",
        ],
    )
    assert_equal(texts(out, "plus_24h")[3], "2024-03-11 13:00:00-04:00")
    assert_equal(texts(out, "hour")[2], "2024-11-03 01:00:00-05:00")
    assert_equal(
        texts(out, "day"),
        [
            "2024-03-09 00:00:00-05:00",
            "2024-11-04 00:00:00-05:00",
            "2024-11-03 00:00:00-04:00",
            "2024-03-10 00:00:00-05:00",
        ],
    )
    # Apia skipped 2011-12-30; a day back from the 31st has nowhere to go.
    var apia = naive(["2011-12-30 10:14:00"]).select(
        col("t")
        .dt()
        .replace_time_zone("UTC")
        .dt()
        .convert_time_zone("Pacific/Apia")
    )
    with assert_raises(contains="non-existent in time zone 'Pacific/Apia'"):
        _ = apia.select(col("t").dt().offset_by("-1d"))


def test_parsing_offsets_and_zones() raises:
    var frame = DataFrame(
        [
            Series(
                "s",
                StringColumn(
                    ["2024-03-10T01:30:00-05:00", "2024-03-10T12:00:00+01:00"]
                ),
            )
        ]
    )
    # %z makes the result UTC-aware, as in Polars.
    var parsed = frame.select(
        col("s").str().strptime("datetime[us]", "%Y-%m-%dT%H:%M:%S%z")
    )
    assert_true(parsed.column("s").dtype() == DataType.datetime("us", "UTC"))
    assert_equal(
        texts(parsed, "s"),
        ["2024-03-10 06:30:00+00:00", "2024-03-10 11:00:00+00:00"],
    )
    # A zone target reads offset-free text as local time there.
    var local = DataFrame(
        [Series("s", StringColumn(["2024-03-10 03:30:00"]))]
    ).select(col("s").str().to_datetime(time_zone=NY))
    assert_equal(texts(local, "s"), ["2024-03-10 03:30:00-04:00"])
    with assert_raises(contains="non-existent"):
        _ = DataFrame(
            [Series("s", StringColumn(["2024-03-10 02:30:00"]))]
        ).select(col("s").str().to_datetime(time_zone=NY))
    var ms = DataType.datetime("ms", "Australia/Lord_Howe")
    var value = parse("2024-01-01 00:15:00", ms)
    assert_equal(format(value, ms, "%Y-%m-%d %H:%M %Z"), "2024-01-01 00:15 +11")


def test_mismatched_zones_do_not_mix() raises:
    var frame = naive(["2024-06-01 12:00:00"]).select_exprs(
        [
            col("t").dt().replace_time_zone(NY).alias("ny"),
            col("t").dt().replace_time_zone("UTC").alias("utc"),
            col("t").alias("plain"),
        ]
    )
    with assert_raises(contains="matching dtypes"):
        _ = frame.select(col("ny") == col("utc"))
    with assert_raises(contains="matching dtypes"):
        _ = frame.select(col("ny") < col("plain"))
    with assert_raises():
        _ = frame.select(col("ny") - col("utc"))
    var left = frame.select(col("ny").alias("k"))
    var right = frame.select(col("utc").alias("k"))
    with assert_raises(contains="Join key dtypes differ"):
        _ = left.join(right, "k", "inner")
    # Same zone: differences are durations.
    var diff = frame.select((col("ny") - col("ny")).alias("d"))
    assert_true(diff.column("d").dtype() == DataType.duration("us"))


def test_datetime_range_in_a_zone() raises:
    var hours = datetime_range(
        "2024-03-10 00:00", "2024-03-10 04:00", "1h", time_zone=NY
    )
    assert_equal(len(hours), 4)
    assert_equal(String(hours.get(2)), "2024-03-10 03:00:00-04:00")
    var days = datetime_range(
        "2024-03-09 00:00", "2024-03-11 00:00", "1d", time_zone=NY
    )
    assert_equal(len(days), 3)
    assert_equal(String(days.get(2)), "2024-03-11 00:00:00-04:00")


def test_csv_reads_and_writes_zones() raises:
    with open(PATH, "w") as handle:
        handle.write(
            "a,b\n2024-03-10T01:30:00-05:00,2024-03-10T01:30:00\n"
            "2024-03-10T12:00:00Z,2024-03-10T12:00:00+01:00\n"
        )
    # All offsets: UTC-aware; mixed with naive text: string, as in Polars.
    var inferred = read_csv(PATH, try_parse_dates=True)
    assert_true(inferred.column("a").dtype() == DataType.datetime("us", "UTC"))
    assert_true(inferred.column("b").dtype() == DataType.STRING)
    # A schema zone reads offset-free text as local time there.
    var typed = read_csv(
        PATH,
        CsvSchema(
            [
                CsvField.datetime("a", time_zone=NY),
                CsvField.datetime("b", time_zone=NY),
            ]
        ),
    )
    assert_equal(
        texts(typed, "b"),
        ["2024-03-10 01:30:00-05:00", "2024-03-10 07:00:00-04:00"],
    )
    # Written with offsets, so reading back gives the same instants.
    var text = to_csv_string(typed.select(col("a")))
    assert_equal(
        text, "a\n2024-03-10 01:30:00-05:00\n2024-03-10 08:00:00-04:00\n"
    )


def test_arrow_format_carries_the_zone() raises:
    var series = (
        naive(["2024-06-01 12:00:00", ""])
        .select(col("t").dt().replace_time_zone(NY))
        .column("t")
    )
    var array = _leak(ArrowArray())
    var schema = _leak(ArrowSchema())
    export_arrow_series(
        series, _at[ArrowArray](array)[], _at[ArrowSchema](schema)[]
    )
    assert_equal(
        _read_c_string(_at[ArrowSchema](schema)[].format),
        "tsu:America/New_York",
    )
    var back = import_arrow_series(array, schema)
    _ = _reclaim[ArrowArray](array)
    _ = _reclaim[ArrowSchema](schema)
    assert_true(back.dtype() == series.dtype())
    assert_true(back.equals(series))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
