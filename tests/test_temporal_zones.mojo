"""ISO zone designators, and format directives with no separator between them.

Both shapes are common in real data and both used to raise: a trailing "Z"
or "+01:00" hit the end-of-input check, and "%Y%m%d" let %Y take six digits.
"""
from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from dataframe import (
    CsvField,
    CsvSchema,
    DataFrame,
    DataType,
    Series,
    read_csv,
)
from dataframe.temporal import parse as parse_temporal


def one(body: String, var field: CsvField, name: String) raises -> DataFrame:
    var path = String("/tmp/claude-1000/tz_") + name + ".csv"
    with open(path, "w") as handle:
        handle.write(body)
    return read_csv(path, CsvSchema([field^]))


def test_utc_designator_is_accepted() raises:
    var naive = parse_temporal("2024-02-28T12:34:56", DataType.datetime("us"))
    for text in ["2024-02-28T12:34:56Z", "2024-02-28T12:34:56z"]:
        assert_equal(
            parse_temporal(text, DataType.datetime("us")),
            naive,
            "Z must mean UTC, which is what a naive datetime already holds",
        )


def test_offsets_convert_to_utc() raises:
    var us = DataType.datetime("us")
    var noon = parse_temporal("2024-02-28T12:00:00", us)
    var hour = Int64(3_600_000_000)
    # East of UTC is earlier in UTC; west is later.
    assert_equal(parse_temporal("2024-02-28T12:00:00+01:00", us), noon - hour)
    assert_equal(parse_temporal("2024-02-28T12:00:00-01:00", us), noon + hour)
    # Minutes count, and the colon is optional; "+HH" alone is legal.
    assert_equal(
        parse_temporal("2024-02-28T12:00:00+0130", us),
        noon - hour - hour // 2,
    )
    assert_equal(
        parse_temporal("2024-02-28T12:00:00+01:30", us),
        noon - hour - hour // 2,
    )
    assert_equal(parse_temporal("2024-02-28T12:00:00+01", us), noon - hour)
    assert_equal(parse_temporal("2024-02-28T12:00:00+00:00", us), noon)


def test_offset_crossing_a_day_boundary() raises:
    var us = DataType.datetime("us")
    # 00:30+01:00 is the previous day in UTC.
    assert_equal(
        parse_temporal("2024-03-01T00:30:00+01:00", us),
        parse_temporal("2024-02-29T23:30:00", us),
    )
    # 23:30-01:00 is the next day in UTC.
    assert_equal(
        parse_temporal("2024-02-28T23:30:00-01:00", us),
        parse_temporal("2024-02-29T00:30:00", us),
    )


def test_offsets_scale_with_the_unit() raises:
    for unit in ["ms", "us", "ns"]:
        var dtype = DataType.datetime(unit)
        var noon = parse_temporal("2024-02-28T12:00:00", dtype)
        var shifted = parse_temporal("2024-02-28T12:00:00+01:00", dtype)
        assert_equal(noon - shifted, 3600 * dtype.per_second())


def test_bad_offsets_raise() raises:
    var us = DataType.datetime("us")
    for text in [
        "2024-02-28T12:00:00+24:00",
        "2024-02-28T12:00:00+01:60",
        "2024-02-28T12:00:00+1:00",
        "2024-02-28T12:00:00Q",
        "2024-02-28T12:00:00+01:00x",
    ]:
        with assert_raises():
            _ = parse_temporal(text, us)


def test_dates_and_times_reject_designators() raises:
    # A zone on a plain date has no meaning here, and TIME stays naive.
    with assert_raises():
        _ = parse_temporal("2024-02-28Z", DataType.DATE)
    with assert_raises():
        _ = parse_temporal("12:00:00Z", DataType.TIME)


def test_packed_format_directives() raises:
    var date = DataType.DATE
    assert_equal(
        parse_temporal("20240228", date, "%Y%m%d"),
        parse_temporal("2024-02-28", date),
    )
    assert_equal(
        parse_temporal("2024-0228", date, "%Y-%m%d"),
        parse_temporal("2024-02-28", date),
    )
    var us = DataType.datetime("us")
    assert_equal(
        parse_temporal("20240228123456", us, "%Y%m%d%H%M%S"),
        parse_temporal("2024-02-28T12:34:56", us),
    )
    # A separator still allows one-digit fields, as before.
    assert_equal(
        parse_temporal("2024-2-8", date, "%Y-%m-%d"),
        parse_temporal("2024-02-08", date),
    )
    # A packed field that is too short must fail rather than borrow digits.
    with assert_raises():
        _ = parse_temporal("2024228", date, "%Y%m%d")


def test_csv_reads_both_shapes() raises:
    var frame = one(
        "t\n2024-02-28T12:34:56Z\n2024-02-28T12:34:56+01:00\n\n",
        CsvField.datetime("t"),
        "zones",
    )
    assert_equal(frame.height(), 3)
    var us = DataType.datetime("us")
    assert_equal(
        frame.item(0, "t").to_physical(),
        parse_temporal("2024-02-28T12:34:56", us),
    )
    assert_equal(
        frame.item(1, "t").to_physical(),
        parse_temporal("2024-02-28T11:34:56", us),
    )
    assert_true(frame.item(2, "t").is_null())

    var packed = one(
        "d\n20240228\n20240229\n", CsvField.date("d", "%Y%m%d"), "packed"
    )
    assert_equal(packed.height(), 2)
    assert_equal(
        packed.item(0, "d").to_physical(),
        parse_temporal("2024-02-28", DataType.DATE),
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
