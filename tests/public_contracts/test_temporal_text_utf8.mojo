"""Independent literals for temporal percent escapes and UTF8 pattern bytes."""
from std.testing import assert_equal
from dataframe import Column, DataFrame, DataType, Series, col
from dataframe.temporal import format, parse


def test_temporal_format_utf8_and_percent_literal_roundtrip() raises:
    var dtype = DataType.DATE
    var pattern = "日期 %Y/%m/%d %%"
    var expected = "日期 1970/01/01 %"
    assert_equal(format(0, dtype, pattern), expected)
    assert_equal(parse(expected, dtype, pattern), Int64(0))
    assert_equal(format(0, dtype, "été %Y %% %Q"), "été 1970 % %Q")
    assert_equal(format(0, dtype, "%é"), "%é")
    assert_equal(format(0, dtype, "%🕰"), "%🕰")
    assert_equal(format(0, dtype, "%%%%"), "%%")
    assert_equal(format(0, dtype, "%%Y %Y%%m fin%"), "%Y 1970%m fin%")
    assert_equal(format(0, dtype, "é🕰日期"), "é🕰日期")


def test_temporal_strftime_utf8_literals_preserve_nulls_and_sliced_rows() raises:
    var source = DataFrame(
        [
            Series(
                "d", Column[Int64]([0, 1, -1], List[Bool]([True, False, True]))
            ).with_dtype(DataType.DATE)
        ]
    )
    var actual = (
        source.lazy()
        .select(col("d").dt().strftime("日期 %Y-%m-%d %% %Q").alias("text"))
        .collect(engine="cpu")
    )
    assert_equal(actual.height(), 3)
    assert_equal(actual.item(0, "text").string(), "日期 1970-01-01 % %Q")
    assert_equal(actual.item(1, "text").is_null(), True)
    assert_equal(actual.item(2, "text").string(), "日期 1969-12-31 % %Q")
    var sliced = source.slice(1, 2).select(
        col("d").dt().strftime("été %Y %%").alias("text")
    )
    assert_equal(sliced.height(), 2)
    assert_equal(sliced.item(0, "text").is_null(), True)
    assert_equal(sliced.item(1, "text").string(), "été 1969 %")


def test_temporal_format_utc_directives_and_unknown_directives_remain_literal() raises:
    assert_equal(
        format(
            -1,
            DataType.datetime("us", "UTC"),
            "%Y-%m-%d %H:%M:%S%f %z %:z %Z %% %Q",
        ),
        "1969-12-31 23:59:59.999999 +0000 +00:00 UTC % %Q",
    )
    assert_equal(format(0, DataType.DATE, "%z|%:z|%Z"), "||")
