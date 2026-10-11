"""Exact native integer temporal fields across storage units and boundaries."""
from std.testing import TestSuite, assert_true, assert_raises
from dataframe import DataFrame, Series, Column, Expr, col, lit
from dataframe.dtype import DataType
from dataframe.metal import MetalRuntime, metal_installed


def temporal_frame(dtype: DataType) raises -> DataFrame:
    var values = List[Int64](
        [
            Int64.MIN,
            Int64.MIN + 719467,
            Int64.MIN + 719468,
            -719469,
            -719468,
            -719467,
            -25509,
            -1,
            0,
            1,
            59,
            60,
            10957,
            11016,
            11017,
            19782,
            2932896,
            86400000000000 - 1,
            86400000000000,
            Int64.MAX - 719468,
            Int64.MAX - 719467,
            Int64.MAX,
        ]
    )
    var valid = List[Bool](length=len(values), fill=True)
    valid[3] = False
    return DataFrame(
        [Series("t", Column[Int64](values^, valid)).with_dtype(dtype)]
    )


def check_fields(dtype: DataType) raises:
    var frame = temporal_frame(dtype)
    var fields = List[Expr](
        [
            col("t").dt().weekday().alias("weekday"),
            col("t").dt().ordinal_day().alias("ordinal"),
        ]
    )
    if not dtype.is_time():
        fields.extend(
            [
                col("t").dt().year().alias("year"),
                col("t").dt().month().alias("month"),
                col("t").dt().day().alias("day"),
            ]
        )
    if not dtype.is_date():
        fields.extend(
            [
                col("t").dt().hour().alias("hour"),
                col("t").dt().minute().alias("minute"),
                col("t").dt().second().alias("second"),
                col("t").dt().nanosecond().alias("nanos"),
            ]
        )
    if dtype.is_datetime():
        fields.extend(
            [
                col("t").dt().date().alias("date"),
                col("t").dt().time().alias("time"),
            ]
        )
    var runtime = MetalRuntime()
    var query = frame.lazy().select_exprs(fields)
    assert_true(
        query.collect(accelerator=runtime).equals(query.collect(engine="cpu"))
    )
    var empty = frame.slice(0, 0).lazy().select_exprs(fields)
    assert_true(
        empty.collect(accelerator=runtime).equals(empty.collect(engine="cpu"))
    )
    var filtered = (
        frame.lazy().filter(col("t").dt().weekday() == 4).select_exprs(fields)
    )
    assert_true(
        filtered.collect(accelerator=runtime).equals(
            filtered.collect(engine="cpu")
        )
    )


def test_temporal_storage_units_and_extremes() raises:
    if not metal_installed():
        return
    check_fields(DataType.DATE)
    check_fields(DataType.TIME)
    for unit in List[String](["ms", "us", "ns"]):
        check_fields(DataType.datetime(unit))


def test_temporal_fields_group_join_and_preflight() raises:
    if not metal_installed():
        return
    var frame = temporal_frame(DataType.DATE)
    var runtime = MetalRuntime()
    var query = (
        frame.lazy()
        .with_columns(col("t").dt().year().alias("year"))
        .group_by("year", maintain_order=True)
        .agg(col("t").len().alias("count"))
    )
    assert_true(
        query.collect(accelerator=runtime).equals(query.collect(engine="cpu"))
    )
    var joined = (
        frame.lazy()
        .with_columns(col("t").dt().year().alias("year"))
        .join(
            frame.lazy().with_columns(col("t").dt().year().alias("year")),
            "year",
        )
        .select_exprs(
            [
                col("t").dt().month().alias("month"),
                col("t_right").dt().day().alias("day"),
            ]
        )
    )
    assert_true(
        joined.collect(accelerator=runtime).equals(joined.collect(engine="cpu"))
    )
    var aware = (
        temporal_frame(DataType.datetime("us", "UTC"))
        .lazy()
        .select(col("t").dt().year())
    )
    with assert_raises(contains="time zone field extraction"):
        _ = aware.collect(accelerator=MetalRuntime(device_id=999999))
    var literal = lit("2000-02-29").str().strptime("date")
    var parsed = (
        frame.lazy()
        .filter(col("t") == literal)
        .select(col("t").dt().ordinal_day())
    )
    assert_true(
        parsed.collect(accelerator=runtime).equals(parsed.collect(engine="cpu"))
    )


def test_temporal_fixed_truncation() raises:
    if not metal_installed():
        return
    var runtime = MetalRuntime()
    var dtypes = List[DataType](
        [
            DataType.DATE,
            DataType.TIME,
            DataType.datetime("ms"),
            DataType.datetime("us"),
            DataType.datetime("ns"),
        ]
    )
    for dtype in dtypes:
        var frame = temporal_frame(dtype)
        var periods = List[String](["1d", "3d", "1w", "2w"])
        if not dtype.is_date():
            periods.extend(["1s", "15m", "3h"])
        for period in periods:
            var query = frame.lazy().select(col("t").dt().truncate(period))
            assert_true(
                query.collect(accelerator=runtime).equals(
                    query.collect(engine="cpu")
                )
            )
            var empty = (
                frame.slice(0, 0).lazy().select(col("t").dt().truncate(period))
            )
            assert_true(
                empty.collect(accelerator=runtime).equals(
                    empty.collect(engine="cpu")
                )
            )
    var frame = temporal_frame(DataType.DATE)
    var monthly = frame.lazy().select(col("t").dt().truncate("1mo"))
    with assert_raises(contains="positive fixed interval"):
        _ = monthly.collect(accelerator=MetalRuntime(device_id=999999))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
