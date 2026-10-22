"""Direct Mojo temporal fields, duration totals and checked calendar rows."""
# contract-group: execution.apple_parity
# contract-group: execution.backends

from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Expr,
    LazyFrame,
    Series,
    col,
    when,
    lit,
)
from dataframe_metal.metal import MetalRuntime
from dataframe.calendar import days_from_civil
from tests.execution_modes.test_backend_metal_float64_emulation import same
from tests.execution_modes.test_mojo_gpu_timezone_conversion import (
    _check_local_literals,
)


def _check(query: LazyFrame) raises -> DataFrame:
    var actual = query.collect(accelerator=MetalRuntime(), engine="accel")
    same(actual, query.collect(engine="cpu", streaming=False))
    return actual^


def _fields(dtype: DataType) -> List[Expr]:
    var expressions: List[Expr] = [
        col("t").dt().weekday().alias("weekday"),
        col("t").dt().ordinal_day().alias("ordinal"),
    ]
    if not dtype.is_time():
        expressions.extend(
            [
                col("t").dt().year().alias("year"),
                col("t").dt().month().alias("month"),
                col("t").dt().day().alias("day"),
                col("t")
                .dt()
                .year()
                .cast(DataType.FLOAT64)
                .alias("floating_year"),
            ]
        )
    if not dtype.is_date():
        expressions.extend(
            [
                col("t").dt().hour().alias("hour"),
                col("t").dt().minute().alias("minute"),
                col("t").dt().second().alias("second"),
                col("t").dt().nanosecond().alias("nanosecond"),
            ]
        )
    if dtype.is_datetime():
        expressions.extend(
            [
                col("t").dt().date().alias("date"),
                col("t").dt().time().alias("time"),
            ]
        )
    return expressions^


# requires: metal
def test_mojo_gpu_temporal_fields_extreme_storage_nullable_layouts_and_metadata() raises:
    var dtypes: List[DataType] = [
        DataType.DATE,
        DataType.TIME,
        DataType.datetime("ms"),
        DataType.datetime("us"),
        DataType.datetime("ns"),
    ]
    var raw: List[Int64] = [
        Int64.MIN,
        Int64.MIN + 719468,
        -719469,
        -719468,
        -1,
        0,
        1,
        10957,
        11016,
        19782,
        Int64.MAX - 719468,
        Int64.MAX,
    ]
    for dtype in dtypes:
        for chunked in [False, True]:
            var values = List[Int64]()
            var valid = List[Bool]()
            for row in range(263):
                values.append(raw[row % len(raw)])
                valid.append(row % 7 != 0)
            var source = (
                Series("t", Column[Int64](values^, valid^))
                .with_dtype(dtype)
                .slice(3, 257)
            )
            if chunked:
                source = Series._from_chunks(
                    [
                        source.slice(0, 129),
                        source.slice(129, 0),
                        source.slice(129, 128),
                    ]
                )
            var frame = DataFrame([source^])
            _ = _check(frame.lazy().select_exprs(_fields(dtype)))
            _ = _check(frame.head(0).lazy().select_exprs(_fields(dtype)))


# requires: metal
def test_mojo_gpu_temporal_calendar_offsets_truncation_and_time_wrap() raises:
    var dtypes: List[DataType] = [
        DataType.DATE,
        DataType.datetime("ms"),
        DataType.datetime("us"),
        DataType.datetime("ns"),
    ]
    for dtype in dtypes:
        var per_day = (
            Int64(1) if dtype.is_date() else dtype.per_second() * 86400
        )
        var within = Int64(0) if dtype.is_date() else per_day - 1
        var values: List[Int64] = [
            days_from_civil(2020, 1, 31) * per_day + within,
            days_from_civil(2021, 3, 31) * per_day,
            -per_day,
            0,
            per_day,
        ]
        var frame = DataFrame(
            [
                Series(
                    "t",
                    Column[Int64](
                        values.copy(), [True, True, True, True, False]
                    ),
                ).with_dtype(dtype)
            ]
        )
        var expressions: List[Expr] = [
            col("t").dt().offset_by("1mo").alias("next"),
            col("t").dt().offset_by("-1mo").alias("previous"),
            col("t").dt().offset_by("1mo2d").alias("combined"),
            col("t").dt().truncate("1mo").alias("month"),
            col("t").dt().truncate("3mo").alias("quarter"),
            col("t").dt().truncate("1y").alias("year"),
            col("t").dt().truncate("1w").alias("week"),
        ]
        if not dtype.is_date():
            expressions.append(col("t").dt().truncate("1h").alias("hour"))
        var result = _check(frame.lazy().select_exprs(expressions))
        assert_equal(result.column("next").dtype(), dtype)
        assert_equal(
            result.column("next").int64()._get(0),
            days_from_civil(2020, 2, 29) * per_day + within,
        )
        assert_equal(
            result.column("previous").int64()._get(1),
            days_from_civil(2021, 2, 28) * per_day,
        )
        assert_equal(
            result.column("month").int64()._get(0),
            days_from_civil(2020, 1, 1) * per_day,
        )
    var time = DataFrame(
        [
            Series(
                "t", Column[Int64]([0, 86399999999999, 86400000000001, -1])
            ).with_dtype(DataType.TIME)
        ]
    )
    var result = _check(time.lazy().select(col("t").dt().offset_by("1ns")))
    assert_equal(result.column("t").dtype(), DataType.TIME)
    assert_equal(result.column("t").int64()._get(1), Int64(0))
    assert_equal(result.column("t").int64()._get(3), Int64(0))
    var high = DataFrame(
        [
            Series("t", Column[Int64]([Int64.MAX])).with_dtype(
                DataType.datetime("ns")
            )
        ]
    )
    with assert_raises(contains="overflow"):
        _ = (
            high.lazy()
            .select(col("t").dt().offset_by("1ns"))
            .collect(accelerator=MetalRuntime(), engine="accel")
        )
    var low = DataFrame(
        [
            Series("t", Column[Int64]([Int64.MIN])).with_dtype(
                DataType.datetime("ns")
            )
        ]
    )
    with assert_raises(contains="overflow"):
        _ = (
            low.lazy()
            .select(col("t").dt().truncate("1mo"))
            .collect(accelerator=MetalRuntime(), engine="accel")
        )
    var masked = DataFrame(
        [
            Series(
                "t", Column[Int64]([Int64.MAX, 0], [False, True])
            ).with_dtype(DataType.datetime("ns"))
        ]
    )
    _ = _check(masked.lazy().select(col("t").dt().offset_by("1ns")))
    _ = _check(
        high.lazy().select(
            when(lit(False))
            .then(col("t").dt().offset_by("1ns"))
            .otherwise(col("t"))
        )
    )


# requires: metal
def test_mojo_gpu_duration_totals_negative_truncation_wrapping_and_rows() raises:
    var units: List[String] = ["ms", "us", "ns"]
    var targets: List[String] = [
        "days",
        "hours",
        "minutes",
        "seconds",
        "milliseconds",
        "microseconds",
        "nanoseconds",
    ]
    for unit in units:
        var dtype = DataType.duration(unit)
        var frame = DataFrame(
            [
                Series(
                    "t",
                    Column[Int64](
                        [
                            Int64.MIN,
                            -1001,
                            -1000,
                            -999,
                            -1,
                            0,
                            1,
                            999,
                            1000,
                            1001,
                            Int64.MAX,
                        ],
                        [
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            True,
                            False,
                        ],
                    ),
                ).with_dtype(dtype)
            ]
        )
        var expressions = List[Expr]()
        for target in targets:
            expressions.append(col("t").dt().total(target).alias(target))
        _ = _check(frame.lazy().select_exprs(expressions))
    var frame = DataFrame(
        [
            Series("t", Column[Int64]([-1001, -1, 0, 1001])).with_dtype(
                DataType.duration("ns")
            )
        ]
    )
    var result = _check(
        frame.lazy().select(col("t").dt().total("microseconds"))
    )
    assert_equal(result.column("t").int64()._get(0), Int64(-1))
    assert_equal(result.column("t").int64()._get(1), Int64(0))


# requires: metal
def test_mojo_gpu_utc_temporal_fields_conversions_calendar_and_checked_edges() raises:
    var units: List[String] = ["ms", "us", "ns"]
    for unit in units:
        var dtype = DataType.datetime(unit, "UTC")
        var per = dtype.per_second()
        var raw: List[Int64] = [
            Int64.MIN,
            -per - 1,
            -1,
            0,
            1,
            per + 1,
            Int64.MAX,
        ]
        var utc = DataFrame(
            [Series("t", Column[Int64](raw.copy())).with_dtype(dtype)]
        )
        _ = _check(utc.lazy().select_exprs(_fields(dtype)))
        var result = _check(
            utc.lazy().select(col("t").dt().convert_time_zone("UTC"))
        )
        assert_equal(result.column("t").dtype(), dtype)
        _ = _check(utc.lazy().select(col("t").dt().replace_time_zone("")))
        var zoned = DataFrame(
            [
                Series("t", Column[Int64](raw.copy())).with_dtype(
                    DataType.datetime(unit, "America/New_York")
                )
            ]
        )
        _ = _check(zoned.lazy().select(col("t").dt().convert_time_zone("UTC")))
        _ = _check(
            zoned.lazy().select(
                col("t").dt().convert_time_zone("UTC").dt().year()
            )
        )
        var safe: List[Int64] = [
            Int64.MIN + per,
            -per - 1,
            -1,
            0,
            1,
            per + 1,
            Int64.MAX,
        ]
        var naive = DataFrame(
            [
                Series("t", Column[Int64](safe^)).with_dtype(
                    DataType.datetime(unit)
                )
            ]
        )
        _ = _check(naive.lazy().select(col("t").dt().replace_time_zone("UTC")))
        var per_day = per * 86400
        var calendar = DataFrame(
            [
                Series(
                    "t",
                    Column[Int64](
                        [
                            days_from_civil(2020, 1, 31) * per_day + per,
                            -per_day,
                            0,
                        ]
                    ),
                ).with_dtype(dtype)
            ]
        )
        _ = _check(
            calendar.lazy().select_exprs(
                [
                    col("t").dt().offset_by("1d").alias("day"),
                    col("t").dt().offset_by("1h").alias("hour"),
                    col("t").dt().offset_by("1mo2d1h").alias("combined"),
                    col("t").dt().truncate("1mo").alias("month"),
                    col("t").dt().truncate("1w").alias("week"),
                ]
            )
        )
        var low = DataFrame(
            [
                Series("t", Column[Int64]([Int64.MIN])).with_dtype(
                    DataType.datetime(unit)
                )
            ]
        )
        with assert_raises(contains="overflow"):
            _ = (
                low.lazy()
                .select(col("t").dt().replace_time_zone("UTC"))
                .collect(accelerator=MetalRuntime(), engine="accel")
            )
        var landing = DataFrame(
            [
                Series("t", Column[Int64]([Int64.MIN + per_day])).with_dtype(
                    dtype
                )
            ]
        )
        with assert_raises(contains="overflow"):
            _ = (
                landing.lazy()
                .select(col("t").dt().offset_by("-1d"))
                .collect(accelerator=MetalRuntime(), engine="accel")
            )
        var replaced = zoned.lazy().select(
            col("t").dt().replace_time_zone("UTC")
        )
        # The original extreme storage values now reach the mutation kernel.
        # Compare any checked reconstruction diagnostic exactly with CPU.
        var expected_error = String("")
        var actual_error = String("")
        try:
            _ = replaced.collect(engine="cpu", optimize=False)
        except error:
            expected_error = String(error)
        try:
            _ = replaced.collect(accelerator=MetalRuntime(), optimize=False)
        except error:
            actual_error = String(error)
        assert_equal(actual_error, expected_error)
        if not expected_error:
            _ = _check(replaced)
        var epoch = DataFrame(
            [
                Series("t", Column[Int64]([0])).with_dtype(
                    DataType.datetime(unit, "America/New_York")
                )
            ]
        )
        _check_local_literals(
            epoch.lazy().select(col("t").dt().replace_time_zone("UTC")),
            DataType.datetime(unit, "UTC"),
            [Int64(-18000) * per],
        )
        _ = _check(
            utc.lazy().select(
                col("t").dt().convert_time_zone("America/New_York").dt().year()
            )
        )
