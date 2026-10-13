"""Calendar buckets preserve proleptic months and checked Datetime storage."""
from std.testing import assert_true, assert_equal, assert_raises
from dataframe import DataFrame, DataType, Series, Column, col, lit, when
from dataframe.metal import MetalRuntime, _capabilities, _Descriptors
from dataframe.accelerator.capabilities import RowCapabilities
from dataframe.accelerator.rows import lower_rows, ROW_CALENDAR_TRUNCATE
from tests.execution_modes.test_backend_metal import require_metal
from tests.execution_modes.test_backend_metal_empty_inputs import check
from tests.support.temporal_reference import civil_days


def fixture(dtype: DataType) raises -> DataFrame:
    var days: List[Int64] = [
        civil_days(2000, 2, 29),
        civil_days(1900, 3, 1),
        civil_days(1969, 12, 31),
        civil_days(1970, 1, 1),
        civil_days(1700, 1, 31),
        civil_days(2001, 12, 31),
        civil_days(2004, 2, 29),
        civil_days(2100, 3, 1),
    ]
    var values = List[Int64]()
    for day in days:
        values.append(
            day if dtype.is_date() else day * 86400 * dtype.per_second() + 123
        )
    return DataFrame(
        [
            Series(
                "t",
                Column[Int64](
                    values^, [True, True, False, True, True, True, True, True]
                ),
            ).with_dtype(dtype)
        ]
    )


def test_calendar_truncation_capability_month_metadata_and_checked_boundaries() raises:
    assert_true(not RowCapabilities().calendar_truncation)
    var frame = fixture(DataType.DATE)
    var query = frame.lazy().select(col("t").dt().truncate("2y3mo"))
    var caps = _capabilities()
    caps.calendar_truncation = False
    with assert_raises(contains="positive fixed interval"):
        _ = lower_rows(query, caps)
    var plan = lower_rows(query, _capabilities())
    assert_equal(plan.code[4], Int64(ROW_CALENDAR_TRUNCATE))
    assert_equal(plan.code[6], Int64(27))
    var desc = _Descriptors(plan)
    assert_equal(desc.code[4], Int64(144))
    assert_true(
        "unsupported" not in MetalRuntime(device_id=999999).describe(query)
    )
    for interval in List[String](["0mo", "-1mo", "1mo1d", "1y2h"]):
        with assert_raises(contains="positive fixed interval"):
            _ = lower_rows(
                frame.lazy().select(col("t").dt().truncate(interval)),
                _capabilities(),
            )
    var time = DataFrame(
        [Series("t", Column[Int64]([0])).with_dtype(DataType.TIME)]
    )
    with assert_raises(contains="Time cannot"):
        _ = lower_rows(
            time.lazy().select(col("t").dt().truncate("1mo")), _capabilities()
        )
    var datetime = fixture(DataType.datetime("ns"))
    with assert_raises(contains="terminal projections without head"):
        _ = lower_rows(
            datetime.lazy().select(col("t").dt().truncate("1mo")).head(1),
            _capabilities(),
        )


def test_native_metal_calendar_truncation_units_leap_centuries_negative_epochs_and_empty_rows() raises:
    var runtime = require_metal()
    for dtype in List[DataType](
        [
            DataType.DATE,
            DataType.datetime("ms"),
            DataType.datetime("us"),
            DataType.datetime("ns"),
        ]
    ):
        var frame = fixture(dtype).slice(1, 7)
        for interval in List[String](
            ["1mo", "3mo", "1q", "1y", "2y", "5mo", "2y3mo"]
        ):
            check(
                frame.lazy().select(col("t").dt().truncate(interval)), runtime
            )
            check(
                frame.head(0).lazy().select(col("t").dt().truncate(interval)),
                runtime,
            )
        var known = DataFrame(
            [
                Series(
                    "t",
                    Column[Int64](
                        [
                            civil_days(
                                2000, 2, 29
                            ) if dtype.is_date() else civil_days(2000, 2, 29)
                            * 86400
                            * dtype.per_second()
                        ]
                    ),
                ).with_dtype(dtype)
            ]
        )
        var answer = (
            known.lazy()
            .select(col("t").dt().truncate("1mo"))
            .collect(accelerator=runtime)
        )
        var expected = civil_days(2000, 2, 1)
        if not dtype.is_date():
            expected *= 86400 * dtype.per_second()
        assert_equal(answer.column("t").int64()._get(0), expected)
    var dates = fixture(DataType.DATE)
    check(
        dates.lazy()
        .with_columns(col("t").dt().truncate("1mo").alias("bucket"))
        .group_by("bucket")
        .agg(col("t").count())
        .sort("bucket"),
        runtime,
    )


def test_native_metal_calendar_truncation_full_date_domain_datetime_overflow_and_masks() raises:
    var runtime = require_metal()
    var dates = DataFrame(
        [
            Series(
                "t",
                Column[Int64](
                    [
                        Int64.MIN,
                        Int64.MIN + 1,
                        Int64.MAX - 1,
                        Int64.MAX,
                        -719528,
                        0,
                    ]
                ),
            ).with_dtype(DataType.DATE)
        ]
    )
    for interval in List[String](["1mo", "1y", "5mo", "999999999999999999mo"]):
        check(dates.lazy().select(col("t").dt().truncate(interval)), runtime)
    for dtype in List[DataType](
        [
            DataType.datetime("ms"),
            DataType.datetime("us"),
            DataType.datetime("ns"),
        ]
    ):
        var frame = DataFrame(
            [
                Series(
                    "t", Column[Int64]([Int64.MIN, 0, Int64.MAX])
                ).with_dtype(dtype)
            ]
        )
        var query = frame.lazy().select(col("t").dt().truncate("1mo"))
        with assert_raises(contains="overflow"):
            _ = query.collect(engine="cpu", streaming=False)
        with assert_raises(contains="overflow"):
            _ = query.collect(accelerator=runtime)
        check(
            frame.lazy().select(
                when(col("t") > lit(Int64.MIN).cast(dtype))
                .then(col("t").dt().truncate("1mo"))
                .otherwise(lit(Int64(0)).cast(dtype))
            ),
            runtime,
        )
        var nulls = DataFrame(
            [Series("t", Column[Int64]([Int64.MIN], [False])).with_dtype(dtype)]
        )
        check(nulls.lazy().select(col("t").dt().truncate("1mo")), runtime)


def test_native_metal_calendar_truncation_profiles_cost_budget_and_reuse() raises:
    var runtime = require_metal()
    var query = (
        fixture(DataType.DATE).lazy().select(col("t").dt().truncate("1y"))
    )
    var estimate = runtime.estimate_auto(query)
    assert_true(";calendar-truncation-v1" in estimate.family)
    assert_true(";metal-calendar-truncation-v1" in estimate.family)
    var exact = MetalRuntime(memory_limit_bytes=estimate.peak_bytes)
    var measured = query.profile(accelerator=exact)
    assert_equal(
        measured[1].item(0, "peak_requested_bytes").int64(),
        Int64(estimate.peak_bytes),
    )
    with assert_raises(contains="budget"):
        _ = query.collect(
            accelerator=MetalRuntime(memory_limit_bytes=estimate.peak_bytes - 1)
        )
    check(query, exact)
