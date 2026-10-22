"""Calendar duration bounds clamp months and retain checked temporal semantics."""
# contract-group: execution.backends
from dataframe_metal.accelerator.gpu.program import Program
from dataframe_metal.accelerator.gpu.runtime import (
    capabilities as _capabilities,
)
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import DataFrame, DataType, Series, Column, Expr, col
from dataframe_metal.metal import MetalRuntime
from dataframe_metal.accelerator.capabilities import RowCapabilities
from dataframe_metal.accelerator.rows import (
    lower_rows,
    ROW_DURATION_KEY,
    ROW_DURATION_CALENDAR,
)
from tests.execution_modes.test_backend_metal import require_metal
from tests.execution_modes.test_backend_metal_duration_rolling import (
    expressions,
    check,
    fails,
)
from tests.support.temporal_reference import civil_days
from tests.support.coverage_generators import offset_layout, chunk_layout


def fixture(dtype: DataType) raises -> DataFrame:
    var days: List[Int64] = [
        civil_days(2000, 2, 28),
        civil_days(2000, 2, 29),
        civil_days(2000, 3, 1),
        civil_days(2000, 3, 31),
    ]
    var per_day = Int64(1) if dtype.is_date() else 86400 * dtype.per_second()
    for i in range(len(days)):
        days[i] = days[i] * per_day + (
            Int64(0) if dtype.is_date() else Int64(12345)
        )
    return DataFrame(
        [
            Series("x", Column[Int64]([1, 2, 4, 8])),
            Series("t", Column[Int64](days^)).with_dtype(dtype),
            Series(
                "k", Column[Int32]([0, 1, 0, 1], [False, True, False, True])
            ),
            Series("j", Column[Int32]([0, 0, 0, 0])),
        ]
    )


def test_calendar_duration_capability_headers_units_and_no_device_preflight() raises:
    assert_true(not RowCapabilities().calendar_duration_rolling)
    for dtype in List[DataType](
        [
            DataType.DATE,
            DataType.datetime("ms"),
            DataType.datetime("us"),
            DataType.datetime("ns"),
        ]
    ):
        var query = (
            fixture(dtype)
            .lazy()
            .select(col("x").rolling_sum_by("t", "2y3mo2d").over(["k", "j"]))
        )
        var caps = _capabilities()
        caps.calendar_duration_rolling = False
        with assert_raises(
            contains="calendar duration rolling requires backend support"
        ):
            _ = lower_rows(query, caps)
        caps = _capabilities()
        caps.max_nodes = 4
        with assert_raises(contains="metadata exceeds backend node limits"):
            _ = lower_rows(query, caps)
        var plan = lower_rows(query, _capabilities())
        var count = 0
        for step in plan.steps:
            if plan.code[4 * step.start] != ROW_DURATION_KEY:
                continue
            count += 1
            assert_equal(step.nodes, 5)
            assert_equal(
                plan.code[4 * (step.start + 2)], Int64(ROW_DURATION_CALENDAR)
            )
            assert_equal(plan.code[4 * (step.start + 2) + 1], Int64(27))
            assert_equal(
                plan.code[4 * (step.start + 2) + 2],
                Int64(
                    0 if dtype.is_date() else 1 if dtype.unit()
                    == "ms" else 2 if dtype.unit()
                    == "us" else 3
                ),
            )
        assert_equal(count, 1)
        _ = Program(plan)
        assert_true(
            "unsupported" not in MetalRuntime(device_id=999999).describe(query)
        )
    with assert_raises(contains="timezone-free"):
        _ = lower_rows(
            fixture(DataType.datetime("ns", "America/New_York"))
            .lazy()
            .select(col("x").rolling_sum_by("t", "1mo")),
            _capabilities(),
        )
    with assert_raises(contains="aligned"):
        _ = lower_rows(
            fixture(DataType.DATE)
            .lazy()
            .select(col("x").rolling_sum_by("t", "1mo1ns")),
            _capabilities(),
        )


# requires: metal
def test_native_metal_calendar_duration_leap_clamps_closed_sides_numeric_decimal_layouts_and_partitions() raises:
    var runtime = require_metal()
    for dtype in List[DataType](
        [
            DataType.DATE,
            DataType.datetime("ms"),
            DataType.datetime("us"),
            DataType.datetime("ns"),
        ]
    ):
        var source = fixture(dtype)
        var expected: List[Int64] = [6, 12, 14, 4]
        var closed: List[String] = ["left", "right", "both", "none"]
        for i in range(4):
            var actual = (
                source.lazy()
                .select(col("x").rolling_sum_by("t", "1mo", closed[i]))
                .collect(accelerator=runtime)
            )
            assert_equal(actual.column("x").int64().value(3), expected[i])
        for value_type in List[DataType](
            [
                DataType.INT64,
                DataType.FLOAT32,
                DataType.FLOAT64,
                DataType.decimal(9, 3, 32),
                DataType.decimal(18, 3, 64),
                DataType.decimal(38, 3),
            ]
        ):
            var typed = source.with_column(source.column("x").cast(value_type))
            for layout in range(3):
                var frame = (
                    typed.copy() if layout
                    == 0 else offset_layout(typed) if layout
                    == 1 else chunk_layout(typed)
                )
                for interval in List[String](["1mo", "1q", "1y", "2y3mo2d"]):
                    for side in closed:
                        check(
                            frame.lazy().select_exprs(
                                expressions(interval, side)
                            ),
                            runtime,
                        )
                    var scoped = expressions(interval, "both", 0, 0)
                    for n in range(len(scoped)):
                        scoped[n] = scoped[n].over(["k", "j"])
                    check(frame.lazy().select_exprs(scoped), runtime)
            check(
                typed.head(0).lazy().select_exprs(expressions("1mo", "both")),
                runtime,
            )


# requires: metal
def test_native_metal_calendar_duration_date_extremes_datetime_overflow_and_index_validation() raises:
    var runtime = require_metal()
    var source = DataFrame(
        [
            Series("x", Column[Float64]([1, 2, 3, 4, 5, 6])),
            Series(
                "t",
                Column[Int64](
                    [Int64.MIN, Int64.MIN + 1, -1, 0, Int64.MAX - 1, Int64.MAX]
                ),
            ).with_dtype(DataType.DATE),
        ]
    )
    for interval in List[String](["1mo", "1q", "1y", "749999999999999999y"]):
        for side in ["left", "right", "both", "none"]:
            check(
                source.lazy().select_exprs(expressions(interval, side)), runtime
            )
    for dtype in List[DataType](
        [
            DataType.datetime("ms"),
            DataType.datetime("us"),
            DataType.datetime("ns"),
        ]
    ):
        var frame = DataFrame(
            [
                Series("x", Column[Int64]([0], [False])),
                Series("t", Column[Int64]([Int64.MIN])).with_dtype(dtype),
            ]
        )
        fails(
            frame.lazy().select(
                col("x").rolling_sum_by("t", "1mo", "right", 4)
            ),
            runtime,
            "overflow",
        )
    var bad = fixture(DataType.DATE).with_column(
        Series("t", Column[Int64]([0, 2, 1, 3])).with_dtype(DataType.DATE)
    )
    fails(
        bad.lazy().select(col("x").rolling_mean_by("t", "1mo")),
        runtime,
        "not sorted",
    )
    bad = fixture(DataType.DATE).with_column(
        Series(
            "t", Column[Int64]([0, 1, 2, 3], [True, True, False, True])
        ).with_dtype(DataType.DATE)
    )
    fails(
        bad.lazy().select(col("x").rolling_mean_by("t", "1mo")),
        runtime,
        "contains nulls",
    )


# requires: metal
def test_native_metal_calendar_duration_resident_axes_consumers_profiles_and_exact_budgets() raises:
    var runtime = require_metal()
    var frame = fixture(DataType.DATE)
    check(
        frame.lazy()
        .filter(col("x") > 1)
        .with_columns(col("x").rolling_sum_by("t", "1mo").alias("s"))
        .filter(col("s") > 2)
        .sort("s"),
        runtime,
    )
    check(
        frame.lazy()
        .with_columns(col("x").rolling_mean_by("t", "1mo").alias("m"))
        .group_by("k")
        .agg(col("m").sum())
        .sort("k"),
        runtime,
    )
    check(
        frame.lazy().select(
            col("x").rolling_var_by("t", "1mo").rolling_std(2, 1, 0)
        ),
        runtime,
    )
    var query = frame.lazy().select_exprs(expressions("1mo2d", "both", 2, 0))
    var estimate = runtime.estimate_auto(query)
    assert_true(":months1:unit0" in estimate.family)
    assert_true(
        estimate.family
        != runtime.estimate_auto(
            frame.lazy().select_exprs(expressions("2mo2d", "both", 2, 0))
        ).family
    )
    var exact = MetalRuntime(memory_limit_bytes=estimate.peak_bytes)
    var profiled = query.profile(accelerator=exact)
    assert_equal(
        profiled[1].item(0, "peak_requested_bytes").int64(),
        Int64(estimate.peak_bytes),
    )
    with assert_raises(contains="budget"):
        _ = query.collect(
            accelerator=MetalRuntime(memory_limit_bytes=estimate.peak_bytes - 1)
        )
    check(query, exact)
