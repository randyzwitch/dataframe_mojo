"""UTC duration windows retain separate calendar and fixed-time stages."""
# contract-group: execution.backends
from dataframe_metal.accelerator.gpu.program import Program
from dataframe_metal.accelerator.gpu.runtime import (
    capabilities as _capabilities,
)
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import DataFrame, DataType, Series, Column, col
from dataframe_metal.metal import MetalRuntime
from dataframe_metal.accelerator.capabilities import RowCapabilities
from dataframe_metal.accelerator.rows import (
    lower_rows,
    ROW_DURATION_KEY,
    ROW_DURATION_UTC,
)
from tests.execution_modes.test_backend_metal import require_metal
from tests.execution_modes.test_backend_metal_calendar_rolling import fixture
from tests.execution_modes.test_backend_metal_duration_rolling import (
    expressions,
    check,
    fails,
)
from tests.support.coverage_generators import offset_layout, chunk_layout


def test_utc_duration_capability_staged_metadata_wide_days_and_no_device_preflight() raises:
    assert_true(not RowCapabilities().utc_duration_rolling)
    for unit in ["ms", "us", "ns"]:
        for zone in ["UTC", "+00:00", "-00:00"]:
            var source = fixture(DataType.datetime(unit, zone))
            var query = source.lazy().select(
                col("x").rolling_sum_by("t", "1mo200000d3h")
            )
            var caps = _capabilities()
            caps.utc_duration_rolling = False
            with assert_raises(
                contains="UTC duration rolling requires backend support"
            ):
                _ = lower_rows(query, caps)
            caps = _capabilities()
            caps.max_nodes = 3
            with assert_raises(contains="metadata exceeds backend node limits"):
                _ = lower_rows(query, caps)
            var plan = lower_rows(query, _capabilities())
            var count = 0
            for step in plan.steps:
                if plan.code[4 * step.start] != ROW_DURATION_KEY:
                    continue
                count += 1
                assert_equal(step.nodes, 4)
                assert_equal(
                    plan.code[4 * (step.start + 3)], Int64(ROW_DURATION_UTC)
                )
                assert_equal(plan.code[4 * (step.start + 3) + 1], Int64(200000))
                assert_equal(
                    plan.code[4 * (step.start + 3) + 2],
                    Int64(1 if unit == "ms" else 2 if unit == "us" else 3),
                )
            assert_equal(count, 1)
            _ = Program(plan)
            assert_true(
                "unsupported"
                not in MetalRuntime(device_id=999999).describe(query)
            )
    with assert_raises(contains="timezone-free or fixed-offset"):
        _ = lower_rows(
            fixture(DataType.datetime("ns", "America/New_York"))
            .lazy()
            .select(col("x").rolling_sum_by("t", "1d")),
            _capabilities(),
        )


# requires: metal
def test_native_metal_utc_duration_units_zero_offset_zones_calendar_days_fixed_ticks_decimals_and_layouts() raises:
    var runtime = require_metal()
    for unit in ["ms", "us", "ns"]:
        for zone in ["UTC", "+00:00", "-00:00"]:
            var source = fixture(DataType.datetime(unit, zone))
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
                var typed = source.with_column(
                    source.column("x").cast(value_type)
                )
                for layout in range(3):
                    var frame = (
                        typed.copy() if layout
                        == 0 else offset_layout(typed) if layout
                        == 1 else chunk_layout(typed)
                    )
                    for interval in [
                        "1d",
                        "1mo",
                        "1mo2d3h",
                        "5ns" if unit
                        == "ns" else "5us" if unit
                        == "us" else "5ms",
                    ]:
                        check(
                            frame.lazy().select_exprs(
                                expressions(interval, "both")
                            ),
                            runtime,
                        )
                    var scoped = expressions("1mo2d3h", "right", 0, 0)
                    for n in range(len(scoped)):
                        scoped[n] = scoped[n].over(["k", "j"])
                    check(frame.lazy().select_exprs(scoped), runtime)
                check(
                    typed.head(0)
                    .lazy()
                    .select_exprs(expressions("999999999999999999d", "both")),
                    runtime,
                )
            for side in ["left", "right", "both", "none"]:
                check(
                    source.lazy().select_exprs(expressions("1mo", side)),
                    runtime,
                )
            var actual = (
                source.lazy()
                .select(col("x").rolling_sum_by("t", "1mo"))
                .collect(accelerator=runtime)
            )
            assert_equal(actual.column("x").int64().value(3), Int64(12))
            if unit != "ns":
                for interval in ["200000d", "1mo200000d3h"]:
                    check(
                        source.lazy().select_exprs(
                            expressions(interval, "both")
                        ),
                        runtime,
                    )
                    var wide = (
                        source.lazy()
                        .select(col("x").rolling_sum_by("t", interval))
                        .collect(accelerator=runtime)
                    )
                    assert_equal(wide.column("x").int64().value(3), Int64(15))


# requires: metal
def test_native_metal_utc_duration_full_width_ticks_checked_day_stages_and_index_observations() raises:
    var runtime = require_metal()
    for unit in ["ms", "us", "ns"]:
        var dtype = DataType.datetime(unit, "UTC")
        var per_day = 86400 * dtype.per_second()
        var source = DataFrame(
            [
                Series("x", Column[Int64]([1, 2, 3, 4])),
                Series(
                    "t",
                    Column[Int64](
                        [
                            -(Int64(1) << 53) - 1,
                            -(Int64(1) << 53),
                            Int64.MAX - 1,
                            Int64.MAX,
                        ]
                    ),
                ).with_dtype(dtype),
            ]
        )
        check(
            source.lazy().select_exprs(
                expressions(
                    "1ms" if unit == "ms" else "1us" if unit == "us" else "1ns",
                    "both",
                )
            ),
            runtime,
        )
        for interval in ["1d", "1mo", "999999999999999999d"]:
            var bad = DataFrame(
                [
                    Series("x", Column[Int64]([0], [False])),
                    Series("t", Column[Int64]([Int64.MIN])).with_dtype(dtype),
                ]
            )
            fails(
                bad.lazy().select(
                    col("x").rolling_sum_by("t", interval, "right", 4)
                ),
                runtime,
                "overflow",
            )
        # The CPU relocalizes each calendar-day stage separately. Its checked
        # whole-second base can underflow even if the final ticks fit Int64.
        var edge = DataFrame(
            [
                Series("x", Column[Int64]([0], [False])),
                Series("t", Column[Int64]([Int64.MIN + per_day])).with_dtype(
                    dtype
                ),
            ]
        )
        fails(
            edge.lazy().select(col("x").rolling_mean_by("t", "1d")),
            runtime,
            "overflow",
        )
        if unit == "ns":
            fails(
                fixture(dtype)
                .lazy()
                .select(col("x").rolling_mean_by("t", "200000d")),
                runtime,
                "overflow",
            )
        var null_index = fixture(dtype).with_column(
            Series(
                "t", Column[Int64]([0, 1, 2, 3], [True, True, False, True])
            ).with_dtype(dtype)
        )
        fails(
            null_index.lazy().select(
                col("x").rolling_mean_by("t", "999999999999999999d")
            ),
            runtime,
            "contains nulls",
        )
        var unsorted = fixture(dtype).with_column(
            Series("t", Column[Int64]([0, 2, 1, 3])).with_dtype(dtype)
        )
        fails(
            unsorted.lazy().select(
                col("x").rolling_mean_by("t", "999999999999999999d")
            ),
            runtime,
            "not sorted",
        )


# requires: metal
def test_native_metal_utc_duration_resident_consumers_distinct_stage_families_and_exact_budgets() raises:
    var runtime = require_metal()
    var frame = fixture(DataType.datetime("us", "UTC"))
    check(
        frame.lazy()
        .filter(col("x") > 1)
        .with_columns(col("x").rolling_sum_by("t", "1mo2d").alias("s"))
        .filter(col("s") > 2)
        .sort("s"),
        runtime,
    )
    check(
        frame.lazy()
        .with_columns(col("x").rolling_mean_by("t", "1mo2d").alias("m"))
        .group_by("k")
        .agg(col("m").sum())
        .sort("k"),
        runtime,
    )
    var query = frame.lazy().select_exprs(
        expressions("1mo200000d3h", "both", 2, 0)
    )
    var estimate = runtime.estimate_auto(query)
    assert_true(":utc2:days200000" in estimate.family)
    assert_true(
        estimate.family
        != runtime.estimate_auto(
            frame.lazy().select_exprs(expressions("1mo199999d3h", "both", 2, 0))
        ).family
    )
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
