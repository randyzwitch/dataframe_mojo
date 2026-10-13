"""Fixed-offset rolling converts each calendar stage between local and UTC ticks."""
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import DataFrame, DataType, Series, Column, col
from dataframe.metal import MetalRuntime, _capabilities, _Descriptors
from dataframe.accelerator.capabilities import RowCapabilities
from dataframe.accelerator.rows import (
    lower_rows,
    ROW_DURATION_KEY,
    ROW_DURATION_UTC,
)
from dataframe.timezone import _fixed_offset
from tests.execution_modes.test_backend_metal import require_metal
from tests.execution_modes.test_backend_metal_calendar_rolling import fixture
from tests.execution_modes.test_backend_metal_duration_rolling import (
    expressions,
    check,
    fails,
)
from tests.support.coverage_generators import offset_layout, chunk_layout


def local_fixture(dtype: DataType) raises -> DataFrame:
    var frame = fixture(dtype)
    var offset = _fixed_offset(dtype.time_zone()).value() * dtype.per_second()
    var times = List[Int64]()
    for row in range(frame.height()):
        times.append(frame.column("t").int64().value(row) - offset)
    return frame.with_column(
        Series("t", Column[Int64](times^)).with_dtype(dtype)
    )


def test_fixed_zone_duration_capability_signed_offsets_no_device_and_independent_utc_gate() raises:
    assert_true(not RowCapabilities().fixed_zone_duration_rolling)
    for unit in ["ms", "us", "ns"]:
        for zone in ["-23:59", "-03:30", "+05:45", "+14:00", "+23:59"]:
            var query = (
                local_fixture(DataType.datetime(unit, zone))
                .lazy()
                .select(col("x").rolling_sum_by("t", "1mo2d3h"))
            )
            var caps = _capabilities()
            caps.fixed_zone_duration_rolling = False
            with assert_raises(
                contains="fixed-offset duration rolling requires backend support"
            ):
                _ = lower_rows(query, caps)
            caps = _capabilities()
            caps.utc_duration_rolling = False
            var plan = lower_rows(query, caps)
            var count = 0
            for step in plan.steps:
                if plan.code[4 * step.start] != ROW_DURATION_KEY:
                    continue
                count += 1
                assert_equal(
                    plan.code[4 * (step.start + 3)], Int64(ROW_DURATION_UTC)
                )
                assert_equal(
                    plan.code[4 * (step.start + 3) + 3],
                    _fixed_offset(zone).value(),
                )
            assert_equal(count, 1)
            _ = _Descriptors(plan)
            assert_true(
                "unsupported"
                not in MetalRuntime(device_id=999999).describe(query)
            )


def test_native_metal_fixed_zone_duration_literal_local_month_clamps_units_offsets_types_layouts_and_partitions() raises:
    var runtime = require_metal()
    for unit in ["ms", "us", "ns"]:
        for zone in ["-23:59", "-03:30", "+05:45", "+14:00", "+23:59"]:
            var source = local_fixture(DataType.datetime(unit, zone))
            var sides: List[String] = ["left", "right", "both", "none"]
            var sums: List[Int64] = [6, 12, 14, 4]
            for i in range(4):
                var actual = (
                    source.lazy()
                    .select(col("x").rolling_sum_by("t", "1mo", sides[i]))
                    .collect(accelerator=runtime)
                )
                assert_equal(actual.column("x").int64().value(3), sums[i])
            for layout in range(3):
                var frame = (
                    source.copy() if layout
                    == 0 else offset_layout(source) if layout
                    == 1 else chunk_layout(source)
                )
                for interval in ["1d", "1mo2d3h"]:
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
            if unit != "ns":
                check(
                    source.lazy().select_exprs(
                        expressions("1mo200000d3h", "both")
                    ),
                    runtime,
                )
                var wide = (
                    source.lazy()
                    .select(col("x").rolling_sum_by("t", "1mo200000d3h"))
                    .collect(accelerator=runtime)
                )
                assert_equal(wide.column("x").int64().value(3), Int64(15))
            check(
                source.head(0)
                .lazy()
                .select_exprs(expressions("999999999999999999d", "both")),
                runtime,
            )
        var source = local_fixture(DataType.datetime(unit, "+05:45"))
        for dtype in List[DataType](
            [
                DataType.FLOAT32,
                DataType.FLOAT64,
                DataType.decimal(9, 3, 32),
                DataType.decimal(18, 3, 64),
                DataType.decimal(38, 3),
            ]
        ):
            var typed = source.with_column(source.column("x").cast(dtype))
            for layout in range(3):
                var frame = (
                    typed.copy() if layout
                    == 0 else offset_layout(typed) if layout
                    == 1 else chunk_layout(typed)
                )
                check(
                    frame.lazy().select_exprs(expressions("1mo2d3h", "both")),
                    runtime,
                )


def test_native_metal_fixed_zone_duration_full_width_local_wrapping_checked_reconstruction_and_pure_fixed_ticks() raises:
    var runtime = require_metal()
    for unit in ["ms", "us", "ns"]:
        for zone in ["-23:59", "+05:45", "+23:59"]:
            var dtype = DataType.datetime(unit, zone)
            var source = DataFrame(
                [
                    Series("x", Column[Int64]([1, 2, 3, 4])),
                    Series(
                        "t",
                        Column[Int64](
                            [Int64.MIN + 1, -1, Int64.MAX - 1, Int64.MAX]
                        ),
                    ).with_dtype(dtype),
                ]
            )
            var tick = (
                "1ms" if unit == "ms" else "1us" if unit == "us" else "1ns"
            )
            # Fixed units never localize, even when adding the offset to a
            # terminal source timestamp would wrap the Int64 local ticks.
            check(
                source.lazy().select_exprs(expressions(tick, "both")), runtime
            )
            var known = (
                source.lazy()
                .select(col("x").rolling_sum_by("t", tick))
                .collect(accelerator=runtime)
            )
            assert_equal(known.column("x").int64().value(0), Int64(1))
            for time in [Int64.MIN, Int64.MAX]:
                var bad = DataFrame(
                    [
                        Series("x", Column[Int64]([0], [False])),
                        Series("t", Column[Int64]([time])).with_dtype(dtype),
                    ]
                )
                # Calendar localization either returns an exact bound or
                # raises; compare both outcomes without hiding GPU errors.
                for interval in ["1d", "1mo", "1mo1d"]:
                    var query = bad.lazy().select(
                        col("x").rolling_sum_by("t", interval, "right", 4)
                    )
                    var cpu_error = False
                    try:
                        _ = query.collect(engine="cpu", streaming=False)
                    except:
                        cpu_error = True
                    if cpu_error:
                        fails(query, runtime, "overflow")
                    else:
                        check(query, runtime)
            if unit == "ns":
                fails(
                    local_fixture(dtype)
                    .lazy()
                    .select(col("x").rolling_mean_by("t", "200000d")),
                    runtime,
                    "overflow",
                )


def test_native_metal_fixed_zone_duration_resident_consumers_offset_families_and_exact_budgets() raises:
    var runtime = require_metal()
    var frame = local_fixture(DataType.datetime("us", "-03:30"))
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
    assert_true(":utc2:days200000:offset-12600" in estimate.family)
    var other = (
        local_fixture(DataType.datetime("us", "+05:45"))
        .lazy()
        .select_exprs(expressions("1mo200000d3h", "both", 2, 0))
    )
    assert_true(estimate.family != runtime.estimate_auto(other).family)
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
