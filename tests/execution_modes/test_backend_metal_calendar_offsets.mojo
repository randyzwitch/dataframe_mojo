"""Signed calendar offsets clamp days, retain time, then add checked fixed ticks."""
# contract-group: execution.backends
from dataframe_metal.accelerator.gpu.program import Program
from dataframe_metal.accelerator.gpu.runtime import (
    capabilities as _capabilities,
)
from std.testing import assert_equal, assert_true, assert_raises
from std.memory import bitcast
from dataframe import DataFrame, DataType, Series, Column, col, lit, when
from dataframe_metal.metal import MetalRuntime
from dataframe_metal.accelerator.capabilities import RowCapabilities
from dataframe_metal.accelerator.rows import lower_rows, ROW_CALENDAR_OFFSET
from tests.execution_modes.test_backend_metal import require_metal
from tests.execution_modes.test_backend_metal_empty_inputs import check
from tests.execution_modes.test_backend_metal_calendar_truncation import fixture
from tests.support.temporal_reference import civil_days


def tagged(
    dtype: DataType, values: List[Int64], valid: List[Bool]
) raises -> DataFrame:
    return DataFrame(
        [
            Series("t", Column[Int64](values.copy(), valid.copy())).with_dtype(
                dtype
            )
        ]
    )


def test_calendar_offsets_capability_signed_metadata_and_checked_boundaries() raises:
    assert_true(not RowCapabilities().calendar_offsets)
    var frame = fixture(DataType.DATE)
    var query = frame.lazy().select(col("t").dt().offset_by("-2y3mo3d"))
    var caps = _capabilities()
    caps.calendar_offsets = False
    with assert_raises(contains="fixed interval aligned"):
        _ = lower_rows(query, caps)
    var plan = lower_rows(query, _capabilities())
    assert_equal(plan.code[4], Int64(ROW_CALENDAR_OFFSET))
    assert_equal(plan.code[6], Int64(-27))
    assert_equal(bitcast[DType.int64](plan.literals[1]), Int64(-3))
    var desc = Program(plan)
    assert_equal(desc.code[4], Int64(ROW_CALENDAR_OFFSET))
    assert_true(
        "unsupported" not in MetalRuntime(device_id=999999).describe(query)
    )
    for interval in List[String](["1mo1ns", "1y1us"]):
        with assert_raises(contains="fixed interval aligned"):
            _ = lower_rows(
                frame.lazy().select(col("t").dt().offset_by(interval)),
                _capabilities(),
            )
    var time = tagged(DataType.TIME, [0], [True])
    with assert_raises(contains="Time cannot"):
        _ = lower_rows(
            time.lazy().select(col("t").dt().offset_by("1mo")), _capabilities()
        )
    var datetime = fixture(DataType.datetime("ns"))
    # Metal decides faults on the CPU, so it needs no fault-order boundary;
    # a backend without that still refuses.
    var headed = datetime.lazy().select(col("t").dt().offset_by("1mo")).head(1)
    var strict = _capabilities()
    strict.adjudicated_faults = False
    with assert_raises(contains="terminal projections without head"):
        _ = lower_rows(headed, strict)
    _ = lower_rows(headed, _capabilities())


# requires: metal
def test_native_metal_calendar_offsets_units_leap_day_clamps_mixed_intervals_and_nulls() raises:
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
            ["1mo", "-1mo", "1q", "-1y", "2y3mo", "1mo2d", "-1y3mo2d"]
        ):
            check(
                frame.lazy().select(col("t").dt().offset_by(interval)), runtime
            )
            check(
                frame.head(0).lazy().select(col("t").dt().offset_by(interval)),
                runtime,
            )
        var per_day = (
            Int64(1) if dtype.is_date() else 86400 * dtype.per_second()
        )
        var extra = Int64(0) if dtype.is_date() else Int64(123)
        var known = tagged(
            dtype,
            [
                civil_days(2000, 1, 31) * per_day + extra,
                civil_days(1900, 1, 31) * per_day + extra,
            ],
            [True, True],
        )
        var actual = (
            known.lazy()
            .select(col("t").dt().offset_by("1mo"))
            .collect(accelerator=runtime)
        )
        assert_equal(
            actual.column("t").int64()._get(0),
            civil_days(2000, 2, 29) * per_day + extra,
        )
        assert_equal(
            actual.column("t").int64()._get(1),
            civil_days(1900, 2, 28) * per_day + extra,
        )
        if not dtype.is_date():
            check(
                frame.lazy().select(col("t").dt().offset_by("1mo2d3h4m5s")),
                runtime,
            )
    var dates = fixture(DataType.DATE)
    check(
        dates.lazy()
        .with_columns(col("t").dt().offset_by("-1mo").alias("moved"))
        .group_by("moved")
        .agg(col("t").count())
        .sort("moved"),
        runtime,
    )


# requires: metal
def test_native_metal_calendar_offsets_extreme_dates_checked_datetime_reconstruction_and_masks() raises:
    var runtime = require_metal()
    var dates = tagged(
        DataType.DATE,
        [Int64.MIN, Int64.MIN + 1, Int64.MAX - 1, Int64.MAX, -719528, 0],
        [True, True, True, True, True, True],
    )
    for interval in List[String](
        [
            "1mo",
            "-1mo",
            "1y",
            "-1y",
            "749999999999999999y",
            "-749999999999999999y",
        ]
    ):
        check(dates.lazy().select(col("t").dt().offset_by(interval)), runtime)
    for dtype in List[DataType](
        [
            DataType.datetime("ms"),
            DataType.datetime("us"),
            DataType.datetime("ns"),
        ]
    ):
        var frame = tagged(dtype, [Int64.MIN, 0, Int64.MAX], [True, True, True])
        var query = frame.lazy().select(col("t").dt().offset_by("1mo"))
        with assert_raises(contains="overflow"):
            _ = query.collect(engine="cpu", streaming=False)
        with assert_raises(contains="overflow"):
            _ = query.collect(accelerator=runtime)
        check(
            frame.lazy().select(
                when(col("t") == lit(Int64(0)).cast(dtype))
                .then(col("t").dt().offset_by("1mo"))
                .otherwise(lit(Int64(0)).cast(dtype))
            ),
            runtime,
        )
        check(
            tagged(dtype, [Int64.MIN, Int64.MAX], [False, False])
            .lazy()
            .select(col("t").dt().offset_by("1mo")),
            runtime,
        )


# requires: metal
def test_native_metal_calendar_offsets_profiles_cost_budget_and_runtime_reuse() raises:
    var runtime = require_metal()
    var query = (
        fixture(DataType.DATE).lazy().select(col("t").dt().offset_by("1mo2d"))
    )
    var estimate = runtime.estimate_auto(query)
    assert_true(";calendar-offsets-v1" in estimate.family)
    assert_true(";mojo-metal-v1;ops=" in estimate.family)
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
