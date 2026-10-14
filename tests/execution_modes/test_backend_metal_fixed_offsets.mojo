"""Fixed temporal offsets preserve checked storage addition and Time wrapping."""
from dataframe.accelerator.gpu.program import Program
from dataframe.accelerator.gpu.runtime import capabilities as _capabilities
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    DataFrame,
    DataType,
    Series,
    Column,
    Expr,
    LazyFrame,
    col,
    lit,
    when,
)
from dataframe.accelerator.capabilities import RowCapabilities
from dataframe.accelerator.rows import lower_rows
from dataframe.metal import MetalRuntime
from tests.execution_modes.test_backend_metal import require_metal
from tests.execution_modes.test_backend_metal_empty_inputs import check


def source(
    dtype: DataType, values: List[Int64], valid: List[Bool]
) raises -> DataFrame:
    return DataFrame(
        [
            Series("t", Column[Int64](values.copy(), valid.copy())).with_dtype(
                dtype
            )
        ]
    )


def test_fixed_offset_capability_units_descriptors_and_observation_boundaries() raises:
    assert_true(not RowCapabilities().temporal_offsets)
    var frame = source(DataType.DATE, [0, 1, -1], [True, True, False])
    var query = frame.lazy().select(col("t").dt().offset_by("-2d"))
    var caps = _capabilities()
    caps.temporal_offsets = False
    with assert_raises(contains="resident expression operation"):
        _ = lower_rows(query, caps)
    var plan = lower_rows(query, _capabilities())
    assert_equal(plan.code[4], Int64(142))
    assert_equal(plan.code[6], Int64(-2))
    assert_equal(plan.code[7], Int64(0))
    _ = Program(plan)
    assert_true(
        "unsupported" not in MetalRuntime(device_id=999999).describe(query)
    )
    for interval in List[String](["1mo1ns", "1y1ns", "1ns"]):
        with assert_raises(contains="fixed interval aligned"):
            _ = lower_rows(
                frame.lazy().select(col("t").dt().offset_by(interval)),
                _capabilities(),
            )
    with assert_raises(contains="terminal projections without head"):
        _ = lower_rows(query.head(1), _capabilities())
    with assert_raises(contains="terminal projections without head"):
        _ = lower_rows(
            frame.lazy().filter(
                col("t").dt().offset_by("1d")
                > lit(Int64(0)).cast(DataType.DATE)
            ),
            _capabilities(),
        )
    var aware = source(DataType.datetime("us", "UTC"), [0], [True])
    var utc_query = aware.lazy().select(col("t").dt().offset_by("1h"))
    var without_utc = _capabilities()
    without_utc.utc_temporal_conversions = False
    with assert_raises(contains="time zone"):
        _ = lower_rows(utc_query, without_utc)
    _ = Program(lower_rows(utc_query, _capabilities()))


def test_native_metal_fixed_offsets_all_temporal_units_nulls_negative_epochs_and_empty_rows() raises:
    var runtime = require_metal()
    var types: List[DataType] = [
        DataType.DATE,
        DataType.datetime("ms"),
        DataType.datetime("us"),
        DataType.datetime("ns"),
        DataType.TIME,
    ]
    for dtype in types:
        var values: List[Int64] = [
            123,
            -123,
            -86400000000000,
            -1,
            0,
            1,
            86400000000000,
            123456789012345,
        ]
        var frame = source(
            dtype, values^, [True, True, False, True, True, True, True, True]
        ).slice(1, 7)
        var intervals: List[String] = ["-2d", "1d", "-1w", "2d3h"]
        if dtype.is_date():
            intervals = ["-2d", "1d", "-1w", "2d"]
        for interval in intervals:
            var query = frame.lazy().select(col("t").dt().offset_by(interval))
            check(query, runtime)
            check(
                frame.head(0).lazy().select(col("t").dt().offset_by(interval)),
                runtime,
            )
        if not dtype.is_date():
            var tick = (
                "1ms" if dtype.per_second()
                == 1000 else "1us" if dtype.per_second()
                == 1000000 else "1ns"
            )
            check(frame.lazy().select(col("t").dt().offset_by(tick)), runtime)
        var offset = "1d"
        var actual = (
            source(dtype, [0], [True])
            .lazy()
            .select(col("t").dt().offset_by(offset))
            .collect(accelerator=runtime)
        )
        var expected = (
            Int64(1) if dtype.is_date() else Int64(
                0
            ) if dtype.is_time() else Int64(86400)
            * dtype.per_second()
        )
        assert_equal(actual.column("t").int64()._get(0), expected)


def test_native_metal_fixed_offset_checked_endpoints_and_conditional_masks() raises:
    var runtime = require_metal()
    var types: List[DataType] = [
        DataType.DATE,
        DataType.datetime("ms"),
        DataType.datetime("us"),
        DataType.datetime("ns"),
        DataType.TIME,
    ]
    for dtype in types:
        var tick = (
            "1d" if dtype.is_date() else "1ms" if dtype.per_second()
            == 1000 else "1us" if dtype.per_second()
            == 1000000 else "1ns"
        )
        var negative = "-" + tick
        var upper = (
            source(dtype, [Int64.MAX], [True])
            .lazy()
            .select(col("t").dt().offset_by(tick))
        )
        var lower = (
            source(dtype, [Int64.MIN], [True])
            .lazy()
            .select(col("t").dt().offset_by(negative))
        )
        for query in List[LazyFrame]([upper^, lower^]):
            with assert_raises(contains="overflow"):
                _ = query.collect(engine="cpu", streaming=False)
            with assert_raises(contains="overflow"):
                _ = query.collect(accelerator=runtime)
        check(
            source(dtype, [Int64.MIN, Int64.MAX - 1], [True, True])
            .lazy()
            .select(col("t").dt().offset_by(tick)),
            runtime,
        )
        check(
            source(dtype, [Int64.MAX, Int64.MIN], [False, False])
            .lazy()
            .select(col("t").dt().offset_by(tick)),
            runtime,
        )
        var frame = source(dtype, [Int64.MAX, Int64.MIN, 0], [True, True, True])
        check(
            frame.lazy().select(
                when(col("t") < lit(Int64.MAX).cast(dtype))
                .then(col("t").dt().offset_by(tick))
                .otherwise(lit(Int64(0)).cast(dtype))
            ),
            runtime,
        )


def test_native_metal_fixed_offset_profiles_cost_budget_and_reuse() raises:
    var runtime = require_metal()
    var query = (
        source(DataType.DATE, [0, -1, 1], [True, True, False])
        .lazy()
        .select(col("t").dt().offset_by("1d"))
    )
    var estimate = runtime.estimate_auto(query)
    assert_true(";fixed-offsets-v1" in estimate.family)
    assert_true(";mojo-metal-v1;ops=" in estimate.family)
    var profiled = query.profile(accelerator=runtime)
    assert_equal(profiled[0].height(), 3)
    assert_true(profiled[1].height() > 0)
    with assert_raises(contains="budget"):
        _ = query.collect(accelerator=MetalRuntime(memory_limit_bytes=1))
    check(query, runtime)
