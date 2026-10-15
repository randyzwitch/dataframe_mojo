"""Decimal duration sums retain precision; floating windows retain decimal scale."""
from dataframe.accelerator.gpu.program import Program
from dataframe.accelerator.gpu.runtime import capabilities as _capabilities
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import DataFrame, Series, Column, DataType, col, lit, when
from dataframe.decimal import pow10
from dataframe.metal import MetalRuntime
from dataframe.accelerator.rows import lower_rows, ROW_DURATION_KEY
from tests.execution_modes.test_backend_metal import require_metal
from tests.execution_modes.test_backend_metal_duration_rolling import (
    check,
    fails,
)
from tests.public_contracts.test_duration_decimal_windows import (
    decimal_frame,
    decimal_expressions,
    anchors,
)
from tests.support.coverage_generators import offset_layout, chunk_layout


def test_duration_decimal_private_widening_precision_float_casts_and_no_device_preflight() raises:
    for dtype in List[DataType](
        [
            DataType.decimal(6, 3, 32),
            DataType.decimal(6, 3, 64),
            DataType.decimal(6, 3),
        ]
    ):
        var source = decimal_frame(dtype)
        var query = source.lazy().select_exprs(decimal_expressions())
        var caps = _capabilities()
        caps.proven_integer_operations = False
        var plan = lower_rows(query, caps)
        var count = 0
        for step in plan.steps:
            if plan.code[step.start * 4] != ROW_DURATION_KEY:
                continue
            count += 1
            var method = plan.code[step.start * 4 + 3]
            var input = Int(plan.code[step.start * 4 + 1])
            if method == 0:
                assert_true(plan.slot_dtypes[input].is_decimal())
                assert_equal(plan.slot_dtypes[input].decimal_width(), 128)
                assert_true(plan.slot_dtypes[step.slot] == dtype.sum_type())
                assert_equal(
                    plan.code[(step.start + 1) * 4 + 3],
                    Int64(dtype.sum_type().precision()),
                )
            else:
                assert_true(plan.slot_dtypes[input] == DataType.FLOAT64)
        assert_equal(count, 4)
        _ = Program(plan)
        assert_true(
            "unsupported" not in MetalRuntime(device_id=999999).describe(query)
        )
        caps = _capabilities()
        caps.decimal128 = False
        with assert_raises(contains="Decimal128"):
            _ = lower_rows(
                source.lazy().select(col("x").rolling_sum_by("t", "3d")), caps
            )


def test_native_metal_duration_decimals_all_widths_scales_signs_layouts_partitions_and_boundaries() raises:
    var runtime = require_metal()
    for width in [32, 64, 128]:
        for precision in [6, 9, 18, 38]:
            if (width == 32 and precision > 9) or (
                width == 64 and precision > 18
            ):
                continue
            for scale in [0, 3, precision]:
                var dtype = DataType.decimal(precision, scale, width)
                for sign in [-1, 1]:
                    var source = decimal_frame(dtype, sign)
                    for layout in range(3):
                        var frame = (
                            source.copy() if layout
                            == 0 else offset_layout(source) if layout
                            == 1 else chunk_layout(source)
                        )
                        var query = frame.lazy().select_exprs(
                            decimal_expressions()
                        )
                        var actual = query.collect(accelerator=runtime)
                        anchors(actual, dtype, sign)
                        check(query, runtime)
                    var grouped = source.with_column(
                        Series(
                            "k",
                            Column[Int32](
                                [0, 1, 0, 1], [False, True, False, True]
                            ),
                        )
                    )
                    for closed in ["left", "right", "both", "none"]:
                        check(
                            grouped.lazy().select_exprs(
                                [
                                    col("x")
                                    .rolling_sum_by("t", "3d", closed, 0)
                                    .over("k")
                                    .alias("s"),
                                    col("x")
                                    .rolling_mean_by("t", "3d", closed, 0)
                                    .over("k")
                                    .alias("m"),
                                    col("x")
                                    .rolling_var_by("t", "3d", closed, 0, 0)
                                    .over("k")
                                    .alias("v"),
                                    col("x")
                                    .rolling_std_by("t", "3d", closed, 0, 0)
                                    .over("k")
                                    .alias("sd"),
                                ]
                            ),
                            runtime,
                        )
                    check(
                        source.head(0)
                        .lazy()
                        .select_exprs(decimal_expressions()),
                        runtime,
                    )


def test_native_metal_duration_decimal_observed_prefix_errors_null_payloads_and_physical_extremes() raises:
    var runtime = require_metal()
    var maximum = pow10(38) - 1
    var source = DataFrame(
        [
            Series(
                "x", Column[Int128]([maximum, maximum, -maximum])
            ).with_dtype(DataType.decimal(38, 0)),
            Series("t", Column[Int64]([0, 0, 0])).with_dtype(DataType.DATE),
            Series("k", Column[Int32]([0, 0, 0])),
        ]
    )
    fails(
        source.lazy().select(col("x").rolling_sum_by("t", "1d")),
        runtime,
        "precision",
    )
    fails(
        source.lazy().select(
            when(col("k") > 0)
            .then(col("x").rolling_sum_by("t", "1d"))
            .otherwise(lit("0").cast(DataType.decimal(38, 0)))
        ),
        runtime,
        "precision",
    )
    check(
        source.lazy().select(col("x").rolling_sum_by("t", "1d", "right", 4)),
        runtime,
    )
    var ignored = source.with_column(
        Series(
            "x",
            Column[Int128](
                [maximum, Int128.MIN, -maximum], [True, False, True]
            ),
        ).with_dtype(DataType.decimal(38, 0))
    )
    check(ignored.lazy().select(col("x").rolling_sum_by("t", "1d")), runtime)
    var nulls = source.with_column(
        Series(
            "x",
            Column[Int128]([Int128.MIN, Int128.MAX, 0], [False, False, False]),
        ).with_dtype(DataType.decimal(38, 0))
    )
    var zero = (
        nulls.lazy()
        .select(col("x").rolling_sum_by("t", "1d", "right", 0))
        .collect(accelerator=runtime)
    )
    for row in range(3):
        assert_equal(
            zero.column("x").numeric[DType.int128]().value(row), Int128(0)
        )
    var narrow = DataFrame(
        [
            Series("x", Column[Int64]([Int64.MAX, Int64.MIN])).with_dtype(
                DataType.decimal(3, 0, 64)
            ),
            Series("t", Column[Int64]([0, 0])).with_dtype(DataType.DATE),
        ]
    )
    var widened = (
        narrow.lazy()
        .select(col("x").rolling_sum_by("t", "1d"))
        .collect(accelerator=runtime)
    )
    assert_true(widened.column("x").dtype() == DataType.decimal(38, 0))
    assert_equal(
        widened.column("x").numeric[DType.int128]().value(0), Int128(-1)
    )
    var small = DataFrame(
        [
            Series("x", Column[Int128]([999, 1])).with_dtype(
                DataType.decimal(3, 0)
            ),
            Series("t", Column[Int64]([0, 0])).with_dtype(DataType.DATE),
        ]
    )
    # Decimal128 inputs also widen to precision 38, just like narrow storage.
    var small_query = small.lazy().select(col("x").rolling_sum_by("t", "1d"))
    var small_result = small_query.collect(accelerator=runtime)
    check(small_query, runtime)
    assert_true(small_result.column("x").dtype() == DataType.decimal(38, 0))
    for row in range(2):
        assert_equal(
            small_result.column("x").numeric[DType.int128]().value(row),
            Int128(1000),
        )
    # CPU validates each observed operand before adding it. Cancellation must
    # not hide a physical Int128 payload outside the result's precision.
    for negative in [False, True]:
        var payload: List[Int128] = [-maximum, Int128.MAX, -maximum]
        if negative:
            payload = [maximum, Int128.MIN, maximum]
        var malformed = source.with_column(
            Series("x", Column[Int128](payload^)).with_dtype(
                DataType.decimal(38, 0)
            )
        )
        fails(
            malformed.lazy().select(col("x").rolling_sum_by("t", "1d")),
            runtime,
            "precision",
        )
        check(
            malformed.lazy().select(
                col("x").rolling_sum_by("t", "1d", "right", 4)
            ),
            runtime,
        )
    check(
        small.lazy().select(col("x").rolling_sum_by("t", "1d", "right", 3)),
        runtime,
    )


def test_native_metal_duration_decimal_consumers_profiles_precision_families_and_exact_budgets() raises:
    var runtime = require_metal()
    var dtype = DataType.decimal(18, 3, 64)
    var frame = decimal_frame(dtype).with_column(
        Series("k", Column[Int32]([0, 1, 0, 1]))
    )
    check(
        frame.lazy()
        .with_columns(col("x").rolling_sum_by("t", "3d").alias("s"))
        .filter(col("s").cast(DataType.FLOAT64) > 2)
        .sort("s"),
        runtime,
    )
    check(
        frame.lazy()
        .with_columns(col("x").rolling_sum_by("t", "3d").alias("s"))
        .group_by("k")
        .agg(col("s").sum())
        .sort("k"),
        runtime,
    )
    check(
        frame.lazy().select(
            col("x").rolling_mean_by("t", "3d").over("k").rolling_std(2, 1, 0)
        ),
        runtime,
    )
    for dtype in List[DataType](
        [
            DataType.decimal(9, 3, 32),
            DataType.decimal(18, 3, 64),
            DataType.decimal(6, 3),
        ]
    ):
        var query = (
            decimal_frame(dtype)
            .lazy()
            .select(col("x").rolling_sum_by("t", "3d"))
        )
        var estimate = runtime.estimate_auto(query)
        assert_true(
            ":dp" + String(dtype.sum_type().precision()) in estimate.family
        )
        assert_true(";mojo-metal-v1;ops=" in estimate.family)
        var exact = MetalRuntime(memory_limit_bytes=estimate.peak_bytes)
        var measured = query.profile(accelerator=exact)
        assert_equal(
            measured[1].item(0, "peak_requested_bytes").int64(),
            Int64(estimate.peak_bytes),
        )
        with assert_raises(contains="budget"):
            _ = query.collect(
                accelerator=MetalRuntime(
                    memory_limit_bytes=estimate.peak_bytes - 1
                )
            )
        check(query, exact)
