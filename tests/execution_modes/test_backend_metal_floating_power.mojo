"""Compensated floating power preserves exceptional values and boundary accuracy."""
# contract-group: execution.backends
from dataframe_metal.accelerator.gpu.program import Program
from dataframe_metal.accelerator.gpu.runtime import (
    capabilities as _capabilities,
)
from std.math import isnan, isfinite
from std.memory import bitcast
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import DataFrame, Series, Column, LazyFrame, col, lit, when
from dataframe.dtype import DataType
from dataframe_metal.metal import MetalRuntime
from dataframe_metal.accelerator.capabilities import RowCapabilities
from dataframe_metal.accelerator.rows import lower_rows
from tests.execution_modes.test_backend_metal import require_metal
from tests.public_contracts.test_arithmetic import power_anchors
from tests.execution_modes.test_backend_metal_interpolation import fixture


def check(query: LazyFrame, runtime: MetalRuntime) raises:
    var actual = query.collect(accelerator=runtime)
    var expected = query.collect(engine="cpu", streaming=False)
    assert_equal(actual.columns(), expected.columns())
    assert_equal(actual.height(), expected.height())
    for name in expected.columns():
        var dtype = expected.column(name).dtype()
        assert_true(actual.column(name).dtype() == dtype)
        if not dtype.is_float():
            assert_true(actual.column(name).equals(expected.column(name)))
            continue
        var a = actual.column(name).cast(DataType.FLOAT64).float64()
        var b = expected.column(name).cast(DataType.FLOAT64).float64()
        for row in range(actual.height()):
            assert_equal(a.is_valid(row), b.is_valid(row))
            if not b.is_valid(row):
                continue
            var x = a._get(row)
            var y = b._get(row)
            if isnan(y):
                assert_true(isnan(x))
            elif not isfinite(y):
                assert_equal(x, y)
            elif x == 0 and y == 0:
                assert_equal(bitcast[DType.uint64](x), bitcast[DType.uint64](y))
            else:
                # Vendor libm can differ by one minimum subnormal at finite
                # underflow boundaries; independent domain anchors below remain exact.
                var relative = Float64(
                    1e-5
                ) if dtype == DataType.FLOAT32 else Float64(5e-12)
                var absolute = Float64(
                    1e-44
                ) if dtype == DataType.FLOAT32 else Float64(1e-323)
                assert_true(
                    isfinite(x) and abs(x - y) <= relative * abs(y) + absolute
                )


def test_floating_power_capability_metadata_types_and_cost_preflight() raises:
    assert_true(not RowCapabilities().float_power)
    var frame = DataFrame([Series("x", Column[Float32]([1, 2]))])
    var caps = _capabilities()
    caps.float_power = False
    with assert_raises(contains="power require CPU"):
        _ = lower_rows(frame.lazy().select(col("x").pow(lit(Float32(2)))), caps)
    for dtype in [DataType.FLOAT32, DataType.FLOAT64]:
        var source = frame.with_columns(col("x").cast(dtype))
        var exponent = lit(Float32(2)) if dtype == DataType.FLOAT32 else lit(
            Float64(2)
        )
        var query = source.lazy().select(col("x").pow(exponent))
        var plan = lower_rows(query, _capabilities())
        _ = Program(plan)
        assert_true(
            "unsupported" not in MetalRuntime(device_id=999999).describe(query)
        )


def anchor[D: DType](runtime: MetalRuntime) raises:
    var frame = power_anchors[D]()
    var actual = (
        frame[0]
        .lazy()
        .select(col("x").pow(col("y")).alias("p"))
        .collect(accelerator=runtime)
    )
    var expected = frame[1].numeric[D]()
    var values = actual.column("p").numeric[D]()
    for row in range(actual.height()):
        assert_equal(values.is_valid(row), expected.is_valid(row))
        if not expected.is_valid(row):
            continue
        var x = Float64(values._get(row))
        var y = Float64(expected._get(row))
        if isnan(y):
            assert_true(isnan(x))
        elif not isfinite(y) or y == 0:
            assert_equal(bitcast[DType.uint64](x), bitcast[DType.uint64](y))
        else:
            assert_true(
                abs(x - y) <= (1e-5 if D == DType.float32 else 5e-12) * abs(y)
            )
    check(frame[0].lazy().select(col("x").pow(col("y"))), runtime)
    check(frame[0].head(0).lazy().select(col("x").pow(col("y"))), runtime)
    var x = Series._from_chunks(
        [frame[0].column("x").slice(0, 5), frame[0].column("x").slice(5, 14)]
    )
    var y = Series._from_chunks(
        [frame[0].column("y").slice(0, 7), frame[0].column("y").slice(7, 12)]
    )
    check(DataFrame([x^, y^]).lazy().select(col("x").pow(col("y"))), runtime)


# requires: metal
def test_native_metal_floating_power_independent_ieee_anchors_widths_nulls_chunks_and_empty() raises:
    var runtime = require_metal()
    anchor[DType.float32](runtime)
    anchor[DType.float64](runtime)
    var x = DataFrame(
        [Series("x", Column[Float64]([1.01, 1.1, 3, 10, 1e-300, 1e300]))]
    )
    for exponent in [-64.5, -3.0, -0.5, 0.5, 3.0, 64.5]:
        check(x.lazy().select(col("x").pow(lit(exponent))), runtime)
        check(x.lazy().select(lit(exponent).pow(col("x"))), runtime)
    var near_one = DataFrame(
        [Series("x", Column[Float64]([1.0000000000000002, 0.9999999999999999]))]
    )
    for exponent in [-3e17, 3e17]:
        check(near_one.lazy().select(col("x").pow(lit(exponent))), runtime)


# requires: metal
def test_native_metal_floating_power_resident_filters_reductions_windows_and_conditional_scopes() raises:
    var runtime = require_metal()
    var frame = fixture().with_columns(col("x").cast(DataType.FLOAT64))
    check(
        frame.lazy()
        .with_columns(col("x").pow(lit(Float64(0.5))).alias("p"))
        .filter(col("p") > 3)
        .sort("p"),
        runtime,
    )
    check(
        frame.lazy()
        .with_columns(col("x").pow(lit(Float64(0.5))).alias("p"))
        .group_by("k")
        .agg(col("p").mean())
        .sort("k"),
        runtime,
    )
    check(
        frame.lazy().select(col("x").pow(lit(Float64(0.5))).rolling_mean(3, 1)),
        runtime,
    )
    check(
        frame.lazy().select(col("x").pow(lit(Float64(0.5))).mean().over("k")),
        runtime,
    )
    check(
        frame.lazy().select(
            when(col("x") > 4)
            .then(col("x").pow(lit(Float64(-0.5))))
            .otherwise(lit(Float64(0)))
        ),
        runtime,
    )
    check(
        frame.head(0).lazy().select(lit(Float64(1)).pow(lit(Float64("nan")))),
        runtime,
    )


# requires: metal
def test_native_metal_floating_power_profile_family_exact_budget_and_runtime_reuse() raises:
    var runtime = require_metal()
    for dtype in [DataType.FLOAT32, DataType.FLOAT64]:
        var frame = fixture().with_columns(col("x").cast(dtype))
        var exponent = lit(Float32(2)) if dtype == DataType.FLOAT32 else lit(
            Float64(2)
        )
        var query = frame.lazy().select(col("x").pow(exponent))
        var estimate = runtime.estimate_auto(query)
        assert_true(";floating-power-v1:p1" in estimate.family)
        assert_true(";mojo-metal-v1;ops=" in estimate.family)
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
