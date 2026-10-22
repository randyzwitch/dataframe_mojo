"""Native binary64 order statistics preserve quantile interpolation semantics."""
# contract-group: execution.backends
from dataframe_metal.accelerator.gpu.program import Program
from dataframe_metal.accelerator.gpu.runtime import (
    capabilities as _capabilities,
)
from std.memory import bitcast
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import DataFrame, Series, Column, Expr, LazyFrame, col, lit
from dataframe.dtype import DataType, NUMERIC_DTYPES
from dataframe_metal.metal import MetalRuntime
from dataframe_metal.accelerator.capabilities import RowCapabilities
from dataframe_metal.accelerator.rows import lower_rows, ROW_QUANTILE_LINEAR
from tests.execution_modes.test_backend_metal import require_metal
from tests.execution_modes.test_backend_metal_moments import near
from tests.execution_modes.test_backend_metal_decimal_casts import (
    fixture as decimal_fixture,
)


def expressions(q: Float64) -> List[Expr]:
    return [
        col("x").median().alias("median"),
        col("x").quantile(q, "linear").alias("linear"),
        col("x").quantile(q, "lower").alias("lower"),
        col("x").quantile(q, "higher").alias("higher"),
        col("x").quantile(q, "nearest").alias("nearest"),
        col("x").quantile(q, "midpoint").alias("midpoint"),
    ]


def check(query: LazyFrame, runtime: MetalRuntime) raises:
    var actual = query.collect(accelerator=runtime)
    var expected = query.collect(engine="cpu", streaming=False)
    try:
        near(actual, expected)
    except error:
        print("QUANTILE MISMATCH", query.explain())
        for name in actual.columns():
            if actual.column(name).dtype() != DataType.FLOAT64:
                continue
            var a = actual.column(name).float64()
            var b = expected.column(name).float64()
            for row in range(actual.height()):
                print(
                    name,
                    row,
                    a.is_valid(row),
                    a._get(row),
                    b.is_valid(row),
                    b._get(row),
                )
        raise Error(String(error))


def test_quantile_capabilities_private_casts_full_probability_metadata_and_binding() raises:
    assert_true(not RowCapabilities().quantiles)
    var frame = DataFrame([Series("x", Column[UInt64]([0, UInt64.MAX]))])
    var caps = _capabilities()
    caps.quantiles = False
    with assert_raises(contains="quantiles require backend support"):
        _ = lower_rows(frame.lazy().select(col("x").median()), caps)
    caps = _capabilities()
    caps.float64 = False
    caps.float64_emulated = False
    with assert_raises(contains="Float64 arithmetic"):
        _ = lower_rows(frame.lazy().select(col("x").median()), caps)
    var probabilities: List[Float64] = [0, -0.0, 0.12345678901234567, 0.5, 1]
    for q in probabilities:
        var query = frame.lazy().select_exprs(expressions(q))
        var plan = lower_rows(query, _capabilities())
        var methods = List[Int]([0, 0, 1, 2, 3, 4])
        for i in range(len(plan.outputs)):
            assert_equal(
                plan.outputs[i].reduction, ROW_QUANTILE_LINEAR + methods[i]
            )
            assert_equal(
                plan.slot_dtypes[plan.outputs[i].slot], DataType.FLOAT64
            )
            assert_equal(
                bitcast[DType.float64](Int64(plan.outputs[i].min_count)),
                Float64(0.5) if i == 0 else q,
            )
        _ = Program(plan)
        assert_true(
            "unsupported" not in MetalRuntime(device_id=999999).describe(query)
        )
    _ = lower_rows(
        decimal_fixture(3, 2).lazy().select(col("wide").median()),
        _capabilities(),
    )
    var invalid_probabilities: List[Float64] = [
        -0.1,
        1.1,
        Float64("nan"),
        Float64("inf"),
    ]
    for q in invalid_probabilities:
        with assert_raises():
            _ = lower_rows(
                frame.lazy().select(col("x").quantile(q)), _capabilities()
            )
    with assert_raises():
        _ = lower_rows(
            frame.lazy().select(col("x").quantile(0.5, "invalid")),
            _capabilities(),
        )


# requires: metal
def test_native_metal_quantiles_numeric_widths_nulls_groups_and_partitions() raises:
    _ = require_metal()
    var runtime = MetalRuntime()
    comptime for D in NUMERIC_DTYPES:
        var values = List[Scalar[D]]()
        var valid = List[Bool]()
        var keys = List[Int32]()
        var key_valid = List[Bool]()
        for i in range(263):
            values.append(Scalar[D](i % 11))
            valid.append(i % 5 != 0)
            keys.append(Int32(i % 3))
            key_valid.append(i % 7 != 0)
        var frame = DataFrame(
            [
                Series("x", Column[Scalar[D]](values^, valid^)),
                Series("k", Column[Int32](keys^, key_valid^)),
            ]
        ).slice(3, 257)
        var probabilities: List[Float64] = [0, 0.12345678901234567, 0.5, 1]
        for q in probabilities:
            check(frame.lazy().select_exprs(expressions(q)), runtime)
            check(
                frame.lazy()
                .group_by("k", maintain_order=True)
                .agg(expressions(q)),
                runtime,
            )
            var partitioned = List[Expr]()
            for expression in expressions(q):
                partitioned.append(expression.over("k"))
            check(frame.lazy().select_exprs(partitioned), runtime)
        for rows in List[Int]([0, 1, 2, 3, 4]):
            check(
                frame.head(rows).lazy().select_exprs(expressions(0.5)), runtime
            )
    var actual = (
        DataFrame([Series("x", Column[Int32]([0, 10, 20, 30]))])
        .lazy()
        .select_exprs(expressions(0.25))
        .collect(accelerator=runtime)
    )
    near(
        actual,
        DataFrame(
            [
                Series("median", Column[Float64]([15])),
                Series("linear", Column[Float64]([7.5])),
                Series("lower", Column[Float64]([0])),
                Series("higher", Column[Float64]([10])),
                Series("nearest", Column[Float64]([10])),
                Series("midpoint", Column[Float64]([5])),
            ]
        ),
    )


# requires: metal
def test_native_metal_quantiles_nonfinite_unsigned_extremes_decimal_casts_and_consumers() raises:
    _ = require_metal()
    var runtime = MetalRuntime()
    var cases = List[List[Float64]]()
    cases.append([Float64("nan"), Float64("nan"), Float64("nan")])
    cases.append([Float64("-inf"), 1, Float64("inf"), Float64("nan")])
    cases.append([Float64("-inf"), Float64("inf")])
    cases.append([Float64("1e308"), Float64("1.7e308")])
    cases.append([Float64("-1e308"), Float64("1e308")])
    cases.append([-0.0, 0.0, Float64("-5e-324"), Float64("5e-324")])
    cases.append([7, 7, 7, 7, 7])
    for values in cases:
        var frame = DataFrame([Series("x", Column[Float64](values.copy()))])
        var probabilities: List[Float64] = [0, -0.0, 0.25, 0.5, 0.75, 1]
        for q in probabilities:
            check(frame.lazy().select_exprs(expressions(q)), runtime)
            check(
                frame.lazy()
                .with_columns(lit(Int32(1)).alias("k"))
                .group_by("k")
                .agg(expressions(q)),
                runtime,
            )
    var unsigned = DataFrame(
        [Series("x", Column[UInt64]([0, 1, UInt64.MAX - 1, UInt64.MAX]))]
    )
    check(unsigned.lazy().select_exprs(expressions(0.75)), runtime)
    for width in List[Int]([3, 6, 12]):
        var frame = (
            decimal_fixture(width, 2)
            .lazy()
            .with_columns(col("wide").cast(DataType.FLOAT64).alias("x"))
        )
        check(frame.select_exprs(expressions(0.37)), runtime)
        for name in List[String](["wide", "large", "small"]):
            var direct = (
                decimal_fixture(width, 2)
                .lazy()
                .with_columns(col(name).alias("x"))
            )
            check(direct.select_exprs(expressions(0.37)), runtime)
    var frame = DataFrame(
        [
            Series("x", Column[Int32]([9, 1, 7, 3, 5])),
            Series("k", Column[Int32]([0, 1, 0, 1, 0])),
        ]
    )
    check(
        frame.lazy()
        .filter(col("x") > 2)
        .head(3)
        .select_exprs(expressions(0.37)),
        runtime,
    )
    check(
        frame.lazy()
        .with_columns(col("x").median().over("k").alias("m"))
        .filter(col("m") > Float64(4)),
        runtime,
    )
    check(
        frame.lazy().select((col("x").quantile(0.37) + Float64(1)).alias("q")),
        runtime,
    )


# requires: metal
def test_native_metal_quantiles_profiles_cost_budget_and_runtime_reuse() raises:
    _ = require_metal()
    var runtime = MetalRuntime()
    var frame = DataFrame([Series("x", Column[Int32]([9, 1, 7, 3, 5]))])
    var query = frame.lazy().select_exprs(expressions(0.37))
    var estimate = runtime.estimate_auto(query)
    assert_true("quantiles-v1" in estimate.family)
    assert_true(";mojo-metal-v1;ops=" in estimate.family)
    var profiled = query.profile(accelerator=runtime)
    near(profiled[0], query.collect(engine="cpu", streaming=False))
    assert_true(profiled[1].height() > 0)
    check(frame.head(0).lazy().select_exprs(expressions(0.5)), runtime)
    check(query, runtime)
    with assert_raises(contains="budget"):
        _ = query.collect(accelerator=MetalRuntime(memory_limit_bytes=1))
    check(query, runtime)
