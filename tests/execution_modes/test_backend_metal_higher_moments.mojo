"""Third and fourth central moments through native Metal reductions."""
# contract-group: execution.backends
from dataframe_metal.accelerator.gpu.program import Program
from dataframe_metal.accelerator.gpu.runtime import (
    capabilities as _capabilities,
)
from std.math import sqrt
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import DataFrame, Series, Column, Expr, LazyFrame, col, lit
from dataframe.dtype import DataType, NUMERIC_DTYPES
from dataframe_metal.metal import MetalRuntime
from dataframe_metal.accelerator.capabilities import RowCapabilities
from dataframe_metal.accelerator.rows import lower_rows
from tests.support.lifecycle_jobs import WorkerSetting
from tests.execution_modes.test_backend_metal import require_metal
from tests.execution_modes.test_backend_metal_moments import near
from tests.execution_modes.test_backend_metal_decimal_casts import (
    fixture as decimal_fixture,
)


def expressions(name: String) -> List[Expr]:
    return [
        col(name).skew().alias("s"),
        col(name).skew(False).alias("su"),
        col(name).kurtosis().alias("kf"),
        col(name).kurtosis(False).alias("kp"),
        col(name).kurtosis(True, False).alias("ku"),
        col(name).kurtosis(False, False).alias("kpu"),
    ]


def check(query: LazyFrame, runtime: MetalRuntime) raises:
    near(
        query.collect(accelerator=runtime),
        query.collect(engine="cpu", streaming=False),
    )


def test_higher_moments_capability_flags_private_casts_and_precision() raises:
    assert_true(not RowCapabilities().higher_moments)
    var setting = WorkerSetting(8)
    var frame = DataFrame([Series("x", Column[UInt64]([0, UInt64.MAX]))])
    var query = frame.lazy().select_exprs(expressions("x"))
    var caps = _capabilities()
    caps.higher_moments = False
    with assert_raises(
        contains="skewness and kurtosis require backend support"
    ):
        _ = lower_rows(query, caps)
    caps = _capabilities()
    caps.float64 = False
    caps.float64_emulated = False
    with assert_raises(contains="Float64 arithmetic"):
        _ = lower_rows(query, caps)
    var plan = lower_rows(query, _capabilities())
    var flags = List[Int]([1, 0, 3, 1, 2, 0])
    for i in range(len(plan.outputs)):
        ref output = plan.outputs[i]
        assert_equal(plan.slot_dtypes[output.slot], DataType.FLOAT64)
        assert_equal(output.min_count & 0x7FFFFFFF, flags[i])
        assert_equal(output.min_count >> 47, 8)
        assert_equal((output.min_count >> 31) & 65535, 0)
    _ = Program(plan)
    assert_true(
        "unsupported" not in MetalRuntime(device_id=999999).describe(query)
    )
    _ = setting^


# requires: metal
def test_native_metal_higher_moments_numeric_widths_flags_nulls_groups_and_partitions() raises:
    _ = require_metal()
    var setting = WorkerSetting(1)
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
        check(frame.lazy().select_exprs(expressions("x")), runtime)
        check(
            frame.lazy()
            .group_by("k", maintain_order=True)
            .agg(expressions("x")),
            runtime,
        )
        var partitioned = List[Expr]()
        for expression in expressions("x"):
            partitioned.append(expression.over("k"))
        check(frame.lazy().select_exprs(partitioned), runtime)
        for rows in List[Int]([0, 1, 2, 3, 4]):
            check(
                frame.head(rows).lazy().select_exprs(expressions("x")), runtime
            )
    var frame = DataFrame([Series("x", Column[Int64]([1, 2, 3, 4]))])
    var actual = (
        frame.lazy().select_exprs(expressions("x")).collect(accelerator=runtime)
    )
    near(
        actual,
        DataFrame(
            [
                Series("s", Column[Float64]([0])),
                Series("su", Column[Float64]([0])),
                Series("kf", Column[Float64]([-1.36])),
                Series("kp", Column[Float64]([1.64])),
                Series("ku", Column[Float64]([-1.2])),
                Series("kpu", Column[Float64]([1.8])),
            ]
        ),
    )
    var asymmetric = DataFrame([Series("x", Column[Int32]([1, 1, 1, 2]))])
    var skew = (
        asymmetric.lazy().select(col("x").skew()).collect(accelerator=runtime)
    )
    near(
        skew,
        DataFrame(
            [Series("x", Column[Float64]([Float64(2) / sqrt(Float64(3))]))]
        ),
    )
    _ = setting^


# requires: metal
def test_native_metal_higher_moments_nonfinite_constants_full_unsigned_and_decimal_casts() raises:
    _ = require_metal()
    var runtime = MetalRuntime()
    var cases = List[List[Float64]]()
    cases.append([4, 4, 4, 4, 4])
    cases.append([1, Float64("nan"), 2, 3])
    cases.append([1, Float64("inf"), 2, 3])
    cases.append([Float64("-inf"), 1, Float64("inf"), 3])
    cases.append([Float64("1e300"), Float64("-1e300"), 0, 1])
    cases.append([Float64("1e-300"), Float64("-1e-300"), 0, 1])
    for values in cases:
        var frame = DataFrame([Series("x", Column[Float64](values.copy()))])
        check(frame.lazy().select_exprs(expressions("x")), runtime)
        var nulls = DataFrame(
            [
                Series(
                    "x",
                    Column[Float64](
                        values.copy(),
                        List[Bool](length=len(values), fill=False),
                    ),
                )
            ]
        )
        check(nulls.lazy().select_exprs(expressions("x")), runtime)
    var unsigned = DataFrame(
        [Series("x", Column[UInt64]([0, UInt64.MAX, UInt64.MAX - 1, 1]))]
    )
    check(unsigned.lazy().select_exprs(expressions("x")), runtime)
    var decimals = (
        decimal_fixture(3, 2)
        .lazy()
        .with_columns(col("wide").cast(DataType.FLOAT64).alias("x"))
    )
    check(decimals.select_exprs(expressions("x")), runtime)


# requires: metal
def test_native_metal_higher_moments_workers_compositions_cost_budget_and_runtime_reuse() raises:
    _ = require_metal()
    var setting = WorkerSetting(8)
    var values = List[Float64]()
    var keys = List[Int32]()
    var valid = List[Bool]()
    for i in range(131073):
        values.append(Float64(1000000) + Float64((i * 17) % 101) / 8)
        keys.append(Int32(i % 5))
        valid.append(i % 13 != 0)
    var frame = DataFrame(
        [
            Series("x", Column[Float64](values^, valid^)),
            Series("k", Column[Int32](keys^)),
        ]
    )
    var runtime = MetalRuntime()
    var query = frame.lazy().select_exprs(expressions("x"))
    check(query, runtime)
    check(
        frame.lazy().group_by("k", maintain_order=True).agg(expressions("x")),
        runtime,
    )
    check(
        frame.lazy().select(col("x").kurtosis(False, False).over("k")), runtime
    )
    check(
        frame.lazy()
        .filter(col("x") > lit(Float64(1000005)))
        .select_exprs(expressions("x")),
        runtime,
    )
    check(
        frame.lazy().select(
            (col("x").skew(False) + lit(Float64(2))).alias("s")
        ),
        runtime,
    )
    check(
        frame.lazy()
        .group_by("k", maintain_order=True)
        .agg((col("x").kurtosis() * lit(Float64(2))).alias("kurt")),
        runtime,
    )
    var estimate = runtime.estimate_auto(query)
    assert_true(";higher-moments-v1" in estimate.family)
    assert_true(";mojo-metal-v1;ops=" in estimate.family)
    assert_true(estimate.peak_bytes > 0)
    with assert_raises(contains="budget"):
        _ = query.collect(accelerator=MetalRuntime(memory_limit_bytes=1))
    check(query, runtime)
    _ = setting^
