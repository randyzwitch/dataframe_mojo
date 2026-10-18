"""Sliding binary64 moments preserve CPU bounds, recovery, and partitions."""
# contract-group: execution.backends
from dataframe.accelerator.gpu.program import Program
from dataframe.accelerator.gpu.runtime import capabilities as _capabilities
from std.math import sqrt, isnan
from std.memory import bitcast
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import DataFrame, Series, Column, Expr, LazyFrame, col, lit
from dataframe.dtype import DataType, NUMERIC_DTYPES
from dataframe.metal import MetalRuntime
from dataframe.accelerator.capabilities import RowCapabilities
from dataframe.accelerator.rows import lower_rows, ROW_CUMULATIVE_KEY
from dataframe.accelerator.windows import lower_partition_aggregates
from tests.support.metal_outcomes import same_outcome
from tests.execution_modes.test_backend_metal import require_metal
from tests.execution_modes.test_backend_metal_moments import near
from tests.execution_modes.test_backend_metal_decimal_casts import (
    fixture as decimal_fixture,
)
from tests.support.lifecycle_jobs import WorkerSetting


def check(query: LazyFrame, runtime: MetalRuntime) raises:
    var actual = query.collect(accelerator=runtime)
    var expected = query.collect(engine="cpu", streaming=False)
    try:
        near(actual, expected)
    except error:
        print(runtime.describe(query))
        for name in expected.columns():
            if expected.column(name).dtype() != DataType.FLOAT64:
                continue
            var a = actual.column(name).float64()
            var b = expected.column(name).float64()
            for i in range(expected.height()):
                if a.is_valid(i) != b.is_valid(i):
                    print(name, i, "validity", a.is_valid(i), b.is_valid(i))
                elif (
                    b.is_valid(i)
                    and a._get(i) != b._get(i)
                    and not (isnan(a._get(i)) and isnan(b._get(i)))
                ):
                    print(name, i, "Metal", a._get(i), "CPU", b._get(i))
        raise error^


def expressions(window: Int, minimum: Int = -1, ddof: Int = 1) -> List[Expr]:
    return [
        col("x").rolling_var(window, minimum, ddof).alias("v"),
        col("x").rolling_std(window, minimum, ddof).alias("s"),
    ]


def test_rolling_moments_capability_packed_bounds_private_casts_and_scopes() raises:
    assert_true(not RowCapabilities().rolling_moments)
    var frame = DataFrame([Series("x", Column[UInt64]([0, UInt64.MAX]))])
    for window in List[Int]([1, 3, Int(Int64.MAX)]):
        for ddof in List[Int]([0, 1, Int(Int64.MAX)]):
            var query = frame.lazy().select_exprs(expressions(window, 1, ddof))
            var caps = _capabilities()
            caps.rolling_moments = False
            with assert_raises(
                contains="rolling variance and standard deviation require backend support"
            ):
                _ = lower_rows(query, caps)
            caps = _capabilities()
            caps.float64 = False
            caps.float64_emulated = False
            with assert_raises(contains="Float64 arithmetic"):
                _ = lower_rows(query, caps)
            var plan = lower_rows(query, _capabilities())
            var found = 0
            for step in plan.steps:
                if plan.code[step.start * 4] != ROW_CUMULATIVE_KEY:
                    continue
                found += 1
                assert_equal(plan.slot_dtypes[step.slot], DataType.FLOAT64)
                assert_equal(plan.node_dtypes[step.start], DataType.FLOAT64)
                assert_equal(plan.code[step.start * 4 + 2], 1)
                assert_true(
                    plan.code[step.start * 4 + 3] == 11
                    or plan.code[step.start * 4 + 3] == 12
                )
                var parameter = bitcast[DType.uint64](plan.literals[step.start])
                assert_equal(
                    parameter & UInt64(UInt32.MAX),
                    UInt64(min(window, Int(UInt32.MAX))),
                )
                assert_equal(
                    parameter >> 32, UInt64(min(ddof, Int(UInt32.MAX)))
                )
            assert_equal(found, 2)
            _ = Program(plan)
            assert_true(
                "unsupported"
                not in MetalRuntime(device_id=999999).describe(query)
            )
    with assert_raises(contains="ddof"):
        _ = lower_rows(
            frame.lazy().select(col("x").rolling_var(2, 1, -1)), _capabilities()
        )
    with assert_raises(contains="min_samples cannot exceed window_size"):
        _ = lower_rows(
            frame.lazy().select(col("x").rolling_std(2, 3)), _capabilities()
        )
    var partitioned = DataFrame(
        [
            Series("x", Column[Float64]([1, 3, 2, 5])),
            Series("k", Column[Int64]([0, 1, 0, 1])),
        ]
    )
    var nested = partitioned.lazy().select(
        col("x").rolling_var(2, 1, 0).over("k").rolling_std(3, 1, 0)
    )
    near(
        lower_partition_aggregates(nested).collect(engine="cpu"),
        nested.collect(engine="cpu"),
    )


# requires: metal
def test_native_metal_rolling_moments_numeric_widths_bounds_nulls_and_anchors() raises:
    _ = require_metal()
    var runtime = MetalRuntime()
    comptime for D in NUMERIC_DTYPES:
        var values = List[Scalar[D]]()
        var valid = List[Bool]()
        var keys = List[Int32]()
        for i in range(67):
            values.append(Scalar[D](i % 13))
            valid.append(i % 5 != 0)
            keys.append(Int32(i % 3))
        var frame = DataFrame(
            [
                Series("x", Column[Scalar[D]](values^, valid^)),
                Series("k", Column[Int32](keys^)),
            ]
        ).slice(3, 61)
        for window in List[Int]([1, 7, Int(Int64.MAX)]):
            for ddof in List[Int]([0, 1, Int(Int64.MAX)]):
                check(
                    frame.lazy().select_exprs(expressions(window, 0, ddof)),
                    runtime,
                )
                check(
                    frame.lazy().select_exprs(
                        [
                            col("x")
                            .rolling_var(window, 1, ddof)
                            .over("k")
                            .alias("v"),
                            col("x")
                            .rolling_std(window, 1, ddof)
                            .over("k")
                            .alias("s"),
                        ]
                    ),
                    runtime,
                )
        for length in List[Int]([0, 1, 3]):
            var nulls = DataFrame(
                [
                    Series(
                        "x",
                        Column[Scalar[D]](
                            List[Scalar[D]](length=length, fill=0),
                            List[Bool](length=length, fill=False),
                        ),
                    )
                ]
            )
            check(nulls.lazy().select_exprs(expressions(3, 0, 0)), runtime)
    var anchor = (
        DataFrame([Series("x", Column[Int64]([1, 3, 5]))])
        .lazy()
        .select_exprs(expressions(2, 1, 1))
        .collect(accelerator=runtime)
    )
    assert_true(not anchor.column("v").float64().is_valid(0))
    assert_equal(anchor.column("v").float64()._get(1), Float64(2))
    assert_equal(anchor.column("s").float64()._get(2), sqrt(Float64(2)))
    var default_minimum = DataFrame([Series("x", Column[Int64]([1, 3, 5]))])
    check(default_minimum.lazy().select_exprs(expressions(3)), runtime)


# requires: metal
def test_native_metal_rolling_moments_nonfinite_recovery_unsigned_and_decimal_inputs() raises:
    _ = require_metal()
    var runtime = MetalRuntime()
    var nan = bitcast[DType.float64](UInt64(0x7FF8000000000042))
    var infinity = Float64("inf")
    var frame = DataFrame(
        [
            Series(
                "x",
                Column[Float64](
                    [
                        nan,
                        1,
                        2,
                        3,
                        infinity,
                        4,
                        5,
                        6,
                        -infinity,
                        7,
                        8,
                        9,
                    ]
                ),
            )
        ]
    )
    for window in List[Int]([1, 2, 3, 17]):
        for ddof in List[Int]([0, 1, 2]):
            check(
                frame.lazy().select_exprs(expressions(window, 1, ddof)), runtime
            )
    var recovered = (
        frame.lazy()
        .select_exprs(expressions(2, 1, 0))
        .collect(accelerator=runtime)
    )
    assert_true(isnan(recovered.column("v").float64()._get(1)))
    assert_equal(recovered.column("v").float64()._get(3), Float64(0.25))
    assert_equal(recovered.column("v").float64()._get(7), Float64(0.25))
    for bits in List[UInt64]([UInt64(1), UInt64(0x7FEFFFFFFFFFFFFF)]):
        var boundary = bitcast[DType.float64](bits)
        var extremes = DataFrame(
            [
                Series(
                    "x",
                    Column[Float64](
                        [
                            boundary,
                            -boundary,
                            0,
                            -Float64(0),
                            boundary,
                            -boundary,
                            1,
                            2,
                        ]
                    ),
                )
            ]
        )
        for window in List[Int]([1, 2, 3, 7]):
            check(
                extremes.lazy().select_exprs(expressions(window, 1, 0)), runtime
            )
    var unsigned = DataFrame(
        [
            Series(
                "x",
                Column[UInt64](
                    [
                        0,
                        UInt64.MAX,
                        UInt64(1) << 63,
                        (UInt64(1) << 63) + 4096,
                    ]
                ),
            )
        ]
    )
    check(unsigned.lazy().select_exprs(expressions(3, 1, 0)), runtime)
    for length in List[Int]([3, 12, 30]):
        var decimal = decimal_fixture(length, 2)
        for name in List[String](["small", "large", "wide"]):
            check(
                decimal.lazy().select_exprs(
                    [
                        col(name).rolling_var(3, 1, 0).alias("v"),
                        col(name).rolling_std(3, 1, 0).alias("s"),
                    ]
                ),
                runtime,
            )
    var chunks = frame.slice(0, 5).vstack(frame.slice(5, 7)).rechunk()
    check(chunks.lazy().select_exprs(expressions(3, 1, 0)), runtime)


# requires: metal
def test_native_metal_rolling_moments_large_offsets_partition_workers_and_nested_consumers() raises:
    _ = require_metal()
    var runtime = MetalRuntime()
    var values = List[Float64]()
    var keys = List[Int32]()
    for i in range(16387):
        values.append(Float64(1e12) + Float64(i % 19))
        keys.append(Int32(i % 7))
    var frame = DataFrame(
        [
            Series("x", Column[Float64](values^)),
            Series("k", Column[Int32](keys^)),
        ]
    )
    for workers in List[Int]([1, 8]):
        var setting = WorkerSetting(workers)
        check(frame.lazy().select_exprs(expressions(127, 1, 1)), runtime)
        check(
            frame.lazy().select(col("x").rolling_std(31, 1, 0).over("k")),
            runtime,
        )
        _ = setting^
    var small = frame.slice(0, 43)
    check(
        small.lazy()
        .with_columns(col("x").rolling_var(7, 1, 0).over("k").alias("v"))
        .filter(col("v") > lit(Float64(3)))
        .sort("v")
        .select(col("v").sum()),
        runtime,
    )
    check(
        small.lazy().select(
            col("x")
            .rolling_var(3, 1, 0)
            .over("k")
            .rolling_std(5, 1, 0)
            .alias("nested")
        ),
        runtime,
    )
    check(
        small.lazy()
        .select((col("x").rolling_var(3, 1, 0) + lit(Float64(2))).alias("v"))
        .head(7),
        runtime,
    )
    var query = small.lazy().select_exprs(expressions(3, 1, 0))
    var estimate = runtime.estimate_auto(query)
    assert_true(";mojo-metal-v1;ops=" in estimate.family)
    assert_true(";rolling-moments-v2" in estimate.family)
    with assert_raises(contains="budget"):
        _ = query.collect(accelerator=MetalRuntime(memory_limit_bytes=1))
    same_outcome(
        small.lazy().select(
            (col("x").cast(DataType.INT64) + lit(Int64(1))).rolling_var(3, 1, 0)
        )
    )
    check(query, runtime)
