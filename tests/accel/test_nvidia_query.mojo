"""Differential tests through the public LazyFrame API; optional CUDA suite."""
from std.math import abs, isnan, isfinite
from std.memory import bitcast
from std.testing import TestSuite, assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, LazyFrame, Series, col, lit, scan_csv
from dataframe_accel.nvidia import NvidiaRuntime
from dataframe.dtype import DataType


def same(actual: DataFrame, expected: DataFrame) raises:
    assert_equal(actual.columns(), expected.columns())
    assert_equal(actual.height(), expected.height())
    for name in expected.columns():
        var a = actual.column(name)
        var e = expected.column(name)
        assert_equal(a.dtype(), e.dtype())
        assert_equal(a.null_count(), e.null_count())
        for row in range(len(a)):
            var av = actual.item(row, name)
            var ev = expected.item(row, name)
            assert_equal(av.is_null(), ev.is_null())
            if ev.is_null():
                continue
            if e.dtype().is_float():
                var x = (
                    av.float32().cast[DType.float64]() if e.dtype()
                    == DataType.FLOAT32 else av.float64()
                )
                var y = (
                    ev.float32().cast[DType.float64]() if e.dtype()
                    == DataType.FLOAT32 else ev.float64()
                )
                if x == y or (isnan(x) and isnan(y)):
                    continue
                assert_true(isfinite(x) and isfinite(y))
                var tolerance = 1e-6 if e.dtype() == DataType.FLOAT32 else 1e-12
                assert_true(abs(x - y) <= tolerance * max(1.0, abs(y)))
            else:
                assert_equal(av.int64(), ev.int64())


def check(plan: LazyFrame, runtime: NvidiaRuntime) raises:
    var actual = plan.collect(engine="accel", accelerator=runtime)
    same(actual, plan.collect())
    same(actual, plan.collect(optimize=False, streaming=False))


def test_nullable_sliced_boundaries_and_empty() raises:
    var runtime = NvidiaRuntime()
    comptime for d in range(2):
        comptime D = DType.float32 if d == 0 else DType.float64
        var values = List[Scalar[D]]()
        var valid = List[Bool]()
        for i in range(1100):
            values.append(Scalar[D](i % 19 - 9) / 8)
            valid.append(i % 3 != 0)
        for bitmap in range(3):
            var source = Column[Scalar[D]](values.copy())
            if bitmap == 1:
                source = Column[Scalar[D]](values.copy(), valid)
            elif bitmap == 2:
                source = Column[Scalar[D]](
                    values.copy(), List[Bool](length=len(values), fill=False)
                )
            for offset in [0, 1, 7, 19]:
                for rows in [0, 1, 255, 256, 257, 1025]:
                    var plan = (
                        DataFrame([Series("x", source.slice(offset, rows))])
                        .lazy()
                        .filter(col("x") > 0)
                        .select_exprs(
                            [
                                (col("x") * 1.25).sum().alias("total"),
                                col("x").count().alias("count"),
                                col("x").sum(min_count=2).alias("minimum"),
                            ]
                        )
                    )
                    check(plan, runtime)


def test_comparisons_arithmetic_and_nonfinite() raises:
    var runtime = NvidiaRuntime()
    comptime for d in range(2):
        comptime D = DType.float32 if d == 0 else DType.float64
        var source = Column[Scalar[D]](
            [
                Scalar[D](Float64("nan")),
                Scalar[D](Float64("inf")),
                Scalar[D](Float64("-inf")),
                -0.0,
                0.0,
                -2.0,
                3.0,
            ]
        )
        var frame = DataFrame([Series("x", source^)])
        var predicates = [
            col("x") > 0,
            col("x") >= 0,
            col("x") < 0,
            col("x") <= 0,
            col("x") == 0,
            col("x") != 0,
            lit(Scalar[D](0)) < col("x"),
        ]
        for predicate in predicates:
            check(
                frame.lazy()
                .filter(predicate)
                .select_exprs(
                    [
                        (col("x") + 0.5).sum().alias("add"),
                        (lit(Scalar[D](0.5)) - col("x")).sum().alias("sub"),
                        (col("x") * -2.0).sum().alias("mul"),
                        (col("x") * 0.0).count().alias("count"),
                    ]
                ),
                runtime,
            )
        check(
            frame.lazy().select_exprs(
                [col("x").sum().alias("sum"), col("x").count().alias("count")]
            ),
            runtime,
        )


def test_public_options_profile_and_fetch() raises:
    var runtime = NvidiaRuntime()
    var plan = (
        DataFrame([Series("x", Column[Float32]([-2, 1, 3]))])
        .lazy()
        .filter(col("x") > 0)
        .select((col("x") * 1.25).sum())
    )
    var expected = plan.collect()
    for engine in ["cpu", "auto", "accel"]:
        for optimize in [False, True]:
            for streaming in [False, True]:
                same(
                    plan.collect(
                        engine=engine,
                        accelerator=runtime,
                        optimize=optimize,
                        streaming=streaming,
                    ),
                    expected,
                )
        same(plan.fetch(1, accelerator=runtime, engine=engine), expected)
        assert_equal(
            plan.fetch(0, accelerator=runtime, engine=engine).height(), 0
        )
    var profiled = plan.profile(accelerator=runtime)
    same(profiled[0], expected)
    assert_equal(profiled[1].item(0, "executor").string(), "nvidia")
    assert_equal(profiled[1].item(0, "input_rows").int64(), Int64(3))
    assert_true("NVIDIA supported" in plan.explain(accelerator=runtime))
    assert_true(
        "auto uses CPU" in plan.explain(accelerator=runtime, engine="auto")
    )
    with assert_raises(contains="Unknown engine"):
        _ = plan.collect(accelerator=runtime, engine="gpu")
    with assert_raises(contains="batch_size must be positive"):
        _ = plan.collect(accelerator=runtime, batch_size=0)


def test_rejection_is_explained_and_runtime_remains_usable() raises:
    var runtime = NvidiaRuntime()
    var frame = DataFrame(
        [
            Series("x", Column[Float32]([1, 2])),
            Series("y", Column[Float32]([3, 4])),
        ]
    )
    var plans = [
        frame.lazy().select(col("x").mean()),
        frame.lazy().select((col("x") / 2.0).sum()),
        frame.lazy().sort("x").select(col("x").sum()),
        DataFrame([Series("x", Column[Int64]([1]))])
        .lazy()
        .select(col("x").sum()),
        scan_csv("/nonexistent/gpu_query.csv").select(col("x").sum()),
    ]
    for plan in plans:
        assert_true("NVIDIA unsupported" in plan.explain(accelerator=runtime))
        with assert_raises(contains="NVIDIA unsupported"):
            _ = plan.collect(accelerator=runtime)
    check(frame.lazy().select(col("x").sum()), runtime)


def test_grid_stride_and_float32_rounding() raises:
    var runtime = NvidiaRuntime()
    var values = List[Float32](length=262145, fill=Float32(1.0000001192092896))
    var plan = (
        DataFrame([Series("x", Column[Float32](values^))])
        .lazy()
        .select_exprs(
            [
                (col("x") * 1.0000001192092896).sum().alias("total"),
                col("x").count().alias("count"),
            ]
        )
    )
    check(plan, runtime)
    assert_equal(
        plan.collect(accelerator=runtime).item(0, "count").int64(),
        Int64(262145),
    )
    var extremes = DataFrame(
        [Series("x", Column[Float32]([Float32.MAX, Float32.MAX, 1e-38]))]
    )
    check(extremes.lazy().select((col("x") * 2.0).sum()), runtime)
    check(
        extremes.lazy().filter(col("x") < 1.0).select((col("x") * 0.5).sum()),
        runtime,
    )
    # Widening before row arithmetic would incorrectly cancel to zero.
    var overflow = (
        DataFrame([Series("x", Column[Float32]([Float32.MAX, -Float32.MAX]))])
        .lazy()
        .select((col("x") * 2.0).sum())
    )
    check(overflow, runtime)
    assert_true(
        isnan(overflow.collect(accelerator=runtime).item(0, "x").float32())
    )
    # Rounding each half-subnormal to Float32 gives zero before summation.
    var tiny = bitcast[DType.float32](UInt32(1))
    var underflow = (
        DataFrame([Series("x", Column[Float32]([tiny, tiny]))])
        .lazy()
        .select((col("x") * 0.5).sum())
    )
    check(underflow, runtime)
    assert_equal(
        underflow.collect(accelerator=runtime).item(0, "x").float32(),
        Float32(0),
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
