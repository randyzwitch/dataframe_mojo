"""CPU differential coverage for resident expressions and stable compaction."""
from std.testing import TestSuite, assert_equal, assert_true, assert_raises
from std.math import isnan, isfinite, abs
from std.memory import bitcast
from dataframe import (
    BoolColumn,
    Column,
    DataFrame,
    LazyFrame,
    Series,
    col,
    lit,
    null,
)
from dataframe.dtype import DataType
from dataframe.accel_rows import lower_rows
from dataframe.accel_row_memory import row_memory
from dataframe_accel.nvidia import NvidiaRuntime


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
            if e.dtype() == DataType.BOOL:
                assert_equal(av.bool(), ev.bool())
            elif e.dtype().is_float():
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
                assert_true(
                    abs(x - y)
                    <= (1e-6 if e.dtype() == DataType.FLOAT32 else 1e-12)
                    * max(1.0, abs(y))
                )
            else:
                assert_equal(av.int64(), ev.int64())


def check(plan: LazyFrame, runtime: NvidiaRuntime) raises:
    var actual = plan.collect(engine="accel", accelerator=runtime)
    same(actual, plan.collect())
    same(actual, plan.collect(optimize=False, streaming=False))


def test_stable_rows_slices_and_byte_boundaries() raises:
    var runtime = NvidiaRuntime()
    comptime for d in range(2):
        comptime D = DType.float32 if d == 0 else DType.float64
        var values = List[Scalar[D]]()
        var valid = List[Bool]()
        var booleans = List[Bool]()
        for i in range(1040):
            values.append(Scalar[D](i))
            valid.append(i % 3 != 0)
            booleans.append(i % 5 != 0)
        for offset in [0, 1, 7]:
            for length in [0, 1, 7, 8, 9, 255, 256, 257, 1025]:
                var frame = DataFrame(
                    [
                        Series(
                            "x",
                            Column[Scalar[D]](values.copy(), valid).slice(
                                offset, length
                            ),
                        ),
                        Series(
                            "y",
                            Column[Scalar[D]](values.copy()).slice(
                                offset, length
                            ),
                        ),
                        Series(
                            "b",
                            BoolColumn(booleans, valid).slice(offset, length),
                        ),
                    ]
                )
                check(frame.lazy(), runtime)
                var plan = (
                    frame.lazy()
                    .with_columns(((col("x") + 1) * 2 - col("y")).alias("z"))
                    .filter((col("x") > 3) & (col("b") | col("x").is_null()))
                    .with_columns(
                        (col("z").fill_null(lit(Scalar[D](-1))) + 0.5).alias(
                            "z"
                        )
                    )
                    .filter(col("y") != 11)
                    .select_exprs(
                        [
                            col("y"),
                            col("z"),
                            (col("x") != col("z")).alias("cmp"),
                            col("b"),
                        ]
                    )
                )
                check(plan, runtime)
                check(plan.head(3), runtime)
                # Shared source slices must survive collection unchanged.
                var source_y = frame.column("y").numeric[D]()
                var source_x = frame.column("x").numeric[D]()
                var source_b = frame.column("b").bool()
                for i in range(length):
                    assert_equal(source_y._get(i), values[offset + i])
                    assert_equal(source_x._valid(i), valid[offset + i])
                    assert_equal(source_b._valid(i), valid[offset + i])
                    if valid[offset + i]:
                        assert_equal(source_x._get(i), values[offset + i])
                        assert_equal(source_b._get(i), booleans[offset + i])


def test_kleene_all_combinations_and_nonfinite() raises:
    var runtime = NvidiaRuntime()
    var a = List[Bool]()
    var b = List[Bool]()
    var av = List[Bool]()
    var bv = List[Bool]()
    for i in range(3):
        for j in range(3):
            a.append(i == 1)
            b.append(j == 1)
            av.append(i != 2)
            bv.append(j != 2)
    var frame = DataFrame(
        [Series("a", BoolColumn(a, av)), Series("b", BoolColumn(b, bv))]
    )
    check(
        frame.lazy().select_exprs(
            [
                (col("a") & col("b")).alias("and"),
                (col("a") | col("b")).alias("or"),
                (col("a") ^ col("b")).alias("xor"),
                (~col("a")).alias("not"),
                col("a").is_null().alias("null"),
                col("a").is_not_null().alias("present"),
                col("a").fill_null(col("b")).alias("fill"),
                (col("a") == col("b")).alias("equal"),
            ]
        ),
        runtime,
    )
    check(frame.lazy().filter(col("a") | col("b")), runtime)
    check(frame.lazy().select(col("a").count()), runtime)
    comptime for d in range(2):
        comptime D = DType.float32 if d == 0 else DType.float64
        var f = DataFrame(
            [
                Series(
                    "x",
                    Column[Scalar[D]](
                        [
                            Scalar[D](Float64("nan")),
                            Scalar[D](Float64("inf")),
                            Scalar[D](Float64("-inf")),
                            -0.0,
                            0.0,
                            -2.0,
                            3.0,
                        ]
                    ),
                )
            ]
        )
        var predicates = [
            col("x") > 0,
            col("x") >= 0,
            col("x") < 0,
            col("x") <= 0,
            col("x") == 0,
            col("x") != 0,
        ]
        for p in predicates:
            check(
                f.lazy()
                .filter(p)
                .select_exprs(
                    [
                        col("x"),
                        (-col("x")).alias("neg"),
                        ((col("x") * 0.0) + col("x")).alias("nested"),
                    ]
                ),
                runtime,
            )
        check(
            f.lazy().select_exprs(
                [null("bool").alias("nb"), col("x"), lit(True).alias("true")]
            ),
            runtime,
        )


def test_all_nulls_empty_filters_and_reductions() raises:
    var runtime = NvidiaRuntime()
    for length in [0, 1, 9, 257]:
        var frame = DataFrame(
            [
                Series(
                    "x",
                    Column[Float64](
                        List[Float64](length=length, fill=Float64("nan")),
                        List[Bool](length=length, fill=False),
                    ),
                ),
                Series(
                    "y", Column[Float64](List[Float64](length=length, fill=2.0))
                ),
            ]
        )
        for predicate in [
            col("x").is_null(),
            col("x") > 0,
            col("y") < 0,
            col("y") > 0,
        ]:
            var plan = (
                frame.lazy()
                .filter(predicate)
                .with_columns((col("x").fill_null(col("y")) * 2 + 1).alias("z"))
            )
            check(plan, runtime)
            check(
                plan.select_exprs(
                    [
                        col("x").sum().alias("s0"),
                        col("x").sum(min_count=1).alias("s1"),
                        col("z").sum().alias("z"),
                        col("x").count().alias("n"),
                    ]
                ),
                runtime,
            )
            check(plan.select(col("z").sum()).head(0), runtime)


def test_grid_stride_multiple_filters_and_binding() raises:
    var runtime = NvidiaRuntime()
    var values = List[Float32]()
    for i in range(262145):
        values.append(Float32(i))
    var frame = DataFrame([Series("x", Column[Float32](values^))])
    var plan = (
        frame.lazy()
        .with_columns([(col("x") + 1).alias("x"), col("x").alias("original")])
        .filter(col("x") > 4)
        .filter(col("original") < 262140)
        .select_exprs([col("original"), col("x")])
    )
    check(plan, runtime)
    check(
        plan.select_exprs(
            [col("x").sum().alias("x"), col("original").count().alias("n")]
        ),
        runtime,
    )
    check(
        frame.lazy()
        .with_columns(lit(Float32(1)).alias("constant"))
        .drop(["x"]),
        runtime,
    )


def test_float_rounding_null_payloads_and_empty_broadcast() raises:
    var runtime = NvidiaRuntime()
    var tiny = bitcast[DType.float32](UInt32(1))
    var frame = DataFrame(
        [
            Series(
                "x",
                Column[Float32](
                    [
                        Float32.MAX,
                        -Float32.MAX,
                        tiny,
                        -tiny,
                        -0.0,
                        Float64("nan").cast[DType.float32](),
                    ],
                    [True, True, True, True, True, False],
                ),
            )
        ]
    )
    var actual = (
        frame.lazy()
        .select_exprs(
            [
                ((col("x") * 2.0) * 0.5).alias("overflow"),
                ((col("x") * 0.5) * 2.0).alias("underflow"),
                col("x"),
            ]
        )
        .collect(accelerator=runtime)
    )
    same(
        actual,
        frame.lazy()
        .select_exprs(
            [
                ((col("x") * 2.0) * 0.5).alias("overflow"),
                ((col("x") * 0.5) * 2.0).alias("underflow"),
                col("x"),
            ]
        )
        .collect(),
    )
    assert_equal(
        bitcast[DType.uint32](actual.item(4, "x").float32()), UInt32(0x80000000)
    )
    var empty = DataFrame([Series("x", Column[Float32](List[Float32]()))])
    check(empty.lazy().with_columns(lit(Float32(3)).alias("constant")), runtime)
    check(
        empty.lazy().select_exprs(
            [col("x"), lit(Float32(3)).alias("constant")]
        ),
        runtime,
    )


def test_profile_memory_and_rejection() raises:
    var runtime = NvidiaRuntime()
    var frame = DataFrame([Series("x", Column[Float32]([1, 2, 3]))])
    var plan = (
        frame.lazy()
        .filter(col("x") > 1)
        .select_exprs(
            [(col("x") * 2 + 1).alias("y"), (col("x") == 2).alias("b")]
        )
    )
    var profile = plan.profile(accelerator=runtime)
    same(profile[0], plan.collect())
    var memory = row_memory(lower_rows(plan))
    assert_equal(
        profile[1].item(0, "peak_requested_device_bytes").int64(),
        Int64(memory.summary.peak_bytes),
    )
    assert_equal(
        profile[1].item(0, "upload_bytes").int64(), Int64(memory.upload_bytes)
    )
    assert_equal(profile[1].item(0, "download_bytes").int64(), Int64(27))
    assert_equal(profile[1].item(0, "kernel_launches").int64(), Int64(9))
    assert_equal(profile[1].item(0, "synchronizations").int64(), Int64(6))
    assert_true(profile[1].item(0, "kernel_ms").float64() >= 0)
    assert_true("resident" in plan.explain(accelerator=runtime))
    var limited = NvidiaRuntime(
        memory_limit_bytes=memory.summary.peak_bytes - 1
    )
    with assert_raises(contains="memory preflight"):
        _ = plan.collect(accelerator=limited)
    var exact = NvidiaRuntime(memory_limit_bytes=memory.summary.peak_bytes)
    check(plan, exact)
    with assert_raises(contains="unsupported"):
        _ = frame.lazy().select(col("x") / 2).collect(accelerator=runtime)
    check(plan, runtime)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
