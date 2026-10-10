"""Metal preflight is CPU-only; installed runtimes also execute parity checks."""
from std.testing import TestSuite, assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, Series, col, lit
from dataframe.metal import MetalRuntime, metal_installed


def test_configuration_and_semantic_rejections_without_device() raises:
    with assert_raises(contains="nonnegative"):
        _ = MetalRuntime(-1)
    var runtime = MetalRuntime()
    var wide = DataFrame([Series("x", Column[Float64]([1, 2]))])
    assert_true(
        "Float64 requires CPU"
        in runtime.describe(wide.lazy().select(col("x") + 1))
    )
    with assert_raises(contains="Float64 requires CPU"):
        _ = wide.lazy().select(col("x") + 1).collect(accelerator=runtime)
    var narrow = DataFrame([Series("x", Column[Float32]([1, 2]))])
    with assert_raises(contains="Float64 accumulation"):
        _ = narrow.lazy().select(col("x").sum()).collect(accelerator=runtime)
    # Auto is conservative until matched end-to-end cost evidence exists.
    var query = narrow.lazy().select(col("x") + 1)
    assert_true(
        query.collect(engine="auto", accelerator=runtime).equals(
            query.collect(engine="cpu")
        )
    )


def test_native_resident_parity_and_profile() raises:
    if not metal_installed():
        return
    var runtime = MetalRuntime()
    var frame = DataFrame(
        [
            Series(
                "x",
                Column[Float32](
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    [
                        True,
                        False,
                        True,
                        True,
                        True,
                        True,
                        False,
                        True,
                        True,
                        True,
                        True,
                    ],
                ),
            ),
            Series(
                "b",
                Column[Bool](
                    [
                        True,
                        False,
                        True,
                        False,
                        True,
                        False,
                        True,
                        False,
                        True,
                        False,
                        True,
                    ],
                    [
                        True,
                        True,
                        False,
                        True,
                        True,
                        True,
                        True,
                        False,
                        True,
                        True,
                        True,
                    ],
                ),
            ),
        ]
    ).slice(3, 8)
    var query = (
        frame.lazy()
        .with_columns(((col("x") + 1) * 2).alias("y"))
        .filter((col("y") > 6) & col("b").fill_null(lit(True)))
        .select_exprs([col("y"), (col("x") != 5).alias("flag"), col("b")])
        .head(3)
    )
    var cpu = query.collect(engine="cpu")
    assert_true(query.collect(engine="accel").equals(cpu))
    var measured = query.profile(accelerator=runtime)
    assert_true(measured[0].equals(cpu))
    assert_equal(measured[1].column("executor").string()._get(0), "metal")
    assert_equal(measured[1].column("synchronizations").int64()._get(0), 1)
    assert_true(measured[1].column("download_bytes").int64()._get(0) > 0)
    var again = query.profile(accelerator=runtime)
    assert_true(again[0].equals(cpu))
    assert_true(again[1].column("pipeline_cache_hit").bool()._get(0))
    assert_equal(
        again[1].column("pipeline_compile_ms").float64()._get(0), Float64(0)
    )
    var denied = MetalRuntime(memory_limit_bytes=0)
    with assert_raises(contains="memory budget"):
        _ = query.collect(accelerator=denied)


def test_native_integer_precision_and_empty_results() raises:
    if not metal_installed():
        return
    var runtime = MetalRuntime()
    var integers = DataFrame(
        [Series("x", Column[Int64]([9007199254740993, -9007199254740993, 1]))]
    )
    var query = integers.lazy().select((col("x") + lit(Int64(2))).alias("y"))
    assert_true(
        query.collect(accelerator=runtime).equals(query.collect(engine="cpu"))
    )
    var narrow = DataFrame([Series("x", Column[Int32]([2147483647, 1, -1]))])
    var sum = narrow.lazy().select_exprs(
        [
            col("x").sum().alias("s"),
            col("x").count().alias("n"),
            col("x").len().alias("l"),
        ]
    )
    assert_true(
        sum.collect(accelerator=runtime).equals(sum.collect(engine="cpu"))
    )
    with assert_raises(contains="overflow"):
        _ = narrow.lazy().select(col("x") + 1).collect(accelerator=runtime)
    var empty = (
        narrow.head(0)
        .lazy()
        .select_exprs([col("x"), (col("x") > 1).alias("b")])
    )
    assert_true(
        empty.collect(accelerator=runtime).equals(empty.collect(engine="cpu"))
    )
    var no_rows = (
        narrow.lazy().filter(col("x") < lit(Int32(-10))).select(col("x"))
    )
    assert_true(
        no_rows.collect(accelerator=runtime).equals(
            no_rows.collect(engine="cpu")
        )
    )
    var empty_sum = narrow.head(0).lazy().select(col("x").sum())
    assert_true(
        empty_sum.collect(accelerator=runtime).equals(
            empty_sum.collect(engine="cpu")
        )
    )


def main() raises:
    var suite = TestSuite.discover_tests[__functions_in_module()]()
    suite^.run()
