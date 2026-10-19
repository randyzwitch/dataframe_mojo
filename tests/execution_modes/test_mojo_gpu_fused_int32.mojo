"""Independent public semantics for bounded native integer scalar reduction."""
# contract-group: execution.backends
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, DataType, Series, col, lit
from dataframe.lazy import LazyFrame
from dataframe.metal import MetalRuntime
from tests.support.coverage_generators import offset_layout, chunk_layout


def _check(query: LazyFrame) raises -> DataFrame:
    var description = MetalRuntime().describe(query)
    if "fused Int32" not in description:
        print("Unexpected fused-family description:", description)
    assert_true("fused Int32" in description)
    var cpu = query.collect(engine="cpu", optimize=False, streaming=False)
    var profiled = query.profile(engine="accel", accelerator=MetalRuntime())
    assert_true(profiled[0].equals(cpu))
    assert_true(
        query.collect(engine="accel", accelerator=MetalRuntime()).equals(cpu)
    )
    assert_equal(
        profiled[1].item(0, "algorithm").string(), "mojo_gpu_fused_int32_sum"
    )
    assert_equal(profiled[0].column("total").dtype(), DataType.INT64)
    return profiled[1].copy()


# requires: mojo_gpu
def test_fused_int32_sum_nulls_layouts_empty_and_min_count() raises:
    var frame = DataFrame(
        [
            Series(
                "x",
                Column[Int32](
                    [-4, -1, 0, 2, 7], [True, False, True, True, True]
                ),
            )
        ]
    )
    for source in [
        frame.copy(),
        offset_layout(frame),
        chunk_layout(frame),
        frame.clear(),
    ]:
        for min_count in [0, 1, 9]:
            for filtered in [False, True]:
                var query = source.lazy()
                if filtered:
                    query = query.filter(col("x") > lit(Int32(0)))
                _ = _check(
                    query.select(
                        ((col("x") + lit(Int32(3))) * lit(Int32(2)))
                        .cast(DataType.INT64)
                        .sum(min_count=min_count)
                        .alias("total")
                    )
                )
                _ = _check(
                    query.select(
                        col("x")
                        .cast(DataType.INT64)
                        .sum(min_count=min_count)
                        .alias("total")
                    )
                )
    var nulls = DataFrame([Series("x", Column[Int32]([1, 2], [False, False]))])
    _ = _check(
        nulls.lazy().select(col("x").cast(DataType.INT64).sum().alias("total"))
    )
    _ = _check(
        nulls.lazy().select(
            col("x").cast(DataType.INT64).sum(min_count=1).alias("total")
        )
    )
    _ = _check(
        frame.lazy()
        .filter(col("x") > lit(Int32(99)))
        .select(col("x").cast(DataType.INT64).sum().alias("total"))
    )


# requires: mojo_gpu
def test_fused_int32_sum_bounded_partials_exact_budget_and_empty_device() raises:
    for rows in [0, 4097, 1048577]:
        var values = List[Int32](capacity=rows)
        for row in range(rows):
            values.append(Int32(row % 97) - 48)
        var frame = DataFrame([Series("x", Column[Int32](values^))])
        var query = (
            frame.lazy()
            .filter(col("x") > lit(Int32(0)))
            .select(
                ((col("x") + lit(Int32(3))) * lit(Int32(2)))
                .cast(DataType.INT64)
                .sum()
                .alias("total")
            )
        )
        var report = _check(query)
        assert_true(report.item(0, "reduction_partials").int64() <= 4096)
        assert_equal(
            report.item(0, "download_bytes").int64(), Int64(24 if rows else 0)
        )
        assert_equal(
            report.item(0, "kernel_launches").int64(), Int64(2 if rows else 0)
        )
        if rows == 0:
            assert_equal(
                report.item(0, "initialization_ms").float64(), Float64(0)
            )
        var peak = Int(report.item(0, "peak_requested_bytes").int64())
        var exact = query.collect(
            engine="accel", accelerator=MetalRuntime(memory_limit_bytes=peak)
        )
        assert_true(exact.equals(query.collect(engine="cpu")))
        var rejected = False
        try:
            _ = query.collect(
                engine="accel",
                accelerator=MetalRuntime(memory_limit_bytes=peak - 1),
            )
        except error:
            rejected = "memory budget" in String(error)
        assert_true(rejected)


# requires: mojo_gpu
def test_fused_int32_sum_independent_signed_endpoints_and_cancellation() raises:
    for endpoints in [True, False]:
        var values = List[Int32]()
        var expected = Int64(0)
        if endpoints:
            values = List[Int32]([Int32.MAX, Int32.MAX, Int32.MIN])
            expected = 2147483646
        else:
            # Cancellation crosses the 256-row partial boundary.
            for row in range(1024):
                values.append(Int32.MAX if row < 512 else -Int32.MAX)
            values.append(7)
            expected = 7
        var frame = DataFrame([Series("x", Column[Int32](values^))])
        var query = frame.lazy().select(
            col("x").cast(DataType.INT64).sum().alias("total")
        )
        _ = _check(query)
        var actual = query.collect(engine="accel", accelerator=MetalRuntime())
        assert_equal(actual.item(0, "total").int64(), expected)


# requires: mojo_gpu
def test_fused_int32_sum_preserves_checked_arithmetic_and_filter_order() raises:
    for multiplier in [Int32(1), Int32(2)]:
        var frame = DataFrame([Series("x", Column[Int32]([Int32.MAX, 1]))])
        var addend = Int32(1) if multiplier == 1 else Int32(0)
        var query = frame.lazy().select(
            ((col("x") + lit(addend)) * lit(multiplier))
            .cast(DataType.INT64)
            .sum()
            .alias("total")
        )
        var cpu_error = String()
        try:
            _ = query.collect(engine="cpu", optimize=False, streaming=False)
        except error:
            cpu_error = String(error)
        assert_true(cpu_error.byte_length() > 0)
        var gpu_error = String()
        try:
            _ = query.collect(
                engine="accel",
                accelerator=MetalRuntime(),
                optimize=False,
                streaming=False,
            )
        except error:
            gpu_error = String(error)
        assert_true(cpu_error in gpu_error)
    var frame = DataFrame([Series("x", Column[Int32]([Int32.MIN, 0, 2]))])
    _ = _check(
        frame.lazy()
        .filter(col("x") > lit(Int32(-1)))
        .select(
            ((col("x") + lit(Int32(-1))) * lit(Int32(2)))
            .cast(DataType.INT64)
            .sum()
            .alias("total")
        )
    )
