"""Native wide sums succeed only after a GPU proof of every partial's range."""
from std.testing import TestSuite, assert_equal, assert_true, assert_raises
from dataframe import DataFrame, Series, Column, col, lit
from dataframe.dtype import DataType
from dataframe.metal import MetalRuntime, metal_installed


def test_bounded_wide_scalar_sums_and_nulls() raises:
    if not metal_installed():
        return
    var runtime = MetalRuntime()
    for n in List[Int]([0, 1, 257, 4097]):
        var signed = List[Int64]()
        var unsigned = List[UInt64]()
        var valid = List[Bool]()
        for i in range(n):
            signed.append(Int64(i % 31 - 15))
            unsigned.append(UInt64(i % 31))
            valid.append(i % 7 != 2)
        var frame = DataFrame(
            [
                Series("i", Column[Int64](signed^, valid)),
                Series("u", Column[UInt64](unsigned^, valid)),
            ]
        )
        var query = frame.lazy().select_exprs(
            [
                col("i").sum().alias("i"),
                col("u").sum().alias("u"),
                col("i").sum(min_count=1).alias("nullable_i"),
                col("u").sum(min_count=n + 1).alias("nullable_u"),
            ]
        )
        var result = query.collect(accelerator=runtime)
        assert_true(result.equals(query.collect(engine="cpu")))
        assert_equal(result.column("i").dtype(), DataType.INT64)
        assert_equal(result.column("u").dtype(), DataType.UINT64)
    var extremes = DataFrame(
        [
            Series("lo", Column[Int64]([Int64.MIN])),
            Series("hi", Column[Int64]([Int64.MAX])),
            Series("u", Column[UInt64]([UInt64.MAX])),
        ]
    )
    var query = extremes.lazy().select_exprs(
        [col("lo").sum(), col("hi").sum(), col("u").sum()]
    )
    assert_true(
        query.collect(accelerator=runtime).equals(query.collect(engine="cpu"))
    )


def test_bounded_wide_grouped_sums_and_resident_inputs() raises:
    if not metal_installed():
        return
    var runtime = MetalRuntime()
    for n in List[Int]([0, 257, 4097]):
        var keys = List[Int32]()
        var signed = List[Int64]()
        var unsigned = List[UInt64]()
        var valid = List[Bool]()
        for i in range(n):
            keys.append(Int32(i % 11))
            signed.append(Int64(i % 37 - 18))
            unsigned.append(UInt64(i % 37))
            valid.append(i % 13 != 5)
        var frame = DataFrame(
            [
                Series("k", Column[Int32](keys^)),
                Series("i", Column[Int64](signed^, valid)),
                Series("u", Column[UInt64](unsigned^, valid)),
            ]
        )
        var grouped = (
            frame.lazy()
            .group_by("k", maintain_order=True)
            .agg([col("i").sum(), col("u").sum(min_count=1)])
        )
        assert_true(
            grouped.collect(accelerator=runtime).equals(
                grouped.collect(engine="cpu")
            )
        )
        var selected = (
            frame.lazy()
            .filter(col("k") == lit(Int32(2)))
            .select(col("u").sum())
        )
        assert_true(
            selected.collect(accelerator=runtime).equals(
                selected.collect(engine="cpu")
            )
        )
        var sorted = (
            frame.lazy().sort("k").slice(-100, 17).select(col("i").sum())
        )
        assert_true(
            sorted.collect(accelerator=runtime).equals(
                sorted.collect(engine="cpu")
            )
        )
    var durations = DataFrame(
        [
            Series("d", Column[Int64]([1, -2, 4])).with_dtype(
                DataType.duration("us")
            )
        ]
    )
    var duration_sum = durations.lazy().select(col("d").sum())
    assert_true(
        duration_sum.collect(accelerator=runtime).equals(
            duration_sum.collect(engine="cpu")
        )
    )
    # Each group has one row, so even extrema outside the global bound fit.
    var extremes = DataFrame(
        [
            Series("k", Column[Int32]([1, 2])),
            Series("v", Column[UInt64]([UInt64.MAX, UInt64.MAX - 1])),
        ]
    )
    var grouped = (
        extremes.lazy().group_by("k", maintain_order=True).agg(col("v").sum())
    )
    assert_true(
        grouped.collect(accelerator=runtime).equals(
            grouped.collect(engine="cpu")
        )
    )
    var right = DataFrame([Series("k", Column[Int32]([1, 2]))])
    var joined = (
        extremes.lazy().join(right.lazy(), "k").select(col("v").count())
    )
    assert_true(
        joined.collect(accelerator=runtime).equals(joined.collect(engine="cpu"))
    )


def test_unproved_sums_reject_and_runtime_recovers() raises:
    if not metal_installed():
        return
    var runtime = MetalRuntime()
    var cancel = DataFrame(
        [Series("v", Column[Int64]([Int64.MAX, -Int64.MAX]))]
    )
    # A wide CPU accumulator can prove the cancelling total. Native partials
    # deliberately decline this input instead of wrapping or approximating.
    assert_equal(
        cancel.lazy()
        .select(col("v").sum())
        .collect(engine="cpu")
        .item(0, "v")
        .int64(),
        Int64(0),
    )
    with assert_raises(contains="exact wide accumulation"):
        _ = cancel.lazy().select(col("v").sum()).collect(accelerator=runtime)
    var unsafe = DataFrame(
        [
            Series("k", Column[Int32]([1, 1])),
            Series("v", Column[UInt64]([UInt64.MAX, UInt64(1)])),
        ]
    )
    with assert_raises(contains="exact wide accumulation"):
        _ = (
            unsafe.lazy()
            .group_by("k")
            .agg(col("v").sum())
            .collect(accelerator=runtime)
        )
    var filtered = (
        unsafe.lazy().filter(col("v") == lit(UInt64(1))).select(col("v").sum())
    )
    assert_true(
        filtered.collect(accelerator=runtime).equals(
            filtered.collect(engine="cpu")
        )
    )
    var safe = DataFrame([Series("v", Column[Int64]([1, 2, 3]))])
    var query = safe.lazy().select(col("v").sum())
    assert_true(
        query.collect(accelerator=runtime).equals(query.collect(engine="cpu"))
    )
    # Checked reductions in pruned child regions retain conservative preflight.
    with assert_raises(contains="terminal boundary"):
        _ = (
            safe.lazy()
            .select(col("v").sum())
            .join(safe.lazy(), "v")
            .collect(accelerator=MetalRuntime(device_id=999999))
        )
    with assert_raises(contains="without head"):
        _ = (
            unsafe.lazy()
            .group_by("k")
            .agg(col("v").sum())
            .head(0)
            .collect(accelerator=MetalRuntime(device_id=999999))
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
