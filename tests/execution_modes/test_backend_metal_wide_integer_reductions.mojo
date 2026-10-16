"""Exact integer sums and means preserve cancellation and checked final results."""
# contract-group: execution.backends
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import DataFrame, Series, Column, col, lit
from dataframe.dtype import DataType
from dataframe.metal import MetalRuntime


from tests.execution_modes.test_backend_metal import require_metal


# requires: metal
def test_native_metal_exact_wide_integer_limits_partials_means_and_nulls() raises:
    _ = require_metal()
    var runtime = MetalRuntime()
    for n in List[Int]([0, 1, 4, 257, 4097, 65537]):
        var signed = List[Int64]()
        var unsigned = List[UInt64]()
        var keys = List[Int32]()
        var valid = List[Bool]()
        for i in range(n):
            # Individual lanes and blocks overflow 64 bits before cancellation.
            signed.append(Int64.MAX if i % 4 < 2 else -Int64.MAX)
            unsigned.append(UInt64.MAX - UInt64(i % 7))
            keys.append(Int32((i // 4) % 7))
            valid.append(i % 4 != 3)
        var frame = DataFrame(
            [
                Series("k", Column[Int32](keys^)),
                Series("i", Column[Int64](signed^)),
                Series("u", Column[UInt64](unsigned^, valid^)),
            ]
        )
        var query = frame.lazy().select_exprs(
            [
                col("i").mean().alias("im"),
                col("u").mean().alias("um"),
                col("u").sum(min_count=n + 1).alias("suppressed"),
            ]
        )
        assert_true(
            query.collect(accelerator=runtime).equals(
                query.collect(engine="cpu")
            )
        )
        var grouped = (
            frame.lazy()
            .group_by("k", maintain_order=True)
            .agg(
                [
                    col("i").mean().alias("im"),
                    col("u").mean().alias("um"),
                    col("u").sum(min_count=n + 1).alias("suppressed"),
                ]
            )
        )
        assert_true(
            grouped.collect(accelerator=runtime).equals(
                grouped.collect(engine="cpu")
            )
        )
        var cancelled = frame.lazy().head(n - n % 4).select(col("i").sum())
        assert_true(
            cancelled.collect(accelerator=runtime).equals(
                cancelled.collect(engine="cpu")
            )
        )
        var sliced = frame.lazy().slice(1, 257).select(col("u").mean())
        assert_true(
            sliced.collect(accelerator=runtime).equals(
                sliced.collect(engine="cpu")
            )
        )
    var extremes = DataFrame(
        [
            Series(
                "i", Column[Int64]([Int64.MIN, Int64.MIN, Int64.MAX, Int64.MAX])
            ),
            Series(
                "u",
                Column[UInt64](
                    [UInt64.MAX, UInt64.MAX, UInt64.MAX, UInt64.MAX]
                ),
            ),
        ]
    )
    var query = extremes.lazy().select_exprs(
        [
            col("i").sum().alias("sum"),
            col("i").mean().alias("mean"),
            col("u").mean().alias("unsigned"),
        ]
    )
    assert_true(
        query.collect(accelerator=runtime).equals(query.collect(engine="cpu"))
    )
    for name in List[String](["i", "u"]):
        var overflow = extremes.lazy().head(2).select(col(name).sum())
        with assert_raises(contains="overflow"):
            _ = overflow.collect(accelerator=runtime)
        with assert_raises(contains="overflow"):
            _ = overflow.collect(engine="cpu")
        # CPU evaluates the SUM before the outer head; so does Metal.
        with assert_raises(contains="overflow"):
            _ = overflow.head(0).collect(accelerator=runtime)
        with assert_raises(contains="overflow"):
            _ = overflow.head(0).collect(engine="cpu")
    assert_true(";mojo-metal-v1;ops=" in runtime.estimate_auto(query).family)


# requires: metal
def test_native_metal_exact_wide_integer_resident_inputs_consumers_and_budget() raises:
    _ = require_metal()
    var runtime = MetalRuntime()
    var frame = DataFrame(
        [
            Series(
                "k",
                Column[String](
                    ["a", "a", "b", "b"], [True, True, False, False]
                ),
            ),
            Series(
                "i",
                Column[Int64]([Int64.MIN, Int64.MAX, Int64.MAX, -Int64.MAX]),
            ),
            Series(
                "u",
                Column[UInt64](
                    [UInt64.MAX, UInt64.MAX, UInt64.MAX, UInt64.MAX]
                ),
            ),
        ]
    )
    var right = DataFrame([Series("k", Column[String](["a", "b"]))])
    var joined = frame.lazy().join(right.lazy(), "k", how="left")
    var query = joined.select_exprs(
        [
            col("i").sum().alias("sum"),
            col("u").mean().alias("mean"),
            col("u").sum(min_count=5).alias("nullsum"),
            col("i").max().alias("maximum"),
        ]
    )
    var measured = query.profile(accelerator=runtime)
    assert_true(measured[0].equals(query.collect(engine="cpu")))
    with assert_raises(contains="budget"):
        _ = query.collect(accelerator=MetalRuntime(memory_limit_bytes=0))
    var grouped = joined.group_by("k", maintain_order=True).agg(
        [
            col("i").sum().alias("sum"),
            col("u").mean().alias("mean"),
            col("u").sum(min_count=5).alias("nullsum"),
        ]
    )
    assert_true(
        grouped.collect(accelerator=runtime).equals(
            grouped.collect(engine="cpu")
        )
    )
    var consumer = joined.with_columns(col("u").mean().alias("mean"))
    assert_true(
        consumer.collect(accelerator=runtime).equals(
            consumer.collect(engine="cpu")
        )
    )
    var sorted_mean = (
        joined.group_by("k", maintain_order=True)
        .agg(col("u").mean().alias("mean"))
        .select(col("mean"))
        .sort("mean")
        .head(1)
    )
    assert_true(
        sorted_mean.collect(accelerator=runtime).equals(
            sorted_mean.collect(engine="cpu")
        )
    )
    var durations = DataFrame(
        [
            Series(
                "d",
                Column[Int64]([Int64.MAX, Int64.MAX, -Int64.MAX, -Int64.MAX]),
            ).with_dtype(DataType.duration("us"))
        ]
    )
    var duration_sum = durations.lazy().select(col("d").sum())
    assert_true(
        duration_sum.collect(accelerator=runtime).equals(
            duration_sum.collect(engine="cpu")
        )
    )
    # A scalar with a 32-byte state remains null after copying to row consumers.
    var null_frame = DataFrame(
        [Series("i", Column[Int64]([Int64.MAX, Int64.MIN], [False, False]))]
    )
    var null_scalar = null_frame.lazy().with_columns(
        col("i").mean().alias("mean")
    )
    assert_true(
        null_scalar.collect(accelerator=runtime).equals(
            null_scalar.collect(engine="cpu")
        )
    )
