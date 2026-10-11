"""Aggregate leaves and their consumers stay resident across GPU regions."""
from std.testing import TestSuite, assert_true, assert_raises
from dataframe import DataFrame, Series, Column, Expr, col, lit, when
from dataframe.lazy import LazyFrame
from dataframe.metal import MetalRuntime, metal_installed


def fixture() raises -> DataFrame:
    return DataFrame(
        [
            Series("g", Column[Int32]([1, 1, 2, 2, 3])),
            Series(
                "x",
                Column[Int16](
                    [4, -1, 8, 2, 0],
                    List[Bool]([True, True, True, False, False]),
                ),
            ),
            Series(
                "__dataframe_metal_aggregate_0", Column[Int32]([1, 2, 3, 4, 5])
            ),
        ]
    )


def expressions() -> List[Expr]:
    return [
        (col("x").len() + 1).alias("rows_plus_one"),
        (col("x").max().cast("int64") - col("x").min().cast("int64")).alias(
            "range"
        ),
        (col("x").sum() + col("x").count()).alias("sum_and_count"),
        when(col("x").count() > 0)
        .then(col("x").min())
        .otherwise(lit(Int16(-7)))
        .alias("conditional"),
        (col("x").null_count() + 2).alias("nulls_plus_two"),
    ]


def check(query: LazyFrame) raises:
    var runtime = MetalRuntime()
    var expected = query.collect(engine="cpu")
    var actual = query.collect(accelerator=runtime)
    assert_true(actual.equals(expected))


def test_grouped_compositions() raises:
    if not metal_installed():
        return
    var frame = fixture()
    check(frame.lazy().group_by("g", maintain_order=True).agg(expressions()))
    check(
        frame.slice(0, 0)
        .lazy()
        .group_by("g", maintain_order=True)
        .agg(expressions())
    )
    var filtered = (
        frame.lazy()
        .filter(col("g") < 3)
        .group_by("g", maintain_order=True)
        .agg(expressions())
    )
    check(filtered)
    with assert_raises(contains="terminal projections without head"):
        _ = filtered.head(1).collect(accelerator=MetalRuntime())


def test_global_compositions_and_consumers() raises:
    if not metal_installed():
        return
    var frame = fixture()
    check(frame.lazy().select_exprs(expressions()))
    check(frame.slice(0, 0).lazy().select_exprs(expressions()))
    check(frame.lazy().filter(col("g") < 3).select_exprs(expressions()))
    var query = (
        frame.lazy()
        .select(col("x").count().alias("n"))
        .with_columns((col("n") + 1).alias("next"))
    )
    check(query)
    var measured = query.profile(accelerator=MetalRuntime())
    assert_true(measured[0].equals(query.collect(engine="cpu")))
    assert_true(measured[1].height() >= 2)
    with assert_raises(contains="budget"):
        _ = query.collect(accelerator=MetalRuntime(memory_limit_bytes=1))


def test_joined_compositions_and_checked_guards() raises:
    if not metal_installed():
        return
    var frame = fixture()
    var right = DataFrame([Series("g", Column[Int32]([1, 2, 3]))])
    check(frame.lazy().join(right.lazy(), "g").select_exprs(expressions()))
    var wide = DataFrame([Series("x", Column[Int64]([1, 2]))])
    with assert_raises(contains="unsupported"):
        _ = (
            wide.lazy()
            .select((col("x").sum() + 1).alias("n"))
            .collect(accelerator=MetalRuntime())
        )
    var overflow = DataFrame(
        [Series("x", Column[Int64]([Int64.MIN, Int64.MAX]))]
    )
    with assert_raises():
        _ = (
            overflow.lazy()
            .select((col("x").max() - col("x").min()).alias("range"))
            .collect(accelerator=MetalRuntime())
        )
    check(frame.lazy().select_exprs(expressions()))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
