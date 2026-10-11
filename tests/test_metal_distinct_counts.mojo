"""Distinct counts reuse resident grouping, including null categories."""
from std.testing import TestSuite, assert_true, assert_raises
from dataframe import DataFrame, Series, Column, StringColumn, col
from dataframe.metal import MetalRuntime, metal_installed


def frame() raises -> DataFrame:
    return DataFrame(
        [
            Series(
                "g",
                Column[Int32](
                    [1, 1, 1, 2, 2, 3],
                    List[Bool]([True, True, True, True, True, False]),
                ),
            ),
            Series(
                "x",
                Column[UInt64](
                    [UInt64.MAX, UInt64.MAX, 0, 3, 4, 5],
                    List[Bool]([True, True, False, True, False, False]),
                ),
            ),
            Series(
                "s",
                StringColumn(
                    List[String](["a", "a", "null", "b", "b", "null"]),
                    List[Bool]([True, True, False, True, True, False]),
                ),
            ),
            Series(
                "__dataframe_metal_distinct_value",
                Column[Int16]([1, 2, 3, 4, 5, 6]),
            ),
        ]
    )


def test_distinct_counts_numeric_text_null_empty() raises:
    if not metal_installed():
        return
    var input = frame()
    var runtime = MetalRuntime()
    for name in List[String](["x", "s"]):
        var query = input.lazy().select(col(name).n_unique().alias("distinct"))
        assert_true(
            query.collect(accelerator=runtime).equals(
                query.collect(engine="cpu")
            )
        )
        var grouped = (
            input.lazy()
            .group_by("g", maintain_order=True)
            .agg(col(name).n_unique().alias("distinct"))
        )
        assert_true(
            grouped.collect(accelerator=runtime).equals(
                grouped.collect(engine="cpu")
            )
        )
        var empty = (
            input.slice(0, 0)
            .lazy()
            .select(col(name).n_unique().alias("distinct"))
        )
        assert_true(
            empty.collect(accelerator=runtime).equals(
                empty.collect(engine="cpu")
            )
        )
        var empty_grouped = (
            input.slice(0, 0)
            .lazy()
            .group_by("g", maintain_order=True)
            .agg(col(name).n_unique().alias("distinct"))
        )
        assert_true(
            empty_grouped.collect(accelerator=runtime).equals(
                empty_grouped.collect(engine="cpu")
            )
        )


def test_distinct_count_composition_filter_join_budget() raises:
    if not metal_installed():
        return
    var input = frame()
    var runtime = MetalRuntime()
    var query = (
        input.lazy()
        .filter(col("g") == 1)
        .select(
            (col("s").str().to_uppercase().n_unique() + 1).alias("distinct")
        )
    )
    assert_true(
        query.collect(accelerator=runtime).equals(query.collect(engine="cpu"))
    )
    var right = DataFrame([Series("g", Column[Int32]([1, 2]))])
    var joined = (
        input.lazy()
        .join(right.lazy(), "g")
        .group_by("g", maintain_order=True)
        .agg(col("x").n_unique().alias("distinct"))
    )
    assert_true(
        joined.collect(accelerator=runtime).equals(joined.collect(engine="cpu"))
    )
    with assert_raises(contains="budget"):
        _ = query.collect(accelerator=MetalRuntime(memory_limit_bytes=1))
    assert_true(
        query.collect(accelerator=runtime).equals(query.collect(engine="cpu"))
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
