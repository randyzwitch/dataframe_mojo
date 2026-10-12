"""Distinct counts reuse resident grouping, including null categories."""
from std.testing import TestSuite, assert_true, assert_raises
from dataframe import DataFrame, Series, Column, StringColumn, Expr, col
from dataframe.lazy import LazyFrame
from dataframe.metal import MetalRuntime, metal_installed, _capabilities
from dataframe.accelerator.distinct_counts import (
    has_distinct_counts,
    lower_distinct_counts,
)
from dataframe.accelerator.joins import lower_join_dag, DAG_JOIN
from dataframe.accelerator.sets import lower_set_operations


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


def check_siblings(query: LazyFrame) raises:
    assert_true(
        query.collect(accelerator=MetalRuntime()).equals(
            query.collect(engine="cpu")
        )
    )


def sibling_expressions() -> List[Expr]:
    return [
        col("s").n_unique().alias("text_distinct"),
        col("x").n_unique().alias("numeric_distinct"),
        col("x").count().alias("valid"),
        col("x").len().alias("rows"),
        col("x").max().alias("largest"),
        col("x").null_count().alias("nulls"),
    ]


def test_distinct_siblings_null_order_empty_and_composition() raises:
    if not metal_installed():
        return
    var input = frame()
    for rows in List[Int]([0, input.height()]):
        var source = input.slice(0, rows)
        check_siblings(source.lazy().select_exprs(sibling_expressions()))
        check_siblings(
            source.lazy()
            .group_by("g", maintain_order=True)
            .agg(sibling_expressions())
        )
    check_siblings(
        input.lazy()
        .group_by("g", maintain_order=True)
        .agg(
            [
                (col("s").n_unique() + col("x").count()).alias("combined"),
                col("x").n_unique().alias("distinct"),
            ]
        )
    )
    check_siblings(
        input.lazy()
        .filter(col("g") < 3)
        .group_by("g", maintain_order=True)
        .agg(sibling_expressions())
        .head(1)
    )
    check_siblings(
        input.lazy()
        .group_by("g", maintain_order=True)
        .agg(sibling_expressions())
        .filter(col("rows") > 1)
    )


def test_distinct_siblings_sum_and_checked_child_restrictions() raises:
    if not metal_installed():
        return
    var input = DataFrame(
        [
            Series("g", Column[Int32]([1, 1, 2, 2])),
            Series("x", Column[Int16]([1, 2, 3, 4])),
        ]
    )
    var expressions = List[Expr](
        [
            col("x").n_unique().alias("distinct"),
            col("x").sum().alias("total"),
        ]
    )
    check_siblings(input.lazy().select_exprs(expressions))
    check_siblings(
        input.lazy().group_by("g", maintain_order=True).agg(expressions)
    )
    var wide = DataFrame(
        [
            Series("g", Column[Int32]([1, 1, 2, 2])),
            Series("x", Column[Int64]([1, 2, 3, 4])),
        ]
    )
    with assert_raises(contains="checked sums require a terminal boundary"):
        _ = (
            wide.lazy()
            .group_by("g")
            .agg(expressions)
            .head(1)
            .collect(accelerator=MetalRuntime())
        )
    with assert_raises(contains="checked sums require a terminal boundary"):
        _ = (
            wide.lazy()
            .select_exprs(expressions)
            .collect(accelerator=MetalRuntime())
        )
    # Window states remain fully observed and permit the checked producer.
    check_siblings(
        wide.lazy().select(
            (col("x").n_unique() + col("x").sum()).over("g").alias("window")
        )
    )


def test_distinct_siblings_join_profile_budget_and_collision() raises:
    if not metal_installed():
        return
    var input = frame().with_columns(
        col("g").alias("__dataframe_metal_distinct_sibling_0")
    )
    var right = DataFrame([Series("g", Column[Int32]([1, 2]))])
    var query = (
        input.lazy()
        .join(right.lazy(), "g")
        .group_by("g", maintain_order=True)
        .agg(sibling_expressions())
    )
    check_siblings(query)
    var measured = query.profile(accelerator=MetalRuntime())
    assert_true(measured[0].equals(query.collect(engine="cpu")))
    assert_true(measured[1].height() >= 3)
    with assert_raises(contains="budget"):
        _ = query.collect(accelerator=MetalRuntime(memory_limit_bytes=1))
    check_siblings(query)


def test_distinct_siblings_metadata_and_shared_prefix() raises:
    var input = DataFrame(
        [
            Series("g", Column[Int32]([1, 1, 2])),
            Series("x", Column[Int16]([1, 2, 3])),
        ]
    )
    var expressions = List[Expr](
        [
            col("x").n_unique().alias("distinct"),
            col("x").count().alias("valid"),
            col("x").max().alias("largest"),
        ]
    )
    for grouped in List[Bool]([False, True]):
        var query = input.lazy().select_exprs(expressions)
        if grouped:
            query = (
                input.lazy().group_by("g", maintain_order=True).agg(expressions)
            )
        var count = len(query._nodes)
        var lowered = lower_distinct_counts(query)
        assert_true(len(query._nodes) == count)
        assert_true(not has_distinct_counts(lowered))
        var dag = lower_join_dag(lower_set_operations(lowered), _capabilities())
        assert_true(dag.joins == 2)
        var joins = 0
        for task in dag.tasks:
            if task.kind == DAG_JOIN:
                assert_true(task.how == (7 if grouped else 6))
                joins += 1
        assert_true(joins == 2)
        assert_true(
            "resident distinct counts"
            in MetalRuntime(device_id=999999).describe(query)
        )


def test_distinct_siblings_multikey_hash_slices_and_null_text() raises:
    if not metal_installed():
        return
    var gs = List[Int32]()
    var xs = List[UInt64]()
    var ss = List[String]()
    var valid = List[Bool]()
    for i in range(8205):
        gs.append(Int32(i % 4101))
        xs.append(UInt64.MAX if i % 3 == 0 else UInt64(i % 7))
        ss.append(String(i % 3))
        valid.append(i % 5 != 0)
    var input = DataFrame(
        [
            Series("g", Column[Int32](gs^, valid.copy())),
            Series("x", Column[UInt64](xs^, valid.copy())),
            Series("s", StringColumn(ss^, valid^)),
        ]
    ).slice(1, 8203)
    check_siblings(
        input.lazy()
        .group_by("g", maintain_order=True)
        .agg(sibling_expressions())
    )
    check_siblings(
        input.lazy()
        .group_by(["g", "s"], maintain_order=True)
        .agg(
            [
                col("x").n_unique().alias("distinct"),
                col("x").count().alias("valid"),
            ]
        )
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
