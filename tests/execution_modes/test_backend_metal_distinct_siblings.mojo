"""Distinct counts reuse resident grouping, including null categories."""
from dataframe.accelerator.gpu.runtime import capabilities as _capabilities
from std.testing import assert_true, assert_raises
from dataframe import DataFrame, Series, Column, StringColumn, Expr, col
from dataframe.lazy import LazyFrame
from dataframe.metal import MetalRuntime
from dataframe.accelerator.distinct_counts import (
    has_distinct_counts,
    lower_distinct_counts,
)
from dataframe.accelerator.joins import lower_join_dag, DAG_JOIN
from dataframe.accelerator.sets import lower_set_operations


from tests.execution_modes.test_backend_metal import require_metal


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


def test_native_metal_distinct_siblings_null_order_empty_and_composition() raises:
    _ = require_metal()
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


def test_native_metal_distinct_siblings_sum_and_checked_child_restrictions() raises:
    _ = require_metal()
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


def test_native_metal_distinct_siblings_join_profile_budget_and_collision() raises:
    _ = require_metal()
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


def test_native_metal_distinct_siblings_multikey_hash_slices_and_null_text() raises:
    _ = require_metal()
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
