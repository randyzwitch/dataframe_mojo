"""The optimizer sees through the plan a rollup, cube or grouping-sets
aggregation expands to: a column no set reads is not scanned, and filters
and selections above it give the unoptimized plan's rows."""
# contract-group: planner.relational
from std.testing import assert_equal, assert_true
from dataframe import (
    Column,
    DataFrame,
    LazyFrame,
    Series,
    col,
    grouping,
    grouping_id,
)
from tests.support.coverage_compare import assert_frame


def source() raises -> DataFrame:
    var a = List[Int64]()
    var a_valid = List[Bool]()
    var b = List[Int64]()
    var m = List[Int64]()
    var unused = List[Int64]()
    for i in range(300):
        a.append(Int64(i % 7))
        a_valid.append(i % 10 != 0)
        b.append(Int64(i % 3))
        m.append(Int64(i))
        unused.append(Int64(-i))
    return DataFrame(
        [
            Series("a", Column[Int64](a^, a_valid.copy())),
            Series("b", Column[Int64](b^)),
            Series("m", Column[Int64](m^)),
            Series("unused", Column[Int64](unused^)),
        ]
    )


def same_both_ways(plan: LazyFrame, order: List[String]) raises:
    """Optimized and unoptimized runs agree, on every executor setting."""
    var want = plan.collect(optimize=False, streaming=False).sort(order)
    for streaming in [True, False]:
        for batch_size in [65536, 7]:
            assert_frame(
                plan.collect(
                    optimize=True, streaming=streaming, batch_size=batch_size
                ).sort(order),
                want,
            )


def test_selection_and_filter_above_expansion() raises:
    var base = (
        source()
        .lazy()
        .group_by_rollup(["a", "b"])
        .agg(
            [
                col("m").sum().alias("total"),
                grouping_id(["a", "b"]).alias("gid"),
            ]
        )
    )
    var order: List[String] = ["a", "b", "gid"]
    same_both_ways(base.select(["a", "total"]), ["a", "total"])
    same_both_ways(base.filter(col("gid") == 1), order)
    same_both_ways(base.filter(col("a") > 2), order)
    same_both_ways(
        base.filter(col("total") > 5000).select(["gid", "total"]),
        ["gid", "total"],
    )


def test_unread_columns_are_not_scanned() raises:
    var plan = (
        source()
        .lazy()
        .group_by_cube(["a", "b"])
        .agg(col("m").sum().alias("total"))
        .select(["total", "a"])
    )
    var text = plan.explain(streaming=False)
    assert_true(text.find("unused") < 0, text)


def test_grouping_sets_with_a_pushed_filter_below() raises:
    var sets: List[List[String]] = [["a"], ["b"], []]
    var plan = (
        source()
        .lazy()
        .filter(col("m") > 100)
        .group_by_grouping_sets(sets)
        .agg([col("m").sum().alias("total"), grouping("a").alias("ga")])
    )
    same_both_ways(plan, ["ga", "a", "b"])
