"""Lazy joins whose left input a filter makes small (#377): a plan's single
join, and the deepest join of a chain, run eagerly so the hash table is
built on the small side, and the rest of a chain streams. Results must
equal the eager plan, row for row once sorted.
"""
from std.testing import TestSuite, assert_equal, assert_true

from dataframe import Column, DataFrame, Series, col

comptime BIG = 600_000


def tables() raises -> Tuple[DataFrame, DataFrame, DataFrame]:
    var keys = List[Int64](capacity=BIG)
    var groups = List[Int64](capacity=BIG)
    var values = List[Float64](capacity=BIG)
    for i in range(BIG):
        keys.append(Int64((i * 7919) % 20_000))
        groups.append(Int64(i % 50))
        values.append(Float64(i % 1000) / 10)
    var fact = DataFrame(
        [
            Series("k", Column[Int64](keys^)),
            Series("g", Column[Int64](groups^)),
            Series("v", Column[Float64](values^)),
        ]
    )
    var ids = List[Int64]()
    var kinds = List[Int64]()
    for i in range(20_000):
        ids.append(Int64(i))
        kinds.append(Int64(i % 97))
    var small = DataFrame(
        [
            Series("id", Column[Int64](ids^)),
            Series("kind", Column[Int64](kinds^)),
        ]
    )
    var gs = List[Int64]()
    var names = List[String]()
    for i in range(50):
        gs.append(Int64(i))
        names.append("group" + String(i))
    var dim = DataFrame(
        [
            Series("g", Column[Int64](gs^)),
            Series("name", Column[String](names^)),
        ]
    )
    return (small^, fact^, dim^)


def same(a: DataFrame, b: DataFrame) raises:
    var names = a.columns()
    assert_equal(a.height(), b.height())
    assert_true(
        a.sort(names).equals(b.select(names).sort(names)),
        "lazy and eager results differ",
    )


def test_single_join_with_small_filtered_left() raises:
    var t = tables()
    var picked = t[0].filter(col("kind") == 3)
    var eager = picked.join(t[1], left_on=["id"], right_on=["k"])
    var lazy = (
        t[0]
        .lazy()
        .filter(col("kind") == 3)
        .join(t[1].lazy(), left_on=["id"], right_on=["k"])
        .collect()
    )
    assert_true(eager.height() > 0)
    same(lazy, eager)


def test_chain_builds_its_deepest_join_on_the_small_side() raises:
    var t = tables()
    var picked = t[0].filter(col("kind") == 3)
    var eager = picked.join(t[1], left_on=["id"], right_on=["k"]).join(
        t[2], "g"
    )
    var lazy = (
        t[0]
        .lazy()
        .filter(col("kind") == 3)
        .join(t[1].lazy(), left_on=["id"], right_on=["k"])
        .join(t[2].lazy(), "g")
        .collect()
    )
    assert_true(eager.height() > 0)
    same(lazy, eager)
    # A left join keeps every left row, with nulls where nothing matches.
    var outer = (
        t[0]
        .lazy()
        .filter(col("kind") == 96)
        .join(t[1].lazy(), left_on=["id"], right_on=["k"], how="left")
        .join(t[2].lazy(), "g", how="left")
        .collect()
    )
    var outer_eager = (
        t[0]
        .filter(col("kind") == 96)
        .join(t[1], left_on=["id"], right_on=["k"], how="left")
        .join(t[2], "g", how="left")
    )
    same(outer, outer_eager)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
