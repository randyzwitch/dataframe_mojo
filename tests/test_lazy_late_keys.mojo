"""Group keys a join gathers from its build side are grouped by their codes
and become their values again in the result: the same groups as the eager
plan through inner and left joins (a missing build row is a null key), two
keys from two builds, a key passed through a select, null build values, a
key the aggregations also read (left as values), and a build with repeated
values (one group per value, not per build row)."""
from std.testing import TestSuite, assert_equal, assert_true
from dataframe import Column, DataFrame, Expr, Series, StringColumn, col, lit


def facts(n: Int) raises -> DataFrame:
    var store = List[Int64](capacity=n)
    var mode = List[Int64](capacity=n)
    var amount = List[Int64](capacity=n)
    for i in range(n):
        store.append(Int64(i % 23))
        mode.append(Int64(i % 7))
        amount.append(Int64(i % 101))
    return DataFrame(
        [
            Series("store", Column[Int64](store^)),
            Series("mode", Column[Int64](mode^)),
            Series("amount", Column[Int64](amount^)),
        ]
    )


def stores() raises -> DataFrame:
    # 20 of the 23 store ids; names repeat (two stores per name) and one
    # name is null.
    var ids = List[Int64]()
    var names = List[String]()
    var valid = List[Bool]()
    for i in range(20):
        ids.append(Int64(i))
        names.append("store name " + String(i // 2) + " with a long tail")
        valid.append(i != 5)
    return DataFrame(
        [
            Series("s_id", Column[Int64](ids^)),
            Series("s_name", StringColumn(names, valid)),
        ]
    )


def modes() raises -> DataFrame:
    var ids = List[Int64]()
    var names = List[String]()
    for i in range(7):
        ids.append(Int64(i))
        names.append("mode" + String(i % 4))
    return DataFrame(
        [
            Series("m_id", Column[Int64](ids^)),
            Series("m_name", StringColumn(names^)),
        ]
    )


def same(got: DataFrame, expected: DataFrame, keys: List[String]) raises:
    assert_equal(got.columns(), expected.columns())
    assert_equal(got.height(), expected.height())
    assert_true(got.sort(keys).equals(expected.sort(keys)))


def test_one_key_inner_and_left() raises:
    var f = facts(150_000)
    var aggregates: List[Expr] = [
        col("amount").sum().alias("total"),
        col("amount").len().alias("n"),
    ]
    var keys: List[String] = ["s_name"]
    for how in ["inner", "left"]:
        var expected = (
            f.filter(col("amount") > 3)
            .join(stores(), left_on=["store"], right_on=["s_id"], how=how)
            .group_by(keys)
            .agg(aggregates)
        )
        var got = (
            f.lazy()
            .filter(col("amount") > 3)
            .join(
                stores().lazy(), left_on=["store"], right_on=["s_id"], how=how
            )
            .group_by(keys)
            .agg(aggregates)
            .collect(batch_size=4096)
        )
        same(got, expected, keys)


def test_two_builds_and_a_select() raises:
    var f = facts(120_000)
    var keys: List[String] = ["m_name", "s_name"]
    var expected = (
        f.join(stores(), left_on=["store"], right_on=["s_id"])
        .join(modes(), left_on=["mode"], right_on=["m_id"])
        .select(["s_name", "m_name", "amount"])
        .group_by(keys)
        .agg([col("amount").sum().alias("total")])
    )
    var got = (
        f.lazy()
        .join(stores().lazy(), left_on=["store"], right_on=["s_id"])
        .join(modes().lazy(), left_on=["mode"], right_on=["m_id"])
        .select(["s_name", "m_name", "amount"])
        .group_by(keys)
        .agg([col("amount").sum().alias("total")])
        .collect(batch_size=4096)
    )
    same(got, expected, keys)


def test_key_also_reduced_keeps_values() raises:
    var f = facts(90_000)
    var keys: List[String] = ["s_name"]
    var aggregates: List[Expr] = [
        col("s_name").str().len_bytes().max().alias("width"),
        col("amount").sum().alias("total"),
    ]
    var expected = (
        f.join(stores(), left_on=["store"], right_on=["s_id"])
        .group_by(keys)
        .agg(aggregates)
    )
    var got = (
        f.lazy()
        .join(stores().lazy(), left_on=["store"], right_on=["s_id"])
        .group_by(keys)
        .agg(aggregates)
        .collect(batch_size=4096)
    )
    same(got, expected, keys)


def test_key_filtered_above_the_join() raises:
    # A filter reading the key keeps it as values.
    var f = facts(90_000)
    var keys: List[String] = ["s_name"]
    var expected = (
        f.join(stores(), left_on=["store"], right_on=["s_id"])
        .filter(col("s_name") != lit("store name 3 with a long tail"))
        .group_by(keys)
        .agg([col("amount").sum().alias("total")])
    )
    var got = (
        f.lazy()
        .join(stores().lazy(), left_on=["store"], right_on=["s_id"])
        .filter(col("s_name") != lit("store name 3 with a long tail"))
        .group_by(keys)
        .agg([col("amount").sum().alias("total")])
        .collect(batch_size=4096)
    )
    same(got, expected, keys)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
