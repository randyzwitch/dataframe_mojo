"""LazyFrame.concat (SQL UNION ALL): the left plan's rows then the right's,
as the eager concat gives them, through both executors, with each input
filtered, aggregated or joined in its own plan, a projection above that
reaches both scans, columns in a different order on the right, a chain of
three, UNION as concat followed by unique, and mismatched inputs rejected.
"""
from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from dataframe import Column, DataFrame, Series, col, concat, lit


def sales(start: Int, n: Int) raises -> DataFrame:
    var ids = List[Int64]()
    var stores = List[Int64]()
    var amounts = List[Float64]()
    for i in range(start, start + n):
        ids.append(Int64(i))
        stores.append(Int64(i % 5))
        amounts.append(Float64(i % 13) * 1.5)
    return DataFrame(
        [
            Series("id", Column[Int64](ids^)),
            Series("store", Column[Int64](stores^)),
            Series("amount", Column[Float64](amounts^)),
        ]
    )


def test_rows_in_order_through_both_executors() raises:
    var a = sales(0, 70_000)
    var b = sales(100_000, 30_000)
    var expected = concat([a.copy(), b.copy()])
    var plan = a.lazy().concat(b.lazy())
    for streaming in [False, True]:
        var got = plan.collect(streaming=streaming, batch_size=4096)
        assert_true(got.equals(expected))
    assert_equal(plan.collect().height(), 100_000)


def test_each_input_keeps_its_own_plan() raises:
    var a = sales(0, 50_000)
    var b = sales(50_000, 50_000)
    var stores = DataFrame(
        [
            Series("store", Column[Int64]([0, 1, 2, 3, 4])),
            Series("region", Column[Int64]([10, 10, 20, 20, 30])),
        ]
    )
    var left = (
        a.lazy()
        .filter(col("amount") > lit(5.0))
        .join(stores.lazy(), on="store")
        .group_by(["region"])
        .agg([col("amount").sum().alias("total")])
    )
    var right = (
        b.lazy()
        .join(stores.lazy(), on="store")
        .group_by(["region"])
        .agg([col("amount").sum().alias("total")])
    )
    var expected = concat(
        [
            a.filter(col("amount") > lit(5.0))
            .join(stores, on="store")
            .group_by(["region"])
            .agg([col("amount").sum().alias("total")]),
            b.join(stores, on="store")
            .group_by(["region"])
            .agg([col("amount").sum().alias("total")]),
        ]
    )
    var got = left.concat(right).collect()
    assert_equal(got.height(), expected.height())
    var by: List[String] = ["region", "total"]
    assert_true(got.sort(by).equals(expected.sort(by)))


def test_aggregate_and_projection_above() raises:
    var a = sales(0, 40_000)
    var b = sales(40_000, 40_000)
    var whole = concat([a.copy(), b.copy()])
    var expected = whole.group_by(["store"], maintain_order=True).agg(
        [col("amount").sum().alias("total"), col("id").len().alias("n")]
    )
    var got = (
        a.lazy()
        .concat(b.lazy())
        .group_by(["store"], maintain_order=True)
        .agg([col("amount").sum().alias("total"), col("id").len().alias("n")])
        .collect()
    )
    assert_true(got.equals(expected))
    # A selection above narrows both inputs, and the output is the same.
    var narrow = a.lazy().concat(b.lazy()).select(["amount"])
    assert_true(narrow.collect().equals(whole.select(["amount"])))
    var described = narrow.explain(streaming=False)
    assert_true("CONCAT" in described)


def test_right_columns_in_another_order() raises:
    var a = sales(0, 1_000)
    var b = sales(1_000, 1_000).select(["amount", "id", "store"])
    var got = a.lazy().concat(b.lazy()).collect()
    assert_equal(got.columns(), a.columns())
    assert_true(got.equals(concat([a.copy(), b.select(a.columns())])))


def test_three_inputs_and_union() raises:
    var a = sales(0, 3_000)
    var b = sales(0, 3_000)
    var c = sales(2_000, 3_000)
    var all_rows = a.lazy().concat(b.lazy()).concat(c.lazy()).collect()
    assert_equal(all_rows.height(), 9_000)
    assert_true(all_rows.equals(concat([a.copy(), b.copy(), c.copy()])))
    # UNION: concat then unique keeps each distinct row once.
    var union = a.lazy().concat(c.lazy()).unique(maintain_order=True).collect()
    assert_equal(union.height(), 5_000)
    assert_true(
        union.equals(concat([a.copy(), c.copy()]).unique(maintain_order=True))
    )


def test_mismatched_inputs_raise() raises:
    var a = sales(0, 100)
    with assert_raises(contains="columns"):
        _ = a.lazy().concat(a.lazy().select(["id", "store"])).collect()
    with assert_raises(contains="no column"):
        _ = (
            a.lazy()
            .concat(
                a.lazy().select_exprs(
                    [col("id"), col("store"), col("amount").alias("other")]
                )
            )
            .collect()
        )
    with assert_raises():
        _ = (
            a.lazy()
            .concat(
                a.lazy().with_columns(
                    [col("store").cast("float64").alias("store")]
                )
            )
            .collect()
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
