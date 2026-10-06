"""LazyFrame.join(other, how="cross"): every pairing, left-major, as the
eager cross join gives it, through both executors, with projection
pushdown, a filter above, chunked inputs, an empty side, name collisions,
and the one-row-by-one-row shape that places summary scalars side by side.
"""
from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from dataframe import Column, DataFrame, Series, col, lit


def left_side() raises -> DataFrame:
    var ids = List[Int64]()
    var amounts = List[Float64]()
    for i in range(1_000):
        ids.append(Int64(i))
        amounts.append(Float64(i % 7))
    return DataFrame(
        [
            Series("id", Column[Int64](ids^)),
            Series("amount", Column[Float64](amounts^)),
        ]
    )


def right_side() raises -> DataFrame:
    return DataFrame(
        [
            Series("rate", Column[Float64]([0.5, 1.0, 2.0])),
            Series("amount", Column[Float64]([10.0, 20.0, 30.0])),
        ]
    )


def test_pairs_every_row_as_the_eager_join_does() raises:
    var left = left_side()
    var right = right_side()
    var want = left.join(right, how="cross")
    var plan = left.lazy().join(right.lazy(), how="cross")
    assert_equal(want.height(), 3_000)
    assert_true(plan.collect().equals(want))
    assert_true(plan.collect(streaming=False).equals(want))
    assert_true(plan.collect(optimize=False).equals(want))
    assert_equal(
        plan.collect().columns(), ["id", "amount", "rate", "amount_right"]
    )
    var chunked = Series._from_chunks(
        [
            left.column("amount").slice(0, 300),
            left.column("amount").slice(300, 700),
        ]
    )
    var pieces = DataFrame([left.column("id"), chunked^])
    assert_true(
        pieces.lazy().join(right.lazy(), how="cross").collect().equals(want)
    )


def test_projection_and_filter_above() raises:
    var left = left_side()
    var right = right_side()
    var plan = (
        left.lazy()
        .join(right.lazy(), how="cross")
        .filter((col("id") % 2 == 0) & (col("rate") > lit(0.75)))
        .select_exprs([(col("amount") * col("rate")).alias("scaled")])
    )
    var want = (
        left.join(right, how="cross")
        .filter((col("id") % 2 == 0) & (col("rate") > lit(0.75)))
        .select_exprs([(col("amount") * col("rate")).alias("scaled")])
    )
    assert_equal(want.height(), 1_000)
    assert_true(plan.collect().equals(want))
    assert_true(plan.collect(streaming=False).equals(want))
    var total = plan.select(col("scaled").sum().alias("t")).collect()
    assert_equal(
        total.column("t").float64()._get(0),
        want.select(col("scaled").sum().alias("t")).item().float64(),
    )


def test_one_row_sides_and_an_empty_side() raises:
    var left = left_side().lazy()
    var a = left.select(col("amount").sum().alias("total"))
    var b = left.select(col("amount").mean().alias("average"))
    var c = left.select(col("id").len().alias("rows"))
    var row = a.join(b, how="cross").join(c, how="cross").collect()
    assert_equal(row.height(), 1)
    assert_equal(row.columns(), ["total", "average", "rows"])
    assert_equal(row.column("rows").get(0).int64(), 1_000)
    var none = left_side().lazy().filter(col("id") < 0)
    assert_equal(
        none.join(right_side().lazy(), how="cross").collect().height(), 0
    )
    assert_equal(
        right_side().lazy().join(none, how="cross").collect().height(), 0
    )
    with assert_raises(contains="requires key columns"):
        _ = left.join(right_side().lazy(), how="inner")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
