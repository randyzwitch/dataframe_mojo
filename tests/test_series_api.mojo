"""Series methods match the equivalent expression over a one-column frame."""
from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
    assert_false,
    assert_raises,
)
from dataframe import AnyValue, Column, DataFrame, Expr, Series, col, lit


def nan() -> Float64:
    return Float64(0) / Float64(0)


def x() raises -> Series:
    return Series(
        "x", Column[Int64]([4, -2, 4, 0, 7], [True, True, True, False, True])
    )


def y() raises -> Series:
    return Series(
        "y", Column[Int64]([1, 2, 3, 4, 0], [True, True, False, True, True])
    )


def frame() raises -> DataFrame:
    return DataFrame([x(), y()])


def via_expr(expr: Expr) raises -> Series:
    return frame().select(expr.alias("x")).column("x")


def test_binary_operators_match_expressions() raises:
    assert_true(
        (x() + y()).equals(via_expr(col("x") + col("y")), check_names=True)
    )
    assert_true((x() - y()).equals(via_expr(col("x") - col("y"))))
    assert_true((x() * y()).equals(via_expr(col("x") * col("y"))))
    assert_true((x() / y()).equals(via_expr(col("x") / col("y"))))
    assert_true((x() // y()).equals(via_expr(col("x") // col("y"))))
    assert_true((x() % y()).equals(via_expr(col("x") % col("y"))))
    assert_true((x() ** y()).equals(via_expr(col("x") ** col("y"))))
    assert_true((x() < y()).equals(via_expr(col("x") < col("y"))))
    assert_true((x() <= y()).equals(via_expr(col("x") <= col("y"))))
    assert_true((x() > y()).equals(via_expr(col("x") > col("y"))))
    assert_true((x() >= y()).equals(via_expr(col("x") >= col("y"))))
    assert_true(x().eq(y()).equals(via_expr(col("x").eq(col("y")))))
    assert_true(x().ne(y()).equals(via_expr(col("x").ne(col("y")))))
    assert_true(
        (x() + lit(Int64(1))).equals(via_expr(col("x") + lit(Int64(1))))
    )
    assert_true(
        (x() > lit(Int64(0))).equals(via_expr(col("x") > lit(Int64(0))))
    )
    assert_true((-x()).equals(via_expr(-col("x"))))
    var p = x() > lit(Int64(0))
    var q = y() > lit(Int64(1))
    assert_true(
        (p & q).equals(
            via_expr((col("x") > lit(Int64(0))) & (col("y") > lit(Int64(1))))
        )
    )
    assert_true(
        (p | q).equals(
            via_expr((col("x") > lit(Int64(0))) | (col("y") > lit(Int64(1))))
        )
    )
    assert_true((~p).equals(via_expr(~(col("x") > lit(Int64(0))))))
    assert_equal((x() + y()).name(), "x")
    # Same-named operands still pair positionally.
    assert_true((x() + x()).equals(via_expr(col("x") * lit(Int64(2)))))
    with assert_raises(contains="Series lengths differ: 5 and 2"):
        _ = x() + x().head(2)
    with assert_raises(contains="requires matching dtypes"):
        _ = x() + Series("f", Column[Float64]([1, 2, 3, 4, 5]))


def test_reductions_return_tagged_values() raises:
    var s = x()
    assert_true(s.sum() == frame().select(col("x").sum()).item())
    assert_equal(s.sum().int64(), Int64(13))
    assert_equal(s.mean().float64(), 13.0 / 4.0)
    assert_equal(s.min().int64(), Int64(-2))
    assert_equal(s.max().int64(), Int64(7))
    assert_equal(s.median().float64(), 4.0)
    assert_equal(s.quantile(0).float64(), -2.0)
    assert_true(s.std() == frame().select(col("x").std()).item())
    assert_true(s.var(0) == frame().select(col("x").var(0)).item())
    assert_equal(s.count(), 4)
    assert_equal(s.n_unique(), 4)
    assert_equal(s.first().int64(), Int64(4))
    assert_equal(s.last().int64(), Int64(7))
    var flags = s > lit(Int64(5))
    assert_true(flags.any().bool())
    assert_false(flags.all().bool())
    assert_true(flags.all(ignore_nulls=False) == AnyValue(False))
    assert_true(Series("e", Column[Int64]([])).max().is_null())


def test_elementwise_helpers_and_access() raises:
    var s = x()
    assert_true(s.is_null().equals(via_expr(col("x").is_null())))
    assert_true(s.is_not_null().equals(via_expr(col("x").is_not_null())))
    assert_true(
        s.fill_null(lit(Int64(9))).equals(
            via_expr(col("x").fill_null(lit(Int64(9))))
        )
    )
    assert_true(s.abs().equals(via_expr(col("x").abs())))
    var f = Series("f", Column[Float64]([1.25, nan(), -2.5]))
    assert_equal(String(f.round(1).get(0)), "1.3")
    assert_true(
        s.apply(col("x").cum_sum()).equals(via_expr(col("x").cum_sum()))
    )
    assert_equal(s[0].int64(), Int64(4))
    assert_equal(s[-1].int64(), Int64(7))
    assert_true(s[3].is_null())
    with assert_raises():
        _ = s[5]
    assert_equal(len(s.to_values()), 5)
    assert_true(s.to_values()[3].is_null())
    assert_equal(len(s.head(2)), 2)
    assert_equal(s.tail(2)[0].is_null(), True)
    assert_true(frame()["y"].equals(y(), check_names=True))


def test_sort_unique_value_counts() raises:
    var s = x()
    var sorted = s.sort()
    assert_equal(String(sorted[0]), "-2")
    assert_true(sorted[4].is_null())
    assert_true(s.sort(descending=True, nulls_last=False)[0].is_null())
    var distinct = s.unique(maintain_order=True)
    assert_equal(len(distinct), 4)
    assert_equal(distinct[0].int64(), Int64(4))
    assert_true(distinct[2].is_null())
    var counts = s.value_counts()
    assert_equal(counts.columns(), [String("x"), "count"])
    assert_equal(counts.item(0, "x").int64(), Int64(4))
    assert_equal(counts.item(0, "count").int64(), Int64(2))
    assert_equal(counts.height(), 4)
    var unsorted = s.value_counts(sort=False, name="n")
    assert_equal(unsorted.item(1, "x").int64(), Int64(-2))
    assert_equal(unsorted.columns()[1], "n")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
