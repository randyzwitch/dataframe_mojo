"""Order-dependent expressions and over() partitions."""
from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
    assert_false,
    assert_raises,
)
from dataframe import Column, DataFrame, Expr, Series, col, lit

comptime MAX = Int64(9223372036854775807)


def nan() -> Float64:
    return Float64(0) / Float64(0)


def frame() raises -> DataFrame:
    return DataFrame(
        [
            Series(
                "k",
                Column[String](
                    ["a", "b", "a", "b", "a", "a"],
                    [True, True, True, True, True, False],
                ),
            ),
            Series(
                "x",
                Column[Int64](
                    [3, 1, 0, 4, 2, 2], [True, True, False, True, True, True]
                ),
            ),
            Series(
                "f",
                Column[Float64](
                    [1.5, nan(), -1, 2, 0, 3],
                    [True, True, True, False, True, True],
                ),
            ),
            Series(
                "s",
                Column[String](
                    ["p", "q", "", "p", "z", "a"],
                    [True, True, True, True, False, True],
                ),
            ),
        ]
    )


def cells(df: DataFrame, expr: Expr) raises -> List[String]:
    var series = df.select(expr.alias("r")).column("r")
    var out = List[String]()
    for i in range(len(series)):
        out.append(String(series.get(i)))
    return out^


def test_cumulative() raises:
    var df = frame()
    assert_equal(
        cells(df, col("x").cum_sum()),
        [String("3"), "4", "null", "8", "10", "12"],
    )
    assert_equal(
        cells(df, col("x").cum_sum(reverse=True)),
        [String("12"), "9", "null", "8", "4", "2"],
    )
    assert_equal(
        cells(df, col("x").cum_count()), [String("1"), "2", "2", "3", "4", "5"]
    )
    assert_equal(
        cells(df, col("x").cum_count(reverse=True)),
        [String("5"), "4", "3", "3", "2", "1"],
    )
    assert_equal(
        cells(df, col("x").cum_min()), [String("3"), "1", "null", "1", "1", "1"]
    )
    assert_equal(
        cells(df, col("s").cum_max()), [String("p"), "q", "q", "q", "null", "q"]
    )
    # NaN sorts above numbers: cum_max becomes NaN, cum_min ignores it.
    assert_equal(
        cells(df, col("f").cum_max()),
        [String("1.5"), "nan", "nan", "null", "nan", "nan"],
    )
    assert_equal(
        cells(df, col("f").cum_min()),
        [String("1.5"), "1.5", "-1.0", "null", "-1.0", "-1.0"],
    )
    var big = DataFrame([Series("i", Column[Int64]([MAX, 1]))])
    with assert_raises(contains="overflow"):
        _ = big.select(col("i").cum_sum())


def test_shift_diff_and_fills() raises:
    var df = frame()
    assert_equal(
        cells(df, col("x").shift()),
        [String("null"), "3", "1", "null", "4", "2"],
    )
    assert_equal(
        cells(df, col("x").shift(-2)),
        [String("null"), "4", "2", "2", "null", "null"],
    )
    assert_equal(cells(df, col("x").shift(0)), cells(df, col("x")))
    assert_equal(
        cells(df, col("x").shift(99)), List[String](length=6, fill="null")
    )
    assert_equal(
        cells(df, col("x").diff()),
        [String("null"), "-2", "null", "null", "-2", "0"],
    )
    assert_equal(
        cells(df, col("x").diff(2)),
        [String("null"), "null", "null", "3", "null", "-2"],
    )
    var pct = cells(df, col("x").pct_change())
    assert_equal(pct[1], String(Float64(-2) / 3))
    var holes = DataFrame(
        [
            Series(
                "v",
                Column[Int64](
                    [0, 1, 0, 0, 0, 5, 0],
                    [False, True, False, False, False, True, False],
                ),
            )
        ]
    )
    assert_equal(
        cells(holes, col("v").forward_fill()),
        [String("null"), "1", "1", "1", "1", "5", "5"],
    )
    assert_equal(
        cells(holes, col("v").forward_fill(2)),
        [String("null"), "1", "1", "1", "null", "5", "5"],
    )
    assert_equal(
        cells(holes, col("v").backward_fill()),
        [String("1"), "1", "5", "5", "5", "5", "null"],
    )
    assert_equal(
        cells(holes, col("v").backward_fill(1)),
        [String("1"), "1", "null", "null", "5", "5", "null"],
    )
    assert_equal(cells(holes, col("v").forward_fill(0)), cells(holes, col("v")))


def test_rank_methods() raises:
    var df = DataFrame(
        [
            Series(
                "v",
                Column[Int64](
                    [20, 10, 20, 0, 30, 10],
                    [True, True, True, False, True, True],
                ),
            )
        ]
    )
    assert_equal(
        cells(df, col("v").rank()),
        [String("3.5"), "1.5", "3.5", "null", "5.0", "1.5"],
    )
    assert_equal(
        cells(df, col("v").rank("min")),
        [String("3"), "1", "3", "null", "5", "1"],
    )
    assert_equal(
        cells(df, col("v").rank("max")),
        [String("4"), "2", "4", "null", "5", "2"],
    )
    assert_equal(
        cells(df, col("v").rank("dense")),
        [String("2"), "1", "2", "null", "3", "1"],
    )
    assert_equal(
        cells(df, col("v").rank("ordinal")),
        [String("3"), "1", "4", "null", "5", "2"],
    )
    assert_equal(
        cells(df, col("v").rank("ordinal", descending=True)),
        [String("2"), "4", "3", "null", "1", "5"],
    )
    assert_equal(
        cells(frame(), col("f").rank("dense")),
        [String("3"), "5", "1", "null", "2", "4"],
    )
    assert_equal(
        cells(frame(), col("s").rank("min")),
        [String("3"), "5", "1", "3", "null", "2"],
    )
    with assert_raises(contains="rank method must be"):
        _ = df.select(col("v").rank("first"))


def test_rolling() raises:
    var df = DataFrame(
        [
            Series(
                "v",
                Column[Int64]([1, 2, 0, 4, 5], [True, True, False, True, True]),
            )
        ]
    )
    assert_equal(
        cells(df, col("v").rolling_sum(2)),
        [String("null"), "3", "null", "null", "9"],
    )
    assert_equal(
        cells(df, col("v").rolling_sum(2, 1)), [String("1"), "3", "2", "4", "9"]
    )
    assert_equal(
        cells(df, col("v").rolling_mean(3, 2)),
        [String("null"), "1.5", "1.5", "3.0", "4.5"],
    )
    assert_equal(
        cells(df, col("v").rolling_min(3, 1)), [String("1"), "1", "1", "2", "4"]
    )
    assert_equal(
        cells(df, col("v").rolling_max(3, 1)), [String("1"), "2", "2", "4", "5"]
    )
    # Windows larger than the input only ever see the available rows.
    assert_equal(cells(df, col("v").rolling_sum(99, 1))[4], "12")
    assert_equal(cells(df, col("v").rolling_sum(99))[4], "null")
    with assert_raises(contains="window_size must be at least 1"):
        _ = df.select(col("v").rolling_sum(0))
    with assert_raises(contains="rolling_sum require a numeric expression"):
        _ = frame().select(col("s").rolling_sum(2))


def test_over_partitions() raises:
    var df = frame()
    assert_equal(
        cells(df, col("x").cum_sum().over("k")),
        [String("3"), "1", "null", "5", "5", "2"],
    )
    assert_equal(
        cells(df, col("x").sum().over("k")),
        [String("5"), "5", "5", "5", "5", "2"],
    )
    assert_equal(
        cells(df, (col("x") - col("x").min()).over("k")),
        [String("1"), "0", "null", "3", "0", "0"],
    )
    assert_equal(
        cells(df, col("x").rank("ordinal").over("k")),
        [String("2"), "1", "null", "2", "1", "1"],
    )
    assert_equal(
        cells(df, col("x").shift().over(["k"])),
        [String("null"), "null", "3", "1", "null", "null"],
    )
    # Composite keys; null is its own key value.
    assert_equal(
        cells(df, col("x").count().over(["k", "s"])),
        [String("1"), "1", "0", "1", "1", "1"],
    )
    assert_equal(
        cells(df, lit(Int64(7)).over("k")), List[String](length=6, fill="7")
    )
    var grouped = df.group_by("k", maintain_order=True).agg(
        [
            col("x").cum_sum().max().alias("peak"),
            col("x").diff().min().alias("drop"),
        ]
    )
    assert_equal(cells(grouped, col("peak")), [String("5"), "5", "2"])
    assert_equal(cells(grouped, col("drop")), [String("null"), "3", "null"])
    with assert_raises(contains="Unknown partition column: nope"):
        _ = df.select(col("x").sum().over("nope"))
    with assert_raises(contains="Window operations require a row-valued input"):
        _ = df.select(col("x").sum().cum_sum())


def test_batch_independence_and_empty() raises:
    var df = frame()
    var exprs: List[Expr] = [
        col("x").cum_sum().alias("a"),
        col("f").rolling_mean(3, 1).alias("b"),
        col("x").rank().over("k").alias("c"),
        (col("x") - col("x").shift()).alias("d"),
        col("s").forward_fill().alias("e"),
    ]
    var reference = df.select_exprs(exprs, batch_size=64)
    for size in range(1, 7):
        assert_true(df.select_exprs(exprs, batch_size=size).equals(reference))
    assert_equal(df.clear().select_exprs(exprs).height(), 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
