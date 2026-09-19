"""min/max/mean/first/last/n_unique/std/var/median/quantile/len reductions."""
from std.math import isnan, sqrt
from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
    assert_false,
    assert_raises,
    assert_almost_equal,
)
from dataframe import DataType, Column, DataFrame, Expr, Series, col, lit
from dataframe.reductions import VarState

comptime MIN = Int64(-9223372036854775807) - 1
comptime MAX = Int64(9223372036854775807)


def nan() -> Float64:
    return Float64(0) / Float64(0)


def inf() -> Float64:
    return Float64(1) / Float64(0)


def frame() raises -> DataFrame:
    return DataFrame(
        [
            Series("k", Column[String](["a", "b", "a", "b", "a", "c", "b"])),
            Series(
                "i",
                Column[Int64](
                    [4, -2, 9, 0, 4, 1, 3],
                    [True, True, True, False, True, False, True],
                ),
            ),
            Series(
                "x",
                Column[Float64](
                    [2.5, nan(), -1, 0, 7, 3, 0.5],
                    [True, True, True, False, True, False, True],
                ),
            ),
            Series(
                "s",
                Column[String](
                    ["pear", "apple", "", "fig", "pear", "kiwi", "fig"],
                    [True, True, True, True, False, True, True],
                ),
            ),
            Series(
                "b",
                Column[Bool](
                    [True, False, True, True, False, True, False],
                    [True, True, False, True, True, True, True],
                ),
            ),
        ]
    )


def one(df: DataFrame, expr: Expr) raises -> Series:
    return df.select(expr.alias("r")).column("r")


def test_min_max_all_dtypes() raises:
    var df = frame()
    assert_equal(one(df, col("i").min()).int64().value(0), Int64(-2))
    assert_equal(one(df, col("i").max()).int64().value(0), Int64(9))
    assert_equal(one(df, col("s").min()).string().value(0), "")
    assert_equal(one(df, col("s").max()).string().value(0), "pear")
    assert_false(one(df, col("b").min()).bool().value(0))
    assert_true(one(df, col("b").max()).bool().value(0))
    # NaN sorts above every number: max is NaN, min ignores it.
    assert_equal(one(df, col("x").min()).float64().value(0), Float64(-1))
    assert_true(isnan(one(df, col("x").max()).float64().value(0)))
    var only_nan = DataFrame([Series("x", Column[Float64]([nan(), nan()]))])
    assert_true(isnan(one(only_nan, col("x").min()).float64().value(0)))
    var infinite = DataFrame([Series("x", Column[Float64]([inf(), -inf(), 1]))])
    assert_equal(one(infinite, col("x").min()).float64().value(0), -inf())
    assert_equal(one(infinite, col("x").max()).float64().value(0), inf())
    var extremes = DataFrame([Series("i", Column[Int64]([MAX, MIN, 0]))])
    assert_equal(one(extremes, col("i").min()).int64().value(0), MIN)
    assert_equal(one(extremes, col("i").max()).int64().value(0), MAX)
    var spread = df.select((col("i").max() - col("i").min()).alias("range"))
    assert_equal(spread.item().int64(), Int64(11))


def test_mean() raises:
    var df = frame()
    assert_equal(one(df, col("i").mean()).dtype(), DataType.FLOAT64)
    assert_almost_equal(
        one(df, col("i").mean()).float64().value(0), 18.0 / 5.0, rtol=1e-15
    )
    assert_true(isnan(one(df, col("x").mean()).float64().value(0)))
    var clean = df.filter(col("x").is_not_nan())
    assert_almost_equal(
        one(clean, col("x").mean()).float64().value(0), 9.0 / 4.0, rtol=1e-15
    )
    # Exact 128-bit totals: the mean of two MAX values is MAX, not overflow.
    var big = DataFrame([Series("i", Column[Int64]([MAX, MAX, MAX]))])
    assert_equal(one(big, col("i").mean()).float64().value(0), Float64(MAX))
    var cancel = DataFrame([Series("i", Column[Int64]([MAX, MIN, MAX, MIN]))])
    assert_equal(one(cancel, col("i").mean()).float64().value(0), -0.5)


def test_first_last_are_order_dependent_and_keep_nulls() raises:
    var df = frame()
    assert_equal(one(df, col("i").first()).int64().value(0), Int64(4))
    assert_equal(one(df, col("i").last()).int64().value(0), Int64(3))
    assert_true(one(df.head(6), col("i").last()).int64().is_null(0))
    assert_equal(one(df, col("s").first()).string().value(0), "pear")
    assert_true(one(df.head(5), col("s").last()).string().is_null(0))
    assert_true(one(df.reverse(), col("b").last()).bool().value(0))
    assert_true(
        one(df, col("x").first() + col("x").last()).float64().value(0) == 3.0
    )


def test_n_unique_counts_null_once_and_canonicalizes_floats() raises:
    var df = frame()
    assert_equal(one(df, col("i").n_unique()).int64().value(0), Int64(5))
    assert_equal(one(df, col("s").n_unique()).int64().value(0), Int64(6))
    assert_equal(one(df, col("b").n_unique()).int64().value(0), Int64(3))
    assert_equal(one(df, col("k").n_unique()).int64().value(0), Int64(3))
    var floats = DataFrame(
        [
            Series(
                "x",
                Column[Float64](
                    [nan(), -nan(), 0.0, -0.0, 1, 1, 5],
                    [True, True, True, True, True, True, False],
                ),
            )
        ]
    )
    assert_equal(one(floats, col("x").n_unique()).int64().value(0), Int64(4))


def test_std_and_var() raises:
    var df = DataFrame(
        [
            Series(
                "v",
                Column[Float64](
                    [2, 4, 4, 4, 5, 5, 7, 9, 0],
                    [True, True, True, True, True, True, True, True, False],
                ),
            )
        ]
    )
    assert_almost_equal(
        one(df, col("v").var(ddof=0)).float64().value(0), 4.0, rtol=1e-12
    )
    assert_almost_equal(
        one(df, col("v").std(ddof=0)).float64().value(0), 2.0, rtol=1e-12
    )
    assert_almost_equal(
        one(df, col("v").var()).float64().value(0), 32.0 / 7.0, rtol=1e-12
    )
    var single = df.head(1)
    assert_true(one(single, col("v").std()).float64().is_null(0))
    assert_equal(one(single, col("v").std(ddof=0)).float64().value(0), 0.0)
    assert_true(one(df.clear(), col("v").var(ddof=0)).float64().is_null(0))
    var ints = DataFrame([Series("i", Column[Int64]([1, 2, 3, 4]))])
    assert_almost_equal(
        one(ints, col("i").var()).float64().value(0), 5.0 / 3.0, rtol=1e-12
    )
    var with_inf = DataFrame([Series("v", Column[Float64]([1, inf()]))])
    assert_true(isnan(one(with_inf, col("v").var()).float64().value(0)))
    with assert_raises(contains="ddof must be nonnegative"):
        _ = df.select(col("v").std(ddof=-1))
    with assert_raises(contains="std requires a numeric expression"):
        _ = frame().select(col("s").std())


def test_median_and_quantile_interpolation() raises:
    var df = DataFrame(
        [
            Series(
                "v",
                Column[Int64](
                    [10, 1, 7, 3, 0], [True, True, True, True, False]
                ),
            )
        ]
    )
    # Valid values sorted: 1, 3, 7, 10.
    assert_equal(one(df, col("v").median()).float64().value(0), 5.0)
    var q = Float64(0.4)  # position 1.2 between 3 and 7
    assert_almost_equal(
        one(df, col("v").quantile(q)).float64().value(0), 3.8, rtol=1e-12
    )
    assert_equal(one(df, col("v").quantile(q, "lower")).float64().value(0), 3.0)
    assert_equal(
        one(df, col("v").quantile(q, "higher")).float64().value(0), 7.0
    )
    assert_equal(
        one(df, col("v").quantile(q, "nearest")).float64().value(0), 3.0
    )
    assert_equal(
        one(df, col("v").quantile(q, "midpoint")).float64().value(0), 5.0
    )
    assert_equal(one(df, col("v").quantile(0)).float64().value(0), 1.0)
    assert_equal(one(df, col("v").quantile(1)).float64().value(0), 10.0)
    assert_equal(
        one(df, col("v").quantile(Float64(1) / 3, "midpoint"))
        .float64()
        .value(0),
        3.0,
    )
    var odd = df.head(3)
    assert_equal(one(odd, col("v").median()).float64().value(0), 7.0)
    assert_true(one(df.clear(), col("v").median()).float64().is_null(0))
    var nans = DataFrame([Series("x", Column[Float64]([1, nan(), 3]))])
    assert_equal(one(nans, col("x").median()).float64().value(0), 3.0)
    assert_true(isnan(one(nans, col("x").quantile(1)).float64().value(0)))
    with assert_raises(contains="quantile must be between 0 and 1"):
        _ = df.select(col("v").quantile(1.5))
    with assert_raises(contains="interpolation must be"):
        _ = df.select(col("v").quantile(0.5, "cubic"))


def test_len_counts_nulls() raises:
    var df = frame()
    assert_equal(one(df, col("i").len()).int64().value(0), Int64(7))
    assert_equal(one(df, col("i").count()).int64().value(0), Int64(5))
    assert_equal(one(df.clear(), col("i").len()).int64().value(0), Int64(0))


def test_empty_and_all_null_inputs() raises:
    var empty = frame().clear()
    var nulls = frame().filter(col("i").is_null())
    for df in [empty.copy(), nulls.copy()]:
        var result = df.select_exprs(
            [
                col("i").min().alias("min"),
                col("i").max().alias("max"),
                col("i").mean().alias("mean"),
                col("x").median().alias("median"),
                col("x").std().alias("std"),
                col("x").min().alias("xmin"),
            ]
        )
        for name in result.columns():
            assert_true(result.item(0, name).is_null(), msg=name)
    assert_equal(one(empty, col("i").n_unique()).int64().value(0), Int64(0))
    assert_equal(one(nulls, col("i").n_unique()).int64().value(0), Int64(1))
    assert_true(one(empty, col("i").first()).int64().is_null(0))


def test_grouped_matches_global_per_group() raises:
    var df = frame()
    var exprs: List[Expr] = [
        col("i").min().alias("imin"),
        col("x").max().alias("xmax"),
        col("i").mean().alias("mean"),
        col("s").first().alias("first"),
        col("b").last().alias("last"),
        col("s").n_unique().alias("nu"),
        col("x").var(ddof=0).alias("var"),
        col("i").median().alias("median"),
        col("x").quantile(0.75, "higher").alias("q"),
        col("i").len().alias("len"),
        (col("i").max() - col("i").min()).alias("range"),
    ]
    for size in [1, 2, 7, 64]:
        var grouped = df.group_by("k", maintain_order=True).agg(
            exprs, batch_size=size
        )
        assert_equal(grouped.height(), 3)
        for g in range(grouped.height()):
            var key = grouped.item(g, "k").string()
            var subset = df.filter(col("k").eq(lit(key)))
            var expected = subset.select_exprs(exprs, batch_size=size)
            for name in expected.columns():
                assert_true(
                    grouped.item(g, name) == expected.item(0, name),
                    msg=key + " " + name,
                )


def test_var_state_merge_matches_whole() raises:
    var values: List[Float64] = [3.5, -2, 1e6, 7, 0.25, -9, 4, 4, 1e-3]
    var whole = VarState()
    for v in values:
        whole.add(v)
    for split in range(len(values) + 1):
        for second in range(split, len(values) + 1):
            var parts = [VarState(), VarState(), VarState()]
            for i in range(len(values)):
                var part = 0 if i < split else (1 if i < second else 2)
                parts[part].add(values[i])
            # Merge in a different association order each time.
            var merged = parts[2].copy()
            merged.merge(parts[0])
            merged.merge(parts[1])
            assert_equal(merged.count, whole.count)
            assert_almost_equal(merged.mean, whole.mean, rtol=1e-12)
            assert_almost_equal(
                merged.variance(1).value(),
                whole.variance(1).value(),
                rtol=1e-10,
            )
    var empty = VarState()
    empty.merge(VarState())
    assert_false(Bool(empty.variance(0)))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
