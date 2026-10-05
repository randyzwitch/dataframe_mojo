"""Means and distinct counts of decimals at every storage width.

A mean adds up the same total as a sum, so it must not be held to the
column's own precision. A distinct count compares scaled integers. Both
were found by TPC-DS q28 on decimal data: the ungrouped mean of a
decimal(7, 2) column raised once its total passed 99,999.99, and the
distinct count crashed.
"""
from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from dataframe import Column, DataFrame, DataType, Expr, Series, col

comptime ROWS = 6_000


def prices() raises -> DataFrame:
    """Prices up to 999.99 whose total far exceeds seven digits, a third
    of them repeated, with nulls."""
    var cents32 = List[Int32](capacity=ROWS)
    var cents64 = List[Int64](capacity=ROWS)
    var cents128 = List[Int128](capacity=ROWS)
    var valid = List[Bool](capacity=ROWS)
    var keys = List[Int64](capacity=ROWS)
    for i in range(ROWS):
        var value = (i * 7919) % 2_000 * 50 - 100
        cents32.append(Int32(value))
        cents64.append(Int64(value))
        cents128.append(Int128(value))
        valid.append(i % 13 != 4)
        keys.append(Int64(i % 5))
    return DataFrame(
        [
            Series("k", Column[Int64](keys^)),
            Series("p32", Column[Int32](cents32^, valid.copy())).with_dtype(
                DataType.decimal(7, 2, 32)
            ),
            Series(
                "p64", Column[Int64](cents64.copy(), valid.copy())
            ).with_dtype(DataType.decimal(15, 2, 64)),
            Series("p128", Column[Int128](cents128^, valid.copy())).with_dtype(
                DataType.decimal(20, 2)
            ),
            Series("f", Column[Int64](cents64^, valid^)),
        ]
    )


def close(a: Float64, b: Float64) -> Bool:
    return abs(a - b) <= 1e-9 * max(1.0, abs(b))


def test_means_do_not_overflow_the_column_precision() raises:
    var data = prices()
    # The same numbers as plain integers of cents.
    var want = data.select(col("f").mean()).item().float64() / 100
    for name in ["p32", "p64", "p128"]:
        var eager = data.select(col(name).mean().alias("m")).item().float64()
        assert_true(close(eager, want), name)
        var lazy = (
            data.lazy().select(col(name).mean().alias("m")).collect().item()
        )
        assert_true(close(lazy.float64(), want), name)
    var by_key = data.group_by("k", maintain_order=True).agg(
        [
            col("p32").mean().alias("a"),
            col("p128").mean().alias("b"),
            (col("f").mean() / 100.0).alias("w"),
        ]
    )
    for g in range(by_key.height()):
        var expected = by_key.column("w").float64()._get(g)
        assert_true(close(by_key.column("a").float64()._get(g), expected))
        assert_true(close(by_key.column("b").float64()._get(g), expected))


def test_distinct_counts_match_the_scaled_integers() raises:
    var data = prices()
    # n_unique counts null once, for decimals as for integers.
    var want = data.select(col("f").n_unique()).item().int64()
    for name in ["p32", "p64", "p128"]:
        assert_equal(
            data.select(col(name).n_unique().alias("u")).item().int64(), want
        )
        assert_equal(
            data.lazy()
            .select(col(name).n_unique().alias("u"))
            .collect()
            .item()
            .int64(),
            want,
        )
    var aggs: List[Expr] = [
        col("p32").n_unique().alias("a"),
        col("p64").n_unique().alias("b"),
        col("p128").n_unique().alias("c"),
        col("f").n_unique().alias("w"),
    ]
    var grouped = data.group_by("k", maintain_order=True).agg(aggs)
    var lazy = data.lazy().group_by(["k"]).agg(aggs).sort(["k"]).collect()
    for frame in [grouped.sort(["k"]), lazy.copy()]:
        for name in ["a", "b", "c"]:
            assert_true(
                frame.column(name).equals(frame.column("w").renamed(name)), name
            )
    # Without nulls, and after a filter.
    var kept = data.filter(col("p32").is_not_null())
    assert_equal(
        kept.select(col("p32").n_unique().alias("u")).item().int64(), want - 1
    )


def test_a_decimal_beyond_64_bits_is_reported_not_miscounted() raises:
    var huge = Int128(Int64.MAX) * 1_000
    var data = DataFrame(
        [
            Series("p", Column[Int128]([huge, huge, 5])).with_dtype(
                DataType.decimal(38, 2)
            )
        ]
    )
    with assert_raises(contains="beyond 64 bits is not supported"):
        _ = data.select(col("p").n_unique())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
