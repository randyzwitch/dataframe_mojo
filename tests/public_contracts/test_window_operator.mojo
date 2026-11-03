"""The window operator (hash-partitioned, sorted per hash group, with
partition and peer boundaries): rank over a two-column order through a
struct, row numbers over many partitions, and framed rolling aggregates
from the segment tree, eagerly and lazily, against Polars' answers and
scalar models."""
# contract-group: expr.ordered_over
from std.testing import assert_equal, assert_true

from dataframe import Column, DataFrame, Expr, Series, as_struct, col, lit
from dataframe.row_expressions import len as row_count, int_range


def _frame() raises -> DataFrame:
    return DataFrame(
        [
            Series("g", Column[Int64]([1, 0, 1, 0, 1, 0, 1, 1])),
            Series(
                "a",
                Column[Int64](
                    [2, 1, 1, 1, 2, 0, 1, 3],
                    [True, True, True, True, True, False, True, True],
                ),
            ),
            Series(
                "b",
                Column[String](
                    ["x", "y", "y", "y", "w", "z", "y", ""],
                    [True, True, True, True, True, True, True, False],
                ),
            ),
        ]
    )


def test_rank_over_a_struct_orders_by_both_fields() raises:
    var key = as_struct([col("a"), col("b")])
    var expressions: List[Expr] = [
        key.copy().rank("min").over("g").alias("m"),
        key.copy().rank("dense", descending=True).over("g").alias("d"),
        key.copy().rank("ordinal").over("g").alias("o"),
    ]
    # Polars 2.0: pl.struct("a", "b").rank(...).over("g").
    var expected: List[List[Int64]] = [
        [4, 2, 1, 2, 3, 1, 1, 5],
        [2, 1, 4, 1, 3, 2, 4, 1],
        [4, 2, 1, 3, 3, 1, 2, 5],
    ]
    var eager = _frame().select_exprs(expressions)
    var lazy = _frame().lazy().select_exprs(expressions).collect()
    var names: List[String] = ["m", "d", "o"]
    for frame in [eager^, lazy^]:
        for c in range(3):
            for row in range(8):
                assert_equal(
                    frame.item(row, names[c]).int64(), expected[c][row]
                )


def test_row_numbers_over_many_partitions_follow_the_order_key() raises:
    comptime ROWS = 6_000
    var g = List[Int64]()
    var t = List[Int64]()
    for row in range(ROWS):
        g.append(Int64((row * 7) % 2_999))
        t.append(Int64((row * 7919) % ROWS))
    var input = DataFrame(
        [
            Series("g", Column[Int64](g.copy())),
            Series("t", Column[Int64](t.copy())),
        ]
    )
    var number = (int_range(row_count()) + lit(Int64(1))).over(
        "g", order_by="t"
    )
    var eager = input.with_columns(number.copy().alias("n"))
    var lazy = input.lazy().with_columns(number.alias("n")).collect()
    for frame in [eager^, lazy^]:
        for row in range(ROWS):
            var expected = Int64(1)
            for other in range(ROWS):
                if g[other] == g[row] and t[other] < t[row]:
                    expected += 1
            assert_equal(frame.item(row, "n").int64(), expected)


def test_rolling_frames_match_a_scalar_model() raises:
    comptime ROWS = 3_000
    var g = List[Int64]()
    var x = List[Int64]()
    var valid = List[Bool]()
    for row in range(ROWS):
        g.append(Int64(row % 7))
        x.append(Int64((row * 37) % 101 - 50))
        valid.append(row % 13 != 5)
    var input = DataFrame(
        [
            Series("g", Column[Int64](g.copy())),
            Series("x", Column[Int64](x.copy(), valid.copy())),
        ]
    )
    var expressions: List[Expr] = [
        col("x").rolling_sum(11, min_samples=1).over("g").alias("sum"),
        col("x").rolling_min(5, min_samples=3).over("g").alias("min"),
        col("x").rolling_max(40).over("g").alias("max"),
        col("x").rolling_mean(3, min_samples=1).over("g").alias("mean"),
    ]
    var eager = input.with_columns(expressions)
    var lazy = input.lazy().with_columns(expressions).collect()
    for frame in [eager^, lazy^]:
        for row in range(ROWS):
            # The partition's rows up to this one, in row order.
            var before = List[Int]()
            for other in range(row + 1):
                if g[other] == g[row]:
                    before.append(other)
            var windows: List[Int] = [11, 5, 40, 3]
            var needed: List[Int] = [1, 3, 40, 1]
            var names: List[String] = ["sum", "min", "max", "mean"]
            for c in range(4):
                var total = Int64(0)
                var low = Int64.MAX
                var high = Int64.MIN
                var count = 0
                for k in range(max(0, len(before) - windows[c]), len(before)):
                    var at = before[k]
                    if not valid[at]:
                        continue
                    count += 1
                    total += x[at]
                    low = min(low, x[at])
                    high = max(high, x[at])
                var value = frame.item(row, names[c])
                assert_equal(value.is_null(), count < needed[c] or count == 0)
                if value.is_null():
                    continue
                if c == 0:
                    assert_equal(value.int64(), total)
                elif c == 1:
                    assert_equal(value.int64(), low)
                elif c == 2:
                    assert_equal(value.int64(), high)
                else:
                    assert_true(
                        abs(value.float64() - Float64(total) / Float64(count))
                        < 1e-9
                    )
