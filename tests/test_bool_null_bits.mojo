"""Boolean `fill_null` and the null-keeping step of numeric `is_in` work on
bitmaps a byte at a time. Results must equal a row-by-row reference with
nulls on either side, a broadcast literal, and sliced inputs that start
mid-byte.
"""
from std.testing import TestSuite, assert_equal, assert_true

from dataframe import Column, DataFrame, Expr, Series, col, lit


def frame(n: Int) raises -> DataFrame:
    var flags = List[Bool](capacity=n)
    var flag_valid = List[Bool](capacity=n)
    var others = List[Bool](capacity=n)
    var other_valid = List[Bool](capacity=n)
    var xs = List[Int64](capacity=n)
    var x_valid = List[Bool](capacity=n)
    for i in range(n):
        flags.append(i % 3 == 0)
        flag_valid.append(i % 5 != 2)
        others.append(i % 4 == 1)
        other_valid.append(i % 7 != 6)
        xs.append(Int64(i % 9 - 2))
        x_valid.append(i % 11 != 4)
    return DataFrame(
        [
            Series("f", Column[Bool](flags^, flag_valid^)),
            Series("g", Column[Bool](others^, other_valid^)),
            Series("x", Column[Int64](xs^, x_valid^)),
        ]
    )


def check(data: DataFrame) raises:
    var sizes: List[Expr] = [Expr(-1), Expr(3), Expr(6)]
    var out = data.select_exprs(
        [
            col("f").fill_null(col("g")).alias("fg"),
            col("f").fill_null(lit(True)).alias("ft"),
            col("x").is_in(sizes).alias("in"),
        ]
    )
    for i in range(data.height()):
        var f = data.item(i, "f")
        var g = data.item(i, "g")
        var fg = out.item(i, "fg")
        if not f.is_null():
            assert_equal(fg.bool(), f.bool())
        elif g.is_null():
            assert_true(fg.is_null())
        else:
            assert_equal(fg.bool(), g.bool())
        var ft = out.item(i, "ft")
        assert_equal(ft.bool(), True if f.is_null() else f.bool())
        var x = data.item(i, "x")
        var found = out.item(i, "in")
        if x.is_null():
            assert_true(found.is_null())
        else:
            var v = x.int64()
            assert_equal(found.bool(), v == -1 or v == 3 or v == 6)


def test_bitmaps_match_rows() raises:
    check(frame(1000))


def test_slices_starting_mid_byte() raises:
    var data = frame(1000)
    check(data.slice(3, 500))
    check(data.slice(13, 7))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
