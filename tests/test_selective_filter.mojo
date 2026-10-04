"""A filter on `a & b & ...` evaluates each part only on the rows the
earlier parts kept (`DataFrame._filter_selective`). It must keep exactly
the rows the whole predicate keeps: nulls in any part drop the row, parts
reading strings run last, and an unselective part ends the narrowing.
"""
from std.testing import TestSuite, assert_equal, assert_true

from dataframe import Column, DataFrame, Expr, Series, col, lit

comptime ROWS = 30_000


def frame() raises -> DataFrame:
    var a = List[Int64](capacity=ROWS)
    var a_valid = List[Bool](capacity=ROWS)
    var b = List[Float64](capacity=ROWS)
    var s = List[String](capacity=ROWS)
    var s_valid = List[Bool](capacity=ROWS)
    var row = List[Int64](capacity=ROWS)
    var nan = Float64(0) / Float64(0)
    for i in range(ROWS):
        a.append(Int64((i * 7919) % 100))
        a_valid.append(i % 37 != 3)
        b.append(nan if i % 41 == 9 else Float64(i % 53))
        s.append("" if i % 5 == 0 else "text" + String(i % 9))
        s_valid.append(i % 29 != 1)
        row.append(Int64(i))
    return DataFrame(
        [
            Series("a", Column[Int64](a^, a_valid^)),
            Series("b", Column[Float64](b^)),
            Series("s", Column[String](s^, s_valid^)),
            Series("row", Column[Int64](row^)),
        ]
    )


def check(data: DataFrame, predicate: Expr) raises:
    """The filter against the whole predicate's mask applied at once."""
    var got = data.filter(predicate)
    var mask = data.select_exprs([predicate.alias("keep")]).column("keep")
    var want = data.filter(mask.bool())
    assert_equal(got.height(), want.height())
    assert_true(got.equals(want), "selective filter kept other rows")


def test_selective_and_unselective_parts() raises:
    var data = frame()
    # Selective first (a == 7 keeps 1%), then a string part.
    check(data, (col("a") == 7) & (col("b") > 10) & col("s").ne(""))
    # Unselective first: the rest run together.
    check(data, (col("b") >= 0) & (col("a") < 50) & (col("s") == "text3"))
    # Selective, then unselective with a string part still to come: the
    # last two run together on the narrowed rows.
    check(data, (col("a") < 10) & (col("b") > 3) & col("s").ne(""))
    # Numbers only: a fifth kept is too many to narrow; 1% is not.
    check(data, (col("a") < 20) & (col("b") < 5))
    check(data, (col("a") == 7) & (col("b") < 5) & (col("row") % 2 == 0))
    # The string part is written first and still runs last.
    check(data, col("s").ne("") & (col("a") > 90))
    # A part reading no column.
    check(data, (col("a") == 3) & lit(True))
    check(data, (col("a") == 3) & lit(False))
    # Nested ANDs and an OR inside one part.
    check(
        data,
        ((col("a") < 20) & ((col("b") < 5) | col("s").is_null()))
        & (col("row") > 100),
    )


def test_chunked_and_small_frames() raises:
    var data = frame()
    var columns = List[Series]()
    for name in data.columns():
        columns.append(
            Series._from_chunks(
                [
                    data.column(name).slice(0, 12_345),
                    data.column(name).slice(12_345, ROWS - 12_345),
                ]
            )
        )
    var pieces = DataFrame(columns^)
    check(pieces, (col("a") == 7) & (col("b") > 10) & col("s").ne(""))
    # Below the row threshold the predicate runs whole.
    check(data.slice(0, 1000), (col("a") == 7) & col("s").ne(""))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
