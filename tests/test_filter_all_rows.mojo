"""A filter whose mask keeps every row returns the frame's columns as they
are; any false or null in the mask, anywhere including a partial last
byte, filters as before. Masks that start mid-byte take the usual path."""
from std.testing import TestSuite, assert_equal, assert_true

from dataframe import Column, DataFrame, Series, col
from dataframe.bool_column import BoolColumn


def frame(n: Int) raises -> DataFrame:
    var xs = List[Int64](capacity=n)
    var names = List[String](capacity=n)
    for i in range(n):
        xs.append(Int64(i))
        names.append("row" + String(i))
    return DataFrame(
        [Series("x", Column[Int64](xs^)), Series("s", Column[String](names^))]
    )


def test_all_true_masks_keep_every_row() raises:
    for n in [1, 7, 8, 9, 64, 1003]:
        var data = frame(n)
        var kept = data.filter(col("x").is_not_null())
        assert_equal(kept.height(), n)
        assert_true(kept.equals(data))
        var sliced = data.slice(8, n - 8) if n > 8 else data.slice(0, n)
        assert_true(sliced.filter(col("x") >= 0).equals(sliced))


def test_one_false_or_null_anywhere_still_filters() raises:
    for n in [9, 64, 1003]:
        var data = frame(n)
        for drop in [0, n // 2, n - 1]:
            var out = data.filter(col("x") != drop)
            assert_equal(out.height(), n - 1)
            for i in range(out.height()):
                assert_true(out.item(i, "x").int64() != Int64(drop))
        # A null mask entry drops its row.
        var values = List[Bool](length=n, fill=True)
        var valid = List[Bool](length=n, fill=True)
        valid[n - 1] = False
        var out = data.filter(BoolColumn(values, valid))
        assert_equal(out.height(), n - 1)
    # A mask that starts mid-byte (a sliced frame) filters as before.
    var data = frame(100).slice(3, 90)
    assert_true(data.filter(col("x") >= 0).equals(data))
    assert_equal(data.filter(col("x") != 50).height(), 89)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
