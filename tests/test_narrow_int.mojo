"""Narrow-integer arithmetic in SIMD blocks and direct reductions of
computed batches (#384). Results must equal row-by-row checked arithmetic;
an overflow anywhere in a block raises the same error; columns with nulls
keep the per-row path; and `sum` of a computed Int16 column is exact.
"""
from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from dataframe import Column, DataFrame, Expr, Series, col, lit


def frame(rows: Int) raises -> DataFrame:
    var a = List[Int16](capacity=rows)
    var b = List[Int8](capacity=rows)
    var c = List[Int32](capacity=rows)
    for i in range(rows):
        a.append(Int16((i * 37) % 2000 - 1000))
        b.append(Int8((i * 11) % 200 - 100))
        c.append(Int32((i * 7919) % 200000 - 100000))
    return DataFrame(
        [
            Series("a", Column[Int16](a^)),
            Series("b", Column[Int8](b^)),
            Series("c", Column[Int32](c^)),
        ]
    )


def test_block_results_match_rows() raises:
    var data = frame(1003)
    var out = data.select_exprs(
        [
            (col("a") + lit(Int16(7))).alias("add"),
            (col("a") - col("a")).alias("zero"),
            (col("b") * lit(Int8(1))).alias("mul"),
            (col("c") * lit(Int32(3))).alias("mul32"),
        ]
    )
    for i in range(data.height()):
        var a = Int(data.column("a").get(i).int16())
        assert_equal(Int(out.column("add").get(i).int16()), a + 7)
        assert_equal(Int(out.column("zero").get(i).int16()), 0)
        assert_equal(
            Int(out.column("mul").get(i).int8()),
            Int(data.column("b").get(i).int8()),
        )
        assert_equal(
            Int(out.column("mul32").get(i).int32()),
            3 * Int(data.column("c").get(i).int32()),
        )


def test_overflow_raises_the_checked_error() raises:
    var values = List[Int16](length=100, fill=1)
    values[37] = 32760
    var data = DataFrame([Series("a", Column[Int16](values^))])
    with assert_raises(contains="int16 addition overflow"):
        _ = data.select_exprs([col("a") + lit(Int16(10))])
    # A tail row past the last full block.
    var tail = List[Int16](length=33, fill=1)
    tail[32] = -32768
    var short = DataFrame([Series("a", Column[Int16](tail^))])
    with assert_raises(contains="int16 subtraction overflow"):
        _ = short.select_exprs([col("a") - lit(Int16(1))])


def test_nulls_keep_the_row_path() raises:
    var data = DataFrame(
        [
            Series(
                "a",
                Column[Int16](
                    [Int16(1), Int16(32767), Int16(3)], [True, False, True]
                ),
            )
        ]
    )
    var out = data.select_exprs([col("a") + lit(Int16(1))])
    assert_true(out.column("a").get(1).is_null())
    assert_equal(Int(out.column("a").get(2).int16()), 4)


def test_sum_of_computed_narrow_column() raises:
    var data = frame(100_000)
    var exprs = List[Expr]()
    var want = List[Int]()
    for k in range(5):
        exprs.append((col("a") + lit(Int16(k))).sum().alias("s" + String(k)))
    var out = data.select_exprs(exprs)
    var base = 0
    for i in range(data.height()):
        base += Int(data.column("a").get(i).int16())
    for k in range(5):
        assert_equal(
            out.column("s" + String(k)).get(0).int64(),
            Int64(base + k * data.height()),
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
