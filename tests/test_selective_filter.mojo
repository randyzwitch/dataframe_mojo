"""A filter on `a & b & ...` evaluates each part only on the rows the
earlier parts kept (`DataFrame._filter_selective`). It must keep exactly
the rows the whole predicate keeps: nulls in any part drop the row, parts
reading strings run last, and an unselective part ends the narrowing.
"""
from std.testing import TestSuite, assert_equal, assert_true

from dataframe import Column, DataFrame, DataType, Expr, Series, col, lit

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


def test_decimal_parts_and_odd_offsets() raises:
    """Decimal comparisons under an AND's mask, and a frame whose rows start
    mid-byte."""
    var data = frame()
    var cents = List[Int128](capacity=ROWS)
    for i in range(ROWS):
        cents.append(Int128((i * 37) % 10_000))
    var priced = data.with_column(
        Series("d", Column[Int128](cents^)).with_dtype(DataType.decimal(15, 2))
    )
    var low = lit("10.00").cast(DataType.decimal(15, 2))
    var high = lit("40.00").cast(DataType.decimal(15, 2))
    check(priced, (col("b") >= 0) & col("d").is_between(low, high))
    check(priced, (col("a") == 7) & col("d").is_between(low, high))
    check(data.slice(3, ROWS - 10), (col("b") >= 0) & (col("a") < 50))


def test_fused_comparisons_every_type_operator_and_offset() raises:
    # Parts comparing a numeric column with a constant run together in one
    # pass (`fused_comparisons`); every width, each operator, the constant
    # on either side, NaN, a validity bitmap with no null, and windows that
    # start mid-byte and end mid-word.
    var n = 20_011
    var i8 = List[Int8](capacity=n)
    var u16 = List[UInt16](capacity=n)
    var i32 = List[Int32](capacity=n)
    var f32 = List[Float32](capacity=n)
    var f64 = List[Float64](capacity=n)
    var all_valid = List[Bool](capacity=n)
    var nan = Float64(0) / Float64(0)
    for i in range(n):
        i8.append(Int8(i % 200 - 100))
        u16.append(UInt16((i * 31) % 60_000))
        i32.append(Int32((i * 7) % 1000 - 500))
        f32.append(Float32(i % 97) / 4)
        f64.append(nan if i % 13 == 5 else Float64(i % 89))
        all_valid.append(True)
    var data = DataFrame(
        [
            Series("i8", Column[Int8](i8^)),
            Series("u16", Column[UInt16](u16^)),
            Series("i32", Column[Int32](i32^, all_valid^)),
            Series("f32", Column[Float32](f32^)),
            Series("f64", Column[Float64](f64^)),
        ]
    )
    var frames = [data.copy(), data.slice(3, n - 10), data.slice(77, 9_000)]
    for f in frames:
        check(
            f,
            (col("i8") >= lit(Int8(-20)))
            & (col("i8") < lit(Int8(60)))
            & (col("u16") != lit(UInt16(310))),
        )
        check(
            f,
            (lit(Int32(0)) < col("i32"))
            & (col("i32") <= lit(Int32(400)))
            & (col("f32") > lit(Float32(3.5))),
        )
        check(f, (col("f64") != 7.0) & (col("f64") <= 40.0))
        check(f, (col("f64") == 12.0) & (lit(100.0) > col("f64")))
        check(f, (col("i32") == lit(Int32(-3))) & (col("f64") >= 0.0))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
