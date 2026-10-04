"""Decimals keep the width their source declares (Arrow decimal32,
decimal64, decimal128). Values, nulls, display, CSV, casts, filters, sorts,
grouping and joins work at every width; arithmetic and sums compute at 128
bits and return decimal128, so a result never overflows its storage.
"""
from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from dataframe import Column, DataFrame, DataType, Series, col, to_csv_string


def prices() raises -> DataFrame:
    var cents: List[Int64] = [Int64(1999), -250, 100000, 1999, 0]
    var valid: List[Bool] = [True, True, True, True, False]
    var counts: List[Int32] = [Int32(15), 7, 3, 15, 1]
    return DataFrame(
        [
            Series("price", Column[Int64](cents^, valid^)).with_dtype(
                DataType.decimal(15, 2, 64)
            ),
            Series("qty", Column[Int32](counts^)).with_dtype(
                DataType.decimal(9, 1, 32)
            ),
            Series("k", Column[Int64]([Int64(1), 2, 1, 1, 2])),
        ]
    )


def test_types_and_limits() raises:
    assert_equal(DataType.decimal(18, 4, 64).name(), "decimal64[18,4]")
    assert_equal(DataType.decimal(9, 0, 32).name(), "decimal32[9,0]")
    assert_equal(DataType.decimal(20, 2).name(), "decimal[20,2]")
    assert_true(DataType.parse("decimal32[9,0]") == DataType.decimal(9, 0, 32))
    assert_true(DataType.decimal(18, 4, 64) != DataType.decimal(18, 4))
    with assert_raises(contains="precision must be between 1 and 18"):
        _ = DataType.decimal(19, 2, 64)
    with assert_raises(contains="precision must be between 1 and 9"):
        _ = DataType.decimal(10, 2, 32)
    with assert_raises(contains="width must be 32, 64 or 128"):
        _ = DataType.decimal(10, 2, 16)


def test_values_display_and_csv() raises:
    var data = prices()
    assert_equal(String(data.column("price").get(0)), "19.99")
    assert_equal(String(data.column("price").get(1)), "-2.50")
    assert_true(data.column("price").get(4).is_null())
    assert_equal(String(data.column("qty").get(0)), "1.5")
    var text = to_csv_string(data.select(["price", "qty"]))
    assert_true(text.startswith("price,qty\n19.99,1.5\n-2.50,0.7\n"))


def test_arithmetic_and_sums_widen() raises:
    var data = prices()
    var out = data.select_exprs(
        [
            (col("price") * col("qty")).alias("total"),
            (col("price") + col("price")).alias("twice"),
        ]
    )
    assert_true(out.column("total").dtype().decimal_width() == 128)
    # The product keeps the larger scale, rounding half to even.
    assert_equal(String(out.column("total").get(0)), "29.98")
    assert_equal(String(out.column("twice").get(2)), "2000.00")
    var sums = data.select_exprs([col("price").sum().alias("s")])
    assert_true(sums.column("s").dtype().decimal_width() == 128)
    assert_equal(String(sums.column("s").get(0)), "1037.48")
    var grouped = data.group_by("k", maintain_order=True).agg(
        [col("price").sum().alias("s"), col("qty").max().alias("m")]
    )
    assert_equal(String(grouped.column("s").get(0)), "1039.98")
    assert_equal(String(grouped.column("s").get(1)), "-2.50")
    assert_equal(String(grouped.column("m").get(0)), "1.5")


def test_rows_keep_their_width() raises:
    var data = prices()
    var kept = data.filter(col("price") > col("qty"))
    assert_equal(kept.height(), 3)
    assert_true(kept.column("price").dtype() == DataType.decimal(15, 2, 64))
    var ordered = data.sort("price")
    assert_true(ordered.column("price").dtype().decimal_width() == 64)
    assert_equal(String(ordered.column("price").get(1)), "19.99")
    var by_price = data.group_by("price", maintain_order=True).agg(
        [col("k").len().alias("n")]
    )
    assert_equal(by_price.height(), 4)
    assert_true(by_price.column("price").dtype().decimal_width() == 64)
    var cast = data.column("qty").cast(DataType.decimal(18, 2, 64))
    assert_equal(String(cast.get(0)), "1.50")
    assert_true(cast.dtype().decimal_width() == 64)
    var back = cast.cast(DataType.FLOAT64)
    assert_equal(back.get(2).float64(), 0.3)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
