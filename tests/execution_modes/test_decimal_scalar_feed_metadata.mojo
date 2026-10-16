"""No-device bounds for conditional checked Decimal Float64 SUM operands."""
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, DataType, Expr, Series, col, lit, when
from dataframe.binding import BoundExpr, bind
from dataframe.accelerator.decimal_scalar_feed import eligible


def fixture() raises -> DataFrame:
    return DataFrame(
        [
            Series("x", Column[Int64]([100, Int64.MAX])).with_dtype(
                DataType.decimal(15, 2, 64)
            ),
            Series("d", Column[Int64]([0, 0])).with_dtype(
                DataType.decimal(15, 2, 64)
            ),
            Series("one", Column[Int64]([100, 100])).with_dtype(
                DataType.decimal(15, 2, 64)
            ),
            Series("p", Column[Bool]([True, False])),
        ]
    )


def bounds(expressions: List[Expr], frame: DataFrame) raises -> List[BoundExpr]:
    var result = List[BoundExpr]()
    for expression in expressions:
        result.append(bind(expression, frame._columns))
    return result^


def test_decimal_scalar_feed_reuses_physical_bounds_for_multiple_sites() raises:
    var frame = fixture()
    var discounted = col("x") * (col("one") - col("d"))
    var numerator = (
        when(col("p"))
        .then(discounted.cast(DataType.FLOAT64))
        .otherwise(lit(0.0))
    )
    var expressions = bounds([numerator.sum(), discounted.sum()], frame)
    assert_true(eligible(expressions, frame._columns))
    var product = bind(discounted, frame._columns)
    assert_equal(
        product.dtypes[len(product.dtypes) - 1], DataType.decimal(18, 4, 64)
    )
    # Int64.MAX is outside the tagged precision. Eligibility must retain its
    # physical-word proof rather than assuming the source has only15digits.
    assert_true(eligible(bounds([numerator.sum()], frame), frame._columns))


def test_decimal_scalar_feed_rejects_foreign_faults_and_wide_raw_sources() raises:
    var frame = fixture()
    var foreign = col("x").cast(DataType.INT8).cast(DataType.FLOAT64).sum()
    assert_true(not eligible(bounds([foreign], frame), frame._columns))
    var wide = DataFrame(
        [
            Series("x", Column[Int128]([1])).with_dtype(
                DataType.decimal(15, 2)
            ),
            Series("one", Column[Int64]([100])).with_dtype(
                DataType.decimal(15, 2, 64)
            ),
        ]
    )
    var wide_sum = ((col("x") * col("one")).cast(DataType.FLOAT64)).sum()
    assert_true(not eligible(bounds([wide_sum], wide), wide._columns))
    var mixed = (
        when(col("p"))
        .then(col("x"))
        .otherwise(col("d").cast(DataType.decimal(15, 4, 64)))
    )
    var mixed_sum = ((mixed * col("one")).cast(DataType.FLOAT64)).sum()
    assert_true(not eligible(bounds([mixed_sum], frame), frame._columns))
