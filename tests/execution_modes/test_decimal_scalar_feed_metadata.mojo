"""No-device bounds for conditional checked Decimal Float64 SUM operands."""
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Expr, Series, col, lit, when
from dataframe.binding import BoundExpr, bind
from dataframe.accelerator.decimal_scalar_feed import eligible
from dataframe.accelerator.constants import fold_literal_text_parsing
from dataframe.expr import LIT_INT, MUL, SUB
from tests.execution_modes.test_conditional_decimal_aggregate_planning import (
    source,
    operand,
)


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


def test_decimal_money_literal_and_arithmetic_bound_metadata() raises:
    var frame = DataFrame(
        [
            Series("price", Column[Int64]([10000, 20000, 30000])).with_dtype(
                DataType.decimal(15, 2, 64)
            ),
            Series("discount", Column[Int64]([0, 50, 0])).with_dtype(
                DataType.decimal(15, 2, 64)
            ),
            Series("p", Column[Bool]([True, False, True])),
        ]
    )
    var one = lit("1").cast(DataType.decimal(15, 2))
    var discounted = col("price") * (one - col("discount"))
    var selected = (
        when(col("p"))
        .then(discounted.cast(DataType.FLOAT64))
        .otherwise(lit(0.0))
    )
    var query = frame.lazy().select(
        (
            lit(100.0)
            * selected.sum()
            / discounted.sum().cast(DataType.FLOAT64)
        ).alias("ratio")
    )
    var canonical = fold_literal_text_parsing(query)
    for planned in [query, canonical]:
        var actual = planned.collect(engine="cpu", optimize=False)
        assert_equal(actual.column("ratio").float64()._get(0), Float64(80))
    var bound = bind(canonical._nodes[1].exprs[0], frame._columns)
    var literal = False
    var subtraction = False
    var product = False
    for at in range(len(bound.expr._nodes)):
        ref node = bound.expr._nodes[at]
        if node.op == LIT_INT and bound.dtypes[at].is_decimal():
            assert_equal(bound.dtypes[at], DataType.decimal(15, 2))
            assert_equal(node.integer, Int64(100))
            literal = True
        elif node.op == SUB:
            assert_equal(bound.dtypes[at], DataType.decimal(16, 2, 64))
            subtraction = True
        elif node.op == MUL and bound.dtypes[at].is_decimal():
            assert_equal(bound.dtypes[at], DataType.decimal(18, 4, 64))
            product = True
    assert_true(literal and subtraction and product)


def test_conditional_decimal_sum_min_count_preserves_operand_faults() raises:
    var masked = source().lazy().select(operand().sum(min_count=3).alias("v"))
    var actual = masked.collect(engine="cpu", optimize=False).column("v")
    assert_equal(actual.dtype(), DataType.FLOAT64)
    assert_true(not actual.float64().is_valid(0))
    var bad = (
        source(True).lazy().select(operand().sum(min_count=999).alias("v"))
    )
    with assert_raises(contains="decimal precision"):
        _ = bad.collect(engine="cpu", optimize=False)
