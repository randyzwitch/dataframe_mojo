"""Aggregate lifting retains WHEN operands and guards conditional scalars."""
# contract-group: execution.backends
# contract-group: execution.apple_parity
from std.testing import assert_equal, assert_raises, assert_true
from dataframe import Column, DataFrame, DataType, Expr, Series, col, lit, when
from dataframe.lazy import LazyFrame
from dataframe.expr import SUM, WHEN
from dataframe_metal.accelerator.aggregates import (
    _conditional_reductions,
    lower_aggregate_compositions,
)


def source(select_bad: Bool = False) raises -> DataFrame:
    return DataFrame(
        [
            Series("x", Column[Int32]([2, 99999])).with_dtype(
                DataType.decimal(4, 0, 32)
            ),
            Series("one", Column[Int32]([1, 1])).with_dtype(
                DataType.decimal(4, 0, 32)
            ),
            Series("p", Column[Bool]([True, select_bad])),
        ]
    )


def operand() -> Expr:
    return (
        when(col("p"))
        .then((col("x") + col("one")).cast(DataType.FLOAT64))
        .otherwise(lit(0.0))
    )


def test_decimal_when_inside_sum_keeps_original_operand_masks() raises:
    var query = source().lazy().select((operand().sum() + lit(1.0)).alias("v"))
    var lowered = lower_aggregate_compositions(query)
    for planned in [query.copy(), lowered.copy()]:
        var actual = planned.collect(engine="cpu", optimize=False)
        assert_equal(actual.column("v").dtype(), DataType.FLOAT64)
        assert_equal(actual.column("v").float64()._get(0), Float64(4))
    var found = False
    for expression in lowered._nodes[len(lowered._nodes) - 2].exprs:
        assert_equal(expression._nodes[len(expression._nodes) - 1].op, SUM)
        for node in expression._nodes:
            found |= node.op == WHEN
    assert_true(found)
    var bad = (
        source(True).lazy().select((operand().sum() + lit(1.0)).alias("v"))
    )
    var lowered_bad = lower_aggregate_compositions(bad)
    for planned in [bad.copy(), lowered_bad.copy()]:
        with assert_raises(contains="decimal precision"):
            _ = planned.collect(engine="cpu", optimize=False)


def test_decimal_sum_inside_when_keeps_conditional_scalar_guard() raises:
    var expression = (
        when(lit(False))
        .then((col("x") + col("one")).cast(DataType.FLOAT64).sum())
        .otherwise(lit(7.0))
    )
    var query = source(True).lazy().select(expression.alias("v"))
    # CPU precomputes reduction states before its outer scalar WHEN.
    with assert_raises(contains="decimal precision"):
        _ = query.collect(engine="cpu", optimize=False)
    with assert_raises(contains="conditional checked decimal arithmetic"):
        _ = lower_aggregate_compositions(query)


def test_shared_reduction_retains_any_conditional_ancestor() raises:
    var leaf = operand().sum()
    var expression = leaf + when(lit(False)).then(leaf).otherwise(lit(0.0))
    var conditional = _conditional_reductions(expression)
    var guarded = 0
    var eager = 0
    for index in range(len(expression._nodes)):
        if expression._nodes[index].op == SUM:
            if conditional[index]:
                guarded += 1
            else:
                eager += 1
    assert_true(guarded > 0)
    assert_true(eager > 0)
    var query = source().lazy().select(expression.alias("v"))
    with assert_raises(contains="conditional checked decimal arithmetic"):
        _ = lower_aggregate_compositions(query)


def ratio_query() raises -> LazyFrame:
    var frame = DataFrame(
        [
            Series("price", Column[Int32]([100, 200, 300])).with_dtype(
                DataType.decimal(4, 0, 32)
            ),
            Series("discount", Column[Int32]([0, 5, 0])).with_dtype(
                DataType.decimal(2, 1, 32)
            ),
            Series("one", Column[Int32]([10, 10, 10])).with_dtype(
                DataType.decimal(2, 1, 32)
            ),
            Series("p", Column[Bool]([True, False, True])),
        ]
    )
    var discounted = col("price") * (col("one") - col("discount"))
    var numerator = (
        when(col("p"))
        .then(discounted.cast(DataType.FLOAT64))
        .otherwise(lit(0.0))
        .sum()
    )
    return frame.lazy().select(
        (
            lit(100.0) * numerator / discounted.sum().cast(DataType.FLOAT64)
        ).alias("ratio")
    )


def test_checked_decimal_conditional_ratio_literal() raises:
    # Discounted values are 100, 100, 300. Selected sum400 / total500.
    var query = ratio_query()
    var actual = query.collect(engine="cpu", optimize=False)
    assert_equal(actual.column("ratio").float64()._get(0), Float64(80))
    var lowered = lower_aggregate_compositions(query)
    var normalized = lowered.collect(engine="cpu", optimize=False)
    assert_equal(normalized.column("ratio").float64()._get(0), Float64(80))
