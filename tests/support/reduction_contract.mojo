"""Shared engine invocation and direct scalar assertions for integer reductions."""
from std.math import sqrt, abs, isnan
from std.testing import assert_equal, assert_true
from dataframe import DataFrame, DataType, Expr, col
from tests.support.reference_model import ReferenceTable, reference_groups
from tests.support.reference_statistics import (
    sorted_integers,
    integer_quantile,
    integer_variance,
)
from tests.support.coverage_compare import assert_reference_cell


def integer_reductions() -> List[Expr]:
    return [
        col("i").sum().alias("sum"),
        col("i").count().alias("count"),
        col("i").len().alias("len"),
        col("i").min().alias("min"),
        col("i").max().alias("max"),
        col("i").mean().alias("mean"),
        col("i").var().alias("var"),
        col("i").std().alias("std"),
        col("i").median().alias("median"),
        col("i").first().alias("first"),
        col("i").last().alias("last"),
        col("i").n_unique().alias("distinct"),
    ]


def assert_statistic(actual: Float64, expected: Float64) raises:
    # Generated integer inputs are bounded by 15. Two-pass and Welford formulas
    # may reassociate; 1e-12 absolute + 1e-12 relative is the declared budget.
    assert_true(not isnan(actual))
    assert_true(abs(actual - expected) <= 1e-12 + 1e-12 * abs(expected))


def assert_integer_reductions(
    actual: DataFrame, output_row: Int, model: ReferenceTable
) raises:
    for name in [
        "sum",
        "count",
        "len",
        "min",
        "max",
        "first",
        "last",
        "distinct",
    ]:
        assert_true(actual.column(name).dtype() == DataType.INT64)
    for name in ["mean", "var", "std", "median"]:
        assert_true(actual.column(name).dtype() == DataType.FLOAT64)
    var count = model.int_count()
    var values = sorted_integers(model)
    assert_equal(actual.item(output_row, "sum").int64(), Int64(model.int_sum()))
    assert_equal(actual.item(output_row, "count").int64(), count)
    assert_equal(actual.item(output_row, "len").int64(), Int64(len(model.rows)))
    assert_equal(
        actual.item(output_row, "distinct").int64(),
        Int64(len(reference_groups(model, ["i"]))),
    )
    for name in ["min", "max", "mean", "median"]:
        assert_equal(actual.item(output_row, name).is_null(), count == 0)
    if count:
        assert_equal(actual.item(output_row, "min").int64(), values[0])
        assert_equal(
            actual.item(output_row, "max").int64(), values[len(values) - 1]
        )
        assert_statistic(
            actual.item(output_row, "mean").float64(),
            Float64(model.int_sum()) / Float64(count),
        )
        assert_statistic(
            actual.item(output_row, "median").float64(),
            integer_quantile(model, 0.5),
        )
    for name in ["var", "std"]:
        assert_equal(actual.item(output_row, name).is_null(), count < 2)
    if count >= 2:
        var variance = integer_variance(model)
        assert_statistic(actual.item(output_row, "var").float64(), variance)
        assert_statistic(
            actual.item(output_row, "std").float64(), sqrt(variance)
        )
    if model.rows:
        assert_reference_cell(
            actual.item(output_row, "first"), model.rows[0], "i"
        )
        assert_reference_cell(
            actual.item(output_row, "last"),
            model.rows[len(model.rows) - 1],
            "i",
        )
    else:
        assert_true(actual.item(output_row, "first").is_null())
        assert_true(actual.item(output_row, "last").is_null())
