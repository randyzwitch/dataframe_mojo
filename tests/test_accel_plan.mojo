"""GPU capability and coercion checks run without the optional GPU SDK."""
from std.testing import TestSuite, assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, Series, col, lit, scan_csv
from dataframe.accel_plan import lower_accel
from dataframe.expr import GT, MUL


def test_shared_binding_and_lowering() raises:
    var frame = DataFrame([Series("x", Column[Float32]([1, 2]))])
    var query = (
        frame.lazy()
        .filter(col("x") > 0)
        .select_exprs(
            [
                (col("x") * 1.25).sum(min_count=2).alias("total"),
                col("x").count().alias("count"),
            ]
        )
    )
    var plan = lower_accel(query)
    assert_equal(plan.predicate.op, GT)
    assert_equal(plan.predicate.literal, Float64(0))
    assert_equal(plan.reductions[0].input.op, MUL)
    assert_equal(plan.reductions[0].input.literal, Float64(1.25))
    assert_equal(plan.reductions[0].min_count, 2)
    assert_equal(plan.reductions[0].name, "total")
    assert_equal(lower_accel(query.head(0)).limit, 0)
    with assert_raises(contains="Unknown expression column"):
        _ = lower_accel(frame.lazy().select(col("missing").sum()))
    with assert_raises(contains="min_count must be nonnegative"):
        _ = lower_accel(frame.lazy().select(col("x").sum(min_count=-1)))


def test_unsupported_regions_fail_before_source_access() raises:
    with assert_raises(contains="in-memory scan"):
        _ = lower_accel(
            scan_csv("/nonexistent/accel-plan.csv").select(col("x").sum())
        )
    var frame = DataFrame([Series("x", Column[Float64]([1, 2]))])
    with assert_raises(contains="column and a bound floating scalar"):
        _ = lower_accel(frame.lazy().select(((col("x") + 1.0) * 2.0).sum()))
    with assert_raises(contains="duplicate output name"):
        _ = lower_accel(
            frame.lazy().select_exprs([col("x").sum(), col("x").count()])
        )
    with assert_raises(contains="sum or count"):
        _ = lower_accel(frame.lazy().select(col("x").mean()))
    with assert_raises(contains="Float32 or Float64"):
        _ = lower_accel(
            DataFrame([Series("x", Column[Int64]([1]))])
            .lazy()
            .select(col("x").sum())
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
