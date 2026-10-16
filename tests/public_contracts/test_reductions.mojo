"""Global integer statistics have explicit empty, null and quantile semantics."""
# contract-group: reduce.global
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, Series, col
from tests.support.reference_model import generated_model
from tests.support.reference_statistics import integer_quantile
from tests.support.coverage_generators import frame_from_model, chunk_layout
from tests.support.reduction_contract import (
    integer_reductions,
    assert_integer_reductions,
    assert_statistic,
)


def test_global_integer_statistics_and_quantile_methods() raises:
    for n in [0, 1, 9, 65]:
        for mode in range(3):
            var model = generated_model(n, 3, mode)
            for layout in range(3):
                var frame = frame_from_model(model, 7 if layout == 1 else 0)
                if layout == 2:
                    frame = chunk_layout(frame)
                var actual = frame.select_exprs(integer_reductions())
                assert_equal(actual.height(), 1)
                assert_integer_reductions(actual, 0, model)
                for q in [0.0, 0.25, 0.5, 0.75, 1.0]:
                    for method in [
                        "nearest",
                        "linear",
                        "lower",
                        "higher",
                        "midpoint",
                    ]:
                        var value = frame.select(
                            col("i").quantile(q, method)
                        ).item()
                        assert_equal(value.is_null(), model.int_count() == 0)
                        if model.int_count():
                            assert_statistic(
                                value.float64(),
                                integer_quantile(model, q, method),
                            )
                for q in [-0.1, 1.1]:
                    with assert_raises():
                        _ = frame.select(col("i").quantile(q))
                with assert_raises():
                    _ = frame.select(col("i").sum(min_count=-1))


def test_integer_sum_checks_final_total_not_partial_and_mean_is_wide() raises:
    var frame = DataFrame([Series("i", Column[Int64]([Int64.MAX, 1, -1]))])
    assert_equal(frame.select(col("i").sum()).item().int64(), Int64.MAX)
    assert_equal(
        frame.select(col("i").mean()).item().float64(), Float64(Int64.MAX) / 3
    )
    var overflow_cases: List[List[Int64]] = [[Int64.MAX, 1], [Int64.MIN, -1]]
    for values in overflow_cases:
        with assert_raises():
            _ = DataFrame([Series("i", Column[Int64](values.copy()))]).select(
                col("i").sum()
            )
    var null_poison = DataFrame(
        [Series("i", Column[Int64]([Int64.MAX, 1], [False, True]))]
    )
    assert_equal(null_poison.select(col("i").sum()).item().int64(), Int64(1))
    assert_true(null_poison.select(col("i").sum(min_count=2)).item().is_null())
