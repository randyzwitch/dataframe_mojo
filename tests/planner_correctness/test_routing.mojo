"""engine="auto" keeps work without device kernels on the CPU."""
# contract-group: planner.routing
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, LazyFrame, Series, col


def test_bool_sums_means_and_cumulative_sums_stay_on_the_cpu() raises:
    # Bool sums count true values (#790); engine="auto" runs them on the CPU.
    var frame = DataFrame(
        [
            Series("k", Column[Int64]([1, 2, 1])),
            Series("b", Column[Bool]([True, False, True])),
        ]
    )
    var queries: List[LazyFrame] = [
        frame.lazy().select(col("b").sum()),
        frame.lazy().select(col("b").mean()),
        frame.lazy().select(col("b").cum_sum()),
        frame.lazy()
        .group_by("k")
        .agg([col("b").sum().alias("total"), col("b").mean().alias("share")]),
    ]
    for query in queries:
        assert_true(
            query.collect(engine="auto").equals(query.collect(engine="cpu"))
        )
    var counted = frame.lazy().select(col("b").sum()).collect(engine="auto")
    assert_equal(counted.item(0, "b").numeric[DType.uint32](), UInt32(2))
