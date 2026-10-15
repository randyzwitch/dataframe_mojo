"""Public IANA producer phases before resident head/order/group consumers."""
from std.testing import assert_equal, assert_raises
from dataframe import Column, DataFrame, DataType, Series, col
from dataframe.metal import MetalRuntime
from tests.support.coverage_compare import assert_frame


def main() raises:
    var runtime = MetalRuntime()
    var naive = DataType.datetime("us")
    var ny = DataType.datetime("us", "America/New_York")
    var source = DataFrame(
        [
            Series("t", Column[Int64]([0, 1])).with_dtype(naive),
            Series("k", Column[Int32]([0, 0])),
        ]
    )
    var produced = source.lazy().with_columns(
        col("t").dt().replace_time_zone("America/New_York").alias("z")
    )
    var selected = produced.sort("z").head(1)
    var expected = DataFrame(
        [
            Series("t", Column[Int64]([0])).with_dtype(naive),
            Series("k", Column[Int32]([0])),
            Series("z", Column[Int64]([18000000000])).with_dtype(ny),
        ]
    )
    assert_frame(selected.collect(engine="cpu"), expected)
    assert_frame(selected.collect(accelerator=runtime), expected)
    var grouped = produced.group_by("k", maintain_order=True).agg(
        col("z").max()
    )
    var grouped_expected = DataFrame(
        [
            Series("k", Column[Int32]([0])),
            Series("z", Column[Int64]([18000000001])).with_dtype(ny),
        ]
    )
    assert_frame(grouped.collect(engine="cpu"), grouped_expected)
    assert_frame(grouped.collect(accelerator=runtime), grouped_expected)
    var gap = DataFrame(
        [Series("t", Column[Int64]([0, 1710037800000000])).with_dtype(naive)]
    )
    var checked = gap.lazy().with_columns(
        col("t").dt().replace_time_zone("America/New_York").alias("z")
    )
    for count in List[Int]([0, 1]):
        var failing = checked.sort("z").head(count)
        with assert_raises(contains="non-existent"):
            _ = failing.collect(engine="cpu")
        with assert_raises(contains="non-existent"):
            _ = failing.collect(accelerator=runtime)
    # The verified optimizer bridge moves this leading head before the
    # checked row producer, so the gap in the second source row is unseen.
    var optimized = checked.head(1).select("z")
    var optimized_expected = DataFrame(
        [Series("z", Column[Int64]([18000000000])).with_dtype(ny)]
    )
    assert_frame(optimized.collect(engine="cpu"), optimized_expected)
    assert_frame(optimized.collect(accelerator=runtime), optimized_expected)
    with assert_raises(contains="budget"):
        _ = selected.collect(accelerator=MetalRuntime(memory_limit_bytes=0))
    assert_frame(selected.collect(accelerator=runtime), expected)
    print(
        "PASS public observed temporal producer head/order/group, faults and recovery"
    )
