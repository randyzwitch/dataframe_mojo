"""An unfiltered lazy group-by over an in-memory frame streams its batches
into states of their own only where the eager group-by would hash-partition
its keys, each batch holds few groups, and there are batches for every
worker (#608); otherwise it runs eagerly. Either way the answer is the eager
group-by's."""
# contract-group: planner.relational
from std.testing import assert_true
from dataframe import Column, DataFrame, Series, col, lit
from tests.support.coverage_compare import assert_frame
from tests.support.group_strategy_reference import (
    ScopedEnvironment,
    StrategyTrace,
)

comptime ROWS = 200_000
comptime BATCH = 4096


def frame(groups: Int) raises -> DataFrame:
    var k = List[Int64](capacity=ROWS)
    var v = List[Int64](capacity=ROWS)
    var s = List[String](capacity=ROWS)
    for row in range(ROWS):
        k.append(Int64((row * 7919) % groups * 104729))
        v.append(Int64(row % 1000))
        s.append("s" + String(row % 97))
    return DataFrame(
        [
            Series("k", Column[Int64](k^)),
            Series("v", Column[Int64](v^)),
            Series("s", Column[String](s^)),
        ]
    )


def batched(t: DataFrame, groups: Int) raises -> Bool:
    """Whether the query took batch states; its answer checked against the
    eager group-by's."""
    var observed = StrategyTrace()
    # A string maximum keeps the plan off the pipelines' aggregate table.
    var rows = (
        t.lazy()
        .with_columns((col("v") * lit(Int64(2))).alias("w"))
        .group_by("k")
        .agg([col("s").max().alias("top"), col("w").sum().alias("total")])
        .collect(batch_size=BATCH)
    )
    var took = "dataframe-path: stream.batch_states\n" in observed.text()
    _ = observed^
    var eager = (
        t.with_columns((col("v") * lit(Int64(2))).alias("w"))
        .group_by("k")
        .agg([col("s").max().alias("top"), col("w").sum().alias("total")])
    )
    assert_true(eager.height() == groups)
    assert_frame(rows.sort("k"), eager.sort("k"))
    return took


def test_few_groups_run_eagerly() raises:
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var threads = ScopedEnvironment("DATAFRAME_THREADS", "4")
    assert_true(not batched(frame(50), 50), "few groups were batched")
    _ = threads^
    _ = trace^


def test_partitioned_keys_with_few_groups_a_batch_take_batch_states() raises:
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var threads = ScopedEnvironment("DATAFRAME_THREADS", "4")
    assert_true(batched(frame(400), 400), "batch states were not taken")
    _ = threads^
    _ = trace^


def test_one_worker_runs_eagerly() raises:
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var threads = ScopedEnvironment("DATAFRAME_THREADS", "1")
    assert_true(not batched(frame(400), 400), "one worker was batched")
    _ = threads^
    _ = trace^


def test_a_batch_showing_many_groups_runs_eagerly() raises:
    # The stream's own test, at a split small enough for a 4K-row batch:
    # it would collect the batches into the eager group-by anyway.
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var threads = ScopedEnvironment("DATAFRAME_THREADS", "4")
    var split = ScopedEnvironment("DATAFRAME_STREAM_SPLIT_GROUPS", "64")
    assert_true(not batched(frame(400), 400), "many groups were batched")
    _ = split^
    _ = threads^
    _ = trace^
