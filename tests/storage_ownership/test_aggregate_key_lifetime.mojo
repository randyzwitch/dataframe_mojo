"""String keys longer than twelve bytes that a step computes in each morsel
live only in that morsel's buffers. The aggregate table keeps a group's key as
the address of its bytes, so it must hold those buffers until its groups are
read out: the result's keys are correct, and stay correct after the source,
the plan and every morsel are gone and their memory is reused."""
from std.testing import assert_equal

from dataframe import DataFrame, Expr, col, lit
from tests.support.aggregate_table_reference import (
    aggregate_frame,
    assert_groups,
    group_models,
)
from tests.support.group_strategy_reference import ScopedEnvironment


def reductions() -> List[Expr]:
    return [
        col("x").sum(min_count=1).alias("x_sum"),
        col("x").mean().alias("x_mean"),
        col("x").min().alias("x_min"),
        col("x").max().alias("x_max"),
        col("x").count().alias("x_count"),
        col("w").sum(min_count=1).alias("w_sum"),
        col("w").null_count().alias("w_nulls"),
        col("x").len().alias("rows"),
        col("y").mean().alias("y_mean"),
        col("y").std().alias("y_std"),
    ]


def grouped_by_computed_keys(rows: Int, groups: Int) raises -> DataFrame:
    var frame = aggregate_frame(rows, groups)
    var result = (
        frame.lazy()
        .with_columns(
            [col("s").str().replace("group key", "GROUPED KEY").alias("s")]
        )
        .filter(col("r") >= lit(Int64(3)))
        .group_by(["s", "i"])
        .agg(reductions())
        .collect(batch_size=2048)
    )
    _ = frame^
    return result^


def test_keys_computed_per_morsel_outlive_their_morsels_and_source() raises:
    var workers = ScopedEnvironment("DATAFRAME_THREADS", "8")
    var rows = 48_000
    var groups = 6_000
    var result = grouped_by_computed_keys(rows, groups)
    # Allocate and drop as much again, so freed morsel buffers are reused.
    for _ in range(3):
        var other = grouped_by_computed_keys(rows, groups)
        assert_equal(other.height(), result.height())
    assert_groups(result, group_models(rows, groups, 3), False, True)
    _ = workers^
