"""Distinct counts reuse resident grouping, including null categories."""
# contract-group: planner.routing
from std.testing import assert_true, assert_raises
from dataframe import DataFrame, Series, Column, StringColumn, Expr, col
from dataframe.lazy import LazyFrame
from dataframe_metal.metal import MetalRuntime, metal_installed, _capabilities
from dataframe_metal.accelerator.distinct_counts import (
    has_distinct_counts,
    lower_distinct_counts,
)
from dataframe_metal.accelerator.joins import lower_join_dag, DAG_JOIN
from dataframe_metal.accelerator.sets import lower_set_operations


def test_distinct_siblings_metadata_and_shared_prefix() raises:
    var input = DataFrame(
        [
            Series("g", Column[Int32]([1, 1, 2])),
            Series("x", Column[Int16]([1, 2, 3])),
        ]
    )
    var expressions = List[Expr](
        [
            col("x").n_unique().alias("distinct"),
            col("x").count().alias("valid"),
            col("x").max().alias("largest"),
        ]
    )
    for grouped in List[Bool]([False, True]):
        var query = input.lazy().select_exprs(expressions)
        if grouped:
            query = (
                input.lazy().group_by("g", maintain_order=True).agg(expressions)
            )
        var count = len(query._nodes)
        var lowered = lower_distinct_counts(query)
        assert_true(len(query._nodes) == count)
        assert_true(not has_distinct_counts(lowered))
        var dag = lower_join_dag(lower_set_operations(lowered), _capabilities())
        assert_true(dag.joins == 2)
        var joins = 0
        for task in dag.tasks:
            if task.kind == DAG_JOIN:
                assert_true(task.how == (7 if grouped else 6))
                joins += 1
        assert_true(joins == 2)
        assert_true(
            "resident distinct counts"
            in MetalRuntime(device_id=999999).describe(query)
        )
