"""Checked producers must finish before downstream consumers can hide faults."""
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, Series, col, lit
from dataframe.lazy import LazyFrame
from dataframe.metal import MetalRuntime, _capabilities
from dataframe.accelerator.joins import (
    lower_join_dag,
    has_checked_integer_consumers,
)
from dataframe.accelerator.rows import lower_rows


def _source(overflow: Bool = False) raises -> DataFrame:
    return DataFrame(
        [
            Series("k", Column[Int64]([0, 0, 1])),
            Series(
                "v", Column[Int64]([Int64.MAX if overflow else Int64(2), 1, 4])
            ),
        ]
    )


def _collect(query: LazyFrame, gpu: Bool) raises -> DataFrame:
    return query.collect(
        accelerator=MetalRuntime(), optimize=False
    ) if gpu else query.collect(engine="cpu", streaming=False, optimize=False)


def test_mojo_gpu_observed_integer_producer_success_and_nested_groups() raises:
    var source = _source()
    var left = (
        source.lazy()
        .with_columns((col("v") + lit(Int64(1))).alias("unused"))
        .select_exprs([col("k"), col("v")])
    )
    var right = (
        source.lazy()
        .with_columns((col("v") + lit(Int64(1))).alias("r"))
        .select_exprs([col("k"), col("r")])
    )
    var joined = left.join(right, on="k", how="inner")
    assert_true(_collect(joined, True).equals(_collect(joined, False)))
    var grouped = (
        source.lazy()
        .group_by("k", maintain_order=True)
        .agg(col("v").sum())
        .group_by("k", maintain_order=True)
        .agg(col("v").sum())
    )
    for gpu in [False, True]:
        var result = _collect(grouped, gpu)
        assert_equal(result.height(), 2)
        assert_equal(result.item(0, "v").int64(), 3)
        assert_equal(result.item(1, "v").int64(), 4)
        assert_equal(_collect(grouped.head(0), gpu).height(), 0)


def test_mojo_gpu_observed_unused_integer_overflow_before_consumers() raises:
    var source = _source(True)
    var producer = (
        source.lazy()
        .with_columns((col("v") + lit(Int64(1))).alias("unused"))
        .select_exprs([col("k"), col("v")])
    )
    var right = _source().lazy().select(col("k"))
    for gpu in [False, True]:
        for empty in [False, True]:
            var query = producer.join(
                right.head(0) if empty else right.copy(), on="k", how="inner"
            )
            with assert_raises(contains="overflow"):
                _ = _collect(query, gpu)
            with assert_raises(contains="overflow"):
                _ = _collect(query.head(0), gpu)
        with assert_raises(contains="overflow"):
            _ = _collect(producer.head(0), gpu)
    # Both expressions fault on the first row; CPU expression order decides
    # which fault is visible even when the consumer requests no rows.
    for gpu in [False, True]:
        for power_first in [False, True]:
            var overflow = (col("v") + lit(Int64(1))).alias("overflow")
            var power = col("v").pow(lit(Int64(-1))).alias("power")
            var ordered = (
                source.lazy()
                .with_columns(
                    [power.copy(), overflow.copy()] if power_first else [
                        overflow.copy(),
                        power.copy(),
                    ]
                )
                .head(0)
            )
            with assert_raises(
                contains="nonnegative exponent" if power_first else "overflow"
            ):
                _ = _collect(ordered, gpu)
    # A previous failed task does not contaminate a fresh query.
    assert_equal(_collect(_source().lazy().select(col("v")), True).height(), 3)


def test_mojo_gpu_observed_first_group_overflow_before_regroup_and_head() raises:
    var first = (
        _source(True)
        .lazy()
        .group_by("k", maintain_order=True)
        .agg(col("v").sum())
    )
    var second = first.group_by("k", maintain_order=True).agg(col("v").sum())
    for gpu in [False, True]:
        with assert_raises(contains="overflow"):
            _ = _collect(second, gpu)
        with assert_raises(contains="overflow"):
            _ = _collect(second.head(0), gpu)
        with assert_raises(contains="overflow"):
            _ = _collect(first.head(0), gpu)


def test_observed_integer_boundary_signatures_and_original_order() raises:
    var query = (
        _source()
        .lazy()
        .with_columns(
            [
                (col("v") + lit(Int64(1))).alias("first"),
                (col("v") * lit(Int64(2))).alias("second"),
            ]
        )
        .head(0)
    )
    assert_true(has_checked_integer_consumers(query))
    var dag = lower_join_dag(
        query._with_accelerator_settings(False, False, 65536), _capabilities()
    )
    var found = False
    for task in dag.tasks:
        ref node = task.query._nodes[len(task.query._nodes) - 1]
        if node.text != "metal_integer_materialization":
            continue
        found = True
        assert_true(task.terminal)
        assert_equal(node.exprs[0]._name, "first")
        assert_equal(node.exprs[1]._name, "second")
        assert_equal(len(node.names2), 3)
        var plan = lower_rows(
            task.query, _capabilities(), terminal_boundary=True
        )
        assert_equal(plan.ordered_mode, 4)
        var changed = task.query.copy()
        changed._nodes[len(changed._nodes) - 1].exprs[0] = (
            col("v") + lit(Int64(2))
        ).alias("first")
        with assert_raises(contains="metadata changed"):
            _ = lower_rows(changed, _capabilities(), terminal_boundary=True)
        with assert_raises(contains="invalid observed producer"):
            _ = lower_rows(task.query, _capabilities(), terminal_boundary=False)
    assert_true(found)
    var grouped = (
        _source()
        .lazy()
        .group_by("k", maintain_order=True)
        .agg(col("v").sum())
        .head(0)
    )
    var groups = lower_join_dag(
        grouped._with_accelerator_settings(False, False, 65536), _capabilities()
    )
    found = False
    for task in groups.tasks:
        ref node = task.query._nodes[len(task.query._nodes) - 1]
        if node.text != "metal_group_materialization":
            continue
        found = True
        assert_true(task.terminal)
        var changed = task.query.copy()
        changed._nodes[len(changed._nodes) - 1].maintain_order = False
        with assert_raises(contains="group metadata changed"):
            _ = lower_rows(changed, _capabilities(), terminal_boundary=True)
    assert_true(found)
