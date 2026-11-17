"""Scheduler output retains selections across worker and morsel cleanup."""
# contract-group: execution.lifecycle
from std.testing import assert_equal, assert_true
from std.memory import ArcPointer
from dataframe import Column, DataFrame, Expr, Series, col, lit
from dataframe.query_execution.pipeline import plan_selection_batches
from tests.support.group_strategy_reference import ScopedEnvironment
from dataframe.planning.context import QueryContext
from dataframe.planning.model import SELECT
from dataframe.vector.selection_batches import SelectionBatches


def _source(rows: Int) raises -> DataFrame:
    var ids = List[Int64](capacity=rows)
    var values = List[Int32](capacity=rows)
    var valid = List[Bool](capacity=rows)
    for row in range(rows):
        ids.append(Int64(row))
        values.append(Int32(row % 101 - 50))
        valid.append(row % 17 != 3)
    return DataFrame(
        [
            Series("row", Column[Int64](ids^)),
            Series("v", Column[Int32](values^, valid)),
        ]
    )


def test_pipeline_selection_batches_keep_source_order_and_storage() raises:
    var source = _source(130_017).slice(3, 130_000)
    var expected_rows = List[Int]()
    for row in range(source.height()):
        if (row + 3) % 7 < 3 and (row + 3) % 11 != 2:
            expected_rows.append(row)
    var expected = source.take(expected_rows).select_exprs(
        [col("v").alias("value"), col("row"), col("v").alias("again")]
    )
    for workers in [1, 4]:
        var configured = ScopedEnvironment("DATAFRAME_THREADS", String(workers))
        var plan = (
            source.lazy()
            .filter(col("row") % lit(Int64(7)) < lit(Int64(3)))
            .filter(col("row") % lit(Int64(11)) != lit(Int64(2)))
            .select_exprs(
                [col("v").alias("value"), col("row"), col("v").alias("again")]
            )
        )
        var batches = plan_selection_batches(
            plan._plan, len(plan._plan._nodes) - 1, 4096
        )
        assert_true(batches[].ready and not batches[].fallback)
        assert_equal(batches[].rows, len(expected_rows))
        for batch_index in range(len(batches[].batches)):
            ref batch = batches[].batches[batch_index]
            assert_equal(batch.copied_columns, 0)
            assert_true(
                batch.columns[0]._shares_buffers_with(source.column("v"))
            )
            assert_equal(batch.lists[0], batch.lists[2])
        _ = plan^
        assert_true(batches[].materialize().equals(expected))
        _ = configured^


def test_pipeline_selection_batches_own_computed_values() raises:
    var source = _source(70_019)
    var plan = (
        source.lazy()
        .filter(col("row") % lit(Int64(5)) == lit(Int64(0)))
        .select_exprs([col("row"), (col("v") + lit(Int32(2))).alias("derived")])
    )
    var batches = plan_selection_batches(
        plan._plan, len(plan._plan._nodes) - 1, 4096
    )
    assert_true(not batches[].fallback)
    var rows = List[Int64]()
    var values = List[Int32]()
    var valid = List[Bool]()
    for row in range(0, source.height(), 5):
        rows.append(Int64(row))
        values.append(Int32(row % 101 - 48))
        valid.append(row % 17 != 3)
    var expected = DataFrame(
        [
            Series("row", Column[Int64](rows^)),
            Series("derived", Column[Int32](values^, valid)),
        ]
    )
    var copied = 0
    for batch_index in range(len(batches[].batches)):
        ref batch = batches[].batches[batch_index]
        copied += batch.copied_columns
        assert_true(batch.columns[0]._shares_buffers_with(source.column("row")))
    assert_true(copied > 0)
    _ = (plan^, source^)
    assert_true(batches[].materialize().equals(expected))


def test_pipeline_selection_batches_preserve_duplicate_join_rows() raises:
    var keys = List[Int64]()
    for row in range(30_019):
        keys.append(Int64(row % 13))
    var fact = _source(30_019).with_column(Series("k", Column[Int64](keys^)))
    var build = DataFrame(
        [
            Series("k", Column[Int64]([1, 1, 3, 8])),
            Series("w", Column[Int32]([10, 20, 30, 80])),
        ]
    )
    var plan = (
        fact.lazy()
        .join(build.lazy(), on="k")
        .select_exprs([col("row"), col("v"), col("w")])
    )
    var batches = plan_selection_batches(
        plan._plan, len(plan._plan._nodes) - 1, 2048
    )
    assert_true(not batches[].fallback)
    var ids = List[Int64]()
    var values = List[Int32]()
    var valid = List[Bool]()
    var weights = List[Int32]()
    for row in range(fact.height()):
        for key in [1, 1, 3, 8]:
            if row % 13 != key:
                continue
            ids.append(Int64(row))
            values.append(Int32(row % 101 - 50))
            valid.append(row % 17 != 3)
            weights.append(Int32(0))
    # The two copies of key 1 carry distinct build payloads.
    var at = 0
    for row in range(fact.height()):
        if row % 13 == 1:
            weights[at] = 10
            weights[at + 1] = 20
            at += 2
        elif row % 13 == 3 or row % 13 == 8:
            weights[at] = Int32(10 * (row % 13))
            at += 1
    var expected = DataFrame(
        [
            Series("row", Column[Int64](ids^)),
            Series("v", Column[Int32](values^, valid)),
            Series("w", Column[Int32](weights^)),
        ]
    ).sort(["row", "w"])
    _ = (plan^, fact^, build^)
    assert_true(batches[].materialize().sort(["row", "w"]).equals(expected))


def _accept_selection(region: QueryContext) -> Bool:
    return len(region._nodes) == 2 and region._nodes[1].kind == SELECT


def _bare_reducer_only(region: QueryContext) -> Bool:
    return len(region._nodes) == 2 and region._nodes[1].kind == SELECT


def _selected_sum(
    region: QueryContext, batches: SelectionBatches
) raises -> Optional[DataFrame]:
    batches.validate()
    assert_true(len(batches.batches) > 0)
    for batch_index in range(len(batches.batches)):
        ref batch = batches.batches[batch_index]
        assert_equal(batch.copied_columns, 0)
        assert_true(len(batch.rows) > 0)
    return batches.materialize().select_exprs(region._nodes[1].exprs)


def _decline_selection(
    region: QueryContext, batches: SelectionBatches
) raises -> Optional[DataFrame]:
    _ = region
    batches.validate()
    return None


def _ordinary_sum(region: QueryContext) raises -> DataFrame:
    return region._frames[0].select_exprs(region._nodes[1].exprs)


def test_pipeline_selection_handoff_and_layout_decline_run_source_once() raises:
    var source = _source(70_019)
    var expected = source.filter(
        col("row") % lit(Int64(7)) < lit(Int64(3))
    ).select_exprs(
        [col("v").sum().alias("sum"), col("v").count().alias("count")]
    )
    for decline in [False, True]:
        var plan = (
            source.lazy()
            .filter(col("row") % lit(Int64(7)) < lit(Int64(3)))
            .select_exprs(
                [col("v").sum().alias("sum"), col("v").count().alias("count")]
            )
            ._with_handoff()
        )
        plan._plan._supports = _bare_reducer_only
        plan._plan._selection_supports = _accept_selection
        plan._plan._selection_execute = (
            _decline_selection if decline else _selected_sum
        )
        plan._plan._handoff_execute = _ordinary_sum
        var executions = ArcPointer(Int(0))
        plan._plan._nodes[0]._executions = executions.copy()
        var measured = plan.profile(
            engine="cpu", optimize=False, batch_size=4096
        )
        assert_equal(executions[], 1)
        assert_true(measured[0].equals(expected))
        var handoffs = 0
        for row in range(measured[1].height()):
            assert_equal(measured[1].item(row, "executions").int64(), Int64(1))
            var algorithm = measured[1].item(row, "algorithm").string()
            if algorithm == ("handoff" if decline else "handoff selections"):
                handoffs += 1
        assert_equal(handoffs, 1)


def _unexpected_selection(
    region: QueryContext, batches: SelectionBatches
) raises -> Optional[DataFrame]:
    _ = region
    _ = batches
    raise Error("a materializing producer must use its completed frame")


def test_pipeline_selection_handoff_limit_and_sort_fallback_run_source_once() raises:
    var source = _source(70_019)
    for ordered in [False, True]:
        var producer = source.lazy().filter(
            col("row") % lit(Int64(7)) < lit(Int64(3))
        )
        var expected = source.filter(col("row") % lit(Int64(7)) < lit(Int64(3)))
        if ordered:
            producer = producer.sort("row", descending=True)
            expected = expected.sort("row", descending=True)
        producer = producer.head(17)
        var expressions: List[Expr] = [
            col("v").sum().alias("sum"),
            col("v").count().alias("count"),
        ]
        expected = expected.head(17).select_exprs(expressions)
        var plan = producer.select_exprs(expressions)._with_handoff()
        plan._plan._supports = _bare_reducer_only
        plan._plan._selection_supports = _accept_selection
        plan._plan._selection_execute = _unexpected_selection
        plan._plan._handoff_execute = _ordinary_sum
        var executions = ArcPointer(Int(0))
        plan._plan._nodes[0]._executions = executions.copy()
        var measured = plan.profile(
            engine="cpu", optimize=False, batch_size=4096
        )
        assert_equal(executions[], 1)
        assert_true(measured[0].equals(expected))
        var handoffs = 0
        for row in range(measured[1].height()):
            if measured[1].item(row, "algorithm").string() == "handoff":
                assert_equal(
                    measured[1].item(row, "executions").int64(), Int64(1)
                )
                handoffs += 1
        assert_equal(handoffs, 1)
