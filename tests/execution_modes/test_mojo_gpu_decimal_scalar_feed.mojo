"""Whole scalar SUM states observe checked Decimal operands on the GPU."""
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Series,
    StringColumn,
    col,
    lit,
    when,
)
from dataframe.lazy import LazyFrame
from dataframe.metal import MetalRuntime
from dataframe.accelerator import decimal_scalar_feed
from dataframe.accelerator.constants import fold_literal_text_parsing
from dataframe.accelerator.aggregates import lower_aggregate_compositions
from dataframe.accelerator.decimal_prefix import prefix_infallible
from dataframe.accelerator.joins import lower_join_dag
from dataframe.accelerator.rows import lower_rows
from dataframe.accelerator.gpu.runtime import capabilities, estimate_resources
from dataframe.accelerator.gpu.program import Program
from tests.execution_modes.test_backend_metal import require_metal
from tests.support.coverage_generators import offset_layout, chunk_layout
from tests.support.lifecycle_jobs import WorkerSetting


def money_source() raises -> DataFrame:
    return DataFrame(
        [
            Series("price", Column[Int64]([10000, 20000, 30000])).with_dtype(
                DataType.decimal(15, 2, 64)
            ),
            Series("discount", Column[Int64]([0, 50, 0])).with_dtype(
                DataType.decimal(15, 2, 64)
            ),
            Series("p", Column[Bool]([True, False, True])),
        ]
    )


def money_ratio(source: DataFrame) raises -> LazyFrame:
    var money = lit("1").cast(DataType.decimal(15, 2))
    var discounted = col("price") * (money - col("discount"))
    var selected = (
        when(col("p"))
        .then(discounted.cast(DataType.FLOAT64))
        .otherwise(lit(0.0))
    )
    return source.lazy().select(
        (
            lit(100.0)
            * selected.sum()
            / discounted.sum().cast(DataType.FLOAT64)
        ).alias("ratio")
    )


def test_decimal_scalar_feed_original_ast_prefix_and_resources() raises:
    var query = lower_aggregate_compositions(
        fold_literal_text_parsing(money_ratio(money_source()))
    )._with_accelerator_settings(False, True, 257)
    var dag = lower_join_dag(query, capabilities())
    var observed = 0
    for task in dag.tasks:
        for node in task.query._nodes:
            if node.text == decimal_scalar_feed.MATERIALIZATION:
                observed += 1
                var plan = lower_rows(
                    task.query, capabilities(), terminal_boundary=True
                )
                assert_equal(plan.ordered_mode, 7)
                assert_true(plan.reductions)
                assert_equal(len(plan.outputs), 2)
                assert_equal(len(plan.ordered_ranks[]), len(plan.literals))
                for step in plan.steps:
                    assert_true(
                        not step.grouped and not step.order and not step.filter
                    )
                _ = Program(plan)
                var facts = estimate_resources(plan, 3)
                assert_true(facts.peak_bytes > 0)
                var forged = task.query.copy()
                forged._nodes[len(forged._nodes) - 1].names2[1] += "changed"
                with assert_raises(contains="logical metadata changed"):
                    _ = lower_rows(
                        forged, capabilities(), terminal_boundary=True
                    )
    assert_equal(observed, 1)
    var checked_prefix = (
        money_source()
        .lazy()
        .with_columns(col("price").cast(DataType.INT8).alias("checked"))
    )
    assert_true(
        not prefix_infallible(checked_prefix, len(checked_prefix._nodes) - 1)
    )


def test_mojo_gpu_decimal_scalar_feed_ratio_money_masks_and_layouts() raises:
    require_metal()
    var frame = money_source()
    for source in [frame, offset_layout(frame), chunk_layout(frame)]:
        var query = money_ratio(source)
        for streaming in [False, True]:
            var cpu = query.collect(
                engine="cpu", optimize=False, streaming=streaming, batch_size=1
            )
            var gpu = query.collect(
                accelerator=MetalRuntime(),
                optimize=False,
                streaming=streaming,
                batch_size=1,
            )
            assert_equal(cpu.column("ratio").float64()._get(0), Float64(80))
            assert_equal(gpu.column("ratio").float64()._get(0), Float64(80))
    # Dictionary predicate transport and an ordinary join retain the same
    # general scalar feed proof; no source values are used for certification.
    var keyed = frame.with_column(Series("id", Column[Int32]([0, 1, 2])))
    var labels = DataFrame(
        [
            Series("id", Column[Int32]([0, 1, 2])),
            Series("label", StringColumn(["PROMO A", "OTHER", "PROMO B"])),
        ]
    )
    var discounted = col("price") * (
        lit("1").cast(DataType.decimal(15, 2)) - col("discount")
    )
    var conditional = (
        when(col("label").str().starts_with("PROMO"))
        .then(discounted.cast(DataType.FLOAT64))
        .otherwise(lit(0.0))
    )
    var joined = (
        keyed.lazy()
        .join(labels.lazy(), on="id")
        .select(
            (
                lit(100.0)
                * conditional.sum()
                / discounted.sum().cast(DataType.FLOAT64)
            ).alias("ratio")
        )
    )
    var joined_cpu = joined.collect(engine="cpu", optimize=False)
    var joined_gpu = joined.collect(accelerator=MetalRuntime(), optimize=False)
    assert_equal(joined_cpu.column("ratio").float64()._get(0), Float64(80))
    assert_equal(joined_gpu.column("ratio").float64()._get(0), Float64(80))


def test_mojo_gpu_decimal_scalar_feed_selected_faults_dead_arms_and_min_count() raises:
    require_metal()
    var frame = DataFrame(
        [
            Series("x", Column[Int64]([100, Int64.MAX])).with_dtype(
                DataType.decimal(15, 2, 64)
            ),
            Series("one", Column[Int64]([100, 100])).with_dtype(
                DataType.decimal(15, 2, 64)
            ),
            Series("p", Column[Bool]([True, False])),
        ]
    )
    var selected = (
        when(col("p"))
        .then((col("x") * col("one")).cast(DataType.FLOAT64))
        .otherwise(lit(0.0))
    )
    var runtime = MetalRuntime()
    for workers in [1, 3]:
        var setting = WorkerSetting(workers)
        for streaming in [False, True]:
            var safe = frame.lazy().select(selected.sum().alias("v"))
            var gpu = safe.collect(
                accelerator=runtime,
                optimize=False,
                streaming=streaming,
                batch_size=1,
            )
            assert_equal(gpu.column("v").float64()._get(0), Float64(1))
            var insufficient = frame.lazy().select(
                selected.sum(min_count=999).alias("v")
            )
            var nulls = insufficient.collect(
                accelerator=runtime,
                optimize=False,
                streaming=streaming,
                batch_size=1,
            )
            assert_true(not nulls.column("v").float64().is_valid(0))
            var bad = (
                frame.with_column(Series("p", Column[Bool]([True, True])))
                .lazy()
                .select(selected.sum(min_count=999).alias("v"))
                .head(0)
            )
            with assert_raises(contains="decimal precision"):
                _ = bad.collect(
                    engine="cpu",
                    optimize=False,
                    streaming=streaming,
                    batch_size=1,
                )
            with assert_raises(contains="decimal precision"):
                _ = bad.collect(
                    accelerator=runtime,
                    optimize=False,
                    streaming=streaming,
                    batch_size=1,
                )
        _ = setting^


def test_mojo_gpu_decimal_scalar_feed_null_empty_budget_and_recovery() raises:
    require_metal()
    var runtime = MetalRuntime()
    var source = money_source()
    var all_null = source.with_column(
        Series(
            "price",
            Column[Int64](
                [Int64.MAX, Int64.MAX, Int64.MAX], [False, False, False]
            ),
        ).with_dtype(DataType.decimal(15, 2, 64))
    )
    var null_operand = (
        col("price") * lit("1").cast(DataType.decimal(15, 2))
    ).cast(DataType.FLOAT64)
    var all_null_sum = all_null.lazy().select(null_operand.sum().alias("v"))
    var null_zero = all_null_sum.collect(accelerator=runtime, optimize=False)
    assert_equal(null_zero.column("v").float64()._get(0), Float64(0))
    var all_null_min = all_null.lazy().select(
        null_operand.sum(min_count=1).alias("v")
    )
    var null_state = all_null_min.collect(accelerator=runtime, optimize=False)
    assert_true(not null_state.column("v").float64().is_valid(0))
    var selected = (
        when(col("p"))
        .then(
            (col("price") * lit("1").cast(DataType.decimal(15, 2))).cast(
                DataType.FLOAT64
            )
        )
        .otherwise(lit(0.0))
    )
    var empty = source.head(0).lazy().select(selected.sum().alias("v"))
    var zero = empty.collect(accelerator=runtime, optimize=False)
    assert_equal(zero.column("v").float64()._get(0), Float64(0))
    var empty_null = (
        source.head(0).lazy().select(selected.sum(min_count=1).alias("v"))
    )
    var nulls = empty_null.collect(accelerator=runtime, optimize=False)
    assert_true(not nulls.column("v").float64().is_valid(0))
    with assert_raises(contains="budget"):
        _ = money_ratio(source).collect(
            accelerator=MetalRuntime(memory_limit_bytes=0), optimize=False
        )
    var recovered = money_ratio(source).collect(
        accelerator=runtime, optimize=False
    )
    assert_equal(recovered.column("ratio").float64()._get(0), Float64(80))
