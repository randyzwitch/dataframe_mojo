"""Infallible Float64 states retain narrow Decimal operand observations."""
# contract-group: execution.backends
# contract-group: execution.apple_parity
from std.memory import bitcast
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Expr, Series, col, lit
from dataframe.binding import bind, BoundExpr
from dataframe.lazy import LazyFrame
from dataframe_metal.metal import MetalRuntime
from dataframe_metal.accelerator import decimal_group_feed
from dataframe_metal.accelerator.joins import lower_join_dag
from dataframe_metal.accelerator.rows import lower_rows
from dataframe_metal.accelerator.gpu.program import Program
from dataframe_metal.accelerator.gpu.runtime import capabilities
from tests.execution_modes.test_backend_metal import require_metal
from tests.support.coverage_generators import offset_layout, chunk_layout
from tests.support.lifecycle_jobs import WorkerSetting


def source(nullable: Bool = False) raises -> DataFrame:
    var columns = List[Series]()
    columns.append(Series("g", Column[Int64]([0, 1, 0])))
    var names: List[String] = ["price", "cost", "discount", "sales"]
    var raw: List[List[Int64]] = [
        [1000, 2000, 3000],
        [100, 200, 300],
        [100, 200, 300],
        [400, 800, 1200],
    ]
    for at in range(len(names)):
        columns.append(
            Series(
                names[at],
                Column[Int64](raw[at].copy(), [True, True, not nullable]),
            ).with_dtype(DataType.decimal(15, 2, 64))
        )
    return DataFrame(columns^)


def operand(divisor: Float64 = 2.0) -> Expr:
    return (col("price") - col("cost") - col("discount") + col("sales")).cast(
        DataType.FLOAT64
    ) / lit(divisor)


def expressions() -> List[Expr]:
    return [
        operand().sum(min_count=1).alias("total"),
        operand().mean().alias("average"),
        col("price").sum(min_count=1).alias("money"),
    ]


def test_decimal_float_group_feed_original_operand_and_state_proof() raises:
    var frame = source()
    var bounds = List[BoundExpr]()
    for expression in expressions():
        bounds.append(bind(expression, frame._columns))
    assert_true(decimal_group_feed.eligible(bounds, frame._columns, ["g"]))
    var query = (
        frame.lazy().group_by("g", maintain_order=True).agg(expressions())
    )
    var dag = lower_join_dag(query, capabilities())
    var observed = 0
    for task in dag.tasks:
        for node in task.query._nodes:
            if node.text == decimal_group_feed.MATERIALIZATION:
                observed += 1
                var plan = lower_rows(
                    task.query, capabilities(), terminal_boundary=True
                )
                assert_equal(plan.ordered_mode, 7)
                assert_true(not plan.reductions)
                _ = Program(plan)
    assert_equal(observed, 1)
    # Exception-infallible Float64 operations never make a foreign checked
    # CAST eligible, even when followed by Float64 division and SUM.
    var foreign = col("price").cast(DataType.INT8).cast(DataType.FLOAT64) / lit(
        2.0
    )
    var foreign_bound = bind(foreign.sum(), frame._columns)
    assert_true(
        not decimal_group_feed.eligible(
            [foreign_bound.copy()], frame._columns, ["g"]
        )
    )


def literal(actual: DataFrame, nullable: Bool) raises:
    assert_equal(actual.column("g").int64().to_list(), [Int64(0), Int64(1)])
    assert_equal(
        actual.column("total").float64().to_list(),
        [Float64(6 if nullable else 24), Float64(12)],
    )
    assert_equal(
        actual.column("average").float64().to_list(),
        [Float64(6 if nullable else 12), Float64(12)],
    )
    assert_equal(actual.column("money").dtype(), DataType.decimal(38, 2))
    assert_equal(
        actual.column("money")._decimal_raw(0),
        Int128(1000 if nullable else 4000),
    )
    assert_equal(actual.column("money")._decimal_raw(1), Int128(2000))


# requires: metal
def test_mojo_gpu_decimal_float_group_feed_literals_views_nulls_and_states() raises:
    require_metal()
    for nullable in [False, True]:
        var frame = source(nullable)
        for layout in range(3):
            var input = (
                frame.copy() if layout
                == 0 else offset_layout(frame) if layout
                == 1 else chunk_layout(frame)
            )
            var query = (
                input.lazy()
                .group_by("g", maintain_order=True)
                .agg(expressions())
            )
            for streaming in [False, True]:
                literal(
                    query.collect(
                        engine="cpu",
                        optimize=False,
                        streaming=streaming,
                        batch_size=1,
                    ),
                    nullable,
                )
                literal(
                    query.collect(
                        accelerator=MetalRuntime(),
                        optimize=False,
                        streaming=streaming,
                        batch_size=1,
                    ),
                    nullable,
                )


# requires: metal
def test_mojo_gpu_decimal_float_group_feed_precision_faults_and_ieee_division() raises:
    require_metal()
    var bad = DataFrame(
        [
            Series("g", Column[Int64]([0])),
            Series("a", Column[Int64]([900000000000000000])).with_dtype(
                DataType.decimal(18, 2, 64)
            ),
            Series("b", Column[Int64]([-900000000000000000])).with_dtype(
                DataType.decimal(18, 2, 64)
            ),
        ]
    )
    var query = (
        bad.lazy()
        .group_by("g")
        .agg(
            ((col("a") - col("b")).cast(DataType.FLOAT64) / lit(2.0))
            .sum(min_count=999)
            .alias("v")
        )
        .head(0)
    )
    for workers in [1, 3]:
        var setting = WorkerSetting(workers)
        for streaming in [False, True]:
            with assert_raises(contains="decimal precision"):
                _ = query.collect(
                    engine="cpu",
                    optimize=False,
                    streaming=streaming,
                    batch_size=1,
                )
            with assert_raises(contains="decimal precision"):
                _ = query.collect(
                    accelerator=MetalRuntime(),
                    optimize=False,
                    streaming=streaming,
                    batch_size=1,
                )
        _ = setting^
    var infinities = (
        source()
        .lazy()
        .group_by("g", maintain_order=True)
        .agg(operand(0.0).sum().alias("v"))
    )
    for gpu in [False, True]:
        var actual = infinities.collect(
            accelerator=MetalRuntime(), optimize=False
        ) if gpu else infinities.collect(engine="cpu", optimize=False)
        for row in range(actual.height()):
            assert_equal(
                bitcast[DType.uint64](actual.column("v").float64()._get(row)),
                UInt64(0x7FF0000000000000),
            )
