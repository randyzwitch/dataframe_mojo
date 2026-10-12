"""Scalar reductions broadcast through resident GPU branches into row expressions."""
from std.memory import bitcast
from tests.execution_modes.test_backend_metal import require_metal
from std.testing import assert_true, assert_equal, assert_raises
from dataframe import DataFrame, Series, Column, col, lit, when, scan_csv
from dataframe.dtype import DataType
from dataframe.lazy import LazyFrame
from dataframe.metal import MetalRuntime, _capabilities
from dataframe.accelerator.aggregates import lower_aggregate_compositions
from dataframe.accelerator.joins import lower_join_dag, DAG_ROWS, DAG_JOIN
from dataframe.accelerator.rows import lower_rows


def fixture() raises -> DataFrame:
    return DataFrame(
        [
            Series("g", Column[Int64]([0, 0, 1, 1, 2, 2])),
            Series(
                "x",
                Column[Float64](
                    [1, 2, 4, 8, 16, 32],
                    List[Bool]([True, True, False, True, True, False]),
                ),
            ),
            Series(
                "__dataframe_metal_aggregate_0",
                Column[Int32]([10, 11, 12, 13, 14, 15]),
            ),
        ]
    )


def same(actual: DataFrame, expected: DataFrame) raises:
    assert_true(actual.equals(expected))
    for name in actual.columns():
        if actual.column(name).dtype() == DataType.FLOAT64:
            var a = actual.column(name).float64()
            var b = expected.column(name).float64()
            for i in range(actual.height()):
                if b.is_valid(i):
                    var bits = bitcast[DType.uint64](b._get(i))
                    if (bits & 0x7FFFFFFFFFFFFFFF) <= 0x7FF0000000000000:
                        assert_equal(bitcast[DType.uint64](a._get(i)), bits)
        elif actual.column(name).dtype() == DataType.FLOAT32:
            var a = actual.column(name).float32()
            var b = expected.column(name).float32()
            for i in range(actual.height()):
                if b.is_valid(i):
                    var bits = bitcast[DType.uint32](b._get(i))
                    if (bits & 0x7FFFFFFF) <= 0x7F800000:
                        assert_equal(bitcast[DType.uint32](a._get(i)), bits)


def check(query: LazyFrame) raises:
    same(query.collect(accelerator=MetalRuntime()), query.collect(engine="cpu"))


def test_scalar_broadcast_planning_shares_prefix_and_preserves_schema() raises:
    var frame = fixture()
    var query = frame.lazy().filter(col("x") > col("x").mean())
    var original_nodes = len(query._nodes)
    var lowered = lower_aggregate_compositions(query)
    assert_equal(len(query._nodes), original_nodes)
    assert_equal(
        lowered._execute(len(lowered._nodes) - 1, True).columns(),
        frame.columns(),
    )
    var dag = lower_join_dag(lowered, _capabilities())
    assert_equal(dag.joins, 1)
    var found = False
    for task in dag.tasks:
        if task.kind == DAG_JOIN:
            assert_equal(task.how, 6)
            assert_equal(dag.tasks[task.source].kind, DAG_ROWS)
            assert_equal(dag.tasks[task.right].kind, DAG_ROWS)
            assert_equal(dag.tasks[task.right].source, task.source)
            found = True
    assert_true(found)
    var runtime = MetalRuntime(device_id=999999)
    assert_true(
        "Metal resident aggregate expressions" in runtime.describe(query)
    )


def test_scalar_broadcast_rejects_file_reads_and_conditional_precision_faults() raises:
    var runtime = MetalRuntime(device_id=999999)
    var missing = scan_csv(
        "/nonexistent/metal-scalar-broadcast-must-not-read.csv"
    )
    with assert_raises(contains="in-memory scans"):
        _ = missing.filter(col("x") > col("x").mean()).collect(
            accelerator=runtime
        )
    var tiny = bitcast[DType.float32](UInt32(1))
    var narrow = DataFrame([Series("x", Column[Float32]([tiny, tiny]))])
    var conditional = narrow.lazy().select(
        when(lit(False)).then((col("x") * col("x")).sum()).otherwise(col("x"))
    )
    same(conditional.collect(engine="cpu"), narrow)
    with assert_raises(contains="conditional scalar Float32 arithmetic"):
        _ = conditional.collect(accelerator=runtime)
    var wide = DataFrame([Series("x", Column[Int64]([Int64.MAX, Int64.MAX]))])
    var mean = wide.lazy().select(
        when(lit(False))
        .then(col("x").mean())
        .otherwise(col("x").cast(DataType.FLOAT64))
    )
    with assert_raises(contains="conditional scalar wide integer mean"):
        _ = mean.collect(accelerator=runtime)
    with assert_raises(contains="terminal boundary"):
        _ = (
            wide.lazy()
            .with_columns(col("x").sum().alias("s"))
            .collect(accelerator=runtime)
        )


def test_native_metal_scalar_broadcast_mixed_projections_and_literal_scalar_outputs() raises:
    _ = require_metal()
    var frame = fixture()
    check(
        frame.lazy().select_exprs(
            [
                col("g"),
                col("x"),
                col("x").mean().alias("mean"),
                (col("x") - col("x").mean()).alias("centered"),
                (col("x") / col("x").sum()).alias("share"),
                col("x").count().alias("count"),
                col("x").max().alias("max"),
                col("x").min().alias("min"),
            ]
        )
    )
    check(
        frame.lazy().select_exprs(
            [lit(Float64(7)).alias("constant"), col("x").mean().alias("mean")]
        )
    )
    check(
        frame.lazy().select_exprs(
            [col("x").mean().alias("mean"), lit(Float64(7)).alias("constant")]
        )
    )


def test_native_metal_scalar_broadcast_filters_and_head_observe_complete_prefix() raises:
    _ = require_metal()
    var frame = fixture()
    check(frame.lazy().filter(col("x") > col("x").mean()))
    check(
        frame.lazy()
        .filter(col("x") > lit(Float64(0.2)) * col("x").mean())
        .head(1)
    )
    check(
        frame.lazy()
        .filter(col("g") < 2)
        .filter(col("x") > col("x").mean())
        .select(col("g"))
    )
    check(
        frame.lazy().head(4).with_columns(col("x").mean().alias("mean")).head(1)
    )
    check(frame.lazy().with_columns(col("x").mean().alias("mean")).head(1))
    check(
        frame.lazy().filter(col("x").count() > lit(Int64(0))).select(col("g"))
    )


def test_native_metal_scalar_broadcast_with_columns_joins_and_grouped_consumers() raises:
    _ = require_metal()
    var frame = fixture()
    var broadcast = frame.lazy().with_columns(col("x").mean().alias("mean"))
    check(broadcast)
    check(frame.lazy().with_columns(col("x").mean()))
    var right = DataFrame([Series("g", Column[Int64]([0, 1]))])
    check(broadcast.join(right.lazy(), "g").select(["g", "x", "mean"]))
    check(
        frame.lazy().join(right.lazy(), "g").filter(col("x") > col("x").mean())
    )
    check(
        broadcast.group_by("g", maintain_order=True).agg(
            [
                (col("x").sum() - col("mean").first()).alias("delta"),
            ]
        )
    )


def test_native_metal_scalar_broadcast_empty_null_and_conditional_inputs() raises:
    _ = require_metal()
    var frame = fixture()
    var nulls = DataFrame(
        [
            Series("g", Column[Int64]([0, 1, 2])),
            Series(
                "x",
                Column[Float64]([0, 0, 0], List[Bool]([False, False, False])),
            ),
        ]
    )
    for sample in List[DataFrame](
        [frame.slice(0, 0), nulls.copy(), frame.copy()]
    ):
        check(sample.lazy().filter(col("x") > col("x").mean()))
        check(sample.lazy().with_columns(col("x").mean().alias("mean")))
        check(
            sample.lazy().select_exprs(
                [col("g"), col("x").mean().alias("mean")]
            )
        )
        check(
            sample.lazy().select(
                when(col("g") < 0).then(col("x").mean()).otherwise(col("x"))
            )
        )
        check(
            sample.lazy().select(
                when(col("g") < 1).then(col("x").sum()).otherwise(col("x"))
            )
        )


def test_native_metal_scalar_broadcast_float32_subnormal_values() raises:
    _ = require_metal()
    var tiny = bitcast[DType.float32](UInt32(1))
    var frame = DataFrame(
        [
            Series("g", Column[Int64]([0, 1, 2])),
            Series("x", Column[Float32]([tiny, tiny, tiny])),
        ]
    )
    check(frame.lazy().with_columns(col("x").mean().alias("mean")))
    check(
        frame.lazy().filter(col("x").cast(DataType.FLOAT64) >= col("x").mean())
    )
    check(
        frame.lazy().select(
            when(col("g") < 0).then(col("x").sum()).otherwise(col("x"))
        )
    )


def test_native_metal_scalar_broadcast_resident_traffic_budgets_and_recovery() raises:
    _ = require_metal()
    var runtime = MetalRuntime()
    var query = fixture().lazy().with_columns(col("x").mean().alias("mean"))
    var measured = query.profile(accelerator=runtime)
    same(measured[0], query.collect(engine="cpu"))
    assert_true(measured[1].height() >= 4)
    var upload = measured[1].column("upload_bytes").int64()
    var download = measured[1].column("download_bytes").int64()
    assert_true(upload._get(0) > 0)
    var dag = lower_join_dag(
        lower_aggregate_compositions(query), _capabilities()
    )
    assert_equal(measured[1].height(), len(dag.tasks))
    for i in range(1, measured[1].height()):
        # Resident row inputs are not uploaded again. The ABI still stages
        # an eight-byte literal entry per expression node, including columns.
        var literal_bytes = Int64(0)
        if dag.tasks[i].kind == DAG_ROWS:
            var plan = lower_rows(
                dag.tasks[i].query,
                _capabilities(),
                terminal_boundary=dag.tasks[i].terminal,
            )
            literal_bytes = Int64(8 * len(plan.literals))
        assert_equal(upload._get(i), literal_bytes)
        assert_true(
            measured[1].column("resident_input_bytes").int64()._get(i) > 0
        )
    for i in range(measured[1].height() - 1):
        # Retained regions synchronize eight bytes of row-count metadata.
        assert_equal(download._get(i), Int64(8))
        assert_equal(
            measured[1].column("host_result_bytes").int64()._get(i), Int64(0)
        )
    with assert_raises(contains="budget"):
        _ = query.collect(accelerator=MetalRuntime(memory_limit_bytes=1))
    same(query.collect(accelerator=runtime), query.collect(engine="cpu"))
    same(
        query.collect(engine="auto", accelerator=runtime),
        query.collect(engine="cpu"),
    )
