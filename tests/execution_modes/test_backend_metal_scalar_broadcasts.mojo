"""Scalar reductions broadcast through resident GPU branches into row expressions."""
from dataframe.accelerator.gpu.runtime import (
    capabilities as _capabilities,
    estimate_resources,
)
from dataframe.accelerator.gpu.join_runtime import preflight as join_layout
from dataframe.accelerator.gpu.joins import METADATA_WORDS
from std.memory import bitcast
from tests.execution_modes.test_backend_metal import require_metal
from std.testing import assert_true, assert_equal, assert_raises
from dataframe import DataFrame, Series, Column, col, lit, when, scan_csv
from dataframe.dtype import DataType
from dataframe.lazy import LazyFrame
from dataframe.metal import MetalRuntime
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


def test_scalar_broadcast_rejects_file_reads_and_checked_integer_children() raises:
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
    assert_true("unsupported" not in runtime.describe(conditional))
    var wide = DataFrame([Series("x", Column[Int64]([Int64.MAX, Int64.MAX]))])
    var mean = wide.lazy().select(
        when(lit(False))
        .then(col("x").mean())
        .otherwise(col("x").cast(DataType.FLOAT64))
    )
    var planned = runtime.describe(mean)
    assert_true("Metal resident aggregate expressions" in planned)
    assert_true("unsupported" not in planned)
    var invalid_child = wide.lazy().select(
        when(lit(False))
        .then((col("x") + lit(Int64(1))).mean())
        .otherwise(col("x").cast(DataType.FLOAT64))
    )
    with assert_raises(contains="checked integer arithmetic"):
        _ = invalid_child.collect(accelerator=runtime)
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
    var report = measured[1].copy()
    var dag = lower_join_dag(
        lower_aggregate_compositions(query), _capabilities()
    )
    var resident = List[Bool](length=len(dag.tasks), fill=False)
    var sizes = List[Int](length=len(dag.tasks), fill=0)
    var region = 0
    for index in range(len(dag.tasks)):
        ref task = dag.tasks[index]
        if (
            task.kind == DAG_ROWS
            and task.source < 0
            and len(task.query._nodes) == 1
        ):
            # Original immutable CPU scans have no submitted region/report.
            sizes[index] = task.query._frames[0].height()
            continue
        assert_true(region < report.height())
        var output_rows = Int(report.item(region, "output_rows").int64())
        var observed_upload = report.item(region, "upload_bytes").int64()
        var observed_download = report.item(region, "download_bytes").int64()
        if task.kind == DAG_ROWS:
            var rows = Int(report.item(region, "input_rows").int64())
            var plan = lower_rows(
                task.query,
                _capabilities(),
                terminal_boundary=task.terminal,
                source_rows=rows,
            )
            var from_resident = task.source >= 0 and resident[task.source]
            var bounds = estimate_resources(
                plan, rows, from_resident, task.terminal
            )
            assert_true(observed_upload <= Int64(bounds.upload_bytes))
            assert_true(observed_download <= Int64(bounds.download_bytes))
            if from_resident:
                assert_true(
                    report.item(region, "resident_input_bytes").int64() > 0
                )
        else:
            var left_resident = resident[task.source]
            var right_resident = resident[task.right]
            var left = dag.tasks[task.source].schema.copy()
            var right = dag.tasks[task.right].schema.copy()
            var layout = join_layout(
                task, left, right, sizes[task.source], sizes[task.right]
            )
            # Only descriptor mappings accompany already resident values.
            var upload = 8 * (
                len(layout.columns) + len(layout.keys) + len(layout.outputs)
            )
            if not left_resident or not right_resident:
                upload += 21 * layout.cells
            if left_resident:
                upload += 24 * left.width()
            if right_resident:
                upload += 24 * right.width()
            assert_equal(observed_upload, Int64(upload))
            # Joins observe five control words, then output errors or the
            # terminal paired-word/validity/error value matrix.
            var bytes_per_cell = 21 if task.terminal else 4
            assert_equal(
                observed_download,
                Int64(
                    METADATA_WORDS * 8
                    + output_rows * len(task.fields) * bytes_per_cell
                ),
            )
        if not task.terminal:
            assert_equal(
                report.item(region, "host_result_bytes").int64(), Int64(0)
            )
            assert_equal(
                report.item(region, "materialization_boundary").string(),
                "Mojo GPU resident table",
            )
        resident[index] = True
        sizes[index] = output_rows
        region += 1
    assert_equal(region, report.height())
    with assert_raises(contains="budget"):
        _ = query.collect(accelerator=MetalRuntime(memory_limit_bytes=1))
    same(query.collect(accelerator=runtime), query.collect(engine="cpu"))
    same(
        query.collect(engine="auto", accelerator=runtime),
        query.collect(engine="cpu"),
    )


def test_native_metal_conditional_integer_means_limits_nulls_empty_and_partitions() raises:
    _ = require_metal()
    var frame = DataFrame(
        [
            Series(
                "k",
                Column[Int32](
                    [0, 0, 1, 1, 2, 2], [True, True, False, False, True, True]
                ),
            ),
            Series(
                "p",
                Column[Bool](
                    [True, False, True, False, True, False],
                    [True, True, False, True, True, False],
                ),
            ),
            Series(
                "i",
                Column[Int64](
                    [Int64.MIN, Int64.MIN, Int64.MAX, Int64.MAX, 0, 1],
                    [True, False, True, True, False, True],
                ),
            ),
            Series(
                "u",
                Column[UInt64](
                    [UInt64.MAX, UInt64.MAX, 1, UInt64.MAX, 2, UInt64.MAX],
                    [True, True, False, True, True, True],
                ),
            ),
        ]
    )
    var nulls = DataFrame(
        [
            Series("k", Column[Int32]([0, 0, 1])),
            Series("p", Column[Bool]([True, False, True])),
            Series(
                "i",
                Column[Int64](
                    [Int64.MIN, Int64.MAX, Int64.MAX], [False, False, False]
                ),
            ),
            Series(
                "u",
                Column[UInt64](
                    [UInt64.MAX, UInt64.MAX, UInt64.MAX], [False, False, False]
                ),
            ),
        ]
    )
    for sample in List[DataFrame](
        [frame.copy(), frame.slice(1, 4), frame.slice(0, 0), nulls^]
    ):
        for name in List[String](["i", "u"]):
            for predicate in [lit(False), lit(True), col("p")]:
                check(
                    sample.lazy().select(
                        when(predicate)
                        .then(col(name).mean())
                        .otherwise(col(name).cast(DataType.FLOAT64))
                        .alias("value")
                    )
                )
            check(
                sample.lazy()
                .group_by("k", maintain_order=True)
                .agg(
                    when(lit(False))
                    .then(col(name).mean())
                    .otherwise(lit(Float64(-1)))
                    .alias("value")
                )
            )
            check(
                sample.lazy()
                .group_by("k", maintain_order=True)
                .agg(
                    when(lit(True))
                    .then(col(name).mean())
                    .otherwise(lit(Float64(-1)))
                    .alias("value")
                )
            )
            check(
                sample.lazy().select(
                    when(col("p"))
                    .then(col(name).cast(DataType.FLOAT64))
                    .otherwise(col(name).mean())
                    .alias("value")
                )
            )
        check(
            sample.lazy()
            .with_columns(
                when(col("p"))
                .then(col("i").mean())
                .otherwise(col("u").mean())
                .alias("value")
            )
            .head(2)
        )
        check(
            sample.lazy().filter(
                when(col("p"))
                .then(col("u").mean())
                .otherwise(col("u").cast(DataType.FLOAT64))
                > lit(Float64(0))
            )
        )
    var extremes = DataFrame(
        [
            Series("i", Column[Int64]([Int64.MIN, Int64.MIN, Int64.MIN])),
            Series("u", Column[UInt64]([UInt64.MAX, UInt64.MAX, UInt64.MAX])),
        ]
    )
    check(
        extremes.lazy().select_exprs(
            [
                when(lit(True))
                .then(col("i").mean())
                .otherwise(lit(Float64(0)))
                .alias("i"),
                when(lit(True))
                .then(col("u").mean())
                .otherwise(lit(Float64(0)))
                .alias("u"),
            ]
        )
    )


def test_native_metal_conditional_integer_means_resident_joins_filters_and_budget() raises:
    _ = require_metal()
    var n = 8193
    var keys = List[Int32]()
    var values = List[UInt64]()
    for i in range(n):
        keys.append(Int32(i % 7))
        values.append(UInt64.MAX - UInt64(i % 9))
    var frame = DataFrame(
        [
            Series("k", Column[Int32](keys^)),
            Series("u", Column[UInt64](values^)),
        ]
    )
    var branch = (
        when(col("k") < lit(Int32(3)))
        .then(col("u").mean())
        .otherwise(col("u").cast(DataType.FLOAT64))
    )
    var right = DataFrame([Series("k", Column[Int32]([0, 1, 2, 3, 4, 5, 6]))])
    var query = frame.lazy().with_columns(branch.alias("value"))
    check(query.join(right.lazy(), "k").select_exprs([col("k"), col("value")]))
    check(
        frame.lazy()
        .join(right.lazy(), "k")
        .filter(branch > lit(Float64(0)))
        .head(17)
    )
    var runtime = MetalRuntime()
    var measured = query.profile(accelerator=runtime)
    same(measured[0], query.collect(engine="cpu"))
    with assert_raises(contains="budget"):
        _ = query.collect(accelerator=MetalRuntime(memory_limit_bytes=0))
    check(query)
