"""Physical-width proof for checked narrow Decimal GROUP operands."""
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Expr,
    Series,
    StringColumn,
    col,
    lit,
    when,
)
from dataframe.lazy import LazyFrame
from dataframe.accelerator import decimal_group_feed
from dataframe.accelerator.joins import lower_join_dag
from dataframe.accelerator.gpu.runtime import capabilities
from dataframe.metal import MetalRuntime
from tests.support.lifecycle_jobs import WorkerSetting
from tests.support.coverage_generators import offset_layout, chunk_layout


def _source(nullable: Bool = False) raises -> DataFrame:
    return DataFrame(
        [
            Series("g", Column[Int32]([0, 1, 0])),
            Series(
                "price",
                Column[Int64]([100, 200, 300], [True, True, not nullable]),
            ).with_dtype(DataType.decimal(15, 2, 64)),
            Series("one", Column[Int32]([100, 100, 100])).with_dtype(
                DataType.decimal(9, 2, 32)
            ),
            Series("discount", Column[Int32]([10, 20, 30])).with_dtype(
                DataType.decimal(9, 2, 32)
            ),
            Series("tax", Column[Int32]([110, 110, 110])).with_dtype(
                DataType.decimal(9, 2, 32)
            ),
        ]
    )


def _expressions() -> List[Expr]:
    var discount = col("price") * (col("one") - col("discount"))
    return [
        discount.sum().alias("total"),
        (discount * col("tax")).sum().alias("charge"),
        discount.mean().alias("average"),
        col("price").count().alias("count"),
    ]


def _collect(
    query: LazyFrame, gpu: Bool, streaming: Bool, batch: Int
) raises -> DataFrame:
    if gpu:
        return query.collect(
            engine="accel",
            accelerator=MetalRuntime(),
            optimize=False,
            streaming=streaming,
            batch_size=batch,
        )
    return query.collect(
        engine="cpu", optimize=False, streaming=streaming, batch_size=batch
    )


def _literal(frame: DataFrame, nullable: Bool = False) raises:
    assert_equal(frame.height(), 2)
    assert_equal(frame.column("g").int32().to_list(), [0, 1])
    assert_equal(frame.column("total").dtype(), DataType.decimal(38, 4))
    assert_equal(frame.column("charge").dtype(), DataType.decimal(38, 6))
    assert_equal(
        frame.column("total")._decimal_raw(0),
        Int128(9000 if nullable else 30000),
    )
    assert_equal(frame.column("total")._decimal_raw(1), Int128(16000))
    assert_equal(
        frame.column("charge")._decimal_raw(0),
        Int128(990000 if nullable else 3300000),
    )
    assert_equal(frame.column("charge")._decimal_raw(1), Int128(1760000))
    assert_equal(
        frame.column("count").int64().to_list(),
        [Int64(1 if nullable else 2), Int64(1)],
    )
    assert_equal(
        frame.column("average").float64().to_list(),
        [Float64(0.9 if nullable else 1.5), Float64(1.6)],
    )


def test_decimal_group_feed_certificate_preserves_whole_original_agg() raises:
    var source = _source()
    var query = (
        source.lazy()
        .group_by("g", maintain_order=True)
        .agg(_expressions())
        .head(0)
    )
    for streaming in [False, True]:
        var configured = query._with_accelerator_settings(False, streaming, 17)
        var dag = lower_join_dag(configured, capabilities())
        var observed = 0
        for task in dag.tasks:
            for node in task.query._nodes:
                if node.text == decimal_group_feed.MATERIALIZATION:
                    observed += 1
                    assert_true(task.terminal)
                    assert_equal(len(node.names2), 7)
                    assert_equal(node.names2[2], "g")
                    assert_equal(len(node.exprs), 4)
        assert_equal(observed, 1)
    assert_true(
        decimal_group_feed.infallible_sum(
            DataType.decimal(18, 18, 64), Int(UInt32.MAX)
        )
    )
    assert_true(
        not decimal_group_feed.infallible_sum(
            DataType.decimal(38, 0), Int(UInt32.MAX)
        )
    )
    # An implicit Decimal WHEN common cast changes stored scale and can
    # throw separately. The narrow arithmetic proof must not certify it.
    var mixed = DataFrame(
        [
            Series("g", Column[Int32]([0, 0])),
            Series("pick", Column[Int32]([1, 0])),
            Series("a", Column[Int64]([100, 999])).with_dtype(
                DataType.decimal(15, 2, 64)
            ),
            Series("b", Column[Int64]([777, 200])).with_dtype(
                DataType.decimal(15, 4, 64)
            ),
            Series("factor", Column[Int64]([1, 1])).with_dtype(
                DataType.decimal(15, 0, 64)
            ),
        ]
    )
    var aligned = (
        when(col("pick") == lit(Int32(1))).then(col("a")).otherwise(col("b"))
    )
    var mixed_query = (
        mixed.lazy()
        .group_by("g")
        .agg((aligned * col("factor")).sum().alias("total"))
    )
    var cpu = mixed_query.collect(
        engine="cpu", optimize=False, streaming=True, batch_size=1
    )
    assert_equal(cpu.column("total").dtype(), DataType.decimal(38, 4))
    assert_equal(cpu.column("total")._decimal_raw(0), Int128(10200))
    with assert_raises(contains="checked decimal"):
        _ = lower_join_dag(
            mixed_query.head(0)._with_accelerator_settings(False, True, 1),
            capabilities(),
        )
    # Unary MIN/NEG and wide precision intermediates are different categories,
    # never covered by a narrow ADD/SUB/MUL proof.
    for expression in [
        (-col("price")).sum(),
        (col("price").cast(DataType.decimal(38, 2)) * col("tax")).sum(),
    ]:
        var unsupported = (
            source.lazy()
            .group_by("g")
            .agg(expression)
            .head(0)
            ._with_accelerator_settings(False, True, 17)
        )
        with assert_raises(contains="checked decimal"):
            _ = lower_join_dag(unsupported, capabilities())


def test_mojo_gpu_decimal_group_feed_literals_nulls_layouts_and_masks() raises:
    var workers = WorkerSetting(1)
    for nullable in [False, True]:
        var source = _source(nullable)
        for frame in [
            source.copy(),
            offset_layout(source),
            chunk_layout(source),
        ]:
            var query = (
                frame.lazy()
                .group_by("g", maintain_order=True)
                .agg(_expressions())
            )
            for streaming in [False, True]:
                for gpu in [False, True]:
                    _literal(_collect(query, gpu, streaming, 17), nullable)
    var text = (
        _source()
        .with_column(Series("returnflag", StringColumn(["N", "R", "N"])))
        .with_column(Series("status", StringColumn(["O", "F", "O"])))
    )
    var text_query = (
        text.lazy()
        .group_by(["returnflag", "status"], maintain_order=True)
        .agg(_expressions())
    )
    for gpu in [False, True]:
        var result = _collect(text_query, gpu, True, 17)
        assert_equal(result.column("returnflag").string().to_list(), ["N", "R"])
        assert_equal(result.column("status").string().to_list(), ["O", "F"])
        assert_equal(result.column("total")._decimal_raw(0), Int128(30000))
        assert_equal(result.column("total")._decimal_raw(1), Int128(16000))
    # An inactive checked multiply must not produce a precision fault.
    var masked = DataFrame(
        [
            Series("g", Column[Int32]([0, 0])),
            Series("a", Column[Int64]([Int64.MAX, 2])).with_dtype(
                DataType.decimal(18, 0, 64)
            ),
            Series("b", Column[Int64]([Int64.MAX, 3])).with_dtype(
                DataType.decimal(18, 0, 64)
            ),
            Series("pick", Column[Int32]([0, 1])),
        ]
    )
    # Keep the false arm valid while its true-arm operands remain extreme.
    var value = (
        when(col("pick") == lit(Int32(1)))
        .then(col("a") * col("b"))
        .otherwise(lit(Int64(0)).cast(DataType.decimal(18, 0, 64)))
    )
    for gpu in [False, True]:
        var result = _collect(
            masked.lazy().group_by("g").agg(value.sum().alias("total")),
            gpu,
            True,
            1,
        )
        assert_equal(result.column("total")._decimal_raw(0), Int128(6))
    var null_operand = masked.with_column(
        Series("a", Column[Int64]([Int64.MAX, 2], [False, True])).with_dtype(
            DataType.decimal(18, 0, 64)
        )
    )
    for gpu in [False, True]:
        var result = _collect(
            null_operand.lazy()
            .group_by("g")
            .agg((col("a") * col("b")).sum().alias("total")),
            gpu,
            True,
            1,
        )
        assert_equal(result.column("total")._decimal_raw(0), Int128(6))
    _ = workers^


def test_mojo_gpu_decimal_group_feed_stream_workers_and_precision_faults() raises:
    var rows = 131072
    var groups = List[Int32](length=rows, fill=0)
    var price = List[Int64](length=rows, fill=100)
    var one = List[Int32](length=rows, fill=100)
    var discount = List[Int32](length=rows, fill=10)
    var tax = List[Int32](length=rows, fill=110)
    for row in range(rows):
        groups[row] = Int32(row % 2)
    var source = DataFrame(
        [
            Series("g", Column[Int32](groups^)),
            Series("price", Column[Int64](price^)).with_dtype(
                DataType.decimal(15, 2, 64)
            ),
            Series("one", Column[Int32](one^)).with_dtype(
                DataType.decimal(9, 2, 32)
            ),
            Series("discount", Column[Int32](discount^)).with_dtype(
                DataType.decimal(9, 2, 32)
            ),
            Series("tax", Column[Int32](tax^)).with_dtype(
                DataType.decimal(9, 2, 32)
            ),
        ]
    )
    for count in [1, 2, 8]:
        var workers = WorkerSetting(count)
        for streaming in [False, True]:
            for batch in [17, 8193]:
                var query = (
                    source.lazy()
                    .group_by("g", maintain_order=True)
                    .agg(_expressions())
                )
                for gpu in [False, True]:
                    var result = _collect(query, gpu, streaming, batch)
                    assert_equal(result.column("g").int32().to_list(), [0, 1])
                    assert_equal(
                        result.column("total")._decimal_raw(0),
                        Int128(589824000),
                    )
                    assert_equal(
                        result.column("total")._decimal_raw(1),
                        Int128(589824000),
                    )
                    assert_equal(
                        result.column("charge")._decimal_raw(0),
                        Int128(64880640000),
                    )
                    assert_equal(
                        result.column("count").int64().to_list(),
                        [Int64(65536), Int64(65536)],
                    )
        _ = workers^
    # Distinct scales and multiple original arithmetic occurrences all map
    # to precision, including physical extrema retagged beyond dtype digits.
    var bad = DataFrame(
        [
            Series("g", Column[Int32]([0, 1, 0])),
            Series("a", Column[Int64]([Int64.MAX, Int64.MIN, 1])).with_dtype(
                DataType.decimal(18, 0, 64)
            ),
            Series("b", Column[Int64]([Int64.MAX, 1, 1])).with_dtype(
                DataType.decimal(18, 0, 64)
            ),
            Series("c", Column[Int64]([Int64.MAX, Int64.MIN, 1])).with_dtype(
                DataType.decimal(18, 18, 64)
            ),
        ]
    )
    var expressions: List[Expr] = [
        (col("a") * (col("b") - col("a"))).sum().alias("total"),
        (col("a") + col("c")).sum().alias("scaled"),
        (col("a") * col("b")).sum().alias("product"),
    ]
    var empty = DataFrame([Series("g", Column[Int32](List[Int32]()))])
    # Rotate the original sibling order so ADD/SUB/MUL are each the CPU's
    # first throwing projection, including different precision/scale metadata.
    for first in range(len(expressions)):
        var ordered = List[Expr]()
        for index in range(len(expressions)):
            ordered.append(
                expressions[(first + index) % len(expressions)].copy()
            )
        var producer = bad.lazy().group_by("g").agg(ordered)
        for streaming in [False, True]:
            for gpu in [False, True]:
                with assert_raises(contains="decimal precision"):
                    _ = _collect(producer, gpu, streaming, 17)
    var producer = bad.lazy().group_by("g").agg(expressions)
    for count in [1, 2, 8]:
        var workers = WorkerSetting(count)
        for streaming in [False, True]:
            for batch in [1, 8193]:
                for query in [
                    producer.copy(),
                    producer.head(0),
                    producer.join(empty.lazy(), on="g", how="inner"),
                ]:
                    for gpu in [False, True]:
                        with assert_raises(contains="decimal precision"):
                            _ = _collect(query, gpu, streaming, batch)
        _ = workers^


def test_mojo_gpu_decimal_group_feed_budget_precedes_device() raises:
    var query = (
        _source()
        .lazy()
        .group_by("g", maintain_order=True)
        .agg(_expressions())
        .head(1)
    )
    var measured = query.profile(
        engine="accel",
        accelerator=MetalRuntime(),
        optimize=False,
        streaming=True,
        batch_size=17,
    )
    var peak_field = (
        "query_peak_requested_bytes" if "query_peak_requested_bytes"
        in measured[1].columns() else "peak_requested_bytes"
    )
    var peak = Int(measured[1].item(0, peak_field).int64())
    var result = query.collect(
        engine="accel",
        accelerator=MetalRuntime(memory_limit_bytes=peak),
        optimize=False,
        streaming=True,
        batch_size=17,
    )
    assert_equal(result.column("total")._decimal_raw(0), Int128(30000))
    with assert_raises(contains="budget"):
        _ = query.collect(
            engine="accel",
            accelerator=MetalRuntime(device_id=999999, memory_limit_bytes=0),
            optimize=False,
            streaming=True,
            batch_size=17,
        )
