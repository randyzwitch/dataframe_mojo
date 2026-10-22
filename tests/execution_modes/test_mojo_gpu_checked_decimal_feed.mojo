"""Checked Decimal operand batches remain observed before SUM consumers."""
# contract-group: execution.apple_parity
# contract-group: execution.backends
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Series, col
from dataframe.lazy import LazyFrame
from dataframe.decimal import pow10
from dataframe_metal.metal import MetalRuntime
from dataframe_metal.accelerator.joins import lower_join_dag
from dataframe_metal.accelerator.gpu.runtime import capabilities
from dataframe_metal.accelerator import decimal_feed
from tests.support.lifecycle_jobs import WorkerSetting
from tests.support.coverage_generators import offset_layout, chunk_layout
from tests.public_contracts.test_checked_decimal_feed_order import (
    feed_fault_fixture,
    feed_expression,
    test_checked_decimal_sum_operand_batches_precede_prefix_observation,
)


def _collect(query: LazyFrame, gpu: Bool) raises -> DataFrame:
    if gpu:
        return query.collect(
            engine="accel",
            accelerator=MetalRuntime(),
            optimize=False,
            streaming=False,
        )
    return query.collect(engine="cpu", optimize=False, streaming=False)


def _assert_total(frame: DataFrame, raw: Int128, valid: Bool = True) raises:
    assert_equal(frame.height(), 1)
    assert_equal(frame.column("total").dtype(), DataType.decimal(38, 4))
    assert_equal(frame.column("total")._decimal_valid(0), valid)
    if valid:
        assert_equal(frame.column("total")._decimal_raw(0), raw)


def _narrow(width: Int) raises -> DataFrame:
    var left = Series("a", Column[Int128]([100, 200, 0], [True, True, False]))
    var right = Series("b", Column[Int128]([10, 20, 30]))
    if width == 32:
        left = Series("a", Column[Int32]([100, 200, 0], [True, True, False]))
        right = Series("b", Column[Int32]([10, 20, 30]))
    elif width == 64:
        left = Series("a", Column[Int64]([100, 200, 0], [True, True, False]))
        right = Series("b", Column[Int64]([10, 20, 30]))
    var dtype = DataType.decimal(
        9 if width == 32 else 18 if width == 64 else 38, 2, width
    )
    return DataFrame([left.with_dtype(dtype), right.with_dtype(dtype)])


# requires: mojo_gpu
def test_checked_decimal_feed_metadata_retains_eager_boundary_and_signature() raises:
    var query = _narrow(128).lazy().select(feed_expression()).head(0)
    var configured = query._with_accelerator_settings(False, False, 17)
    var dag = lower_join_dag(configured, capabilities())
    var observed = 0
    for task in dag.tasks:
        for node in task.query._nodes:
            if node.text == decimal_feed.MATERIALIZATION:
                observed += 1
                assert_true(task.terminal)
                assert_equal(len(node.names2), 2)
    assert_equal(observed, 1)
    # A second fallible arithmetic occurrence has no ranked feed proof yet.
    var unsupported = (
        _narrow(128)
        .lazy()
        .select(((col("a") * col("b")) + col("a")).sum().alias("total"))
        .head(0)
        ._with_accelerator_settings(False, False, 17)
    )
    var strict = capabilities()
    strict.adjudicated_faults = False
    with assert_raises(contains="checked decimal arithmetic"):
        _ = lower_join_dag(unsupported, strict)
    # Metal decides faults on the CPU and needs no ranked feed proof.
    _ = lower_join_dag(unsupported, capabilities())


# requires: metal
def test_mojo_gpu_checked_decimal_sum_feed_success_layouts_nulls_and_consumers() raises:
    var workers = WorkerSetting(1)
    for width in [32, 64, 128]:
        var source = _narrow(width)
        for frame in [
            source.copy(),
            offset_layout(source),
            chunk_layout(source),
        ]:
            var producer = frame.lazy().select(feed_expression())
            for gpu in [False, True]:
                _assert_total(_collect(producer, gpu), 5000)
                _assert_total(
                    _collect(
                        producer.filter(col("total").is_not_null()).head(1), gpu
                    ),
                    5000,
                )
                var right = DataFrame(
                    [
                        Series("total", Column[Int128]([5000])).with_dtype(
                            DataType.decimal(38, 4)
                        ),
                        Series("label", Column[Int32]([7])),
                    ]
                )
                var joined = _collect(
                    producer.join(right.lazy(), on="total", how="left"), gpu
                )
                _assert_total(joined, 5000)
                assert_equal(joined.column("label").int32().to_list(), [7])
                var empty = frame.head(0).lazy().select(feed_expression())
                _assert_total(_collect(empty, gpu), 0)
                var nullable = (
                    frame.head(0)
                    .lazy()
                    .select(
                        (col("a") * col("b")).sum(min_count=1).alias("total")
                    )
                )
                _assert_total(_collect(nullable, gpu), 0, False)
    var measured_query = _narrow(128).lazy().select(feed_expression()).head(1)
    var measured = measured_query.profile(
        engine="accel",
        accelerator=MetalRuntime(),
        optimize=False,
        streaming=False,
    )
    var peak_field = (
        "query_peak_requested_bytes" if "query_peak_requested_bytes"
        in measured[1].columns() else "peak_requested_bytes"
    )
    var peak = Int(measured[1].item(0, peak_field).int64())
    _assert_total(
        measured_query.collect(
            engine="accel",
            accelerator=MetalRuntime(memory_limit_bytes=peak),
            optimize=False,
            streaming=False,
        ),
        5000,
    )
    with assert_raises(contains="budget"):
        _ = measured_query.collect(
            engine="accel",
            accelerator=MetalRuntime(device_id=999999, memory_limit_bytes=0),
            optimize=False,
            streaming=False,
        )
    _ = workers^


def _worker_fixture(
    early_prefix: Bool, later_operand: Bool
) raises -> DataFrame:
    var left = List[Int128](length=131072, fill=0)
    var right = List[Int128](length=131072, fill=1)
    left[0] = 6 * pow10(37)
    left[1 if early_prefix else 65536] = 6 * pow10(37)
    if later_operand:
        left[131071] = pow10(38) - 1
        right[131071] = 2
    return DataFrame(
        [
            Series("a", Column[Int128](left^)).with_dtype(
                DataType.decimal(38, 0)
            ),
            Series("b", Column[Int128](right^)).with_dtype(
                DataType.decimal(38, 0)
            ),
        ]
    )


# requires: metal
def test_mojo_gpu_checked_decimal_sum_feed_faults_before_head_and_empty_join() raises:
    test_checked_decimal_sum_operand_batches_precede_prefix_observation()
    var workers = WorkerSetting(1)
    var source = feed_fault_fixture()
    var producer = source.lazy().select(feed_expression())
    var empty = DataFrame(
        [
            Series("total", Column[Int128](List[Int128]())).with_dtype(
                DataType.decimal(38, 0)
            )
        ]
    )
    for query in [
        producer.copy(),
        producer.head(0),
        producer.join(empty.lazy(), on="total", how="inner"),
    ]:
        with assert_raises(contains="decimal precision"):
            _ = _collect(query, False)
        with assert_raises(contains="decimal precision"):
            _ = _collect(query, True)
    _ = workers^
    var parallel = WorkerSetting(2)
    for early_prefix in [False, True]:
        for later_operand in [False, True]:
            var query = (
                _worker_fixture(early_prefix, later_operand)
                .lazy()
                .select(feed_expression())
                .head(0)
            )
            var message = (
                "decimal multiplication overflow" if later_operand
                and not early_prefix else "decimal precision"
            )
            with assert_raises(contains=message):
                _ = _collect(query, False)
            with assert_raises(contains=message):
                _ = _collect(query, True)
    _ = parallel^
