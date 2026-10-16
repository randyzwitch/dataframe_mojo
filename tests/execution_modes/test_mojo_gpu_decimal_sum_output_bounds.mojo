"""Original successful SUM bounds authorize only infallible consumers."""
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import DataFrame, Series, Column, col, lit
from dataframe.dtype import DataType
from dataframe.lazy import LazyFrame
from dataframe.binding import bind
from dataframe.accelerator.constants import fold_literal_text_parsing
from dataframe.accelerator.joins import lower_join_dag
from dataframe.accelerator import decimal_sum_bounds as raw
from dataframe.accelerator import decimal_sum_receipts
from dataframe.accelerator.gpu.runtime import capabilities, _preflight
from dataframe.metal import MetalRuntime
from tests.support.coverage_generators import offset_layout, chunk_layout
from tests.support.lifecycle_jobs import WorkerSetting


def source() raises -> DataFrame:
    var dtype = DataType.decimal(18, 2, 64)
    var keys = List[Int32]([-2, -2, 0, 1])
    var key_valid = List[Bool]([True, True, True, False])
    var sale = List[Int64]([1000, 0, 1000, Int64.MIN])
    var refund = List[Int64]([200, 200, 600, Int64.MAX])
    var zero = List[Int64]([0, 0, 0, 1])
    var valid = List[Bool]([True, True, True, False])
    return DataFrame(
        [
            Series("g", Column[Int32](keys^, key_valid^)),
            Series("sale", Column[Int64](sale^, valid)).with_dtype(dtype),
            Series("refund", Column[Int64](refund^, valid^)).with_dtype(dtype),
            Series("zero", Column[Int64](zero^)).with_dtype(dtype),
        ]
    )


def query(frame: DataFrame) raises -> LazyFrame:
    var dtype = frame.column("sale").dtype()
    return (
        frame.lazy()
        .group_by("g", maintain_order=True)
        .agg(
            [
                col("sale").sum(min_count=1).alias("sale"),
                (col("refund") + col("zero")).sum(min_count=1).alias("refund"),
            ]
        )
        .filter(col("sale") > lit("2").cast(dtype) * col("refund"))
        .select(["g"])
    )


def test_sum_output_bound_raw_math_and_original_slots() raises:
    var cap = raw.sum_bound(
        raw.Magnitude(Int128(1) << 63, True), Int(UInt32.MAX)
    )
    assert_true(cap.known)
    assert_equal(cap.maximum, Int128(UInt32.MAX) * (Int128(1) << 63))
    assert_true(
        raw.precision_safe(
            raw.product(cap, raw.Magnitude(200, True)), DataType.decimal(38, 4)
        )
    )
    assert_true(
        not raw.product(
            raw.Magnitude(Int128.MAX, True), raw.Magnitude(2, True)
        ).known
    )
    assert_true(not raw.sum_bound(cap, Int(UInt32.MAX) + 1).known)
    assert_true(not raw.physical(DataType.decimal(38, 2, 128)).known)
    var normalized = fold_literal_text_parsing(query(source()))
    var dag = lower_join_dag(
        normalized._with_accelerator_settings(False, True, 17), capabilities()
    )
    var receipts = 0
    for task in dag.tasks:
        if task.sum_receipt:
            receipts += 1
            var receipt = task.sum_receipt.value()
            assert_equal(receipt[].origin[].original_root, 1)
            assert_equal(receipt[].origin[].maxima[1].maximum, cap.maximum)
            assert_true(receipt[].origin[].retained_bytes > 0)
            assert_equal(
                receipt[].origin[].outputs[1].dtype(),
                DataType.decimal(38, 2, 128),
            )
            decimal_sum_receipts.validate(
                receipt[], task.source, dag.tasks[task.source].query, task.query
            )
            with assert_raises(contains="task lineage"):
                decimal_sum_receipts.validate(
                    receipt[],
                    task.source + 1,
                    dag.tasks[task.source].query,
                    task.query,
                )
            var forged = task.query.copy()
            forged._frames[0] = forged._frames[0].select(
                ["refund", "sale", "g"]
            )
            with assert_raises(contains="slot or dtype"):
                decimal_sum_receipts.validate(
                    receipt[], task.source, dag.tasks[task.source].query, forged
                )
    assert_equal(receipts, 1)


def test_mojo_gpu_sum_output_bounds_scaled_literal_nulls_layouts() raises:
    var frame = source()
    for layout in [frame.copy(), offset_layout(frame), chunk_layout(frame)]:
        var planned = query(layout)
        for workers in [1, 2]:
            var setting = WorkerSetting(workers)
            for streaming in [False, True]:
                for gpu in [False, True]:
                    var result: DataFrame
                    if gpu:
                        result = planned.collect(
                            engine="accel",
                            accelerator=MetalRuntime(),
                            optimize=False,
                            streaming=streaming,
                            batch_size=1,
                        )
                    else:
                        result = planned.collect(
                            engine="cpu",
                            optimize=False,
                            streaming=streaming,
                            batch_size=1,
                        )
                    assert_equal(result.height(), 1)
                    assert_equal(result.column("g").dtype(), DataType.INT32)
                    assert_equal(
                        result.column("g").int32().to_list(), [Int32(-2)]
                    )
            _ = setting^


def test_mojo_gpu_sum_output_bounds_preserve_producer_fault_and_budget() raises:
    var good = query(source())
    var measured = good.profile(
        engine="accel",
        accelerator=MetalRuntime(),
        optimize=False,
        streaming=True,
        batch_size=17,
    )
    var field = (
        "query_peak_requested_bytes" if "query_peak_requested_bytes"
        in measured[1].columns() else "peak_requested_bytes"
    )
    var peak = Int(measured[1].item(0, field).int64())
    var result = good.collect(
        engine="accel",
        accelerator=MetalRuntime(memory_limit_bytes=peak),
        optimize=False,
        streaming=True,
        batch_size=17,
    )
    assert_equal(result.column("g").int32().to_list(), [Int32(-2)])
    with assert_raises(contains="budget"):
        _ = good.collect(
            engine="accel",
            accelerator=MetalRuntime(device_id=999999, memory_limit_bytes=0),
            optimize=False,
            streaming=True,
        )
    var frame = source().with_column(
        Series("refund", Column[Int64]([Int64.MAX, 200, 600, 0])).with_dtype(
            DataType.decimal(18, 2, 64)
        )
    )
    for gpu in [False, True]:
        for streaming in [False, True]:
            with assert_raises(contains="decimal precision"):
                if gpu:
                    _ = (
                        query(frame)
                        .head(0)
                        .collect(
                            engine="accel",
                            accelerator=MetalRuntime(),
                            optimize=False,
                            streaming=streaming,
                        )
                    )
                else:
                    _ = (
                        query(frame)
                        .head(0)
                        .collect(
                            engine="cpu", optimize=False, streaming=streaming
                        )
                    )


def test_sum_output_bounds_refuse_arbitrary_wide_columns_and_scale18_products() raises:
    var payload = List[Int128]([1, 2])
    var wide = DataFrame(
        [
            Series("v", Column[Int128](payload^)).with_dtype(
                DataType.decimal(38, 2, 128)
            )
        ]
    )
    var arbitrary = wide.lazy().filter(
        col("v") > lit("2").cast(DataType.decimal(18, 2, 64)) * col("v")
    )
    with assert_raises(contains="checked decimal"):
        _ = _preflight(fold_literal_text_parsing(arbitrary))
    var dtype = DataType.decimal(18, 18, 64)
    var frame = DataFrame(
        [
            Series("g", Column[Int32]([0, 0])),
            Series("v", Column[Int64]([1, 1])).with_dtype(dtype),
            Series("z", Column[Int64]([0, 0])).with_dtype(dtype),
        ]
    )
    var unsafe = (
        frame.lazy()
        .group_by("g")
        .agg((col("v") + col("z")).sum(min_count=1).alias("s"))
        .filter(
            col("s") > col("s") * lit("2").cast(DataType.decimal(19, 18, 128))
        )
    )
    with assert_raises(contains="checked decimal"):
        _ = lower_join_dag(
            fold_literal_text_parsing(unsafe)._with_accelerator_settings(
                False, True, 17
            ),
            capabilities(),
        )
