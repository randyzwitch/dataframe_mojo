"""Public grouped IMPLODE: real GPU layout, resident consumers and budgets."""
# contract-group: execution.apple_parity
# contract-group: execution.backends
from std.memory import ArcPointer, bitcast
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Expr, Series, col
from dataframe.metal import MetalRuntime
from dataframe.accelerator.grouped_lists import prepare
from dataframe.accelerator.gpu.calibration import estimate
from dataframe.accelerator.gpu.grouped_lists_runtime import (
    execute as execute_grouped_lists,
)
from tests.support.coverage_generators import offset_layout, chunk_layout
from tests.support.lifecycle_jobs import WorkerSetting
from dataframe.parallel import configured_workers
from dataframe.partition import low_cardinality, _mix


def _source() raises -> DataFrame:
    var x = Series(
        "x",
        Column[Int32](
            [10, 20, 30, 40, 50, 60], [True, False, True, True, True, True]
        ),
    )
    var metadata: List[UInt8] = [1, 0, 0, 0, 1, 0, 0, 0, 107, 1, 0, 0, 0, 118]
    x._field_metadata = ArcPointer(metadata^)
    var wide: List[Int128] = [Int128(1) << 80, -(Int128(1) << 80), 7, 8, 9, -7]
    return DataFrame(
        [
            Series(
                "k",
                Column[Int64](
                    [2, 1, 2, 3, 1, 2], [True, True, True, False, True, True]
                ),
            ),
            Series("j", Column[Float64]([0, -0.0, -0.0, 1, 0, 0])),
            x^,
            Series("wide", Column[Int128](wide^)).with_dtype(
                DataType.decimal(38, 0)
            ),
        ]
    )


def _expressions() -> List[Expr]:
    return [
        col("x").implode().alias("items"),
        col("x").sum().alias("sum"),
        col("wide").implode().alias("wide_items"),
        col("x").count().alias("count"),
    ]


# requires: metal
def test_mojo_gpu_grouped_implode_mixed_keys_children_reductions_layouts_and_residency() raises:
    var raw = _source()
    for layout in range(3):
        var frame = (
            raw.copy() if layout
            == 0 else offset_layout(raw) if layout
            == 1 else chunk_layout(raw)
        )
        var query = (
            frame.lazy()
            .group_by(["k", "j"], maintain_order=True)
            .agg(_expressions())
        )
        var expected = query.collect(
            engine="cpu", optimize=False, streaming=False
        )
        var profiled = query.profile(
            accelerator=MetalRuntime(), engine="accel", optimize=False
        )
        ref actual = profiled[0]
        assert_true(actual.equals(expected))
        assert_equal(expected.column("sum").dtype(), DataType.INT32)
        assert_equal(actual.column("sum").dtype(), DataType.INT32)
        assert_equal(actual.height(), 3)
        assert_equal(actual.item(0, "k").int64(), Int64(2))
        assert_equal(actual.item(1, "k").int64(), Int64(1))
        assert_true(actual.item(2, "k").is_null())
        assert_equal(
            bitcast[DType.uint64](actual.column("j").float64()._get(0)),
            UInt64(0),
        )
        assert_equal(
            bitcast[DType.uint64](actual.column("j").float64()._get(1)),
            UInt64(1) << 63,
        )
        assert_equal(
            actual.column("sum").int32().to_list(),
            [Int32(100), Int32(50), Int32(40)],
        )
        assert_equal(
            actual.column("count").int64().to_list(),
            [Int64(3), Int64(1), Int64(1)],
        )
        var items = actual.column("items").list_column()
        assert_equal(items._offsets[], [Int64(0), Int64(3), Int64(5), Int64(6)])
        assert_true(
            items.child().equals(
                Series(
                    "item",
                    Column[Int32](
                        [10, 30, 60, 20, 50, 40],
                        [True, True, True, False, True, True],
                    ),
                )
            )
        )
        assert_true(
            items.child()._field_metadata.value().ptr()
            == frame.column("x")._field_metadata.value().ptr()
        )
        ref report = profiled[1]
        assert_equal(
            report.item(0, "algorithm").string(),
            "mojo_gpu_resident_grouped_lists",
        )
        assert_equal(
            report.item(0, "list_descriptor_upload_bytes").int64(), Int64(0)
        )
        assert_equal(report.item(0, "resident_list_exports").int64(), Int64(2))
        # The intermediate Lists remain device owners across the next SELECT.
        # Primitive len/get output does not export any offsets/child List owner.
        var selected = query.select_exprs(
            [
                col("k"),
                col("items").list().len().alias("n"),
                col("wide_items").list().get(-1).alias("last"),
                col("sum"),
            ]
        )
        var consumed = selected.profile(
            accelerator=MetalRuntime(), engine="accel", optimize=False
        )
        assert_true(
            consumed[0].equals(
                selected.collect(engine="cpu", optimize=False, streaming=False)
            )
        )
        assert_equal(
            consumed[0].column("n").int64().to_list(),
            [Int64(3), Int64(2), Int64(1)],
        )
        assert_equal(
            consumed[0].column("last")._data[Column[Int128]]._get(0), Int128(-7)
        )
        assert_equal(
            consumed[0].column("last")._data[Column[Int128]]._get(1), Int128(9)
        )
        assert_equal(
            consumed[0].column("last")._data[Column[Int128]]._get(2), Int128(8)
        )
        assert_equal(
            consumed[1].item(0, "resident_list_exports").int64(), Int64(0)
        )


# requires: metal
def test_mojo_gpu_grouped_implode_empty_nulls_projection_errors_head_and_exact_budget() raises:
    var raw = _source()
    var query = (
        raw.lazy().group_by(["k", "j"], maintain_order=True).agg(_expressions())
    )
    var measured = query.profile(
        accelerator=MetalRuntime(), engine="accel", optimize=False
    )
    var peak = Int(measured[1].item(0, "query_peak_requested_bytes").int64())
    assert_true(peak > 0)
    assert_true(
        query.collect(
            accelerator=MetalRuntime(memory_limit_bytes=peak),
            engine="accel",
            optimize=False,
        ).equals(measured[0])
    )
    with assert_raises(contains="memory budget before submission"):
        _ = query.collect(
            accelerator=MetalRuntime(memory_limit_bytes=peak - 1),
            engine="accel",
            optimize=False,
        )
    assert_true(
        query.collect(
            accelerator=MetalRuntime(), engine="accel", optimize=False
        ).equals(measured[0])
    )
    var observed = execute_grouped_lists(
        query, 0, -1, True, observation_only=True
    )
    assert_equal(observed[0].height(), 0)
    assert_equal(
        observed[1].item(0, "query_host_result_bytes").int64(), Int64(0)
    )
    assert_equal(observed[1].item(0, "resident_list_exports").int64(), Int64(0))
    assert_true(
        observed[1].item(0, "query_download_bytes").int64()
        < measured[1].item(0, "query_download_bytes").int64()
    )
    var empty = (
        raw.head(0)
        .lazy()
        .group_by(["k", "j"], maintain_order=True)
        .agg(_expressions())
    )
    var empty_result = empty.collect(
        accelerator=MetalRuntime(device_id=999999, memory_limit_bytes=0),
        engine="accel",
        optimize=False,
    )
    assert_equal(empty_result.height(), 0)
    assert_true(
        empty_result.equals(
            empty.collect(engine="cpu", optimize=False, streaming=False)
        )
    )
    var faults = DataFrame(
        [
            Series("k", Column[Int32]([0, 0])),
            Series("u", Column[UInt64]([UInt64.MAX, 0])),
            Series("v", Column[Int64]([Int64.MAX, 1])),
        ]
    )
    for first in [True, False]:
        var checked = col("u").cast(DataType.INT32).implode().alias("items")
        var sum = col("v").sum().alias("sum")
        var expressions = List[Expr](
            [
                checked.copy(),
                sum.copy(),
            ]
        ) if first else List[Expr]([sum.copy(), checked.copy()])
        var failing = (
            faults.lazy().group_by("k", maintain_order=True).agg(expressions)
        )
        with assert_raises(
            contains="cast from uint64 to int32 failed" if first else "overflow"
        ):
            _ = failing.collect(engine="cpu", optimize=False, streaming=False)
        with assert_raises(
            contains="strict cast failed" if first else "overflow"
        ):
            _ = failing.head(0).collect(
                accelerator=MetalRuntime(), engine="accel", optimize=False
            )
    var all_null = (
        DataFrame(
            [
                Series("k", Column[Int32]([0, 0, 1])),
                Series("x", Column[Int64]([7, 8, 9], [False, False, False])),
            ]
        )
        .lazy()
        .group_by("k", maintain_order=True)
        .agg(col("x").implode().alias("items"))
    )
    var result = all_null.collect(
        accelerator=MetalRuntime(), engine="accel", optimize=False
    )
    assert_equal(result.column("items").null_count(), 0)
    assert_equal(result.column("items").list_column().child().null_count(), 3)
    assert_true(
        result.equals(
            all_null.collect(engine="cpu", optimize=False, streaming=False)
        )
    )


# requires: metal
def test_mojo_gpu_grouped_implode_metadata_admission_and_parallel_schedule_guards() raises:
    var frame = _source()
    var query = (
        frame.lazy()
        .group_by(["k", "j"], maintain_order=True)
        .agg(_expressions())
    )
    var description = MetalRuntime(device_id=999999).describe(query)
    assert_true("resident grouped primitive Lists" in description)
    with assert_raises(contains="no analytical calibration"):
        _ = estimate(query)
    var mixed = (
        frame.lazy().group_by("k").agg((col("x") + col("x").sum()).implode())
    )
    # A naked nested SUM is rejected by the public CPU binder before any
    # provider-specific admission. OVER gives the state a valid row shape.
    with assert_raises(
        contains="Aggregate input must be row-valued without nested aggregates"
    ):
        _ = mixed.collect(engine="cpu", optimize=False, streaming=False)
    with assert_raises(
        contains="Aggregate input must be row-valued without nested aggregates"
    ):
        _ = prepare(mixed)
    var stateful = (
        frame.lazy()
        .group_by("k", maintain_order=True)
        .agg((col("x") + col("x").sum().over("k")).implode().alias("items"))
    )
    var stateful_cpu = stateful.collect(
        engine="cpu", optimize=False, streaming=False
    )
    var values = stateful_cpu.column("items").list_column().child()
    assert_true(
        values.equals(
            Series(
                "item",
                Column[Int32](
                    [110, 130, 160, 0, 100, 80],
                    [True, True, True, False, True, True],
                ),
            )
        )
    )
    with assert_raises(contains="state-containing operands"):
        _ = prepare(stateful)
    var checked_reducer = (
        DataFrame(
            [
                Series("k", Column[Int32]([0])),
                Series("x", Column[Int64]([128])),
                Series("base", Column[Int64]([2])),
                Series("exp", Column[Int64]([-1])),
            ]
        )
        .lazy()
        .group_by("k")
        .agg(
            [
                col("base").implode().alias("items"),
                (
                    col("x").cast(DataType.INT8).cast(DataType.INT64)
                    + col("base").pow(col("exp"))
                )
                .sum()
                .alias("sum"),
            ]
        )
    )
    with assert_raises(contains="cast from int64 to int8 failed"):
        _ = checked_reducer.collect(
            engine="cpu", optimize=False, streaming=False
        )
    with assert_raises(contains="source provenance"):
        _ = prepare(checked_reducer)
    assert_true(
        "resident grouped primitive Lists"
        in MetalRuntime(device_id=999999).describe(checked_reducer)
    )
    var large = (
        DataFrame(
            [
                Series("k", Column[Int32](List[Int32](length=32768, fill=0))),
                Series("x", Column[Int64](List[Int64](length=32768, fill=1))),
            ]
        )
        .lazy()
        .group_by("k")
        .agg([col("x").implode().alias("items"), col("x").sum().alias("sum")])
    )
    with assert_raises(contains="parallel bucket/job"):
        _ = prepare(large)
    assert_true(
        "parallel bucket/job" in MetalRuntime(device_id=999999).describe(large)
    )


# requires: metal
def test_mojo_gpu_grouped_implode_large_checked_cpu_bucket_and_heavy_fault_order() raises:
    assert_equal(_mix(UInt64(4)), UInt64(0xB7A4712C74562914))
    for kind in range(3):
        var setting = WorkerSetting(8 if kind == 2 else 4)
        assert_equal(configured_workers(), 8 if kind == 2 else 4)
        var n = 262145 if kind == 2 else 65537
        var hot = 2 * n // 5
        var keys = List[Int64]()
        for row in range(n):
            var key = Int64(0) if kind == 0 else Int64(row)
            if kind == 2:
                key = (
                    Int64(4) if row
                    < hot else Int64(0) if row
                    == hot else Int64(1000000 + row)
                )
            keys.append(key)
        var integers = List[Int64](length=n, fill=0)
        var exponents = List[Int64](length=n, fill=0)
        integers[hot if kind == 2 else 4] = 128
        exponents[0] = -1
        var frame = DataFrame(
            [
                Series("k", Column[Int64](keys^)),
                Series("u", Column[Int64](integers^)),
                Series("base", Column[Int64](List[Int64](length=n, fill=2))),
                Series("exp", Column[Int64](exponents^)),
            ]
        )
        assert_equal(low_cardinality([frame.column("k")]), kind == 0)
        var query = (
            frame.lazy()
            .group_by("k", maintain_order=True)
            .agg(
                [
                    col("u").cast(DataType.INT8).implode().alias("first"),
                    col("base").pow(col("exp")).implode().alias("later"),
                    col("u").count().alias("count"),
                ]
            )
        )
        assert_true(prepare(query).scheduled)
        with assert_raises(
            contains="cast from int64 to int8 failed" if kind
            == 0 else "nonnegative exponent"
        ):
            _ = query.collect(engine="cpu", optimize=False, streaming=False)
        for head in List[Int]([-1, 0]):
            var selected = query.copy() if head < 0 else query.head(head)
            with assert_raises(
                contains="strict cast failed" if kind
                == 0 else "nonnegative exponent"
            ):
                _ = selected.collect(
                    accelerator=MetalRuntime(), engine="accel", optimize=False
                )
        _ = setting^
    # A successful checked large producer uses the same path, with nullable
    # payload and stable group/source order retained through len/get consumers.
    var n = 65537
    var keys = List[Int32]()
    var values = List[Int64]()
    var valid = List[Bool]()
    for row in range(n):
        keys.append(Int32(row % 257))
        values.append(Int64(row % 100))
        valid.append(row % 11 != 0)
    var frame = DataFrame(
        [
            Series("k", Column[Int32](keys^)),
            Series("x", Column[Int64](values^, valid^)),
        ]
    )
    var query = (
        frame.lazy()
        .group_by("k", maintain_order=True)
        .agg(
            [
                col("x").cast(DataType.INT8).implode().alias("items"),
                col("x").count().alias("count"),
            ]
        )
        .select_exprs(
            [
                col("k"),
                col("count"),
                col("items").list().len().alias("length"),
                col("items").list().get(-1).alias("last"),
            ]
        )
    )
    var setting = WorkerSetting(4)
    assert_equal(configured_workers(), 4)
    var expected = query.collect(engine="cpu", optimize=False, streaming=False)
    var runtime = MetalRuntime()
    var measured = runtime.execute_profiled(query)
    assert_true(measured[0].equals(expected))
    for group in range(257):
        var count = (n - 1 - group) // 257 + 1
        var last = group + (count - 1) * 257
        assert_equal(measured[0].item(group, "length").int64(), Int64(count))
        assert_equal(measured[0].item(group, "last").is_null(), last % 11 == 0)
        if last % 11 != 0:
            assert_equal(
                measured[0].item(group, "last").int8(), Int8(last % 100)
            )
    var peak = Int(measured[1].item(0, "query_peak_requested_bytes").int64())
    var exact = query.collect(
        accelerator=MetalRuntime(memory_limit_bytes=peak),
        engine="accel",
        optimize=False,
    )
    assert_true(exact.equals(expected))
    with assert_raises(contains="memory budget before submission"):
        _ = query.collect(
            accelerator=MetalRuntime(memory_limit_bytes=peak - 1),
            engine="accel",
            optimize=False,
        )
    _ = setting^
