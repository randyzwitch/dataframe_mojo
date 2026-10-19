"""Bounded context-affine scratch reuse preserves results and query budgets."""
# contract-group: execution.backends
from std.testing import assert_equal, assert_raises, assert_true
from dataframe import Column, DataFrame, DataType, Series, col, lit
from dataframe.lazy import LazyFrame
from dataframe.metal import MetalRuntime
from dataframe.accelerator.gpu.query_scope import DeviceService, Scope
from dataframe.accelerator.gpu.typed_storage import _NativeReservation


def _sum_query(value: Int32, rows: Int = 513) raises -> LazyFrame:
    var source = DataFrame(
        [Series("x", Column[Int32](List[Int32](length=rows, fill=value)))]
    )
    return (
        source.lazy()
        .filter(col("x") > lit(Int32(0)))
        .select(
            ((col("x") + lit(Int32(3))) * lit(Int32(2)))
            .cast(DataType.INT64)
            .sum()
            .alias("total")
        )
    )


def _plain_sum_query(value: Int32, rows: Int = 513) raises -> LazyFrame:
    var source = DataFrame(
        [Series("x", Column[Int32](List[Int32](length=rows, fill=value)))]
    )
    return source.lazy().select(
        col("x").cast(DataType.INT64).sum().alias("total")
    )


def _nullable_sum_query() raises -> LazyFrame:
    var source = DataFrame([Series("x", Column[Int32]([2, 2], [True, False]))])
    return (
        source.lazy()
        .filter(col("x") > lit(Int32(0)))
        .select(
            ((col("x") + lit(Int32(3))) * lit(Int32(2)))
            .cast(DataType.INT64)
            .sum()
            .alias("total")
        )
    )


def test_mojo_gpu_scratch_limits_are_metadata_only() raises:
    var service = DeviceService(idle_scratch_bytes=24, idle_scratch_entries=1)
    assert_equal(service.cached_generations(), 0)
    assert_equal(service.scratch_info(), (0, 0, 0, 0))
    assert_equal(service.pipeline_info(), (0, 0, 0, 0))
    with assert_raises(contains="idle limit"):
        _ = DeviceService(idle_scratch_bytes=-1)
    with assert_raises(contains="idle limit"):
        _ = DeviceService(idle_scratch_entries=-1)


# requires: metal
def test_mojo_gpu_scratch_reuses_capacity_but_charges_each_query() raises:
    var runtime = MetalRuntime()
    var first = _sum_query(2).profile(engine="accel", accelerator=runtime)
    assert_equal(first[0].item(0, "total").int64(), Int64(5130))
    assert_equal(runtime._device_service.scratch_info(), (72, 4, 0, 4))
    var copied = runtime.copy()
    var second = _sum_query(4).profile(engine="accel", accelerator=copied)
    assert_equal(second[0].item(0, "total").int64(), Int64(7182))
    assert_equal(first[0].item(0, "total").int64(), Int64(5130))
    assert_equal(runtime._device_service.scratch_info(), (72, 4, 4, 4))
    assert_equal(second[1].item(0, "service_scratch_hits").int64(), Int64(4))
    assert_equal(runtime._device_service.cached_generations(), 1)
    var peak = Int(second[1].item(0, "peak_requested_bytes").int64())
    var bounded = runtime.copy()
    bounded._budget = peak - 1
    with assert_raises(contains="memory budget"):
        _ = _sum_query(2).collect(engine="accel", accelerator=bounded)
    assert_equal(runtime._device_service.scratch_info(), (72, 4, 4, 4))
    bounded._budget = peak
    var healthy = _sum_query(2).profile(engine="accel", accelerator=bounded)
    assert_equal(healthy[0].item(0, "total").int64(), Int64(5130))
    assert_equal(
        healthy[1].item(0, "peak_requested_bytes").int64(), Int64(peak)
    )
    assert_equal(runtime._device_service.scratch_info(), (72, 4, 8, 4))


# requires: metal
def test_mojo_gpu_scratch_idle_caps_and_failure_evict_cached_buffers() raises:
    var limited = MetalRuntime()
    limited._device_service = DeviceService(
        idle_scratch_bytes=24, idle_scratch_entries=1
    )
    _ = _sum_query(2).collect(engine="accel", accelerator=limited)
    assert_equal(limited._device_service.scratch_info(), (24, 1, 0, 4))
    _ = _sum_query(4).collect(engine="accel", accelerator=limited)
    assert_equal(limited._device_service.scratch_info(), (24, 1, 1, 7))
    var disabled = MetalRuntime()
    disabled._device_service = DeviceService(idle_scratch_bytes=0)
    _ = _sum_query(2).collect(engine="accel", accelerator=disabled)
    _ = _sum_query(4).collect(engine="accel", accelerator=disabled)
    assert_equal(disabled._device_service.scratch_info(), (0, 0, 0, 8))

    var runtime = MetalRuntime()
    _ = _sum_query(2).collect(engine="accel", accelerator=runtime)
    with assert_raises():
        _ = _sum_query(Int32.MAX, 1).collect(
            engine="accel", accelerator=runtime
        )
    assert_equal(runtime._device_service.scratch_info()[0], 0)
    assert_equal(runtime._device_service.scratch_info()[1], 0)
    var hits_after_failure = runtime._device_service.scratch_info()[2]
    var healthy = _sum_query(4).collect(engine="accel", accelerator=runtime)
    assert_equal(healthy.item(0, "total").int64(), Int64(7182))
    assert_equal(runtime._device_service.cached_generations(), 2)
    assert_equal(runtime._device_service.scratch_info()[2], hits_after_failure)


# requires: metal
def test_mojo_gpu_overlapping_private_scope_cannot_consume_idle_scratch() raises:
    var runtime = MetalRuntime()
    _ = _sum_query(2).collect(engine="accel", accelerator=runtime)
    var service = runtime._device_service.copy()
    var first = Scope(service=service.copy())
    _ = first.context()
    var overlapping = Scope(budget=12, service=service.copy())
    var reservation = _NativeReservation(overlapping, 12)
    var private_scratch = overlapping.scratch[DType.uint32](3)
    assert_equal(overlapping.uses_cached_context(), False)
    overlapping.synchronize()
    overlapping.recycle_scratch(private_scratch^)
    _ = reservation^
    assert_equal(overlapping.info()[0], 0)
    overlapping.mark_success()
    _ = overlapping^
    first.synchronize()
    first.mark_success()
    _ = first^
    assert_equal(service.scratch_info(), (72, 4, 0, 4))


# requires: metal
def test_mojo_gpu_fused_pipeline_cache_hits_variants_budget_and_poison() raises:
    var runtime = MetalRuntime()
    var first = _sum_query(2).profile(engine="accel", accelerator=runtime)
    assert_equal(first[0].item(0, "total").int64(), Int64(5130))
    assert_equal(runtime._device_service.pipeline_info(), (2, 0, 2, 0))
    assert_equal(first[1].item(0, "service_pipeline_entries").int64(), Int64(2))
    assert_equal(first[1].item(0, "service_pipeline_misses").int64(), Int64(2))
    assert_equal(
        first[1].item(0, "service_pipeline_capacity").int64(), Int64(9)
    )

    var repeated = _sum_query(4).profile(engine="accel", accelerator=runtime)
    assert_equal(repeated[0].item(0, "total").int64(), Int64(7182))
    assert_equal(runtime._device_service.pipeline_info(), (2, 2, 2, 0))
    assert_equal(repeated[1].item(0, "service_pipeline_hits").int64(), Int64(2))

    # Removing filter and row arithmetic selects another precompiled partial
    # specialization while reusing the common finish function.
    var plain = _plain_sum_query(7).profile(engine="accel", accelerator=runtime)
    assert_equal(plain[0].item(0, "total").int64(), Int64(3591))
    assert_equal(runtime._device_service.pipeline_info(), (3, 3, 3, 0))

    var nullable = _nullable_sum_query().profile(
        engine="accel", accelerator=runtime
    )
    assert_equal(nullable[0].item(0, "total").int64(), Int64(10))
    assert_equal(runtime._device_service.pipeline_info(), (4, 4, 4, 0))

    var budget = Int(repeated[1].item(0, "peak_requested_bytes").int64())
    var bounded = runtime.copy()
    bounded._budget = budget - 1
    with assert_raises(contains="memory budget"):
        _ = _sum_query(2).collect(engine="accel", accelerator=bounded)
    assert_equal(runtime._device_service.pipeline_info(), (4, 4, 4, 0))
    bounded._budget = budget
    var exact = _sum_query(2).profile(engine="accel", accelerator=bounded)
    assert_equal(exact[0].item(0, "total").int64(), Int64(5130))
    assert_true(
        exact[1].item(0, "peak_requested_bytes").int64() <= Int64(budget)
    )
    assert_equal(runtime._device_service.pipeline_info(), (4, 6, 4, 0))

    with assert_raises():
        _ = _sum_query(Int32.MAX, 1).collect(
            engine="accel", accelerator=runtime
        )
    assert_equal(runtime._device_service.pipeline_info(), (0, 8, 4, 1))
    var after_poison = _sum_query(4).collect(
        engine="accel", accelerator=runtime
    )
    assert_equal(after_poison.item(0, "total").int64(), Int64(7182))
    assert_equal(runtime._device_service.pipeline_info(), (2, 8, 6, 1))
    assert_equal(runtime._device_service.cached_generations(), 2)


# requires: metal
def test_mojo_gpu_fused_pipeline_cache_bypasses_overlapping_scope() raises:
    var service = DeviceService()
    var cached_scope = Scope(0, 4096, service=service.copy())
    _ = cached_scope.context()
    assert_true(cached_scope.uses_cached_context())

    var private_scope = Scope(0, 4096, service=service.copy())
    var private_context = private_scope.context()
    assert_equal(private_scope.uses_cached_context(), False)
    var values = private_context.enqueue_create_buffer[DType.int32](1)
    var validity = private_context.enqueue_create_buffer[DType.uint8](1)
    var sums = private_context.enqueue_create_buffer[DType.int64](1)
    var counts = private_context.enqueue_create_buffer[DType.uint32](1)
    var faults = private_context.enqueue_create_buffer[DType.uint32](1)
    var output = private_context.enqueue_create_buffer[DType.int64](3)

    private_scope.enqueue_fused_partials[False, False, False](
        values,
        validity,
        UInt32(0),
        UInt32(0),
        UInt32(256),
        Int32(0),
        Int32(0),
        Int32(1),
        sums,
        counts,
        faults,
        1,
    )
    private_scope.enqueue_fused_finish(sums, counts, faults, 1, output)
    private_scope.synchronize()
    assert_equal(service.pipeline_info(), (0, 0, 0, 0))
    private_scope.mark_success()
    cached_scope.synchronize()
    cached_scope.mark_success()
