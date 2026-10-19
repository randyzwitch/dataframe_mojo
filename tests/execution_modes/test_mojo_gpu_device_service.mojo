"""A Metal runtime shares one idle context and discards failed cached work."""
# contract-group: execution.backends
# contract-group: execution.apple_parity
from std.testing import assert_equal, assert_false, assert_true, assert_raises
from dataframe import Column, DataFrame, Series, col, lit
from dataframe.accelerator.gpu.query_scope import DeviceService, Scope
from dataframe.accelerator.gpu.runtime import _preflight
from dataframe.metal import MetalRuntime


def test_mojo_gpu_device_service_metadata_copy_and_budget_are_context_free() raises:
    var runtime = MetalRuntime(memory_limit_bytes=8)
    var copied = runtime.copy()
    assert_true(runtime._device_service.same_service(copied._device_service))
    assert_equal(runtime._device_service.device_id(), 0)
    assert_equal(runtime._device_service.cached_generations(), 0)

    var scope = Scope(0, 8, service=runtime._device_service)
    assert_equal(scope.info()[2], 8)
    assert_equal(scope.check(8), 8)
    with assert_raises():
        _ = scope.check(9)
    assert_equal(scope.info()[0], 0)
    assert_equal(scope.info()[1], 8)
    assert_equal(runtime._device_service.cached_generations(), 0)


def _complete_successful_scope(service: DeviceService) raises:
    var scope = Scope(service.device_id(), 4096, service=service)
    _ = scope.context()
    assert_true(scope.uses_cached_context())
    scope.synchronize()
    scope.mark_success()


def _overlap_scope_leases_private_context(service: DeviceService) raises:
    var first = Scope(service.device_id(), 4096, service=service)
    _ = first.context()
    assert_true(first.uses_cached_context())

    var overlapping = Scope(service.device_id(), 4096, service=service)
    _ = overlapping.context()
    assert_false(overlapping.uses_cached_context())

    overlapping.synchronize()
    overlapping.mark_success()
    first.synchronize()
    first.mark_success()


def _poison_cached_context(service: DeviceService) raises:
    var scope = Scope(service.device_id(), 4096, service=service)
    _ = scope.context()
    assert_true(scope.uses_cached_context())
    scope.synchronize()
    scope.mark_failed()


# requires: metal
def test_mojo_gpu_device_service_reuses_idle_context_and_discards_poison() raises:
    var service = DeviceService()
    _complete_successful_scope(service)
    assert_equal(service.cached_generations(), 1)

    _overlap_scope_leases_private_context(service)
    assert_equal(service.cached_generations(), 1)
    _complete_successful_scope(service)
    assert_equal(service.cached_generations(), 1)

    _poison_cached_context(service)
    _complete_successful_scope(service)
    assert_equal(service.cached_generations(), 2)


# requires: metal
def test_mojo_gpu_direct_runtime_reports_final_service_drain() raises:
    var runtime = MetalRuntime()
    var source = DataFrame([Series("x", Column[Int32]([1, 2, 3]))])
    var query = source.lazy().select_exprs([col("x")])
    var measured = query.profile(accelerator=runtime)
    assert_equal(measured[0].item(2, "x").int32(), 3)
    assert_true(measured[1].item(0, "synchronizations").int64() > 0)
    assert_equal(runtime._device_service.cached_generations(), 1)


# requires: metal
def test_mojo_gpu_device_service_repeated_typed_collections_share_context() raises:
    var runtime = MetalRuntime()
    var copied = runtime.copy()
    var first_source = DataFrame(
        [Series("x", Column[Float32]([Float32(1), Float32(2)]))]
    )
    var first_query = first_source.lazy().select(
        (col("x") + lit(Float32(1))).alias("value")
    )
    var first = first_query.collect(accelerator=runtime, engine="accel")
    assert_equal(first.item(0, "value").float32(), Float32(2))
    assert_equal(first.item(1, "value").float32(), Float32(3))
    assert_equal(runtime._device_service.cached_generations(), 1)

    var second_source = DataFrame(
        [Series("x", Column[Float32]([Float32(10), Float32(-4)]))]
    )
    var second_query = second_source.lazy().select(
        (col("x") * lit(Float32(2))).alias("value")
    )
    var second = second_query.collect(accelerator=copied, engine="accel")
    assert_equal(second.item(0, "value").float32(), Float32(20))
    assert_equal(second.item(1, "value").float32(), Float32(-8))
    # Later executions through the copied runtime do not mutate the first host result.
    assert_equal(first.item(0, "value").float32(), Float32(2))
    assert_equal(first.item(1, "value").float32(), Float32(3))
    assert_equal(runtime._device_service.cached_generations(), 1)

    var repeated = first_query.collect(accelerator=runtime, engine="accel")
    assert_equal(repeated.item(0, "value").float32(), Float32(2))
    assert_equal(repeated.item(1, "value").float32(), Float32(3))
    assert_equal(runtime._device_service.cached_generations(), 1)


# requires: metal
def test_mojo_gpu_device_service_empty_and_preflight_leave_no_context() raises:
    var runtime = MetalRuntime()
    var empty_source = DataFrame(
        [Series("x", Column[Float32](List[Float32]()))]
    )
    var empty_query = empty_source.lazy().select(
        (col("x") + lit(Float32(1))).alias("value")
    )
    var empty = empty_query.collect(accelerator=runtime, engine="accel")
    assert_equal(empty.height(), 0)
    assert_equal(runtime._device_service.cached_generations(), 0)

    var source = DataFrame([Series("x", Column[Int64](List[Int64]()))])
    var invalid_query = source.lazy().select(col("x"))
    with assert_raises(contains="source row override"):
        _ = _preflight(invalid_query, source_rows=Int(Int32.MAX) + 1)
    assert_equal(runtime._device_service.cached_generations(), 0)

    var zero_budget = MetalRuntime(memory_limit_bytes=0)
    var nonempty = DataFrame(
        [Series("x", Column[Float32]([Float32(1), Float32(2)]))]
    )
    with assert_raises(contains="memory budget"):
        _ = nonempty.lazy().select(col("x")).collect(
            accelerator=zero_budget, engine="accel"
        )
    assert_equal(zero_budget._device_service.cached_generations(), 0)


# requires: metal
def test_mojo_gpu_device_service_budget_rejection_preserves_warm_context() raises:
    var source = DataFrame(
        [Series("x", Column[Float32]([Float32(1), Float32(2)]))]
    )
    var healthy = source.lazy().select(
        (col("x") + lit(Float32(1))).alias("value")
    )
    var baseline = healthy.profile(accelerator=MetalRuntime(), engine="accel")
    var budget = Int(baseline[1].item(0, "peak_requested_bytes").int64())
    var runtime = MetalRuntime(memory_limit_bytes=budget)
    var warm = healthy.profile(accelerator=runtime, engine="accel")
    assert_true(
        warm[1].item(0, "peak_requested_bytes").int64()
        <= Int64(runtime._budget)
    )
    assert_equal(runtime._device_service.cached_generations(), 1)

    var large_values = List[Float32](length=4097, fill=Float32(3))
    var large = DataFrame([Series("x", Column[Float32](large_values^))])
    with assert_raises(contains="memory budget"):
        _ = large.lazy().select(col("x")).collect(
            accelerator=runtime, engine="accel"
        )
    assert_equal(runtime._budget, budget)
    assert_equal(runtime._device_service.cached_generations(), 1)

    var after = healthy.profile(accelerator=runtime, engine="accel")
    assert_equal(after[0].item(0, "value").float32(), Float32(2))
    assert_true(
        after[1].item(0, "peak_requested_bytes").int64()
        <= Int64(runtime._budget)
    )
    assert_equal(runtime._device_service.cached_generations(), 1)
