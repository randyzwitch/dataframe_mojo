"""A Metal runtime shares one idle context and discards failed cached work."""
# contract-group: execution.backends
# contract-group: execution.apple_parity
from std.testing import assert_equal, assert_false, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Series, col, lit
from dataframe.accelerator.gpu.query_scope import DeviceService, Scope
from dataframe.accelerator.gpu.dag_runtime import execute as execute_dag
from dataframe.accelerator.gpu.runtime import capabilities
from dataframe.accelerator.gpu.runtime import _preflight
from dataframe.metal import MetalRuntime
from tests.support.resident_grouped_publication_reference import (
    _source,
    _expressions,
    _literal,
)


def _region_synchronizations(report: DataFrame) raises -> Int64:
    var total = Int64(0)
    for row in range(report.height()):
        total += report.item(row, "synchronizations").int64()
    return total


def _assert_region_synchronization_ms_sums(report: DataFrame) raises:
    assert_true(report.height() > 1)
    var total = Float64(0)
    for row in range(report.height()):
        var measured = report.item(row, "synchronization_ms")
        assert_true(not measured.is_null())
        total += measured.float64()
    var query_total = report.item(0, "query_synchronization_ms")
    assert_true(not query_total.is_null())
    var difference = total - query_total.float64()
    # Each region is rounded to milliseconds separately; allow their combined
    # sub-microsecond rounding error, without asserting an absolute duration.
    var rounding_tolerance = Float64(report.height()) / 1000
    assert_true(difference <= rounding_tolerance)
    assert_true(difference >= -rounding_tolerance)


def test_mojo_gpu_device_service_metadata_copy_and_budget_are_context_free() raises:
    var runtime = MetalRuntime(memory_limit_bytes=8)
    var copied = runtime.copy()
    assert_true(runtime._device_service.same_service(copied._device_service))
    assert_equal(runtime._device_service.device_id(), 0)
    assert_equal(runtime._device_service.cached_generations(), 0)

    var scope = Scope(0, 8, service=runtime._device_service.copy())
    assert_equal(scope.info()[2], 8)
    assert_equal(scope.check(8), 8)
    with assert_raises():
        _ = scope.check(9)
    assert_equal(scope.info()[0], 0)
    assert_equal(scope.info()[1], 8)
    assert_equal(runtime._device_service.cached_generations(), 0)


def _complete_successful_scope(service: DeviceService) raises:
    var scope = Scope(service.device_id(), 4096, service=service.copy())
    _ = scope.context()
    assert_true(scope.uses_cached_context())
    scope.synchronize()
    scope.mark_success()


def _overlap_scope_leases_private_context(service: DeviceService) raises:
    var first = Scope(service.device_id(), 4096, service=service.copy())
    _ = first.context()
    assert_true(first.uses_cached_context())

    var overlapping = Scope(service.device_id(), 4096, service=service.copy())
    _ = overlapping.context()
    assert_false(overlapping.uses_cached_context())

    overlapping.synchronize()
    overlapping.mark_success()
    first.synchronize()
    first.mark_success()


def _poison_cached_context(service: DeviceService) raises:
    var scope = Scope(service.device_id(), 4096, service=service.copy())
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
        _ = (
            nonempty.lazy()
            .select(col("x"))
            .collect(accelerator=zero_budget, engine="accel")
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
        _ = (
            large.lazy()
            .select(col("x"))
            .collect(accelerator=runtime, engine="accel")
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


# requires: metal
def test_mojo_gpu_service_backed_branched_join_dag_reuses_runtime_copies() raises:
    var source = DataFrame(
        [
            Series("k", Column[Int64]([1, 2, 2, 3])),
            Series("x", Column[Float64]([10, 20, 21, 30])),
        ]
    )
    var branch = source.lazy().with_columns((col("x") + 1).alias("x"))
    var query = branch.join(branch, on="k").with_columns(
        (col("x") + col("x_right")).alias("sum")
    )
    var runtime = MetalRuntime()
    var copied = runtime.copy()
    var expected = query.collect(engine="cpu", streaming=False)

    var first = query.profile(accelerator=runtime, engine="accel")
    assert_true(first[0].equals(expected))
    assert_equal(
        first[1].item(0, "query_synchronizations").int64(),
        _region_synchronizations(first[1]),
    )
    _assert_region_synchronization_ms_sums(first[1])
    for field in List[String](
        [
            "service_context_generations",
            "service_idle_scratch_bytes",
            "service_idle_scratch_entries",
            "service_scratch_hits",
            "service_scratch_misses",
            "service_pipeline_entries",
            "service_pipeline_hits",
            "service_pipeline_misses",
            "service_pipeline_clears",
            "service_pipeline_capacity",
            "synchronization_ms",
            "query_synchronization_ms",
        ]
    ):
        assert_true(field in first[1].columns())
    assert_true(
        first[1].item(first[1].height() - 1, "synchronizations").int64() >= 1
    )
    for row in range(first[1].height()):
        assert_equal(
            first[1].item(row, "service_context_generations").int64(), 1
        )

    var second = query.profile(accelerator=copied, engine="accel")
    assert_true(second[0].equals(expected))
    assert_equal(runtime._device_service.cached_generations(), 1)
    assert_equal(
        second[1].item(0, "query_synchronizations").int64(),
        _region_synchronizations(second[1]),
    )
    _assert_region_synchronization_ms_sums(second[1])

    var empty = DataFrame([Series("k", Column[Int64](List[Int64]()))])
    var empty_runtime = MetalRuntime(device_id=999999)
    var empty_result = (
        empty.lazy()
        .join(empty.lazy(), on="k")
        .collect(accelerator=empty_runtime, engine="accel")
    )
    assert_equal(empty_result.height(), 0)
    assert_equal(empty_runtime._device_service.cached_generations(), 0)

    var no_match = source.lazy().filter(col("x") > lit(Float64(1000)))
    var zero_query = no_match.join(source.lazy(), on="k")
    var zero_baseline = execute_dag(
        zero_query, runtime._device, runtime._budget, True, capabilities()
    )
    assert_true(zero_baseline[1].item(0, "kernel_launches").int64() > 0)
    assert_true(zero_baseline[1].item(0, "synchronizations").int64() > 0)
    var baseline_waits = _region_synchronizations(zero_baseline[1])
    var baseline_terminal_waits = (
        zero_baseline[1]
        .item(zero_baseline[1].height() - 1, "synchronizations")
        .int64()
    )
    var zero = zero_query.profile(
        accelerator=runtime, engine="accel", optimize=False
    )
    assert_equal(zero[0].height(), 0)
    assert_equal(
        zero[1].item(0, "query_synchronizations").int64(), baseline_waits + 1
    )
    assert_equal(
        zero[1].item(zero[1].height() - 1, "synchronizations").int64(),
        baseline_terminal_waits + 1,
    )
    assert_true(zero[1].item(0, "query_synchronizations").int64() > 0)
    assert_equal(
        zero[1].item(0, "query_synchronizations").int64(),
        _region_synchronizations(zero[1]),
    )
    _assert_region_synchronization_ms_sums(zero[1])


# requires: metal
def test_mojo_gpu_service_backed_grouped_fields_dag_counts_terminal_drain() raises:
    var source = _source()
    var query = (
        source.lazy()
        .group_by("k", maintain_order=True)
        .agg(_expressions())
        .explode(["xs", "wide"])
    )
    var runtime = MetalRuntime()
    var baseline = execute_dag(
        query,
        runtime._device,
        runtime._budget,
        True,
        capabilities(),
        ordered_source=True,
    )
    var baseline_waits = _region_synchronizations(baseline[1])
    var measured = query.profile(
        accelerator=runtime, engine="accel", optimize=False
    )
    _literal(measured[0])
    assert_true(
        measured[0].equals(
            query.collect(engine="cpu", streaming=False, optimize=False)
        )
    )
    assert_equal(
        measured[1].item(0, "query_synchronizations").int64(),
        baseline_waits + 1,
    )
    assert_equal(
        measured[1].item(0, "query_synchronizations").int64(),
        _region_synchronizations(measured[1]),
    )
    _assert_region_synchronization_ms_sums(measured[1])
    assert_true(
        measured[1].item(measured[1].height() - 1, "synchronizations").int64()
        >= 1
    )
    for row in range(measured[1].height()):
        assert_equal(
            measured[1].item(row, "service_context_generations").int64(), 1
        )
    assert_equal(runtime._device_service.cached_generations(), 1)


# requires: metal
def test_mojo_gpu_service_backed_dag_overlap_and_post_submission_poison() raises:
    var left = DataFrame(
        [
            Series("k", Column[Int32]([1, 2])),
            Series("x", Column[Int32]([10, 20])),
        ]
    )
    var right = DataFrame(
        [
            Series("k", Column[Int32]([1, 2])),
            Series("tag", Column[Int32]([100, 200])),
        ]
    )
    var query = (
        left.lazy()
        .with_columns((col("x") + 1).alias("x"))
        .join(right.lazy(), on="k")
    )
    var runtime = MetalRuntime()
    var expected = query.collect(engine="cpu", streaming=False)
    assert_true(
        query.collect(accelerator=runtime, engine="accel").equals(expected)
    )
    assert_equal(runtime._device_service.cached_generations(), 1)

    var held = Scope(0, -1, service=runtime._device_service.copy())
    _ = held.context()
    assert_true(held.uses_cached_context())
    var private = execute_dag(
        query,
        runtime._device,
        runtime._budget,
        True,
        capabilities(),
        ordered_source=True,
        device_service=runtime._device_service.copy(),
    )
    assert_true(private[0].equals(expected))
    assert_equal(runtime._device_service.cached_generations(), 1)
    held.synchronize()
    held.mark_success()
    _ = held^
    assert_true(
        query.collect(accelerator=runtime.copy(), engine="accel").equals(
            expected
        )
    )
    assert_equal(runtime._device_service.cached_generations(), 1)

    var bad_left = DataFrame(
        [
            Series("k", Column[Int32]([1])),
            Series("x", Column[Int32]([128])),
        ]
    )
    var bad = (
        bad_left.lazy()
        .with_columns(col("x").cast(DataType.INT8).alias("narrow"))
        .join(right.lazy(), on="k")
    )
    with assert_raises(contains="cast"):
        _ = bad.collect(accelerator=runtime, engine="accel")
    var recovered = query.collect(accelerator=runtime, engine="accel")
    assert_true(recovered.equals(expected))
    assert_equal(runtime._device_service.cached_generations(), 2)


# requires: metal
def test_mojo_gpu_service_backed_dag_budget_failure_keeps_warm_context() raises:
    var small = DataFrame([Series("k", Column[Int32]([1, 2]))])
    var query = small.lazy().join(small.lazy(), on="k")
    var runtime = MetalRuntime()
    _ = query.collect(accelerator=runtime, engine="accel")
    assert_equal(runtime._device_service.cached_generations(), 1)

    var previous_budget = runtime._budget
    runtime._budget = 0
    with assert_raises(contains="memory budget"):
        _ = query.collect(accelerator=runtime, engine="accel")
    assert_equal(runtime._device_service.cached_generations(), 1)
    runtime._budget = previous_budget
    _ = query.collect(accelerator=runtime, engine="accel")
    assert_equal(runtime._device_service.cached_generations(), 1)
