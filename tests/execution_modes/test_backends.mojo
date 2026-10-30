"""Public backend dispatch contracts, independent of installed device hardware."""
# contract-group: planner.routing
# contract-group: execution.backends
from std.memory import Pointer
from std.sys import CompilationTarget
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, Series, col, lit, scan_csv
from dataframe._accel_provider import (
    installed,
    devices,
    transfer_model,
    pipeline_supports,
    pipeline_estimate,
    run_pipeline,
)
from dataframe.devices import (
    ByteCost,
    PipelineFeatures,
    TransferModel,
)
from dataframe_metal import provider as metal_provider
from dataframe.lazy import LazyFrame
from dataframe_metal.metal import MetalRuntime
from dataframe_metal.accelerator.gpu.target import compiled_for_metal
from tests.support.backend_reference import (
    BackendTrace,
    CountedBackend,
    DefaultPolicyBackend,
    backend_plan,
    assert_backend_rows,
)


def assert_no_provider_calls(trace: BackendTrace) raises:
    assert_equal(trace.executes, 0)
    assert_equal(trace.profiled, 0)
    assert_equal(trace.descriptions, 0)


def test_cpu_engine_ignores_explicit_provider_policy_and_execution() raises:
    for n in [0, 1, 9, 65]:
        for optimize in [False, True]:
            for streaming in [False, True]:
                var trace = BackendTrace("provider must not run")
                var provider = CountedBackend(Int(Pointer(to=trace)))
                var plan = backend_plan(n)
                assert_backend_rows(
                    plan.collect(
                        accelerator=provider,
                        engine="cpu",
                        optimize=optimize,
                        streaming=streaming,
                        batch_size=7,
                    ),
                    n,
                )
                var profiled = plan.profile(
                    accelerator=provider,
                    engine="cpu",
                    optimize=optimize,
                    streaming=streaming,
                    batch_size=7,
                )
                assert_backend_rows(profiled[0], n)
                assert_true(profiled[1].height() > 0)
                assert_equal(
                    plan.explain(accelerator=provider, engine="cpu"),
                    plan.explain(engine="cpu"),
                )
                assert_backend_rows(
                    plan.fetch(2, accelerator=provider, engine="cpu"),
                    n,
                    limit=2,
                )
                assert_no_provider_calls(trace)


def test_forced_and_omitted_engine_dispatch_execute_profile_describe_and_fetch_once() raises:
    for explicit in [False, True]:
        var trace = BackendTrace()
        var provider = CountedBackend(Int(Pointer(to=trace)))
        var plan = backend_plan(65)
        if explicit:
            assert_backend_rows(
                plan.collect(
                    accelerator=provider,
                    engine="accel",
                    optimize=False,
                    streaming=False,
                    batch_size=1,
                ),
                65,
            )
        else:
            assert_backend_rows(plan.collect(accelerator=provider), 65)
        assert_equal(trace.executes, 1)
        var profiled = plan.profile(accelerator=provider)
        assert_backend_rows(profiled[0], 65)
        assert_equal(
            profiled[1].item(0, "executor").string(), "counted-provider"
        )
        assert_equal(trace.profiled, 1)
        assert_equal(trace.executes, 1)
        assert_equal(
            plan.explain(accelerator=provider),
            "ENGINE accel: counted provider description\n",
        )
        assert_equal(trace.descriptions, 1)
        assert_backend_rows(plan.fetch(3, accelerator=provider), 65, limit=3)
        assert_equal(trace.executes, 2)


def test_automatic_engine_never_runs_a_whole_plan_on_the_provider() raises:
    # Neither provider reports devices, so placement leaves every pipeline
    # on the CPU and no whole-plan function is called.
    for optimize in [False, True]:
        for streaming in [False, True]:
            for batch in [1, 7, 64, 257]:
                var trace = BackendTrace("provider must not run")
                var provider = CountedBackend(Int(Pointer(to=trace)))
                var plan = backend_plan(65)
                assert_backend_rows(
                    plan.collect(
                        accelerator=provider,
                        engine="auto",
                        optimize=optimize,
                        streaming=streaming,
                        batch_size=batch,
                    ),
                    65,
                )
                var profiled = plan.profile(
                    accelerator=provider,
                    engine="auto",
                    optimize=optimize,
                    streaming=streaming,
                    batch_size=batch,
                )
                assert_backend_rows(profiled[0], 65)
                assert_true(profiled[1].height() > 0)
                assert_true("selection_reason" not in profiled[1].columns())
                assert_no_provider_calls(trace)
    var trace = BackendTrace()
    var provider = DefaultPolicyBackend(Int(Pointer(to=trace)))
    var plan = backend_plan(65)
    assert_backend_rows(plan.collect(accelerator=provider, engine="auto"), 65)
    assert_backend_rows(
        plan.profile(accelerator=provider, engine="auto")[0], 65
    )
    assert_no_provider_calls(trace)
    var forced = plan.profile(accelerator=provider)
    assert_backend_rows(forced[0], 65)
    assert_equal(trace.executes, 1)


def test_automatic_explain_describes_placement_and_fetch_stays_on_the_cpu() raises:
    var trace = BackendTrace("provider must not run")
    var provider = CountedBackend(Int(Pointer(to=trace)))
    var plan = backend_plan(65)
    var text = plan.explain(
        accelerator=provider,
        engine="auto",
        optimize=False,
        streaming=False,
        batch_size=257,
    )
    assert_true(text.startswith("ENGINE auto: placed by measured cost"), text)
    assert_true(
        text.endswith(
            plan.explain(
                engine="cpu", optimize=False, streaming=False, batch_size=257
            )
        ),
        text,
    )
    assert_backend_rows(
        plan.fetch(0, accelerator=provider, engine="auto"), 65, limit=0
    )
    assert_backend_rows(
        plan.fetch(5, accelerator=provider, engine="auto"), 65, limit=5
    )
    assert_no_provider_calls(trace)


def test_invalid_engines_batches_and_provider_decisions_reject_before_execution() raises:
    var plan = backend_plan(9)
    for bad_engine in ["", "CPU", "cuda", "mixed"]:
        var trace = BackendTrace()
        var provider = CountedBackend(Int(Pointer(to=trace)))
        with assert_raises():
            _ = plan.collect(accelerator=provider, engine=bad_engine)
        with assert_raises():
            _ = plan.profile(accelerator=provider, engine=bad_engine)
        with assert_raises():
            _ = plan.explain(accelerator=provider, engine=bad_engine)
        with assert_raises():
            _ = plan.fetch(1, accelerator=provider, engine=bad_engine)
        assert_no_provider_calls(trace)
        with assert_raises():
            _ = plan.collect(engine=bad_engine)
        with assert_raises():
            _ = plan.profile(engine=bad_engine)
        with assert_raises():
            _ = plan.explain(engine=bad_engine)
    for batch in [0, -1]:
        for engine in ["cpu", "auto", "accel"]:
            var trace = BackendTrace()
            var provider = CountedBackend(Int(Pointer(to=trace)))
            with assert_raises():
                _ = plan.collect(
                    accelerator=provider, engine=engine, batch_size=batch
                )
            with assert_raises():
                _ = plan.profile(
                    accelerator=provider, engine=engine, batch_size=batch
                )
            with assert_raises():
                _ = plan.explain(
                    accelerator=provider, engine=engine, batch_size=batch
                )
            assert_no_provider_calls(trace)


def test_provider_execution_errors_propagate_once_without_cpu_retry() raises:
    # A CPU retry would fail on the nonexistent scan rather than the provider's
    # distinctive error, so this observes both dispatch and no-fallback behavior.
    var plan = scan_csv("/this/path/does/not/exist/backend-no-fallback.csv")
    for operation in range(3):
        var trace = BackendTrace("independent provider execution failure")
        var provider = CountedBackend(Int(Pointer(to=trace)))
        var message = String()
        try:
            if operation == 0:
                _ = plan.collect(accelerator=provider, engine="accel")
            elif operation == 1:
                _ = plan.profile(accelerator=provider, engine="accel")
            else:
                _ = plan.fetch(2, accelerator=provider, engine="accel")
        except error:
            message = String(error)
        assert_equal(message, "independent provider execution failure")
        assert_equal(trace.executes + trace.profiled, 1)


def test_registered_slot_cpu_fallback_and_absence_are_explicit() raises:
    # Presence is tested as a capability fact rather than attempting native work
    # on every host. Native-present execution has separately gated witnesses.
    comptime if not CompilationTarget.is_macos():
        assert_equal(installed(), False)
    var plan = backend_plan(65)
    assert_backend_rows(plan.collect(engine="cpu"), 65)
    if not installed():
        assert_backend_rows(plan.collect(engine="auto"), 65)
        var profiled = plan.profile(engine="auto")
        assert_backend_rows(profiled[0], 65)
        assert_true("selection_reason" not in profiled[1].columns())
        assert_true(
            plan.explain(engine="auto").startswith(
                "ENGINE auto: placed by measured cost"
            )
        )
        for operation in range(3):
            var message = String()
            try:
                if operation == 0:
                    _ = plan.collect(engine="accel")
                elif operation == 1:
                    _ = plan.profile(engine="accel")
                else:
                    _ = plan.fetch(1, engine="accel")
            except error:
                message = String(error)
            assert_true("Accelerator provider is not installed" in message)
        assert_true(
            "Accelerator provider is not installed"
            in plan.explain(engine="accel")
        )


def test_native_provider_configuration_and_unsupported_types_reject_before_driver_setup() raises:
    for device in [-1, -2]:
        with assert_raises():
            _ = MetalRuntime(device)
    with assert_raises():
        _ = MetalRuntime(memory_limit_bytes=-2)
    _ = MetalRuntime(memory_limit_bytes=0)
    _ = MetalRuntime(memory_limit_bytes=-1)
    var runtime = MetalRuntime()
    var frame = DataFrame([Series("x", Column[Float64]([1.25, 2.5]))])
    var unsupported = frame.lazy().select(
        col("x").implode().list().len().alias("x")
    )
    comptime if compiled_for_metal():
        # Metal supports no node of this plan, so the CPU executor runs all
        # of it.
        assert_true(
            unsupported.collect(accelerator=runtime).equals(
                unsupported.collect(engine="cpu")
            )
        )
    else:
        var message = String()
        try:
            _ = unsupported.collect(accelerator=runtime)
        except error:
            message = String(error)
        assert_true("unsupported" in message)
    assert_true("unsupported" in unsupported.explain(accelerator=runtime))


def test_metal_cpu_only_target_keeps_cpu_dispatch_and_rejects_gpu_submission() raises:
    # This must compile with an unknown accelerator as well as on GPU hosts.
    # A runtime presence branch alone still specializes unreachable kernels.
    comptime if not compiled_for_metal():
        assert_equal(installed(), False)
        assert_equal(MetalRuntime.device_count(), 0)
        var plan = backend_plan(65)
        assert_backend_rows(plan.collect(engine="cpu"), 65)
        assert_backend_rows(plan.collect(engine="auto"), 65)
        var runtime = MetalRuntime()
        with assert_raises(contains="Apple Metal compilation target"):
            _ = plan.collect(engine="accel", accelerator=runtime)
        var frame = DataFrame([Series("k", Column[Int64]([1, 2, 3]))])
        var joined = frame.lazy().join(frame.lazy(), on="k")
        with assert_raises(contains="Apple Metal compilation target"):
            _ = joined.collect(engine="accel", accelerator=runtime)


def test_device_functions_report_nothing_without_a_device() raises:
    # Linux contract builds use the CPU-only provider; Metal compiled for a
    # CPU target has no device either.
    var plan = backend_plan(8)
    var features = PipelineFeatures.rows(8)
    if not installed():
        assert_equal(len(devices()), 0)
        assert_true(not transfer_model(0).measured())
        assert_true(not pipeline_supports(plan, 0))
        assert_equal(pipeline_estimate(plan, 0, features), Float64(-1))
        with assert_raises(contains="no device runs pipelines"):
            _ = run_pipeline(plan, 0)
    if not compiled_for_metal():
        assert_equal(len(metal_provider.devices()), 0)
        assert_true(not metal_provider.transfer_model(0).measured())
        assert_true(not metal_provider.pipeline_supports(plan, 0))
        assert_equal(
            metal_provider.pipeline_estimate(plan, 0, features), Float64(-1)
        )
        with assert_raises(contains="no device runs pipelines"):
            _ = metal_provider.run_pipeline(plan, 0)


def test_device_cost_types_mean_what_they_say() raises:
    var link = ByteCost(1000, 0.5)
    assert_true(link.measured())
    assert_equal(link.cost(0), Float64(1000))
    assert_equal(link.cost(4096), Float64(3048))
    assert_true(not ByteCost(-1, 0.5).measured())
    var shared = TransferModel.shared_memory()
    assert_true(shared.measured())
    assert_equal(shared.to_device.cost(1 << 30), Float64(0))
    assert_equal(shared.to_host.cost(1 << 30), Float64(0))
    assert_true(shared.startup_ns < 0)
    assert_true(not TransferModel.unmeasured().measured())
    var features = PipelineFeatures.rows(100)
    assert_equal(features.input_rows, 100)
    assert_equal(features.groups, -1)
    assert_equal(features.build_rows, -1)


def test_explicit_runtimes_default_to_no_devices() raises:
    var trace = BackendTrace()
    var backend = DefaultPolicyBackend(Int(Pointer(to=trace)))
    var plan = backend_plan(8)
    assert_equal(len(backend.devices()), 0)
    assert_true(not backend.transfer_model(0).measured())
    assert_true(not backend.pipeline_supports(plan, 0))
    assert_equal(
        backend.pipeline_estimate(plan, 0, PipelineFeatures.rows(8)),
        Float64(-1),
    )
    with assert_raises(contains="does not run pipelines"):
        _ = backend.run_pipeline(plan, 0)
    assert_no_provider_calls(trace)
    _ = trace^
