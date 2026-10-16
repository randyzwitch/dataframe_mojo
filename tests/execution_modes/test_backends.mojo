"""Public backend dispatch contracts, independent of installed device hardware."""
# contract-group: planner.routing
# contract-group: execution.backends
from std.memory import Pointer
from std.sys import CompilationTarget
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, Series, col, lit, scan_csv
from dataframe._accel_provider import installed
from dataframe.lazy import LazyFrame
from dataframe.metal import MetalRuntime
from dataframe.accelerator.gpu.target import compiled_for_metal
from tests.support.backend_reference import (
    BackendTrace,
    CountedBackend,
    DefaultPolicyBackend,
    backend_plan,
    assert_backend_rows,
)


def assert_no_provider_calls(trace: BackendTrace) raises:
    assert_equal(trace.selections, 0)
    assert_equal(trace.executes, 0)
    assert_equal(trace.profiled, 0)
    assert_equal(trace.automatic, 0)
    assert_equal(trace.descriptions, 0)


def test_cpu_engine_ignores_explicit_provider_policy_and_execution() raises:
    for n in [0, 1, 9, 65]:
        for optimize in [False, True]:
            for streaming in [False, True]:
                var trace = BackendTrace(
                    "invalid-selection", "provider must not run"
                )
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
        var trace = BackendTrace("cpu")
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
        assert_equal(trace.selections, 0)
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
        assert_equal(trace.automatic, 0)
        assert_equal(trace.selections, 0)


def test_automatic_cpu_accel_and_mixed_routes_forward_every_execution_setting() raises:
    for decision in ["cpu", "accel", "mixed"]:
        for optimize in [False, True]:
            for streaming in [False, True]:
                for batch in [1, 7, 64, 257]:
                    var trace = BackendTrace(decision)
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
                    assert_equal(trace.selections, 1)
                    assert_equal(trace.automatic, Int(decision != "cpu"))
                    assert_equal(trace.optimize, optimize)
                    assert_equal(trace.streaming, streaming)
                    assert_equal(trace.batch, batch)
                    assert_equal(trace.profiling, False)
                    var profiled = plan.profile(
                        accelerator=provider,
                        engine="auto",
                        optimize=optimize,
                        streaming=streaming,
                        batch_size=batch,
                    )
                    assert_backend_rows(profiled[0], 65)
                    assert_equal(trace.selections, 2)
                    assert_equal(trace.automatic, 2 * Int(decision != "cpu"))
                    assert_equal(trace.profiling, decision != "cpu")
                    assert_equal(trace.optimize, optimize)
                    assert_equal(trace.streaming, streaming)
                    assert_equal(trace.batch, batch)
                    assert_true(profiled[1].height() > 0)
                    for row in range(profiled[1].height()):
                        assert_equal(
                            profiled[1].item(row, "selection_reason").string(),
                            "counted placement " + decision,
                        )
                        if decision != "cpu":
                            assert_equal(
                                profiled[1].item(row, "executor").string(),
                                "counted-provider",
                            )
                    assert_equal(trace.executes, 0)
                    assert_equal(trace.profiled, 0)


def test_automatic_explain_is_selection_only_and_fetch_forwards_default_settings() raises:
    for decision in ["cpu", "accel", "mixed"]:
        var trace = BackendTrace(decision)
        var provider = CountedBackend(Int(Pointer(to=trace)))
        var plan = backend_plan(65)
        var text = plan.explain(
            accelerator=provider,
            engine="auto",
            optimize=False,
            streaming=False,
            batch_size=257,
        )
        assert_true(
            "ENGINE " + decision + ": counted placement " + decision in text
        )
        assert_equal(trace.selections, 1)
        assert_equal(trace.optimize, False)
        assert_equal(trace.streaming, False)
        assert_equal(trace.batch, 257)
        assert_equal(trace.automatic, 0)
        assert_equal(trace.executes, 0)
        assert_equal(trace.profiled, 0)
        assert_equal(trace.descriptions, 0)
        assert_backend_rows(
            plan.fetch(0, accelerator=provider, engine="auto"), 65, limit=0
        )
        assert_backend_rows(
            plan.fetch(5, accelerator=provider, engine="auto"), 65, limit=5
        )
        assert_equal(trace.selections, 3)
        assert_equal(trace.automatic, 2 * Int(decision != "cpu"))
        assert_equal(trace.optimize, True)
        assert_equal(trace.streaming, True)
        assert_equal(trace.batch, 65536)


def test_third_party_default_policy_selects_cpu_and_default_profile_executes_once() raises:
    var trace = BackendTrace()
    var provider = DefaultPolicyBackend(Int(Pointer(to=trace)))
    var plan = backend_plan(65)
    assert_backend_rows(plan.collect(accelerator=provider, engine="auto"), 65)
    var profiled = plan.profile(accelerator=provider, engine="auto")
    assert_backend_rows(profiled[0], 65)
    for row in range(profiled[1].height()):
        assert_equal(
            profiled[1].item(row, "selection_reason").string(),
            "provider has no automatic placement policy",
        )
    assert_true(
        "provider has no automatic placement policy"
        in plan.explain(accelerator=provider, engine="auto")
    )
    assert_no_provider_calls(trace)
    var forced = plan.profile(accelerator=provider)
    assert_backend_rows(forced[0], 65)
    assert_equal(trace.executes, 1)
    # Trait execute_auto defaults to execute/execute_profiled, without recursion.
    _ = provider.execute_auto(plan, False, False, 7, False)
    _ = provider.execute_auto(plan, False, False, 7, True)
    assert_equal(trace.executes, 3)
    with assert_raises():
        _ = provider.estimate_auto(plan)


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
    for decision in ["", "CPU", "gpu"]:
        var trace = BackendTrace(decision)
        var provider = CountedBackend(Int(Pointer(to=trace)))
        for operation in range(3):
            var message = String()
            try:
                if operation == 0:
                    _ = plan.collect(accelerator=provider, engine="auto")
                elif operation == 1:
                    _ = plan.profile(accelerator=provider, engine="auto")
                else:
                    _ = plan.explain(accelerator=provider, engine="auto")
            except error:
                message = String(error)
            assert_equal(
                message, "Invalid automatic provider decision: " + decision
            )
        assert_equal(trace.selections, 3)
        assert_equal(
            trace.automatic
            + trace.executes
            + trace.profiled
            + trace.descriptions,
            0,
        )


def test_provider_execution_errors_propagate_once_without_cpu_retry() raises:
    # A CPU retry would fail on the nonexistent scan rather than the provider's
    # distinctive error, so this observes both dispatch and no-fallback behavior.
    var plan = scan_csv("/this/path/does/not/exist/backend-no-fallback.csv")
    for engine in ["accel", "auto"]:
        for decision in ["accel", "mixed"]:
            for operation in range(3):
                var trace = BackendTrace(
                    decision, "independent provider execution failure"
                )
                var provider = CountedBackend(Int(Pointer(to=trace)))
                var message = String()
                try:
                    if operation == 0:
                        _ = plan.collect(accelerator=provider, engine=engine)
                    elif operation == 1:
                        _ = plan.profile(accelerator=provider, engine=engine)
                    else:
                        _ = plan.fetch(2, accelerator=provider, engine=engine)
                except error:
                    message = String(error)
                assert_equal(message, "independent provider execution failure")
                assert_equal(
                    trace.executes + trace.profiled + trace.automatic, 1
                )
                assert_equal(trace.selections, Int(engine == "auto"))


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
        for row in range(profiled[1].height()):
            assert_true(
                "provider is not installed"
                in profiled[1].item(row, "selection_reason").string()
            )
        assert_true("provider is not installed" in plan.explain(engine="auto"))
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
