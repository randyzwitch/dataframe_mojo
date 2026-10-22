"""Mojo analytical facts describe metadata without source reads or execution."""
# contract-group: execution.backends
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import DataFrame, Series, Column, Expr, col, lit, scan_csv
from dataframe_metal.metal import MetalRuntime
from dataframe.lazy import LazyFrame
from dataframe_metal.accelerator.cost import PlanEstimate, cost_feature_names
from dataframe_metal.accelerator.rows import lower_rows
from dataframe._build_identity import CPU_BUILD_ID
from tests.execution_modes.test_backend_metal import (
    require_metal,
    assert_native_report,
)
from tests.support.metal_domain_checks import assert_domain
from tests.support.lifecycle_jobs import WorkerSetting


def estimate_feature(
    facts: PlanEstimate, name: String, device: Bool
) raises -> Float64:
    var names = cost_feature_names()
    for at in range(len(names)):
        if names[at] == name:
            return facts.accelerator[at] if device else facts.cpu[at]
    raise Error("Missing documented analytical feature " + name)


def cost_query(
    source: DataFrame, constant_offset: Int = 0
) raises -> Tuple[LazyFrame, DataFrame]:
    # 21 independent Float32 outputs create a dedicated shader shape for this
    # cache witness. Runtime literal values, row counts and names vary below.
    var expressions = List[Expr]()
    var expected = List[Series]()
    for index in range(21):
        var name = "cost_" + String(index + constant_offset)
        var value = Float32(index + constant_offset)
        expressions.append((col("x") + lit(value)).alias(name))
        var values = List[Float32]()
        var bits = List[Bool]()
        for row in range(source.height()):
            var cell = source.item(row, "x")
            values.append(
                Float32(0) if cell.is_null() else Float32(
                    cell.float32() + value
                )
            )
            bits.append(not cell.is_null())
        expected.append(Series(name, Column[Float32](values^, bits^)))
    return (source.lazy().select_exprs(expressions), DataFrame(expected^))


# requires: metal
def test_native_metal_analytical_estimates_are_value_blind_and_observe_cold_to_warm_shader_metadata() raises:
    var runtime = require_metal()
    var setting = WorkerSetting(1)
    var a = DataFrame(
        [Series("x", Column[Float32]([1, 2, 3], [True, True, False]))]
    )
    var b = DataFrame(
        [Series("x", Column[Float32]([99, -7, 0], [False, True, True]))]
    )
    var first = cost_query(a)
    var alternate = cost_query(b)
    var before = first[0].explain(engine="cpu")
    var cold = runtime.estimate_auto(first[0])
    var same = runtime.estimate_auto(alternate[0])
    var repeated = runtime.estimate_auto(first[0])
    assert_equal(cold.cpu, same.cpu)
    # Both sources have a validity bitmap: payload and bitmap contents must
    # not influence the estimate, while their physical buffer sizes may.
    assert_equal(cold.accelerator, same.accelerator)
    assert_equal(cold.accelerator, repeated.accelerator)
    var dense = cost_query(DataFrame([Series("x", Column[Float32]([1, 2, 3]))]))
    var dense_facts = runtime.estimate_auto(dense[0])
    assert_equal(cold.cpu, dense_facts.cpu)
    # Mojo uploads a byte validity matrix for every cell, independent of
    # whether the source owns a packed bitmap or has an all-valid shortcut.
    assert_equal(cold.accelerator, dense_facts.accelerator)
    assert_true(";mojo-metal-v1;ops=" in cold.family)
    assert_equal(cold.placement, "accel")
    assert_equal(estimate_feature(cold, "fixed", False), Float64(1))
    assert_equal(estimate_feature(cold, "rows", False), Float64(3))
    assert_equal(estimate_feature(cold, "source_bytes", False), Float64(12))
    assert_equal(estimate_feature(cold, "filter_node_rows", False), Float64(0))
    assert_equal(estimate_feature(cold, "reduction_rows", False), Float64(0))
    assert_equal(estimate_feature(cold, "cast_node_rows", False), Float64(0))
    assert_true(estimate_feature(cold, "projection_node_rows", False) > 0)
    assert_equal(estimate_feature(cold, "synchronizations", True), Float64(3))
    assert_equal(estimate_feature(cold, "cold", True), Float64(1))
    assert_equal(
        estimate_feature(cold, "allocation_bytes", True),
        Float64(cold.peak_bytes),
    )
    assert_true(estimate_feature(cold, "upload_bytes", True) >= 12)
    assert_true(estimate_feature(cold, "launches", True) > 0)
    assert_equal(first[0].explain(engine="cpu"), before)
    assert_domain(
        a,
        DataFrame(
            [Series("x", Column[Float32]([1, 2, 3], [True, True, False]))]
        ),
    )
    var measured = first[0].profile(accelerator=runtime)
    assert_domain(measured[0], first[1])
    assert_native_report(measured[1])
    var warm = runtime.estimate_auto(first[0])
    assert_equal(warm.cpu, cold.cpu)
    # Standalone calls each own a context/buffer scope. They never inherit
    # the deleted native process-wide shader-cache cold/warm flag.
    assert_equal(warm.accelerator, cold.accelerator)
    var resized = cost_query(
        DataFrame([Series("x", Column[Float32]([4, 5, 6, 7, 8]))]), 101
    )
    var resized_facts = runtime.estimate_auto(resized[0])
    assert_equal(estimate_feature(resized_facts, "rows", False), Float64(5))
    assert_equal(
        estimate_feature(resized_facts, "source_bytes", False), Float64(20)
    )
    assert_equal(estimate_feature(resized_facts, "cold", True), Float64(1))
    var reused = resized[0].profile(accelerator=runtime)
    assert_domain(reused[0], resized[1])
    assert_native_report(reused[1])
    for optimize in [False, True]:
        for streaming in [False, True]:
            for batch in [7, 65]:
                var identity = runtime.calibration_key(
                    optimize, streaming, batch
                )
                assert_equal(identity.provider, "metal")
                assert_true(identity.device.startswith("mojo-metal;ordinal=0;"))
                assert_true("unified=" in identity.device)
                assert_equal(identity.cpu_build_id, CPU_BUILD_ID)
                if CPU_BUILD_ID == "development-source-no-calibration":
                    assert_equal(identity.runtime_build_id, CPU_BUILD_ID)
                else:
                    assert_true(
                        identity.runtime_build_id.startswith(
                            "dataframe-mojo-metal-v1;source-compiler-sdk="
                        )
                    )
                assert_true(
                    "model=" in identity.host and "os=" in identity.host
                )
                assert_equal(identity.threads, 1)
                assert_equal(identity.optimize, optimize)
                assert_equal(identity.streaming, streaming)
                assert_equal(identity.batch_size, batch)
    _ = setting^


def cost_preflight() raises:
    var invalid = MetalRuntime(device_id=999999)
    var wide = DataFrame([Series("x", Column[Float64]([1, 2]))])
    with assert_raises(contains="no analytical calibration"):
        _ = invalid.estimate_auto(wide.lazy().select(col("x").implode()))
    with assert_raises(contains="in-memory"):
        _ = invalid.estimate_auto(
            scan_csv(
                "/nonexistent/analytical-metadata-must-not-open.csv"
            ).select(col("x"))
        )


# requires: metal
def test_native_metal_analytical_unsupported_requests_reject_before_source_or_device_access() raises:
    _ = require_metal()
    cost_preflight()


def verify_metal_cost_cpu_models() raises:
    for values in [
        [Float32(1), Float32(2), Float32(3)],
        [Float32(99), Float32(-7), Float32(0)],
    ]:
        var prepared = cost_query(
            DataFrame(
                [
                    Series(
                        "x", Column[Float32]([values[0], values[1], values[2]])
                    )
                ]
            )
        )
        _ = lower_rows(prepared[0])
        for optimize in [False, True]:
            for streaming in [False, True]:
                assert_domain(
                    prepared[0].collect(
                        engine="cpu",
                        optimize=optimize,
                        streaming=streaming,
                        batch_size=7,
                    ),
                    prepared[1],
                )
    cost_preflight()
