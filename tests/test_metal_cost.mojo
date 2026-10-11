"""Metal supplies metadata facts to the shared analytical cost policy."""
from std.testing import TestSuite, assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, Series, col, lit, scan_csv
from dataframe.metal import MetalRuntime, metal_installed


def test_plan_estimates_ignore_values_and_observe_pipeline_cache() raises:
    if not metal_installed():
        return
    var runtime = MetalRuntime()
    var a = DataFrame([Series("x", Column[Float32]([1, 2, 3]))])
    var b = DataFrame([Series("x", Column[Float32]([99, -7, 0]))])
    var before = a.copy()
    var first = a.lazy().select((col("x") * lit(Float32(3))).alias("y"))
    var second = b.lazy().select((col("x") * lit(Float32(3))).alias("y"))
    var x = runtime.estimate_auto(first)
    var y = runtime.estimate_auto(second)
    assert_equal(x.cpu, y.cpu)
    assert_equal(x.accelerator, y.accelerator)
    assert_equal(x.family, "float32")
    assert_equal(x.cpu[1], Float64(3))
    assert_equal(x.cpu[2], Float64(12))
    assert_equal(x.accelerator[9], Float64(1))
    assert_equal(x.accelerator[13], Float64(1))
    assert_true(x.peak_bytes > 0 and x.accelerator[8] > 0)
    assert_true(a.equals(before))
    assert_true(
        first.collect(accelerator=runtime).equals(first.collect(engine="cpu"))
    )
    var warm = runtime.estimate_auto(first)
    assert_equal(warm.accelerator[13], Float64(0))
    assert_equal(warm.cpu, x.cpu)
    var key = runtime.calibration_key(True, True, 65536)
    assert_equal(key.provider, "metal")
    assert_true(key.runtime_build_id.byte_length() == 64)
    assert_true("model=" in key.host and "os=" in key.host)


def test_unsupported_cost_plans_reject_without_reading_sources() raises:
    var runtime = MetalRuntime()
    var wide = DataFrame([Series("x", Column[Float64]([1, 2]))])
    with assert_raises(contains="Float64 requires CPU"):
        _ = runtime.estimate_auto(wide.lazy().select(col("x") + 1))
    var missing = scan_csv("/nonexistent/metal-cost-must-not-read.csv").select(
        col("x")
    )
    with assert_raises(contains="unsupported"):
        _ = runtime.estimate_auto(missing)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
