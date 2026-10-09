"""GPU memory gates and measured reports, including ordinary collect parity."""
from std.testing import TestSuite, assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, LazyFrame, Series, col
from dataframe_accel.nvidia import NvidiaRuntime


def query() raises -> LazyFrame:
    var values = List[Float32]()
    var valid = List[Bool]()
    for i in range(280):
        values.append(Float32(i % 17 - 8))
        valid.append(i % 3 != 0)
    return (
        DataFrame([Series("x", Column[Float32](values^, valid).slice(19, 257))])
        .lazy()
        .filter(col("x") > 0)
        .select_exprs(
            [
                (col("x") * 1.25).sum().alias("sum"),
                col("x").count().alias("count"),
            ]
        )
    )


def test_preflight_budget_and_cpu_policy() raises:
    var plan = query()
    var denied = NvidiaRuntime(memory_limit_bytes=1108)
    assert_true("memory preflight rejected" in plan.explain(accelerator=denied))
    with assert_raises(contains="exceeds budget 1108"):
        _ = plan.collect(accelerator=denied)
    with assert_raises(contains="exceeds budget 1108"):
        _ = plan.profile(accelerator=denied)
    assert_true(
        plan.collect(engine="auto", accelerator=denied).equals(plan.collect())
    )
    var exact = NvidiaRuntime(memory_limit_bytes=1109)
    assert_true(plan.collect(accelerator=exact).equals(plan.collect()))
    with assert_raises(contains="memory_limit_bytes must"):
        _ = NvidiaRuntime(memory_limit_bytes=-2)


def test_report_accounts_for_real_allocations_and_timing() raises:
    var plan = query()
    var runtime = NvidiaRuntime(memory_limit_bytes=1109)
    var expected = plan.collect()
    var ordinary = runtime.execute(plan)
    assert_true(ordinary[0].equals(expected))
    assert_true(ordinary[1].item(0, "kernel_ms").is_null())
    assert_equal(ordinary[1].item(0, "synchronizations").int64(), Int64(5))
    var profiled = plan.profile(accelerator=runtime)
    assert_true(profiled[0].equals(expected))
    var report = profiled[1].copy()
    assert_equal(report.item(0, "upload_bytes").int64(), Int64(1061))
    assert_equal(report.item(0, "download_bytes").int64(), Int64(32))
    assert_equal(report.item(0, "workspace_bytes").int64(), Int64(32))
    assert_equal(report.item(0, "device_output_bytes").int64(), Int64(16))
    assert_equal(
        report.item(0, "peak_requested_device_bytes").int64(), Int64(1109)
    )
    assert_equal(report.item(0, "memory_budget_bytes").int64(), Int64(1109))
    assert_equal(report.item(0, "kernel_launches").int64(), Int64(4))
    assert_equal(report.item(0, "synchronizations").int64(), Int64(7))
    assert_equal(report.item(0, "device_id").int64(), Int64(0))
    assert_true(report.item(0, "kernel_ms").float64() > 0)
    assert_true(report.item(0, "wall_ms").float64() > 0)
    assert_true(report.item(0, "free_device_bytes").int64() >= Int64(1109))
    assert_true("scalar download" in report.item(0, "boundaries").string())
    assert_true("workspace=32" in plan.explain(accelerator=runtime))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
