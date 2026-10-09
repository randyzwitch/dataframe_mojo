"""Same public imports in CPU and GPU distributions; modes exercise discovery."""
from std.sys import argv
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, Series, col, scan_csv


def main() raises:
    var args = argv()
    var mode = args[1] if len(args) > 1 else "gpu"
    var plan = (
        DataFrame([Series("x", Column[Float32]([-2, 1, 3]))])
        .lazy()
        .filter(col("x") > 0)
        .select((col("x") * 1.25).sum().alias("total"))
    )
    var cpu = plan.collect(engine="cpu")
    assert_equal(cpu.item(0, "total").float32(), Float32(5))
    assert_true(plan.collect(engine="auto").equals(cpu))
    if mode == "cpu":
        assert_true("provider is not installed" in plan.explain(engine="accel"))
        with assert_raises(contains="provider is not installed"):
            _ = plan.collect(engine="accel")
    elif mode == "unavailable":
        assert_true("device is unavailable" in plan.explain(engine="accel"))
        with assert_raises(contains="device is unavailable"):
            _ = plan.collect(engine="accel")
    elif mode == "invalid-device":
        assert_true("DATAFRAME_ACCEL_DEVICE" in plan.explain(engine="accel"))
        with assert_raises(contains="DATAFRAME_ACCEL_DEVICE"):
            _ = plan.collect(engine="accel")
    elif mode == "unsupported":
        var unsupported = scan_csv(
            "/nonexistent/registered-provider.csv"
        ).select(col("x").sum())
        with assert_raises(contains="NVIDIA unsupported"):
            _ = unsupported.collect(engine="accel")
        assert_true("NVIDIA unsupported" in unsupported.explain(engine="accel"))
    else:
        for _ in range(3):
            assert_true(plan.collect(engine="accel").equals(cpu))
        assert_true(
            "registered provider: NVIDIA" in plan.explain(engine="accel")
        )
        assert_true(plan.fetch(1, engine="accel").equals(cpu))
        var report = plan.profile(engine="accel")
        assert_true(report[0].equals(cpu))
        assert_equal(report[1].item(0, "executor").string(), "nvidia")
        with assert_raises(contains="batch_size must be positive"):
            _ = plan.collect(engine="accel", batch_size=0)
    print("registered provider:", mode, "passed")
