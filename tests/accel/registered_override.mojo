"""Explicit runtime/device ownership overrides optional provider defaults."""
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, Series, col
from dataframe_accel.nvidia import NvidiaRuntime


def main() raises:
    # The caller intentionally sets an invalid DATAFRAME_ACCEL_DEVICE. An
    # explicit runtime must bypass discovery and keep its own chosen device.
    var runtime = NvidiaRuntime(device_id=0)
    var query = (
        DataFrame([Series("x", Column[Float64]([1, 2]))])
        .lazy()
        .select(col("x").sum())
    )
    for _ in range(3):
        assert_equal(
            query.collect(engine="accel", accelerator=runtime)
            .item(0, "x")
            .float64(),
            Float64(3),
        )
    assert_true("NVIDIA supported" in query.explain(accelerator=runtime))
    assert_equal(runtime.device_id(), 0)
    print("explicit provider override passed")
