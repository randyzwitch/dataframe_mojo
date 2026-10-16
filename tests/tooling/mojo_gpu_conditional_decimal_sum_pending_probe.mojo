"""Public integration probe; pending ranked checked operand feed foundation."""
from std.testing import assert_equal
from dataframe.metal import MetalRuntime
from tests.execution_modes.test_backend_metal import require_metal
from tests.execution_modes.test_conditional_decimal_aggregate_planning import (
    ratio_query,
    source,
    operand,
)
from dataframe import lit


def main() raises:
    require_metal()
    var runtime = MetalRuntime()
    var ratio = ratio_query()
    var cpu = ratio.collect(engine="cpu", optimize=False)
    assert_equal(cpu.column("ratio").float64()._get(0), Float64(80))
    var gpu = ratio.collect(accelerator=runtime, optimize=False)
    assert_equal(gpu.column("ratio").float64()._get(0), Float64(80))
    print("PASS public conditional checked Decimal ratio literal80")
    var masked = source().lazy().select((operand().sum() + lit(1.0)).alias("v"))
    var masked_gpu = masked.collect(accelerator=runtime, optimize=False)
    assert_equal(masked_gpu.column("v").float64()._get(0), Float64(4))
    print("PASS masked invalid Decimal SUM operand literal4")
