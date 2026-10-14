"""Run the bounded public temporal GPU contract subset on Apple Silicon."""

from tests.execution_modes.test_mojo_gpu_temporal_arithmetic import (
    test_mojo_gpu_temporal_arithmetic_preflight_rejects_checked_producers_and_resident_sources,
    test_mojo_gpu_temporal_binary_arithmetic_extremes_layouts_descriptors_and_budget,
    test_mojo_gpu_temporal_physical_casts_conditional_masks_and_overflow_contract,
)


def main() raises:
    test_mojo_gpu_temporal_arithmetic_preflight_rejects_checked_producers_and_resident_sources()
    test_mojo_gpu_temporal_binary_arithmetic_extremes_layouts_descriptors_and_budget()
    test_mojo_gpu_temporal_physical_casts_conditional_masks_and_overflow_contract()
    print("PASS all 3 public temporal GPU contracts")
