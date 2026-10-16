"""Focused original-proof and public strict-cast composition validation."""
from tests.execution_modes.test_strict_cast_producer_metadata import (
    test_strict_cast_original_prefix_owner_and_bridge,
)
from tests.execution_modes.test_mojo_gpu_strict_cast_producers import (
    test_mojo_gpu_strict_cast_composition_filter_group_join_and_layouts,
    test_mojo_gpu_strict_cast_complete_phase_faults_and_conditionals,
    test_mojo_gpu_strict_cast_projection_fault_order_before_consumer,
    test_mojo_gpu_strict_cast_nullable_original_values_and_dtype,
)


def main() raises:
    test_strict_cast_original_prefix_owner_and_bridge()
    test_mojo_gpu_strict_cast_composition_filter_group_join_and_layouts()
    test_mojo_gpu_strict_cast_complete_phase_faults_and_conditionals()
    test_mojo_gpu_strict_cast_projection_fault_order_before_consumer()
    test_mojo_gpu_strict_cast_nullable_original_values_and_dtype()
