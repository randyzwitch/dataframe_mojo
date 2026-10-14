"""Focused public source Struct adapter proof."""
from tests.execution_modes.test_mojo_gpu_source_structs import (
    test_mojo_gpu_source_struct_fields_nullable_layouts_literal_values_and_metadata,
    test_mojo_gpu_source_struct_errors_masks_profile_budget_and_recovery,
    test_mojo_gpu_source_struct_refusals_and_explanation_remain_metadata_only,
)


def main() raises:
    test_mojo_gpu_source_struct_refusals_and_explanation_remain_metadata_only()
    print("PASS source Struct refusals/metadata-only explanation")
    test_mojo_gpu_source_struct_fields_nullable_layouts_literal_values_and_metadata()
    print("PASS source Struct literal fields/layouts/nulls/metadata")
    test_mojo_gpu_source_struct_errors_masks_profile_budget_and_recovery()
    print("PASS source Struct errors/masks/budgets/recovery")
