"""Focused terminal primitive List packaging proof."""
from tests.execution_modes.test_mojo_gpu_implode import (
    test_mojo_gpu_implode_primitive_literals_nulls_layouts_metadata_and_empty,
    test_mojo_gpu_implode_prior_rows_trailing_slices_and_profiles_budget,
    test_mojo_gpu_implode_gpu_errors_masks_projection_order_head_and_observer,
    test_mojo_gpu_implode_explicit_boundaries_and_metadata_only_description,
)


def main() raises:
    test_mojo_gpu_implode_primitive_literals_nulls_layouts_metadata_and_empty()
    print("PASS primitive implode values/nulls/layouts/metadata/empty")
    test_mojo_gpu_implode_prior_rows_trailing_slices_and_profiles_budget()
    print("PASS primitive implode ordering/slices/profiles/budget/recovery")
    test_mojo_gpu_implode_gpu_errors_masks_projection_order_head_and_observer()
    print(
        "PASS primitive implode GPU errors/masks/producer order/head/observer"
    )
    test_mojo_gpu_implode_explicit_boundaries_and_metadata_only_description()
    print("PASS primitive implode explicit boundaries/description")
