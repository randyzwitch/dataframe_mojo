"""Focused CPU/GPU primitive Struct identity-cast metadata contracts."""
from tests.public_contracts.test_struct_field_metadata import (
    test_struct_field_identity_cast_metadata_empty_batched_masks_and_computed,
    test_struct_field_identity_cast_worker_and_zero_row_packed_siblings,
)
from tests.execution_modes.test_mojo_gpu_source_structs import (
    test_mojo_gpu_source_struct_identity_cast_field_metadata_and_masks,
    test_mojo_gpu_source_struct_refusals_and_explanation_remain_metadata_only,
)


def main() raises:
    test_struct_field_identity_cast_metadata_empty_batched_masks_and_computed()
    test_struct_field_identity_cast_worker_and_zero_row_packed_siblings()
    print(
        "PASS CPU Struct field identity casts, accumulators and empty metadata"
    )
    test_mojo_gpu_source_struct_refusals_and_explanation_remain_metadata_only()
    test_mojo_gpu_source_struct_identity_cast_field_metadata_and_masks()
    print(
        "PASS GPU Struct field identity casts, masks, metadata and retained refusals"
    )
