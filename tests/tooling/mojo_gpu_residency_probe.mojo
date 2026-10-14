"""Isolated ownership probe; does not import the query runtime."""
from tests.execution_modes.test_mojo_gpu_residency import (
    test_mojo_gpu_residency_lazy_empty_identity_and_zero_column_cardinality,
    test_mojo_gpu_residency_shared_paired_matrix_device_copy_and_lifetime,
    test_mojo_gpu_residency_export_budget_and_deferred_errors,
    test_mojo_gpu_residency_structural_identity_lazy_upload_and_target_offset,
)


def main() raises:
    test_mojo_gpu_residency_lazy_empty_identity_and_zero_column_cardinality()
    print("PASS lazy structural identities and zero-column cardinality")
    test_mojo_gpu_residency_shared_paired_matrix_device_copy_and_lifetime()
    print("PASS immutable owners and paired device copies")
    test_mojo_gpu_residency_export_budget_and_deferred_errors()
    print("PASS export budget and deferred errors")
    test_mojo_gpu_residency_structural_identity_lazy_upload_and_target_offset()
    print("PASS structural identity upload and target offset")
