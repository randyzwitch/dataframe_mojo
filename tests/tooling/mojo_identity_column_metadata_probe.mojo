"""Primitive identity metadata CPU and real GPU contract proof."""
from tests.public_contracts.test_identity_column_metadata import (
    test_identity_column_metadata_primitive_logical_empty_and_batched,
    test_identity_column_metadata_full_worker_and_nested_boundaries,
)
from tests.execution_modes.test_mojo_gpu_identity_column_metadata import (
    test_mojo_gpu_identity_column_metadata_logical_views_empty_and_budget,
    test_mojo_gpu_identity_column_metadata_struct_transport_and_fresh_results,
)


def main() raises:
    test_identity_column_metadata_primitive_logical_empty_and_batched()
    print("PASS CPU identity metadata primitive/logical/empty/batched")
    test_identity_column_metadata_full_worker_and_nested_boundaries()
    print("PASS CPU identity metadata full/worker/nested boundaries")
    test_mojo_gpu_identity_column_metadata_logical_views_empty_and_budget()
    print("PASS GPU identity metadata logical/views/empty/budget")
    test_mojo_gpu_identity_column_metadata_struct_transport_and_fresh_results()
    print("PASS GPU identity metadata Struct transport/fresh results")
