"""Literal CPU/GPU categorical transport witnesses."""
from tests.execution_modes.test_mojo_gpu_categorical_transport import (
    test_mojo_gpu_categorical_original_rows_filter_sort_group_and_over,
    test_mojo_gpu_categorical_join_duplicate_codes_and_dictionary_union,
    test_mojo_gpu_categorical_asof_temporary_by_union_preserves_output_owner,
    test_mojo_gpu_categorical_dictionary_budget_owners,
    test_mojo_gpu_categorical_mixed_join_control_readback_and_budget,
    test_mojo_gpu_coded_string_group_publication_drops_original_code_identity,
)


def main() raises:
    test_mojo_gpu_categorical_original_rows_filter_sort_group_and_over()
    test_mojo_gpu_categorical_join_duplicate_codes_and_dictionary_union()
    test_mojo_gpu_categorical_asof_temporary_by_union_preserves_output_owner()
    test_mojo_gpu_categorical_dictionary_budget_owners()
    test_mojo_gpu_categorical_mixed_join_control_readback_and_budget()
    test_mojo_gpu_coded_string_group_publication_drops_original_code_identity()
    print("PASS original categorical ROW FILTER SORT GROUP OVER JOIN ASOF")
