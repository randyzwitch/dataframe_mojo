"""Source List public CPU/GPU and resident transport proof."""
from tests.execution_modes.test_mojo_gpu_source_lists import (
    test_mojo_gpu_source_list_get_len_null_parents_layouts_metadata_and_computed_rows,
    test_mojo_gpu_source_list_oob_empty_error_order_observer_and_exact_budget,
    test_mojo_gpu_source_list_metadata_only_admission_and_retained_boundaries,
)


def main() raises:
    test_mojo_gpu_source_list_get_len_null_parents_layouts_metadata_and_computed_rows()
    print("PASS source List resident len/get layouts/metadata/nullable values")
    test_mojo_gpu_source_list_oob_empty_error_order_observer_and_exact_budget()
    print("PASS source List OOB/empty/producer errors/observer/budget/recovery")
    test_mojo_gpu_source_list_metadata_only_admission_and_retained_boundaries()
    print("PASS source List metadata-only admission/explicit boundaries")
