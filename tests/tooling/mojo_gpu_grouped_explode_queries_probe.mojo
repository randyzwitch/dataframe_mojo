from tests.execution_modes.test_mojo_gpu_grouped_explode import (
    test_mojo_gpu_grouped_explode_layouts_metadata_stable_children_and_retained_producer,
    test_mojo_gpu_grouped_explode_resident_row_join_group_slices_and_source_failure,
    test_mojo_gpu_grouped_explode_original_phase_errors_empty_observer_and_exact_budget,
)


def main() raises:
    test_mojo_gpu_grouped_explode_layouts_metadata_stable_children_and_retained_producer()
    print("PASS grouped EXPLODE layouts/metadata/resident owners")
    test_mojo_gpu_grouped_explode_resident_row_join_group_slices_and_source_failure()
    print("PASS grouped EXPLODE ROW/JOIN/GROUP/slices/source errors")
    test_mojo_gpu_grouped_explode_original_phase_errors_empty_observer_and_exact_budget()
    print(
        "PASS grouped EXPLODE producer order/budgets/observer/empty/provenance"
    )
