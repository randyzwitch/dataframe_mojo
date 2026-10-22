"""Focused physical-device gate for host profiling and resident execution."""
from tests.execution_modes.test_metal_profile_phases import (
    test_metal_profile_regions_union_optional_metrics_in_first_seen_order,
    test_metal_profile_phase_schema_keeps_unknown_measurements_null,
    test_metal_profile_phases_preserve_results_and_empty_execution,
    test_metal_profile_phases_survive_join_between_row_regions,
    test_metal_profile_native_join_followed_by_scalar_sum,
)
from tests.execution_modes.test_mojo_gpu_resident_dag import (
    test_mojo_gpu_resident_dag_branches_final_export_profile_and_budget,
    test_mojo_gpu_resident_dag_empty_scalar_identity_and_zero_column_consumers,
)


def main() raises:
    test_metal_profile_phase_schema_keeps_unknown_measurements_null()
    test_metal_profile_regions_union_optional_metrics_in_first_seen_order()
    test_metal_profile_phases_preserve_results_and_empty_execution()
    test_metal_profile_phases_survive_join_between_row_regions()
    test_metal_profile_native_join_followed_by_scalar_sum()
    print("PASS host phase attribution and mixed report schemas")
    test_mojo_gpu_resident_dag_branches_final_export_profile_and_budget()
    test_mojo_gpu_resident_dag_empty_scalar_identity_and_zero_column_consumers()
    print("PASS resident results, immutable branches, exports and budgets")
