from tests.execution_modes.test_mojo_gpu_grouped_lists import (
    test_mojo_gpu_grouped_implode_mixed_keys_children_reductions_layouts_and_residency,
    test_mojo_gpu_grouped_implode_empty_nulls_projection_errors_head_and_exact_budget,
    test_mojo_gpu_grouped_implode_metadata_admission_and_parallel_schedule_guards,
    test_mojo_gpu_grouped_implode_large_checked_cpu_bucket_and_heavy_fault_order,
)


def main() raises:
    test_mojo_gpu_grouped_implode_mixed_keys_children_reductions_layouts_and_residency()
    print("PASS grouped List producers and resident consumers")
    test_mojo_gpu_grouped_implode_empty_nulls_projection_errors_head_and_exact_budget()
    print("PASS grouped empty/errors/head/budgets")
    test_mojo_gpu_grouped_implode_metadata_admission_and_parallel_schedule_guards()
    print("PASS grouped metadata boundaries")
    test_mojo_gpu_grouped_implode_large_checked_cpu_bucket_and_heavy_fault_order()
    print("PASS large checked grouped CPU bucket/heavy priority")
