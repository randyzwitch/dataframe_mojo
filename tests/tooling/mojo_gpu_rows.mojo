"""Exercise the dataframe's direct Mojo GPU implementation on a real device."""
from tests.execution_modes.test_mojo_gpu_rows import (
    test_mojo_gpu_integer_widths_nullable_slices_chunks_and_chained_programs,
    test_mojo_gpu_conditional_errors_are_observed_only_in_selected_values,
    test_mojo_gpu_integer_endpoints_and_width_overflow_match_exact_anchors,
    test_mojo_gpu_preserves_float64_storage_and_reports_direct_execution,
    test_mojo_gpu_boolean_kleene_nulls_empty_rows_and_zero_columns,
    test_mojo_gpu_floating_arithmetic_predicates_rounding_and_subnormals,
    test_mojo_gpu_transcendentals_power_and_numeric_casts,
    test_mojo_gpu_filters_slices_preserve_order_nulls_chunks_and_bits,
    test_mojo_gpu_compaction_observes_checked_work_on_surviving_rows,
)


def main() raises:
    test_mojo_gpu_integer_widths_nullable_slices_chunks_and_chained_programs()
    test_mojo_gpu_conditional_errors_are_observed_only_in_selected_values()
    test_mojo_gpu_integer_endpoints_and_width_overflow_match_exact_anchors()
    test_mojo_gpu_preserves_float64_storage_and_reports_direct_execution()
    test_mojo_gpu_boolean_kleene_nulls_empty_rows_and_zero_columns()
    test_mojo_gpu_floating_arithmetic_predicates_rounding_and_subnormals()
    test_mojo_gpu_transcendentals_power_and_numeric_casts()
    test_mojo_gpu_filters_slices_preserve_order_nulls_chunks_and_bits()
    test_mojo_gpu_compaction_observes_checked_work_on_surviving_rows()
    print(
        "PASS direct Mojo dataframe GPU rows: widths, nulls, ownership, errors and bit encodings"
    )
