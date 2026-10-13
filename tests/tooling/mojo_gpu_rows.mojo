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

from tests.execution_modes.test_mojo_gpu_reductions import (
    test_mojo_gpu_integer_reductions_layouts_empty_nulls_and_exact_results,
    test_mojo_gpu_wide_reduction_cancellation_overflow_and_mean,
    test_mojo_gpu_float_boolean_extrema_and_filtered_reductions,
)

from tests.execution_modes.test_mojo_gpu_ordering import (
    test_mojo_gpu_ordering_integer_layouts_nulls_mixed_keys_and_stability,
    test_mojo_gpu_ordering_float_encodings_nan_zero_subnormal_and_nulls,
    test_mojo_gpu_ordering_compositions_key_errors_budget_and_recovery,
)

from tests.execution_modes.test_mojo_gpu_statistics import (
    test_mojo_gpu_statistics_numeric_layouts_empty_masks_and_mixed_outputs,
    test_mojo_gpu_statistics_nonfinite_constant_and_scalar_anchors,
    test_mojo_gpu_statistics_filtered_sliced_and_pair_intersection,
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
    test_mojo_gpu_integer_reductions_layouts_empty_nulls_and_exact_results()
    test_mojo_gpu_wide_reduction_cancellation_overflow_and_mean()
    test_mojo_gpu_float_boolean_extrema_and_filtered_reductions()
    test_mojo_gpu_ordering_integer_layouts_nulls_mixed_keys_and_stability()
    test_mojo_gpu_ordering_float_encodings_nan_zero_subnormal_and_nulls()
    test_mojo_gpu_ordering_compositions_key_errors_budget_and_recovery()
    test_mojo_gpu_statistics_numeric_layouts_empty_masks_and_mixed_outputs()
    test_mojo_gpu_statistics_nonfinite_constant_and_scalar_anchors()
    test_mojo_gpu_statistics_filtered_sliced_and_pair_intersection()
    print(
        "PASS direct Mojo dataframe GPU: rows, filtering, reductions, ordering and statistics"
    )
