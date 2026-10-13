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

from tests.execution_modes.test_mojo_gpu_grouping import (
    test_mojo_gpu_grouping_mixed_keys_all_reductions_and_nullable_layouts,
    test_mojo_gpu_grouping_key_only_stability_distinct_groups_and_compositions,
    test_mojo_gpu_grouping_exact_wide_sum_local_arg_positions_and_overflow,
)

from tests.execution_modes.test_mojo_gpu_ranking import (
    test_mojo_gpu_rank_integer_widths_chunks_methods_and_partitions,
    test_mojo_gpu_rank_floating_nan_direction_zero_ties_nulls_and_empty,
    test_mojo_gpu_rank_compositions_key_errors_budget_and_reuse,
)

from tests.execution_modes.test_mojo_gpu_windows import (
    test_mojo_gpu_windows_integer_layouts_partitions_directions_shifts_and_fills,
    test_mojo_gpu_windows_float_encodings_logical_metadata_and_empty_nulls,
    test_mojo_gpu_windows_compositions_checked_prefix_errors_budget_and_recovery,
)

from tests.execution_modes.test_mojo_gpu_quantiles import (
    test_mojo_gpu_quantiles_numeric_layouts_empty_masks_and_mixed_reductions,
    test_mojo_gpu_quantiles_nonfinite_zero_and_interpolation_anchors,
    test_mojo_gpu_quantiles_grouped_filtered_sliced_and_checked_errors,
)


def main() raises:
    print(
        "RUN test_mojo_gpu_integer_widths_nullable_slices_chunks_and_chained_programs"
    )
    test_mojo_gpu_integer_widths_nullable_slices_chunks_and_chained_programs()
    print(
        "RUN test_mojo_gpu_conditional_errors_are_observed_only_in_selected_values"
    )
    test_mojo_gpu_conditional_errors_are_observed_only_in_selected_values()
    print(
        "RUN test_mojo_gpu_integer_endpoints_and_width_overflow_match_exact_anchors"
    )
    test_mojo_gpu_integer_endpoints_and_width_overflow_match_exact_anchors()
    print(
        "RUN test_mojo_gpu_preserves_float64_storage_and_reports_direct_execution"
    )
    test_mojo_gpu_preserves_float64_storage_and_reports_direct_execution()
    print("RUN test_mojo_gpu_boolean_kleene_nulls_empty_rows_and_zero_columns")
    test_mojo_gpu_boolean_kleene_nulls_empty_rows_and_zero_columns()
    print(
        "RUN test_mojo_gpu_floating_arithmetic_predicates_rounding_and_subnormals"
    )
    test_mojo_gpu_floating_arithmetic_predicates_rounding_and_subnormals()
    print("RUN test_mojo_gpu_transcendentals_power_and_numeric_casts")
    test_mojo_gpu_transcendentals_power_and_numeric_casts()
    print(
        "RUN test_mojo_gpu_filters_slices_preserve_order_nulls_chunks_and_bits"
    )
    test_mojo_gpu_filters_slices_preserve_order_nulls_chunks_and_bits()
    print(
        "RUN test_mojo_gpu_compaction_observes_checked_work_on_surviving_rows"
    )
    test_mojo_gpu_compaction_observes_checked_work_on_surviving_rows()
    print(
        "RUN test_mojo_gpu_integer_reductions_layouts_empty_nulls_and_exact_results"
    )
    test_mojo_gpu_integer_reductions_layouts_empty_nulls_and_exact_results()
    print("RUN test_mojo_gpu_wide_reduction_cancellation_overflow_and_mean")
    test_mojo_gpu_wide_reduction_cancellation_overflow_and_mean()
    print("RUN test_mojo_gpu_float_boolean_extrema_and_filtered_reductions")
    test_mojo_gpu_float_boolean_extrema_and_filtered_reductions()
    print(
        "RUN test_mojo_gpu_ordering_integer_layouts_nulls_mixed_keys_and_stability"
    )
    test_mojo_gpu_ordering_integer_layouts_nulls_mixed_keys_and_stability()
    print(
        "RUN test_mojo_gpu_ordering_float_encodings_nan_zero_subnormal_and_nulls"
    )
    test_mojo_gpu_ordering_float_encodings_nan_zero_subnormal_and_nulls()
    print(
        "RUN test_mojo_gpu_ordering_compositions_key_errors_budget_and_recovery"
    )
    test_mojo_gpu_ordering_compositions_key_errors_budget_and_recovery()
    print(
        "RUN test_mojo_gpu_statistics_numeric_layouts_empty_masks_and_mixed_outputs"
    )
    test_mojo_gpu_statistics_numeric_layouts_empty_masks_and_mixed_outputs()
    print("RUN test_mojo_gpu_statistics_nonfinite_constant_and_scalar_anchors")
    test_mojo_gpu_statistics_nonfinite_constant_and_scalar_anchors()
    print("RUN test_mojo_gpu_statistics_filtered_sliced_and_pair_intersection")
    test_mojo_gpu_statistics_filtered_sliced_and_pair_intersection()
    print(
        "RUN test_mojo_gpu_grouping_mixed_keys_all_reductions_and_nullable_layouts"
    )
    test_mojo_gpu_grouping_mixed_keys_all_reductions_and_nullable_layouts()
    print(
        "RUN test_mojo_gpu_grouping_key_only_stability_distinct_groups_and_compositions"
    )
    test_mojo_gpu_grouping_key_only_stability_distinct_groups_and_compositions()
    print(
        "RUN test_mojo_gpu_grouping_exact_wide_sum_local_arg_positions_and_overflow"
    )
    test_mojo_gpu_grouping_exact_wide_sum_local_arg_positions_and_overflow()
    print("RUN test_mojo_gpu_rank_integer_widths_chunks_methods_and_partitions")
    test_mojo_gpu_rank_integer_widths_chunks_methods_and_partitions()
    print(
        "RUN test_mojo_gpu_rank_floating_nan_direction_zero_ties_nulls_and_empty"
    )
    test_mojo_gpu_rank_floating_nan_direction_zero_ties_nulls_and_empty()
    print("RUN test_mojo_gpu_rank_compositions_key_errors_budget_and_reuse")
    test_mojo_gpu_rank_compositions_key_errors_budget_and_reuse()
    print(
        "RUN test_mojo_gpu_windows_integer_layouts_partitions_directions_shifts_and_fills"
    )
    test_mojo_gpu_windows_integer_layouts_partitions_directions_shifts_and_fills()
    print(
        "RUN test_mojo_gpu_windows_float_encodings_logical_metadata_and_empty_nulls"
    )
    test_mojo_gpu_windows_float_encodings_logical_metadata_and_empty_nulls()
    print(
        "RUN test_mojo_gpu_windows_compositions_checked_prefix_errors_budget_and_recovery"
    )
    test_mojo_gpu_windows_compositions_checked_prefix_errors_budget_and_recovery()
    print(
        "RUN test_mojo_gpu_quantiles_numeric_layouts_empty_masks_and_mixed_reductions"
    )
    test_mojo_gpu_quantiles_numeric_layouts_empty_masks_and_mixed_reductions()
    print(
        "RUN test_mojo_gpu_quantiles_nonfinite_zero_and_interpolation_anchors"
    )
    test_mojo_gpu_quantiles_nonfinite_zero_and_interpolation_anchors()
    print(
        "RUN test_mojo_gpu_quantiles_grouped_filtered_sliced_and_checked_errors"
    )
    test_mojo_gpu_quantiles_grouped_filtered_sliced_and_checked_errors()
    print(
        "PASS direct Mojo dataframe GPU: rows, filtering, reductions, ordering, statistics, groups, ranks, windows and quantiles"
    )
