"""Focused legacy behavioral witnesses against the Mojo Metal provider.

The imported fixtures keep their independent result/error/budget anchors.
Run with DATAFRAME_METAL_LIBRARY pointing to an unavailable dylib to ensure
the native dataframe ABI cannot participate. No fixtures are defined here.
"""

from tests.execution_modes.test_backend_metal_decimal_casts import (
    test_native_metal_decimal_cast_all_scales_widths_and_integer_rounding,
)

from tests.execution_modes.test_backend_metal_decimal_rescaling import (
    test_native_metal_decimal_rescaling_all_scales_widths_and_numeric_targets,
    test_native_metal_integer_decimal_casts_exact_words_nulls_and_consumers,
    test_native_metal_decimal_cast_errors_observation_lax_budget_and_reuse,
)

from tests.execution_modes.test_backend_metal_decimal_arithmetic import (
    test_native_metal_decimal_add_subtract_all_widths_scales_and_unary_metadata,
)

from tests.execution_modes.test_backend_metal_decimal_multiplicative import (
    test_native_metal_decimal_multiplication_division_all_scales_widths_nulls_and_slices,
    test_native_metal_decimal_multiplicative_errors_conditionals_zero_divisor_and_reuse,
)

from tests.execution_modes.test_backend_metal_decimal_means import (
    test_native_metal_decimal_mean_all_scales_signs_and_grouped_rounding,
    test_native_metal_decimal_mean_empty_nulls_partial_boundaries_and_resident_rows,
    test_native_metal_decimal_mean_profile_cost_budget_and_recovery,
)

from tests.execution_modes.test_backend_metal_duration_decimals import (
    test_native_metal_duration_decimals_all_widths_scales_signs_layouts_partitions_and_boundaries,
)

from tests.execution_modes.test_backend_metal_moments import (
    test_native_metal_moments_large_worker_merges_and_conditioned_accuracy,
)

from tests.execution_modes.test_backend_metal_higher_moments import (
    test_native_metal_higher_moments_workers_compositions_cost_budget_and_runtime_reuse,
)

from tests.execution_modes.test_backend_metal_pair_moments import (
    test_native_metal_pair_moments_large_workers_centered_accuracy_and_spearman_ties,
    test_native_metal_pair_moments_ieee_null_pairs_unsigned_and_wide_physical_slots,
)

from tests.execution_modes.test_backend_metal_interpolation import (
    test_native_metal_interpolation_numeric_widths_decimals_null_partitions_slices_and_chunks,
)

from tests.execution_modes.test_backend_metal_coordinate_interpolation import (
    test_native_metal_coordinate_interpolation_numeric_decimals_partitions_and_chunked_sources,
    test_native_metal_coordinate_interpolation_independent_distances_duplicate_axes_and_consumers,
)

from tests.execution_modes.test_backend_metal_wide_plans import (
    test_native_metal_256_column_mixed_payloads_physical_words_and_resident_boundaries,
    test_native_metal_wide_metal_programs_steps_and_resident_consumers,
)

from tests.execution_modes.test_backend_metal_dictionaries import (
    test_native_metal_categorical_join_styles_reconcile_independent_codes_and_restore_dictionary_identity,
    test_native_metal_dictionary_distinct_counts_include_nulls_and_feed_aggregate_compositions,
)

from tests.execution_modes.test_backend_metal_string_concat import (
    test_native_metal_concat_dictionary_group_filter_join_order_and_duplicate_outputs,
)

from tests.execution_modes.test_backend_metal_resident import (
    test_native_metal_nested_union_and_extreme_slices_preserve_all_typed_payloads_and_order,
    test_native_metal_distinct_keep_modes_use_logical_null_nan_zero_classes_and_current_slice_positions,
)

from tests.execution_modes.test_backend_metal_partition_aggregates import (
    test_native_metal_deep_shared_partition_metadata_stays_resident,
    test_native_metal_nested_partition_floating_worker_order,
    test_native_metal_nested_partition_aggregates_composition_consumers_and_nulls,
)

from tests.execution_modes.test_backend_metal_scalar_broadcasts import (
    test_native_metal_scalar_broadcast_resident_traffic_budgets_and_recovery,
)

from tests.execution_modes.test_backend_metal_cumulative_windows import (
    test_native_metal_cumulative_nested_scopes_resident_consumers_empty_and_budget,
)

from tests.execution_modes.test_backend_metal_column_slots import (
    test_native_metal_column_slot_gpu_resident_joins_reductions_and_dictionaries,
)

from tests.execution_modes.test_backend_metal_shift_fill import (
    test_native_metal_shift_fill_partition_boundaries_and_nested_resident_consumers,
)

from tests.execution_modes.test_mojo_gpu_integers import (
    test_mojo_gpu_integer_division_modulo_power_rounding_and_membership,
    test_mojo_gpu_integer_operator_endpoints_conditional_errors_and_reuse,
)


def main() raises:
    print(
        "RUN test_native_metal_decimal_cast_all_scales_widths_and_integer_rounding"
    )
    test_native_metal_decimal_cast_all_scales_widths_and_integer_rounding()
    print(
        "RUN test_native_metal_decimal_rescaling_all_scales_widths_and_numeric_targets"
    )
    test_native_metal_decimal_rescaling_all_scales_widths_and_numeric_targets()
    print(
        "RUN test_native_metal_integer_decimal_casts_exact_words_nulls_and_consumers"
    )
    test_native_metal_integer_decimal_casts_exact_words_nulls_and_consumers()
    print(
        "RUN test_native_metal_decimal_cast_errors_observation_lax_budget_and_reuse"
    )
    test_native_metal_decimal_cast_errors_observation_lax_budget_and_reuse()
    print(
        "RUN test_native_metal_decimal_add_subtract_all_widths_scales_and_unary_metadata"
    )
    test_native_metal_decimal_add_subtract_all_widths_scales_and_unary_metadata()
    print(
        "RUN test_native_metal_decimal_multiplication_division_all_scales_widths_nulls_and_slices"
    )
    test_native_metal_decimal_multiplication_division_all_scales_widths_nulls_and_slices()
    print(
        "RUN test_native_metal_decimal_multiplicative_errors_conditionals_zero_divisor_and_reuse"
    )
    test_native_metal_decimal_multiplicative_errors_conditionals_zero_divisor_and_reuse()
    print(
        "RUN test_native_metal_decimal_mean_all_scales_signs_and_grouped_rounding"
    )
    test_native_metal_decimal_mean_all_scales_signs_and_grouped_rounding()
    print(
        "RUN test_native_metal_decimal_mean_empty_nulls_partial_boundaries_and_resident_rows"
    )
    test_native_metal_decimal_mean_empty_nulls_partial_boundaries_and_resident_rows()
    print("RUN test_native_metal_decimal_mean_profile_cost_budget_and_recovery")
    test_native_metal_decimal_mean_profile_cost_budget_and_recovery()
    print(
        "RUN test_native_metal_duration_decimals_all_widths_scales_signs_layouts_partitions_and_boundaries"
    )
    test_native_metal_duration_decimals_all_widths_scales_signs_layouts_partitions_and_boundaries()
    print(
        "RUN test_native_metal_moments_large_worker_merges_and_conditioned_accuracy"
    )
    test_native_metal_moments_large_worker_merges_and_conditioned_accuracy()
    print(
        "RUN test_native_metal_higher_moments_workers_compositions_cost_budget_and_runtime_reuse"
    )
    test_native_metal_higher_moments_workers_compositions_cost_budget_and_runtime_reuse()
    print(
        "RUN test_native_metal_pair_moments_large_workers_centered_accuracy_and_spearman_ties"
    )
    test_native_metal_pair_moments_large_workers_centered_accuracy_and_spearman_ties()
    print(
        "RUN test_native_metal_pair_moments_ieee_null_pairs_unsigned_and_wide_physical_slots"
    )
    test_native_metal_pair_moments_ieee_null_pairs_unsigned_and_wide_physical_slots()
    print(
        "RUN test_native_metal_interpolation_numeric_widths_decimals_null_partitions_slices_and_chunks"
    )
    test_native_metal_interpolation_numeric_widths_decimals_null_partitions_slices_and_chunks()
    print(
        "RUN test_native_metal_coordinate_interpolation_numeric_decimals_partitions_and_chunked_sources"
    )
    test_native_metal_coordinate_interpolation_numeric_decimals_partitions_and_chunked_sources()
    print(
        "RUN test_native_metal_coordinate_interpolation_independent_distances_duplicate_axes_and_consumers"
    )
    test_native_metal_coordinate_interpolation_independent_distances_duplicate_axes_and_consumers()
    print(
        "RUN test_native_metal_256_column_mixed_payloads_physical_words_and_resident_boundaries"
    )
    test_native_metal_256_column_mixed_payloads_physical_words_and_resident_boundaries()
    print(
        "RUN test_native_metal_wide_metal_programs_steps_and_resident_consumers"
    )
    test_native_metal_wide_metal_programs_steps_and_resident_consumers()
    print(
        "RUN test_native_metal_categorical_join_styles_reconcile_independent_codes_and_restore_dictionary_identity"
    )
    test_native_metal_categorical_join_styles_reconcile_independent_codes_and_restore_dictionary_identity()
    print(
        "RUN test_native_metal_dictionary_distinct_counts_include_nulls_and_feed_aggregate_compositions"
    )
    test_native_metal_dictionary_distinct_counts_include_nulls_and_feed_aggregate_compositions()
    print(
        "RUN test_native_metal_concat_dictionary_group_filter_join_order_and_duplicate_outputs"
    )
    test_native_metal_concat_dictionary_group_filter_join_order_and_duplicate_outputs()
    print(
        "RUN test_native_metal_nested_union_and_extreme_slices_preserve_all_typed_payloads_and_order"
    )
    test_native_metal_nested_union_and_extreme_slices_preserve_all_typed_payloads_and_order()
    print(
        "RUN test_native_metal_distinct_keep_modes_use_logical_null_nan_zero_classes_and_current_slice_positions"
    )
    test_native_metal_distinct_keep_modes_use_logical_null_nan_zero_classes_and_current_slice_positions()
    print("RUN test_native_metal_deep_shared_partition_metadata_stays_resident")
    test_native_metal_deep_shared_partition_metadata_stays_resident()
    print("RUN test_native_metal_nested_partition_floating_worker_order")
    test_native_metal_nested_partition_floating_worker_order()
    print(
        "RUN test_native_metal_nested_partition_aggregates_composition_consumers_and_nulls"
    )
    test_native_metal_nested_partition_aggregates_composition_consumers_and_nulls()
    print(
        "RUN test_native_metal_scalar_broadcast_resident_traffic_budgets_and_recovery"
    )
    test_native_metal_scalar_broadcast_resident_traffic_budgets_and_recovery()
    print(
        "RUN test_native_metal_cumulative_nested_scopes_resident_consumers_empty_and_budget"
    )
    test_native_metal_cumulative_nested_scopes_resident_consumers_empty_and_budget()
    print(
        "RUN test_native_metal_column_slot_gpu_resident_joins_reductions_and_dictionaries"
    )
    test_native_metal_column_slot_gpu_resident_joins_reductions_and_dictionaries()
    print(
        "RUN test_native_metal_shift_fill_partition_boundaries_and_nested_resident_consumers"
    )
    test_native_metal_shift_fill_partition_boundaries_and_nested_resident_consumers()
    print(
        "RUN test_mojo_gpu_integer_division_modulo_power_rounding_and_membership"
    )
    test_mojo_gpu_integer_division_modulo_power_rounding_and_membership()
    print(
        "RUN test_mojo_gpu_integer_operator_endpoints_conditional_errors_and_reuse"
    )
    test_mojo_gpu_integer_operator_endpoints_conditional_errors_and_reuse()
    print("PASS Mojo GPU legacy behavior: 34 selected functions")
