"""Metadata-only legacy witnesses inspect the exact direct Mojo programs."""
from tests.execution_modes.test_backend_metal_calendar_offsets import (
    test_calendar_offsets_capability_signed_metadata_and_checked_boundaries,
)
from tests.execution_modes.test_backend_metal_calendar_rolling import (
    test_calendar_duration_capability_headers_units_and_no_device_preflight,
)
from tests.execution_modes.test_backend_metal_calendar_truncation import (
    test_calendar_truncation_capability_month_metadata_and_checked_boundaries,
)
from tests.execution_modes.test_backend_metal_chunked_inputs import (
    test_chunked_inputs_capability_owned_descriptors_and_value_blind_estimates,
)
from tests.execution_modes.test_backend_metal_column_slots import (
    test_column_slot_preflight_reuses_values_and_bounds_outputs,
)
from tests.execution_modes.test_backend_metal_coordinate_interpolation import (
    test_coordinate_interpolation_capability_private_axis_metadata_and_checked_inputs,
)
from tests.execution_modes.test_backend_metal_cumulative_windows import (
    test_nested_window_lowering_preserves_cpu_scopes_without_device,
)
from tests.execution_modes.test_backend_metal_decimal_arithmetic import (
    test_decimal_unary_cpu_storage_widths_and_arithmetic_descriptors,
)
from tests.execution_modes.test_backend_metal_decimal_casts import (
    test_decimal_cast_capability_scale_descriptors_and_rejections,
)
from tests.execution_modes.test_backend_metal_decimal_comparisons import (
    test_decimal_comparison_capability_and_packed_metadata,
)
from tests.execution_modes.test_backend_metal_decimal_means import (
    test_decimal_mean_capabilities_and_scale_descriptors,
)
from tests.execution_modes.test_backend_metal_decimal_multiplicative import (
    test_decimal_multiplicative_capability_proofs_and_metadata,
)
from tests.execution_modes.test_backend_metal_decimal_rescaling import (
    test_decimal_rescaling_capability_and_strict_metadata,
)
from tests.execution_modes.test_backend_metal_duration_decimals import (
    test_duration_decimal_private_widening_precision_float_casts_and_no_device_preflight,
)
from tests.execution_modes.test_backend_metal_duration_rolling import (
    test_duration_rolling_capability_intervals_private_indices_types_and_scopes,
)
from tests.execution_modes.test_backend_metal_duration_totals import (
    test_duration_totals_capability_metadata_units_and_minimum_cpu_sign,
)
from tests.execution_modes.test_backend_metal_empty_inputs import (
    test_empty_inputs_capability_schema_descriptors_and_empty_selectors,
)
from tests.execution_modes.test_backend_metal_empty_results import (
    test_empty_outputs_capability_metadata_and_distinct_cpu_selection_shapes,
)
from tests.execution_modes.test_backend_metal_empty_results import (
    test_empty_selection_slices_preserve_cpu_scalar_and_row_shapes,
)
from tests.execution_modes.test_backend_metal_fixed_offsets import (
    test_fixed_offset_capability_units_descriptors_and_observation_boundaries,
)
from tests.execution_modes.test_backend_metal_fixed_zone_rolling import (
    test_fixed_zone_duration_capability_signed_offsets_no_device_and_independent_utc_gate,
)
from tests.execution_modes.test_backend_metal_floating_power import (
    test_floating_power_capability_metadata_types_and_cost_preflight,
)
from tests.execution_modes.test_backend_metal_higher_moments import (
    test_higher_moments_capability_flags_private_casts_and_precision,
)
from tests.execution_modes.test_backend_metal_interpolation import (
    test_interpolation_capability_private_types_partition_metadata_and_errors,
)
from tests.execution_modes.test_backend_metal_moments import (
    test_moments_capability_precision_metadata_and_private_casts,
)
from tests.execution_modes.test_backend_metal_pair_moments import (
    test_pair_moment_capability_right_slots_ddof_and_spearman_staging,
)
from tests.execution_modes.test_backend_metal_partition_aggregates import (
    test_partition_aggregate_metadata_preflight,
)
from tests.execution_modes.test_backend_metal_partition_aggregates import (
    test_shared_partition_metadata_preserves_cpu_empty_execution,
)
from tests.execution_modes.test_backend_metal_proven_arithmetic import (
    test_integer_proofs_cover_unsigned_product_and_literal_boundaries,
)
from tests.execution_modes.test_backend_metal_proven_rank import (
    test_proven_rank_preflight_preserves_opt_in_and_unsafe_children,
)
from tests.execution_modes.test_backend_metal_quantiles import (
    test_quantile_capabilities_private_casts_full_probability_metadata_and_binding,
)
from tests.execution_modes.test_backend_metal_rank import (
    test_rank_capability_and_checked_input_preflight,
)
from tests.execution_modes.test_backend_metal_rolling import (
    test_rolling_capability_threshold_metadata_and_nested_scope_lowering,
)
from tests.execution_modes.test_backend_metal_rolling_moments import (
    test_rolling_moments_capability_packed_bounds_private_casts_and_scopes,
)
from tests.execution_modes.test_backend_metal_scalar_broadcasts import (
    test_scalar_broadcast_planning_shares_prefix_and_preserves_schema,
)
from tests.execution_modes.test_backend_metal_scalar_broadcasts import (
    test_scalar_broadcast_rejects_file_reads_and_checked_integer_children,
)
from tests.execution_modes.test_backend_metal_scalar_projections import (
    test_scalar_projection_normalization_shape_aliases_and_metadata,
)
from tests.execution_modes.test_backend_metal_selectors import (
    test_metal_selectors_metadata_names_positions_dtypes_and_errors,
)
from tests.execution_modes.test_backend_metal_shift_fill import (
    test_shift_fill_capability_raw_parameters_and_nested_scopes,
)
from tests.execution_modes.test_backend_metal_temporal import (
    test_metal_temporal_lowering_and_cpu_models,
)
from tests.execution_modes.test_backend_metal_transcendentals import (
    test_transcendental_capability_metadata_types_and_no_device_preflight,
)
from tests.execution_modes.test_backend_metal_utc_rolling import (
    test_utc_duration_capability_staged_metadata_wide_days_and_no_device_preflight,
)
from tests.execution_modes.test_backend_metal_wide_plans import (
    test_wide_metal_metadata_and_default_backend_limits,
)


def main() raises:
    var failures = 0
    try:
        test_calendar_offsets_capability_signed_metadata_and_checked_boundaries()
        print(
            "PASS test_calendar_offsets_capability_signed_metadata_and_checked_boundaries"
        )
    except error:
        failures += 1
        print(
            "FAIL test_calendar_offsets_capability_signed_metadata_and_checked_boundaries",
            error,
        )
    try:
        test_calendar_duration_capability_headers_units_and_no_device_preflight()
        print(
            "PASS test_calendar_duration_capability_headers_units_and_no_device_preflight"
        )
    except error:
        failures += 1
        print(
            "FAIL test_calendar_duration_capability_headers_units_and_no_device_preflight",
            error,
        )
    try:
        test_calendar_truncation_capability_month_metadata_and_checked_boundaries()
        print(
            "PASS test_calendar_truncation_capability_month_metadata_and_checked_boundaries"
        )
    except error:
        failures += 1
        print(
            "FAIL test_calendar_truncation_capability_month_metadata_and_checked_boundaries",
            error,
        )
    try:
        test_chunked_inputs_capability_owned_descriptors_and_value_blind_estimates()
        print(
            "PASS test_chunked_inputs_capability_owned_descriptors_and_value_blind_estimates"
        )
    except error:
        failures += 1
        print(
            "FAIL test_chunked_inputs_capability_owned_descriptors_and_value_blind_estimates",
            error,
        )
    try:
        test_column_slot_preflight_reuses_values_and_bounds_outputs()
        print(
            "PASS test_column_slot_preflight_reuses_values_and_bounds_outputs"
        )
    except error:
        failures += 1
        print(
            "FAIL test_column_slot_preflight_reuses_values_and_bounds_outputs",
            error,
        )
    try:
        test_coordinate_interpolation_capability_private_axis_metadata_and_checked_inputs()
        print(
            "PASS test_coordinate_interpolation_capability_private_axis_metadata_and_checked_inputs"
        )
    except error:
        failures += 1
        print(
            "FAIL test_coordinate_interpolation_capability_private_axis_metadata_and_checked_inputs",
            error,
        )
    try:
        test_nested_window_lowering_preserves_cpu_scopes_without_device()
        print(
            "PASS test_nested_window_lowering_preserves_cpu_scopes_without_device"
        )
    except error:
        failures += 1
        print(
            "FAIL test_nested_window_lowering_preserves_cpu_scopes_without_device",
            error,
        )
    try:
        test_decimal_unary_cpu_storage_widths_and_arithmetic_descriptors()
        print(
            "PASS test_decimal_unary_cpu_storage_widths_and_arithmetic_descriptors"
        )
    except error:
        failures += 1
        print(
            "FAIL test_decimal_unary_cpu_storage_widths_and_arithmetic_descriptors",
            error,
        )
    try:
        test_decimal_cast_capability_scale_descriptors_and_rejections()
        print(
            "PASS test_decimal_cast_capability_scale_descriptors_and_rejections"
        )
    except error:
        failures += 1
        print(
            "FAIL test_decimal_cast_capability_scale_descriptors_and_rejections",
            error,
        )
    try:
        test_decimal_comparison_capability_and_packed_metadata()
        print("PASS test_decimal_comparison_capability_and_packed_metadata")
    except error:
        failures += 1
        print(
            "FAIL test_decimal_comparison_capability_and_packed_metadata", error
        )
    try:
        test_decimal_mean_capabilities_and_scale_descriptors()
        print("PASS test_decimal_mean_capabilities_and_scale_descriptors")
    except error:
        failures += 1
        print(
            "FAIL test_decimal_mean_capabilities_and_scale_descriptors", error
        )
    try:
        test_decimal_multiplicative_capability_proofs_and_metadata()
        print("PASS test_decimal_multiplicative_capability_proofs_and_metadata")
    except error:
        failures += 1
        print(
            "FAIL test_decimal_multiplicative_capability_proofs_and_metadata",
            error,
        )
    try:
        test_decimal_rescaling_capability_and_strict_metadata()
        print("PASS test_decimal_rescaling_capability_and_strict_metadata")
    except error:
        failures += 1
        print(
            "FAIL test_decimal_rescaling_capability_and_strict_metadata", error
        )
    try:
        test_duration_decimal_private_widening_precision_float_casts_and_no_device_preflight()
        print(
            "PASS test_duration_decimal_private_widening_precision_float_casts_and_no_device_preflight"
        )
    except error:
        failures += 1
        print(
            "FAIL test_duration_decimal_private_widening_precision_float_casts_and_no_device_preflight",
            error,
        )
    try:
        test_duration_rolling_capability_intervals_private_indices_types_and_scopes()
        print(
            "PASS test_duration_rolling_capability_intervals_private_indices_types_and_scopes"
        )
    except error:
        failures += 1
        print(
            "FAIL test_duration_rolling_capability_intervals_private_indices_types_and_scopes",
            error,
        )
    try:
        test_duration_totals_capability_metadata_units_and_minimum_cpu_sign()
        print(
            "PASS test_duration_totals_capability_metadata_units_and_minimum_cpu_sign"
        )
    except error:
        failures += 1
        print(
            "FAIL test_duration_totals_capability_metadata_units_and_minimum_cpu_sign",
            error,
        )
    try:
        test_empty_inputs_capability_schema_descriptors_and_empty_selectors()
        print(
            "PASS test_empty_inputs_capability_schema_descriptors_and_empty_selectors"
        )
    except error:
        failures += 1
        print(
            "FAIL test_empty_inputs_capability_schema_descriptors_and_empty_selectors",
            error,
        )
    try:
        test_empty_outputs_capability_metadata_and_distinct_cpu_selection_shapes()
        print(
            "PASS test_empty_outputs_capability_metadata_and_distinct_cpu_selection_shapes"
        )
    except error:
        failures += 1
        print(
            "FAIL test_empty_outputs_capability_metadata_and_distinct_cpu_selection_shapes",
            error,
        )
    try:
        test_empty_selection_slices_preserve_cpu_scalar_and_row_shapes()
        print(
            "PASS test_empty_selection_slices_preserve_cpu_scalar_and_row_shapes"
        )
    except error:
        failures += 1
        print(
            "FAIL test_empty_selection_slices_preserve_cpu_scalar_and_row_shapes",
            error,
        )
    try:
        test_fixed_offset_capability_units_descriptors_and_observation_boundaries()
        print(
            "PASS test_fixed_offset_capability_units_descriptors_and_observation_boundaries"
        )
    except error:
        failures += 1
        print(
            "FAIL test_fixed_offset_capability_units_descriptors_and_observation_boundaries",
            error,
        )
    try:
        test_fixed_zone_duration_capability_signed_offsets_no_device_and_independent_utc_gate()
        print(
            "PASS test_fixed_zone_duration_capability_signed_offsets_no_device_and_independent_utc_gate"
        )
    except error:
        failures += 1
        print(
            "FAIL test_fixed_zone_duration_capability_signed_offsets_no_device_and_independent_utc_gate",
            error,
        )
    try:
        test_floating_power_capability_metadata_types_and_cost_preflight()
        print(
            "PASS test_floating_power_capability_metadata_types_and_cost_preflight"
        )
    except error:
        failures += 1
        print(
            "FAIL test_floating_power_capability_metadata_types_and_cost_preflight",
            error,
        )
    try:
        test_higher_moments_capability_flags_private_casts_and_precision()
        print(
            "PASS test_higher_moments_capability_flags_private_casts_and_precision"
        )
    except error:
        failures += 1
        print(
            "FAIL test_higher_moments_capability_flags_private_casts_and_precision",
            error,
        )
    try:
        test_interpolation_capability_private_types_partition_metadata_and_errors()
        print(
            "PASS test_interpolation_capability_private_types_partition_metadata_and_errors"
        )
    except error:
        failures += 1
        print(
            "FAIL test_interpolation_capability_private_types_partition_metadata_and_errors",
            error,
        )
    try:
        test_moments_capability_precision_metadata_and_private_casts()
        print(
            "PASS test_moments_capability_precision_metadata_and_private_casts"
        )
    except error:
        failures += 1
        print(
            "FAIL test_moments_capability_precision_metadata_and_private_casts",
            error,
        )
    try:
        test_pair_moment_capability_right_slots_ddof_and_spearman_staging()
        print(
            "PASS test_pair_moment_capability_right_slots_ddof_and_spearman_staging"
        )
    except error:
        failures += 1
        print(
            "FAIL test_pair_moment_capability_right_slots_ddof_and_spearman_staging",
            error,
        )
    try:
        test_partition_aggregate_metadata_preflight()
        print("PASS test_partition_aggregate_metadata_preflight")
    except error:
        failures += 1
        print("FAIL test_partition_aggregate_metadata_preflight", error)
    try:
        test_shared_partition_metadata_preserves_cpu_empty_execution()
        print(
            "PASS test_shared_partition_metadata_preserves_cpu_empty_execution"
        )
    except error:
        failures += 1
        print(
            "FAIL test_shared_partition_metadata_preserves_cpu_empty_execution",
            error,
        )
    try:
        test_integer_proofs_cover_unsigned_product_and_literal_boundaries()
        print(
            "PASS test_integer_proofs_cover_unsigned_product_and_literal_boundaries"
        )
    except error:
        failures += 1
        print(
            "FAIL test_integer_proofs_cover_unsigned_product_and_literal_boundaries",
            error,
        )
    try:
        test_proven_rank_preflight_preserves_opt_in_and_unsafe_children()
        print(
            "PASS test_proven_rank_preflight_preserves_opt_in_and_unsafe_children"
        )
    except error:
        failures += 1
        print(
            "FAIL test_proven_rank_preflight_preserves_opt_in_and_unsafe_children",
            error,
        )
    try:
        test_quantile_capabilities_private_casts_full_probability_metadata_and_binding()
        print(
            "PASS test_quantile_capabilities_private_casts_full_probability_metadata_and_binding"
        )
    except error:
        failures += 1
        print(
            "FAIL test_quantile_capabilities_private_casts_full_probability_metadata_and_binding",
            error,
        )
    try:
        test_rank_capability_and_checked_input_preflight()
        print("PASS test_rank_capability_and_checked_input_preflight")
    except error:
        failures += 1
        print("FAIL test_rank_capability_and_checked_input_preflight", error)
    try:
        test_rolling_capability_threshold_metadata_and_nested_scope_lowering()
        print(
            "PASS test_rolling_capability_threshold_metadata_and_nested_scope_lowering"
        )
    except error:
        failures += 1
        print(
            "FAIL test_rolling_capability_threshold_metadata_and_nested_scope_lowering",
            error,
        )
    try:
        test_rolling_moments_capability_packed_bounds_private_casts_and_scopes()
        print(
            "PASS test_rolling_moments_capability_packed_bounds_private_casts_and_scopes"
        )
    except error:
        failures += 1
        print(
            "FAIL test_rolling_moments_capability_packed_bounds_private_casts_and_scopes",
            error,
        )
    try:
        test_scalar_broadcast_planning_shares_prefix_and_preserves_schema()
        print(
            "PASS test_scalar_broadcast_planning_shares_prefix_and_preserves_schema"
        )
    except error:
        failures += 1
        print(
            "FAIL test_scalar_broadcast_planning_shares_prefix_and_preserves_schema",
            error,
        )
    try:
        test_scalar_broadcast_rejects_file_reads_and_checked_integer_children()
        print(
            "PASS test_scalar_broadcast_rejects_file_reads_and_checked_integer_children"
        )
    except error:
        failures += 1
        print(
            "FAIL test_scalar_broadcast_rejects_file_reads_and_checked_integer_children",
            error,
        )
    try:
        test_scalar_projection_normalization_shape_aliases_and_metadata()
        print(
            "PASS test_scalar_projection_normalization_shape_aliases_and_metadata"
        )
    except error:
        failures += 1
        print(
            "FAIL test_scalar_projection_normalization_shape_aliases_and_metadata",
            error,
        )
    try:
        test_metal_selectors_metadata_names_positions_dtypes_and_errors()
        print(
            "PASS test_metal_selectors_metadata_names_positions_dtypes_and_errors"
        )
    except error:
        failures += 1
        print(
            "FAIL test_metal_selectors_metadata_names_positions_dtypes_and_errors",
            error,
        )
    try:
        test_shift_fill_capability_raw_parameters_and_nested_scopes()
        print(
            "PASS test_shift_fill_capability_raw_parameters_and_nested_scopes"
        )
    except error:
        failures += 1
        print(
            "FAIL test_shift_fill_capability_raw_parameters_and_nested_scopes",
            error,
        )
    try:
        test_metal_temporal_lowering_and_cpu_models()
        print("PASS test_metal_temporal_lowering_and_cpu_models")
    except error:
        failures += 1
        print("FAIL test_metal_temporal_lowering_and_cpu_models", error)
    try:
        test_transcendental_capability_metadata_types_and_no_device_preflight()
        print(
            "PASS test_transcendental_capability_metadata_types_and_no_device_preflight"
        )
    except error:
        failures += 1
        print(
            "FAIL test_transcendental_capability_metadata_types_and_no_device_preflight",
            error,
        )
    try:
        test_utc_duration_capability_staged_metadata_wide_days_and_no_device_preflight()
        print(
            "PASS test_utc_duration_capability_staged_metadata_wide_days_and_no_device_preflight"
        )
    except error:
        failures += 1
        print(
            "FAIL test_utc_duration_capability_staged_metadata_wide_days_and_no_device_preflight",
            error,
        )
    try:
        test_wide_metal_metadata_and_default_backend_limits()
        print("PASS test_wide_metal_metadata_and_default_backend_limits")
    except error:
        failures += 1
        print("FAIL test_wide_metal_metadata_and_default_backend_limits", error)
    if failures:
        raise Error("Mojo legacy metadata failures: " + String(failures))
    print("PASS 43 metadata-only legacy Mojo contracts")
