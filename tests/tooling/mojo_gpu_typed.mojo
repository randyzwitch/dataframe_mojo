"""Focused device gate for typed ownership and native Float32 row execution."""
from tests.execution_modes.test_gpu_typed_storage_metadata import (
    test_native_float32_column_capacity_includes_only_present_sidecars,
    test_native_column_capacity_checks_budget_before_allocation,
    test_native_reservation_precedes_allocation_and_releases_shared_scope,
    test_native_float32_upload_roundtrips_slice_and_packed_nulls,
    test_native_float32_upload_rechunks_only_validity_for_chunked_series,
    test_native_float32_empty_and_budget_failure_allocate_nothing,
    test_native_float32_views_retain_one_owner_and_readback_checks_budget,
)
from tests.execution_modes.test_mojo_gpu_typed_rows import (
    test_mojo_gpu_typed_rows_float32_projection_and_arithmetic15,
    test_mojo_gpu_typed_rows_nullable_offset_slices_chunks_and_field_metadata,
    test_mojo_gpu_typed_rows_filter_head_after_filter_and_zero_rows,
    test_mojo_gpu_typed_rows_computed_head_preserves_limits,
    test_mojo_gpu_typed_rows_float32_special_values_and_sum_fallback,
    test_mojo_gpu_typed_filter_multiple_blocks_and_query_budget,
    test_mojo_gpu_typed_float32_join_uses_narrow_native_input_bridge,
    test_mojo_gpu_typed_float32_join_bridge_float_keys_empty_and_compacted_filter,
    test_mojo_gpu_typed_float32_join_keeps_multicolumn_producer_generic,
    test_mojo_gpu_typed_float32_join_keeps_branched_producers_scoped,
)
from tests.execution_modes.test_metal_profile_phases import (
    test_metal_profile_phases_preserve_results_and_empty_execution,
)


def main() raises:
    test_native_float32_column_capacity_includes_only_present_sidecars()
    test_native_column_capacity_checks_budget_before_allocation()
    test_native_reservation_precedes_allocation_and_releases_shared_scope()
    test_native_float32_upload_roundtrips_slice_and_packed_nulls()
    test_native_float32_upload_rechunks_only_validity_for_chunked_series()
    test_native_float32_empty_and_budget_failure_allocate_nothing()
    test_native_float32_views_retain_one_owner_and_readback_checks_budget()
    print("PASS native storage slices, chunks, nulls, lifetimes and budgets")
    test_mojo_gpu_typed_rows_float32_projection_and_arithmetic15()
    test_mojo_gpu_typed_rows_nullable_offset_slices_chunks_and_field_metadata()
    test_mojo_gpu_typed_rows_filter_head_after_filter_and_zero_rows()
    test_mojo_gpu_typed_rows_computed_head_preserves_limits()
    test_mojo_gpu_typed_rows_float32_special_values_and_sum_fallback()
    test_mojo_gpu_typed_filter_multiple_blocks_and_query_budget()
    test_mojo_gpu_typed_float32_join_uses_narrow_native_input_bridge()
    test_mojo_gpu_typed_float32_join_bridge_float_keys_empty_and_compacted_filter()
    test_mojo_gpu_typed_float32_join_keeps_multicolumn_producer_generic()
    test_mojo_gpu_typed_float32_join_keeps_branched_producers_scoped()
    test_metal_profile_phases_preserve_results_and_empty_execution()
    print(
        "PASS native Float32 arithmetic, stable filters, metadata and profiles"
    )
