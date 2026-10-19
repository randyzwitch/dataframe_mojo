"""Focused integration gate for typed storage, context reuse and fused sums."""
from tests.execution_modes.test_mojo_gpu_device_service import (
    test_mojo_gpu_device_service_metadata_copy_and_budget_are_context_free,
    test_mojo_gpu_device_service_reuses_idle_context_and_discards_poison,
    test_mojo_gpu_direct_runtime_reports_final_service_drain,
    test_mojo_gpu_device_service_repeated_typed_collections_share_context,
    test_mojo_gpu_device_service_empty_and_preflight_leave_no_context,
    test_mojo_gpu_device_service_budget_rejection_preserves_warm_context,
)
from dataframe.accelerator.gpu.target import METAL_TARGET
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
    test_mojo_gpu_typed_rows_float32_special_values_and_sum_fallback,
    test_mojo_gpu_typed_filter_multiple_blocks_and_query_budget,
)
from tests.execution_modes.test_metal_profile_phases import (
    test_metal_profile_phases_preserve_results_and_empty_execution,
)
from tests.execution_modes.test_mojo_gpu_fused_int32 import (
    test_fused_int32_sum_nulls_layouts_empty_and_min_count,
    test_fused_int32_sum_bounded_partials_exact_budget_and_empty_device,
    test_fused_int32_sum_preserves_checked_arithmetic_and_filter_order,
    test_fused_int32_sum_independent_signed_endpoints_and_cancellation,
)


def main() raises:
    test_mojo_gpu_device_service_metadata_copy_and_budget_are_context_free()
    print(
        "PASS context-free metadata, copied runtime identity and query budget"
    )
    comptime if METAL_TARGET:
        test_mojo_gpu_device_service_reuses_idle_context_and_discards_poison()
        test_mojo_gpu_direct_runtime_reports_final_service_drain()
        test_mojo_gpu_device_service_repeated_typed_collections_share_context()
        test_mojo_gpu_device_service_empty_and_preflight_leave_no_context()
        test_mojo_gpu_device_service_budget_rejection_preserves_warm_context()
        print(
            "PASS public typed execution, overlapping leases, poison and budget"
        )
    else:
        print("SKIP Metal service execution cases on a non-Metal target")
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
    test_mojo_gpu_typed_rows_float32_special_values_and_sum_fallback()
    test_mojo_gpu_typed_filter_multiple_blocks_and_query_budget()
    test_metal_profile_phases_preserve_results_and_empty_execution()
    print(
        "PASS native Float32 arithmetic, stable filters, metadata and profiles"
    )
    test_fused_int32_sum_nulls_layouts_empty_and_min_count()
    print("PASS fused sum layouts/nulls/empty/min_count")
    test_fused_int32_sum_bounded_partials_exact_budget_and_empty_device()
    print("PASS fused sum bounded partials and budgets")
    test_fused_int32_sum_independent_signed_endpoints_and_cancellation()
    print("PASS fused sum independent endpoints and cross-partial cancellation")
    test_fused_int32_sum_preserves_checked_arithmetic_and_filter_order()
    print("PASS fused sum checked arithmetic and filter ordering")
