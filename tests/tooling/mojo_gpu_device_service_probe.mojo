"""Focused public-runtime probe for lazy Metal context reuse and cleanup."""
from tests.execution_modes.test_mojo_gpu_device_service import (
    test_mojo_gpu_device_service_metadata_copy_and_budget_are_context_free,
    test_mojo_gpu_device_service_reuses_idle_context_and_discards_poison,
    test_mojo_gpu_direct_runtime_reports_final_service_drain,
    test_mojo_gpu_device_service_repeated_typed_collections_share_context,
    test_mojo_gpu_device_service_empty_and_preflight_leave_no_context,
    test_mojo_gpu_device_service_budget_rejection_preserves_warm_context,
)
from dataframe.accelerator.gpu.target import METAL_TARGET


def main() raises:
    test_mojo_gpu_device_service_metadata_copy_and_budget_are_context_free()
    print("PASS context-free metadata, copied runtime identity and query budget")
    comptime if METAL_TARGET:
        test_mojo_gpu_device_service_reuses_idle_context_and_discards_poison()
        test_mojo_gpu_direct_runtime_reports_final_service_drain()
        test_mojo_gpu_device_service_repeated_typed_collections_share_context()
        test_mojo_gpu_device_service_empty_and_preflight_leave_no_context()
        test_mojo_gpu_device_service_budget_rejection_preserves_warm_context()
        print("PASS public typed execution, overlapping leases, poison and budget")
    else:
        print("SKIP Metal service execution cases on a non-Metal target")
