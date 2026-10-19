"""Focused gate for exact-capacity, bounded reduction scratch reuse."""
from tests.execution_modes.test_mojo_gpu_scratch_pool import (
    test_mojo_gpu_scratch_limits_are_metadata_only,
    test_mojo_gpu_scratch_reuses_capacity_but_charges_each_query,
    test_mojo_gpu_scratch_idle_caps_and_failure_evict_cached_buffers,
    test_mojo_gpu_overlapping_private_scope_cannot_consume_idle_scratch,
)


def main() raises:
    test_mojo_gpu_scratch_limits_are_metadata_only()
    test_mojo_gpu_scratch_reuses_capacity_but_charges_each_query()
    test_mojo_gpu_scratch_idle_caps_and_failure_evict_cached_buffers()
    test_mojo_gpu_overlapping_private_scope_cannot_consume_idle_scratch()
    print("PASS scratch reuse, exact query budgets, idle caps, poison and overlap")
