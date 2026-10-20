"""Focused device gate for the native join gather foundation."""

from tests.execution_modes.test_gpu_heterogeneous_native_table import (
    test_native_join_gather_bounds_errors_budget_and_empty,
    test_native_join_gather_coalesces_device_maps_with_one_drain,
    test_native_join_gather_small_payloads_use_safe_unused_error_buffers,
    test_native_join_gather_float64_bits_and_coalescing_precedence,
)


def main() raises:
    test_native_join_gather_coalesces_device_maps_with_one_drain()
    test_native_join_gather_small_payloads_use_safe_unused_error_buffers()
    test_native_join_gather_float64_bits_and_coalescing_precedence()
    test_native_join_gather_bounds_errors_budget_and_empty()
    print("PASS native join pair-map gather, coalescing, errors and budgets")
