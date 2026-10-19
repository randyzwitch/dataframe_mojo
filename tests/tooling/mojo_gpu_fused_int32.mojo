"""Focused device gate for native fused integer reduction."""
from tests.execution_modes.test_mojo_gpu_fused_int32 import (
    test_fused_int32_sum_nulls_layouts_empty_and_min_count,
    test_fused_int32_sum_bounded_partials_exact_budget_and_empty_device,
    test_fused_int32_sum_preserves_checked_arithmetic_and_filter_order,
    test_fused_int32_sum_independent_signed_endpoints_and_cancellation,
)


def main() raises:
    test_fused_int32_sum_nulls_layouts_empty_and_min_count()
    print("PASS fused sum layouts/nulls/empty/min_count")
    test_fused_int32_sum_bounded_partials_exact_budget_and_empty_device()
    print("PASS fused sum bounded partials and budgets")
    test_fused_int32_sum_independent_signed_endpoints_and_cancellation()
    print("PASS fused sum independent endpoints and cross-partial cancellation")
    test_fused_int32_sum_preserves_checked_arithmetic_and_filter_order()
    print("PASS fused sum checked arithmetic and filter ordering")
