"""Focused public semantics for bounded native wide integer SUMs."""
from tests.execution_modes.test_mojo_gpu_wide_integer_sum import (
    test_wide_integer_sum_exact_cancellation_across_partials_and_outputs,
    test_wide_integer_sum_unsigned_endpoints_nulls_empty_and_min_count,
    test_wide_integer_sum_final_overflow_and_late_checked_row_faults,
    test_wide_integer_sum_bounded_state_budget_and_reuse,
)


def main() raises:
    test_wide_integer_sum_exact_cancellation_across_partials_and_outputs()
    print("PASS wide integer cancellation, layouts and scalar outputs")
    test_wide_integer_sum_unsigned_endpoints_nulls_empty_and_min_count()
    print("PASS unsigned endpoints, nulls, empty and min_count")
    test_wide_integer_sum_final_overflow_and_late_checked_row_faults()
    print("PASS final overflow, late row faults and distinct expressions")
    test_wide_integer_sum_bounded_state_budget_and_reuse()
    print("PASS bounded wide state, exact budgets and scratch reuse")
