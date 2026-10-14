"""Focused exact order-statistic parity contracts."""
from tests.execution_modes.test_mojo_gpu_quantile_zero import (
    test_mojo_gpu_quantile_zero_scalar_exact_ranks_and_reciprocal,
    test_mojo_gpu_quantile_zero_quickselect_fallback_and_nonfinite,
    test_mojo_gpu_quantile_zero_grouped_disjoint_scratch_and_budget,
    test_mojo_gpu_quantile_linear_separate_rounding_literal_anchors,
)


def main() raises:
    test_mojo_gpu_quantile_zero_scalar_exact_ranks_and_reciprocal()
    print("PASS scalar signed zeros and resident reciprocal")
    test_mojo_gpu_quantile_zero_quickselect_fallback_and_nonfinite()
    print("PASS quickselect/fallback and null/NaN zero representatives")
    test_mojo_gpu_quantile_zero_grouped_disjoint_scratch_and_budget()
    print("PASS grouped disjoint scratch and exact budgets")
    test_mojo_gpu_quantile_linear_separate_rounding_literal_anchors()
    print("PASS independent separate-rounding scalar/grouped anchors")
