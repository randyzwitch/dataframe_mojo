"""Focused radix and tiled-comparison ordering validation."""
from tests.execution_modes.test_mojo_gpu_ordering import (
    test_mojo_gpu_ordering_integer_layouts_nulls_mixed_keys_and_stability,
    test_mojo_gpu_ordering_float_encodings_nan_zero_subnormal_and_nulls,
    test_mojo_gpu_ordering_compositions_key_errors_budget_and_recovery,
    test_mojo_gpu_ordering_tiled_merge_boundaries_ties_nulls_and_float_encodings,
)


def main() raises:
    test_mojo_gpu_ordering_integer_layouts_nulls_mixed_keys_and_stability()
    test_mojo_gpu_ordering_float_encodings_nan_zero_subnormal_and_nulls()
    test_mojo_gpu_ordering_compositions_key_errors_budget_and_recovery()
    test_mojo_gpu_ordering_tiled_merge_boundaries_ties_nulls_and_float_encodings()
    print("PASS radix and tiled ordering, stability, errors and budgets")
