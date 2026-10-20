"""Focused regression gate for equality hash grouping."""
from tests.execution_modes.test_mojo_gpu_hash_grouping import (
    test_mojo_gpu_hash_grouping_boundaries_and_integer_states,
    test_mojo_gpu_hash_grouping_composite_null_nan_keys_and_source_order,
    test_mojo_gpu_hash_grouping_probe_exhaustion_and_checked_sum,
)
from tests.execution_modes.test_mojo_gpu_ordering import (
    test_mojo_gpu_ordering_integer_layouts_nulls_mixed_keys_and_stability,
    test_mojo_gpu_ordering_float_encodings_nan_zero_subnormal_and_nulls,
    test_mojo_gpu_ordering_compositions_key_errors_budget_and_recovery,
)
from tests.execution_modes.test_mojo_gpu_grouping import (
    test_mojo_gpu_grouping_mixed_keys_all_reductions_and_nullable_layouts,
    test_mojo_gpu_grouping_key_only_stability_distinct_groups_and_compositions,
    test_mojo_gpu_grouping_exact_wide_sum_local_arg_positions_and_overflow,
)


def main() raises:
    test_mojo_gpu_hash_grouping_boundaries_and_integer_states()
    test_mojo_gpu_hash_grouping_composite_null_nan_keys_and_source_order()
    test_mojo_gpu_hash_grouping_probe_exhaustion_and_checked_sum()
    print("PASS hash admission boundaries, composite keys, states and fallback")
    test_mojo_gpu_ordering_integer_layouts_nulls_mixed_keys_and_stability()
    test_mojo_gpu_ordering_float_encodings_nan_zero_subnormal_and_nulls()
    test_mojo_gpu_ordering_compositions_key_errors_budget_and_recovery()
    print("PASS ordering types, layouts, compositions, errors and budgets")
    test_mojo_gpu_grouping_mixed_keys_all_reductions_and_nullable_layouts()
    test_mojo_gpu_grouping_key_only_stability_distinct_groups_and_compositions()
    test_mojo_gpu_grouping_exact_wide_sum_local_arg_positions_and_overflow()
    print("PASS grouped reducers, layouts, order and checked arithmetic")
