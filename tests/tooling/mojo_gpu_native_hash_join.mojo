"""Focused device gate for the native Mojo GPU hash join."""
from tests.execution_modes.test_mojo_gpu_joins import (
    test_mojo_gpu_join_styles_typed_composite_keys_coalescing_and_empty_sides,
    test_mojo_gpu_join_nan_zero_subnormal_keys_and_exact_payloads,
    test_mojo_gpu_native_hash_join_float32_encodings_and_nulls,
    test_mojo_gpu_native_hash_join_null_equality_all_keyed_styles,
    test_mojo_gpu_native_hash_join_collision_fallback_preserves_null_equal,
    test_mojo_gpu_native_hash_join_empty_and_full_profile_launch_counts,
    test_mojo_gpu_native_hash_join_mixed_payload_budget_and_recovery,
    test_mojo_gpu_join_bounded_cross_fanout_and_exact_memory_budget,
)


def main() raises:
    test_mojo_gpu_join_styles_typed_composite_keys_coalescing_and_empty_sides()
    test_mojo_gpu_join_nan_zero_subnormal_keys_and_exact_payloads()
    test_mojo_gpu_native_hash_join_float32_encodings_and_nulls()
    test_mojo_gpu_native_hash_join_null_equality_all_keyed_styles()
    test_mojo_gpu_native_hash_join_collision_fallback_preserves_null_equal()
    test_mojo_gpu_native_hash_join_empty_and_full_profile_launch_counts()
    test_mojo_gpu_native_hash_join_mixed_payload_budget_and_recovery()
    test_mojo_gpu_join_bounded_cross_fanout_and_exact_memory_budget()
    print(
        "PASS native hash join styles, keys, nulls, fallback, profiles and budgets"
    )
