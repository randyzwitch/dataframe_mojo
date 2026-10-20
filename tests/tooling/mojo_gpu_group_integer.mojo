"""Focused grouped integer partials and retained grouping semantics."""
from tests.execution_modes.test_mojo_gpu_grouping import (
    test_grouped_integer_metadata_certificate_accepts_normal_groupby_agg,
    test_mojo_gpu_grouping_mixed_keys_all_reductions_and_nullable_layouts,
    test_mojo_gpu_grouping_key_only_stability_distinct_groups_and_compositions,
    test_mojo_gpu_grouping_exact_wide_sum_local_arg_positions_and_overflow,
    test_mojo_gpu_grouping_integer_partials_shared_merge_and_fallback,
)


def main() raises:
    test_grouped_integer_metadata_certificate_accepts_normal_groupby_agg()
    print("PASS normal grouped plan certificate")
    test_mojo_gpu_grouping_mixed_keys_all_reductions_and_nullable_layouts()
    print("PASS mixed keys, reducers and nullable layouts")
    test_mojo_gpu_grouping_key_only_stability_distinct_groups_and_compositions()
    print("PASS key representatives and compositions")
    test_mojo_gpu_grouping_exact_wide_sum_local_arg_positions_and_overflow()
    print("PASS exact wide totals, argument positions and overflow")
    test_mojo_gpu_grouping_integer_partials_shared_merge_and_fallback()
    print("PASS parallel grouped integers, budgets and generic consumers")
