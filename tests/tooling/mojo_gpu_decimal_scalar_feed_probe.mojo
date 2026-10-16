from tests.execution_modes.test_mojo_gpu_decimal_scalar_feed import (
    test_decimal_scalar_feed_original_ast_prefix_and_resources,
    test_mojo_gpu_decimal_scalar_feed_ratio_money_masks_and_layouts,
    test_mojo_gpu_decimal_scalar_feed_selected_faults_dead_arms_and_min_count,
    test_mojo_gpu_decimal_scalar_feed_null_empty_budget_and_recovery,
)


def main() raises:
    test_decimal_scalar_feed_original_ast_prefix_and_resources()
    print("PASS scalar feed original AST, prefix and bounded resources")
    test_mojo_gpu_decimal_scalar_feed_ratio_money_masks_and_layouts()
    print("PASS scalar feed ratio80, canonical money, masks and layouts")
    test_mojo_gpu_decimal_scalar_feed_selected_faults_dead_arms_and_min_count()
    print("PASS scalar feed selected faults, dead arms and min_count")
    test_mojo_gpu_decimal_scalar_feed_null_empty_budget_and_recovery()
    print("PASS scalar feed empty, budget and recovery")
