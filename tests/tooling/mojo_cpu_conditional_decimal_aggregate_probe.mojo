from tests.execution_modes.test_conditional_decimal_aggregate_planning import (
    test_decimal_when_inside_sum_keeps_original_operand_masks,
    test_decimal_sum_inside_when_keeps_conditional_scalar_guard,
    test_shared_reduction_retains_any_conditional_ancestor,
)


def main() raises:
    test_decimal_when_inside_sum_keeps_original_operand_masks()
    print("PASS internal WHEN retains checked operand masks")
    test_decimal_sum_inside_when_keeps_conditional_scalar_guard()
    print("PASS external WHEN keeps conservative scalar guard")
    test_shared_reduction_retains_any_conditional_ancestor()
    print("PASS shared reduction ancestry")
