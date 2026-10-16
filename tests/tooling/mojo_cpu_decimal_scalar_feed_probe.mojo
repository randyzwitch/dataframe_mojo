from tests.execution_modes.test_decimal_scalar_feed_metadata import (
    test_decimal_scalar_feed_reuses_physical_bounds_for_multiple_sites,
    test_decimal_scalar_feed_rejects_foreign_faults_and_wide_raw_sources,
    test_decimal_money_literal_and_arithmetic_bound_metadata,
    test_conditional_decimal_sum_min_count_preserves_operand_faults,
)


def main() raises:
    test_decimal_scalar_feed_reuses_physical_bounds_for_multiple_sites()
    print("PASS scalar precision-only operand/state proof")
    test_decimal_scalar_feed_rejects_foreign_faults_and_wide_raw_sources()
    print("PASS foreign fault and wide raw source rejection")
    test_decimal_money_literal_and_arithmetic_bound_metadata()
    print("PASS default-width money literal, arithmetic metadata, ratio80")
    test_conditional_decimal_sum_min_count_preserves_operand_faults()
    print("PASS min_count null state retains selected operand precision fault")
