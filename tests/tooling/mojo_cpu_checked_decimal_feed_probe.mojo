"""CPU-only independent checked Decimal feed-event oracle."""
from tests.public_contracts.test_checked_decimal_feed_order import (
    test_checked_decimal_sum_operand_batches_precede_prefix_observation,
)


def main() raises:
    test_checked_decimal_sum_operand_batches_precede_prefix_observation()
    print("PASS checked Decimal SUM operand-batch and prefix fault order")
