"""CPU feed-order oracle; run before the isolated device kernel probe."""
from tests.public_contracts.test_grouped_feed_error_order import (
    test_grouped_feed_input_batch_precedes_decimal_update_same_batch_only,
    test_grouped_feed_checked_ast_and_integer_sum_finish_order,
    test_grouped_feed_worker_errors_precede_worker_ordered_merges,
)


def main() raises:
    test_grouped_feed_input_batch_precedes_decimal_update_same_batch_only()
    print("PASS CPU input batch versus Decimal128 state update")
    test_grouped_feed_checked_ast_and_integer_sum_finish_order()
    test_grouped_feed_worker_errors_precede_worker_ordered_merges()
    print("PASS CPU checked AST and integer SUM finish order")
