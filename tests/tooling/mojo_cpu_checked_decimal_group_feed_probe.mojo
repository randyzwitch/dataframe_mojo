"""Independent group/OVER checked Decimal event-order proof."""
from tests.public_contracts.test_checked_decimal_group_feed_order import (
    test_checked_decimal_group_and_over_distinct_operand_batches,
    test_checked_decimal_group_range_workers_before_partial_merges,
    test_checked_decimal_over_partition_job_order_precedes_source_fault_order,
    test_checked_decimal_multiple_occurrences_follow_original_ast_order,
)


def main() raises:
    test_checked_decimal_group_and_over_distinct_operand_batches()
    print("PASS 1/4 eager GROUP1024 vs OVER8192 feed boundaries")
    test_checked_decimal_group_range_workers_before_partial_merges()
    print("PASS 2/4 GROUP range4096 and worker-before-merge faults")
    test_checked_decimal_over_partition_job_order_precedes_source_fault_order()
    print("PASS 3/4 OVER whole-partition job fault order")
    test_checked_decimal_multiple_occurrences_follow_original_ast_order()
    print("PASS 4/4 multiple Decimal arithmetic occurrence order")
