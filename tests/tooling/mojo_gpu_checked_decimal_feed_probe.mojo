"""Focused eager checked Decimal SUM replay proof; CPU literals precede GPU."""
from tests.execution_modes.test_mojo_gpu_checked_decimal_feed import (
    test_checked_decimal_feed_metadata_retains_eager_boundary_and_signature,
    test_mojo_gpu_checked_decimal_sum_feed_success_layouts_nulls_and_consumers,
    test_mojo_gpu_checked_decimal_sum_feed_faults_before_head_and_empty_join,
)


def main() raises:
    test_checked_decimal_feed_metadata_retains_eager_boundary_and_signature()
    print("PASS 1/3 checked Decimal feed boundary metadata")
    test_mojo_gpu_checked_decimal_sum_feed_success_layouts_nulls_and_consumers()
    print("PASS 2/3 checked Decimal SUM layouts, nulls, joins and heads")
    test_mojo_gpu_checked_decimal_sum_feed_faults_before_head_and_empty_join()
    print(
        "PASS 3/3 checked Decimal operand/prefix/worker/merge fault precedence"
    )
