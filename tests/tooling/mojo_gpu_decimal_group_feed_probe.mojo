"""Narrow Decimal GROUP: original metadata, literal values and precision faults."""
from tests.execution_modes.test_mojo_gpu_decimal_group_feed import (
    test_decimal_group_feed_certificate_preserves_whole_original_agg,
    test_mojo_gpu_decimal_group_feed_literals_nulls_layouts_and_masks,
    test_mojo_gpu_decimal_group_feed_stream_workers_and_precision_faults,
    test_mojo_gpu_decimal_group_feed_budget_precedes_device,
)
from tests.public_contracts.test_checked_decimal_group_feed_order import (
    test_checked_decimal_over_partition_job_order_precedes_source_fault_order,
)


def main() raises:
    test_checked_decimal_over_partition_job_order_precedes_source_fault_order()
    print("PASS 1/5 corrected first-seen OVER CPU oracle")
    test_decimal_group_feed_certificate_preserves_whole_original_agg()
    print("PASS 2/5 whole AGG certificate and refusals")
    test_mojo_gpu_decimal_group_feed_literals_nulls_layouts_and_masks()
    print("PASS 3/5 narrow Decimal GROUP literals, layouts, masks and nulls")
    test_mojo_gpu_decimal_group_feed_stream_workers_and_precision_faults()
    print("PASS 4/5 streaming/workers/batches and commuting precision faults")
    test_mojo_gpu_decimal_group_feed_budget_precedes_device()
    print("PASS 5/5 exact measured budget and budget-before-device")
