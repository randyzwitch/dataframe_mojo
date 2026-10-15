"""Focused checked-producer observation proof; requires Metal."""
from tests.execution_modes.test_mojo_gpu_observed_integer_producers import (
    test_mojo_gpu_observed_integer_producer_success_and_nested_groups,
    test_mojo_gpu_observed_unused_integer_overflow_before_consumers,
    test_mojo_gpu_observed_first_group_overflow_before_regroup_and_head,
    test_observed_integer_boundary_signatures_and_original_order,
)


def main() raises:
    test_observed_integer_boundary_signatures_and_original_order()
    print("PASS 1/4 signature/order guards", flush=True)
    test_mojo_gpu_observed_integer_producer_success_and_nested_groups()
    print("PASS 2/4 observed producer success", flush=True)
    test_mojo_gpu_observed_unused_integer_overflow_before_consumers()
    print("PASS 3/4 unused producer faults", flush=True)
    test_mojo_gpu_observed_first_group_overflow_before_regroup_and_head()
    print("PASS 4/4 first group faults", flush=True)
