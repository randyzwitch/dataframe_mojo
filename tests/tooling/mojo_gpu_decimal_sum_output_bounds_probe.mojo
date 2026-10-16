"""Standalone independent catches for successful-SUM consumer receipts."""
from tests.execution_modes.test_mojo_gpu_decimal_sum_output_bounds import (
    test_sum_output_bound_raw_math_and_original_slots,
    test_mojo_gpu_sum_output_bounds_scaled_literal_nulls_layouts,
    test_mojo_gpu_sum_output_bounds_preserve_producer_fault_and_budget,
    test_sum_output_bounds_refuse_arbitrary_wide_columns_and_scale18_products,
)


def main() raises:
    var failures = 0
    try:
        test_sum_output_bound_raw_math_and_original_slots()
        print("PASS 1/4 raw bounds and original slots")
    except e:
        failures += 1
        print("FAIL 1/4 raw bounds and original slots:", e)
    try:
        test_mojo_gpu_sum_output_bounds_scaled_literal_nulls_layouts()
        print("PASS 2/4 scaled literal nulls layouts")
    except e:
        failures += 1
        print("FAIL 2/4 scaled literal nulls layouts:", e)
    try:
        test_mojo_gpu_sum_output_bounds_preserve_producer_fault_and_budget()
        print("PASS 3/4 producer fault and exact budget")
    except e:
        failures += 1
        print("FAIL 3/4 producer fault and exact budget:", e)
    try:
        test_sum_output_bounds_refuse_arbitrary_wide_columns_and_scale18_products()
        print("PASS 4/4 arbitrary wide and unproved scale18 refused")
    except e:
        failures += 1
        print("FAIL 4/4 arbitrary wide and unproved scale18 refused:", e)
    if failures:
        raise Error("successful SUM output bounds probe failed")
