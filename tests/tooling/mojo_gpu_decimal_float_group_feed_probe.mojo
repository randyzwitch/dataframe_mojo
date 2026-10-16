from tests.execution_modes.test_mojo_gpu_decimal_float_group_feed import (
    test_decimal_float_group_feed_original_operand_and_state_proof,
    test_mojo_gpu_decimal_float_group_feed_literals_views_nulls_and_states,
    test_mojo_gpu_decimal_float_group_feed_precision_faults_and_ieee_division,
)


def main() raises:
    var failures = 0
    try:
        test_decimal_float_group_feed_original_operand_and_state_proof()
        print(
            "PASS original Float64 operand/state proof and foreign CAST refusal"
        )
    except error:
        failures += 1
        print("FAIL original Float64 operand/state proof", String(error))
    try:
        test_mojo_gpu_decimal_float_group_feed_literals_views_nulls_and_states()
        print("PASS independent Float64/Decimal states, views and nulls")
    except error:
        failures += 1
        print("FAIL Float64/Decimal state literals", String(error))
    try:
        test_mojo_gpu_decimal_float_group_feed_precision_faults_and_ieee_division()
        print("PASS precision faults before min_count/head and IEEE division")
    except error:
        failures += 1
        print("FAIL precision fault and IEEE division", String(error))
    if failures:
        raise Error("Decimal Float64 grouped feed validation failed")
