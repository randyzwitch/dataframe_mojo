from tests.execution_modes.test_mojo_gpu_decimal_float_group_feed import (
    test_decimal_float_group_feed_original_operand_and_state_proof,
    test_mojo_gpu_decimal_float_group_feed_literals_views_nulls_and_states,
    test_mojo_gpu_decimal_float_group_feed_precision_faults_and_ieee_division,
)

from tests.execution_modes.test_mojo_gpu_decimal_same_scale_selection import (
    test_decimal_same_scale_selection_literals_and_cast_refusals,
    test_mojo_gpu_decimal_same_scale_selection_nulls_masks_and_layouts,
    test_mojo_gpu_decimal_same_scale_fill_observes_both_children_under_when,
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
    try:
        test_decimal_same_scale_selection_literals_and_cast_refusals()
        print("PASS exact-scale literal proof and unsafe cast refusals")
    except error:
        failures += 1
        print("FAIL exact-scale literal metadata proof", String(error))
    try:
        test_mojo_gpu_decimal_same_scale_selection_nulls_masks_and_layouts()
        print("PASS independent net totals3900/5300, masks/nulls/views/chunks")
    except error:
        failures += 1
        print("FAIL exact-scale literal values", String(error))
    try:
        test_mojo_gpu_decimal_same_scale_fill_observes_both_children_under_when()
        print("PASS eager FILL_NULL children under original WHEN mask")
    except error:
        failures += 1
        print("FAIL FILL_NULL child error/mask contract", String(error))
    print("Decimal q4/q40 groups:", 6 - failures, "PASS", failures, "FAIL")
    if failures:
        raise Error("Decimal q4/q40 validation failed")
