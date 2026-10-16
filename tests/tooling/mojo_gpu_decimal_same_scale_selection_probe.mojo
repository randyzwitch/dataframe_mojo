from tests.execution_modes.test_mojo_gpu_decimal_same_scale_selection import (
    test_decimal_same_scale_selection_literals_and_cast_refusals,
    test_mojo_gpu_decimal_same_scale_selection_nulls_masks_and_layouts,
    test_mojo_gpu_decimal_same_scale_fill_observes_both_children_under_when,
)


def main() raises:
    var failures = 0
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
    if failures:
        raise Error("Decimal same-scale selection validation failed")
