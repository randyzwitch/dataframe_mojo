"""Each focused case reports independently; any failure makes the driver fail."""
from tests.public_contracts.test_decimal_exact_product import (
    test_decimal_exact_product_layout_and_nulls,
)
from tests.public_contracts.test_decimal_exact_product import (
    test_decimal_exact_product_precision_and_mask,
)
from tests.public_contracts.test_decimal_exact_product import (
    test_decimal_product_physical_values_check_result_before_narrowing,
)
from tests.public_contracts.test_decimal_add_physical_bounds import (
    test_decimal_add_subtract_dense_and_nullable_precision_and_alignment,
)
from tests.public_contracts.test_decimal_add_physical_bounds import (
    test_decimal_add_subtract_dense_precision_and_mask_literals,
)


def main() raises:
    var failures = 0
    try:
        test_decimal_exact_product_layout_and_nulls()
        print("PASS test_decimal_exact_product_layout_and_nulls")
    except error:
        failures += 1
        print(
            "FAIL test_decimal_exact_product_layout_and_nulls:", String(error)
        )
    try:
        test_decimal_exact_product_precision_and_mask()
        print("PASS test_decimal_exact_product_precision_and_mask")
    except error:
        failures += 1
        print(
            "FAIL test_decimal_exact_product_precision_and_mask:", String(error)
        )
    try:
        test_decimal_product_physical_values_check_result_before_narrowing()
        print(
            "PASS test_decimal_product_physical_values_check_result_before_narrowing"
        )
    except error:
        failures += 1
        print(
            "FAIL test_decimal_product_physical_values_check_result_before_narrowing:",
            String(error),
        )
    try:
        test_decimal_add_subtract_dense_and_nullable_precision_and_alignment()
        print(
            "PASS test_decimal_add_subtract_dense_and_nullable_precision_and_alignment"
        )
    except error:
        failures += 1
        print(
            "FAIL test_decimal_add_subtract_dense_and_nullable_precision_and_alignment:",
            String(error),
        )
    try:
        test_decimal_add_subtract_dense_precision_and_mask_literals()
        print(
            "PASS test_decimal_add_subtract_dense_precision_and_mask_literals"
        )
    except error:
        failures += 1
        print(
            "FAIL test_decimal_add_subtract_dense_precision_and_mask_literals:",
            String(error),
        )
    print("RESULT cpu_arithmetic: total", 5, "failures", failures)
    if failures:
        raise Error("Focused cases failed: " + String(failures))
