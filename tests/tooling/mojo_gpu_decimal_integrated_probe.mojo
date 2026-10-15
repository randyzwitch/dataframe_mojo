"""Each focused case reports independently; any failure makes the driver fail."""
from tests.execution_modes.test_backend_metal_decimal_arithmetic import (
    test_decimal_unary_cpu_storage_widths_and_arithmetic_descriptors,
)
from tests.execution_modes.test_backend_metal_decimal_arithmetic import (
    test_native_metal_decimal_add_subtract_all_widths_scales_and_unary_metadata,
)
from tests.execution_modes.test_backend_metal_decimal_arithmetic import (
    test_native_metal_decimal_arithmetic_proven_consumers_aggregates_and_windows,
)
from tests.execution_modes.test_backend_metal_decimal_arithmetic import (
    test_native_metal_decimal_arithmetic_checked_errors_conditionals_budget_and_reuse,
)
from tests.execution_modes.test_backend_metal_decimal_arithmetic import (
    test_native_metal_decimal_add_subtract_literal_result_types_and_boundaries,
)
from tests.execution_modes.test_backend_metal_decimal_multiplicative import (
    test_decimal_multiplicative_capability_proofs_and_metadata,
)
from tests.execution_modes.test_backend_metal_decimal_multiplicative import (
    test_native_metal_decimal_multiplication_division_all_scales_widths_nulls_and_slices,
)
from tests.execution_modes.test_backend_metal_decimal_multiplicative import (
    test_native_metal_decimal_multiplicative_proven_resident_consumers,
)
from tests.execution_modes.test_backend_metal_decimal_multiplicative import (
    test_native_metal_decimal_multiplicative_errors_conditionals_zero_divisor_and_reuse,
)
from tests.execution_modes.test_backend_metal_decimal_multiplicative import (
    test_native_metal_decimal_exact_product_literals_layouts_and_precision,
)
from tests.execution_modes.test_backend_metal_decimal_multiplicative import (
    test_native_metal_decimal_product_width_and_quotient_literal_bits,
)
from tests.execution_modes.test_backend_metal_cumulative_windows import (
    test_native_metal_cumulative_checked_prefixes_remain_observable_under_head_and_when,
)
from tests.execution_modes.test_backend_metal_duration_decimals import (
    test_native_metal_duration_decimal_observed_prefix_errors_null_payloads_and_physical_extremes,
)


def main() raises:
    var failures = 0
    try:
        test_decimal_unary_cpu_storage_widths_and_arithmetic_descriptors()
        print(
            "PASS test_decimal_unary_cpu_storage_widths_and_arithmetic_descriptors"
        )
    except error:
        failures += 1
        print(
            "FAIL test_decimal_unary_cpu_storage_widths_and_arithmetic_descriptors:",
            String(error),
        )
    try:
        test_native_metal_decimal_add_subtract_all_widths_scales_and_unary_metadata()
        print(
            "PASS test_native_metal_decimal_add_subtract_all_widths_scales_and_unary_metadata"
        )
    except error:
        failures += 1
        print(
            "FAIL test_native_metal_decimal_add_subtract_all_widths_scales_and_unary_metadata:",
            String(error),
        )
    try:
        test_native_metal_decimal_arithmetic_proven_consumers_aggregates_and_windows()
        print(
            "PASS test_native_metal_decimal_arithmetic_proven_consumers_aggregates_and_windows"
        )
    except error:
        failures += 1
        print(
            "FAIL test_native_metal_decimal_arithmetic_proven_consumers_aggregates_and_windows:",
            String(error),
        )
    try:
        test_native_metal_decimal_arithmetic_checked_errors_conditionals_budget_and_reuse()
        print(
            "PASS test_native_metal_decimal_arithmetic_checked_errors_conditionals_budget_and_reuse"
        )
    except error:
        failures += 1
        print(
            "FAIL test_native_metal_decimal_arithmetic_checked_errors_conditionals_budget_and_reuse:",
            String(error),
        )
    try:
        test_native_metal_decimal_add_subtract_literal_result_types_and_boundaries()
        print(
            "PASS test_native_metal_decimal_add_subtract_literal_result_types_and_boundaries"
        )
    except error:
        failures += 1
        print(
            "FAIL test_native_metal_decimal_add_subtract_literal_result_types_and_boundaries:",
            String(error),
        )
    try:
        test_decimal_multiplicative_capability_proofs_and_metadata()
        print("PASS test_decimal_multiplicative_capability_proofs_and_metadata")
    except error:
        failures += 1
        print(
            "FAIL test_decimal_multiplicative_capability_proofs_and_metadata:",
            String(error),
        )
    try:
        test_native_metal_decimal_multiplication_division_all_scales_widths_nulls_and_slices()
        print(
            "PASS test_native_metal_decimal_multiplication_division_all_scales_widths_nulls_and_slices"
        )
    except error:
        failures += 1
        print(
            "FAIL test_native_metal_decimal_multiplication_division_all_scales_widths_nulls_and_slices:",
            String(error),
        )
    try:
        test_native_metal_decimal_multiplicative_proven_resident_consumers()
        print(
            "PASS test_native_metal_decimal_multiplicative_proven_resident_consumers"
        )
    except error:
        failures += 1
        print(
            "FAIL test_native_metal_decimal_multiplicative_proven_resident_consumers:",
            String(error),
        )
    try:
        test_native_metal_decimal_multiplicative_errors_conditionals_zero_divisor_and_reuse()
        print(
            "PASS test_native_metal_decimal_multiplicative_errors_conditionals_zero_divisor_and_reuse"
        )
    except error:
        failures += 1
        print(
            "FAIL test_native_metal_decimal_multiplicative_errors_conditionals_zero_divisor_and_reuse:",
            String(error),
        )
    try:
        test_native_metal_decimal_exact_product_literals_layouts_and_precision()
        print(
            "PASS test_native_metal_decimal_exact_product_literals_layouts_and_precision"
        )
    except error:
        failures += 1
        print(
            "FAIL test_native_metal_decimal_exact_product_literals_layouts_and_precision:",
            String(error),
        )
    try:
        test_native_metal_decimal_product_width_and_quotient_literal_bits()
        print(
            "PASS test_native_metal_decimal_product_width_and_quotient_literal_bits"
        )
    except error:
        failures += 1
        print(
            "FAIL test_native_metal_decimal_product_width_and_quotient_literal_bits:",
            String(error),
        )
    try:
        test_native_metal_cumulative_checked_prefixes_remain_observable_under_head_and_when()
        print(
            "PASS test_native_metal_cumulative_checked_prefixes_remain_observable_under_head_and_when"
        )
    except error:
        failures += 1
        print(
            "FAIL test_native_metal_cumulative_checked_prefixes_remain_observable_under_head_and_when:",
            String(error),
        )
    try:
        test_native_metal_duration_decimal_observed_prefix_errors_null_payloads_and_physical_extremes()
        print(
            "PASS test_native_metal_duration_decimal_observed_prefix_errors_null_payloads_and_physical_extremes"
        )
    except error:
        failures += 1
        print(
            "FAIL test_native_metal_duration_decimal_observed_prefix_errors_null_payloads_and_physical_extremes:",
            String(error),
        )
    print("RESULT gpu_focused: total", 13, "failures", failures)
    if failures:
        raise Error("Focused cases failed: " + String(failures))
