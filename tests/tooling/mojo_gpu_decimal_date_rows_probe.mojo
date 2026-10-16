"""Physical Apple probe of maintained Date.YEAR/Decimal ROW witnesses."""
from tests.execution_modes.test_mojo_gpu_decimal_date_rows import (
    test_decimal_date_rows_certificate_preserves_original_prefix_and_states,
    test_mojo_gpu_decimal_date_rows_literals_extremes_nulls_and_layouts,
    test_mojo_gpu_decimal_date_rows_precision_before_group_and_join,
)


def main() raises:
    test_decimal_date_rows_certificate_preserves_original_prefix_and_states()
    print("PASS decimal Date.YEAR original prefix and state certificates")
    test_mojo_gpu_decimal_date_rows_literals_extremes_nulls_and_layouts()
    print(
        "PASS decimal Date.YEAR independent literals, extremes, nulls and layouts"
    )
    test_mojo_gpu_decimal_date_rows_precision_before_group_and_join()
    print(
        "PASS decimal Date.YEAR precision before GROUP/JOIN across CPU streaming geometry"
    )
