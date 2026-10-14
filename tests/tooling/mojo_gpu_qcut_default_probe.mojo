"""Public default QCUT dictionary/control and downstream consumer proof."""
from tests.execution_modes.test_mojo_gpu_qcut_default import (
    test_mojo_gpu_qcut_default_literal_labels_nulls_closure_and_layouts,
    test_mojo_gpu_qcut_default_downstream_filter_join_and_domain_union,
    test_mojo_gpu_qcut_default_breakpoints_over_keys_and_exact_controls,
    test_mojo_gpu_qcut_default_duplicates_signed_zero_and_ordered_failures,
)


def main() raises:
    test_mojo_gpu_qcut_default_literal_labels_nulls_closure_and_layouts()
    print("PASS default QCUT labels/layouts/null/NaN/closure")
    test_mojo_gpu_qcut_default_downstream_filter_join_and_domain_union()
    print("PASS default QCUT filters/joins/two dynamic domains")
    test_mojo_gpu_qcut_default_breakpoints_over_keys_and_exact_controls()
    print("PASS default QCUT terminal Struct/OVER/control traffic/budget")
    test_mojo_gpu_qcut_default_duplicates_signed_zero_and_ordered_failures()
    print("PASS default QCUT duplicate/zero/error order/recovery")
