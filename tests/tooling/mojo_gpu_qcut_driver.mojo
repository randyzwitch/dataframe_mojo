"""Focused resident custom-label QCUT public contracts."""
from tests.execution_modes.test_mojo_gpu_qcut import (
    test_mojo_gpu_qcut_custom_labels_layouts_closure_and_consumers,
    test_mojo_gpu_qcut_duplicate_labels_null_nan_and_control_order,
    test_mojo_gpu_qcut_exact_breakpoints_association_and_zero_permutation,
    test_mojo_gpu_qcut_eager_producers_scalar_siblings_budget_and_recovery,
)


def main() raises:
    test_mojo_gpu_qcut_custom_labels_layouts_closure_and_consumers()
    print("PASS QCUT labels/layouts/closure/consumers/empty controls")
    test_mojo_gpu_qcut_duplicate_labels_null_nan_and_control_order()
    print("PASS QCUT duplicate remapping and null/NaN/error controls")
    test_mojo_gpu_qcut_exact_breakpoints_association_and_zero_permutation()
    print("PASS QCUT exact interpolation and zero representative")
    test_mojo_gpu_qcut_eager_producers_scalar_siblings_budget_and_recovery()
    print("PASS QCUT eager ordered phases, budgets and recovery")
