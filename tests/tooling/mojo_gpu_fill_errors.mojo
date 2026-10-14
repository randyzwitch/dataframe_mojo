"""Isolated device regression for eager fill errors."""
from tests.execution_modes.test_mojo_gpu_fill_errors import (
    test_mojo_gpu_fill_observes_unselected_row_rhs_strict_cast,
    test_mojo_gpu_fill_outer_when_masks_rhs_fault_and_lax_cast_is_safe,
)


def main() raises:
    test_mojo_gpu_fill_observes_unselected_row_rhs_strict_cast()
    test_mojo_gpu_fill_outer_when_masks_rhs_fault_and_lax_cast_is_safe()
    print(
        "PASS eager fill faults, terminal preflight, WHEN masks and lax casts"
    )
