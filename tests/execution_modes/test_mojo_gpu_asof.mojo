"""Independent direct-kernel as-of witnesses before public DAG integration."""
from tests.support.mojo_gpu_asof_reference import (
    directions_null_runs_and_integer_distances,
    ieee_comparisons_and_tolerance,
    grouped_keys_sortedness_and_error_observation,
)


def test_mojo_gpu_asof_raw_kernel_directions_null_duplicate_runs_and_exact_integer_distance() raises:
    directions_null_runs_and_integer_distances()


def test_mojo_gpu_asof_raw_kernel_ieee_nonfinite_fractional_tolerance_and_cpu_cursor_policy() raises:
    ieee_comparisons_and_tolerance()


def test_mojo_gpu_asof_raw_kernel_grouped_typed_keys_complete_sortedness_and_error_observation() raises:
    grouped_keys_sortedness_and_error_observation()
