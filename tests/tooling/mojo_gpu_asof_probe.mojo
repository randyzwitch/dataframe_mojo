"""Executable direct as-of kernel probe; reusable cases live outside main."""
from tests.support.mojo_gpu_asof_reference import (
    directions_null_runs_and_integer_distances,
    ieee_comparisons_and_tolerance,
    grouped_keys_sortedness_and_error_observation,
)


def main() raises:
    directions_null_runs_and_integer_distances()
    ieee_comparisons_and_tolerance()
    grouped_keys_sortedness_and_error_observation()
    print(
        "PASS fixed-width as-of directions, IEEE distances, null runs, grouped keys and complete sortedness"
    )
