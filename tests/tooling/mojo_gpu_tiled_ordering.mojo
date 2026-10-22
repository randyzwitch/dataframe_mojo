"""Focused real-device gate for bounded tiled stable ordering."""
from tests.execution_modes.test_mojo_gpu_ordering import (
    test_mojo_gpu_ordering_tiled_merge_boundaries_ties_nulls_and_float_encodings,
)


def main() raises:
    test_mojo_gpu_ordering_tiled_merge_boundaries_ties_nulls_and_float_encodings()
    print("PASS tiled merge boundaries, stable ties, nulls and float encodings")
