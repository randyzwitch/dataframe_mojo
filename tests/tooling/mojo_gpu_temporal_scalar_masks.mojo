"""Dedicated public scalar temporal mask regression."""
from tests.execution_modes.test_mojo_gpu_temporal_scalar_masks import (
    test_mojo_gpu_temporal_scalar_faults_any_active_and_empty_rows,
    test_mojo_gpu_temporal_safe_literal_cast_folding_preserves_dt_fields,
)


def main() raises:
    test_mojo_gpu_temporal_scalar_faults_any_active_and_empty_rows()
    test_mojo_gpu_temporal_safe_literal_cast_folding_preserves_dt_fields()
    print(
        "PASS public scalar temporal cast/arithmetic ANY-active and empty ROWS"
    )
