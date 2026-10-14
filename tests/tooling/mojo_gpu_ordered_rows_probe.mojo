"""Independent direct-ROWS ordered errors on actual public CPU/GPU paths."""
from tests.public_contracts.test_ordered_rows_metadata import (
    test_ordered_rows_rank_resources_calibration_and_file_provenance,
)
from tests.execution_modes.test_mojo_gpu_ordered_errors import (
    test_mojo_gpu_ordered_rows_cross_row_batch_projection_and_workers,
    test_mojo_gpu_ordered_rows_masks_nulls_empty_and_mixed_numeric_faults,
    test_mojo_gpu_ordered_rows_profile_budget_estimate_and_provenance,
)


def main() raises:
    test_ordered_rows_rank_resources_calibration_and_file_provenance()
    print("PASS ordered ROWS metadata/resources/provenance", flush=True)
    test_mojo_gpu_ordered_rows_cross_row_batch_projection_and_workers()
    print(
        "PASS ordered ROWS projection/batch/AST and CPU workers1/2/4/8",
        flush=True,
    )
    test_mojo_gpu_ordered_rows_masks_nulls_empty_and_mixed_numeric_faults()
    print(
        "PASS ordered ROWS selected masks/shared occurrences/empty", flush=True
    )
    test_mojo_gpu_ordered_rows_profile_budget_estimate_and_provenance()
    print(
        "PASS ordered ROWS budget/estimate/explain and source guards",
        flush=True,
    )
