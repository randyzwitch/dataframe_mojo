"""Independent CPU IMPLODE child metadata and accumulator proof."""
from tests.public_contracts.test_implode_field_metadata import (
    test_implode_field_metadata_scalar_empty_batches_identity_and_computed,
    test_implode_field_metadata_grouped_over_full_and_worker_accumulators,
    test_implode_field_metadata_seed_does_not_evaluate_inactive_payloads,
)

from tests.execution_modes.test_mojo_gpu_resident_metadata import (
    test_mojo_gpu_resident_cardinality_metadata_bounds,
)


def main() raises:
    test_mojo_gpu_resident_cardinality_metadata_bounds()
    print("PASS metadata-only empty resident budget and positive mapping bound")
    test_implode_field_metadata_scalar_empty_batches_identity_and_computed()
    print("PASS CPU IMPLODE scalar/empty/batches/identity/computed metadata")
    test_implode_field_metadata_grouped_over_full_and_worker_accumulators()
    print("PASS CPU IMPLODE grouped/OVER/full/worker metadata")
    test_implode_field_metadata_seed_does_not_evaluate_inactive_payloads()
    print("PASS CPU IMPLODE inactive/empty checked payloads")
