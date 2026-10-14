"""Independent both-empty public join contracts; no hardware is touched."""
from tests.execution_modes.test_mojo_gpu_empty_joins import (
    test_mojo_gpu_both_empty_join_styles_schema_zero_budget_without_device,
    test_mojo_gpu_both_empty_join_invalid_metadata_precedes_device_setup,
)


def main() raises:
    test_mojo_gpu_both_empty_join_styles_schema_zero_budget_without_device()
    print("PASS both-empty join styles, typed schemas, zero budget, no device")
    test_mojo_gpu_both_empty_join_invalid_metadata_precedes_device_setup()
    print("PASS both-empty join metadata errors precede device setup")
