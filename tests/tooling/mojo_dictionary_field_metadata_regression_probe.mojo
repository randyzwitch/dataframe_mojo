"""Focused CPU normalization and original Metal field-owner regressions."""
from tests.public_contracts.test_dictionary_lookup_field_metadata import (
    test_dictionary_lookup_metadata_schema_cpu_literals_and_decimal_admission,
    test_dictionary_lookup_does_not_change_public_metadata_conflicts,
)
from tests.public_contracts.test_accelerator_field_metadata import (
    test_accelerator_field_metadata_group_keys_and_reductions_follow_cpu,
)
from tests.execution_modes.test_mojo_gpu_field_metadata import (
    test_mojo_gpu_field_metadata_dictionary_aliases_asof_and_empty_owners,
    test_mojo_gpu_field_metadata_rows_groups_empty_and_computed_outputs,
    test_mojo_gpu_field_metadata_asof_resident_aliases_and_public_export,
)


def main() raises:
    test_dictionary_lookup_metadata_schema_cpu_literals_and_decimal_admission()
    print("PASS 1/6 dictionary lookup metadata-only schema and CPU literals")
    test_dictionary_lookup_does_not_change_public_metadata_conflicts()
    print("PASS 2/6 public metadata conflicts retained")
    test_accelerator_field_metadata_group_keys_and_reductions_follow_cpu()
    print("PASS 3/6 public grouped field owners")
    test_mojo_gpu_field_metadata_dictionary_aliases_asof_and_empty_owners()
    print("PASS 4/6 original failing Metal dictionary/ASOF/empty owners")
    test_mojo_gpu_field_metadata_rows_groups_empty_and_computed_outputs()
    print("PASS 5/6 Metal ROW/GROUP/empty field owners")
    test_mojo_gpu_field_metadata_asof_resident_aliases_and_public_export()
    print("PASS 6/6 Metal resident ASOF field owners")
