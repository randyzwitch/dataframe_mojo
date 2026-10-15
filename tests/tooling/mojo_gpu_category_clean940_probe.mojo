"""Focused existing category contracts plus PR726 metadata/budget guards."""
from std.testing import assert_equal
from tests.execution_modes.test_mojo_gpu_categorical_transport import (
    test_mojo_gpu_categorical_original_rows_filter_sort_group_and_over,
    test_mojo_gpu_categorical_join_duplicate_codes_and_dictionary_union,
    test_mojo_gpu_categorical_asof_temporary_by_union_preserves_output_owner,
    test_mojo_gpu_categorical_dictionary_budget_owners,
    test_mojo_gpu_categorical_mixed_join_control_readback_and_budget,
    test_mojo_gpu_coded_string_group_publication_drops_original_code_identity,
)
from tests.public_contracts.test_dictionary_lookup_field_metadata import (
    test_dictionary_lookup_metadata_schema_cpu_literals_and_decimal_admission,
    test_dictionary_lookup_does_not_change_public_metadata_conflicts,
)
from tests.execution_modes.test_mojo_gpu_field_metadata import (
    test_mojo_gpu_field_metadata_dictionary_aliases_asof_and_empty_owners,
    test_mojo_gpu_field_metadata_rows_groups_empty_and_computed_outputs,
)
from tests.support.source_explode_reference import (
    simultaneous_scratch_and_output_literal_budget,
)
from tests.execution_modes.test_mojo_gpu_explode import (
    test_mojo_gpu_explode_empty_metadata_guards_exact_budget_observer_and_recovery,
)


def main() raises:
    var failures = 0
    try:
        test_mojo_gpu_categorical_original_rows_filter_sort_group_and_over()
        print(
            "PASS 1/12 test_mojo_gpu_categorical_original_rows_filter_sort_group_and_over",
            flush=True,
        )
    except error:
        failures += 1
        print(
            "FAIL 1/12 test_mojo_gpu_categorical_original_rows_filter_sort_group_and_over",
            String(error),
            flush=True,
        )
    try:
        test_mojo_gpu_categorical_join_duplicate_codes_and_dictionary_union()
        print(
            "PASS 2/12 test_mojo_gpu_categorical_join_duplicate_codes_and_dictionary_union",
            flush=True,
        )
    except error:
        failures += 1
        print(
            "FAIL 2/12 test_mojo_gpu_categorical_join_duplicate_codes_and_dictionary_union",
            String(error),
            flush=True,
        )
    try:
        test_mojo_gpu_categorical_asof_temporary_by_union_preserves_output_owner()
        print(
            "PASS 3/12 test_mojo_gpu_categorical_asof_temporary_by_union_preserves_output_owner",
            flush=True,
        )
    except error:
        failures += 1
        print(
            "FAIL 3/12 test_mojo_gpu_categorical_asof_temporary_by_union_preserves_output_owner",
            String(error),
            flush=True,
        )
    try:
        test_mojo_gpu_categorical_dictionary_budget_owners()
        print(
            "PASS 4/12 test_mojo_gpu_categorical_dictionary_budget_owners",
            flush=True,
        )
    except error:
        failures += 1
        print(
            "FAIL 4/12 test_mojo_gpu_categorical_dictionary_budget_owners",
            String(error),
            flush=True,
        )
    try:
        test_mojo_gpu_categorical_mixed_join_control_readback_and_budget()
        print(
            "PASS 5/12 test_mojo_gpu_categorical_mixed_join_control_readback_and_budget",
            flush=True,
        )
    except error:
        failures += 1
        print(
            "FAIL 5/12 test_mojo_gpu_categorical_mixed_join_control_readback_and_budget",
            String(error),
            flush=True,
        )
    try:
        test_mojo_gpu_coded_string_group_publication_drops_original_code_identity()
        print(
            "PASS 6/12 test_mojo_gpu_coded_string_group_publication_drops_original_code_identity",
            flush=True,
        )
    except error:
        failures += 1
        print(
            "FAIL 6/12 test_mojo_gpu_coded_string_group_publication_drops_original_code_identity",
            String(error),
            flush=True,
        )
    try:
        test_dictionary_lookup_metadata_schema_cpu_literals_and_decimal_admission()
        print(
            "PASS 7/12 test_dictionary_lookup_metadata_schema_cpu_literals_and_decimal_admission",
            flush=True,
        )
    except error:
        failures += 1
        print(
            "FAIL 7/12 test_dictionary_lookup_metadata_schema_cpu_literals_and_decimal_admission",
            String(error),
            flush=True,
        )
    try:
        test_dictionary_lookup_does_not_change_public_metadata_conflicts()
        print(
            "PASS 8/12 test_dictionary_lookup_does_not_change_public_metadata_conflicts",
            flush=True,
        )
    except error:
        failures += 1
        print(
            "FAIL 8/12 test_dictionary_lookup_does_not_change_public_metadata_conflicts",
            String(error),
            flush=True,
        )
    try:
        test_mojo_gpu_field_metadata_dictionary_aliases_asof_and_empty_owners()
        print(
            "PASS 9/12 test_mojo_gpu_field_metadata_dictionary_aliases_asof_and_empty_owners",
            flush=True,
        )
    except error:
        failures += 1
        print(
            "FAIL 9/12 test_mojo_gpu_field_metadata_dictionary_aliases_asof_and_empty_owners",
            String(error),
            flush=True,
        )
    try:
        test_mojo_gpu_field_metadata_rows_groups_empty_and_computed_outputs()
        print(
            "PASS 10/12 test_mojo_gpu_field_metadata_rows_groups_empty_and_computed_outputs",
            flush=True,
        )
    except error:
        failures += 1
        print(
            "FAIL 10/12 test_mojo_gpu_field_metadata_rows_groups_empty_and_computed_outputs",
            String(error),
            flush=True,
        )
    try:
        simultaneous_scratch_and_output_literal_budget()
        print(
            "PASS 11/12 simultaneous_scratch_and_output_literal_budget",
            flush=True,
        )
    except error:
        failures += 1
        print(
            "FAIL 11/12 simultaneous_scratch_and_output_literal_budget",
            String(error),
            flush=True,
        )
    try:
        test_mojo_gpu_explode_empty_metadata_guards_exact_budget_observer_and_recovery()
        print(
            "PASS 12/12 test_mojo_gpu_explode_empty_metadata_guards_exact_budget_observer_and_recovery",
            flush=True,
        )
    except error:
        failures += 1
        print(
            "FAIL 12/12 test_mojo_gpu_explode_empty_metadata_guards_exact_budget_observer_and_recovery",
            String(error),
            flush=True,
        )
    assert_equal(failures, 0)
