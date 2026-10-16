"""Uncompiled validation integration: six category and three grouped probes."""
from tests.execution_modes.test_mojo_gpu_categorical_transport import (
    test_mojo_gpu_categorical_original_rows_filter_sort_group_and_over,
    test_mojo_gpu_categorical_join_duplicate_codes_and_dictionary_union,
    test_mojo_gpu_categorical_asof_temporary_by_union_preserves_output_owner,
    test_mojo_gpu_categorical_dictionary_budget_owners,
    test_mojo_gpu_categorical_mixed_join_control_readback_and_budget,
    test_mojo_gpu_coded_string_group_publication_drops_original_code_identity,
)
from tests.execution_modes.test_mojo_gpu_grouped_typed_explode import (
    test_grouped_typed_expand_literal_raw_domains_and_consumers,
    test_grouped_typed_expand_duplicate_categorical_and_coded_string_identity,
)
from tests.support.grouped_typed_owner_reference import (
    scoped_grouped_typed_owner_contract,
)


def main() raises:
    scoped_grouped_typed_owner_contract()
    print("PASS 1/9 grouped typed scoped owner metadata")
    test_mojo_gpu_categorical_dictionary_budget_owners()
    print("PASS 2/9 category dictionary budget owners")
    test_mojo_gpu_categorical_original_rows_filter_sort_group_and_over()
    print("PASS 3/9 category ROW FILTER SORT GROUP OVER")
    test_mojo_gpu_categorical_join_duplicate_codes_and_dictionary_union()
    print("PASS 4/9 category JOIN same and different dictionaries")
    test_mojo_gpu_categorical_asof_temporary_by_union_preserves_output_owner()
    print("PASS 5/9 category ASOF original output owners")
    test_mojo_gpu_categorical_mixed_join_control_readback_and_budget()
    print("PASS 6/9 mixed JOIN dictionary controls and budget")
    test_mojo_gpu_coded_string_group_publication_drops_original_code_identity()
    print("PASS 7/9 coded String GROUP publication semantics")
    test_grouped_typed_expand_literal_raw_domains_and_consumers()
    print("PASS 8/9 typed grouped EXPLODE raw domains and consumers")
    test_grouped_typed_expand_duplicate_categorical_and_coded_string_identity()
    print("PASS 9/9 typed grouped EXPLODE original dictionary identity")
