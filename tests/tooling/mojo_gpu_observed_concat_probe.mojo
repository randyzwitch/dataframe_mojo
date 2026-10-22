"""Run the sparse observed-concat and existing concat contracts on Metal."""
from tests.execution_modes.test_backend_metal_observed_concat import (
    test_native_metal_observed_concat_sparse_pairs_and_consumers,
)
from tests.execution_modes.test_backend_metal_string_concat import (
    test_native_metal_concat_dictionary_pairs_nulls_unicode_literals_and_chains,
    test_native_metal_concat_dictionary_domains_replacement_aliases_and_internal_names,
    test_native_metal_concat_dictionary_group_filter_join_order_and_duplicate_outputs,
    test_native_metal_concat_dictionary_pair_capacity_byte_budget_and_recovery,
)


def main() raises:
    test_native_metal_observed_concat_sparse_pairs_and_consumers()
    test_native_metal_concat_dictionary_pairs_nulls_unicode_literals_and_chains()
    test_native_metal_concat_dictionary_domains_replacement_aliases_and_internal_names()
    test_native_metal_concat_dictionary_group_filter_join_order_and_duplicate_outputs()
    test_native_metal_concat_dictionary_pair_capacity_byte_budget_and_recovery()
    print("PASS all existing and observed Metal string concat contracts")
