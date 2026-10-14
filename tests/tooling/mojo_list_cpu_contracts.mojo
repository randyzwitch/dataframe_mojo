"""Separate CPU metadata and nullable List EXPLODE audit proof."""
from tests.public_contracts.test_list_get_metadata import (
    test_source_list_get_primitive_metadata_seed_empty_batched_identity_and_workers,
    test_source_list_null_parent_nonempty_span_cpu_explode_current_contract,
)


def main() raises:
    test_source_list_get_primitive_metadata_seed_empty_batched_identity_and_workers()
    test_source_list_null_parent_nonempty_span_cpu_explode_current_contract()
    print("PASS CPU List GET metadata and literal null-parent EXPLODE audit")
