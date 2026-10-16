"""CPU metadata and mask references for original IANA mutation phases."""
# contract-group: execution.apple_parity
# contract-group: execution.iana
# contract-group: execution.backends
from tests.tooling.iana_scalar_any_proofs import (
    metadata_scalar_occurrence_slots_and_resources,
)
from tests.tooling.iana_mutation_proofs import (
    cpu_scalar_any_inactive_predicate_eager_replace,
    metadata_unique_tables_and_relocation,
    metadata_deferred_intervals_and_unconditional_policy,
    metadata_occurrence_scopes,
    cpu_unselected_replace_and_masked_operand,
    cpu_fused_inactive_predicate_eager_replace,
)


def test_iana_metadata_scalar_occurrence_slots_and_resources() raises:
    metadata_scalar_occurrence_slots_and_resources()


def test_iana_cpu_scalar_any_inactive_predicate_eager_replace() raises:
    cpu_scalar_any_inactive_predicate_eager_replace()


def test_iana_metadata_unique_tables_and_relocation() raises:
    metadata_unique_tables_and_relocation()


def test_iana_metadata_deferred_intervals_and_unconditional_policy() raises:
    metadata_deferred_intervals_and_unconditional_policy()


def test_iana_metadata_occurrence_scopes() raises:
    metadata_occurrence_scopes()


def test_iana_cpu_unselected_replace_and_masked_operand() raises:
    cpu_unselected_replace_and_masked_operand()


def test_iana_cpu_fused_inactive_predicate_eager_replace() raises:
    cpu_fused_inactive_predicate_eager_replace()
