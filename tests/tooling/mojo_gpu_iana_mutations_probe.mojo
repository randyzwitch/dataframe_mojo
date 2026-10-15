from tests.tooling.iana_mutation_proofs import (
    metadata_unique_tables_and_relocation,
    metadata_deferred_intervals_and_unconditional_policy,
    metadata_occurrence_scopes,
    cpu_unselected_replace_and_masked_operand,
    gpu_public_fault_replay_and_recovery,
)


def main() raises:
    metadata_unique_tables_and_relocation()
    metadata_deferred_intervals_and_unconditional_policy()
    metadata_occurrence_scopes()
    cpu_unselected_replace_and_masked_operand()
    gpu_public_fault_replay_and_recovery()
    print("PASS all 5 IANA mutation metadata/CPU/device proofs")
