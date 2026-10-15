from tests.tooling.iana_mutation_proofs import (
    metadata_unique_tables_and_relocation,
    metadata_deferred_intervals_and_unconditional_policy,
    metadata_occurrence_scopes,
    cpu_unselected_replace_and_masked_operand,
    cpu_fused_inactive_predicate_eager_replace,
)


def main() raises:
    var failures = 0
    try:
        metadata_unique_tables_and_relocation()
        print("PASS metadata_unique_tables_and_relocation", flush=True)
    except error:
        failures += 1
        print(
            "FAIL metadata_unique_tables_and_relocation:",
            String(error),
            flush=True,
        )
    try:
        metadata_deferred_intervals_and_unconditional_policy()
        print(
            "PASS metadata_deferred_intervals_and_unconditional_policy",
            flush=True,
        )
    except error:
        failures += 1
        print(
            "FAIL metadata_deferred_intervals_and_unconditional_policy:",
            String(error),
            flush=True,
        )
    try:
        metadata_occurrence_scopes()
        print("PASS metadata_occurrence_scopes", flush=True)
    except error:
        failures += 1
        print("FAIL metadata_occurrence_scopes:", String(error), flush=True)
    try:
        cpu_unselected_replace_and_masked_operand()
        print("PASS cpu_unselected_replace_and_masked_operand", flush=True)
    except error:
        failures += 1
        print(
            "FAIL cpu_unselected_replace_and_masked_operand:",
            String(error),
            flush=True,
        )
    try:
        cpu_fused_inactive_predicate_eager_replace()
        print("PASS cpu_fused_inactive_predicate_eager_replace", flush=True)
    except error:
        failures += 1
        print(
            "FAIL cpu_fused_inactive_predicate_eager_replace:",
            String(error),
            flush=True,
        )
    if failures:
        raise Error("IANA CPU proof failures: " + String(failures))
    print("PASS all 5 IANA metadata/CPU proofs", flush=True)
