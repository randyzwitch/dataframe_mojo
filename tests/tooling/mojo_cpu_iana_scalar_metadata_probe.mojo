from tests.tooling.iana_scalar_any_proofs import (
    metadata_scalar_occurrence_slots_and_resources,
)
from tests.tooling.iana_mutation_proofs import (
    cpu_scalar_any_inactive_predicate_eager_replace,
)


def main() raises:
    metadata_scalar_occurrence_slots_and_resources()
    print("PASS metadata_scalar_occurrence_slots_and_resources", flush=True)
    cpu_scalar_any_inactive_predicate_eager_replace()
    print("PASS cpu_scalar_any_inactive_predicate_eager_replace", flush=True)
