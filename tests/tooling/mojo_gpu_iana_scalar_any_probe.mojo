from tests.tooling.iana_scalar_any_proofs import (
    metadata_scalar_occurrence_slots_and_resources,
    gpu_scalar_any_partial_all_false_and_nested,
    gpu_scalar_any_cpu_batch_boundaries,
    gpu_scalar_any_nonzero_sliced_replay,
    gpu_scalar_any_budget_and_recovery,
)


def main() raises:
    metadata_scalar_occurrence_slots_and_resources()
    print("PASS metadata_scalar_occurrence_slots_and_resources", flush=True)
    gpu_scalar_any_partial_all_false_and_nested()
    print("PASS gpu_scalar_any_partial_all_false_and_nested", flush=True)
    gpu_scalar_any_cpu_batch_boundaries()
    print("PASS gpu_scalar_any_cpu_batch_boundaries", flush=True)
    gpu_scalar_any_nonzero_sliced_replay()
    print("PASS gpu_scalar_any_nonzero_sliced_replay", flush=True)
    gpu_scalar_any_budget_and_recovery()
    print("PASS gpu_scalar_any_budget_and_recovery", flush=True)
