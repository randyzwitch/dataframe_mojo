from tests.tooling.iana_mutation_proofs import (
    gpu_fused_inactive_predicate_eager_replace,
    gpu_iana_replay_selected_row,
)


def main() raises:
    gpu_fused_inactive_predicate_eager_replace()
    gpu_iana_replay_selected_row()
    print(
        "PASS GPU IANA retained fusion, masked Float32 and one-lane replay witnesses",
        flush=True,
    )
