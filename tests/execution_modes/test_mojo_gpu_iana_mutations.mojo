"""Metal IANA mutations preserve CPU masks, diagnostics and recovery."""
# contract-group: execution.apple_parity
# contract-group: execution.iana
# contract-group: execution.backends
from tests.tooling.iana_scalar_any_proofs import (
    gpu_scalar_any_partial_all_false_and_nested,
    gpu_scalar_any_cpu_batch_boundaries,
    gpu_scalar_any_nonzero_sliced_replay,
    gpu_scalar_any_budget_and_recovery,
)
from tests.tooling.iana_mutation_proofs import (
    gpu_public_fault_replay_and_recovery,
    gpu_fused_inactive_predicate_eager_replace,
    gpu_iana_replay_selected_row,
)


# requires: metal
def test_mojo_gpu_iana_scalar_any_partial_all_false_and_nested() raises:
    gpu_scalar_any_partial_all_false_and_nested()


# requires: metal
def test_mojo_gpu_iana_scalar_any_cpu_batch_boundaries() raises:
    gpu_scalar_any_cpu_batch_boundaries()


# requires: metal
def test_mojo_gpu_iana_scalar_any_nonzero_sliced_replay() raises:
    gpu_scalar_any_nonzero_sliced_replay()


# requires: metal
def test_mojo_gpu_iana_scalar_any_budget_and_recovery() raises:
    gpu_scalar_any_budget_and_recovery()


# requires: metal
def test_mojo_gpu_iana_public_fault_replay_and_recovery() raises:
    gpu_public_fault_replay_and_recovery()


# requires: metal
def test_mojo_gpu_iana_fused_inactive_predicate_eager_replace() raises:
    gpu_fused_inactive_predicate_eager_replace()


# requires: metal
def test_mojo_gpu_iana_replay_selected_row() raises:
    gpu_iana_replay_selected_row()
