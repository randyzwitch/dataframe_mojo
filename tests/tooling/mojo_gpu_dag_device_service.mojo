"""Focused device gate for the runtime-owned resident DAG service."""

from tests.execution_modes.test_mojo_gpu_device_service import (
    test_mojo_gpu_service_backed_branched_join_dag_reuses_runtime_copies,
    test_mojo_gpu_service_backed_grouped_fields_dag_counts_terminal_drain,
    test_mojo_gpu_service_backed_dag_overlap_and_post_submission_poison,
    test_mojo_gpu_service_backed_dag_budget_failure_keeps_warm_context,
)


def main() raises:
    test_mojo_gpu_service_backed_branched_join_dag_reuses_runtime_copies()
    test_mojo_gpu_service_backed_grouped_fields_dag_counts_terminal_drain()
    test_mojo_gpu_service_backed_dag_overlap_and_post_submission_poison()
    test_mojo_gpu_service_backed_dag_budget_failure_keeps_warm_context()
    print("PASS runtime-owned DAG context reuse, lifetime, poison and budget")
