"""Requested provider settings and effective eager expression batches."""
from tests.public_contracts.test_accelerator_execution_settings import (
    test_accelerator_settings_preserve_legacy_provider_dispatch_and_plan_copy,
    test_lazy_requested_stream_batch_does_not_override_eager_expression_batches,
)
from tests.execution_modes.test_mojo_gpu_execution_settings import (
    test_mojo_gpu_settings_control_file_planning_and_preserve_eager_fault_batches,
)


def main() raises:
    test_accelerator_settings_preserve_legacy_provider_dispatch_and_plan_copy()
    print("PASS provider settings envelope and backward-compatible dispatch")
    test_lazy_requested_stream_batch_does_not_override_eager_expression_batches()
    print("PASS CPU requested stream versus effective expression batches")
    test_mojo_gpu_settings_control_file_planning_and_preserve_eager_fault_batches()
    print("PASS Metal file planning options and eager fault batches")
