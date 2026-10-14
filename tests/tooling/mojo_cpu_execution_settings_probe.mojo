"""Context-free provider options and CPU eager-versus-stream batch proof."""
from tests.public_contracts.test_accelerator_execution_settings import (
    test_accelerator_settings_preserve_legacy_provider_dispatch_and_plan_copy,
    test_lazy_requested_stream_batch_does_not_override_eager_expression_batches,
)


def main() raises:
    test_accelerator_settings_preserve_legacy_provider_dispatch_and_plan_copy()
    print("PASS provider settings envelope and unchanged legacy interface")
    test_lazy_requested_stream_batch_does_not_override_eager_expression_batches()
    print("PASS CPU requested stream versus effective expression batches")
