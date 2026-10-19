"""Focused integration gate for typed storage, context reuse and fused sums."""
from tests.tooling.mojo_gpu_device_service_probe import main as service_checks
from tests.tooling.mojo_gpu_typed import main as typed_checks
from tests.tooling.mojo_gpu_fused_int32 import main as fused_checks


def main() raises:
    service_checks()
    typed_checks()
    fused_checks()
