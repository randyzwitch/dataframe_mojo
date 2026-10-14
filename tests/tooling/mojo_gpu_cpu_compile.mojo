"""Compile and exercise backend dispatch with no accelerator architecture."""
from std.testing import assert_true
from dataframe.accelerator.gpu.target import compiled_for_metal
from tests.execution_modes.test_backends import (
    test_metal_cpu_only_target_keeps_cpu_dispatch_and_rejects_gpu_submission,
    test_registered_slot_cpu_fallback_and_absence_are_explicit,
    test_native_provider_configuration_and_unsupported_types_reject_before_driver_setup,
)


def main() raises:
    assert_true(not compiled_for_metal())
    test_metal_cpu_only_target_keeps_cpu_dispatch_and_rejects_gpu_submission()
    test_registered_slot_cpu_fallback_and_absence_are_explicit()
    test_native_provider_configuration_and_unsupported_types_reject_before_driver_setup()
    print(
        "PASS CPU-only Mojo provider compilation, CPU dispatch and explicit GPU rejection"
    )
