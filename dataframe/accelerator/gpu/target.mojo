"""Compile-time Metal eligibility without instantiating an unknown GPU target."""
from std.sys import has_apple_gpu_accelerator
from std.sys.info import _accelerator_arch


def compiled_for_metal() -> Bool:
    # StaticString.startswith miscompiles in deep comptime instantiations.
    # Match the standard library's vendor classification with substring tests.
    comptime if "metal" in _accelerator_arch() or "apple" in _accelerator_arch():
        return has_apple_gpu_accelerator()
    else:
        return False


comptime METAL_TARGET = compiled_for_metal()
