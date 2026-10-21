"""Compile-time Metal eligibility without instantiating an unknown GPU target."""
from std.sys import default_accelerator
from std.sys.info import _accelerator_arch


def compiled_for_metal() -> Bool:
    # StaticString.startswith miscompiles in deep comptime instantiations.
    # Match the standard library's vendor classification with substring tests.
    comptime if default_accelerator().is_apple_gpu():
        return True
    else:
        return False


comptime METAL_TARGET = compiled_for_metal()
