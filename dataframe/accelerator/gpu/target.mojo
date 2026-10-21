"""Compile-time Metal eligibility without instantiating an unknown GPU target."""
from std.sys.info import _accelerator_arch


def compiled_for_metal() -> Bool:
    # StaticString.startswith miscompiles in deep comptime instantiations.
    # Match the standard library's vendor classification with substring tests.
    # `default_accelerator()` fails its architecture constraint under
    # `--target-accelerator=unknown`, which the CPU-only build uses.
    comptime if "metal" in _accelerator_arch() or "apple" in _accelerator_arch():
        return True
    else:
        return False


comptime METAL_TARGET = compiled_for_metal()
