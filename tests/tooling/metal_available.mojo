"""Report hardware absence separately from a broken native installation."""
from std.sys import CompilationTarget
from dataframe.metal import MetalRuntime, metal_installed


def main() raises:
    if not CompilationTarget.is_macos():
        print("absent")
        return
    if not metal_installed():
        raise Error("Metal capability probe requires the built native library")
    print("available" if MetalRuntime.device_count() > 0 else "absent")
