"""Report Mojo Metal hardware capability without a dataframe native library."""
from std.sys import CompilationTarget
from dataframe.metal import MetalRuntime, metal_installed


def main() raises:
    if not CompilationTarget.is_macos():
        print("absent")
        return
    if not metal_installed():
        print("absent")
        return
    print("available" if MetalRuntime.device_count() > 0 else "absent")
