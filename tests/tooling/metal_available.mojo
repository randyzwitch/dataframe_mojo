"""Report Mojo Metal hardware capability without a dataframe native library.

Prints `available` or `absent` on stdout, and on stderr why: the accelerator
this program was compiled for (none unless the compiler detected or was told
one), and the devices found."""
from std.io import RawFileDescriptor
from std.sys import CompilationTarget
from std.sys.info import _accelerator_arch
from dataframe.accelerator.gpu.target import compiled_for_metal
from dataframe.metal import MetalRuntime, metal_installed


def main() raises:
    var why = RawFileDescriptor(2)
    print(
        "metal probe: accelerator '", _accelerator_arch(), "'", sep="", file=why
    )
    if not CompilationTarget.is_macos():
        print("metal probe: not macOS", file=why)
        print("absent")
        return
    if not compiled_for_metal():
        print(
            "metal probe: not compiled for an Apple GPU (name one with"
            " --target-accelerator=apple-mN)",
            file=why,
        )
        print("absent")
        return
    if not metal_installed():
        print("metal probe: the default device is not Metal", file=why)
        print("absent")
        return
    var devices = MetalRuntime.device_count()
    print("metal probe: devices", devices, file=why)
    print("available" if devices > 0 else "absent")
