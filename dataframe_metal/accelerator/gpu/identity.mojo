"""Context-free Mojo Metal identity and Apple host memory headroom.

Importing this module and reading calibration identity never constructs a GPU
context. Only the explicit device_count query discovers devices through MAX.
The operating-system calls below read host facts; no dataframe native ABI is
loaded. Hardware and power state are part of every calibration match.
"""
from std.ffi import OwnedDLHandle, external_call
from std.memory import Pointer
from std.sys import CompilationTarget, size_of
from .target import MetalContext as DeviceContext
from dataframe._build_identity import CPU_BUILD_ID
from ..._build_identity import METAL_BUILD_ID
from .target import METAL_TARGET

comptime PROVIDER_BUILD = "dataframe-mojo-metal-v2"


def available() -> Bool:
    comptime if not CompilationTarget.is_macos():
        return False
    comptime if not METAL_TARGET:
        return False
    else:
        return String(DeviceContext.default_device_info.api) == "metal"


def device_count() raises -> Int:
    comptime if not METAL_TARGET:
        return 0
    else:
        if not available():
            return 0
        return DeviceContext.number_of_devices()


def _sysctl_string(var name: String) raises -> String:
    comptime if not CompilationTarget.is_macos():
        raise Error("Mojo Metal host identity requires macOS")
    else:
        var key = name.as_c_string_span()
        var bytes = List[UInt8](length=256, fill=0)
        var size = len(bytes)
        if (
            external_call["sysctlbyname", Int32](
                key.ptr(), bytes.unsafe_ptr(), Pointer(to=size), 0, 0
            )
            != 0
        ):
            raise Error("Cannot read Apple calibration host identity: " + name)
        var length = 0
        while length < min(size, len(bytes)) and bytes[length] != 0:
            length += 1
        return String(unsafe_from_utf8=bytes[0:length])


def _sysctl_integer(var name: String) raises -> UInt64:
    comptime if not CompilationTarget.is_macos():
        raise Error("Mojo Metal host identity requires macOS")
    else:
        var key = name.as_c_string_span()
        var value = UInt64(0)
        var size = size_of[UInt64]()
        if external_call["sysctlbyname", Int32](
            key.ptr(), Pointer(to=value), Pointer(to=size), 0, 0
        ) != 0 or (size != 4 and size != 8):
            raise Error("Cannot read Apple calibration host identity: " + name)
        return value


def _power_state() raises -> Tuple[UInt8, Int64]:
    comptime if not CompilationTarget.is_macos():
        raise Error("Mojo Metal power identity requires macOS")
    else:
        var library = OwnedDLHandle(
            "/System/Library/Frameworks/Foundation.framework/Foundation"
        )
        var class_lookup = library.get_function[Int]("objc_getClass")
        var selector_lookup = library.get_function[Int]("sel_registerName")
        var message = library.get_function[Int]("objc_msgSend")
        var low_message = library.get_function[UInt8]("objc_msgSend")
        var thermal_message = library.get_function[Int64]("objc_msgSend")
        var name = String("NSProcessInfo")
        var cls = class_lookup(name.as_c_string_span().ptr())
        var process_name = String("processInfo")
        var process_selector = selector_lookup(
            process_name.as_c_string_span().ptr()
        )
        var process = message(cls, process_selector)
        if not cls or not process:
            raise Error("Cannot identify Apple host power state")
        var low_name = String("isLowPowerModeEnabled")
        var low_selector = selector_lookup(low_name.as_c_string_span().ptr())
        var thermal_name = String("thermalState")
        var thermal_selector = selector_lookup(
            thermal_name.as_c_string_span().ptr()
        )
        return (
            low_message(process, low_selector),
            thermal_message(process, thermal_selector),
        )


def _memory_headroom(total: Int) raises -> Int:
    comptime if not CompilationTarget.is_macos():
        raise Error("Mojo Metal memory headroom requires macOS")
    else:
        var host = external_call["mach_host_self", UInt32]()
        var task = external_call["mach_task_self", UInt32]()
        var page = UInt64(0)
        var page_status = external_call["host_page_size", Int32](
            host, Pointer(to=page)
        )
        # vm_statistics64 starts with four UInt32 page counts. An aligned
        # oversized buffer accepts all SDK revisions without reading tails.
        var statistics = List[UInt64](length=64, fill=0)
        var count = UInt32(128)
        var state_status = external_call["host_statistics64", Int32](
            host, Int32(4), statistics.unsafe_ptr(), Pointer(to=count)
        )
        _ = external_call["mach_port_deallocate", Int32](task, host)
        if page_status != 0 or state_status != 0 or count < 4 or page == 0:
            raise Error("Cannot read Apple host memory headroom")
        var free = UInt64(UInt32(statistics[0]))
        var inactive = UInt64(UInt32(statistics[1]))
        var pages = free + inactive
        if pages > UInt64(Int.MAX) // page:
            return total
        return min(total, Int(pages * page))


@fieldwise_init
struct HardwareFacts(Copyable):
    var device: String
    var host: String
    var total_bytes: Int
    var available_bytes: Int


def hardware(device: Int = 0) raises -> HardwareFacts:
    comptime if METAL_TARGET:
        return _metal_hardware(device)
    else:
        raise Error("Mojo Metal calibration requires an Apple Metal target")


def _metal_hardware(device: Int) raises -> HardwareFacts:
    if not available():
        raise Error("Mojo Metal calibration requires an Apple Metal target")
    if device != 0:
        raise Error(
            "Mojo Metal automatic calibration requires default Apple GPU device 0"
        )
    var model = _sysctl_string("hw.model")
    var chip = _sysctl_string("machdep.cpu.brand_string")
    var total = _sysctl_integer("hw.memsize")
    if total > UInt64(Int.MAX):
        raise Error("Apple physical memory exceeds supported allocation range")
    var power = _power_state()
    var device_key = (
        "mojo-metal;ordinal=0;model="
        + model
        + ";chip="
        + chip
        + ";target="
        + String(DeviceContext.default_device_info.arch_name)
        + ";unified=1"
    )
    var host_key = (
        _sysctl_string("kern.hostname")
        + ";model="
        + model
        + ";memory="
        + String(total)
        + ";processors="
        + String(_sysctl_integer("hw.activecpu"))
        + ";os="
        + _sysctl_string("kern.osrelease")
        + ";os_build="
        + _sysctl_string("kern.osversion")
        + ";lowPower="
        + String(power[0])
        + ";thermal="
        + String(power[1])
    )
    return HardwareFacts(
        device_key^, host_key^, Int(total), _memory_headroom(Int(total))
    )


def runtime_build_id() -> String:
    if CPU_BUILD_ID == "development-source-no-calibration":
        return CPU_BUILD_ID
    if METAL_BUILD_ID == "development-metal-no-calibration":
        return METAL_BUILD_ID
    # Both generated identities bind sources, compiler/SDK and build flags.
    # A CPU rewrite invalidates the CPU hash and the dependent Metal hash.
    return PROVIDER_BUILD + ";cpu=" + CPU_BUILD_ID + ";metal=" + METAL_BUILD_ID
