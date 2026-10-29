"""Context-free Mojo Metal identity and Apple host memory headroom.

Importing this module and reading calibration identity never constructs a GPU
context. Only the explicit device_count query discovers devices through MAX.
The operating-system calls below read host facts; no dataframe native ABI is
loaded. Hardware and power state are part of every calibration match.
"""
from std.ffi import external_call
from std.memory import Pointer
from std.sys import CompilationTarget, size_of
from max.gpu.host import DeviceContext
from dataframe._build_identity import CPU_BUILD_ID
from .target import METAL_TARGET

comptime PROVIDER_BUILD = "dataframe-mojo-metal-v1"
comptime _NameMessage = def(Int) thin abi("C") -> Int
comptime _ObjectMessage = def(Int, Int) thin abi("C") -> Int
comptime _BoolMessage = def(Int, Int) thin abi("C") -> UInt8
comptime _IntegerMessage = def(Int, Int) thin abi("C") -> Int64


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


def _c_string(value: String) -> List[UInt8]:
    var bytes = List[UInt8]()
    bytes.extend(value.as_bytes())
    bytes.append(0)
    return bytes^


def _sysctl_string(name: String) raises -> String:
    comptime if not CompilationTarget.is_macos():
        raise Error("Mojo Metal host identity requires macOS")
    else:
        var key = _c_string(name)
        var bytes = List[UInt8](length=256, fill=0)
        var size = len(bytes)
        if (
            external_call["sysctlbyname", Int32](
                key.unsafe_ptr(), bytes.unsafe_ptr(), Pointer(to=size), 0, 0
            )
            != 0
        ):
            raise Error("Cannot read Apple calibration host identity: " + name)
        var length = 0
        while length < min(size, len(bytes)) and bytes[length] != 0:
            length += 1
        return String(unsafe_from_utf8=bytes[0:length])


def _sysctl_integer(name: String) raises -> UInt64:
    comptime if not CompilationTarget.is_macos():
        raise Error("Mojo Metal host identity requires macOS")
    else:
        var key = _c_string(name)
        var value = UInt64(0)
        var size = size_of[UInt64]()
        if external_call["sysctlbyname", Int32](
            key.unsafe_ptr(), Pointer(to=value), Pointer(to=size), 0, 0
        ) != 0 or (size != 4 and size != 8):
            raise Error("Cannot read Apple calibration host identity: " + name)
        return value


def _power_state() raises -> Tuple[UInt8, Int64]:
    comptime if not CompilationTarget.is_macos():
        raise Error("Mojo Metal power identity requires macOS")
    else:
        # Foundation supplies host power/thermal facts without a Metal device.
        var path = _c_string(
            "/System/Library/Frameworks/Foundation.framework/Foundation"
        )
        var handle = external_call["dlopen", Int](path.unsafe_ptr(), Int32(1))
        if not handle:
            raise Error("Cannot load Foundation host power metadata")
        # Resolve Objective-C entry points through Foundation's dependencies;
        # standalone executables need no additional framework linker flags.
        var class_name = _c_string("objc_getClass")
        var class_lookup = external_call["dlsym", Int](
            handle, class_name.unsafe_ptr()
        )
        var selector_name = _c_string("sel_registerName")
        var selector_lookup = external_call["dlsym", Int](
            handle, selector_name.unsafe_ptr()
        )
        if not class_lookup or not selector_lookup:
            _ = external_call["dlclose", Int32](handle)
            raise Error("Cannot resolve Foundation host metadata entry points")
        var name = _c_string("NSProcessInfo")
        var cls = Pointer(to=class_lookup).unsafe_bitcast[_NameMessage]()[](
            Int(name.unsafe_ptr())
        )
        # Integer-address callback parameters erase pointer origins. Retain
        # every C-string owner until its indirect Foundation call completes.
        _ = name^
        var process_name = _c_string("processInfo")
        var process_selector = Pointer(to=selector_lookup).unsafe_bitcast[
            _NameMessage
        ]()[](Int(process_name.unsafe_ptr()))
        _ = process_name^
        var message_name = _c_string("objc_msgSend")
        var message = external_call["dlsym", Int](
            handle, message_name.unsafe_ptr()
        )
        if not message:
            _ = external_call["dlclose", Int32](handle)
            raise Error("Cannot read Foundation host metadata")
        var process = Pointer(to=message).unsafe_bitcast[_ObjectMessage]()[](
            cls, process_selector
        )
        if not cls or not process:
            _ = external_call["dlclose", Int32](handle)
            raise Error("Cannot identify Apple host power state")
        var low_name = _c_string("isLowPowerModeEnabled")
        var low_selector = Pointer(to=selector_lookup).unsafe_bitcast[
            _NameMessage
        ]()[](Int(low_name.unsafe_ptr()))
        _ = low_name^
        var thermal_name = _c_string("thermalState")
        var thermal_selector = Pointer(to=selector_lookup).unsafe_bitcast[
            _NameMessage
        ]()[](Int(thermal_name.unsafe_ptr()))
        _ = thermal_name^
        var low = Pointer(to=message).unsafe_bitcast[_BoolMessage]()[](
            process, low_selector
        )
        var thermal = Pointer(to=message).unsafe_bitcast[_IntegerMessage]()[](
            process, thermal_selector
        )
        _ = external_call["dlclose", Int32](handle)
        return (low, thermal)


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
    # The package builder hashes every dataframe source, the complete lock,
    # full `mojo --version` and compiler flags into CPU_BUILD_ID. This binds
    # the provider to actual compiler/SDK builds without duplicating pins.
    return PROVIDER_BUILD + ";source-compiler-sdk=" + CPU_BUILD_ID
