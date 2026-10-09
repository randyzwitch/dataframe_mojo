"""NVIDIA provider registered by the optional package build.

A default collection owns its context reference until all work completes. This
provider retains no process-wide runtime; the SDK may retain allocation caches.
Explicit runtime handles remain available for deliberate context reuse.
"""
from std.os import getenv
from dataframe.accel_plan import lower_accel
from dataframe.frame import DataFrame
from dataframe.lazy import LazyFrame
from .nvidia import NvidiaRuntime
from .query import describe as describe_plan


def installed() -> Bool:
    return True


def selected_device() raises -> Int:
    var setting = getenv("DATAFRAME_ACCEL_DEVICE")
    if not setting:
        return 0
    var device: Int
    try:
        device = Int(setting)
    except:
        raise Error(
            "DATAFRAME_ACCEL_DEVICE must be a nonnegative CUDA device ordinal"
        )
    if device < 0:
        raise Error(
            "DATAFRAME_ACCEL_DEVICE must be a nonnegative CUDA device ordinal"
        )
    return device


def _available_device() raises -> Int:
    var device = selected_device()
    if device >= NvidiaRuntime.device_count():
        raise Error("NVIDIA device is unavailable: " + String(device))
    return device


def execute(plan: LazyFrame) raises -> Tuple[DataFrame, DataFrame]:
    # Reject unsupported sources/expressions before creating a CUDA context.
    _ = lower_accel(plan)
    var runtime = NvidiaRuntime(_available_device())
    return runtime.execute(plan)


def describe(plan: LazyFrame) -> String:
    try:
        _ = lower_accel(plan)
        var device = _available_device()
        return (
            describe_plan(plan)
            + "  registered provider: NVIDIA; device="
            + String(device)
            + "; context scoped to collection\n"
        )
    except error:
        return "ENGINE accel: " + String(error) + "\n"
