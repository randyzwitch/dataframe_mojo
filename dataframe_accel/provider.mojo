"""NVIDIA provider registered by the optional package build.

A default collection owns its context reference until all work completes. This
provider retains no process-wide runtime; the SDK may retain allocation caches.
Explicit runtime handles remain available for deliberate context reuse.
"""
from std.os import getenv
from std.time import perf_counter_ns
from dataframe.column import Column
from dataframe.series import Series
from dataframe.accel_plan import lower_accel
from dataframe.accel_rows import lower_rows
from dataframe.frame import DataFrame
from dataframe.lazy import LazyFrame
from .nvidia import NvidiaRuntime


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


def _memory_limit() raises -> Int:
    var setting = getenv("DATAFRAME_ACCEL_MEMORY_LIMIT")
    if not setting:
        return -1
    var limit: Int
    try:
        limit = Int(setting)
    except:
        raise Error("DATAFRAME_ACCEL_MEMORY_LIMIT must be nonnegative bytes")
    if limit < 0:
        raise Error("DATAFRAME_ACCEL_MEMORY_LIMIT must be nonnegative bytes")
    return limit


def _execute(
    plan: LazyFrame, profiling: Bool
) raises -> Tuple[DataFrame, DataFrame]:
    validate(plan)
    var device = _available_device()
    var limit = _memory_limit()
    var start = perf_counter_ns()
    var runtime = NvidiaRuntime(device, memory_limit_bytes=limit)
    var initialization_ms = Float64(perf_counter_ns() - start) / 1e6
    var result = runtime.execute_profiled(
        plan
    ) if profiling else runtime.execute(plan)
    var report = result[1].with_column(
        Series("initialization_ms", Column[Float64]([initialization_ms]))
    )
    return (result[0].copy(), report^)


def execute(plan: LazyFrame) raises -> Tuple[DataFrame, DataFrame]:
    return _execute(plan, False)


def execute_profiled(plan: LazyFrame) raises -> Tuple[DataFrame, DataFrame]:
    return _execute(plan, True)


def describe(plan: LazyFrame) -> String:
    try:
        validate(plan)
        var runtime = NvidiaRuntime(
            _available_device(), memory_limit_bytes=_memory_limit()
        )
        return (
            runtime.describe(plan)
            + "  registered provider: NVIDIA; context scoped to collection\n"
        )
    except error:
        return "ENGINE accel: " + String(error) + "\n"


def validate(plan: LazyFrame) raises:
    var supported = True
    try:
        _ = lower_accel(plan)
    except:
        supported = False
    if not supported:
        _ = lower_rows(plan)
