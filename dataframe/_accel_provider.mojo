"""Default provider slot with optional native Metal discovery.

Presence checks do not open a driver. CPU collection never initializes Metal.
The optional NVIDIA package builder can replace this module in its staged tree.
"""
from .frame import DataFrame
from .lazy import LazyFrame
from std.os import getenv
from .metal import MetalRuntime, metal_installed


def installed() -> Bool:
    return metal_installed()


def _runtime() raises -> MetalRuntime:
    var device = 0
    var limit = -1
    var configured_device = getenv("DATAFRAME_ACCEL_DEVICE")
    var configured_limit = getenv("DATAFRAME_ACCEL_MEMORY_LIMIT")
    if configured_device:
        device = Int(configured_device)
        if device < 0:
            raise Error("DATAFRAME_ACCEL_DEVICE must be nonnegative")
    if configured_limit:
        limit = Int(configured_limit)
        if limit < 0:
            raise Error("DATAFRAME_ACCEL_MEMORY_LIMIT must be nonnegative")
    return MetalRuntime(device, memory_limit_bytes=limit)


def execute(plan: LazyFrame) raises -> Tuple[DataFrame, DataFrame]:
    if installed():
        return _runtime().execute(plan)
    raise Error(
        "Accelerator provider is not installed; use the optional GPU package or accelerator=runtime"
    )


def describe(plan: LazyFrame) -> String:
    if installed():
        try:
            return _runtime().describe(plan)
        except error:
            return "ENGINE accel: " + String(error) + "\n"
    return "ENGINE accel: Accelerator provider is not installed; use the optional GPU package or accelerator=runtime\n"


def execute_profiled(plan: LazyFrame) raises -> Tuple[DataFrame, DataFrame]:
    if installed():
        return _runtime().execute_profiled(plan)
    return execute(plan)


def select_auto(
    plan: LazyFrame,
    optimize: Bool,
    streaming: Bool,
    batch_size: Int,
) -> Tuple[String, String]:
    """Select before execution; optional providers own capability and cost gates."""
    if installed():
        return (
            "cpu",
            "auto uses CPU; Metal has no matched end-to-end cost evidence",
        )
    return ("cpu", "auto uses CPU; accelerator provider is not installed")


def execute_auto(
    plan: LazyFrame,
    optimize: Bool,
    streaming: Bool,
    batch_size: Int,
    profiling: Bool,
) raises -> Tuple[DataFrame, DataFrame]:
    """Execute a selected region; execution errors must propagate to callers."""
    return execute(plan)
