"""Default provider slot with context-free Mojo Metal availability.

Presence checks do not open a driver. CPU collection never initializes Metal.
The optional NVIDIA package builder can replace this module in its staged tree.
"""
from .accelerator.cost import PlanEstimate, CalibrationKey
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


def calibration_runtime() raises -> MetalRuntime:
    """Explicit runtime factory for the provider-neutral benchmark driver."""
    return _runtime()


def calibration_key(
    optimize: Bool, streaming: Bool, batch_size: Int
) raises -> CalibrationKey:
    """Provider metadata for the common calibration harness; no submission."""
    return _runtime().calibration_key(optimize, streaming, batch_size)


def estimate_auto(plan: LazyFrame) raises -> PlanEstimate:
    """Plan-time facts required by the provider-neutral calibration runner."""
    return _runtime().estimate_auto(plan)


def select_auto(
    plan: LazyFrame,
    optimize: Bool,
    streaming: Bool,
    batch_size: Int,
) -> Tuple[String, String]:
    """Select before execution; providers supply facts to the shared analytical cost selector."""
    if installed():
        try:
            return _runtime().select_auto(plan, optimize, streaming, batch_size)
        except error:
            return (
                "cpu",
                "automatic analytical preflight rejected: " + String(error),
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
    return (
        _runtime()
        .execute_profiled(plan) if profiling else _runtime()
        .execute(plan)
    )
