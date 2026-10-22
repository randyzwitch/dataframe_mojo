"""The Metal provider: the six provider functions over the Metal runtime.

The root `dataframe_provider` package re-exports this module, so a plain
`-I .` build uses it. `-I providers/cpu -I .` replaces it with the CPU-only
provider.

The provider surface is six functions whose signatures mention only core
types. Calibration is backend-specific and is imported from the backend.
Presence checks do not open a driver. CPU collection never initializes Metal.
"""
from dataframe.frame import DataFrame
from dataframe.lazy import LazyFrame
from .metal import MetalRuntime, default_runtime, metal_installed


def installed() -> Bool:
    return metal_installed()


def _runtime() raises -> MetalRuntime:
    return default_runtime()


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
