"""CPU distribution's provider slot; optional builds register a backend here.

This module must not import accelerator packages. The optional package builder
replaces it in a temporary source tree, never in the checked-out CPU sources.
"""
from .frame import DataFrame
from .lazy import LazyFrame


def installed() -> Bool:
    return False


def execute(plan: LazyFrame) raises -> Tuple[DataFrame, DataFrame]:
    raise Error(
        "Accelerator provider is not installed; use the optional GPU package or accelerator=runtime"
    )


def describe(plan: LazyFrame) -> String:
    return "ENGINE accel: Accelerator provider is not installed; use the optional GPU package or accelerator=runtime\n"


def execute_profiled(plan: LazyFrame) raises -> Tuple[DataFrame, DataFrame]:
    return execute(plan)


def select_auto(
    plan: LazyFrame,
    optimize: Bool,
    streaming: Bool,
    batch_size: Int,
) -> Tuple[String, String]:
    """Select before execution; optional providers own capability and cost gates."""
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
