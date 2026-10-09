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
