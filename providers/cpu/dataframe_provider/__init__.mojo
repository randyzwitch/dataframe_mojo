"""The CPU-only accelerator provider: no backend is installed.

`dataframe/_accel_provider.mojo` imports the nine functions below. Put this
package first on the import path (`-I providers/cpu -I .`) for a build that
compiles no GPU code. The default `dataframe_provider` in the repository root
is the Metal provider.
"""
from dataframe.frame import DataFrame
from dataframe.lazy import LazyFrame


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
    return ("cpu", "auto uses CPU; accelerator provider is not installed")


def execute_auto(
    plan: LazyFrame,
    optimize: Bool,
    streaming: Bool,
    batch_size: Int,
    profiling: Bool,
) raises -> Tuple[DataFrame, DataFrame]:
    return execute(plan)


def handoff_supports(region: LazyFrame) -> Bool:
    return False


def handoff_execute(region: LazyFrame) raises -> DataFrame:
    raise Error("Accelerator provider is not installed; nothing is handed off")


def handoff_estimate(region: LazyFrame, rows: Int) -> Tuple[String, Float64]:
    return (String(), Float64(-1))
