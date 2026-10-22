"""The CPU-only accelerator provider: no backend is installed.

`dataframe/_accel_provider.mojo` imports the six functions below. A backend
build puts its own `dataframe_provider` package ahead of this one on the import
path, for example `-I providers/metal -I .`.
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
