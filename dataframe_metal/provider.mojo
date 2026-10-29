"""The Metal provider: the fourteen provider functions over the Metal runtime.

The root `dataframe_provider` package re-exports this module, so a plain
`-I .` build uses it. `-I providers/cpu -I .` replaces it with the CPU-only
provider.

The provider surface is fourteen functions whose signatures mention only core
types. Calibration is backend-specific and is imported from the backend.
Presence checks do not open a driver. CPU collection never initializes Metal.
"""
from dataframe.devices import (
    DeviceInfo,
    MEMORY_UNIFIED,
    PipelineFeatures,
    TransferModel,
)
from dataframe.frame import DataFrame
from dataframe.lazy import LazyFrame
from .metal import MetalRuntime, default_runtime, metal_installed
from . import handoff


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


def handoff_supports(region: LazyFrame) -> Bool:
    """Whether Metal runs `region`, a node the CPU executor offers with its
    input already computed; decided from the region's plan and types."""
    if not installed():
        return False
    return handoff.supports(region)


def handoff_execute(region: LazyFrame) raises -> DataFrame:
    """Run an accepted region on Metal."""
    return handoff.execute(region)


def handoff_estimate(region: LazyFrame, rows: Int) -> Tuple[String, Float64]:
    """The region's calibration key and Metal's measured cost for `rows`
    input rows, in nanoseconds; -1 when there is none."""
    if not installed():
        return (String(), Float64(-1))
    return handoff.estimate(region, rows)


def devices() -> List[DeviceInfo]:
    """The default Apple GPU when Metal can run here; no device is opened.
    Unified memory is shared with the CPU, so there is no planner budget."""
    var result = List[DeviceInfo]()
    if installed():
        result.append(DeviceInfo("metal", 0, "Apple GPU", MEMORY_UNIFIED, -1))
    return result^


def transfer_model(device: Int) -> TransferModel:
    """The GPU reads host buffers in place (`wrap_host_memory`); start-up is
    not measured yet."""
    if not installed() or device != 0:
        return TransferModel.unmeasured()
    return TransferModel.shared_memory()


def pipeline_supports(pipeline: LazyFrame, device: Int) -> Bool:
    """Whether Metal runs `pipeline`: the operators hand-off accepts, as a
    plan over the pipeline's inputs."""
    return device == 0 and handoff_supports(pipeline)


def pipeline_estimate(
    pipeline: LazyFrame, device: Int, features: PipelineFeatures
) -> Float64:
    """Metal's measured nanoseconds for the pipeline's input rows; -1 when
    there is no measurement."""
    if device != 0 or features.input_rows < 0:
        return Float64(-1)
    return handoff_estimate(pipeline, features.input_rows)[1]


def run_pipeline(pipeline: LazyFrame, device: Int) raises -> DataFrame:
    if not installed():
        raise Error("Metal is not available here; no device runs pipelines")
    if device != 0:
        raise Error("Metal runs pipelines on device 0 only")
    return handoff_execute(pipeline)
