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
from .accelerator.faults import is_data_fault
from .calibration import shape_key
from .region_plan import normalize_region
from .accelerator.gpu.identity import runtime_build_id
from dataframe.json.value import json_quote
from std.os import getenv


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
    if device != 0:
        return False
    try:
        var normalized = normalize_region(pipeline)
        return Bool(normalized) and handoff_supports(normalized.value())
    except:
        return False


def pipeline_estimate(
    pipeline: LazyFrame, device: Int, features: PipelineFeatures
) -> Float64:
    """Metal's measured nanoseconds for the pipeline's input rows; -1 when
    there is no measurement."""
    if device != 0 or features.input_rows < 0:
        return Float64(-1)
    # A schema-only input cannot prove its physical layout. The pipeline
    # planner preserves actual in-memory scans; defer unknown inputs to CPU.
    var frame_rows = pipeline._plan._frames[
        pipeline._plan._nodes[0].source_slot()
    ].height()
    if frame_rows != features.input_rows:
        return Float64(-1)
    try:
        var normalized = normalize_region(pipeline)
        if not normalized:
            return Float64(-1)
        return handoff_estimate(normalized.value(), features.input_rows)[1]
    except:
        return Float64(-1)


def run_pipeline(pipeline: LazyFrame, device: Int) raises -> DataFrame:
    if not installed():
        raise Error("Metal is not available here; no device runs pipelines")
    if device != 0:
        raise Error("Metal runs pipelines on device 0 only")
    try:
        var result = handoff_execute(pipeline)
        _trace_placement(pipeline, "metal")
        return result^
    except error:
        if not is_data_fault(error):
            raise error^
        var result = pipeline.collect(engine="cpu")
        _trace_placement(pipeline, "cpu-adjudication")
        return result^


def _trace_placement(pipeline: LazyFrame, engine: String) raises:
    var path = getenv("DATAFRAME_METAL_PLACEMENT_TRACE")
    if not path:
        return
    var normalized = normalize_region(pipeline)
    var selected = normalized.take()
    var key = shape_key(selected, handoff.region_key(selected))
    var rows = pipeline._plan._frames[
        pipeline._plan._nodes[0].source_slot()
    ].height()
    var record = (
        '{"engine":'
        + json_quote(engine)
        + ',"key":'
        + json_quote(key)
        + ',"rows":'
        + String(rows)
        + ',"runtime_build_id":'
        + json_quote(runtime_build_id())
        + "}\n"
    )
    with open(path, "a") as target:
        target.write(record)
