"""The CPU-only accelerator provider: no backend is installed.

`dataframe/_accel_provider.mojo` imports the twelve functions below. Put this
package first on the import path (`-I providers/cpu -I .`) for a build that
compiles no GPU code. The default `dataframe_provider` in the repository root
is the Metal provider.
"""
from dataframe.devices import DeviceInfo, PipelineFeatures, TransferModel
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


def handoff_supports(region: LazyFrame) -> Bool:
    return False


def handoff_execute(region: LazyFrame) raises -> DataFrame:
    raise Error("Accelerator provider is not installed; nothing is handed off")


def handoff_estimate(region: LazyFrame, rows: Int) -> Tuple[String, Float64]:
    return (String(), Float64(-1))


def devices() -> List[DeviceInfo]:
    return List[DeviceInfo]()


def transfer_model(device: Int) -> TransferModel:
    return TransferModel.unmeasured()


def pipeline_supports(pipeline: LazyFrame, device: Int) -> Bool:
    return False


def pipeline_estimate(
    pipeline: LazyFrame, device: Int, features: PipelineFeatures
) -> Float64:
    return Float64(-1)


def run_pipeline(pipeline: LazyFrame, device: Int) raises -> DataFrame:
    raise Error(
        "Accelerator provider is not installed; no device runs pipelines"
    )
