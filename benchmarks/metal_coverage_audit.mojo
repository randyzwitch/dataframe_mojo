"""Coverage-only wrapper recording observed pipeline execution and adjudication."""
from dataframe import DataFrame
from dataframe.execution_report import pipeline_record_json
from dataframe.lazy import AcceleratorBackend, LazyFrame
from dataframe_metal.metal import MetalRuntime


def _adjudication(report: DataFrame) raises -> String:
    """A collection summary; detailed pipeline records remain the evidence."""
    if "fault_adjudication" not in report.columns():
        if (
            "gpu_completed_launches" not in report.columns()
            or report.height() == 0
        ):
            return "unknown"
        var launched = False
        var host = False
        for row in range(report.height()):
            if report.item(row, "device_status").string() == "unverified":
                return "unknown"
            launched |= report.item(row, "gpu_completed_launches").int64() > 0
            host |= (
                report.item(row, "device_status").string() == "no_device_work"
            )
        return (
            "gpu_observed" if launched else "no_device_work" if host else "cpu_only"
        )
    if report.height() < 1:
        return "unknown"
    for row in range(report.height()):
        if report.item(row, "fault_adjudication").string() != "cpu":
            return "unknown"
    return "cpu"


def _record(report: DataFrame):
    var status: String
    try:
        status = _adjudication(report)
        for row in range(report.height()):
            if (
                report.item(row, "node").int64()
                == report.item(row, "pipeline").int64()
            ):
                print("runtime_pipeline\t" + pipeline_record_json(report, row))
    except error:
        status = "unknown"
    print("runtime_audit\t" + status)


@fieldwise_init
struct CoverageAuditRuntime(AcceleratorBackend):
    """Delegate to MetalRuntime and audit its returned report per collect."""

    var _inner: MetalRuntime

    def __init__(out self) raises:
        self._inner = MetalRuntime()

    def execute(self, plan: LazyFrame) raises -> Tuple[DataFrame, DataFrame]:
        var result: Tuple[DataFrame, DataFrame]
        try:
            result = self._inner.execute_profiled(plan)
        except error:
            print("runtime_audit\tfailed")
            raise error^
        _record(result[1])
        return result^

    def execute_profiled(
        self, plan: LazyFrame
    ) raises -> Tuple[DataFrame, DataFrame]:
        var result: Tuple[DataFrame, DataFrame]
        try:
            result = self._inner.execute_profiled(plan)
        except error:
            print("runtime_audit\tfailed")
            raise error^
        _record(result[1])
        return result^

    def describe(self, plan: LazyFrame) -> String:
        return self._inner.describe(plan)
