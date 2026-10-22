"""Coverage-only wrapper that preserves Metal execution and reports adjudication."""
from dataframe import DataFrame
from dataframe.lazy import AcceleratorBackend, LazyFrame
from dataframe_metal.metal import MetalRuntime


def _adjudication(report: DataFrame) raises -> String:
    """Classify the existing report marker without changing execution."""
    if "fault_adjudication" not in report.columns():
        return "no_marker"
    if report.height() < 1:
        return "unknown"
    for row in range(report.height()):
        if report.item(row, "fault_adjudication").string() != "cpu":
            return "unknown"
    return "cpu"


def _record(report: DataFrame):
    var status = "unknown"
    try:
        status = _adjudication(report)
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
            result = self._inner.execute(plan)
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
