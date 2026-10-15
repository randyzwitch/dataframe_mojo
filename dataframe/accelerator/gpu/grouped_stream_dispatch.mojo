"""Original-route dispatch for ordered grouped basic floating reductions."""
from dataframe import Column, DataFrame, Series
from dataframe.execution_report import ExecutionReport
from dataframe.accelerator.metadata import schema_frame
from dataframe.accelerator.grouped_stream_routes import (
    BASIC,
    StreamRoute,
)
from .grouped_float_stream_query import execute_candidate as execute_basic


def execute(
    route: StreamRoute,
    device: Int,
    budget: Int,
    batch_size: Int,
    *,
    profiling: Bool = False,
    observation_only: Bool = False,
    started: Int = 0,
) raises -> Tuple[DataFrame, DataFrame]:
    if budget < -1:
        raise Error("Grouped float stream memory budget is invalid")
    if not route.rows:
        # A grouped empty frame has no groups. Preserve the bound schema and
        # avoid device discovery, allocation, source upload and sampling.
        var result = schema_frame(route.query, 1).clear()
        var report = ExecutionReport()
        report.record(
            1,
            "GROUPED FLOAT STREAM",
            "metal",
            0,
            0,
            algorithm="mojo_gpu_single_job_float_epochs",
        )
        var fields = report.frame()._columns.copy()
        for name in [
            "kernel_launches",
            "synchronizations",
            "upload_bytes",
            "download_bytes",
            "peak_requested_bytes",
            "host_result_bytes",
            "query_kernel_launches",
            "query_synchronizations",
            "query_upload_bytes",
            "query_download_bytes",
            "query_peak_requested_bytes",
            "query_host_result_bytes",
        ]:
            fields.append(Series(name, Column[Int64]([Int64(0)])))
        return (result^, DataFrame(fields^))
    if route.family == BASIC:
        return execute_basic(
            route.query,
            device,
            budget,
            route.jobs,
            batch_size,
            UInt64(0x7FF8000000000000),
            profiling=profiling,
            original_source=True,
            streaming=True,
            observation_only=observation_only,
            captured_ranges=route.ranges,
            captured_flush_groups=route.flush_groups,
            captured_peak=route.capture_peak,
            started=started,
        )
    raise Error("Grouped float stream state family changed after capture")
