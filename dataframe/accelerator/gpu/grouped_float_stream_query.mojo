"""Original ordered terminal floating reductions over certified CPU epochs."""
from std.collections import Optional
from std.time import perf_counter_ns
from dataframe import Column, DataFrame, Series
from dataframe.lazy import LazyFrame
from dataframe.execution_report import ExecutionReport
from dataframe.accelerator.grouped_float_stream import prepare_query
from .residency import Scope, Table, checked_add, checked_mul
from .runtime import _preflight, execute_plan, materialize
from .grouped_float_geometry import build_groups
from .grouped_float_geometry import build_destinations
from .grouped_float_stream_runtime import (
    execute as execute_float,
    whole_geometry,
    uses_insert,
)
from .grouped_float_geometry import _host_bytes
from .grouped_float_geometry import _Reservation


def execute_candidate(
    query: LazyFrame,
    device: Int,
    budget: Int,
    workers: Int,
    batch_size: Int,
    canonical_nan: UInt64,
    *,
    profiling: Bool = False,
    original_source: Bool = False,
    streaming: Bool = True,
    observation_only: Bool = False,
    mut captured_ranges: List[UInt32],
    captured_flush_groups: Int = -1,
    captured_peak: Int = 0,
    started: Int = 0,
) raises -> Tuple[DataFrame, DataFrame]:
    var began = started if started > 0 else Int(perf_counter_ns())
    var scope = Scope(device, budget)
    _ = scope.check(captured_peak)
    # Keep the immutable source receipt. Per-feed copies are deferred until
    # the device confirms the inserted-table route actually needs them.
    var captured_bytes = checked_mul(captured_ranges.capacity(), 4)
    var captured_owner = _Reservation(scope, captured_bytes)
    var plan = prepare_query(
        query,
        workers,
        batch_size,
        original_source=original_source,
        streaming=streaming,
        captured_ranges=captured_ranges,
        captured_flush_groups=captured_flush_groups,
        defer_ranges=len(captured_ranges) > 0,
    )
    # Bind the ordinary collected GPU path before any device submission.
    var collected = _preflight(
        plan.source.source.lazy()
        .group_by(plan.names, maintain_order=True)
        .agg(query._nodes[len(query._nodes) - 1].exprs),
        source_rows=plan.rows,
        ordered_source=True,
    )
    var source_result = execute_plan(
        plan.source, scope, profiling=profiling, export_values=False
    )
    var source = source_result[0].copy()
    _ = source_result^
    plan.scan._record_execution()
    source.observe_errors()
    if not uses_insert(source, 0, scope):
        # Reuse the admitted main GPU implementation in the same Scope.
        # This preserves its behavior without asserting CPU-exact collected
        # floating association. Probe/capture/source costs remain visible.
        # The sample and source observation have synchronized. End their
        # transport/receipt lifetimes before the generic matrix is allocated,
        # preserving main's tight budget instead of retaining both matrices.
        # Reupload the immutable source through main's GPU executor; its
        # traffic and the probe peak remain charged in this same Scope.
        _ = source^
        captured_ranges = List[UInt32]()
        _ = captured_owner^
        var executed = execute_plan(
            collected,
            scope,
            None,
            profiling,
            export_values=not observation_only,
        )
        var resident = executed[0].copy()
        _ = executed^
        var result = plan.schema.clear()
        var host_bytes = 0
        if observation_only:
            resident.observe_errors()
        else:
            host_bytes = _host_bytes(resident)
            result = materialize(resident)
        var report = _report(
            scope,
            plan.rows,
            resident.rows,
            host_bytes,
            began,
            "mojo_gpu_collected_float_groups",
            observation_only,
        )
        return (result^, report^)
    # Feed copies are now needed. Reserve before constructing any of them.
    var feed_bytes = checked_mul(
        checked_mul(len(plan.feeds), len(captured_ranges)), 4
    )
    var feed_owner = _Reservation(scope, feed_bytes)
    if len(captured_ranges):
        for at in range(len(plan.feeds)):
            plan.feeds[at].ranges = captured_ranges.copy()
    var layout = build_groups(source, plan.names, scope)
    var destinations = build_destinations(layout.value(), scope)
    var geometry = whole_geometry(destinations, scope)
    var key_result = execute_plan(
        plan.keys,
        scope,
        Optional(source.copy()),
        profiling,
        export_values=False,
    )
    var keys = key_result[0].copy()
    keys.observe_errors()
    var outputs = List[Table]()
    for feed in plan.feeds:
        outputs.append(
            execute_float(feed, source, geometry, scope, canonical_nan)
        )
    var result = plan.schema.clear()
    var host_reserved = 0
    try:
        if not observation_only:
            var columns = List[Series]()
            var key_bytes = _host_bytes(keys)
            scope._retain(key_bytes)
            host_reserved += key_bytes
            var key_frame = materialize(keys)
            columns.append(key_frame._columns[0].copy())
            for output in outputs:
                var bytes = _host_bytes(output)
                scope._retain(bytes)
                host_reserved += bytes
                var frame = materialize(output)
                columns.append(frame._columns[0].copy())
            result = DataFrame(columns^, height=geometry.groups)
        scope._release(host_reserved)
    except error:
        scope.synchronize()
        scope._release(host_reserved)
        raise error^
    var report = _report(
        scope,
        plan.rows,
        geometry.groups,
        host_reserved,
        began,
        "mojo_gpu_single_worker_float_epochs",
        observation_only,
    )
    _ = feed_owner^
    _ = captured_owner^
    return (result^, report^)


def _report(
    scope: Scope,
    rows: Int,
    groups: Int,
    host_result_bytes: Int,
    started: Int,
    algorithm: String,
    observation_only: Bool,
) raises -> DataFrame:
    var traffic = scope.transfers()
    var peak = scope.info()[1]
    var report = ExecutionReport()
    report.record(
        1,
        "GROUPED FLOAT ERROR OBSERVATION" if observation_only else "GROUPED FLOAT",
        "metal",
        rows,
        groups,
        algorithm=algorithm,
        wall_ns=Int(perf_counter_ns()) - started,
    )
    var fields = report.frame()._columns.copy()
    for field in List[Tuple[String, Int64]](
        [
            ("kernel_launches", Int64(traffic[2])),
            ("synchronizations", Int64(traffic[3])),
            ("upload_bytes", Int64(traffic[0])),
            ("download_bytes", Int64(traffic[1])),
            ("peak_requested_bytes", Int64(peak)),
            ("host_result_bytes", Int64(host_result_bytes)),
            ("query_kernel_launches", Int64(traffic[2])),
            ("query_synchronizations", Int64(traffic[3])),
            ("query_upload_bytes", Int64(traffic[0])),
            ("query_download_bytes", Int64(traffic[1])),
            ("query_peak_requested_bytes", Int64(peak)),
            ("query_host_result_bytes", Int64(host_result_bytes)),
        ]
    ):
        fields.append(Series(field[0], Column[Int64]([field[1]])))
    fields.append(
        Series(
            "query_wall_ms",
            Column[Float64]([Float64(Int(perf_counter_ns()) - started) / 1e6]),
        )
    )
    return DataFrame(fields^)
