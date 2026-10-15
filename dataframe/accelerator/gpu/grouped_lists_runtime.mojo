"""Public grouped primitive Lists retain their GPU descriptors between phases."""
from std.time import perf_counter_ns
from dataframe.parallel import configured_workers
from dataframe import Column, DataFrame, Series
from dataframe.lazy import LazyFrame
from dataframe.execution_report import ExecutionReport
from dataframe.accelerator.grouped_lists import prepare
from dataframe.accelerator.grouped_key_queries import prepare_keys
from .residency import Scope, Table, checked_add, checked_mul
from .runtime import execute_plan, materialize, _width
from .device_lists import (
    DeviceLists,
    grouped_implode,
    empty_grouped,
    export,
    consume,
)
from .list_groups import build_groups
from .grouped_phase_groups import PhaseGroups
from .grouped_feed_runtime import whole_geometry, execute as execute_feed
from .grouped_schedule import build as build_schedule, read_phases, gather_phase
from .grouped_list_targets import (
    ListTarget,
    build_destinations,
    create_target,
    append_phase,
    finish_target,
)


def _column(source: Table, name: String) raises -> Table:
    var result = source.copy()
    for index in range(source.schema.width()):
        if source.schema._columns[index].name() == name:
            result.schema = DataFrame([source.schema._columns[index].copy()])
            if (
                not source.rows
                and not source.storage
                and not source.identity_storage
            ):
                return Table.empty(result.schema.copy())
            result.columns = [source.columns[index]]
            return result^
    raise Error("Grouped IMPLODE output column is missing")


def _host_bytes(source: Table) raises -> Int:
    var result = 0
    for field in source.schema._columns:
        result = checked_add(
            result,
            checked_add(
                checked_mul(source.rows, _width(field.dtype())),
                (source.rows + 7) // 8,
            ),
        )
    return result


def _execute_primitive(
    query: LazyFrame,
    device: Int,
    budget: Int,
    profiling: Bool,
    *,
    observation_only: Bool = False,
    ordered_source: Bool = False,
    eager_feed: Bool = False,
) raises -> Tuple[DataFrame, DataFrame]:
    # All schemas, supported operations and producer plans bind before context.
    var plan = prepare(
        query, ordered_source=ordered_source, eager_feed=eager_feed
    )
    var feed_producers = 0
    for descriptor in plan.feed_plans:
        feed_producers += Int(Bool(descriptor))
    var started = Int(perf_counter_ns())
    var scope = Scope(device, budget)
    var source_result = execute_plan(
        plan.source, scope, profiling=profiling, export_values=False
    )
    var source = source_result[0].copy()
    plan.scan._record_execution()
    var layout = build_groups(source, plan.names, scope)
    var key_result = execute_plan(
        plan.keys,
        scope,
        Optional(source.copy()),
        profiling,
        export_values=False,
    )
    var keys = key_result[0].copy()
    keys.observe_errors()
    var lists = List[DeviceLists]()
    var scalars = List[Table]()
    var indices = List[Int]()
    if plan.scheduled:
        var schedule = build_schedule(
            source,
            plan.names,
            scope,
            configured_workers(),
            provenance=plan.key_provenance,
        )
        var phases = read_phases(schedule, scope)
        var destinations = build_destinations(layout.value(), scope)
        var targets = List[ListTarget]()
        for index in range(len(plan.producers)):
            if plan.list_outputs[index]:
                ref descriptor = plan.producers[index][].outputs[0]
                var child = Series.full_null(
                    descriptor.name, descriptor.dtype, 0
                )
                child._field_metadata = descriptor.field_metadata
                indices.append(len(targets))
                targets.append(
                    create_target(
                        destinations, schedule, child, scope, child.name()
                    )
                )
            else:
                # Admission excludes input/feed/state/association faults on
                # this path. Exact infallible scalar siblings can be computed
                # independently, while checked List operands obey CPU phases.
                var produced = execute_plan(
                    plan.producers[index][],
                    scope,
                    Optional(source.copy()),
                    profiling,
                    export_values=False,
                )
                var output = _column(
                    produced[0],
                    plan.grouped_schema._columns[
                        len(plan.names) + index
                    ].name(),
                )
                output.observe_errors()
                indices.append(len(scalars))
                scalars.append(output^)
        # CPU job priority precedes projection priority: heavy buckets first,
        # then light buckets, each in ascending bucket order. A private target
        # only publishes after every successful phase filled its stable slots.
        for phase in range(Int(schedule[].facts[4])):
            var bucket = gather_phase(source, schedule, phases, phase, scope)
            for index in range(len(plan.producers)):
                if not plan.list_outputs[index]:
                    continue
                var produced = execute_plan(
                    plan.producers[index][],
                    scope,
                    Optional(bucket.copy()),
                    profiling,
                    export_values=False,
                )
                var output = _column(
                    produced[0],
                    plan.grouped_schema._columns[
                        len(plan.names) + index
                    ].name(),
                )
                append_phase(
                    targets[indices[index]], output, phases, phase, scope
                )
        # Pop in reverse, then restore original List output order.
        var reversed = List[DeviceLists]()
        while len(targets):
            reversed.append(finish_target(targets.pop(), scope))
        while len(reversed):
            lists.append(reversed.pop())
    else:
        var geometry = Optional[PhaseGroups]()
        var feed_count = 0
        for descriptor in plan.feed_plans:
            feed_count += Int(Bool(descriptor))
        if feed_count and layout:
            var destinations = build_destinations(layout.value(), scope)
            geometry = whole_geometry(destinations, scope)
        for index in range(len(plan.producers)):
            if plan.feed_plans[index]:
                var output = execute_feed(
                    plan.feed_plans[index].value(),
                    source,
                    geometry.value(),
                    scope,
                ) if geometry else Table.empty(
                    DataFrame([plan.feed_plans[index].value().prototype.copy()])
                )
                indices.append(len(scalars))
                scalars.append(output^)
                continue
            var produced = execute_plan(
                plan.producers[index][],
                scope,
                Optional(source.copy()),
                profiling,
                export_values=False,
            )
            var output = _column(
                produced[0],
                plan.grouped_schema._columns[len(plan.names) + index].name(),
            )
            # CPU serial grouped route completes each original aggregate before
            # preparing the next one's states/operand. No multi-output region can
            # prepare a later reduction ahead of an earlier checked List operand.
            if plan.list_outputs[index]:
                var value = grouped_implode(
                    output,
                    layout.value(),
                    scope,
                    output.schema._columns[0].name(),
                ) if layout else empty_grouped(
                    output.schema._columns[0],
                    scope,
                    output.schema._columns[0].name(),
                )
                indices.append(len(lists))
                lists.append(value^)
            else:
                output.observe_errors()
                indices.append(len(scalars))
                scalars.append(output.copy())
            _ = output^
            _ = produced^
    # Lists cross the aggregate -> row-consumer boundary as device owners.
    # No offsets or child values are exported/reuploaded for len/get.
    var final_tables = List[Table]()
    var final_lists = List[DeviceLists]()
    var final_indices = List[Int]()
    var final_kinds = List[Bool]()
    for request in plan.consumers:
        if request.kind:
            var producer = request.source - len(plan.names)
            var value = lists[indices[producer]].copy()
            if request.kind == 1:
                value.prototype = value.prototype.renamed(request.name)
                final_indices.append(len(final_lists))
                final_lists.append(value^)
                final_kinds.append(True)
            else:
                var output = consume[True](
                    value, scope, request.name
                ) if request.kind == 2 else consume[False](
                    value, scope, request.name, request.index
                )
                output.observe_errors()
                final_indices.append(len(final_tables))
                final_tables.append(output^)
                final_kinds.append(False)
        else:
            var table = (
                _column(
                    keys, plan.grouped_schema._columns[request.source].name()
                ) if request.source
                < len(plan.names) else scalars[
                    indices[request.source - len(plan.names)]
                ].copy()
            )
            table.schema = DataFrame(
                [table.schema._columns[0].renamed(request.name)]
            )
            final_indices.append(len(final_tables))
            final_tables.append(table^)
            final_kinds.append(False)
    for slice in plan.slices:
        if slice.source_failure:
            raise Error(slice.source_failure)
        if slice.length < -1:
            raise Error("Slice length must be nonnegative")
    var host_reserved = 0
    var host_result_bytes = 0
    var result = plan.schema.clear()
    try:
        if not observation_only:
            var columns = List[Series]()
            for index in range(len(plan.consumers)):
                if final_kinds[index]:
                    ref value = final_lists[final_indices[index]]
                    var output = export(value, scope)
                    var bytes = checked_add(
                        checked_add(
                            checked_mul(value.rows + 1, 8),
                            (value.rows + 7) // 8,
                        ),
                        _host_bytes(value.child),
                    ) if value.rows else 0
                    scope._retain(bytes)
                    host_reserved = checked_add(host_reserved, bytes)
                    host_result_bytes = checked_add(host_result_bytes, bytes)
                    output._field_metadata = plan.schema._columns[
                        index
                    ]._field_metadata
                    columns.append(output^)
                else:
                    ref table = final_tables[final_indices[index]]
                    var output = materialize(table)
                    var bytes = _host_bytes(table)
                    scope._retain(bytes)
                    host_reserved = checked_add(host_reserved, bytes)
                    host_result_bytes = checked_add(host_result_bytes, bytes)
                    columns.append(output._columns[0].copy())
            result = DataFrame(columns^, height=keys.rows)
            for slice in plan.slices:
                result = result.slice(slice.offset, slice.length)
        scope._release(host_reserved)
    except error:
        scope.synchronize()
        scope._release(host_reserved)
        _ = source^
        _ = layout^
        _ = keys^
        _ = lists^
        _ = scalars^
        _ = final_tables^
        _ = final_lists^
        raise error^
    var traffic = scope.transfers()
    var peak = scope.info()[1]
    var report = ExecutionReport()
    report.record(
        plan.root,
        "GROUPED IMPLODE ERROR OBSERVATION" if observation_only else "GROUPED IMPLODE",
        "metal",
        plan.rows,
        keys.rows,
        algorithm="mojo_gpu_resident_grouped_lists",
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
            ("resident_list_count", Int64(len(lists))),
            (
                "resident_list_exports",
                Int64(0 if observation_only else len(final_lists)),
            ),
            ("list_descriptor_upload_bytes", Int64(0)),
            ("grouped_cpu_bucket_schedule", Int64(plan.scheduled)),
            ("grouped_checked_feed_producers", Int64(feed_producers)),
        ]
    ):
        fields.append(Series(field[0], Column[Int64]([field[1]])))
    fields.append(Series("runtime_implementation", Column[String](["mojo"])))
    _ = source^
    _ = layout^
    _ = keys^
    _ = lists^
    _ = scalars^
    _ = final_tables^
    _ = final_lists^
    return (result^, DataFrame(fields^))


def execute(
    query: LazyFrame,
    device: Int,
    budget: Int,
    profiling: Bool,
    *,
    observation_only: Bool = False,
    ordered_source: Bool = False,
    eager_feed: Bool = False,
) raises -> Tuple[DataFrame, DataFrame]:
    # Original schemas/admission bind before key values are staged. Only key
    # carriers change: checked numeric payload occurrence proofs remain exact.
    var staged = prepare_keys(
        query,
        memory_limit_bytes=budget,
        ordered_source=ordered_source,
        eager_feed=eager_feed,
    )
    var remaining = budget
    if remaining >= 0:
        remaining -= staged.staging_bytes
    var executed = _execute_primitive(
        staged.query,
        device,
        remaining,
        profiling,
        observation_only=observation_only,
        ordered_source=ordered_source,
        eager_feed=eager_feed,
    )
    var report = executed[1].copy()
    var inner_peak = Int(
        report.column("query_peak_requested_bytes").int64().value(0)
    )
    var restoration = 0
    var result = staged.schema.copy()
    if not observation_only:
        # The encoded result remains alive while typed key owners are built.
        # Core reports full retained export backing even after trailing slices.
        restoration = Int(
            report.column("query_host_result_bytes").int64().value(0)
        )
        for index in range(len(staged.key_outputs)):
            var source = staged.key_outputs[index]
            if source >= 0:
                restoration = checked_add(
                    restoration,
                    staged.domains[source].restoration_bytes(
                        executed[0]._columns[index]
                    ),
                )
        if remaining >= 0 and restoration > remaining:
            raise Error("Metal grouped key restoration exceeds memory budget")
        var fields = List[Series]()
        for index in range(len(staged.key_outputs)):
            var source = staged.key_outputs[index]
            var output = (
                staged.domains[source].restore(
                    executed[0]._columns[index]
                ) if source
                >= 0 else executed[0]._columns[index].copy()
            )
            output._field_metadata = staged.schema._columns[
                index
            ]._field_metadata
            fields.append(output^)
        result = DataFrame(fields^, height=executed[0].height())
    var peak = checked_add(staged.staging_bytes, max(inner_peak, restoration))
    for name in List[String](
        ["peak_requested_bytes", "query_peak_requested_bytes"]
    ):
        report = report.with_column(Series(name, Column[Int64]([Int64(peak)])))
    for field in List[Tuple[String, Int64]](
        [
            ("grouped_key_staging_bytes", Int64(staged.staging_bytes)),
            ("grouped_key_restoration_bytes", Int64(restoration)),
        ]
    ):
        report = report.with_column(Series(field[0], Column[Int64]([field[1]])))
    return (result^, report^)
