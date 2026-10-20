"""Direct Mojo execution of shared join and scalar-consumer region DAGs.

Every region is metadata-bound before device initialization. Computed regions
share immutable Mojo-owned device tables and copy selected input columns on
the device. Original scans stage on first consumption; only the final table
exports CPU values. Shared owner accounting includes all live intermediates,
region scratch and the final host export before allocation.
"""
from dataframe.accelerator import decimal_sum_receipts

from std.time import perf_counter_ns
from std.sys import size_of
from std.memory import ArcPointer, Pointer
from dataframe.accelerator.code_domains import TextQuery, TextField
from .residency import TextDomain, TextColumn
from dataframe.accelerator.code_domains import TextField
from .text_domains import union_domain, source_maps, MapPlan, ColumnMap
from .text_programs import recode_region
from dataframe import Column, DataFrame, DataType, Series
from dataframe.lazy import LazyFrame
from dataframe.execution_report import ExecutionReport
from dataframe.accelerator.explodes import schema as explode_schema
from dataframe.accelerator.cuts import control_failure as cut_control_failure
from dataframe.accelerator.capabilities import RowCapabilities
from dataframe.accelerator.joins import (
    JoinDag,
    DAG_JOIN,
    DAG_ASOF,
    DAG_ROWS,
    DAG_SOURCE_FIELDS,
    DAG_EXPLODE,
    DAG_GROUPED_FIELDS,
    lower_join_dag,
)
from . import runtime as rows
from . import qcut_runtime as qcuts
from dataframe.accelerator.qcuts import is_qcut_marker
from dataframe.accelerator.partition_cuts import (
    is_marker as is_partition_cut_marker,
)
from . import partition_runtime as partition_cuts
from .temporal_descriptors import ROW_TEMPORAL_BINARY, ROW_TEMPORAL_CAST
from . import join_runtime as joins
from . import asof_runtime as asof
from .query_scope import DeviceService, Scope
from .residency import Table
from .resident_fields import ResidentFrame
from . import source_fields
from .producer_phases_runtime import MetadataReservation
from . import explode
from .grouped_lists_producer import produce as produce_grouped_fields
from .grouped_key_owner import GroupedKeyOwner, staging_bytes
from .device_lists import HostReservation


def preflight(
    query: LazyFrame,
    capabilities: RowCapabilities,
    *,
    ordered_source: Bool = False,
    eager_feed: Bool = False,
) raises -> JoinDag:
    """Check every reachable task and operation without device initialization."""
    var semantic = capabilities.copy()
    semantic.scalar_regions = True
    var dag = lower_join_dag(
        query, semantic, ordered_source=ordered_source, eager_feed=eager_feed
    )
    for task in dag.tasks:
        if task.kind == DAG_JOIN:
            _ = joins.preflight(
                task,
                dag.tasks[task.source].schema,
                dag.tasks[task.right].schema,
            )
        elif task.kind == DAG_ASOF:
            _ = asof.preflight(
                task,
                dag.tasks[task.source].schema,
                dag.tasks[task.right].schema,
            )
        elif task.kind == DAG_GROUPED_FIELDS:
            if not task.grouped:
                raise Error("Resident grouped task prepared owner is missing")
        elif task.kind == DAG_SOURCE_FIELDS:
            source_fields.preflight(task.query._frames[0])
        elif task.kind == DAG_EXPLODE:
            _ = explode_schema(
                dag.tasks[task.source].schema,
                task.query._nodes[len(task.query._nodes) - 1].names,
                semantic,
            )
        elif task.kind == DAG_ROWS:
            if is_partition_cut_marker(task.query):
                _ = partition_cuts.preflight(task.query)
                continue
            if is_qcut_marker(task.query):
                _ = qcuts.preflight(task.query)
                continue
            if task.sum_receipt:
                decimal_sum_receipts.validate(
                    task.sum_receipt.value()[],
                    task.source,
                    dag.tasks[task.source].query,
                    task.query,
                )
            var plan = rows._preflight(
                task.query,
                terminal_boundary=task.terminal,
                sum_receipt=task.sum_receipt,
            )
            if plan.ordered_mode == 6 and (
                not plan.producer_region
                or plan.producer_region.value()[].source_task != task.source
            ):
                raise Error(
                    "Observed temporal producer planned input lineage differs"
                )
            if task.source >= 0:
                if plan.ordered_mode == 2:
                    raise Error(
                        "Metal unsupported [ordered rows]: resident producer provenance is not proved"
                    )
                for node in range(len(plan.node_dtypes)):
                    var op = plan.code[4 * node]
                    if (
                        (op == ROW_TEMPORAL_BINARY or op == ROW_TEMPORAL_CAST)
                        and plan.ordered_mode != 3
                        and not capabilities.adjudicated_faults
                    ):
                        raise Error(
                            "Metal unsupported [temporal]: prototype requires a direct scan before DAG submission"
                        )
        else:
            raise Error("Mojo GPU unexpected execution DAG task")
    return dag^


def _frame(
    dag: JoinDag, values: List[Optional[Table]], index: Int
) raises -> DataFrame:
    if values[index]:
        return values[index].value().schema.copy()
    ref source = dag.tasks[index]
    if (
        source.kind != DAG_ROWS
        or source.source >= 0
        or len(source.query._nodes) != 1
    ):
        raise Error("Mojo GPU resident DAG input has been released")
    return source.query._frames[0].copy()


def _int_column(
    report: DataFrame, name: String, value: Int
) raises -> DataFrame:
    return report.with_column(
        Series(
            name,
            Column[Int64](
                List[Int64](length=report.height(), fill=Int64(value))
            ),
        )
    )


def _float_column(
    report: DataFrame, name: String, value: Float64
) raises -> DataFrame:
    return report.with_column(
        Series(
            name,
            Column[Float64](List[Float64](length=report.height(), fill=value)),
        )
    )


def _profile_phases(report: DataFrame) raises -> DataFrame:
    """Keep measured row timings when other region types have no attribution."""
    var result = report.copy()
    for name in List[String](
        [
            "host_prepare_ms",
            "staging_copy_ms",
            "initialization_ms",
            "submit_wait_ms",
            "synchronization_ms",
            "readback_ms",
            "result_copy_ms",
        ]
    ):
        if name not in result.columns():
            result = result.with_column(
                Series.full_null(name, DataType.FLOAT64, result.height())
            )
    return result^


def _storage_report(
    root: Int,
    operator: String,
    algorithm: String,
    input_rows: Int,
    output_rows: Int,
    scope: Scope,
    started: Int,
    input_bytes: Int,
    output_bytes: Int,
) raises -> DataFrame:
    var report = ExecutionReport()
    report.record(
        root,
        operator,
        "metal",
        input_rows,
        output_rows,
        algorithm=algorithm,
        wall_ns=Int(perf_counter_ns()) - started,
    )
    var result = report.frame()
    for entry in List[Tuple[String, Int]](
        [
            ("kernel_launches", 0),
            ("synchronizations", 0),
            ("upload_bytes", 0),
            ("download_bytes", 0),
            ("peak_requested_bytes", scope.info()[1]),
            ("host_result_bytes", 0),
            ("resident_input_bytes", input_bytes),
            ("resident_output_bytes", output_bytes),
        ]
    ):
        result = _int_column(result, entry[0], entry[1])
    result = result.with_column(
        Series("runtime_implementation", Column[String](["mojo"]))
    )
    result = result.with_column(
        Series(
            "query_wall_ms",
            Column[Float64]([Float64(Int(perf_counter_ns()) - started) / 1e6]),
        )
    )
    return result^


def execute(
    query: LazyFrame,
    device_id: Int,
    budget: Int,
    profiling: Bool,
    capabilities: RowCapabilities,
    *,
    observation_only: Bool = False,
    text_context: Optional[ArcPointer[TextQuery]] = None,
    ordered_source: Bool = False,
    eager_feed: Bool = False,
    device_service: Optional[DeviceService] = None,
) raises -> Tuple[DataFrame, DataFrame]:
    var started = Int(perf_counter_ns())
    var dag = preflight(
        query,
        capabilities,
        ordered_source=ordered_source,
        eager_feed=eager_feed,
    )
    var preflight_ns = Int(perf_counter_ns()) - started if profiling else 0
    var scope = Scope(
        device_id,
        budget,
        profiling=profiling,
        service=device_service,
    )
    var producer_metadata = MetadataReservation(scope, dag)
    var partition_context = ArcPointer(partition_cuts.StateContext())
    var base_domain = Optional[ArcPointer[TextDomain]]()
    if text_context:
        base_domain = ArcPointer(
            TextDomain(scope, text_context.value()[].base_domain, False)
        )
    var uses = List[Int](length=len(dag.tasks), fill=0)
    for task in dag.tasks:
        if task.source >= 0:
            uses[task.source] += 1
        if task.kind == DAG_JOIN or task.kind == DAG_ASOF:
            uses[task.right] += 1
    var values = List[Optional[Table]](capacity=len(dag.tasks))
    # A nested source is a persistent descriptor + child owner, never a
    # fictional primitive matrix slot or an intermediate CPU List column.
    var nested = List[Optional[ResidentFrame]](capacity=len(dag.tasks))
    var report = DataFrame(List[Series]())
    var result = DataFrame(List[Series]())
    var query_host_result_bytes = 0
    var service_drain_done = False
    try:
        for index in range(len(dag.tasks)):
            var task = dag.tasks[index].copy()
            if (
                task.kind == DAG_ROWS
                and task.source < 0
                and len(task.query._nodes) == 1
                and (not text_context or index != dag.root)
            ):
                # Share original immutable scan buffers; no data evaluation or
                # device staging occurs until the first computed consumer.
                if task.query._nodes[0].source_failure:
                    raise Error(task.query._nodes[0].source_failure)
                values.append(None)
                nested.append(None)
                if index == dag.root:
                    result = task.schema.copy() if observation_only else task.query._frames[
                        0
                    ].copy()
                continue
            if (
                task.kind == DAG_ROWS
                or task.kind == DAG_SOURCE_FIELDS
                or task.kind == DAG_GROUPED_FIELDS
            ) and task.query._nodes[0].source_failure:
                raise Error(task.query._nodes[0].source_failure)
            var live_before = scope.info()[0]
            var traffic_before = scope.transfers()
            var region_sync_before = scope.timing_ns()[1]
            var region_started = Int(perf_counter_ns())
            if (
                task.kind == DAG_SOURCE_FIELDS
                or task.kind == DAG_GROUPED_FIELDS
            ):
                var fields: ResidentFrame
                var input_rows: Int
                if task.kind == DAG_GROUPED_FIELDS:
                    ref prepared = task.grouped.value()[]
                    var staging = Optional[ArcPointer[HostReservation]]()
                    if prepared.original_keys:
                        staging = ArcPointer(
                            HostReservation(
                                scope,
                                staging_bytes(prepared.original_keys.value()[]),
                            )
                        )
                    var source = rows.execute_plan(
                        prepared.source,
                        scope,
                        profiling=profiling,
                        export_values=False,
                        text_domain=base_domain,
                    )
                    var typed = Optional[ArcPointer[GroupedKeyOwner]]()
                    if prepared.original_keys:
                        typed = ArcPointer(
                            GroupedKeyOwner(
                                prepared.original_keys.value(),
                                task.grouped.value(),
                                source[0],
                                scope,
                                staging.value(),
                            )
                        )
                    prepared.scan._record_execution()
                    fields = produce_grouped_fields(
                        prepared, source[0], scope, profiling, typed=typed
                    )
                    input_rows = prepared.rows
                    _ = source^
                else:
                    task.query._nodes[0]._record_execution()
                    fields = source_fields.stage(task.query._frames[0], scope)
                    input_rows = fields.rows
                var owned = scope.info()[0] - live_before
                var region = _storage_report(
                    task.root,
                    "GROUPED FIELDS" if task.kind
                    == DAG_GROUPED_FIELDS else "SOURCE FIELDS",
                    "mojo_gpu_grouped_list_producer" if task.kind
                    == DAG_GROUPED_FIELDS else "mojo_gpu_source_fields",
                    input_rows,
                    fields.rows,
                    scope,
                    region_started,
                    0,
                    owned,
                )
                if device_service and index == dag.root:
                    # Lowering normally emits source/grouped fields as producer
                    # tasks before EXPLODE; if terminal directly, keep its drain
                    # in this region's timing and synchronization counters.
                    scope.synchronize()
                    service_drain_done = True
                var after = scope.transfers()
                region = _int_column(
                    region, "upload_bytes", after[0] - traffic_before[0]
                )
                region = _int_column(
                    region, "download_bytes", after[1] - traffic_before[1]
                )
                region = _int_column(
                    region, "kernel_launches", after[2] - traffic_before[2]
                )
                region = _int_column(
                    region, "synchronizations", after[3] - traffic_before[3]
                )
                region = _int_column(region, "node", task.root)
                region = _int_column(
                    region, "retained_intermediate_bytes", live_before
                )
                region = _int_column(region, "intermediate_output_bytes", owned)
                region = region.with_column(
                    Series(
                        "materialization_boundary",
                        Column[String](["Mojo GPU resident fields"]),
                    )
                )
                if profiling:
                    region = _profile_phases(region)
                    region = _float_column(
                        region,
                        "synchronization_ms",
                        Float64(scope.timing_ns()[1] - region_sync_before)
                        / 1e6,
                    )
                report = region^ if report.width() == 0 else report.vstack(
                    region.select(report.columns())
                )
                values.append(None)
                nested.append(fields^)
                continue
            var executed: Tuple[Table, DataFrame]
            var target_domain = Optional[ArcPointer[TextDomain]]()
            var join_maps = Optional[MapPlan]()
            var join_source_text = List[TextField]()
            if base_domain:
                var input_fields = 0
                if task.source >= 0 and values[task.source]:
                    input_fields += len(values[task.source].value().text)
                if (task.kind == DAG_JOIN or task.kind == DAG_ASOF) and values[
                    task.right
                ]:
                    input_fields += len(values[task.right].value().text)
                var text_workspace = rows._checked_mul(
                    input_fields, size_of[Optional[TextColumn]]()
                )
                _ = scope.check(text_workspace)
                var input_text = List[Optional[TextColumn]](
                    capacity=input_fields
                )
                if task.source >= 0 and values[task.source]:
                    input_text.extend(values[task.source].value().text.copy())
                if (task.kind == DAG_JOIN or task.kind == DAG_ASOF) and values[
                    task.right
                ]:
                    input_text.extend(values[task.right].value().text.copy())
                target_domain = union_domain(
                    scope, base_domain.value(), input_text, text_workspace
                )
                recode_region(task.query, target_domain.value())
                if task.kind == DAG_JOIN or task.kind == DAG_ASOF:
                    var left = _frame(dag, values, task.source)
                    var right = _frame(dag, values, task.right)
                    ref left_node = dag.tasks[task.source].query._nodes[
                        len(dag.tasks[task.source].query._nodes) - 1
                    ]
                    ref right_node = dag.tasks[task.right].query._nodes[
                        len(dag.tasks[task.right].query._nodes) - 1
                    ]
                    var left_rows = (
                        values[task.source]
                        .value()
                        .rows if values[task.source] else left.height()
                    )
                    var right_rows = (
                        values[task.right]
                        .value()
                        .rows if values[task.right] else right.height()
                    )
                    _ = scope.check(
                        rows._checked_mul(
                            len(left_node.text_fields)
                            + len(right_node.text_fields),
                            3 * size_of[ColumnMap](),
                        )
                    )
                    var columns = source_maps(
                        left,
                        left_node.text_fields,
                        base_domain.value(),
                        values[task.source],
                        left_rows,
                    )
                    columns.extend(
                        source_maps(
                            right,
                            right_node.text_fields,
                            base_domain.value(),
                            values[task.right],
                            right_rows,
                            left.width(),
                        )
                    )
                    join_maps = MapPlan(target_domain.value(), columns)
                    for side in range(2):
                        var schema = right.copy() if side else left.copy()
                        var fields = (
                            right_node.text_fields.copy() if side else left_node.text_fields.copy()
                        )
                        for field in fields:
                            for column in range(schema.width()):
                                if schema._columns[column].name() == field.name:
                                    var renamed = field.copy()
                                    renamed.name = String(
                                        column + (left.width() if side else 0)
                                    )
                                    var resident = (
                                        values[task.right]
                                        .copy() if side else values[task.source]
                                        .copy()
                                    )
                                    if resident:
                                        for at in range(
                                            resident.value().schema.width()
                                        ):
                                            if (
                                                resident.value()
                                                .schema._columns[at]
                                                .name()
                                                == field.name
                                            ):
                                                renamed.category = (
                                                    resident.value()
                                                    .categories[at]
                                                    .copy()
                                                )
                                    join_source_text.append(renamed^)
            if task.kind == DAG_EXPLODE:
                if not nested[task.source]:
                    raise Error(
                        "Resident EXPLODE source owner has been released"
                    )
                var output = explode.execute(
                    nested[task.source].value(),
                    task.query._nodes[len(task.query._nodes) - 1].names,
                    scope,
                )
                var region = _storage_report(
                    task.root,
                    "EXPLODE",
                    "mojo_gpu_primitive_explode",
                    nested[task.source].value().rows,
                    output.rows,
                    scope,
                    region_started,
                    live_before,
                    output.retained_bytes(),
                )
                executed = (output^, region^)
            elif task.kind == DAG_JOIN:
                if task.equal_rows:
                    # Marked distribution zip children are complete observed
                    # producers. Unequal lengths fail before join allocation.
                    var left_rows = (
                        values[task.source]
                        .value()
                        .rows if values[task.source] else _frame(
                            dag, values, task.source
                        )
                        .height()
                    )
                    var right_rows = (
                        values[task.right]
                        .value()
                        .rows if values[task.right] else _frame(
                            dag, values, task.right
                        )
                        .height()
                    )
                    if left_rows != right_rows:
                        raise Error(
                            "explode: columns have different element counts at row 0"
                        )
                executed = joins.execute_plan(
                    task,
                    _frame(dag, values, task.source),
                    _frame(dag, values, task.right),
                    scope,
                    profiling,
                    left_resident=values[task.source],
                    right_resident=values[task.right],
                    export_values=not (observation_only and index == dag.root),
                    text_remap=join_maps^,
                    source_text=join_source_text,
                )
                if not task.terminal:
                    executed[0].observe_errors()
            elif task.kind == DAG_ASOF:
                executed = asof.execute_plan(
                    task,
                    _frame(dag, values, task.source),
                    _frame(dag, values, task.right),
                    scope,
                    profiling,
                    left_resident=values[task.source],
                    right_resident=values[task.right],
                    export_values=not (observation_only and index == dag.root),
                    text_remap=join_maps^,
                    source_text=join_source_text,
                )
            else:
                var resident = Optional[Table]()
                if task.source >= 0:
                    task.query._frames[0] = _frame(dag, values, task.source)
                    resident = values[task.source].copy()
                var source_rows = resident.value().rows if resident else -1
                if is_partition_cut_marker(task.query):
                    if not resident:
                        raise Error(
                            "Partition CUT original resident producer is missing"
                        )
                    executed = partition_cuts.execute_plan(
                        task.query,
                        resident.value(),
                        scope,
                        export_values=not (
                            observation_only and index == dag.root
                        ),
                        text_domain=target_domain,
                        state_context=partition_context,
                    )
                elif is_qcut_marker(task.query):
                    if not resident:
                        raise Error(
                            "QCUT observed resident producer is missing"
                        )
                    # Observed operand precedes dynamic quantile controls;
                    # the classifier keeps values and codes device resident.
                    executed = qcuts.execute_plan(
                        task.query,
                        resident.value(),
                        scope,
                        export_values=not (
                            observation_only and index == dag.root
                        ),
                        text_domain=target_domain,
                    )
                else:
                    if task.sum_receipt:
                        # Producer execute/observe completed synchronously before
                        # this task; no failed or masked GROUP result can authorize
                        # arithmetic. The same DAG retains its exact source owner.
                        decimal_sum_receipts.validate(
                            task.sum_receipt.value()[],
                            task.source,
                            dag.tasks[task.source].query,
                            task.query,
                        )
                    var plan = rows._preflight(
                        task.query,
                        terminal_boundary=task.terminal,
                        source_rows=source_rows,
                        sum_receipt=task.sum_receipt,
                    )
                    executed = rows.execute_plan(
                        plan,
                        scope,
                        resident_input=resident,
                        profiling=profiling,
                        native_output=task.terminal
                        and index == dag.root
                        and not observation_only,
                        terminal_boundary=task.terminal,
                        export_values=not (
                            observation_only and index == dag.root
                        ),
                        text_domain=target_domain,
                    )
            if target_domain and (
                task.kind == DAG_JOIN or task.kind == DAG_ASOF
            ):
                executed[0].attach_text(
                    task.query._nodes[len(task.query._nodes) - 1].text_fields,
                    target_domain.value(),
                )
            if task.terminal and index != dag.root:
                # Window producers permit checked terminal expressions, but their
                # complete result is observed before a consumer can prune rows.
                executed[0].observe_errors()
            var cut_failure = cut_control_failure(task.query)
            if cut_failure:
                if not task.terminal or index == dag.root:
                    executed[0].observe_errors()
                raise Error(cut_failure)
            var region = executed[1].copy()
            var owned = executed[0].retained_bytes()
            var host_result_bytes = 0
            if index == dag.root:
                if observation_only:
                    executed[0].observe_errors()
                    result = task.schema.copy()
                else:
                    if text_context:
                        _ = scope.check(
                            rows._checked_mul(
                                executed[0].schema.width(),
                                3 * size_of[Optional[DataType]](),
                            )
                        )
                        var domains = List[Optional[DataType]](
                            capacity=executed[0].schema.width()
                        )
                        for column in range(len(executed[0].text)):
                            var field = executed[0].text[column].copy()
                            if executed[0].categories[column]:
                                domains.append(
                                    executed[0]
                                    .categories[column]
                                    .value()
                                    .owner[]
                                    .dtype
                                )
                            elif field:
                                domains.append(field.value().domain[].dtype)
                            else:
                                domains.append(None)
                        var state = Pointer(
                            to=text_context.value()[]
                        ).unsafe_origin_cast[MutAnyOrigin]()
                        state[].record_final(domains)
                    if profiling:
                        var readback_started = Int(perf_counter_ns())
                        var readback_ns = 0
                        var result_ns = 0
                        if len(executed[0].native_columns):
                            var host = rows._readback_native(executed[0])
                            readback_ns = (
                                Int(perf_counter_ns()) - readback_started
                            )
                            var result_started = Int(perf_counter_ns())
                            result = rows._materialize_native_host(
                                executed[0], host^
                            )
                            result_ns = Int(perf_counter_ns()) - result_started
                        else:
                            var host = executed[0].readback()
                            readback_ns = (
                                Int(perf_counter_ns()) - readback_started
                            )
                            var result_started = Int(perf_counter_ns())
                            result = rows._materialize_host(executed[0], host)
                            result_ns = Int(perf_counter_ns()) - result_started
                        for field in List[Tuple[String, Int]](
                            [
                                ("readback_ms", readback_ns),
                                ("result_copy_ms", result_ns),
                            ]
                        ):
                            region = region.with_column(
                                Series(
                                    field[0],
                                    Column[Float64]([Float64(field[1]) / 1e6]),
                                )
                            )
                    else:
                        result = rows.materialize(executed[0])
                for column in result._columns:
                    host_result_bytes = rows._checked_add(
                        host_result_bytes,
                        rows._host_column_bytes(column),
                    )
            query_host_result_bytes = (
                host_result_bytes if index
                == dag.root else query_host_result_bytes
            )
            if task.kind == DAG_EXPLODE:
                # Its storage report precedes the common final readback. Include
                # that region's compact host matrix and typed export envelope.
                region = _int_column(
                    region, "peak_requested_bytes", scope.info()[1]
                )
            if device_service and index == dag.root:
                scope.synchronize()
                service_drain_done = True
            var traffic_after = scope.transfers()
            region = _int_column(
                region, "upload_bytes", traffic_after[0] - traffic_before[0]
            )
            region = _int_column(
                region, "download_bytes", traffic_after[1] - traffic_before[1]
            )
            region = _int_column(
                region, "kernel_launches", traffic_after[2] - traffic_before[2]
            )
            region = _int_column(
                region, "synchronizations", traffic_after[3] - traffic_before[3]
            )
            region = _int_column(region, "host_result_bytes", host_result_bytes)
            region = _int_column(region, "node", task.root)
            region = _int_column(
                region, "retained_intermediate_bytes", live_before
            )
            region = _int_column(region, "intermediate_output_bytes", owned)
            region = region.with_column(
                Series(
                    "materialization_boundary",
                    Column[String](
                        List[String](
                            length=region.height(),
                            fill="Mojo GPU error observation" if observation_only
                            and index
                            == dag.root else "Mojo GPU -> CPU columns" if index
                            == dag.root else "Mojo GPU resident table",
                        )
                    ),
                )
            )
            if profiling:
                region = _profile_phases(region)
                region = _float_column(
                    region,
                    "synchronization_ms",
                    Float64(scope.timing_ns()[1] - region_sync_before) / 1e6,
                )
            report = region^ if report.width() == 0 else report.vstack(
                region.select(report.columns())
            )
            values.append(executed[0].copy())
            nested.append(None)
            if task.source >= 0:
                uses[task.source] -= 1
                if uses[task.source] == 0:
                    values[task.source] = None
                    nested[task.source] = None
            if task.kind == DAG_JOIN or task.kind == DAG_ASOF:
                uses[task.right] -= 1
                if uses[task.right] == 0:
                    values[task.right] = None
                    nested[task.right] = None
        if device_service and not service_drain_done:
            # An empty/direct scan DAG has no terminal region to own a drain. The
            # scope guard avoids initializing a context when the DAG did no work.
            scope.synchronize()
        if profiling:
            report = report.with_column(
                Series(
                    "query_preflight_ms",
                    Column[Float64](
                        List[Float64](
                            length=report.height(),
                            fill=Float64(preflight_ns) / 1e6,
                        )
                    ),
                )
            )
        report = _int_column(
            report, "query_peak_requested_bytes", scope.info()[1]
        )
        report = _int_column(
            report, "query_host_result_bytes", query_host_result_bytes
        )
        report = _int_column(report, "query_upload_bytes", scope.transfers()[0])
        report = _int_column(
            report, "query_download_bytes", scope.transfers()[1]
        )
        report = _int_column(
            report, "query_kernel_launches", scope.transfers()[2]
        )
        report = _int_column(
            report, "query_synchronizations", scope.transfers()[3]
        )
        report = report.with_column(
            Series(
                "query_wall_ms",
                Column[Float64](
                    List[Float64](
                        length=report.height(),
                        fill=Float64(Int(perf_counter_ns()) - started) / 1e6,
                    )
                ),
            )
        )
        if profiling:
            report = _float_column(
                report,
                "query_synchronization_ms",
                Float64(scope.timing_ns()[1]) / 1e6,
            )
        if device_service and profiling:
            var scratch_cache = device_service.value().scratch_info()
            var pipeline_cache = device_service.value().pipeline_info()
            for field in List[Tuple[String, Int]](
                [
                    (
                        "service_context_generations",
                        device_service.value().cached_generations(),
                    ),
                    ("service_idle_scratch_bytes", scratch_cache[0]),
                    ("service_idle_scratch_entries", scratch_cache[1]),
                    ("service_scratch_hits", scratch_cache[2]),
                    ("service_scratch_misses", scratch_cache[3]),
                    ("service_pipeline_entries", pipeline_cache[0]),
                    ("service_pipeline_hits", pipeline_cache[1]),
                    ("service_pipeline_misses", pipeline_cache[2]),
                    ("service_pipeline_clears", pipeline_cache[3]),
                    ("service_pipeline_capacity", 9),
                ]
            ):
                report = report.with_column(
                    Series(
                        field[0],
                        Column[Int64](
                            List[Int64](
                                length=report.height(), fill=Int64(field[1])
                            )
                        ),
                    )
                )
        _ = producer_metadata^
        if device_service:
            scope.mark_success()
        return (result^, report^)
    except error:
        if device_service:
            scope.mark_failed()
            try:
                scope.synchronize()
            except cleanup_error:
                _ = String(cleanup_error)
        _ = values^
        _ = nested^
        _ = base_domain^
        _ = producer_metadata^
        raise error^


def describe(
    query: LazyFrame,
    capabilities: RowCapabilities,
    *,
    ordered_source: Bool = False,
    eager_feed: Bool = False,
) -> String:
    try:
        var dag = preflight(
            query,
            capabilities,
            ordered_source=ordered_source,
            eager_feed=eager_feed,
        )
        return (
            "ENGINE accel: Mojo GPU joins and row regions; "
            + String(dag.joins)
            + " joins, "
            + String(len(dag.tasks))
            + " regions; immutable Mojo GPU resident tables; final CPU export\n"
        )
    except error:
        return "ENGINE accel: " + String(error) + "\n"
