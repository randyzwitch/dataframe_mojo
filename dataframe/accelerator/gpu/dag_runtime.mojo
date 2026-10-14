"""Direct Mojo execution of shared join and scalar-consumer region DAGs.

Every region is metadata-bound before device initialization. Computed regions
share immutable Mojo-owned device tables and copy selected input columns on
the device. Original scans stage on first consumption; only the final table
exports CPU values. Shared owner accounting includes all live intermediates,
region scratch and the final host export before allocation.
"""
from std.time import perf_counter_ns
from dataframe import Column, DataFrame, Series
from dataframe.lazy import LazyFrame
from dataframe.accelerator.capabilities import RowCapabilities
from dataframe.accelerator.joins import (
    JoinDag,
    DAG_JOIN,
    DAG_ASOF,
    DAG_ROWS,
    lower_join_dag,
)
from . import runtime as rows
from .temporal_descriptors import ROW_TEMPORAL_BINARY, ROW_TEMPORAL_CAST
from . import join_runtime as joins
from . import asof_runtime as asof
from .residency import Scope, Table


def preflight(
    query: LazyFrame, capabilities: RowCapabilities
) raises -> JoinDag:
    """Check every reachable task and operation without device initialization."""
    var semantic = capabilities.copy()
    semantic.scalar_regions = True
    var dag = lower_join_dag(query, semantic)
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
        elif task.kind == DAG_ROWS:
            var plan = rows._preflight(
                task.query, terminal_boundary=task.terminal
            )
            if task.source >= 0:
                for node in range(len(plan.node_dtypes)):
                    var op = plan.code[4 * node]
                    if op == ROW_TEMPORAL_BINARY or op == ROW_TEMPORAL_CAST:
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


def execute(
    query: LazyFrame,
    device_id: Int,
    budget: Int,
    profiling: Bool,
    capabilities: RowCapabilities,
    *,
    observation_only: Bool = False,
) raises -> Tuple[DataFrame, DataFrame]:
    var started = Int(perf_counter_ns())
    var dag = preflight(query, capabilities)
    var scope = Scope(device_id, budget)
    var uses = List[Int](length=len(dag.tasks), fill=0)
    for task in dag.tasks:
        if task.source >= 0:
            uses[task.source] += 1
        if task.kind == DAG_JOIN or task.kind == DAG_ASOF:
            uses[task.right] += 1
    var values = List[Optional[Table]](capacity=len(dag.tasks))
    var report = DataFrame(List[Series]())
    var result = DataFrame(List[Series]())
    for index in range(len(dag.tasks)):
        var task = dag.tasks[index].copy()
        if (
            task.kind == DAG_ROWS
            and task.source < 0
            and len(task.query._nodes) == 1
        ):
            # Share original immutable scan buffers; no data evaluation or
            # device staging occurs until the first computed consumer.
            if task.query._nodes[0].source_failure:
                raise Error(task.query._nodes[0].source_failure)
            values.append(None)
            if index == dag.root:
                result = task.schema.copy() if observation_only else task.query._frames[
                    0
                ].copy()
            continue
        var live_before = scope.info()[0]
        var traffic_before = scope.transfers()
        var executed: Tuple[Table, DataFrame]
        if task.kind == DAG_JOIN:
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
            )
        else:
            var resident = Optional[Table]()
            if task.source >= 0:
                task.query._frames[0] = _frame(dag, values, task.source)
                resident = values[task.source].copy()
            var source_rows = resident.value().rows if resident else -1
            var plan = rows._preflight(
                task.query,
                terminal_boundary=task.terminal,
                source_rows=source_rows,
            )
            executed = rows.execute_plan(
                plan,
                scope,
                resident_input=resident,
                profiling=profiling,
                terminal_boundary=task.terminal,
                export_values=not (observation_only and index == dag.root),
            )
        if task.terminal and index != dag.root:
            # Window producers permit checked terminal expressions, but their
            # complete result is observed before a consumer can prune rows.
            executed[0].observe_errors()
        var region = executed[1].copy()
        var owned = executed[0].retained_bytes()
        var host_result_bytes = 0
        if index == dag.root:
            if observation_only:
                executed[0].observe_errors()
                result = task.schema.copy()
            else:
                result = rows.materialize(executed[0])
            for column in result._columns:
                host_result_bytes = rows._checked_add(
                    host_result_bytes,
                    rows._checked_add(
                        rows._checked_mul(
                            result.height(), rows._width(column.dtype())
                        ),
                        (result.height() + 7) // 8,
                    ),
                )
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
        region = _int_column(region, "retained_intermediate_bytes", live_before)
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
        report = region^ if report.width() == 0 else report.vstack(
            region.select(report.columns())
        )
        values.append(executed[0].copy())
        if task.source >= 0:
            uses[task.source] -= 1
            if uses[task.source] == 0:
                values[task.source] = None
        if task.kind == DAG_JOIN or task.kind == DAG_ASOF:
            uses[task.right] -= 1
            if uses[task.right] == 0:
                values[task.right] = None
    report = _int_column(report, "query_peak_requested_bytes", scope.info()[1])
    report = _int_column(report, "query_upload_bytes", scope.transfers()[0])
    report = _int_column(report, "query_download_bytes", scope.transfers()[1])
    report = _int_column(report, "query_kernel_launches", scope.transfers()[2])
    report = _int_column(report, "query_synchronizations", scope.transfers()[3])
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
    return (result^, report^)


def describe(query: LazyFrame, capabilities: RowCapabilities) -> String:
    try:
        var dag = preflight(query, capabilities)
        return (
            "ENGINE accel: Mojo GPU joins and row regions; "
            + String(dag.joins)
            + " joins, "
            + String(len(dag.tasks))
            + " regions; immutable Mojo GPU resident tables; final CPU export\n"
        )
    except error:
        return "ENGINE accel: " + String(error) + "\n"
