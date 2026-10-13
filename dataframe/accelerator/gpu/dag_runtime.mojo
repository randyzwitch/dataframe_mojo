"""Direct Mojo execution of shared join and scalar-consumer region DAGs.

Every region is metadata-bound and checked before the first device context.
Each computed region runs Mojo GPU kernels and materializes CPU columns at its
boundary; those columns are staged for the next region. Original immutable
scans are reused directly. Live computed frames count against the same query
budget until their last consumer releases them. Reports expose these boundaries
and the complete observed query peak rather than implying device residency.
"""
from std.time import perf_counter_ns
from dataframe import Column, DataFrame, Series
from dataframe.lazy import LazyFrame
from dataframe.accelerator.capabilities import RowCapabilities
from dataframe.accelerator.joins import (
    JoinDag,
    DAG_JOIN,
    DAG_ROWS,
    lower_join_dag,
)
from . import runtime as rows
from . import join_runtime as joins


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
        elif task.kind == DAG_ROWS:
            _ = rows._preflight(task.query, terminal_boundary=task.terminal)
        else:
            raise Error("Mojo GPU unexpected execution DAG task")
    return dag^


def retained_bytes(frame: DataFrame) raises -> Int:
    """Conservative owned typed payload and validity bytes for a materialization."""
    var retained = 0
    for column in frame._columns:
        retained = rows._checked_add(
            retained,
            rows._checked_mul(frame.height(), rows._width(column.dtype())),
        )
        retained = rows._checked_add(retained, (frame.height() + 7) // 8)
    return retained


def execute(
    query: LazyFrame,
    device_id: Int,
    budget: Int,
    profiling: Bool,
    capabilities: RowCapabilities,
) raises -> Tuple[DataFrame, DataFrame]:
    var started = Int(perf_counter_ns())
    var dag = preflight(query, capabilities)
    var uses = List[Int](length=len(dag.tasks), fill=0)
    for task in dag.tasks:
        if task.source >= 0:
            uses[task.source] += 1
        if task.kind == DAG_JOIN:
            uses[task.right] += 1
    var values = List[DataFrame](capacity=len(dag.tasks))
    var owned = List[Int](length=len(dag.tasks), fill=0)
    var live = 0
    var peak = 0
    var report = DataFrame(List[Series]())
    for index in range(len(dag.tasks)):
        var task = dag.tasks[index].copy()
        if (
            task.kind == DAG_ROWS
            and task.source < 0
            and len(task.query._nodes) == 1
        ):
            # Identity scans share the caller's immutable buffers. The first
            # computed consumer owns staging; this is not CPU evaluation.
            values.append(task.query._frames[0].copy())
            continue
        if budget >= 0 and live > budget:
            raise Error(
                "Mojo GPU DAG retained intermediates exceed memory budget"
            )
        var available = budget - live if budget >= 0 else -1
        var executed: Tuple[DataFrame, DataFrame]
        if task.kind == DAG_JOIN:
            executed = joins.execute(
                task,
                values[task.source],
                values[task.right],
                device_id,
                available,
                profiling,
            )
        else:
            if task.source >= 0:
                task.query._frames[0] = values[task.source].copy()
            executed = rows.execute(
                task.query,
                device_id,
                available,
                profiling,
                terminal_boundary=task.terminal,
            )
        var region = executed[1].copy()
        var region_peak = 0
        for row in range(region.height()):
            region_peak = max(
                region_peak,
                Int(region.item(row, "peak_requested_bytes").int64()),
            )
        peak = max(peak, rows._checked_add(live, region_peak))
        owned[index] = retained_bytes(executed[0])
        region = region.with_column(
            Series(
                "node",
                Column[Int64](
                    List[Int64](length=region.height(), fill=Int64(task.root))
                ),
            )
        )
        region = region.with_column(
            Series(
                "retained_intermediate_bytes",
                Column[Int64](
                    List[Int64](length=region.height(), fill=Int64(live))
                ),
            )
        )
        region = region.with_column(
            Series(
                "intermediate_output_bytes",
                Column[Int64](
                    List[Int64](
                        length=region.height(), fill=Int64(owned[index])
                    )
                ),
            )
        )
        region = region.with_column(
            Series(
                "materialization_boundary",
                Column[String](
                    List[String](
                        length=region.height(), fill="GPU result -> CPU columns"
                    )
                ),
            )
        )
        report = region^ if report.width() == 0 else report.vstack(region)
        values.append(executed[0].copy())
        live = rows._checked_add(live, owned[index])
        peak = max(peak, live)
        if budget >= 0 and peak > budget:
            raise Error("Mojo GPU DAG exceeds memory budget")
        if task.source >= 0:
            uses[task.source] -= 1
            if uses[task.source] == 0:
                live -= owned[task.source]
                owned[task.source] = 0
                values[task.source] = DataFrame(List[Series]())
        if task.kind == DAG_JOIN:
            uses[task.right] -= 1
            if uses[task.right] == 0:
                live -= owned[task.right]
                owned[task.right] = 0
                values[task.right] = DataFrame(List[Series]())
    report = report.with_column(
        Series(
            "query_peak_requested_bytes",
            Column[Int64](
                List[Int64](length=report.height(), fill=Int64(peak))
            ),
        )
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
    return (values[dag.root].copy(), report^)


def describe(query: LazyFrame, capabilities: RowCapabilities) -> String:
    try:
        var dag = preflight(query, capabilities)
        return (
            "ENGINE accel: Mojo GPU joins and row regions; "
            + String(dag.joins)
            + " joins, "
            + String(len(dag.tasks))
            + " regions; GPU results materialize CPU columns between regions\n"
        )
    except error:
        return "ENGINE accel: " + String(error) + "\n"
