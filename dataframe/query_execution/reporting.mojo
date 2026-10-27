"""Observed per-node execution reporting."""
from dataframe.planning.context import (
    QueryContext,
)
from dataframe.planning.explain import (
    plan_label,
)


def plan_record(
    self: QueryContext,
    node: Int,
    executor: String,
    input_rows: Int,
    output_rows: Int,
    *,
    executions: Int = 1,
    build_rows: Int = 0,
    builds: Int = 0,
    algorithm: String = "",
    build_side: String = "",
    busy_ns: Int = 0,
    wall_ns: Int = 0,
):
    """Add to the attached report, if any."""
    if not self._report:
        return
    var report = self._report.value()
    report[].record(
        node,
        plan_label(self, node),
        executor,
        input_rows,
        output_rows,
        executions=executions,
        build_rows=build_rows,
        builds=builds,
        algorithm=algorithm,
        build_side=build_side,
        busy_ns=busy_ns,
        wall_ns=wall_ns,
    )
