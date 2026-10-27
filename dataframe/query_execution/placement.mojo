"""CPU/accelerator region placement through provider-neutral runtime ports."""
from dataframe.frame import (
    DataFrame,
)
from dataframe.optimizer.passes import (
    plan_optimized,
)
from dataframe.placement import (
    LinearCost,
    crossover_rows,
    load_costs,
)
from dataframe.planning.context import (
    QueryContext,
    _context_estimate,
    _context_handoff,
    _context_supports,
)
from dataframe.planning.explain import (
    plan_label,
)
from dataframe.planning.graph import (
    plan_estimated_rows,
)
from dataframe.planning.model import (
    AGG,
    FILTER,
    HANDOFF_AUTO,
    SELECT,
)
from dataframe.query_execution.reporting import (
    plan_record,
)
from std.collections import (
    Dict,
    Optional,
)
from std.time import (
    perf_counter_ns,
)


def _placement(
    region: QueryContext, rows: Int, cpu_costs: Dict[String, LinearCost]
) -> Tuple[Bool, String]:
    """Whether the accelerator is estimated faster for `region` at `rows`
    input rows, and both estimates as text for profiles. False, keeping the
    CPU, when either side has no measured cost."""
    var estimate = _context_estimate(region, rows)
    var cpu = cpu_costs.get(estimate[0])
    if estimate[1] < 0 or not cpu:
        return (False, String(" (no measured cost)"))
    var cpu_ns = cpu.value().ns(rows)
    var costs = (
        " (accel "
        + String(round(estimate[1] / 1e6, 3))
        + " ms, cpu "
        + String(round(cpu_ns / 1e6, 3))
        + " ms, "
        + String(rows)
        + " rows)"
    )
    return (estimate[1] < cpu_ns, costs^)


def plan_handoff_input(self: QueryContext, index: Int) -> Tuple[Int, List[Int]]:
    """The input below node `index` and its filters: the node offered
    with the filters directly beneath it, outermost first."""
    var filters = List[Int]()
    var input = self._nodes[index].left
    while input >= 0 and self._nodes[input].kind == FILTER:
        filters.append(input)
        input = self._nodes[input].left
    return (input, filters^)


def plan_handoff_plan(
    self: QueryContext, index: Int, filters: List[Int], columns: DataFrame
) -> QueryContext:
    """Node `index` and its filters as a plan over `columns`, the
    input's result: what a provider receives, columns in."""
    var region = self.with_input(columns)
    for f in reversed(range(len(filters))):
        region = region.push(self._nodes[filters[f]].copy())
    return region.push(self._nodes[index].copy())


def plan_handoff_region(
    self: QueryContext, index: Int, streaming: Bool, batch_size: Int
) raises -> Optional[DataFrame]:
    """Node `index`'s result when it is handed to the provider; None to
    execute it as usual.

    A projection or aggregation is offered with the filters directly
    below it. The provider decides from the input's schema alone, before
    any input runs. HANDOFF_FORCED hands every accepted node over.
    HANDOFF_AUTO compares the provider's estimate with the CPU's
    measured cost for the same operator (`placement.mojo`) at the
    input's rows: estimated from the plan when `_estimated_rows` can,
    before anything runs, so a node the CPU keeps executes exactly as
    it would have; otherwise counted once the input has run, and the CPU
    then runs the node on those columns, so nothing runs twice."""
    var kind = self._nodes[index].kind
    if kind != SELECT and kind != AGG:
        return None
    var cpu_costs = Dict[String, LinearCost]()
    if self._handoff == HANDOFF_AUTO:
        cpu_costs = load_costs("cpu")
        if len(cpu_costs) == 0:
            return None
    var found = plan_handoff_input(self, index)
    var input = found[0]
    if input < 0:
        return None
    var schema = self._execute(input, True, streaming, batch_size)
    var offered = plan_handoff_plan(self, index, found[1], schema)
    if not _context_supports(offered):
        return None
    var algorithm = String("handoff")
    var decided = False
    if self._handoff == HANDOFF_AUTO:
        var estimated = plan_estimated_rows(self, input)
        if estimated >= 0:
            var verdict = _placement(offered, estimated, cpu_costs)
            if not verdict[0]:
                plan_record(
                    self,
                    index,
                    "placement",
                    estimated,
                    -1,
                    algorithm="placed cpu" + verdict[1],
                )
                return None
            algorithm += verdict[1]
            decided = True
    var columns = self._execute(input, False, streaming, batch_size)
    var rows = columns.height()
    var region = plan_handoff_plan(self, index, found[1], columns)
    if self._handoff == HANDOFF_AUTO and not decided:
        var verdict = _placement(region, rows, cpu_costs)
        if not verdict[0]:
            var began = Int(perf_counter_ns())
            var placed = region._execute(
                len(region._nodes) - 1, False, streaming, batch_size
            )
            plan_record(
                self,
                index,
                "eager",
                rows,
                placed.height(),
                algorithm="placed cpu" + verdict[1],
                wall_ns=Int(perf_counter_ns()) - began,
            )
            return placed^
        algorithm += verdict[1]
    var began = Int(perf_counter_ns())
    var result = _context_handoff(region)
    plan_record(
        self,
        index,
        "accel",
        rows,
        result.height(),
        algorithm=algorithm,
        wall_ns=Int(perf_counter_ns()) - began,
    )
    return result^


def plan_placement_lines(self: QueryContext) raises -> String:
    """One `PLACE` line per node the provider supports: both measured
    cost models and the row count from which the accelerator is
    estimated faster, since rows are known only when the node runs."""
    var cpu_costs = load_costs("cpu")
    var out = String()
    for index in range(len(self._nodes)):
        var kind = self._nodes[index].kind
        if kind != SELECT and kind != AGG:
            continue
        var found = plan_handoff_input(self, index)
        if found[0] < 0:
            continue
        var schema = self._execute(found[0], True)
        var region = plan_handoff_plan(self, index, found[1], schema)
        if not _context_supports(region):
            continue
        var at_zero = _context_estimate(region, 0)
        var at_million = _context_estimate(region, 1_000_000)
        var cpu = cpu_costs.get(at_zero[0])
        out += "PLACE " + plan_label(self, index) + ": " + at_zero[0] + ": "
        if at_zero[1] < 0 or not cpu:
            out += "no measured cost; cpu\n"
            continue
        var accel = LinearCost(at_zero[1], (at_million[1] - at_zero[1]) / 1e6)
        var from_rows = crossover_rows(cpu.value(), accel)
        out += (
            "cpu "
            + String(round(cpu.value().fixed_ns / 1e3, 1))
            + " us + "
            + String(round(cpu.value().row_ns, 3))
            + " ns/row, accel "
            + String(round(accel.fixed_ns / 1e3, 1))
            + " us + "
            + String(round(accel.row_ns, 3))
            + " ns/row; "
            + (
                "accel from " + String(from_rows) + " rows" if from_rows
                >= 0 else "cpu at every size"
            )
            + "\n"
        )
    return out^


def plan_handoff_nodes(self: QueryContext) raises -> List[String]:
    """The nodes of the optimized plan the provider would run, as
    `explain` labels them: each projection or aggregation it accepts
    from its input's schema."""
    var plan = plan_optimized(self)
    var labels = List[String]()
    for index in range(len(plan._nodes)):
        var kind = plan._nodes[index].kind
        if kind != SELECT and kind != AGG:
            continue
        var found = plan_handoff_input(plan, index)
        if found[0] < 0:
            continue
        var schema = plan._execute(found[0], True)
        if _context_supports(plan_handoff_plan(plan, index, found[1], schema)):
            labels.append(plan_label(plan, index))
    return labels^
