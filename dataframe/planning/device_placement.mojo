"""Where each pipeline of a plan runs: the CPU or an accelerator device.

A plan is cut into placement units (`plan_pipelines`): a chain of row-local
steps (filters, projections) that ends in a sink, the operator that consumes
the chain (an aggregate, a sort, a join, ...), or in the plan's result. The
sink's other inputs, such as a join's build side, are child units. Joins end
a unit here, though the CPU executor streams through them; a unit is the
grain of placement, not of execution.

`solve_placement` then chooses a device per unit exactly, by dynamic
programming over the tree of units:

    cost(p, d) = run(p, d) + move(p's source, to d)
               + sum over children c of min over d' of
                     [cost(c, d') + move(c's output, d' -> d)]

with the result moved to the host at the root. Moving data between the same
device costs nothing; between the host and a device it costs the device's
measured `TransferModel`, which is zero on unified memory. A device's
start-up cost is paid once if any unit uses it, so the program is solved
for each set of devices allowed and the cheapest kept. A unit whose
working bytes exceed a device's budget cannot run there. Anything not
measured (a run cost, a transfer, bytes that matter) rules a device out for
that unit, so an unmeasured plan stays on the CPU.

Nothing here knows a backend exists: the caller asks the backend for its
devices and estimates and passes the numbers in.
"""
from std.collections import Optional
from dataframe.aggregation.expressions import _row_local
from dataframe.devices import ByteCost, PipelineFeatures, TransferModel
from dataframe.dtype import DataType
from dataframe.expr import changes_length
from dataframe.frame import DataFrame
from dataframe.io.csv import CsvSchema
from dataframe.planning.context import QueryContext
from dataframe.planning.graph import plan_estimated_rows
from dataframe.planning.model import (
    AGG,
    CONCAT,
    DROP,
    FILTER,
    GROUPING_SETS,
    HCONCAT,
    JOIN,
    SCAN_FRAME,
    SELECT,
    SLICE,
    SORT,
    UNIQUE,
    WITH_COLUMNS,
)
from dataframe.planning.node_builders import _frame_scan_node
from dataframe.planning.properties import _is_scan, _stream_rows

# Larger than any real cost; a choice that is not allowed.
comptime _NEVER = Float64(1e300)
# Device 0 of a placement problem is always the CPU (the host).
comptime CPU = 0


struct PlacementProblem(Copyable, Movable):
    """The numbers a placement is chosen from. Units are numbered with
    children before parents; devices with the CPU as 0."""

    var children: List[List[Int]]
    var root: Int
    # Nanoseconds to run unit p on device d, excluding data movement; the
    # CPU's must be >= 0, a negative device cost rules that device out.
    var run_ns: List[List[Float64]]
    # Bytes each unit produces, and reads from its own source; -1 unknown.
    var output_bytes: List[Int]
    var source_bytes: List[Int]
    # The device each unit's source already lives on (CPU for the host).
    var source_device: List[Int]
    # Bytes a unit holds on its device at once; -1 unknown.
    var working_bytes: List[Int]
    # Per device; entry 0 (the CPU) is unused.
    var transfers: List[TransferModel]
    var budgets: List[Int]
    var startup_ns: List[Float64]

    def __init__(out self, units: Int, devices: Int):
        self.children = List[List[Int]](length=units, fill=List[Int]())
        self.root = units - 1
        self.run_ns = List[List[Float64]](
            length=units, fill=List[Float64](length=devices, fill=-1)
        )
        self.output_bytes = List[Int](length=units, fill=-1)
        self.source_bytes = List[Int](length=units, fill=0)
        self.source_device = List[Int](length=units, fill=CPU)
        self.working_bytes = List[Int](length=units, fill=-1)
        self.transfers = List[TransferModel](
            length=devices, fill=TransferModel.shared_memory()
        )
        self.budgets = List[Int](length=devices, fill=-1)
        self.startup_ns = List[Float64](length=devices, fill=0)

    def devices(self) -> Int:
        return len(self.transfers)

    def move(self, bytes: Int, source: Int, target: Int) -> Float64:
        """Nanoseconds to move `bytes` from device `source` to `target`."""
        if source == target:
            return 0
        var total = Float64(0)
        if source != CPU:
            var cost = _leg(self.transfers[source].to_host, bytes)
            if cost >= _NEVER:
                return _NEVER
            total += cost
        if target != CPU:
            var cost = _leg(self.transfers[target].to_device, bytes)
            if cost >= _NEVER:
                return _NEVER
            total += cost
        return total

    def allowed(self, unit: Int, device: Int) -> Bool:
        if device == CPU:
            return True
        if self.run_ns[unit][device] < 0:
            return False
        var budget = self.budgets[device]
        if budget < 0:
            return True
        var working = self.working_bytes[unit]
        return working >= 0 and working <= budget


def _leg(cost: ByteCost, bytes: Int) -> Float64:
    if not cost.measured():
        return _NEVER
    if cost.ns_per_byte == 0:
        return cost.fixed_ns
    if bytes < 0:
        return _NEVER
    return cost.cost(bytes)


@fieldwise_init
struct Placement(Copyable, Movable):
    """The device of each unit and the estimated nanoseconds."""

    var devices: List[Int]
    var total_ns: Float64
    # Every unit on the CPU, for comparison.
    var cpu_ns: Float64


def _solve_subset(
    problem: PlacementProblem, allowed: List[Bool]
) -> Tuple[Float64, List[Int]]:
    var units = len(problem.children)
    var devices = problem.devices()
    var cost = List[List[Float64]](
        length=units, fill=List[Float64](length=devices, fill=_NEVER)
    )
    # For each unit, device, child: the child's device in the best choice.
    var pick = List[List[List[Int]]]()
    for p in range(units):
        var per_device = List[List[Int]]()
        for d in range(devices):
            var children = List[Int]()
            if not allowed[d] or not problem.allowed(p, d):
                for _ in problem.children[p]:
                    children.append(CPU)
                per_device.append(children^)
                continue
            var run = problem.run_ns[p][d]
            if d == CPU:
                run = max(run, 0)
            var total = run + problem.move(
                problem.source_bytes[p], problem.source_device[p], d
            )
            for c in problem.children[p]:
                var best = _NEVER
                var best_device = CPU
                for e in range(devices):
                    if cost[c][e] >= _NEVER:
                        continue
                    var option = cost[c][e] + problem.move(
                        problem.output_bytes[c], e, d
                    )
                    if option < best:
                        best = option
                        best_device = e
                children.append(best_device)
                total = min(total + best, _NEVER)
            cost[p][d] = total
            per_device.append(children^)
        pick.append(per_device^)
    var root = problem.root
    var best = _NEVER
    var device = CPU
    for d in range(devices):
        if cost[root][d] >= _NEVER:
            continue
        var option = cost[root][d] + problem.move(
            problem.output_bytes[root], d, CPU
        )
        if option < best:
            best = option
            device = d
    var chosen = List[Int](length=units, fill=CPU)
    var stack = List[Tuple[Int, Int]]()
    stack.append((root, device))
    while len(stack):
        var top = stack.pop()
        chosen[top[0]] = top[1]
        for i in range(len(problem.children[top[0]])):
            stack.append((problem.children[top[0]][i], pick[top[0]][top[1]][i]))
    return (best, chosen^)


def solve_placement(problem: PlacementProblem) -> Placement:
    """The cheapest device per unit, with each device's start-up paid once
    when it is used. All-CPU is always a candidate."""
    var units = len(problem.children)
    var cpu = Float64(0)
    for p in range(units):
        cpu += max(problem.run_ns[p][CPU], 0)
    var others = problem.devices() - 1
    var best = Placement(List[Int](length=units, fill=CPU), cpu, cpu)
    for subset in range(1, 1 << others):
        var allowed = List[Bool](length=problem.devices(), fill=False)
        allowed[CPU] = True
        var startup = Float64(0)
        for d in range(1, problem.devices()):
            if (subset >> (d - 1)) & 1:
                allowed[d] = True
                startup += max(problem.startup_ns[d], 0)
        var solved = _solve_subset(problem, allowed)
        var total = solved[0] + startup
        if total < best.total_ns:
            best = Placement(solved[1].copy(), total, cpu)
    return best^


@fieldwise_init
struct PlacementPipeline(Copyable, Movable):
    """One placement unit of a plan."""

    # The unit's top node: its sink, or the last row-local step when the
    # unit ends in the plan's result or feeds another unit's second input.
    var top: Int
    # The sink node, or -1 when the unit is steps only.
    var sink: Int
    # Row-local steps from the top down.
    var steps: List[Int]
    # The node the steps read: `source`, or a child unit's top.
    var input: Int
    # The scan the unit reads, or -1 when it reads a child unit.
    var source: Int
    # Child units, by index into the list `plan_pipelines` returns.
    var children: List[Int]
    # The unit as a plan over its inputs, for a backend to inspect.
    var region: QueryContext
    # The CPU calibration key: sink and step kinds and input types.
    var key: String
    var features: PipelineFeatures
    var output_bytes: Int
    var source_bytes: Int
    var working_bytes: Int


def _kind_name(kind: Int) -> String:
    if kind == FILTER:
        return "filter"
    if kind == SELECT:
        return "select"
    if kind == WITH_COLUMNS:
        return "with_columns"
    if kind == DROP:
        return "drop"
    if kind == AGG:
        return "agg"
    if kind == JOIN:
        return "join"
    if kind == SORT:
        return "sort"
    if kind == SLICE:
        return "slice"
    if kind == UNIQUE:
        return "unique"
    if kind == CONCAT:
        return "concat"
    if kind == HCONCAT:
        return "hconcat"
    if kind == GROUPING_SETS:
        return "grouping_sets"
    return "op" + String(kind)


def _row_bytes(schema: DataFrame) -> Int:
    """Estimated bytes per row: fixed widths by type, 32 for strings and
    other variable-width values (as `_frame_bytes` estimates them)."""
    var total = 0
    for column in schema._columns:
        var bits = column.dtype().bit_width()
        total += (bits + 7) // 8 if bits > 0 else 32
    return total


def _is_step(self: QueryContext, index: Int) -> Bool:
    ref node = self._nodes[index]
    return node.kind == DROP or _stream_rows(node)


def _bounded_rows(
    self: QueryContext, top: Int, sink: Int, input_rows: Int
) raises -> Int:
    """The planner's estimate of a unit's output rows, or else an upper
    bound for operations that cannot expand their input. Scalar selects
    can produce one row even from empty input. Length-changing selects
    and joins have no general bound; -1 when unknown."""
    if sink >= 0 and self._nodes[sink].kind == SELECT:
        for expression in self._nodes[sink].exprs:
            if changes_length(expression):
                return -1
    var estimated = plan_estimated_rows(self, top)
    if estimated >= 0 or input_rows < 0:
        return estimated
    if sink < 0:
        return input_rows
    var kind = self._nodes[sink].kind
    if kind == SLICE:
        var length = self._nodes[sink].slice_length()
        return min(length, input_rows) if length >= 0 else input_rows
    if kind == SELECT:
        return max(1, input_rows)
    if kind == AGG or kind == UNIQUE or kind == SORT:
        return input_rows
    return -1


def plan_pipelines(self: QueryContext) raises -> List[PlacementPipeline]:
    """The plan's placement units, children before parents; the last one
    produces the plan's result."""
    var units = plan_pipeline_cuts(self)
    plan_pipeline_estimates(self, units)
    return units^


def plan_pipeline_cuts(self: QueryContext) raises -> List[PlacementPipeline]:
    """The placement units without their row and byte estimates (all -1).
    Estimating samples filters, which costs more than cutting, so a planner
    asks the backends first and estimates only when one can run a unit."""
    var units = List[PlacementPipeline]()
    _ = _unit(self, len(self._nodes) - 1, units)
    return units^


def plan_pipeline_estimates(
    self: QueryContext, mut units: List[PlacementPipeline]
) raises:
    """Fill in each unit's rows and bytes, children before parents, so a
    unit over a child reads the child's estimated output."""
    for p in range(len(units)):
        var top = units[p].top
        var sink = units[p].sink
        var cursor = units[p].input
        var source = units[p].source
        var children = units[p].children.copy()
        var row_bytes = _row_bytes(units[p].region._frames[0])
        var right = -1
        if sink >= 0 and self._nodes[sink].right >= 0:
            right = children[len(children) - 1]
        var input_rows = -1
        if cursor >= 0:
            input_rows = units[
                children[0]
            ].features.output_rows if source < 0 else plan_estimated_rows(
                self, cursor
            )
        var output_rows = _bounded_rows(self, top, sink, input_rows)
        var groups = -1
        if sink >= 0 and self._nodes[sink].kind == AGG:
            groups = output_rows
        var build_rows = -1
        if sink >= 0 and self._nodes[sink].kind == JOIN:
            build_rows = plan_estimated_rows(self, self._nodes[sink].right)
        var input_bytes = -1
        if input_rows >= 0:
            input_bytes = input_rows * row_bytes
        var output_bytes = -1
        if output_rows >= 0:
            output_bytes = output_rows * _row_bytes(self._execute(top, True))
        # Input, build side and output held at once.
        var working = -1
        if input_bytes >= 0 and output_bytes >= 0:
            working = input_bytes + output_bytes
            if right >= 0:
                var build = units[right].output_bytes
                working = working + build if build >= 0 else -1
        units[p].features = PipelineFeatures(
            input_rows, input_bytes, output_rows, groups, build_rows
        )
        units[p].output_bytes = output_bytes
        units[p].source_bytes = input_bytes if source >= 0 else 0
        units[p].working_bytes = working


def _input(self: QueryContext, index: Int) raises -> DataFrame:
    """What a unit reads at node `index`: an in-memory frame as it is (its
    types and where its buffers live), anything else as its schema."""
    if self._nodes[index].kind == SCAN_FRAME:
        return self._execute(index, False)
    return self._execute(index, True)


def _region(
    self: QueryContext,
    input: DataFrame,
    steps: List[Int],
    sink: Int,
    build: Optional[DataFrame],
) -> QueryContext:
    """Steps and sink as a plan over `input`, with `build` as the sink's
    second input."""
    var region = self.with_input(input)
    for s in reversed(range(len(steps))):
        region = region.push(self._nodes[steps[s]].copy())
    if sink >= 0:
        var node = self._nodes[sink].copy()
        node.left = len(region._nodes) - 1
        if build:
            region._frames.append(build.value().copy())
            region._schemas.append(Optional[CsvSchema]())
            region._nodes.append(_frame_scan_node(slot=len(region._frames) - 1))
            node.right = len(region._nodes) - 1
        region._nodes.append(node^)
    return region^


def plan_unit_region(
    self: QueryContext, unit: PlacementPipeline
) raises -> QueryContext:
    """The unit as a plan over its actual inputs, to run it: its source
    read, or the child units' results. Children run first, so a child
    placed on a device is already a frame in `self`; a child on the CPU
    runs here, once, since nothing reads below this unit afterwards."""
    var input = self._execute(
        unit.input, False
    ) if unit.input >= 0 else self._execute(unit.top, False)
    var build = Optional[DataFrame]()
    if unit.sink >= 0 and self._nodes[unit.sink].right >= 0:
        build = self._execute(self._nodes[unit.sink].right, False)
    return _region(self, input, unit.steps, unit.sink, build)


def plan_substitute(
    mut self: QueryContext, unit: PlacementPipeline, var result: DataFrame
):
    """Replace the unit's top with its result, so the rest of the plan
    reads it as an in-memory frame."""
    plan_substitute_node(self, unit.top, result^)


def plan_substitute_node(
    mut self: QueryContext, index: Int, var result: DataFrame
):
    """Replace node `index` with a frame already computed for it."""
    self._frames.append(result^)
    self._schemas.append(Optional[CsvSchema]())
    self._nodes[index] = _frame_scan_node(slot=len(self._frames) - 1)


def _unit(
    self: QueryContext, top: Int, mut units: List[PlacementPipeline]
) raises -> Int:
    # A plan that is only a scan is a unit reading that scan.
    var scan = _is_scan(self._nodes[top].kind)
    var sink = -1 if scan or _is_step(self, top) else top
    var steps = List[Int]()
    var cursor = top if sink < 0 else self._nodes[top].left
    while cursor >= 0 and _is_step(self, cursor):
        steps.append(cursor)
        cursor = self._nodes[cursor].left
    var children = List[Int]()
    var source = -1
    if cursor >= 0:
        if _is_scan(self._nodes[cursor].kind):
            source = cursor
        else:
            children.append(_unit(self, cursor, units))
    var right = -1
    if sink >= 0 and self._nodes[sink].right >= 0:
        right = _unit(self, self._nodes[sink].right, units)
        children.append(right)

    # The unit as a plan over its inputs.
    var input = _input(self, cursor) if cursor >= 0 else self._execute(
        top, True
    )
    var build = Optional[DataFrame]()
    if right >= 0:
        build = self._execute(self._nodes[sink].right, True)
    var region = _region(self, input, steps, sink, build)
    var key = String("pipeline:")
    if sink >= 0:
        key += _kind_name(self._nodes[sink].kind)
    else:
        key += "steps"
    for s in steps:
        key += ":" + _kind_name(self._nodes[s].kind)
    for column in input._columns:
        key += ":" + column.dtype().name()

    units.append(
        PlacementPipeline(
            top,
            sink,
            steps^,
            cursor,
            source,
            children^,
            region^,
            key^,
            PipelineFeatures(-1, -1, -1, -1, -1),
            -1,
            -1,
            -1,
        )
    )
    return len(units) - 1
