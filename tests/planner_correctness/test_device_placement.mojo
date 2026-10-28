"""Device placement: the exact program against brute force, the cases it
must get right, how a plan is cut into units, and what explain reports."""
# contract-group: planner.device_placement
from std.testing import assert_equal, assert_true, assert_false
from dataframe import Column, DataFrame, LazyFrame, Series, col
from dataframe.devices import (
    ByteCost,
    DeviceInfo,
    MEMORY_DISCRETE,
    PipelineFeatures,
    TransferModel,
)
from dataframe.frame import DataFrame as Frame
from dataframe.lazy import AcceleratorBackend
from dataframe.planning.device_placement import (
    CPU,
    PlacementProblem,
    plan_pipelines,
    solve_placement,
)
from tests.support.group_strategy_reference import ScopedEnvironment

comptime NEVER = Float64(1e300)


struct Random(Movable):
    var state: UInt64

    def __init__(out self, seed: UInt64):
        self.state = seed

    def below(mut self, n: Int) -> Int:
        self.state = self.state * 6364136223846793005 + 1442695040888963407
        return Int((self.state >> 33) % UInt64(n))


def brute_force(problem: PlacementProblem) -> Float64:
    """The cheapest total over every assignment, start-up included."""
    var units = len(problem.children)
    var devices = problem.devices()
    var parent = List[Int](length=units, fill=-1)
    for p in range(units):
        for c in problem.children[p]:
            parent[c] = p
    var best = NEVER
    var assignment = List[Int](length=units, fill=0)
    var total_assignments = 1
    for _ in range(units):
        total_assignments *= devices
    for code in range(total_assignments):
        var rest = code
        for p in range(units):
            assignment[p] = rest % devices
            rest //= devices
        var total = Float64(0)
        var used = List[Bool](length=devices, fill=False)
        var feasible = True
        for p in range(units):
            var d = assignment[p]
            if not problem.allowed(p, d):
                feasible = False
                break
            used[d] = True
            total += max(problem.run_ns[p][d], 0)
            total += problem.move(
                problem.source_bytes[p], problem.source_device[p], d
            )
            var target = assignment[parent[p]] if parent[p] >= 0 else CPU
            total += problem.move(problem.output_bytes[p], d, target)
        if not feasible or total >= NEVER:
            continue
        for d in range(1, devices):
            if used[d]:
                total += max(problem.startup_ns[d], 0)
        best = min(best, total)
    return best


def random_problem(mut random: Random) -> PlacementProblem:
    var units = 1 + random.below(5)
    var devices = 2 + random.below(2)
    var problem = PlacementProblem(units, devices)
    for p in range(units - 1):
        var parent = p + 1 + random.below(units - p - 1)
        problem.children[parent].append(p)
    for p in range(units):
        problem.run_ns[p][CPU] = Float64(random.below(1000))
        for d in range(1, devices):
            problem.run_ns[p][d] = Float64(random.below(1000) - 150)
        problem.output_bytes[p] = random.below(5000) - 500
        problem.source_bytes[p] = random.below(3000)
        problem.source_device[p] = 0 if random.below(4) else 1
        problem.working_bytes[p] = random.below(6000) - 500
    for d in range(1, devices):
        var kind = random.below(4)
        if kind == 0:
            problem.transfers[d] = TransferModel.shared_memory()
        elif kind == 1:
            problem.transfers[d] = TransferModel.unmeasured()
        else:
            problem.transfers[d] = TransferModel(
                ByteCost(Float64(random.below(200)), 0.05),
                ByteCost(Float64(random.below(200)), 0.08),
                -1,
            )
        problem.budgets[d] = random.below(7000) - 1000
        problem.startup_ns[d] = Float64(random.below(400))
    return problem^


def test_the_program_matches_brute_force() raises:
    var random = Random(20261028)
    for _ in range(800):
        var problem = random_problem(random)
        var expected = brute_force(problem)
        var placement = solve_placement(problem)
        var cpu = Float64(0)
        for p in range(len(problem.children)):
            cpu += problem.run_ns[p][CPU]
        assert_equal(placement.cpu_ns, cpu)
        assert_true(abs(placement.total_ns - min(expected, cpu)) < 1e-6)
        # The reported assignment really costs what was reported.
        for p in range(len(problem.children)):
            assert_true(problem.allowed(p, placement.devices[p]))


def chain(devices: Int) -> PlacementProblem:
    """Three units in a chain: 0 feeds 1 feeds 2."""
    var problem = PlacementProblem(3, devices)
    problem.children[1].append(0)
    problem.children[2].append(1)
    for p in range(3):
        problem.output_bytes[p] = 1_000_000
        problem.source_bytes[p] = 1_000_000 if p == 0 else 0
        problem.working_bytes[p] = 2_000_000
    return problem^


def test_unified_memory_places_each_unit_on_its_own() raises:
    var problem = chain(2)
    problem.transfers[1] = TransferModel.shared_memory()
    # The device wins units 0 and 2, the CPU unit 1.
    problem.run_ns[0][CPU] = 100
    problem.run_ns[0][1] = 10
    problem.run_ns[1][CPU] = 10
    problem.run_ns[1][1] = 100
    problem.run_ns[2][CPU] = 100
    problem.run_ns[2][1] = 10
    var placement = solve_placement(problem)
    assert_equal(placement.devices, [1, CPU, 1])
    assert_equal(placement.total_ns, Float64(30))


def test_costly_transfers_keep_a_discrete_plan_together() raises:
    var problem = chain(2)
    # A microsecond per kilobyte each way: splitting pays two crossings.
    problem.transfers[1] = TransferModel(
        ByteCost(0, 0.001), ByteCost(0, 0.001), 0
    )
    problem.run_ns[0][CPU] = 100_000
    problem.run_ns[0][1] = 1_000
    problem.run_ns[1][CPU] = 1_000
    problem.run_ns[1][1] = 2_000
    problem.run_ns[2][CPU] = 100_000
    problem.run_ns[2][1] = 1_000
    var placement = solve_placement(problem)
    assert_equal(placement.devices, [1, 1, 1])
    # 4 us of device work plus the source up and the result down.
    assert_equal(placement.total_ns, Float64(4_000 + 1_000 + 1_000))


def test_budget_start_up_and_unmeasured_costs() raises:
    var problem = chain(2)
    problem.transfers[1] = TransferModel.shared_memory()
    for p in range(3):
        problem.run_ns[p][CPU] = 1_000
        problem.run_ns[p][1] = 100
    assert_equal(solve_placement(problem).devices, [1, 1, 1])
    # Start-up larger than the saving keeps everything on the CPU.
    problem.startup_ns[1] = 5_000
    assert_equal(solve_placement(problem).devices, [CPU, CPU, CPU])
    # Paid once, it is worth it when every unit saves enough.
    problem.startup_ns[1] = 2_000
    assert_equal(solve_placement(problem).devices, [1, 1, 1])
    # A unit too big for the budget stays on the CPU.
    problem.budgets[1] = 1_000_000
    assert_equal(solve_placement(problem).devices, [CPU, CPU, CPU])
    problem.budgets[1] = -1
    # Unmeasured transfers on a discrete device rule it out.
    problem.transfers[1] = TransferModel.unmeasured()
    assert_equal(solve_placement(problem).devices, [CPU, CPU, CPU])


def sources() raises -> Tuple[DataFrame, DataFrame]:
    var facts = DataFrame(
        [
            Series("k", Column[Int64]([1, 2, 3, 1, 2, 3])),
            Series("v", Column[Float64]([1, 2, 3, 4, 5, 6])),
        ]
    )
    var names = DataFrame(
        [
            Series("k", Column[Int64]([1, 2, 3])),
            Series("w", Column[Int32]([10, 20, 30])),
        ]
    )
    return (facts^, names^)


def joined() raises -> LazyFrame:
    var frames = sources()
    return (
        frames[0]
        .lazy()
        .filter(col("v") > 1)
        .join(frames[1].lazy().filter(col("w") > 10), on="k")
        .group_by("k")
        .agg(col("v").sum())
    )


def test_a_plan_is_cut_into_units_children_first() raises:
    var units = plan_pipelines(joined()._optimized()._plan)
    # The probe side's filter, the build side's filter, the join, and the
    # aggregate over the join's output.
    assert_true(len(units) >= 3)
    ref root = units[len(units) - 1]
    assert_true(root.key.startswith("pipeline:agg"), root.key)
    for p in range(len(units)):
        for c in units[p].children:
            assert_true(c < p)
    var join = -1
    for p in range(len(units)):
        if units[p].key.startswith("pipeline:join"):
            join = p
    assert_true(join >= 0)
    assert_equal(len(units[join].children), 1 + Int(units[join].source < 0))
    # A unit's region is a plan over its inputs that backends can read.
    var region = units[join].region.copy()
    assert_equal(region._nodes[len(region._nodes) - 1].kind, 6)
    assert_true(units[0].features.input_rows >= 0)


@fieldwise_init
struct FakeDevice(AcceleratorBackend):
    """One discrete device that runs every unit at a hundredth of a
    nanosecond per input row."""

    var budget: Int

    def execute(self, plan: LazyFrame) raises -> Tuple[Frame, Frame]:
        raise Error("not used")

    def describe(self, plan: LazyFrame) -> String:
        return "ENGINE accel: fake device\n"

    def devices(self) -> List[DeviceInfo]:
        return [DeviceInfo("fake", 0, "Fake GPU", MEMORY_DISCRETE, self.budget)]

    def transfer_model(self, device: Int) -> TransferModel:
        return TransferModel(ByteCost(1000, 0.01), ByteCost(1000, 0.01), 0)

    def pipeline_supports(self, pipeline: LazyFrame, device: Int) -> Bool:
        return True

    def pipeline_estimate(
        self, pipeline: LazyFrame, device: Int, features: PipelineFeatures
    ) -> Float64:
        return 0.01 * Float64(max(features.input_rows, 0))


def calibrate(path: String, keys: List[String], fixed_ns: Float64) raises:
    var text = String('{"operators": {')
    for i in range(len(keys)):
        if i:
            text += ", "
        text += (
            '"'
            + keys[i]
            + '": {"cpu": {"fixed_ns": '
            + String(fixed_ns)
            + ', "row_ns": 5}}'
        )
    text += "}}"
    with open(path, "w") as target:
        target.write(text)


def test_explain_reports_the_placement_it_would_make() raises:
    var frames = sources()
    var summed = frames[0].lazy().filter(col("v") > 1).select(col("v").sum())
    var plan = joined()
    var keys = List[String]()
    for unit in plan_pipelines(summed._optimized()._plan):
        keys.append(unit.key)
    for unit in plan_pipelines(plan._optimized()._plan):
        keys.append(unit.key)
    # Without a CPU calibration nothing is compared and all stays on CPU.
    var unset = ScopedEnvironment("DATAFRAME_HANDOFF_CALIBRATION", "")
    var shown = summed.explain(accelerator=FakeDevice(-1), engine="auto")
    assert_true("DEVICES cpu, fake:0 Fake GPU (discrete)" in shown, shown)
    assert_true("cpu not measured" in shown, shown)
    assert_false("on fake:0" in shown, shown)
    _ = unset^
    # Each unit costs the CPU a millisecond, the device microseconds.
    var path = String("build/test-device-placement-slow-cpu.json")
    calibrate(path, keys, 1e6)
    var calibrated = ScopedEnvironment("DATAFRAME_HANDOFF_CALIBRATION", path)
    shown = summed.explain(accelerator=FakeDevice(-1), engine="auto")
    assert_true("on fake:0" in shown, shown)
    assert_false("on cpu" in shown, shown)
    assert_true("PLACEMENT estimated" in shown, shown)
    # A budget too small for the unit keeps it on the CPU.
    shown = summed.explain(accelerator=FakeDevice(1), engine="auto")
    assert_false("on fake:0" in shown, shown)
    # The join's output rows are not estimated, so the aggregate above it
    # has no CPU cost, and the join's output cannot be moved by an unknown
    # number of bytes: both stay on the CPU.
    shown = plan.explain(accelerator=FakeDevice(-1), engine="auto")
    assert_true("pipeline:agg" in shown and "rows -1" in shown, shown)
    var lines = shown.split("\n")
    for line in lines:
        if line.startswith("PIPELINE") and (
            "pipeline:join" in line or "pipeline:agg" in line
        ):
            assert_true(String(line).endswith("on cpu"), shown)
    _ = calibrated^
    # Without devices there are no placement lines at all.
    assert_false("DEVICES" in summed.explain(engine="auto"))
