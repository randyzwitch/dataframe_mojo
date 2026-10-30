"""Device placement: the exact program against brute force, the cases it
must get right, how a plan is cut into units, and what explain reports."""
# contract-group: planner.device_placement
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from dataframe import (
    Column,
    DataFrame,
    LazyFrame,
    Series,
    col,
    int_range,
    len as row_len,
)
from dataframe.devices import (
    ByteCost,
    DeviceInfo,
    MEMORY_DISCRETE,
    PipelineFeatures,
    TransferModel,
)
from dataframe.frame import DataFrame as Frame
from dataframe.dtype import DataType
from std.memory import Layout, Allocation, alloc, dealloc
from dataframe.lazy import AcceleratorBackend
from dataframe.planning.device_placement import (
    CPU,
    PlacementProblem,
    plan_pipeline_cuts,
    plan_pipeline_estimates,
    plan_pipelines,
    solve_placement,
)
from dataframe.lazy_api import _solve_devices
from tests.support.group_strategy_reference import ScopedEnvironment
from std.atomic import Atomic
from std.memory import ArcPointer, Pointer, unsafe_memcpy
from std.sys import size_of
from dataframe.storage.buffer import _Buffer
from dataframe.storage.residence import (
    _CopyToHost,
    _DeviceResidence,
    _ReleaseDevice,
)

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


def test_length_changing_select_has_unknown_device_capacity() raises:
    var frame = DataFrame([Series("k", Column[Int64]([1, 1]))])
    for expression in [
        int_range(0, 100).alias("expanded"),
        int_range(0, 100)
        .over("k", mapping_strategy="explode")
        .alias("expanded"),
    ]:
        var units = plan_pipelines(frame.lazy().select(expression)._plan)
        ref unit = units[len(units) - 1]
        assert_equal(unit.features.input_rows, 2)
        assert_equal(unit.features.output_rows, -1)
        assert_equal(unit.output_bytes, -1)
        assert_equal(unit.working_bytes, -1)
        # A measured, cheap device still cannot claim a finite budget can
        # hold an output whose expansion is unknown.
        var problem = PlacementProblem(1, 2)
        problem.run_ns[0][CPU] = 1000
        problem.run_ns[0][1] = 1
        problem.working_bytes[0] = unit.working_bytes
        problem.budgets[1] = 1024
        assert_false(problem.allowed(0, 1))
        assert_equal(solve_placement(problem).devices, [CPU])


def test_empty_scalar_select_accounts_for_its_one_output_row() raises:
    var frame = DataFrame([Series("k", Column[Int64](List[Int64]()))])
    var units = plan_pipelines(frame.lazy().select(row_len().alias("n"))._plan)
    ref unit = units[len(units) - 1]
    assert_equal(unit.features.input_rows, 0)
    assert_equal(unit.features.output_rows, 1)
    assert_true(unit.output_bytes > 0)
    assert_equal(unit.working_bytes, unit.output_bytes)
    var problem = PlacementProblem(1, 2)
    problem.run_ns[0][CPU] = 1000
    problem.run_ns[0][1] = 1
    problem.working_bytes[0] = unit.working_bytes
    problem.budgets[1] = 0
    assert_false(problem.allowed(0, 1))
    problem.budgets[1] = unit.working_bytes
    assert_true(problem.allowed(0, 1))


struct Runs(Movable):
    """What the fake device was asked to do."""

    var pipelines: Int
    var copies: Atomic[Int64]
    var fail: Bool
    # Device memory: every Float64 result column the fake device returned.
    var memory: List[List[Float64]]

    def __init__(out self, fail: Bool = False):
        self.pipelines = 0
        self.copies = Atomic[Int64](0)
        self.fail = fail
        self.memory = List[List[Float64]]()


def runs_at(address: Int) -> Pointer[Runs, MutUntrackedOrigin]:
    return Pointer[Runs, MutUntrackedOrigin](unsafe_from_address=address)


struct Held(Movable):
    var runs: Int
    var values: List[Float64]

    def __init__(out self, runs: Int, var values: List[Float64]):
        self.runs = runs
        self.values = values^


def held_to_host(context: Int, address: Int, bytes: Int) abi("C") -> Int:
    var held = Pointer[Held, MutUntrackedOrigin](unsafe_from_address=context)
    _ = runs_at(held[].runs)[].copies.fetch_add(1)
    unsafe_memcpy(
        dest=Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=address),
        src=Pointer[UInt8, MutUntrackedOrigin](
            unsafe_from_address=Int(held[].values.unsafe_ptr())
        ),
        count=bytes,
    )
    return 0


def held_release(context: Int) abi("C"):
    var pointer = Pointer[Held, MutUntrackedOrigin](unsafe_from_address=context)
    _ = pointer.unsafe_take_pointee()^
    dealloc(
        Allocation[Held](
            unsafe_owned_ptr=pointer, layout=Layout[Held](length=1)
        )
    )


def on_device(column: Column[Float64], runs: Int) raises -> Column[Float64]:
    """The column's values moved into fake device memory."""
    var values = List[Float64]()
    for i in range(len(column)):
        values.append(column._get(i))
    var slot = alloc(Layout[Held](length=1)).unsafe_leak()
    var length = len(values)
    slot.unsafe_write(Held(runs, values^))
    var copy: _CopyToHost = held_to_host
    var release: _ReleaseDevice = held_release
    var residence = ArcPointer(
        _DeviceResidence(
            backend="fake",
            device=0,
            context=Int(slot),
            bytes=length * size_of[Float64](),
            to_host=Pointer(to=copy).unsafe_bitcast[Int]()[],
            release=Pointer(to=release).unsafe_bitcast[Int]()[],
        )
    )
    return Column[Float64]._from_buffers(
        _Buffer[Float64].on_device(length, residence),
        column._bits.copy(),
        0,
        length,
    )


@fieldwise_init
struct FakeDevice(AcceleratorBackend):
    """One discrete device that runs every unit at a hundredth of a
    nanosecond per input row. It runs a unit by computing it on the CPU and
    returning its Float64 columns in fake device memory."""

    var budget: Int
    var runs: Int
    # When positive: the device's cost for a unit under 1,000 input rows,
    # as a large fixed launch cost would make it.
    var small_fixed_ns: Float64

    def run_pipeline(self, pipeline: LazyFrame, device: Int) raises -> Frame:
        ref runs = runs_at(self.runs)[]
        runs.pipelines += 1
        if runs.fail:
            raise Error("fake device fault")
        var result = pipeline.collect(engine="cpu", optimize=False)
        var columns = List[Series]()
        for series in result._columns:
            if series.dtype() == DataType.FLOAT64:
                columns.append(
                    Series(
                        series.name(),
                        on_device(series.float64(), self.runs),
                    )
                )
            else:
                columns.append(series.copy())
        return Frame(columns^)

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
        if self.small_fixed_ns > 0 and features.input_rows < 1000:
            return self.small_fixed_ns
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


def test_estimates_wait_until_a_device_can_run_a_unit() raises:
    var plan = joined()._optimized()
    var cuts = plan_pipeline_cuts(plan._plan)
    var estimated = plan_pipelines(plan._plan)
    assert_equal(len(cuts), len(estimated))
    for p in range(len(cuts)):
        assert_equal(cuts[p].key, estimated[p].key)
        assert_equal(cuts[p].features.input_rows, -1)
        assert_equal(cuts[p].output_bytes, -1)
    plan_pipeline_estimates(plan._plan, cuts)
    for p in range(len(cuts)):
        assert_equal(
            cuts[p].features.input_rows, estimated[p].features.input_rows
        )
        assert_equal(cuts[p].output_bytes, estimated[p].output_bytes)
        assert_equal(cuts[p].working_bytes, estimated[p].working_bytes)
    # No unit has a CPU cost, so none is compared and nothing is
    # estimated, though the device would run every unit.
    var path = String("build/test-device-placement-unrelated.json")
    calibrate(path, ["pipeline:unrelated"], 1e6)
    var calibrated = ScopedEnvironment("DATAFRAME_HANDOFF_CALIBRATION", path)
    var device = FakeDevice(-1, 0, 0)
    var solved = _solve_devices(plan, device, device.devices())
    for p in range(len(solved.units)):
        assert_equal(solved.units[p].features.input_rows, -1)
        assert_equal(solved.placement.devices[p], CPU)
    # Explain still reports every unit's estimates.
    solved = _solve_devices(plan, device, device.devices(), estimate_all=True)
    assert_true(solved.units[0].features.input_rows >= 0)
    _ = calibrated^


def test_estimates_cover_compared_units_and_those_below() raises:
    var plan = joined()._optimized()
    var estimated = plan_pipelines(plan._plan)
    # Only the scans' units have a CPU cost, so only they are compared.
    var leaves = List[String]()
    for unit in estimated:
        if len(unit.children) == 0:
            leaves.append(unit.key)
    var path = String("build/test-device-placement-leaves.json")
    calibrate(path, leaves, 1e6)
    var calibrated = ScopedEnvironment("DATAFRAME_HANDOFF_CALIBRATION", path)
    var device = FakeDevice(-1, 0, 0)
    var solved = _solve_devices(plan, device, device.devices())
    var skipped = 0
    for p in range(len(solved.units)):
        ref unit = solved.units[p]
        if len(unit.children) == 0:
            assert_equal(
                unit.features.input_rows, estimated[p].features.input_rows
            )
            assert_equal(unit.output_bytes, estimated[p].output_bytes)
        else:
            assert_equal(unit.features.input_rows, -1)
            skipped += 1
    assert_true(skipped > 0)
    _ = calibrated^
    # A compared unit's children are estimated with it. Costs are read
    # once per path, so this calibration has its own.
    var root = String("build/test-device-placement-root.json")
    calibrate(root, [estimated[len(estimated) - 1].key], 1e6)
    calibrated = ScopedEnvironment("DATAFRAME_HANDOFF_CALIBRATION", root)
    solved = _solve_devices(plan, device, device.devices())
    for p in range(len(solved.units)):
        assert_equal(
            solved.units[p].features.input_rows,
            estimated[p].features.input_rows,
        )
    _ = calibrated^


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
    var shown = summed.explain(accelerator=FakeDevice(-1, 0, 0), engine="auto")
    assert_true("DEVICES cpu, fake:0 Fake GPU (discrete)" in shown, shown)
    assert_true("cpu not measured" in shown, shown)
    assert_false("on fake:0" in shown, shown)
    _ = unset^
    # Each unit costs the CPU a millisecond, the device microseconds.
    var path = String("build/test-device-placement-slow-cpu.json")
    calibrate(path, keys, 1e6)
    var calibrated = ScopedEnvironment("DATAFRAME_HANDOFF_CALIBRATION", path)
    shown = summed.explain(accelerator=FakeDevice(-1, 0, 0), engine="auto")
    assert_true("on fake:0" in shown, shown)
    assert_false("on cpu" in shown, shown)
    assert_true("PLACEMENT estimated" in shown, shown)
    # A budget too small for the unit keeps it on the CPU.
    shown = summed.explain(accelerator=FakeDevice(1, 0, 0), engine="auto")
    assert_false("on fake:0" in shown, shown)
    # The join's output rows are not estimated, so the aggregate above it
    # has no CPU cost, and the join's output cannot be moved by an unknown
    # number of bytes: both stay on the CPU.
    shown = plan.explain(accelerator=FakeDevice(-1, 0, 0), engine="auto")
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


def test_collect_runs_placed_units_on_the_device() raises:
    var frames = sources()
    var summed = frames[0].lazy().filter(col("v") > 1).select(col("v").sum())
    var plan = joined()
    var keys = List[String]()
    for unit in plan_pipelines(summed._optimized()._plan):
        keys.append(unit.key)
    for unit in plan_pipelines(plan._optimized()._plan):
        keys.append(unit.key)
    var path = String("build/test-device-placement-collect.json")
    calibrate(path, keys, 1e6)
    var calibrated = ScopedEnvironment("DATAFRAME_HANDOFF_CALIBRATION", path)
    var runs = Runs()
    var device = FakeDevice(-1, Int(Pointer(to=runs)), 0)
    # The sum runs on the device; its device-resident result is copied to
    # the host once, before collect returns.
    var result = summed.collect(accelerator=device, engine="auto")
    assert_true(result.equals(summed.collect(engine="cpu")))
    assert_equal(runs.pipelines, 1)
    assert_equal(runs.copies.load(), Int64(1))
    assert_true(result.column("v").float64()._data.on_host())
    # The join plan's units above the join stay on the CPU, and its
    # result matches the CPU's.
    var joined_result = plan.collect(accelerator=device, engine="auto")
    var expected = plan.collect(engine="cpu").sort("k")
    assert_true(joined_result.sort("k").equals(expected))
    # Without devices or measured costs nothing runs on the device.
    var pipelines = runs.pipelines
    var unset = ScopedEnvironment("DATAFRAME_HANDOFF_CALIBRATION", "")
    _ = summed.collect(accelerator=device, engine="auto")
    assert_equal(runs.pipelines, pipelines)
    _ = unset^
    # A device fault propagates; the CPU does not rerun the query.
    var failing = Runs(fail=True)
    with assert_raises(contains="fake device fault"):
        _ = summed.collect(
            accelerator=FakeDevice(-1, Int(Pointer(to=failing)), 0),
            engine="auto",
        )
    assert_equal(failing.pipelines, 1)
    _ = calibrated^
    _ = runs^
    _ = failing^


def test_a_unit_is_placed_again_when_it_starts() raises:
    var values = List[Float64]()
    for i in range(200_000):
        values.append(Float64(i))
    var frame = DataFrame([Series("v", Column[Float64](values^))])
    # The filter keeps 9 rows. The sort's unit runs on the device; the sum
    # over its output is planned from more rows than it gets.
    var plan = (
        frame.lazy()
        .filter(col("v") > 199_990.0)
        .sort("v")
        .select(col("v").sum())
    )
    var keys = List[String]()
    for unit in plan_pipelines(plan._optimized()._plan):
        keys.append(unit.key)
    var path = String("build/test-device-placement-start.json")
    calibrate(path, keys, 1e6)
    var calibrated = ScopedEnvironment("DATAFRAME_HANDOFF_CALIBRATION", path)
    var runs = Runs()
    var device = FakeDevice(-1, Int(Pointer(to=runs)), 2e6)
    var shown = plan.explain(accelerator=device, engine="auto")
    assert_false("on cpu" in shown, shown)
    var result = plan.collect(accelerator=device, engine="auto")
    assert_true(result.equals(plan.collect(engine="cpu")))
    # At its start the sum has 9 rows, where the device's 2 ms fixed cost
    # loses to the CPU: only the sort ran on the device.
    assert_equal(runs.pipelines, 1)
    _ = calibrated^
    _ = runs^
