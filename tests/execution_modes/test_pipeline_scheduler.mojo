"""Pipelines scheduled as events run each pipeline's stages in order, start a
pipeline only after the pipelines it depends on complete, finish sibling
join builds only after all of them executed, run independent pipelines at
the same time, survive nested parallel work inside a task, run the tasks a
sink's Finalize adds, round after round, before the finish event finishes,
and report the first task error."""
# contract-group: execution.lifecycle
from std.atomic import Atomic
from std.memory import ArcPointer
from std.testing import assert_equal, assert_raises, assert_true
from std.time import perf_counter_ns

from dataframe.parallel import Job, configured_workers, run_jobs
from dataframe.scheduler import (
    META_JOIN_BUILD,
    META_REGULAR,
    MetaPipeline,
    SchedulablePipeline,
    build_pipeline_schedule,
    run_pipelines,
)


struct Clock(Movable):
    var tick: Atomic[Int64]
    var started: Atomic[Int64]

    def __init__(out self):
        self.tick = Atomic[Int64](0)
        self.started = Atomic[Int64](0)


struct SumJob(Job):
    var value: Int
    var result: Int

    def __init__(out self, value: Int):
        self.value = value
        self.result = 0

    def run(mut self) raises:
        self.result = self.value * 2


struct Recorded(SchedulablePipeline):
    """Records when each of its stages ran on a shared clock."""

    var clock: ArcPointer[Clock]
    var tasks: Int
    var initialized: Int
    var executed: List[Int]
    var prepared: Int
    var finalized: Int
    # Wait, in execute, until this many pipelines have started executing
    # (bounded), and record whether they had.
    var meet: Int
    var met: Bool
    var fail: Bool
    # Nested parallel work inside the execute task.
    var nested: Bool
    var nested_total: Int
    # Tasks its Finalize inserts per round (a join build's finalize
    # event), the rounds, when each task ran (round-major), and when each
    # round's serial step ran.
    var finals: Int
    var rounds: Int
    var round: Int
    var final_ticks: List[Int]
    var round_ticks: List[Int]
    var finished_finals: Int

    def __init__(
        out self,
        clock: ArcPointer[Clock],
        tasks: Int = 1,
        meet: Int = 0,
        fail: Bool = False,
        nested: Bool = False,
        finals: Int = 0,
        rounds: Int = 1,
    ):
        self.clock = clock.copy()
        self.tasks = tasks
        self.initialized = -1
        self.executed = List[Int](length=tasks, fill=-1)
        self.prepared = -1
        self.finalized = -1
        self.meet = meet
        self.met = False
        self.fail = fail
        self.nested = nested
        self.nested_total = 0
        self.finals = finals
        self.rounds = rounds
        self.round = 0
        self.final_ticks = List[Int](length=finals * rounds, fill=-1)
        self.round_ticks = List[Int]()
        self.finished_finals = -1

    def _now(self) -> Int:
        return Int(self.clock[].tick.fetch_add(1))

    def initialize(mut self) raises:
        self.initialized = self._now()

    def max_execute_tasks(self) -> Int:
        return self.tasks

    def execute_tasks(mut self) raises -> Int:
        return self.tasks

    def execute(mut self, task: Int) raises:
        _ = self.clock[].started.fetch_add(1)
        if self.meet > 0:
            var deadline = perf_counter_ns() + 5_000_000_000
            while (
                self.clock[].started.load() < Int64(self.meet)
                and perf_counter_ns() < deadline
            ):
                pass
            self.met = self.clock[].started.load() >= Int64(self.meet)
        if self.nested:
            var jobs = List[SumJob]()
            for k in range(16):
                jobs.append(SumJob(k))
            run_jobs(jobs)
            for k in range(len(jobs)):
                self.nested_total += jobs[k].result
            var inner = List[Recorded]()
            var metas = List[MetaPipeline]()
            for _ in range(3):
                inner.append(Recorded(ArcPointer(Clock())))
                metas.append(MetaPipeline(META_JOIN_BUILD))
            run_pipelines(inner, metas)
            for k in range(len(inner)):
                assert_true(inner[k].finalized >= 0)
        self.executed[task] = self._now()

    def prepare_finalize(mut self) raises:
        self.prepared = self._now()

    def finalize(mut self) raises:
        if self.fail:
            raise Error("finalize failed on purpose")
        self.finalized = self._now()

    def max_finalize_tasks(self) -> Int:
        return self.finals

    def finalize_tasks(mut self) raises -> Int:
        if self.finals == 0 or self.round == self.rounds:
            return 0
        self.round += 1
        return self.finals

    def finalize_task(mut self, task: Int) raises:
        self.final_ticks[(self.round - 1) * self.finals + task] = self._now()

    def finish_finalize(mut self) raises:
        self.finished_finals = self._now()
        self.round_ticks.append(self.finished_finals)


def _last_execute(p: Recorded) -> Int:
    var last = -1
    for t in p.executed:
        last = max(last, t)
    return last


def _first_execute(p: Recorded) -> Int:
    var first = Int.MAX
    for t in p.executed:
        first = min(first, t)
    return first


def test_stages_follow_dependencies_and_sibling_builds_finish_together() raises:
    """Pipelines 0-2 are sibling join builds under one parent, 3 a regular
    pipeline that reads pipeline 0's sink, 4 a build under another parent:
    every pipeline's stages run in order and each runs exactly once, 3
    executes only after 0 finalized, and no sibling of 0-2 prepares its
    finish before all three executed."""
    var clock = ArcPointer(Clock())
    var pipelines = List[Recorded]()
    var metas = List[MetaPipeline]()
    for _ in range(3):
        pipelines.append(Recorded(clock, tasks=3))
        metas.append(MetaPipeline(META_JOIN_BUILD, parent=-1))
    pipelines.append(Recorded(clock, tasks=2))
    metas.append(MetaPipeline(META_REGULAR, parent=-1, dependencies=[0]))
    pipelines.append(Recorded(clock))
    metas.append(MetaPipeline(META_JOIN_BUILD, parent=3))
    run_pipelines(pipelines, metas)
    var latest_execute = -1
    var earliest_prepare = Int.MAX
    for p in range(len(pipelines)):
        ref q = pipelines[p]
        assert_true(q.initialized >= 0)
        assert_true(q.initialized < _first_execute(q))
        assert_true(_last_execute(q) < q.prepared)
        assert_true(q.prepared < q.finalized)
        if p < 3:
            latest_execute = max(latest_execute, _last_execute(q))
            earliest_prepare = min(earliest_prepare, q.prepared)
    assert_true(latest_execute < earliest_prepare)
    assert_true(pipelines[0].finalized < _first_execute(pipelines[3]))
    # Five pipelines' stages, every one once: 4 initialize, prepare and
    # finalize ticks plus one per execute task.
    assert_equal(Int(clock[].tick.load()), 5 * 3 + 3 * 3 + 2 + 1)


def test_a_finish_group_runs_after_the_base_executes_and_before_it_finishes() raises:
    """Pipelines 1 and 2 are finish-group members of 0 and 1: each executes
    after the one it names, and pipeline 0 prepares its finish only after
    both executed."""
    var clock = ArcPointer(Clock())
    var pipelines = List[Recorded]()
    var metas = List[MetaPipeline]()
    pipelines.append(Recorded(clock, tasks=2))
    metas.append(MetaPipeline(META_REGULAR))
    pipelines.append(Recorded(clock, tasks=2))
    metas.append(MetaPipeline(META_REGULAR, finish_group=0))
    pipelines.append(Recorded(clock, tasks=2))
    metas.append(MetaPipeline(META_REGULAR, finish_group=1))
    run_pipelines(pipelines, metas)
    assert_true(_last_execute(pipelines[0]) < _first_execute(pipelines[1]))
    assert_true(_last_execute(pipelines[1]) < _first_execute(pipelines[2]))
    assert_true(_last_execute(pipelines[2]) < pipelines[0].prepared)


def test_finalize_tasks_run_before_the_finish_event_finishes() raises:
    """Pipeline 0's Finalize inserts four tasks (a join build's finalize
    event): each runs after Finalize, the serial finish step after all of
    them, and pipeline 1, which reads 0's sink, executes only after."""
    var clock = ArcPointer(Clock())
    var pipelines = List[Recorded]()
    var metas = List[MetaPipeline]()
    pipelines.append(Recorded(clock, finals=4))
    metas.append(MetaPipeline(META_JOIN_BUILD, parent=1))
    pipelines.append(Recorded(clock, tasks=2))
    metas.append(MetaPipeline(META_REGULAR, dependencies=[0]))
    run_pipelines(pipelines, metas)
    ref build = pipelines[0]
    for t in build.final_ticks:
        assert_true(build.finalized < t)
        assert_true(t < build.finished_finals)
    assert_true(build.finished_finals < _first_execute(pipelines[1]))


def test_finalize_rounds_run_one_after_another() raises:
    """Pipeline 0's finish runs 40 rounds of 8 tasks, many more tasks than
    the queue holds at once: every round's tasks run after the round
    before ended and before its own serial step, and pipeline 1, which
    reads 0's sink, executes only after the last round."""
    var clock = ArcPointer(Clock())
    var pipelines = List[Recorded]()
    var metas = List[MetaPipeline]()
    pipelines.append(Recorded(clock, finals=8, rounds=40))
    metas.append(MetaPipeline(META_JOIN_BUILD, parent=1))
    pipelines.append(Recorded(clock, tasks=2))
    metas.append(MetaPipeline(META_REGULAR, dependencies=[0]))
    run_pipelines(pipelines, metas)
    ref build = pipelines[0]
    assert_equal(len(build.round_ticks), 40)
    for r in range(40):
        var before = build.finalized if r == 0 else build.round_ticks[r - 1]
        for t in range(8):
            var tick = build.final_ticks[r * 8 + t]
            assert_true(before < tick)
            assert_true(tick < build.round_ticks[r])
    assert_true(build.finished_finals < _first_execute(pipelines[1]))


def test_independent_pipelines_run_at_the_same_time() raises:
    """Two pipelines with no dependency between them each wait in their
    execute task until both have started: they meet only when they run at
    once, as independent join builds do."""
    if configured_workers() < 2:
        return
    var clock = ArcPointer(Clock())
    var pipelines = List[Recorded]()
    var metas = List[MetaPipeline]()
    for _ in range(2):
        pipelines.append(Recorded(clock, meet=2))
        metas.append(MetaPipeline(META_JOIN_BUILD))
    run_pipelines(pipelines, metas)
    assert_true(pipelines[0].met)
    assert_true(pipelines[1].met)


def test_nested_parallel_work_inside_tasks_completes() raises:
    """Execute tasks that run jobs and schedules of their own, while their
    siblings wait on the crew, all complete: a waiting thread never takes
    an executor's worker whose schedule waits for that thread's task."""
    var clock = ArcPointer(Clock())
    var pipelines = List[Recorded]()
    var metas = List[MetaPipeline]()
    for _ in range(4):
        pipelines.append(Recorded(clock, nested=True))
        metas.append(MetaPipeline(META_JOIN_BUILD))
    run_pipelines(pipelines, metas)
    for p in range(len(pipelines)):
        assert_equal(pipelines[p].nested_total, 2 * (15 * 16 // 2))
        assert_true(pipelines[p].finalized >= 0)


def test_the_first_task_error_reaches_the_caller() raises:
    var clock = ArcPointer(Clock())
    var pipelines = List[Recorded]()
    var metas = List[MetaPipeline]()
    pipelines.append(Recorded(clock))
    metas.append(MetaPipeline(META_JOIN_BUILD))
    pipelines.append(Recorded(clock, fail=True))
    metas.append(MetaPipeline(META_JOIN_BUILD))
    with assert_raises(contains="finalize failed on purpose"):
        run_pipelines(pipelines, metas)
    # The schedule is reusable after a failure.
    var again = List[Recorded]()
    again.append(Recorded(clock))
    run_pipelines(again, [MetaPipeline(META_REGULAR)])
    assert_true(again[0].finalized >= 0)


def test_a_cyclic_schedule_is_rejected() raises:
    var metas = List[MetaPipeline]()
    metas.append(MetaPipeline(META_REGULAR, dependencies=[1]))
    metas.append(MetaPipeline(META_REGULAR, dependencies=[0]))
    with assert_raises(contains="Cyclic dependency"):
        _ = build_pipeline_schedule(metas)


def test_serial_scheduling_runs_every_stage_on_the_caller() raises:
    var clock = ArcPointer(Clock())
    var pipelines = List[Recorded]()
    var metas = List[MetaPipeline]()
    for _ in range(3):
        pipelines.append(Recorded(clock, tasks=2))
        metas.append(MetaPipeline(META_JOIN_BUILD))
    run_pipelines(pipelines, metas, parallel=False)
    for p in range(len(pipelines)):
        assert_true(pipelines[p].finalized >= 0)
    assert_equal(Int(clock[].tick.load()), 3 * 3 + 3 * 2)
