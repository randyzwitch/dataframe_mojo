"""Pipelines scheduled by events, ported from DuckDB 2.0's executor
(src/parallel: executor.cpp, event.cpp, pipeline_schedule.cpp,
pipeline_initialize_event.cpp, pipeline_event.cpp,
pipeline_prepare_finish_event.cpp, pipeline_finish_event.cpp,
pipeline_complete_event.cpp).

A query's independent pipelines -- the build sides of a stream's joins,
the inputs of an eager join, shared subplans -- are scheduled as DuckDB
schedules its MetaPipelines (`BuildPipelineSchedule`). Each pipeline
becomes five events: initialize, execute, prepare finish, finish and
complete, each depending on the one before. A pipeline that depends on
another (its source is the other's sink) executes only after the other
completes, and sibling join builds under one parent all execute before
any prepares its finish, and all prepare before any finishes. An event
whose dependencies are done schedules its tasks on one queue
(`Event::CompleteDependency`, `Event::SetTasks`); an event with no
tasks finishes at once, and finishing completes a dependency of each
parent (`Event::Finish`). The query is done when every pipeline's
complete event has finished (`Executor::ExecutionIsFinished`).

The crew's threads and the caller take tasks from that queue until then
(`Executor::WorkOnTasks`), so a pipeline never waits for another it does
not depend on, and independent builds run at the same time. A thread
with no task queued runs the jobs that running tasks post to the crew
(`_Crew._help`): until every operator runs as a task of its pipeline, a
task's own morsels still run through `run_jobs`. The workers themselves
are posted so that only idle crew threads join them (`_UNHELPABLE`).
"""
from std.atomic import Atomic
from std.ffi import external_call
from std.memory import Pointer

from dataframe.parallel import (
    Job,
    _Crew,
    _current_crew,
    _run_jobs,
    configured_workers,
)

# PipelineScheduleStageType (pipeline_schedule.hpp).
comptime STAGE_INITIALIZE = 0
comptime STAGE_EXECUTE = 1
comptime STAGE_PREPARE_FINISH = 2
comptime STAGE_FINISH = 3
comptime STAGE_COMPLETE = 4

# MetaPipelineType (meta_pipeline.hpp): a join's build side is a
# JOIN_BUILD child of the pipeline that probes it.
comptime META_REGULAR = 0
comptime META_JOIN_BUILD = 1

# A queued task: its event in the high 32 bits, its index in the low.
comptime _TASK_BITS = 32


trait SchedulablePipeline(Deinitable, Movable):
    """A pipeline's work, as its events' tasks run it."""

    def initialize(mut self) raises:
        """`PipelineInitializeTask`: reset the sink."""
        ...

    def execute_tasks(self) -> Int:
        """How many tasks `Pipeline::Schedule` makes for the execute
        event (at least one)."""
        ...

    def execute(mut self, task: Int) raises:
        """One execute task: run the pipeline's source into its sink."""
        ...

    def prepare_finalize(mut self) raises:
        """`PipelinePreFinishTask`: `Sink::PrepareFinalize`."""
        ...

    def finalize(mut self) raises:
        """`PipelineFinishTask`: `Sink::Finalize`."""
        ...


struct MetaPipeline(Copyable, Movable):
    """One MetaPipeline of a schedule, holding one pipeline (its base).

    `parent` names the MetaPipeline whose pipeline reads this one's sink
    (-1 for the caller's own pipeline, which runs after the schedule);
    JOIN_BUILD children of one parent are siblings. `dependencies` are the
    pipelines (by index) that must complete before this one executes
    (`Pipeline::GetDependencies`)."""

    var type: Int
    var parent: Int
    var dependencies: List[Int]

    def __init__(
        out self,
        type: Int,
        parent: Int = -1,
        dependencies: List[Int] = List[Int](),
    ):
        self.type = type
        self.parent = parent
        self.dependencies = dependencies.copy()


struct PipelineScheduleStage(Copyable, Movable):
    """One event to create: its stage, its pipeline, and the stages it
    waits for."""

    var type: Int
    var pipeline: Int
    var dependencies: List[Int]

    def __init__(out self, type: Int, pipeline: Int):
        self.type = type
        self.pipeline = pipeline
        self.dependencies = List[Int]()


def _add_stage(
    mut stages: List[PipelineScheduleStage], type: Int, pipeline: Int
) -> Int:
    stages.append(PipelineScheduleStage(type, pipeline))
    return len(stages) - 1


def build_pipeline_schedule(
    metas: List[MetaPipeline],
) raises -> List[PipelineScheduleStage]:
    """`BuildPipelineSchedule` for MetaPipelines of one pipeline each:
    stage `5p + s` is pipeline p's stage s. Raises on a cycle."""
    var stages = List[PipelineScheduleStage](capacity=5 * len(metas))
    for p in range(len(metas)):
        # AddBasePipeline: initialize -> execute -> prepare finish ->
        # finish -> complete.
        var initialize = _add_stage(stages, STAGE_INITIALIZE, p)
        var execute = _add_stage(stages, STAGE_EXECUTE, p)
        var prepare_finish = _add_stage(stages, STAGE_PREPARE_FINISH, p)
        var finish = _add_stage(stages, STAGE_FINISH, p)
        var complete = _add_stage(stages, STAGE_COMPLETE, p)
        stages[execute].dependencies.append(initialize)
        stages[prepare_finish].dependencies.append(execute)
        stages[finish].dependencies.append(prepare_finish)
        stages[complete].dependencies.append(finish)
    for p in range(len(metas)):
        # A pipeline dependency is complete only after its sink has
        # finalized.
        for d in metas[p].dependencies:
            if d < 0 or d >= len(metas) or d == p:
                raise Error("pipeline dependency out of range")
            stages[5 * p + STAGE_EXECUTE].dependencies.append(
                5 * d + STAGE_COMPLETE
            )
    # Sibling join builds: all execute through Combine before any runs
    # PrepareFinalize, and all prepare before any runs Finalize, so the
    # memory every build needs is known when they finalize.
    for a in range(len(metas)):
        if metas[a].type != META_JOIN_BUILD:
            continue
        for b in range(len(metas)):
            if (
                b == a
                or metas[b].type != META_JOIN_BUILD
                or metas[b].parent != metas[a].parent
            ):
                continue
            stages[5 * a + STAGE_PREPARE_FINISH].dependencies.append(
                5 * b + STAGE_EXECUTE
            )
            stages[5 * a + STAGE_FINISH].dependencies.append(
                5 * b + STAGE_PREPARE_FINISH
            )
    if _has_cycle(stages):
        raise Error("Cyclic dependency in pipeline schedule")
    return stages^


def _has_cycle(stages: List[PipelineScheduleStage]) -> Bool:
    """`PipelineSchedule::HasCycle`: a depth-first search over the
    dependencies (0 unvisited, 1 on the path, 2 done)."""
    var states = List[Int](length=len(stages), fill=0)
    for start in range(len(stages)):
        if states[start] != 0:
            continue
        var path = List[Int]()
        var next = List[Int]()
        path.append(start)
        next.append(0)
        states[start] = 1
        while len(path) > 0:
            var at = path[len(path) - 1]
            var k = next[len(next) - 1]
            if k == len(stages[at].dependencies):
                states[at] = 2
                _ = path.pop()
                _ = next.pop()
                continue
            next[len(next) - 1] = k + 1
            var dependency = stages[at].dependencies[k]
            if states[dependency] == 1:
                return True
            if states[dependency] == 0:
                states[dependency] = 1
                path.append(dependency)
                next.append(0)
    return False


def _counter(base: Int, i: Int) -> ref[MutAnyOrigin] Atomic[Int64]:
    return Pointer[Atomic[Int64], MutAnyOrigin](
        unsafe_from_address=base + 8 * i
    )[]


struct _Executor[P: SchedulablePipeline](Movable):
    """The events of one schedule and the queue of their tasks. Lives on
    the caller's stack while the crew works on it; workers reach it by
    address."""

    # The caller's List[P], which stays put while the events run.
    var pipelines: Int
    var stage: List[Int]
    var pipeline: List[Int]
    var parents: List[List[Int]]
    var total_dependencies: List[Int]
    var total_tasks: List[Int]
    # Atomics, one per event: dependencies finished, tasks finished.
    var finished_dependencies: List[Int64]
    var finished_tasks: List[Int64]
    # The task queue: a slot per task the events make, 0 until a task is
    # published there; `reserved` slots are taken, `head` is the next to
    # run (`ConcurrentQueue`, fixed at the total task count).
    var queue: List[Int64]
    var reserved: Atomic[Int64]
    var head: Atomic[Int64]
    var total_pipelines: Int
    var completed_pipelines: Atomic[Int64]
    var failed: Atomic[Int64]
    var error: String

    def __init__(
        out self,
        mut pipelines: List[Self.P],
        stages: List[PipelineScheduleStage],
    ):
        self.pipelines = Int(Pointer(to=pipelines))
        self.stage = List[Int](capacity=len(stages))
        self.pipeline = List[Int](capacity=len(stages))
        self.parents = List[List[Int]](capacity=len(stages))
        self.total_dependencies = List[Int](capacity=len(stages))
        self.total_tasks = List[Int](capacity=len(stages))
        var capacity = 0
        for s in stages:
            self.stage.append(s.type)
            self.pipeline.append(s.pipeline)
            self.parents.append(List[Int]())
            self.total_dependencies.append(len(s.dependencies))
            var tasks = 0
            if s.type == STAGE_EXECUTE:
                tasks = max(1, pipelines[s.pipeline].execute_tasks())
            elif s.type != STAGE_COMPLETE:
                # Initialize, prepare finish and finish each make one task;
                # complete makes none and finishes when scheduled.
                tasks = 1
            self.total_tasks.append(tasks)
            capacity += tasks
        # `Event::AddDependency`: the dependency learns its parent.
        for e in range(len(stages)):
            for d in stages[e].dependencies:
                self.parents[d].append(e)
        self.finished_dependencies = List[Int64](length=len(stages), fill=0)
        self.finished_tasks = List[Int64](length=len(stages), fill=0)
        self.queue = List[Int64](length=max(capacity, 1), fill=0)
        self.reserved = Atomic[Int64](0)
        self.head = Atomic[Int64](0)
        self.total_pipelines = len(pipelines)
        self.completed_pipelines = Atomic[Int64](0)
        self.failed = Atomic[Int64](0)
        self.error = String()

    def _pipelines(self) -> ref[MutAnyOrigin] List[Self.P]:
        return Pointer[List[Self.P], MutAnyOrigin](
            unsafe_from_address=self.pipelines
        )[]

    def schedule_events(mut self):
        """`ScheduleEventsInternal`: schedule the events without
        dependencies."""
        for e in range(len(self.stage)):
            if self.total_dependencies[e] == 0:
                self._schedule(e)

    def finished(self) -> Bool:
        """`Executor::ExecutionIsFinished`."""
        return (
            self.completed_pipelines.load() >= Int64(self.total_pipelines)
            or self.failed.load() != 0
        )

    def _schedule(mut self, event: Int):
        """`Event::Schedule` then `SetTasks`; an event without tasks
        finishes now (`AutoFinishWithoutTasks`)."""
        var tasks = self.total_tasks[event]
        if tasks == 0:
            self._finish(event)
            return
        for t in range(tasks):
            var at = Int(self.reserved.fetch_add(1))
            _counter(Int(self.queue.unsafe_ptr()), at).store(
                Int64((event << _TASK_BITS) | t) + 1
            )

    def _complete_dependency(mut self, event: Int):
        """`Event::CompleteDependency`."""
        var base = Int(self.finished_dependencies.unsafe_ptr())
        var done = Int(_counter(base, event).fetch_add(1)) + 1
        if done == self.total_dependencies[event]:
            self._schedule(event)

    def _finish(mut self, event: Int):
        """`Event::Finish`: complete a dependency of every parent, then
        `FinalizeFinish` (a complete event completes its pipeline)."""
        for k in range(len(self.parents[event])):
            var parent = self.parents[event][k]
            self._complete_dependency(parent)
        if self.stage[event] == STAGE_COMPLETE:
            _ = self.completed_pipelines.fetch_add(1)

    def _finish_task(mut self, event: Int):
        """`Event::FinishTask`: the event's last task finishes it."""
        var base = Int(self.finished_tasks.unsafe_ptr())
        var done = Int(_counter(base, event).fetch_add(1)) + 1
        if done == self.total_tasks[event]:
            self._finish(event)

    def _dequeue(mut self) -> Int:
        """The next published task, or -1."""
        var base = Int(self.queue.unsafe_ptr())
        while True:
            var head = self.head.load()
            if head >= self.reserved.load():
                return -1
            var record = _counter(base, Int(head)).load()
            if record == 0:
                # Reserved, not yet written: its event is scheduling.
                return -1
            if self.head.compare_exchange(head, head + 1):
                return Int(record - 1)

    def _run_task(mut self, record: Int):
        var event = record >> _TASK_BITS
        var task = record & ((1 << _TASK_BITS) - 1)
        ref pipeline = self._pipelines()[self.pipeline[event]]
        var stage = self.stage[event]
        try:
            if stage == STAGE_INITIALIZE:
                pipeline.initialize()
            elif stage == STAGE_EXECUTE:
                pipeline.execute(task)
            elif stage == STAGE_PREPARE_FINISH:
                pipeline.prepare_finalize()
            else:
                pipeline.finalize()
        except e:
            # `Executor::PushError`: the first error is the query's; the
            # failed event never finishes, so every worker stops.
            var none = Int64(0)
            if self.failed.compare_exchange(none, Int64(1)):
                self.error = String(e)
            return
        self._finish_task(event)

    def work_on_tasks(mut self):
        """`Executor::WorkOnTasks` until the schedule is done. With no
        task queued, the tasks left are running on other threads: run
        the jobs they post to the crew meanwhile, else yield."""
        while not self.finished():
            var record = self._dequeue()
            if record >= 0:
                self._run_task(record)
                continue
            var crew = Int(_current_crew().load())
            if crew == 0 or not Pointer[_Crew, MutAnyOrigin](
                unsafe_from_address=crew
            )[]._help(-1):
                _ = external_call["sched_yield", Int32]()


struct _EventWorker[P: SchedulablePipeline](Job):
    """One thread working on an executor's tasks."""

    var executor: Int

    def __init__(out self, executor: Int):
        self.executor = executor

    def run(mut self) raises:
        Pointer[_Executor[Self.P], MutAnyOrigin](
            unsafe_from_address=self.executor
        )[].work_on_tasks()


def run_pipelines[
    P: SchedulablePipeline
](
    mut pipelines: List[P],
    metas: List[MetaPipeline],
    parallel: Bool = True,
) raises:
    """Schedule `pipelines` (pipeline p is MetaPipeline `metas[p]`) as
    events and run their tasks on the crew's threads and the caller
    until every pipeline completes; re-raise the first task error.
    `parallel=False` runs every task on the caller, in the order the
    events schedule them (an execution report records one at a time)."""
    if len(pipelines) != len(metas):
        raise Error("one MetaPipeline per pipeline")
    if len(pipelines) == 0:
        return
    var stages = build_pipeline_schedule(metas)
    var executor = _Executor[P](pipelines, stages)
    executor.schedule_events()
    # As many threads as tasks can run at once: every pipeline's execute
    # tasks, within the thread count.
    var width = 0
    for p in range(len(pipelines)):
        width += max(1, pipelines[p].execute_tasks())
    var threads = min(width, configured_workers()) if parallel else 1
    if threads <= 1:
        executor.work_on_tasks()
    else:
        var address = Int(Pointer(to=executor))
        var workers = List[_EventWorker[P]](capacity=threads)
        for _ in range(threads):
            workers.append(_EventWorker[P](address))
        # Posted for idle crew threads only: a thread waiting inside one
        # of these tasks must not take a worker that waits for that task.
        _run_jobs(workers, Int.MAX, helpable=False)
    if executor.failed.load() != 0:
        raise Error(executor.error)
    if executor.completed_pipelines.load() < Int64(len(pipelines)):
        raise Error("pipeline schedule stopped before completing")
