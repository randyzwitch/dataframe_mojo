"""Pipelines scheduled by events, ported from DuckDB 2.0's executor
(src/parallel: executor.cpp, event.cpp, pipeline_schedule.cpp,
pipeline_initialize_event.cpp, pipeline_event.cpp,
pipeline_prepare_finish_event.cpp, pipeline_finish_event.cpp,
pipeline_complete_event.cpp).

A query's independent pipelines -- the build sides of a stream's joins
and the probe that depends on them, the inputs of an eager join, shared
subplans -- are scheduled by their data dependencies. Each pipeline
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
not depend on, and independent builds run at the same time. A pipeline
says how many tasks its execute event makes when that event schedules
(`Pipeline::Schedule`): a stream's probe makes one per morsel worker. A
thread with no task queued runs the jobs that running tasks post to the
crew (`_Crew._help`): a build side's own plan still runs inside its one
execute task. A finish event whose sink's Finalize has parallel work (filling a join
build's pointer table) adds that work's tasks to the queue, and finishes
after the last of them, so the work spreads over every thread while
nothing reads the sink early. The serial step after those tasks may add
another round of them, as a merge that halves its runs each round does,
so a finish of any number of parallel rounds never waits on a nested
run. The workers themselves are posted so that only idle crew
threads join them (`_UNHELPABLE`).
"""
from std.atomic import Atomic
from std.ffi import external_call
from std.memory import Pointer
from std.time import perf_counter_ns

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


def _queue_capacity(outstanding: Int) -> Int:
    """The ring's slot count: a power of two of at least `outstanding`."""
    var capacity = 1
    while capacity < outstanding:
        capacity <<= 1
    return capacity


trait SchedulablePipeline(Deinitable, Movable):
    """A pipeline's work, as its events' tasks run it."""

    def initialize(mut self) raises:
        """`PipelineInitializeTask`: reset the sink."""
        ...

    def max_execute_tasks(self) -> Int:
        """The most execute tasks `execute_tasks` can make, known before
        any event runs: the queue is sized from it."""
        ...

    def execute_tasks(mut self) raises -> Int:
        """`Pipeline::Schedule`, called when the execute event is
        scheduled (its dependencies complete): set up the source and
        operators, and say how many tasks to make (at least one, at most
        `max_execute_tasks`)."""
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

    def max_finalize_tasks(self) -> Int:
        """The most tasks one call of `finalize_tasks` can make, known
        before any event runs: the queue is sized from it."""
        return 0

    def finalize_tasks(mut self) raises -> Int:
        """After `finalize`, and again after each `finish_finalize`: how
        many tasks the sink's next round of finalize work splits into, all
        of which must run before the finish event's parents may start (a
        join build filling its pointer table); 0 when no work is left."""
        return 0

    def finalize_task(mut self, task: Int) raises:
        """Task `task` of the current round; they run at the same time."""
        pass

    def finish_finalize(mut self) raises:
        """The serial step after the last task of a round."""
        pass


struct MetaPipeline(Copyable, Movable):
    """Pipeline scheduling metadata. parent identifies the consumer of this
    pipeline's sink, and dependencies must complete before execution.
    JOIN_BUILD children with the same parent are siblings. finish_group
    identifies a pipeline whose execution must finish first, such as the
    probe before an unmatched-build scan; -1 means none. Prepare-finish
    waits until all group members finish execution."""

    var type: Int
    var parent: Int
    var dependencies: List[Int]
    var finish_group: Int

    def __init__(
        out self,
        type: Int,
        parent: Int = -1,
        dependencies: List[Int] = List[Int](),
        finish_group: Int = -1,
    ):
        self.type = type
        self.parent = parent
        self.dependencies = dependencies.copy()
        self.finish_group = finish_group


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
    # A finish group: each member executes after the one it names, and the
    # group's base prepares its finish after every member executed.
    for p in range(len(metas)):
        var g = metas[p].finish_group
        if g < 0:
            continue
        if g >= len(metas) or g == p:
            raise Error("pipeline finish group out of range")
        stages[5 * p + STAGE_EXECUTE].dependencies.append(5 * g + STAGE_EXECUTE)
        var base = g
        while metas[base].finish_group >= 0:
            base = metas[base].finish_group
        stages[5 * base + STAGE_PREPARE_FINISH].dependencies.append(
            5 * p + STAGE_EXECUTE
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
    # The task queue: a ring of slots, each a task record and a sequence
    # number (atomics). Slot `p & mask` holds the task at position p once
    # its sequence is p + 1, and is free for position p + capacity once
    # its sequence is p + capacity. `tail` is the next position to fill,
    # `head` the next to take. A pipeline's events run one after another,
    # and so do its finalize rounds, so at most the largest of its
    # events' task counts is queued for it at once: the ring holds the
    # sum of those over the pipelines, however many rounds there are.
    var records: List[Int64]
    var sequences: List[Int64]
    var mask: Int
    var tail: Atomic[Int64]
    var head: Atomic[Int64]
    var total_pipelines: Int
    var completed_pipelines: Atomic[Int64]
    var failed: Atomic[Int64]
    var error: String
    # When the schedule began: orders it against schedules nested in its
    # tasks for `_Crew._help`.
    var stamp: Int

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
        for s in stages:
            self.stage.append(s.type)
            self.pipeline.append(s.pipeline)
            self.parents.append(List[Int]())
            self.total_dependencies.append(len(s.dependencies))
            # The execute event learns its count when it schedules, a
            # finish event its finalize rounds' as they come. Initialize, prepare finish and finish each
            # make one task; complete makes none and finishes when
            # scheduled.
            var tasks = 0 if s.type == STAGE_COMPLETE else 1
            self.total_tasks.append(tasks)
        var outstanding = 0
        for p in range(len(pipelines)):
            outstanding += max(
                1,
                pipelines[p].max_execute_tasks(),
                pipelines[p].max_finalize_tasks(),
            )
        var capacity = _queue_capacity(outstanding)
        # `Event::AddDependency`: the dependency learns its parent.
        for e in range(len(stages)):
            for d in stages[e].dependencies:
                self.parents[d].append(e)
        self.finished_dependencies = List[Int64](length=len(stages), fill=0)
        self.finished_tasks = List[Int64](length=len(stages), fill=0)
        self.records = List[Int64](length=capacity, fill=0)
        self.sequences = List[Int64](capacity=capacity)
        for i in range(capacity):
            self.sequences.append(Int64(i))
        self.mask = capacity - 1
        self.tail = Atomic[Int64](0)
        self.head = Atomic[Int64](0)
        self.total_pipelines = len(pipelines)
        self.completed_pipelines = Atomic[Int64](0)
        self.failed = Atomic[Int64](0)
        self.error = String()
        self.stamp = Int(perf_counter_ns())

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
        if self.stage[event] == STAGE_EXECUTE:
            ref pipeline = self._pipelines()[self.pipeline[event]]
            try:
                tasks = max(1, pipeline.execute_tasks())
                if tasks > max(1, pipeline.max_execute_tasks()):
                    raise Error("more execute tasks than max_execute_tasks")
            except e:
                self._fail(e)
                return
            self.total_tasks[event] = tasks
        if tasks == 0:
            self._finish(event)
            return
        for t in range(tasks):
            if not self._push((event << _TASK_BITS) | t):
                return

    def _push(mut self, record: Int) -> Bool:
        """Queue a task record; fails the schedule when the ring has no
        free slot, which its sizing rules out."""
        var sequences = Int(self.sequences.unsafe_ptr())
        while True:
            var position = self.tail.load()
            var slot = Int(position) & self.mask
            var free = _counter(sequences, slot).load() - position
            if free == 0:
                if self.tail.compare_exchange(position, position + 1):
                    _counter(Int(self.records.unsafe_ptr()), slot).store(
                        Int64(record)
                    )
                    _counter(sequences, slot).store(position + 1)
                    return True
            elif free < 0:
                self._fail(Error("pipeline task queue overflow"))
                return False

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
        """Count a finished task; the event's last task finishes it. A
        finish event whose round of finalize tasks just ended first runs
        their serial step (`finish_finalize`), which may start another
        round instead."""
        var base = Int(self.finished_tasks.unsafe_ptr())
        var done = Int(_counter(base, event).fetch_add(1)) + 1
        if done == self.total_tasks[event]:
            if self.stage[event] == STAGE_FINISH and done > 1:
                try:
                    self._pipelines()[self.pipeline[event]].finish_finalize()
                    if self._insert_finalize_tasks(event):
                        return
                except e:
                    self._fail(e)
                    return
            self._finish(event)

    def _insert_finalize_tasks(mut self, event: Int) raises -> Bool:
        """After a finish event's Finalize, or a round's serial step: the
        tasks of the next round of its finalize work, as more tasks of the
        finish event, round task t queued as index t + 1 (index 0 is
        Finalize itself). The finish event finishes, and its parents'
        dependencies complete, only after every round. Whether a round
        started."""
        ref pipeline = self._pipelines()[self.pipeline[event]]
        var tasks = pipeline.finalize_tasks()
        if tasks == 0:
            return False
        if tasks > pipeline.max_finalize_tasks():
            raise Error("more finalize tasks than max_finalize_tasks")
        # Set before any is queued: no task of this round can finish
        # before the count includes it.
        self.total_tasks[event] += tasks
        for t in range(tasks):
            if not self._push((event << _TASK_BITS) | (t + 1)):
                return True
        return True

    def _dequeue(mut self) -> Int:
        """The next queued task, or -1."""
        var sequences = Int(self.sequences.unsafe_ptr())
        while True:
            var position = self.head.load()
            var slot = Int(position) & self.mask
            var filled = _counter(sequences, slot).load() - (position + 1)
            if filled < 0:
                # Empty, or its task still being written.
                return -1
            if filled == 0 and self.head.compare_exchange(
                position, position + 1
            ):
                var record = _counter(
                    Int(self.records.unsafe_ptr()), slot
                ).load()
                _counter(sequences, slot).store(position + Int64(self.mask + 1))
                return Int(record)

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
            elif task > 0:
                pipeline.finalize_task(task - 1)
            else:
                pipeline.finalize()
                _ = self._insert_finalize_tasks(event)
        except e:
            self._fail(e)
            return
        self._finish_task(event)

    def _fail(mut self, e: Error):
        """`Executor::PushError`: the first error is the query's; the
        failed event never finishes, so every worker stops."""
        var none = Int64(0)
        if self.failed.compare_exchange(none, Int64(1)):
            self.error = String(e)

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
            )[]._help(-1, self.stamp):
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
    # tasks or finalize tasks, within the thread count.
    var width = 0
    for p in range(len(pipelines)):
        width += max(
            1,
            pipelines[p].max_execute_tasks(),
            pipelines[p].max_finalize_tasks(),
        )
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
        _run_jobs(workers, Int.MAX, helpable=False, schedule=executor.stamp)
    if executor.failed.load() != 0:
        raise Error(executor.error)
    if executor.completed_pipelines.load() < Int64(len(pipelines)):
        raise Error("pipeline schedule stopped before completing")
