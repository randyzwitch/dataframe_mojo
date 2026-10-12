"""Count actual worker entries and job destruction, including exceptional unwind."""
from std.atomic import Atomic
from std.ffi import external_call
from std.memory import ArcPointer
from std.os import getenv
from dataframe.parallel import Job, _helpers
from std.testing import assert_equal


struct WorkerSetting(Movable):
    var previous: String
    var existed: Bool

    def __init__(out self, workers: Int):
        # An environment value cannot contain NUL. A default NUL therefore
        # distinguishes absent from present-empty without redeclaring libc.
        var old = getenv("DATAFRAME_THREADS", "\0")
        self.previous = old.copy() if old != "\0" else String("")
        var name = String("DATAFRAME_THREADS\0")
        self.existed = old != "\0"
        var value = String(workers) + "\0"
        _ = external_call["setenv", Int32](
            Int(name.unsafe_ptr()), Int(value.unsafe_ptr()), Int32(1)
        )
        _ = name^
        _ = value^

    def __deinit__(deinit self):
        var name = String("DATAFRAME_THREADS\0")
        if self.existed:
            var value = self.previous + "\0"
            _ = external_call["setenv", Int32](
                Int(name.unsafe_ptr()),
                Int(value.unsafe_ptr()),
                Int32(1),
            )
            _ = value^
        else:
            _ = external_call["unsetenv", Int32](Int(name.unsafe_ptr()))
        _ = name^


struct JobCounts(Movable):
    var started: List[Atomic[Int64]]
    var exited: List[Atomic[Int64]]
    var destroyed: List[Atomic[Int64]]
    var thread: List[Atomic[Int64]]
    var active: Atomic[Int64]
    var destroyed_while_active: Atomic[Int64]

    def __init__(out self, count: Int):
        self.started = []
        self.exited = []
        self.destroyed = []
        self.thread = []
        for _ in range(count):
            self.started.append(Atomic[Int64](0))
            self.exited.append(Atomic[Int64](0))
            self.destroyed.append(Atomic[Int64](0))
            self.thread.append(Atomic[Int64](0))
        self.active = Atomic[Int64](0)
        self.destroyed_while_active = Atomic[Int64](0)


struct CountedJob(Job):
    var counts: ArcPointer[JobCounts]
    var id: Int
    var fail: Bool
    var result: Int

    def __init__(
        out self, counts: ArcPointer[JobCounts], id: Int, fail: Bool = False
    ):
        self.counts = counts.copy()
        self.id = id
        self.fail = fail
        self.result = -1

    def run(mut self) raises:
        _ = self.counts[].started[self.id].fetch_add(1)
        self.counts[].thread[self.id].store(
            external_call["pthread_self", UInt64]().cast[DType.int64]()
        )
        _ = self.counts[].active.fetch_add(1)
        # Yield so a producer can fail while real helper threads own work.
        for _ in range(16):
            _ = external_call["sched_yield", Int32]()
        self.result = self.id * self.id
        _ = self.counts[].active.fetch_sub(1)
        _ = self.counts[].exited[self.id].fetch_add(1)
        if self.fail:
            raise Error(
                "deliberate lifecycle worker failure " + String(self.id)
            )

    def __deinit__(deinit self):
        if self.counts[].active.load() != 0:
            _ = self.counts[].destroyed_while_active.fetch_add(1)
        _ = self.counts[].destroyed[self.id].fetch_add(1)


def assert_job_counts(
    counts: ArcPointer[JobCounts], submitted: Int, destroyed: Bool
) raises:
    assert_equal(counts[].active.load(), Int64(0))
    assert_equal(counts[].destroyed_while_active.load(), Int64(0))
    for id in range(len(counts[].started)):
        assert_equal(
            counts[].started[id].load(), Int64(1 if id < submitted else 0)
        )
        assert_equal(
            counts[].exited[id].load(), Int64(1 if id < submitted else 0)
        )
        assert_equal(
            counts[].destroyed[id].load(),
            Int64(1 if destroyed and id < submitted else 0),
        )
    assert_equal(_helpers().load(), Int64(0))


def jobs_for(
    counts: ArcPointer[JobCounts], count: Int, failure: Int = -1
) -> List[CountedJob]:
    var jobs = List[CountedJob]()
    for id in range(count):
        jobs.append(CountedJob(counts, id, id == failure))
    return jobs^
