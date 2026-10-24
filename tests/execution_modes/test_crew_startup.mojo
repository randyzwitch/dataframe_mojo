"""A crew creates its threads on the first `run_jobs` that has work for
them, not when its operation starts: an operation with nothing to run in
parallel creates none. The first thread creates the others, so that first
call is served by helpers as they come up. Release parks a crew whose
threads were created and the next operation takes it back, with no thread
created; a crew whose threads were never created is freed, and one sized for
another thread setting is joined and replaced."""
# contract-group: execution.lifecycle
from std.atomic import Atomic
from std.ffi import external_call
from std.memory import ArcPointer, Pointer
from std.time import perf_counter_ns
from std.testing import assert_equal, assert_true
from dataframe.parallel import Crew, Job, _Crew, _helpers, run_jobs
from tests.support.lifecycle_jobs import WorkerSetting


struct Meeting(Movable):
    var arrived: Atomic[Int64]
    var peak: Atomic[Int64]
    var active: Atomic[Int64]

    def __init__(out self):
        self.arrived = Atomic[Int64](0)
        self.peak = Atomic[Int64](0)
        self.active = Atomic[Int64](0)


struct MeetingJob(Job):
    """Waits, up to two seconds, until `expected` jobs run at once."""

    var meeting: ArcPointer[Meeting]
    var expected: Int

    def __init__(out self, meeting: ArcPointer[Meeting], expected: Int):
        self.meeting = meeting.copy()
        self.expected = expected

    def run(mut self) raises:
        var active = self.meeting[].active.fetch_add(1) + 1
        var peak = self.meeting[].peak.load()
        while active > peak:
            if self.meeting[].peak.compare_exchange(peak, active):
                break
            peak = self.meeting[].peak.load()
        _ = self.meeting[].arrived.fetch_add(1)
        var deadline = perf_counter_ns() + 2_000_000_000
        while self.meeting[].arrived.load() < Int64(self.expected):
            if perf_counter_ns() > deadline:
                break
            _ = external_call["sched_yield", Int32]()
        _ = self.meeting[].active.fetch_sub(1)


def state(crew: Crew) -> ref[MutAnyOrigin] _Crew:
    return Pointer[_Crew, MutAnyOrigin](unsafe_from_address=crew.address)[]


def test_a_crew_creates_no_thread_until_work_arrives() raises:
    var setting = WorkerSetting(4)
    Crew.shutdown()
    var idle = Crew.start()
    assert_true(idle.address != 0)
    assert_equal(state(idle).spawned.load(), Int64(0))
    assert_equal(state(idle).threads, 0)
    # Released without work: nothing to join.
    idle.release()
    assert_equal(_helpers().load(), Int64(0))
    _ = setting^


def test_the_first_call_is_served_by_helpers_and_the_crew_is_parked_on_release() raises:
    for workers in [2, 4, 8]:
        var setting = WorkerSetting(workers)
        var crew = Crew.start()
        var meeting = ArcPointer(Meeting())
        var jobs = List[MeetingJob]()
        for _ in range(workers):
            jobs.append(MeetingJob(meeting, workers))
        run_jobs(jobs)
        # Every job met the others: the helpers the first call created ran
        # them alongside the caller.
        assert_equal(meeting[].arrived.load(), Int64(workers))
        assert_equal(meeting[].peak.load(), Int64(workers))
        assert_equal(state(crew).spawned.load(), Int64(2))
        assert_equal(state(crew).threads, workers - 1)
        # Later calls reuse the threads.
        var again = ArcPointer(Meeting())
        var more = List[MeetingJob]()
        for _ in range(workers):
            more.append(MeetingJob(again, workers))
        run_jobs(more)
        assert_equal(again[].peak.load(), Int64(workers))
        assert_equal(state(crew).threads, workers - 1)
        crew.release()
        crew.release()
        assert_equal(_helpers().load(), Int64(0))
        _ = setting^


def _run_meeting(workers: Int) raises -> Int64:
    var meeting = ArcPointer(Meeting())
    var jobs = List[MeetingJob]()
    for _ in range(workers):
        jobs.append(MeetingJob(meeting, workers))
    run_jobs(jobs)
    return meeting[].peak.load()


def test_a_released_crew_is_taken_back_without_creating_threads() raises:
    var setting = WorkerSetting(4)
    var first = Crew.start()
    assert_equal(_run_meeting(4), Int64(4))
    var address = first.address
    first.release()
    assert_equal(first.address, 0)
    var second = Crew.start()
    # The same state, its threads already up.
    assert_equal(second.address, address)
    assert_equal(state(second).spawned.load(), Int64(2))
    assert_equal(state(second).threads, 3)
    assert_equal(_run_meeting(4), Int64(4))
    assert_equal(state(second).threads, 3)
    second.release()
    assert_equal(_helpers().load(), Int64(0))
    _ = setting^


def test_a_parked_crew_follows_the_thread_setting() raises:
    var first_setting = WorkerSetting(4)
    var first = Crew.start()
    assert_equal(_run_meeting(4), Int64(4))
    first.release()
    _ = first_setting^
    # Another setting joins the parked threads and starts a crew of its size.
    var other_setting = WorkerSetting(2)
    var second = Crew.start()
    assert_equal(state(second).spawned.load(), Int64(0))
    assert_equal(_run_meeting(2), Int64(2))
    assert_equal(state(second).threads, 1)
    second.release()
    assert_equal(_helpers().load(), Int64(0))
    _ = other_setting^
