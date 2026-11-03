"""Atomic publication, shared cache lifetimes, and actual concurrent readers."""
# contract-group: storage.buffer_facts
from std.atomic import Atomic
from std.ffi import external_call
from std.memory import ArcPointer
from std.time import perf_counter_ns
from std.testing import assert_equal, assert_true, assert_false
from dataframe import Column, Series
from dataframe.storage.facts import _BufferFacts
from dataframe.parallel import Job, Pool, run_jobs
from dataframe.join_hash import int64_progression
from dataframe.kernels.segment_statistics import (
    ensure_statistics,
    series_statistics,
)
from dataframe.pipeline import _zoned_bounds
from tests.support.lifecycle_jobs import WorkerSetting


struct Meeting(Movable):
    var arrived: Atomic[Int64]

    def __init__(out self):
        self.arrived = Atomic[Int64](0)


def meet(meeting: ArcPointer[Meeting], workers: Int) raises:
    _ = meeting[].arrived.fetch_add(1)
    var deadline = perf_counter_ns() + 2_000_000_000
    while meeting[].arrived.load() < Int64(workers):
        if perf_counter_ns() > deadline:
            raise Error("buffer-facts workers did not overlap")
        _ = external_call["sched_yield", Int32]()


struct PublishJob(Job):
    var facts: ArcPointer[_BufferFacts]
    var meeting: ArcPointer[Meeting]
    var workers: Int
    var id: Int
    var observed: Int64

    def __init__(
        out self,
        facts: ArcPointer[_BufferFacts],
        meeting: ArcPointer[Meeting],
        workers: Int,
        id: Int,
    ):
        self.facts = facts.copy()
        self.meeting = meeting.copy()
        self.workers = workers
        self.id = id
        self.observed = -1

    def run(mut self) raises:
        meet(self.meeting, self.workers)
        for _ in range(64):
            self.facts[].publish_progression(True)
        self.observed = Int64(self.facts[].progression_state())


def test_concurrent_facts_publication_exposes_one_answer() raises:
    var setting = WorkerSetting(4)
    var pool = Pool(4)
    for _ in range(32):
        var facts = ArcPointer(_BufferFacts())
        var meeting = ArcPointer(Meeting())
        var jobs = List[PublishJob]()
        for id in range(4):
            jobs.append(PublishJob(facts, meeting, 4, id))
        pool.run(jobs)
        assert_equal(meeting[].arrived.load(), Int64(4))
        assert_equal(facts[].progression_state(), 1)
        for i in range(len(jobs)):
            assert_equal(jobs[i].observed, Int64(1))
    _ = pool^
    _ = setting^


struct CacheJob(Job):
    var column: Column[Int64]
    var meeting: ArcPointer[Meeting]
    var workers: Int
    var progression: Bool
    var least: Int64
    var most: Int64

    def __init__(
        out self,
        column: Column[Int64],
        meeting: ArcPointer[Meeting],
        workers: Int,
    ):
        self.column = column.copy()
        self.meeting = meeting.copy()
        self.workers = workers
        self.progression = False
        self.least = 0
        self.most = 0

    def run(mut self) raises:
        meet(self.meeting, self.workers)
        self.progression = int64_progression(Series("k", self.column.copy()))[0]
        # Simultaneous initializers publish one record; the others drop
        # theirs, and every reader sees a complete one.
        ensure_statistics(Series("k", self.column.copy()))
        var bounds = _zoned_bounds(Series("k", self.column.copy()))
        assert_true(bounds[0])
        self.least = bounds[1]
        self.most = bounds[2]


def test_progression_and_statistics_consumers_share_cold_and_warm_facts() raises:
    var setting = WorkerSetting(4)
    var pool = Pool(4)
    for variant in range(3):
        var values = List[Int64]()
        for row in range(65537):
            values.append(
                Int64(
                    row if variant
                    == 0 else (65536 - row if variant == 1 else row // 2)
                )
            )
        var column = Column[Int64](values^)
        for _ in range(2):
            var meeting = ArcPointer(Meeting())
            var jobs = List[CacheJob]()
            for _ in range(4):
                jobs.append(CacheJob(column, meeting, 4))
            pool.run(jobs)
            for i in range(len(jobs)):
                ref job = jobs[i]
                assert_equal(job.progression, variant == 0)
                assert_equal(job.least, Int64(0))
                assert_equal(job.most, Int64(32768 if variant == 2 else 65536))
            assert_equal(
                column._facts[].progression_state(), 1 if variant == 0 else 2
            )
            assert_true(column._data[].statistics())
    _ = pool^
    _ = setting^


def test_fact_invalidation_keeps_copied_windows_and_old_published_statistics_alive() raises:
    var column = Column[Int64]([0, 1, 2, 3, 4])
    assert_true(int64_progression(Series("k", column.copy()))[0])
    ensure_statistics(Series("k", column.copy()))
    var view = column.slice(1, 3)
    var original = column.copy()
    column._append_column(Column[Int64]([-1]))
    assert_equal(column._facts[].progression_state(), 0)
    assert_false(column._data[].statistics())
    assert_false(int64_progression(Series("k", column.copy()))[0])
    assert_equal(original._facts[].progression_state(), 1)
    assert_true(view._data[].statistics())
    assert_equal(_zoned_bounds(Series("v", view.copy()))[1], Int64(0))
    assert_equal(_zoned_bounds(Series("v", view.copy()))[2], Int64(4))
    assert_equal(view.to_list(), List[Int64]([Int64(1), Int64(2), Int64(3)]))
    # The sole owner's values change in place: their statistics go.
    var owned = Column[Int64]([5, 6])
    ensure_statistics(Series("o", owned.copy()))
    assert_true(owned._data[].statistics())
    owned._append_column(Column[Int64]([-7]))
    assert_false(owned._data[].statistics())
    var bounds = _zoned_bounds(Series("o", owned.copy()))
    assert_equal(bounds[1], Int64(-7))
    assert_equal(bounds[2], Int64(6))
    column._reserve_rows(100, 0)
    assert_false(column._data[].statistics())
    assert_equal(
        original.to_list(),
        List[Int64]([Int64(0), Int64(1), Int64(2), Int64(3), Int64(4)]),
    )


def test_declared_window_bounds_and_negative_facts_are_independent() raises:
    var facts = _BufferFacts(2, 5)
    assert_equal(facts.first(), 2)
    assert_equal(facts.last(100), 5)
    assert_equal(facts.progression_state(), 0)
    facts.publish_progression(False)
    assert_equal(facts.progression_state(), 2)
    var unknown = _BufferFacts()
    assert_equal(unknown.last(17), 17)
