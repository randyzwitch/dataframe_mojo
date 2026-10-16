"""Nested scheduling shares one thread budget and survives changing/error rounds."""
# contract-group: execution.lifecycle
from std.atomic import Atomic
from std.ffi import external_call
from std.memory import ArcPointer
from std.sys import num_physical_cores
from std.testing import assert_equal, assert_true, assert_raises
from dataframe.parallel import (
    Job,
    Pool,
    Crew,
    _ProducedJobs,
    run_jobs,
    _helpers,
    _pool_spin_limit,
)
from tests.support.lifecycle_jobs import WorkerSetting, JobCounts, CountedJob
from tests.support.lifecycle_jobs import assert_job_counts, jobs_for


struct Gauge(Movable):
    var active: Atomic[Int64]
    var peak: Atomic[Int64]

    def __init__(out self):
        self.active = Atomic[Int64](0)
        self.peak = Atomic[Int64](0)


struct BudgetJob(Job):
    var gauge: ArcPointer[Gauge]
    var id: Int
    var result: Int

    def __init__(out self, gauge: ArcPointer[Gauge], id: Int):
        self.gauge = gauge.copy()
        self.id = id
        self.result = -1

    def run(mut self) raises:
        var active = self.gauge[].active.fetch_add(1) + 1
        var peak = self.gauge[].peak.load()
        while active > peak:
            if self.gauge[].peak.compare_exchange(peak, active):
                break
        _ = external_call["usleep", Int32](UInt32(1000))
        self.result = self.id * self.id
        _ = self.gauge[].active.fetch_sub(1)


struct NestedBudgetJob(Job):
    var gauge: ArcPointer[Gauge]
    var id: Int
    var result: Int

    def __init__(out self, gauge: ArcPointer[Gauge], id: Int):
        self.gauge = gauge.copy()
        self.id = id
        self.result = 0

    def run(mut self) raises:
        var inner = List[BudgetJob]()
        for index in range(7):
            inner.append(BudgetJob(self.gauge, self.id * 7 + index))
        run_jobs(inner)
        for index in range(7):
            assert_equal(inner[index].result, (self.id * 7 + index) ** 2)
            self.result += inner[index].result


def test_nested_jobs_and_crew_pool_rounds_share_the_configured_budget_and_preserve_result_positions() raises:
    for workers in [1, 2, 4]:
        var setting = WorkerSetting(workers)
        for use_crew in [False, True]:
            var crew = Crew.start() if use_crew else Crew()
            if use_crew:
                var second = Crew.start()
                assert_equal(second.address, 0)
                second.release()
            var gauge = ArcPointer(Gauge())
            for round in range(8):
                var jobs = List[NestedBudgetJob]()
                for id in range(round + 1):
                    jobs.append(NestedBudgetJob(gauge, id))
                run_jobs(jobs)
                for id in range(len(jobs)):
                    var expected = 0
                    for index in range(7):
                        expected += (id * 7 + index) ** 2
                    assert_equal(jobs[id].result, expected)
                assert_equal(gauge[].active.load(), Int64(0))
                assert_true(gauge[].peak.load() <= Int64(workers))
                assert_equal(_helpers().load(), Int64(0))
            var pool = Pool(workers)
            for round in range(41):
                var n = round * 7 % 37
                var failure = 1 if round % 9 == 4 and n > 2 else -1
                var counts = ArcPointer(JobCounts(n))
                var jobs = jobs_for(counts, n, failure)
                if failure >= 0:
                    with assert_raises(
                        contains="deliberate lifecycle worker failure 1"
                    ):
                        pool.run(jobs, claim=round % 2 == 0)
                    assert_job_counts(counts, n, True)
                else:
                    pool.run(jobs, claim=round % 2 == 0)
                    for id in range(n):
                        assert_equal(jobs[id].result, id * id)
                    _ = jobs^
                    assert_job_counts(counts, n, True)
            pool.release()
            pool.release()
            crew.release()
            crew.release()
            assert_equal(_helpers().load(), Int64(0))
        _ = setting^


def _implicit_round(workers: Int, count: Int) raises:
    var pool = Pool(workers)
    var counts = ArcPointer(JobCounts(count))
    var jobs = jobs_for(counts, count)
    pool.run(jobs)
    for id in range(count):
        assert_equal(jobs[id].result, id * id)
    # Pool destructor must join its threads before this function returns.


def _retry_after_capacity(mut pool: Pool, counts: ArcPointer[JobCounts]) raises:
    var jobs = jobs_for(counts, 7)
    pool.run(jobs)
    for id in range(7):
        assert_equal(jobs[id].result, id * id)


def _capacity_error(
    mut pool: Pool,
    counts: ArcPointer[JobCounts],
    rejected: ArcPointer[JobCounts],
) raises:
    var producer = _ProducedJobs[CountedJob](1)
    pool.run_produced(producer)
    producer.submit(CountedJob(counts, 0))
    producer.submit(CountedJob(rejected, 0))


def test_capacity_failure_drains_produced_work_and_implicit_pools_leave_no_worker_at_process_exit() raises:
    for workers in [1, 2, 4]:
        var setting = WorkerSetting(workers)
        var pool = Pool(workers)
        var counts = ArcPointer(JobCounts(1))
        var rejected = ArcPointer(JobCounts(1))
        with assert_raises(contains="capacity exceeded"):
            _capacity_error(pool, counts, rejected)
        assert_job_counts(counts, 1, True)
        assert_equal(rejected[].started[0].load(), Int64(0))
        assert_equal(rejected[].exited[0].load(), Int64(0))
        assert_equal(rejected[].destroyed[0].load(), Int64(1))
        var retry_counts = ArcPointer(JobCounts(7))
        _retry_after_capacity(pool, retry_counts)
        assert_job_counts(retry_counts, 7, True)
        pool.release()
        for round in range(30):
            _implicit_round(workers, round % 17 + 1)
        assert_equal(_helpers().load(), Int64(0))
        _ = setting^


def test_oversubscribed_parked_pool_preserves_rounds_first_error_and_retry() raises:
    var workers = num_physical_cores() + 1
    assert_equal(_pool_spin_limit(workers), 0)
    var pool = Pool(workers)
    for round in range(13):
        var n = round % 7 + 1
        var counts = ArcPointer(JobCounts(n))
        var jobs = jobs_for(counts, n, 1 if n > 2 else -1)
        if n > 2:
            with assert_raises(
                contains="deliberate lifecycle worker failure 1"
            ):
                pool.run(jobs, claim=round % 2 == 0)
        else:
            pool.run(jobs, claim=round % 2 == 0)
            for id in range(n):
                assert_equal(jobs[id].result, id * id)
        _ = jobs^
        assert_job_counts(counts, n, True)
    var retry = ArcPointer(JobCounts(7))
    _retry_after_capacity(pool, retry)
    assert_job_counts(retry, 7, True)
    pool.release()
