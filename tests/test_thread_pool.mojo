"""A scoped worker pool: same results as run_jobs, and no threads at exit.

The exit behaviour is the point of the design. PR #114 kept a process-wide
pool whose workers were never joined, and the process crashed on the way out
about one run in three under `mojo run`, after every test had passed. So the
last test here runs many pools back to back and the suite's own clean exit is
the check -- a leaked worker would take the process down with it.
"""
from std.ffi import external_call
from std.sys import num_physical_cores
from std.testing import TestSuite, assert_equal, assert_true, assert_raises

from dataframe.parallel import (
    Crew,
    Job,
    Pool,
    configured_workers,
    run_jobs,
    _ProducedJobs,
)


def c_string(text: String) -> List[UInt8]:
    var bytes = List[UInt8]()
    bytes.extend(text.as_bytes())
    bytes.append(0)
    return bytes^


def set_threads(n: Int):
    var name = c_string("DATAFRAME_THREADS")
    var value = c_string(String(n))
    _ = external_call["setenv", Int32](
        Int(name.unsafe_ptr()), Int(value.unsafe_ptr()), Int32(1)
    )
    _ = name^
    _ = value^


struct Square(Job):
    var input: Int
    var output: Int

    def __init__(out self, input: Int):
        self.input = input
        self.output = -1

    def run(mut self) raises:
        self.output = self.input * self.input


struct Failing(Job):
    var index: Int

    def __init__(out self, index: Int):
        self.index = index

    def run(mut self) raises:
        if self.index % 3 == 1:
            raise Error("job " + String(self.index) + " failed")


def test_results_come_back_in_order() raises:
    set_threads(8)
    var pool = Pool(configured_workers())
    var jobs = List[Square]()
    for i in range(50):
        jobs.append(Square(i))
    pool.run(jobs)
    assert_equal(len(jobs), 50)
    for i in range(50):
        assert_equal(jobs[i].output, i * i, "job " + String(i))
    pool.release()


def test_more_jobs_than_workers() raises:
    set_threads(4)
    var pool = Pool(configured_workers())
    var jobs = List[Square]()
    for i in range(500):
        jobs.append(Square(i))
    pool.run(jobs)
    for i in range(500):
        assert_equal(jobs[i].output, i * i)
    pool.release()


def test_many_rounds_on_one_pool() raises:
    """The reason every batch field is atomic and read after the claim: a
    worker still in the claim loop from the previous round must not test a
    stale count against a fresh index. When that was wrong it dropped one
    task per hang, so the rounds here are short and numerous."""
    set_threads(16)
    var pool = Pool(configured_workers())
    for round in range(200):
        var jobs = List[Square]()
        var n = 2 + round % 31
        for i in range(n):
            jobs.append(Square(i))
        pool.run(jobs)
        assert_equal(len(jobs), n, "round " + String(round))
        for i in range(n):
            assert_equal(jobs[i].output, i * i, "round " + String(round))
    pool.release()


def test_first_error_in_job_order_is_raised() raises:
    set_threads(8)
    var pool = Pool(configured_workers())
    var jobs = List[Failing]()
    for i in range(20):
        jobs.append(Failing(i))
    with assert_raises(contains="job 1 failed"):
        pool.run(jobs)
    pool.release()


def test_single_job_and_empty_round() raises:
    set_threads(8)
    var pool = Pool(configured_workers())
    var none = List[Square]()
    pool.run(none)
    assert_equal(len(none), 0)
    var one = List[Square]()
    one.append(Square(7))
    pool.run(one)
    assert_equal(one[0].output, 49)
    pool.release()


def test_one_thread_runs_everything_on_the_caller() raises:
    set_threads(1)
    var pool = Pool(configured_workers())
    assert_equal(pool.workers(), 0, "no threads should start at one worker")
    var jobs = List[Square]()
    for i in range(10):
        jobs.append(Square(i))
    pool.run(jobs)
    for i in range(10):
        assert_equal(jobs[i].output, i * i)
    pool.release()
    set_threads(32)


def test_release_is_idempotent_and_implicit() raises:
    set_threads(8)
    var pool = Pool(configured_workers())
    var jobs = List[Square]()
    jobs.append(Square(3))
    jobs.append(Square(4))
    pool.run(jobs)
    pool.release()
    pool.release()
    assert_equal(jobs[1].output, 16)

    # A pool that is never released must still join its workers when it goes
    # out of scope, or the threads outlive the test and reach process exit.
    for _ in range(20):
        var scoped = Pool(configured_workers())
        var work = List[Square]()
        for i in range(8):
            work.append(Square(i))
        scoped.run(work)
        assert_equal(work[7].output, 49)


def test_many_pools_back_to_back() raises:
    """Create and release many pools, at several sizes. Any worker that
    survived its pool would still be parked in JIT-compiled code when this
    process exits, which is exactly how #114 crashed."""
    set_threads(32)
    for round in range(60):
        var size = 2 + (round % 16)
        var pool = Pool(size)
        var jobs = List[Square]()
        for i in range(size * 3):
            jobs.append(Square(i))
        pool.run(jobs)
        for i in range(size * 3):
            assert_equal(jobs[i].output, i * i, "round " + String(round))
        pool.release()


struct Increment(Job):
    var count: Int

    def __init__(out self):
        self.count = 0

    def run(mut self) raises:
        self.count += 1


def test_claimed_and_produced_rounds_run_exactly_once() raises:
    for workers in [1, 2, 7]:
        var pool = Pool(workers)
        for round in range(30):
            var n = round * 7 % 101
            var produced = _ProducedJobs[Increment](n)
            pool.run_produced(produced)
            for _ in range(n):
                produced.submit(Increment())
            var result = produced.finish()
            assert_equal(len(result), n)
            for i in range(len(result)):
                assert_equal(result[i].count, 1)
            var jobs = List[Increment]()
            for _ in range(n):
                jobs.append(Increment())
            pool.run(jobs, claim=True)
            for i in range(len(jobs)):
                assert_equal(jobs[i].count, 1)
        pool.release()


def test_producer_errors_drain_before_reuse() raises:
    for workers in [1, 4]:
        var pool = Pool(workers)
        var produced = _ProducedJobs[Failing](10)
        pool.run_produced(produced)
        for i in range(10):
            produced.submit(Failing(i))
        with assert_raises(contains="job 1 failed"):
            _ = produced.finish()
        var jobs = List[Square]()
        jobs.append(Square(9))
        pool.run(jobs)
        assert_equal(jobs[0].output, 81)
        pool.release()


def test_producer_capacity_failure_drains_on_unwind() raises:
    var pool = Pool(4)
    var refused = False
    try:
        var produced = _ProducedJobs[Square](1)
        pool.run_produced(produced)
        produced.submit(Square(2))
        produced.submit(Square(3))
    except e:
        refused = True
        assert_true("capacity exceeded" in String(e))
    assert_true(refused)
    var jobs = List[Square]()
    jobs.append(Square(7))
    pool.run(jobs)
    assert_equal(jobs[0].output, 49)
    pool.release()


def test_oversubscribed_parked_pool_reuses_rounds_and_propagates_errors() raises:
    var pool = Pool(num_physical_cores() + 1)
    for round in range(8):
        var jobs = List[Square]()
        for i in range(19):
            jobs.append(Square(i + round))
        pool.run(jobs)
        for i in range(len(jobs)):
            assert_equal(jobs[i].output, (i + round) * (i + round))
    var failed = False
    try:
        var jobs = List[Failing]()
        for i in range(9):
            jobs.append(Failing(i))
        pool.run(jobs)
    except:
        failed = True
    assert_true(failed)
    var final = List[Square]()
    final.append(Square(7))
    pool.run(final)
    assert_equal(final[0].output, 49)
    pool.release()


struct Nested(Job):
    """A job that runs jobs of its own, as a stream batch's work does."""

    var base: Int
    var total: Int

    def __init__(out self, base: Int):
        self.base = base
        self.total = 0

    def run(mut self) raises:
        var inner = List[Square]()
        for i in range(5):
            inner.append(Square(self.base + i))
        run_jobs(inner)
        for i in range(5):
            self.total += inner[i].output


def nested_total(base: Int) -> Int:
    var total = 0
    for i in range(5):
        total += (base + i) * (base + i)
    return total


def test_crew_runs_jobs_nested_calls_and_pool_rounds() raises:
    set_threads(8)
    var crew = Crew.start()
    for round in range(40):
        var n = round % 13 + 1
        var jobs = List[Square]()
        for i in range(n):
            jobs.append(Square(round + i))
        run_jobs(jobs)
        for i in range(n):
            assert_equal(jobs[i].output, (round + i) * (round + i))
        var nested = List[Nested]()
        for i in range(n):
            nested.append(Nested(i))
        run_jobs(nested)
        for i in range(n):
            assert_equal(nested[i].total, nested_total(i))
        # A pool inside the operation runs its rounds on the crew, capped
        # at its own size, and starts threads only for a produced round.
        var pool = Pool(3)
        var squares = List[Square]()
        for i in range(n):
            squares.append(Square(i))
        pool.run(squares, claim=round % 2 == 0)
        for i in range(n):
            assert_equal(squares[i].output, i * i)
        if round % 10 == 0:
            var produced = _ProducedJobs[Increment](n)
            pool.run_produced(produced)
            for _ in range(n):
                produced.submit(Increment())
            var result = produced.finish()
            for i in range(n):
                assert_equal(result[i].count, 1)
        pool.release()
    crew.release()
    crew.release()


def test_crew_raises_the_first_error_and_stays_usable() raises:
    set_threads(4)
    var crew = Crew.start()
    for _ in range(5):
        var jobs = List[Failing]()
        for i in range(9):
            jobs.append(Failing(i))
        with assert_raises(contains="job 1 failed"):
            run_jobs(jobs)
    var jobs = List[Square]()
    for i in range(9):
        jobs.append(Square(i))
    run_jobs(jobs)
    assert_equal(jobs[8].output, 64)
    crew.release()


def test_second_crew_start_does_nothing() raises:
    set_threads(4)
    var first = Crew.start()
    var second = Crew.start()
    assert_equal(second.address, 0)
    second.release()
    var jobs = List[Square]()
    for i in range(20):
        jobs.append(Square(i))
    run_jobs(jobs)
    assert_equal(jobs[19].output, 361)
    first.release()
    # With no crew running, run_jobs starts threads as before.
    var after = List[Square]()
    for i in range(6):
        after.append(Square(i))
    run_jobs(after)
    assert_equal(after[5].output, 25)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
