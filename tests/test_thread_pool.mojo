"""A scoped worker pool: same results as run_jobs, and no threads at exit.

The exit behaviour is the point of the design. PR #114 kept a process-wide
pool whose workers were never joined, and the process crashed on the way out
about one run in three under `mojo run`, after every test had passed. So the
last test here runs many pools back to back and the suite's own clean exit is
the check -- a leaked worker would take the process down with it.
"""
from std.ffi import external_call
from std.testing import TestSuite, assert_equal, assert_true, assert_raises

from dataframe.parallel import Job, Pool, configured_workers


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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
