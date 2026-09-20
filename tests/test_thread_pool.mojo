"""The worker pool: reuse across batches, nesting, oversubscription, errors."""
from std.testing import TestSuite, assert_equal, assert_raises, assert_true
from std.ffi import external_call
from std.memory import Pointer
from std.time import monotonic

from dataframe.parallel import Job, run_jobs


struct Square(Job):
    var value: Int
    var result: Int

    def __init__(out self, value: Int):
        self.value = value
        self.result = -1

    def run(mut self) raises:
        self.result = self.value * self.value


struct Failing(Job):
    var index: Int
    var fail: Bool

    def __init__(out self, index: Int, fail: Bool):
        self.index = index
        self.fail = fail

    def run(mut self) raises:
        if self.fail:
            raise Error("job " + String(self.index) + " failed")


struct Nested(Job):
    """A job that submits its own batch: must run inline, not deadlock."""

    var total: Int

    def __init__(out self):
        self.total = 0

    def run(mut self) raises:
        var inner = List[Square]()
        for i in range(8):
            inner.append(Square(i))
        run_jobs(inner)
        for i in range(len(inner)):
            self.total += inner[i].result


def test_results_come_back_in_order() raises:
    var jobs = List[Square]()
    for i in range(16):
        jobs.append(Square(i))
    run_jobs(jobs)
    assert_equal(len(jobs), 16)
    for i in range(16):
        assert_equal(jobs[i].value, i)
        assert_equal(jobs[i].result, i * i)


def test_many_more_jobs_than_workers() raises:
    var jobs = List[Square]()
    for i in range(500):
        jobs.append(Square(i))
    run_jobs(jobs)
    var total = 0
    for i in range(len(jobs)):
        total += jobs[i].result
    # sum of squares 0..499
    assert_equal(total, 499 * 500 * 999 // 6)


def test_repeated_batches_reuse_the_pool() raises:
    # 200 batches exercise the sleep/wake path; each must be complete
    # and none may see another batch's tasks.
    for batch in range(200):
        var jobs = List[Square]()
        for i in range(8):
            jobs.append(Square(batch * 8 + i))
        run_jobs(jobs)
        for i in range(8):
            assert_equal(jobs[i].result, (batch * 8 + i) * (batch * 8 + i))


def test_nested_submission_runs_inline() raises:
    var outer = List[Nested]()
    for _ in range(4):
        outer.append(Nested())
    run_jobs(outer)
    for i in range(len(outer)):
        assert_equal(outer[i].total, 0 + 1 + 4 + 9 + 16 + 25 + 36 + 49)


def test_first_error_in_job_order_is_raised() raises:
    var jobs = List[Failing]()
    for i in range(12):
        jobs.append(Failing(i, i == 5 or i == 9))
    with assert_raises(contains="job 5 failed"):
        run_jobs(jobs)


def test_single_job_runs_on_the_caller() raises:
    var jobs = List[Square]()
    jobs.append(Square(7))
    run_jobs(jobs)
    assert_equal(jobs[0].result, 49)


def _noop(address: Int) abi("C") -> Int:
    return 0


def test_pool_dispatch_beats_thread_creation() raises:
    """The reason the pool exists: after warm-up, a batch costs a wake-up,
    not thread creation. Measured relative to creating the same number of
    threads in this process, so machine load does not decide the result."""
    var n = 8
    var rounds = 12
    var warm = List[Square]()
    for i in range(n):
        warm.append(Square(i))
    run_jobs(warm)

    var start = monotonic()
    for _ in range(rounds):
        var jobs = List[Square]()
        for i in range(n):
            jobs.append(Square(i))
        run_jobs(jobs)
    var pool_ns = monotonic() - start

    var entry: def(Int) thin abi("C") -> Int = _noop
    var entry_address = Pointer(to=entry).unsafe_bitcast[Int]()[]
    start = monotonic()
    for _ in range(rounds):
        var threads = List[UInt64](length=n, fill=0)
        for t in range(n):
            var rc = external_call["pthread_create", Int32](
                Int(threads.unsafe_ptr()) + 8 * t, 0, entry_address, 0
            )
            if rc != 0:
                raise Error("pthread_create failed")
        for t in range(n):
            _ = external_call["pthread_join", Int32](threads[t], 0)
    var create_ns = monotonic() - start

    assert_true(
        pool_ns < create_ns,
        "pool batch "
        + String(pool_ns // rounds // 1000)
        + " us vs thread creation "
        + String(create_ns // rounds // 1000)
        + " us",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
