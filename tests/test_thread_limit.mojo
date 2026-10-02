"""`run_jobs` never runs more jobs at once than `DATAFRAME_THREADS` (#390):
more jobs than threads are claimed by that many threads, every job runs
once, jobs come back in order, and a failing job still raises.
"""
from std.atomic import Atomic
from std.ffi import external_call
from std.memory import Pointer
from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from dataframe.parallel import Job, run_jobs


def set_threads(n: Int):
    var name = String("DATAFRAME_THREADS")
    var value = String(n)
    _ = external_call["setenv", Int32](
        Int(name.unsafe_ptr()), Int(value.unsafe_ptr()), Int32(1)
    )
    _ = name^
    _ = value^


struct Gauge(Movable):
    var running: Atomic[Int64]
    var peak: Atomic[Int64]

    def __init__(out self):
        self.running = Atomic[Int64](0)
        self.peak = Atomic[Int64](0)


struct CountedJob(Job):
    var gauge: Int
    var index: Int
    var fail: Bool
    var ran: Int

    def __init__(out self, gauge: Int, index: Int, fail: Bool = False):
        self.gauge = gauge
        self.index = index
        self.fail = fail
        self.ran = 0

    def run(mut self) raises:
        ref gauge = Pointer[Gauge, MutAnyOrigin](
            unsafe_from_address=self.gauge
        )[]
        var now = gauge.running.fetch_add(1) + 1
        var peak = gauge.peak.load()
        while now > peak:
            if gauge.peak.compare_exchange(peak, now):
                break
        # Long enough for every started thread to overlap.
        _ = external_call["usleep", Int32](UInt32(2000))
        _ = gauge.running.fetch_sub(1)
        self.ran += 1
        if self.fail:
            raise Error("job " + String(self.index) + " failed")


struct NestingJob(Job):
    """Starts a round of counted jobs from inside a job, as a grouped
    reduce does inside a hash bucket."""

    var gauge: Int
    var inner: Int

    def __init__(out self, gauge: Int, inner: Int):
        self.gauge = gauge
        self.inner = inner

    def run(mut self) raises:
        var jobs = List[CountedJob]()
        for i in range(self.inner):
            jobs.append(CountedJob(self.gauge, i))
        run_jobs(jobs)
        for i in range(self.inner):
            assert_equal(jobs[i].ran, 1)


def run_counted(threads: Int, count: Int) raises -> Int:
    set_threads(threads)
    var gauge = Gauge()
    var jobs = List[CountedJob]()
    for i in range(count):
        jobs.append(CountedJob(Int(Pointer(to=gauge)), i))
    run_jobs(jobs)
    assert_equal(len(jobs), count)
    for i in range(count):
        assert_equal(jobs[i].index, i)
        assert_equal(jobs[i].ran, 1)
    var peak = Int(gauge.peak.load())
    _ = gauge^
    return peak


def test_more_jobs_than_threads_are_capped() raises:
    var peak = run_counted(4, 24)
    assert_true(peak <= 4, "peak " + String(peak))
    # The cap still uses every allowed thread.
    assert_true(peak >= 2, "peak " + String(peak))


def test_up_to_the_limit_each_job_gets_a_thread() raises:
    var peak = run_counted(8, 3)
    assert_true(peak <= 3, "peak " + String(peak))
    assert_equal(run_counted(1, 5), 1)


def test_nested_rounds_share_the_limit() raises:
    set_threads(4)
    var gauge = Gauge()
    var jobs = List[NestingJob]()
    for _ in range(6):
        jobs.append(NestingJob(Int(Pointer(to=gauge)), 4))
    run_jobs(jobs)
    var peak = Int(gauge.peak.load())
    _ = gauge^
    assert_true(peak <= 4, "peak " + String(peak))


def test_a_failing_claimed_job_raises() raises:
    set_threads(3)
    var gauge = Gauge()
    var jobs = List[CountedJob]()
    for i in range(10):
        jobs.append(CountedJob(Int(Pointer(to=gauge)), i, fail=i == 6))
    with assert_raises(contains="job 6 failed"):
        run_jobs(jobs)
    _ = gauge^


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
