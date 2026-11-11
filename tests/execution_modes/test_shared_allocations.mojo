"""Shared allocations outlive moved owners and worker error paths."""
# contract-group: execution.lifecycle
from std.memory import ArcPointer
from std.testing import assert_equal
from dataframe.parallel import Crew, Pool, _helpers
from tests.support.lifecycle_jobs import (
    WorkerSetting,
    JobCounts,
    assert_job_counts,
    jobs_for,
)


def _moved_pool(workers: Int) -> Pool:
    var pool = Pool(workers)
    return pool^


def test_moved_pool_keeps_worker_state_and_arguments_until_join() raises:
    Crew.shutdown()
    var setting = WorkerSetting(4)
    var pool = _moved_pool(4)
    for _ in range(4):
        var counts = ArcPointer(JobCounts(16))
        var jobs = jobs_for(counts, 16)
        pool.run(jobs, claim=False)
        assert_job_counts(counts, 16, False)
        _ = jobs^
        assert_job_counts(counts, 16, True)
    pool.release()
    pool.release()
    assert_equal(pool.address, 0)
    assert_equal(_helpers().load(), Int64(0))
    _ = setting^
