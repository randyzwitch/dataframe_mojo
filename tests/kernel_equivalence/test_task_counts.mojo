"""An eager kernel runs one task per thread, each at least one vector of
rows: tiny inputs stay on the calling thread, large ones use every thread,
and the answer does not depend on the split."""
# contract-group: reduce.global
from std.testing import assert_equal, assert_raises, assert_true

from dataframe import Column, DataFrame, Series, col
from dataframe.parallel import _helpers, task_count, worker_count
from dataframe.vector.types import STANDARD_VECTOR_SIZE
from tests.support.lifecycle_jobs import WorkerSetting


def test_task_counts_follow_vectors_and_threads() raises:
    var setting = WorkerSetting(8)
    comptime V = STANDARD_VECTOR_SIZE
    for rows in [0, 1, V - 1, V, 2 * V - 1]:
        assert_equal(worker_count(rows), 1)
    assert_equal(worker_count(2 * V), 2)
    assert_equal(worker_count(5 * V + 3), 5)
    assert_equal(worker_count(8 * V), 8)
    assert_equal(worker_count(1_000_000), 8)
    assert_equal(task_count(1_000_000, 3), 3)
    assert_equal(task_count(1_000_000, 0), 1)
    _ = setting^
    var serial = WorkerSetting(1)
    assert_equal(worker_count(1_000_000), 1)
    _ = serial^


def _frame(n: Int) raises -> DataFrame:
    var values = List[Int64](capacity=n)
    for i in range(n):
        values.append(Int64((i * 7919) % 1000 - 500))
    return DataFrame([Series("i", Column[Int64](values^))])


def test_reductions_match_on_every_task_count() raises:
    comptime V = STANDARD_VECTOR_SIZE
    for n in [1, V - 1, 2 * V, 3 * V + 1, 100_000]:
        var total = Int64(0)
        var low = Int64.MAX
        var high = Int64.MIN
        for i in range(n):
            var v = Int64((i * 7919) % 1000 - 500)
            total += v
            low = min(low, v)
            high = max(high, v)
        var frame = _frame(n)
        for threads in [1, 4, 8]:
            var setting = WorkerSetting(threads)
            var out = frame.select_exprs(
                [
                    col("i").sum().alias("s"),
                    col("i").min().alias("lo"),
                    col("i").max().alias("hi"),
                    col("i").count().alias("c"),
                ]
            )
            assert_equal(out.item(0, "s").int64(), total)
            assert_equal(out.item(0, "lo").int64(), low)
            assert_equal(out.item(0, "hi").int64(), high)
            assert_equal(out.item(0, "c").int64(), Int64(n))
            assert_equal(_helpers().load(), Int64(0))
            _ = setting^


def test_overflow_raises_on_every_task_count() raises:
    # The values that overflow sit in different vectors, so each task's
    # partial sum fits and only the combined total overflows.
    comptime V = STANDARD_VECTOR_SIZE
    var values = List[Int64](length=4 * V, fill=0)
    values[1] = Int64.MAX
    values[3 * V + 1] = 1
    var frame = DataFrame([Series("i", Column[Int64](values^))])
    for threads in [1, 4]:
        var setting = WorkerSetting(threads)
        with assert_raises():
            _ = frame.select(col("i").sum())
        assert_true(_helpers().load() == Int64(0))
        _ = setting^
