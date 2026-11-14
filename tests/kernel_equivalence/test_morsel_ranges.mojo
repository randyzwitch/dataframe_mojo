"""Morsels cover source rows once in order without crossing chunk boundaries.
Round the target morsel count up to a multiple of the workers so the final
scheduling wave stays balanced."""
# contract-group: table.join
from std.testing import assert_equal, assert_true

from dataframe import Column, DataFrame, Series, col
from dataframe.parallel import source_tasks
from dataframe.pipeline import _morsel_ranges
from tests.support.lifecycle_jobs import WorkerSetting


def rows(n: Int) raises -> DataFrame:
    var values = List[Int64](length=n, fill=0)
    return DataFrame([Series("x", Column[Int64](values^))])


def check_cover(ranges: List[Int], n: Int) raises:
    var at = 0
    for k in range(len(ranges) // 2):
        assert_equal(ranges[2 * k], at)
        assert_true(ranges[2 * k + 1] > 0)
        at += ranges[2 * k + 1]
    assert_equal(at, n)


def test_counts_round_to_a_multiple_of_the_workers() raises:
    # Start from a target of 100,000 rows per morsel and round the count up to
    # a worker multiple: 1M rows on eight workers gives 16 morsels.
    for n in [1_000_000, 100_000, 20_000, 5, 0]:
        var ranges = _morsel_ranges(rows(n), 8)
        check_cover(ranges, n)
        var pieces = len(ranges) // 2
        if n == 1_000_000:
            assert_equal(pieces, 16)
        elif n == 100_000 or n == 20_000:
            assert_equal(pieces, 8)
        elif n > 0:
            assert_equal(pieces, 1)
    # One worker: only the ideal count.
    assert_equal(len(_morsel_ranges(rows(250_000), 1)) // 2, 2)


def test_tasks_are_one_a_thread_up_to_the_morsels() raises:
    # Every thread runs a task while the source has morsels for it; a
    # source of fewer morsels than threads runs a task per morsel.
    assert_equal(source_tasks(8, 8), 8)
    assert_equal(source_tasks(16, 8), 8)
    assert_equal(source_tasks(5, 8), 5)
    assert_equal(source_tasks(1, 8), 1)
    assert_equal(source_tasks(0, 8), 1)
    assert_equal(source_tasks(3, 1), 1)
    # A file's batches are not known ahead: a task a thread.
    assert_equal(source_tasks(-1, 8), 8)
    # 288,464 rows on eight threads: eight morsels, so eight tasks.
    var morsels = len(_morsel_ranges(rows(288_464), 8)) // 2
    assert_equal(morsels, 8)
    assert_equal(source_tasks(morsels, 8), 8)


def sequence(n: Int) raises -> DataFrame:
    var values = List[Int64](capacity=n)
    for i in range(n):
        values.append(Int64(i))
    return DataFrame([Series("x", Column[Int64](values^))])


def test_tasks_taking_many_morsels_keep_source_order() raises:
    # Two threads over four morsels: each task takes morsels until the
    # source is empty, and the collected rows stay in source order.
    var setting = WorkerSetting(2)
    var n = 400_000
    var frame = sequence(n)
    assert_equal(len(_morsel_ranges(frame, 2)) // 2, 4)
    var kept = (
        frame.lazy()
        .filter(col("x") % 3 == 0)
        .with_columns([(col("x") * 2).alias("y")])
        .collect()
    )
    assert_equal(kept.height(), (n + 2) // 3)
    var x = kept.column("x").rechunk()
    var y = kept.column("y").rechunk()
    ref xs = x._data[Column[Int64]]
    ref ys = y._data[Column[Int64]]
    for i in range(kept.height()):
        if xs._get(i) != Int64(3 * i) or ys._get(i) != Int64(6 * i):
            assert_equal(xs.value(i), Int64(3 * i))
            assert_equal(ys.value(i), Int64(6 * i))
    # Every morsel is reduced once.
    var total = frame.lazy().select(col("x").sum().alias("s")).collect()
    assert_equal(
        total.column("s").rechunk()._data[Column[Int64]].value(0),
        Int64(n) * Int64(n - 1) // 2,
    )
    _ = setting^


def test_a_one_morsel_source_on_many_threads() raises:
    var setting = WorkerSetting(8)
    var frame = sequence(5)
    assert_equal(len(_morsel_ranges(frame, 8)) // 2, 1)
    var kept = frame.lazy().filter(col("x") > 1).collect()
    assert_equal(kept.height(), 3)
    var x = kept.column("x").rechunk()
    for i in range(3):
        assert_equal(x._data[Column[Int64]].value(i), Int64(i + 2))
    _ = setting^
