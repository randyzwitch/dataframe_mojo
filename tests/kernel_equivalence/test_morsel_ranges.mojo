"""Morsels cover source rows once in order without crossing chunk boundaries.
Round the target morsel count up to a multiple of the workers so the final
scheduling wave stays balanced."""
# contract-group: table.join
from std.testing import assert_equal, assert_true

from dataframe import Column, DataFrame, Series
from dataframe.pipeline import _morsel_ranges


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
