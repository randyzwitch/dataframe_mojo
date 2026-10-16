"""A pipeline's morsels cover its source's rows once, in order, never
across a chunk boundary. A source of one chunk about a batch long is cut
into a piece a worker of at least a batch's share per worker (and one
vector); a smaller one stays whole."""
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


def test_one_chunk_sources_split_by_their_batch_share() raises:
    # A batch of 65,536 rows on 8 workers: a 65,536-row source is cut in
    # eight. A 20,000-row one holds two shares of 8,192 rows, and two
    # pieces are then cut to a multiple of the workers, as sources of
    # several chunks are. Under one share (8,000 rows) it stays whole.
    for n in [65_536, 20_000, 8_000, 5, 0]:
        var ranges = _morsel_ranges(rows(n), 65_536, 8)
        check_cover(ranges, n)
        var pieces = len(ranges) // 2
        if n == 65_536:
            assert_equal(pieces, 8)
        elif n == 20_000:
            assert_equal(pieces, 8)
        elif n > 0:
            assert_equal(pieces, 1)
    # One worker: one piece.
    assert_equal(len(_morsel_ranges(rows(65_536), 65_536, 1)) // 2, 1)
