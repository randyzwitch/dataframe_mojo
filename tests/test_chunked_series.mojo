"""Chunked CSV result storage shares arrays and preserves values."""
from std.testing import TestSuite, assert_equal, assert_true
from dataframe import Column, Series


def test_chunk_views_and_rechunk() raises:
    var first = Series("x", Column[Int64]([1, 2]))
    var second = Series("x", Column[Int64]([3, 4, 5], [True, False, True]))
    var series = Series._from_chunks([first.copy(), second.copy()])
    assert_equal(series.n_chunks(), 2)
    assert_equal(len(series), 5)
    assert_equal(series.null_count(), 1)
    assert_equal(series.get(2).int64(), 3)
    assert_true(series.chunks()[0].int64()._shares_buffers_with(first.int64()))
    var slice = series.slice(1, 3)
    assert_equal(slice.n_chunks(), 2)
    assert_equal(slice.get(0).int64(), 2)
    assert_equal(slice.get(1).int64(), 3)
    assert_equal(slice.null_count(), 1)
    var compact = series.rechunk()
    assert_equal(compact.n_chunks(), 1)
    assert_true(series.equals(compact))
    assert_true(slice.equals(compact.slice(1, 3)))


def main() raises:
    var suite = TestSuite.discover_tests[__functions_in_module()]()
    suite^.run()
