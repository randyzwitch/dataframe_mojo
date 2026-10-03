"""Rechunking a chunked series: the first merge is kept and shared by every
copy, later rechunks return the same values, and string-view chunks merge
with their buffer indexes kept or remapped, nulls included.
"""
from std.testing import TestSuite, assert_equal, assert_true

from dataframe import Column, Series, StringColumn
from dataframe.string_view import StringViewBuilder


def views(first: Int, count: Int, null_every: Int) raises -> Series:
    var builder = StringViewBuilder()
    for i in range(first, first + count):
        if null_every > 0 and i % null_every == 0:
            builder.append_null()
        elif i % 3 == 0:
            builder.append(StringSlice("short" + String(i)))
        else:
            builder.append(
                StringSlice("a value longer than twelve bytes " + String(i))
            )
    return Series("s", StringColumn(builder^.finish()))


def expected(first: Int, count: Int, null_every: Int, i: Int) -> String:
    var k = first + i
    if null_every > 0 and k % null_every == 0:
        return "<null>"
    if k % 3 == 0:
        return "short" + String(k)
    return "a value longer than twelve bytes " + String(k)


def text(series: Series, i: Int) raises -> String:
    var cell = series.get(i)
    return String("<null>") if cell.is_null() else cell.string()


def test_view_chunks_merge_in_order() raises:
    for null_every in [0, 7]:
        # Separate builders: each chunk has its own buffers, so the merge
        # remaps every chunk after the first.
        var parts: List[Series] = [
            views(0, 500, null_every),
            views(500, 300, null_every),
            views(800, 9, null_every),
        ]
        var chunked = Series._from_chunks(parts)
        assert_true(chunked.is_chunked())
        var merged = chunked.rechunk()
        assert_equal(len(merged), 809)
        for i in range(809):
            assert_equal(text(merged, i), expected(0, 809, null_every, i))
        # Slices of one chunk share its buffers: indexes stay as they are.
        var whole = views(0, 600, null_every)
        var pieces: List[Series] = [
            whole.slice(0, 200),
            whole.slice(200, 250),
            whole.slice(450, 150),
        ]
        var again = Series._from_chunks(pieces).rechunk()
        for i in range(600):
            assert_equal(text(again, i), expected(0, 600, null_every, i))


def test_second_rechunk_reuses_the_first() raises:
    var parts: List[Series] = [
        Series("x", Column[Int64]([1, 2, 3])),
        Series("x", Column[Int64]([4, 5])),
    ]
    var chunked = Series._from_chunks(parts)
    var copy = chunked.copy()
    var first = chunked.rechunk()
    var second = copy.rechunk()
    assert_equal(len(second), 5)
    for i in range(5):
        assert_equal(second.get(i).int64(), Int64(i + 1))
        assert_equal(first.get(i).int64(), second.get(i).int64())
    assert_equal(second.name(), "x")
    assert_true(second.dtype() == chunked.dtype())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
