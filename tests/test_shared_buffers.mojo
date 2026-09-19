"""Zero-copy column windows and copy-on-write safety."""
from std.testing import TestSuite, assert_equal, assert_true, assert_false
from dataframe import Column, DataFrame, Series, col, concat, lit


def ints(n: Int) raises -> Column[Int64]:
    var values = List[Int64]()
    var valid = List[Bool]()
    for i in range(n):
        values.append(Int64(i))
        valid.append(i % 3 != 1)
    return Column[Int64](values^, valid^)


def frame() raises -> DataFrame:
    return DataFrame(
        [
            Series("a", ints(20)),
            Series("s", Column[String](List[String](length=20, fill="x"))),
        ]
    )


def test_slices_share_buffers_and_read_through_offsets() raises:
    var column = ints(20)
    var window = column.slice(5, 9)
    assert_true(window._shares_buffers_with(column))
    assert_equal(len(window), 9)
    for i in range(9):
        assert_equal(window._valid(i), column._valid(5 + i))
        assert_equal(window._get(i), column._get(5 + i))
    var nested = window.slice(2, 3)
    assert_true(nested._shares_buffers_with(column))
    assert_true(nested.is_null(0))  # row 7: 7 % 3 == 1
    assert_equal(nested.value(1), Int64(8))
    assert_equal(nested.null_count(), window.slice(2, 3).null_count())
    assert_equal(nested._to_list(), [Int64(7), 8, 9])


def test_frame_operations_share_instead_of_copying() raises:
    var df = frame()
    ref original = df._columns[0]._data[Column[Int64]]
    assert_true(
        df.select(["a"])
        ._columns[0]
        ._data[Column[Int64]]
        ._shares_buffers_with(original)
    )
    assert_true(
        df.rename({"a": "b"})
        ._columns[0]
        ._data[Column[Int64]]
        ._shares_buffers_with(original)
    )
    assert_true(
        df.drop("s")
        ._columns[0]
        ._data[Column[Int64]]
        ._shares_buffers_with(original)
    )
    assert_true(
        df.head(3)
        ._columns[0]
        ._data[Column[Int64]]
        ._shares_buffers_with(original)
    )
    assert_true(
        df.slice(4, 5)
        ._columns[0]
        ._data[Column[Int64]]
        ._shares_buffers_with(original)
    )
    assert_true(
        df.column("a")._data[Column[Int64]]._shares_buffers_with(original)
    )
    # Computed columns are new buffers.
    var doubled = df.with_columns((col("a") * lit(Int64(2))).alias("b"))
    assert_false(
        doubled.column("b")._data[Column[Int64]]._shares_buffers_with(original)
    )
    assert_true(
        doubled.column("a")._data[Column[Int64]]._shares_buffers_with(original)
    )


def test_appending_never_mutates_shared_buffers() raises:
    var column = ints(10)
    var before = column._to_list()
    var copy = column.copy()
    copy._append_column(ints(3))
    assert_equal(len(copy), 13)
    assert_equal(len(column), 10)
    assert_equal(column._to_list(), before)
    assert_false(copy._shares_buffers_with(column))
    # Appending to a window must not overwrite the parent's later rows.
    var window = column.slice(0, 4)
    window._append_column(ints(2))
    assert_equal(column._get(4), Int64(4))
    assert_equal(window._to_list(), [Int64(0), 1, 2, 3, 0, 1])


def test_unaligned_windows_append_correctly() raises:
    var source = ints(40)
    for start in range(0, 9):
        for extra in range(0, 11):
            var left = source.slice(start, 13)
            var right = source.slice(start + 5, extra)
            var joined = left.copy()
            joined._append_column(right)
            assert_equal(len(joined), 13 + extra)
            for i in range(13 + extra):
                var expected = start + i if i < 13 else start + 5 + i - 13
                assert_equal(joined._valid(i), source._valid(expected))
                assert_equal(joined._get(i), source._get(expected))
    var stacked = concat([frame().slice(3, 7), frame().slice(11, 5)])
    assert_equal(stacked.height(), 12)
    assert_true(stacked.column("a").get(0) == frame().column("a").get(3))
    assert_true(stacked.column("a").get(7) == frame().column("a").get(11))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
