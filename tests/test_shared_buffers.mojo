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
    assert_equal(nested.to_list(), [Int64(7), 8, 9])


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
    var before = column.to_list()
    var copy = column.copy()
    copy._append_column(ints(3))
    assert_equal(len(copy), 13)
    assert_equal(len(column), 10)
    assert_equal(column.to_list(), before)
    assert_false(copy._shares_buffers_with(column))
    # Appending to a window must not overwrite the parent's later rows.
    var window = column.slice(0, 4)
    window._append_column(ints(2))
    assert_equal(column._get(4), Int64(4))
    assert_equal(window.to_list(), [Int64(0), 1, 2, 3, 0, 1])


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


# Tests from test_chunked_series.mojo.
# Chunked CSV result storage shares arrays and preserves values.
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


def test_slices_find_only_overlapping_chunks() raises:
    var parts = List[Series]()
    for i in range(17):
        parts.append(
            Series("x", Column[Int64]([Int64(2 * i), Int64(2 * i + 1)]))
        )
    var series = Series._from_chunks(parts)
    var compact = series.rechunk()
    for offset in range(len(series) + 1):
        for length in range(len(series) - offset + 1):
            var actual = series.slice(offset, length)
            var expected = compact.slice(offset, length)
            assert_true(actual.equals(expected))
            if length > 0 and offset // 2 == (offset + length - 1) // 2:
                assert_equal(actual.n_chunks(), 1)


# Tests from test_buffer_access.mojo.
# Public Arrow buffer access: a consumer reads values and validity directly.
from std.testing import TestSuite, assert_equal, assert_false, assert_true
from dataframe import BoolColumn, Column, DataFrame, Series, StringColumn


def _bit_at(bits: Pointer[UInt8, MutAnyOrigin], i: Int) -> Bool:
    """Read an Arrow validity bit the way a consumer would."""
    return (bits.unsafe_offset(i >> 3)[] >> UInt8(i & 7)) & 1 == 1


def test_values_pointer_reads_the_window() raises:
    var c = Column[Float64]([1.5, 2.5, 3.5, 4.5])
    var values = c.unsafe_values()
    assert_equal(values[], Float64(1.5))
    assert_equal(values.unsafe_offset(3)[], Float64(4.5))
    # A slice shifts the values pointer to its own row 0.
    var window = c.slice(2, 2)
    assert_equal(window.unsafe_values()[], Float64(3.5))
    assert_equal(len(window), 2)


def test_validity_is_indexed_from_the_window_offset() raises:
    var c = Column[Int64]([1, 2, 3, 4, 5], [True, False, True, True, False])
    assert_equal(c.null_count(), 2)
    var bits = c.unsafe_validity()
    assert_equal(c.validity_offset(), 0)
    assert_true(_bit_at(bits, 0))
    assert_false(_bit_at(bits, 1))
    # The bitmap is NOT shifted to the window; the offset says where row 0 is.
    var window = c.slice(1, 3)
    assert_equal(window.validity_offset(), 1)
    var wbits = window.unsafe_validity()
    var base = window.validity_offset()
    assert_false(_bit_at(wbits, base + 0))  # original row 1
    assert_true(_bit_at(wbits, base + 1))  # original row 2
    assert_equal(window.null_count(), 1)


def test_is_valid_matches_the_bitmap() raises:
    var c = Column[Int64]([1, 2, 3], [True, False, True])
    var bits = c.unsafe_validity()
    for i in range(len(c)):
        assert_equal(c.is_valid(i), _bit_at(bits, c.validity_offset() + i))
    assert_true(c.is_valid(0))
    assert_false(c.is_valid(1))


def test_buffers_are_shared_not_copied() raises:
    var df = DataFrame([Series("a", Column[Int64]([0, 1, 2, 3, 4, 5]))])
    var whole = df.column("a").numeric[DType.int64]()
    var window = df.slice(2, 3).column("a").numeric[DType.int64]()
    # Same underlying allocation: the window points into the original buffer.
    var base = Int(whole.unsafe_values().unsafe_offset(2))
    assert_equal(Int(window.unsafe_values()), base)


def test_to_list_keeps_null_slots_as_stored() raises:
    var c = Column[Int64]([10, 20, 30], [True, False, True])
    var values = c.to_list()
    assert_equal(len(values), 3)
    assert_equal(values[0], Int64(10))
    assert_equal(values[2], Int64(30))
    # The null slot carries its stored payload, as Arrow leaves it undefined.
    assert_equal(values[1], Int64(20))
    assert_false(c.is_valid(1))


def test_bool_values_are_a_bitmap() raises:
    var b = BoolColumn([True, False, True, True], [True, True, False, True])
    var values = b.unsafe_values()
    assert_true(_bit_at(values, b.validity_offset() + 0))
    assert_false(_bit_at(values, b.validity_offset() + 1))
    assert_true(_bit_at(values, b.validity_offset() + 3))
    assert_false(b.is_valid(2))
    assert_equal(b.true_count(), 3)


def test_string_bytes_and_offsets() raises:
    var s = StringColumn(Column[String](["ab", "", "xyz"]))
    var offsets = s.unsafe_offsets()
    var bytes = s.unsafe_bytes()
    var base = s.validity_offset()
    assert_equal(offsets.unsafe_offset(base)[], Int64(0))
    assert_equal(offsets.unsafe_offset(base + 1)[], Int64(2))
    assert_equal(offsets.unsafe_offset(base + 3)[], Int64(5))
    # Row 0 is "ab".
    assert_equal(bytes[], UInt8(ord("a")))
    assert_equal(bytes.unsafe_offset(1)[], UInt8(ord("b")))
    assert_equal(s.to_list(), ["ab", "", "xyz"])


def test_empty_column_has_a_length_of_zero() raises:
    var c = Column[Int64]([])
    assert_equal(len(c), 0)
    assert_equal(c.null_count(), 0)
    assert_equal(len(c.to_list()), 0)


def test_all_valid_take_keeps_absent_validity() raises:
    var source = Column[Int64]([10, 20, 30, 40])
    assert_equal(len(source._bits[]), 0)
    var selected = source.slice(1, 3).take([2, 0, 2])
    assert_equal(selected.to_list(), [Int64(40), 20, 40])
    assert_equal(selected.null_count(), 0)
    assert_equal(len(selected._bits[]), 0)
    var nullable = Column[Int64]([10, 20, 30], [True, False, True])
    var selected_nullable = nullable.take([2, 1, 0])
    assert_equal(selected_nullable.null_count(), 1)
    assert_false(selected_nullable.is_valid(1))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
