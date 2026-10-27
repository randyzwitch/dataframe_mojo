"""Frame exports iterate incrementally and estimate visible column buffers."""
# contract-group: api.inspect
# contract-group: storage.views
from std.collections import Optional
from std.testing import assert_equal, assert_raises, assert_true
from dataframe import (
    AnyValue,
    ArrowArray,
    ArrowSchema,
    Column,
    DataFrame,
    DataType,
    ListColumn,
    RowView,
    Series,
    StringColumn,
    StructColumn,
    export_arrow_series,
    import_arrow_series,
)
from dataframe.string_view import StringViewBuilder
from tests.support.coverage_compare import assert_frame
from tests.support.coverage_generators import offset_layout, chunk_layout
from tests.support.reduction_types_reference import (
    typed_reduction_columns,
    reduction_validity,
)


def _source() raises -> DataFrame:
    var columns = typed_reduction_columns(reduction_validity(1))
    for i in range(len(columns)):
        columns[i] = columns[i].renamed("c" + String(i))
    columns.append(
        Series(
            "list",
            ListColumn(
                [Int64(0), 1, 2, 3, 4, 5, 6],
                Series("item", Column[Int64]([1, 2, 3, 4, 5, 6])),
                [UInt8(0b101101)],
            ),
        )
    )
    columns.append(
        Series(
            "struct",
            StructColumn(
                [
                    Series("a", Column[Int64]([1, 2, 3, 4, 5, 6])),
                    Series("s", Column[String](["a", "b", "c", "d", "e", "f"])),
                ],
                [UInt8(0b101101)],
                length=6,
            ),
        )
    )
    columns.append(Series.full_null("null", DataType.NULL, 6))
    return DataFrame(columns^)


def _retained_row() raises -> RowView:
    var frame = _source()
    for row in frame.iter_rows():
        return row.copy()
    raise Error("Expected a row")


def test_frame_dictionary_exports_preserve_typed_cells_and_column_ownership() raises:
    var source = _source()
    for frame in [source.copy(), offset_layout(source), chunk_layout(source)]:
        var dictionary = frame.to_dict()
        assert_equal(len(dictionary), frame.width())
        for name in frame.columns():
            assert_true(dictionary[name].equals(frame.column(name)))
            if not dictionary[name].dtype().is_nested():
                assert_true(
                    dictionary[name]._shares_buffers_with(frame.column(name))
                )
        var rows = frame.to_dicts()
        assert_equal(len(rows), frame.height())
        for row in range(frame.height()):
            assert_equal(len(rows[row]), frame.width())
            for name in frame.columns():
                assert_true(rows[row][name] == frame.column(name).get(row))
        assert_equal(len(frame.clear().to_dicts()), 0)
    var zero = DataFrame(List[Series](), height=3)
    assert_equal(len(zero.to_dict()), 0)
    var rows = zero.to_dicts()
    assert_equal(len(rows), 3)
    assert_equal(len(rows[2]), 0)


def test_frame_row_and_column_iterators_keep_ownership_and_convert_only_visited_cells() raises:
    var frame = _source()
    var index = 0
    for row in frame.iter_rows():
        assert_equal(len(row), frame.width())
        var values = row.to_list()
        for c in range(frame.width()):
            var expected = frame._columns[c].get(index)
            assert_true(row[c] == expected)
            assert_true(row[frame._columns[c].name()] == expected)
            assert_true(values[c] == expected)
        index += 1
    assert_equal(index, frame.height())
    var retained = _retained_row()
    assert_true(retained[0] == frame._columns[0].get(0))
    with assert_raises():
        _ = retained[-1]
    with assert_raises():
        _ = retained[frame.width()]
    with assert_raises():
        _ = retained["missing"]
    index = 0
    for column in frame.iter_columns():
        assert_true(column.equals(frame._columns[index]))
        index += 1
    assert_equal(index, frame.width())
    index = 0
    for row in frame.clear().iter_rows():
        _ = row
        index += 1
    assert_equal(index, 0)
    # An invalid label is deliberately behind an unvisited row. Iterator
    # creation and early termination must not decode it.
    var encoded = Series("x", Column[String](["a"])).cast(DataType.CATEGORICAL)
    var bad = Series("x", Column[UInt32]([0, 99])).with_dtype(encoded.dtype())
    var late = DataFrame([bad.copy()])
    for row in late.iter_rows():
        assert_equal(row[0].string(), "a")
        break


def test_frame_slice_iterator_is_incremental_zero_copy_and_handles_every_storage_window() raises:
    var source = _source()
    for frame in [source.copy(), offset_layout(source), chunk_layout(source)]:
        for n in [Int(1), 2, 4, 9, Int.MAX]:
            var offset = 0
            for part in frame.iter_slices(n):
                var expected = frame.slice(
                    offset, min(n, frame.height() - offset)
                )
                assert_frame(part, expected)
                for name in frame.columns():
                    if not part.column(name).dtype().is_nested():
                        assert_true(
                            part.column(name)._shares_buffers_with(
                                expected.column(name)
                            )
                        )
                offset += part.height()
            assert_equal(offset, frame.height())
        for n in [Int(0), -1]:
            with assert_raises():
                _ = frame.iter_slices(n)
        var count = 0
        for part in frame.clear().iter_slices(3):
            _ = part
            count += 1
        assert_equal(count, 0)
    var zero = DataFrame(List[Series](), height=5)
    var heights = List[Int]()
    for part in zero.iter_slices(2):
        heights.append(part.height())
        assert_equal(part.width(), 0)
    assert_equal(heights, List[Int]([2, 2, 1]))


def test_frame_size_counts_native_buffers_nested_spans_bit_offsets_and_units() raises:
    var numeric = DataFrame([Series("x", Column[Int64]([1, 2, 3]))])
    assert_equal(numeric.estimated_size(), 24.0)
    var nullable = DataFrame(
        [Series("x", Column[Float32]([1, 2, 3], [True, False, True]))]
    )
    assert_equal(nullable.estimated_size(), 13.0)
    var text = DataFrame([Series("x", StringColumn(["a", "bb", ""]))])
    assert_equal(text.estimated_size(), 35.0)
    assert_equal(text.slice(1, 1).estimated_size(), 18.0)
    var bits = DataFrame(
        [
            Series("b", Column[Bool](List[Bool](length=10, fill=True))),
            Series(
                "f",
                Column[Float64](
                    List[Float64](length=10, fill=1),
                    [
                        False,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                    ],
                ),
            ),
        ]
    )
    assert_equal(bits.slice(7, 2).estimated_size(), 22.0)
    var child = Series("item", Column[Int64]([9, 1, 2, 3, 4, 9]))
    var lists = DataFrame([Series("x", ListColumn([Int64(1), 3, 5], child^))])
    assert_equal(lists.estimated_size(), 56.0)
    assert_equal(lists.slice(1, 1).estimated_size(), 32.0)
    var structs = DataFrame(
        [
            Series(
                "x",
                StructColumn(
                    [
                        Series("a", Column[Int64]([1, 2])),
                        Series("b", Column[Float32]([3, 4])),
                    ],
                    [UInt8(0b01)],
                    length=2,
                ),
            )
        ]
    )
    assert_equal(structs.estimated_size(), 25.0)
    assert_equal(structs.slice(1, 1).estimated_size(), 13.0)
    var kilo = DataFrame(
        [Series("x", Column[Int64](List[Int64](length=128, fill=0)))]
    )
    for unit in [String("kb"), "kilobytes"]:
        assert_equal(kilo.estimated_size(unit), 1.0)
    assert_equal(kilo.estimated_size("mb"), 1.0 / 1024)
    assert_equal(kilo.estimated_size("gb"), 1.0 / (1024.0 * 1024))
    assert_equal(kilo.estimated_size("tb"), 1.0 / (1024.0 * 1024 * 1024))
    assert_equal(kilo.estimated_size("bytes"), 1024.0)
    var empty = DataFrame(List[Series]())
    assert_equal(empty.estimated_size(), 0.0)
    for unit in [String("KB"), "unknown"]:
        with assert_raises():
            _ = empty.estimated_size(unit)


def test_frame_size_covers_views_chunks_categoricals_and_borrowed_arrow() raises:
    var builder = StringViewBuilder()
    builder.append("a")
    builder.append("abcdefghijklmnop")
    builder.append_null()
    var views = DataFrame([Series("x", StringColumn(builder^.finish()))])
    assert_equal(views.estimated_size(), 17.0)
    assert_equal(views.slice(1, 2).estimated_size(), 16.0)
    assert_equal(views.slice(2, 1).estimated_size(), 0.0)
    var values = DataFrame(
        [Series("x", Column[Int32]([1, 2, 3, 4, 5, 6], reduction_validity(1)))]
    )
    assert_equal(values.estimated_size(), 25.0)
    assert_equal(chunk_layout(values).estimated_size(), 26.0)
    var encoded = DataFrame(
        [Series("x", Column[String](["a", "bb"])).cast(DataType.CATEGORICAL)]
    )
    assert_equal(encoded.estimated_size(), 9.0)
    assert_equal(
        DataFrame([Series.full_null("x", DataType.NULL, 9)]).estimated_size(),
        0.0,
    )
    var source = Series("x", Column[Int64]([1, 2, 3], [True, False, True]))
    var array = ArrowArray()
    var schema = ArrowSchema()
    export_arrow_series(source, array, schema)
    var imported = import_arrow_series(array, schema, copy=False)
    var borrowed = DataFrame([imported.copy()])
    assert_equal(borrowed.estimated_size(), 25.0)
    assert_equal(borrowed.slice(1, 1).estimated_size(), 9.0)
