"""Polars-style absent validity until the first null, across consumers."""
from std.memory import ArcPointer
from std.testing import TestSuite, assert_equal, assert_true
from dataframe import (
    Column,
    Series,
    DataFrame,
    ArrowArray,
    ArrowSchema,
    export_arrow_series,
    import_arrow_series,
)
from dataframe.arrow import _buffer
from dataframe.bool_column import BoolColumn
from dataframe.string_column import StringBuilder
from dataframe.csv import CsvField, CsvOptions
from dataframe.csv_buffers import CsvBuffer
from dataframe.dtype import DataType


def _add(mut buffer: CsvBuffer, text: String) raises:
    var bytes = Span[UInt8, ImmutAnyOrigin](
        unsafe_ptr=text.as_bytes()
        .unsafe_ptr()
        .unsafe_origin_cast[ImmutAnyOrigin](),
        length=text.byte_length(),
    )
    buffer.add(
        bytes, False, CsvOptions(",", '"', "", 0, -1, [], False, False, "utf8")
    )


def test_csv_builders_allocate_validity_only_on_first_null() raises:
    var numeric = CsvBuffer(CsvField.int64("x"), 17)
    var boolean = CsvBuffer(CsvField.bool("b"), 17)
    var strings = CsvBuffer(CsvField.string("s"), 17)
    for _ in range(17):
        _add(numeric, "7")
        _add(boolean, "true")
        _add(strings, "text")
    var n = numeric.finish().int64()
    var b = boolean.finish().bool()
    var s = strings.finish().string()
    assert_equal(len(n._bits[]), 0)
    assert_equal(len(b._bits[]), 0)
    assert_equal(len(s._bits[]), 0)
    assert_equal(n.null_count(), 0)
    assert_equal(b.true_count(), 17)
    assert_equal(s.value(16), "text")
    assert_equal(n.take([16, 0]).value(0), 7)
    assert_true(b.take([16, 0]).value(1))
    assert_equal(s.take([16, 0]).value(0), "text")


def test_first_null_and_chunk_append_at_every_bitmap_alignment() raises:
    for boundary in range(18):
        var left = CsvBuffer(CsvField.int64("x"), boundary + 4)
        for _ in range(boundary):
            _add(left, "7")
        left.add_null()
        _add(left, "9")
        var nullable = left.finish()
        var right = CsvBuffer(CsvField.int64("x"), 11)
        for _ in range(11):
            _add(right, "11")
        var valid = right.finish()
        var combined = Series._from_chunks(
            [valid.copy(), nullable.copy(), valid.copy()]
        ).rechunk()
        assert_equal(len(combined), 24 + boundary)
        assert_equal(combined.null_count(), 1)
        assert_true(combined.int64().is_null(11 + boundary))
        assert_equal(combined.int64().value(12 + boundary), 9)
        assert_equal(combined.int64().value(23 + boundary), 11)
        var all_valid = (
            Series._from_chunks([valid.copy(), valid.copy()]).rechunk().int64()
        )
        assert_equal(len(all_valid._bits[]), 0)
        assert_equal(all_valid.value(21), 11)


def test_string_builder_rollback_with_and_without_validity() raises:
    var builder = StringBuilder(3)
    builder.append("a")
    builder.append("b")
    builder._pop()
    builder.append_null()
    builder._pop()
    builder.append("c")
    var result = builder^.finish()
    assert_equal(len(result), 2)
    assert_equal(result.null_count(), 0)
    assert_equal(result.value(1), "c")


def test_arrow_absent_validity_round_trip() raises:
    var buffer = CsvBuffer(CsvField.int64("x"), 9)
    for _ in range(9):
        _add(buffer, "42")
    var series = buffer.finish().slice(1, 7)
    var array = ArrowArray()
    var schema = ArrowSchema()
    export_arrow_series(series, array, schema)
    assert_equal(_buffer(array, 0), 0)
    var imported = import_arrow_series(array, schema)
    assert_equal(len(imported.int64()._bits[]), 0)
    assert_equal(imported.int64().value(6), 42)
    assert_true(imported.equals(series))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
