"""Arrow C Data Interface: export/import round trips without Python.

pyarrow interoperability is checked separately in tests/oracle/arrow.mojo.
"""
from std.memory import Pointer
from std.testing import TestSuite, assert_equal, assert_true, assert_false
from dataframe import (
    ArrowArray,
    ArrowSchema,
    Column,
    DataFrame,
    DataType,
    Series,
    StringColumn,
    export_arrow,
    export_arrow_series,
    import_arrow,
    import_arrow_series,
)
from dataframe.string_view import StringViewBuilder
from dataframe.arrow import (
    _at,
    _buffer,
    _export_frame,
    _export_series,
    _read_c_string,
)


def sample() raises -> DataFrame:
    var valid = List[Bool]()
    for i in range(20):
        valid.append(i % 4 != 2)
    var ints = List[Int64]()
    var floats = List[Float64]()
    var bools = List[Bool]()
    var texts = List[String]()
    for i in range(20):
        ints.append(Int64(i * 1000 - 7))
        floats.append(Float64(i) / 4.0)
        bools.append(i % 3 == 0)
        texts.append("" if i % 5 == 0 else String("é") * (i % 4) + String(i))
    return DataFrame(
        [
            Series("i", Column[Int64](ints.copy(), valid)),
            Series("f", Column[Float64](floats^, valid)),
            Series("b", Column[Bool](bools^, valid)),
            Series("s", StringColumn(texts, valid)),
            Series("d", Column[Int64](ints.copy(), valid)).with_dtype(
                DataType.DATE
            ),
            Series("ts", Column[Int64](ints.copy(), valid)).with_dtype(
                DataType.datetime("us")
            ),
            Series("du", Column[Int64](ints.copy(), valid)).with_dtype(
                DataType.duration("ns")
            ),
            Series("t", Column[Int64](ints^, valid)).with_dtype(DataType.TIME),
        ]
    )


def round_trip(frame: DataFrame) raises -> DataFrame:
    var array = ArrowArray()
    var schema = ArrowSchema()
    var releases = 0
    _export_frame(
        frame,
        array,
        schema,
        Int(Pointer(to=releases)),
    )
    var result = import_arrow(array, schema)
    assert_equal(releases, 1 + frame.width())  # the parent and each child
    assert_equal(array.release, 0)
    assert_equal(schema.release, 0)
    return result^


def assert_frames_equal(a: DataFrame, b: DataFrame) raises:
    assert_equal(a.height(), b.height())
    assert_equal(a.width(), b.width())
    for k in range(a.width()):
        assert_equal(a._columns[k].name(), b._columns[k].name())
        assert_true(a._columns[k].dtype() == b._columns[k].dtype())
        assert_true(a._columns[k].equals(b._columns[k]))


def test_every_dtype_round_trips() raises:
    var frame = sample()
    assert_frames_equal(round_trip(frame), frame)


def test_sliced_and_empty_frames() raises:
    var frame = sample()
    for start in [0, 1, 3, 7, 8, 13]:
        for length in [0, 1, 5, 7]:
            var window = frame.slice(start, length)
            assert_frames_equal(round_trip(window), window)
    assert_frames_equal(round_trip(frame.slice(20, 0)), frame.slice(20, 0))


def test_schema_formats_and_offsets() raises:
    var frame = sample().slice(3, 10)
    var array = ArrowArray()
    var schema = ArrowSchema()
    export_arrow(frame, array, schema)
    assert_equal(_read_c_string(schema.format), "+s")
    assert_equal(Int(schema.n_children), 8)
    var formats = List[String]()
    var offsets = List[Int64]()
    for k in range(8):
        ref child = _at[ArrowSchema](_at[Int](schema.children + 8 * k)[])[]
        formats.append(_read_c_string(child.format))
        ref child_array = _at[ArrowArray](_at[Int](array.children + 8 * k)[])[]
        offsets.append(child_array.offset)
        assert_equal(child_array.length, 10)
    assert_equal(formats, ["l", "g", "b", "U", "tdD", "tsu:", "tDn", "ttn"])
    # Shared buffers keep the window offset; only Date converts (to Int32),
    # so it starts at 0. Bool is zero-copy now that values are bit-packed.
    assert_equal(offsets, [Int64(3), 3, 3, 3, 0, 3, 3, 3])
    _ = import_arrow(array, schema)


def export_temporary(
    mut array: ArrowArray, mut schema: ArrowSchema, releases: Int
) raises -> Int:
    """Export a column that is destroyed on return; give its bytes address."""
    var column = StringColumn(["alpha", "beta", "gamma"])
    var address = Int(column._bytes[].unsafe_ptr())
    _export_series(Series("s", column^), array, schema, releases)
    return address


def test_export_shares_buffers_and_outlives_source() raises:
    var array = ArrowArray()
    var schema = ArrowSchema()
    var releases = 0
    var data_address = export_temporary(
        array, schema, Int(Pointer(to=releases))
    )
    # The source is gone; the export still owns the same bytes (zero-copy).
    assert_equal(_buffer(array, 2), data_address)
    assert_equal(releases, 0)
    var series = import_arrow_series(array, schema)
    assert_equal(releases, 1)
    assert_equal(series.string().value(2), "gamma")


def test_native_string_views_export_via_retained_large_utf8_adapter() raises:
    var builder = StringViewBuilder()
    builder.append("short")
    builder.append("native string beyond inline")
    builder.append_null()
    builder.append("日本")
    var source = Series("s", StringColumn(builder^.finish()))
    assert_true(source.string()._is_view_storage())
    var array = ArrowArray()
    var schema = ArrowSchema()
    export_arrow_series(source, array, schema)
    # The current complete C adapter is U. It retains its materialization in
    # ArrowArray.private_data, so no pointer escapes a temporary conversion.
    assert_equal(_read_c_string(schema.format), "U")
    assert_equal(array.n_buffers, 3)
    var back = import_arrow_series(array, schema)
    assert_true(back.equals(source))
    assert_false(back.string()._is_view_storage())


def test_series_round_trip_and_release_once() raises:
    var source = sample()
    for k in range(source.width()):
        var array = ArrowArray()
        var schema = ArrowSchema()
        export_arrow_series(
            source._columns[k].slice(5, 9),
            array,
            schema,
        )
        var back = import_arrow_series(array, schema)
        assert_true(back.equals(source._columns[k].slice(5, 9)))
        assert_equal(array.release, 0)
        # Released structs are rejected rather than read.
        var failed = False
        try:
            _ = import_arrow_series(array, schema)
        except:
            failed = True
        assert_true(failed)


def test_unsupported_formats_raise_and_release() raises:
    var array = ArrowArray()
    var schema = ArrowSchema()
    var releases = 0
    _export_series(
        Series("x", Column[Int64]([1, 2])),
        array,
        schema,
        Int(Pointer(to=releases)),
    )
    # Pretend the producer declared a type we do not support.
    var fake = List[UInt8]([UInt8(ord("z")), 0])
    var real_format = schema.format
    schema.format = Int(fake.unsafe_ptr())
    var message = String()
    try:
        _ = import_arrow_series(array, schema)
    except e:
        message = String(e)
    assert_true("Unsupported Arrow format 'z'" in message)
    assert_equal(releases, 1)
    _ = real_format
    _ = fake^


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
