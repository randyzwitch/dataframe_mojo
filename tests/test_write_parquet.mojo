"""Parquet writing: logical types, chunks, codecs, empty files and errors."""
from std.testing import TestSuite, assert_equal, assert_true, assert_raises
from std.memory import Pointer
from dataframe import (
    ArrowArray,
    ArrowSchema,
    Column,
    DataFrame,
    DataType,
    Series,
    concat,
    read_parquet,
    write_parquet,
    parquet_backend_version,
    parquet_row_group_statistics,
)
from dataframe.arrow import _export_frame, _release_imported
from dataframe.parquet import (
    _load_library,
    _Library,
    _call_writer,
    _raise_backend_error,
)

comptime PATH = "/tmp/dataframe_write_parquet_test.parquet"


def equal(left: DataFrame, right: DataFrame) raises:
    assert_equal(left.height(), right.height())
    assert_equal(left.columns(), right.columns())
    for c in range(left.width()):
        assert_equal(left._columns[c].dtype(), right._columns[c].dtype())
        assert_true(left._columns[c].equals(right._columns[c]))


def test_codecs_chunks_projection_and_empty() raises:
    var source = read_parquet("tests/fixtures/types.parquet")
    var chunked = concat([source.slice(1, 2), source.slice(0, 4)])
    for codec in ["zstd", "snappy", "uncompressed"]:
        write_parquet(chunked, PATH, compression=codec, row_group_size=2)
        equal(chunked, read_parquet(PATH))
        assert_equal(parquet_row_group_statistics(PATH).height(), 3)
        equal(
            chunked.select(["s", "i"]), read_parquet(PATH, columns=["s", "i"])
        )
        write_parquet(source.clear(), PATH, compression=codec)
        equal(source.clear(), read_parquet(PATH))
    # Writing has not consumed or mutated the source.
    equal(source, read_parquet("tests/fixtures/types.parquet"))


def test_nested_and_parameterized_types() raises:
    for fixture in ["list_int.parquet", "struct.parquet"]:
        var source = read_parquet("tests/fixtures/" + fixture)
        write_parquet(source, PATH, row_group_size=1)
        equal(source, read_parquet(PATH))
    var values = Column[Int64]([0, 123, -456], [True, False, True])
    var columns = List[Series]()
    for unit in ["ns", "us", "ms"]:
        columns.append(
            Series("ts_" + unit, values.copy()).with_dtype(
                DataType.datetime(unit)
            )
        )
        columns.append(
            Series("du_" + unit, values.copy()).with_dtype(
                DataType.duration(unit)
            )
        )
    columns.append(
        Series("time", Column[Int64]([0, 123, 456])).with_dtype(DataType.TIME)
    )
    for dtype in [
        DataType.INT8,
        DataType.INT16,
        DataType.INT32,
        DataType.UINT8,
        DataType.UINT16,
        DataType.UINT32,
        DataType.UINT64,
        DataType.FLOAT32,
        DataType.FLOAT64,
    ]:
        columns.append(
            Series(
                dtype.name(), Column[Int64]([0, 1, 2], [True, False, True])
            ).cast(dtype)
        )
    var source = DataFrame(columns^)
    write_parquet(source, PATH, row_group_size=2)
    equal(source, read_parquet(PATH))


def test_invalid_options_and_io_error() raises:
    var source = DataFrame([Series("a", Column[Int64]([1, 2]))])
    with assert_raises(contains="unsupported compression"):
        write_parquet(source, PATH, compression="unknown")
    with assert_raises(contains="positive"):
        write_parquet(source, PATH, row_group_size=0)
    with assert_raises():
        write_parquet(source, "/dev/null/not-a-directory.parquet")
    # Exercise the native ownership handshake, both before and after import.
    var library = _load_library()
    var writer = _Library._symbol(library.handle, "dfq_write_parquet")
    for codec in ["unknown", "zstd"]:
        var array = ArrowArray()
        var schema = ArrowSchema()
        var releases = 0
        _export_frame(source, array, schema, Int(Pointer(to=releases)))
        var error = 0
        var status = _call_writer(
            writer,
            "/dev/null/not-a-directory.parquet",
            codec,
            100,
            array,
            schema,
            error,
        )
        assert_true(status != 0)
        _release_imported(array, schema)
        assert_equal(releases, 1 + source.width())
        assert_equal(array.release, 0)
        assert_equal(schema.release, 0)
        with assert_raises():
            _raise_backend_error(library, error, "write_parquet")


def main() raises:
    try:
        print("libdfparquet: arrow", parquet_backend_version())
    except e:
        print("skipped:", e)
        return
    TestSuite.discover_tests[__functions_in_module()]().run()
