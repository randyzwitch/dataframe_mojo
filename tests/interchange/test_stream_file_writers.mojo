"""Incremental writers publish once, preserve schema, and clean failed output."""
# contract-group: io.sinks
from std.os import getenv
from std.os.path import exists
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    Column,
    DataFrame,
    Series,
    StringColumn,
    read_csv,
    read_parquet,
    parquet_row_group_statistics,
)
from dataframe.io.sinks import FileSink


def _path(name: String) -> String:
    return getenv("COVERAGE_TEST_TMP") + "/stream-writer-" + name


def _source() raises -> DataFrame:
    return DataFrame(
        [
            Series("i", Column[Int64]([1, 2, 3])),
            Series("s", StringColumn(["a,b", "", 'quoted"value'])),
        ]
    )


def _text(path: String) raises -> String:
    with open(path, "r") as file:
        return file.read()


def test_csv_batches_publish_once_with_one_header_and_typed_empty_sentinel() raises:
    var source = _source()
    var path = _path("batches.csv")
    with open(path, "w") as file:
        file.write("original")
    var sink = FileSink(source.clear(), path, "csv", buffer_size=1)
    sink.append(source.head(1))
    assert_equal(_text(path), "original")
    sink.append(source.slice(1, 2))
    assert_equal(sink.emitted_rows, 3)
    assert_true(sink.schema.equals(source.clear()))
    sink.finish()
    assert_equal(_text(path), 'i,s\n1,"a,b"\n2,""\n3,"quoted""value"\n')
    with assert_raises():
        sink.append(source)


def test_empty_csv_writes_only_optional_header() raises:
    var source = _source().clear()
    for header in [True, False]:
        var path = _path("empty-header.csv" if header else "empty.csv")
        var sink = FileSink(source, path, "csv", has_header=header)
        sink.finish()
        assert_equal(_text(path), "i,s\n" if header else "")


def test_categorical_batches_use_their_own_dictionary_with_empty_schema_dictionary() raises:
    var schema = DataFrame(
        [Series("s", StringColumn(List[String]())).cast("categorical")]
    )
    var first = DataFrame(
        [Series("s", StringColumn(["red", "blue"])).cast("categorical")]
    )
    var second = DataFrame(
        [
            Series("s", StringColumn(["blue", "red", "green"])).cast(
                "categorical"
            )
        ]
    )
    var expected: List[String] = ["red", "blue", "blue", "red", "green"]
    for format in ["csv", "parquet"]:
        var path = _path("dictionary-batches." + format)
        var sink = FileSink(schema, path, format, row_group_size=3)
        sink.append(first)
        sink.append(second)
        assert_equal(sink.emitted_rows, 5)
        sink.finish()
        var back = read_csv(path) if format == "csv" else read_parquet(path)
        assert_equal(back.columns(), List[String](["s"]))
        assert_equal(back.height(), len(expected))
        for row in range(len(expected)):
            assert_equal(back.column("s").get(row).string(), expected[row])


def _abandon_csv(path: String) raises -> String:
    var source = _source()
    var sink = FileSink(source.clear(), path, "csv", buffer_size=1)
    var temporary = sink.csv[0].output.temporary
    sink.append(source)
    assert_true(exists(temporary))
    return temporary


def test_abandoned_writer_removes_only_owned_temporary() raises:
    var path = _path("abandoned.csv")
    with open(path, "w") as file:
        file.write("original")
    var temporary = _abandon_csv(path)
    assert_true(not exists(temporary))
    assert_equal(_text(path), "original")
    with assert_raises():
        _ = FileSink(_source(), path, "csv", separator="bad")
    assert_equal(_text(path), "original")


def test_incremental_parquet_groups_span_batch_boundaries_and_empty_schema() raises:
    var source = _source()
    var path = _path("groups.parquet")
    var sink = FileSink(source.clear(), path, "parquet", row_group_size=2)
    for row in range(source.height()):
        sink.append(source.slice(row, 1))
    sink.finish()
    assert_true(read_parquet(path).equals(source))
    var stats = parquet_row_group_statistics(path)
    assert_equal(stats.height(), 2)
    assert_equal(stats.item(0, "rows").int64(), Int64(2))
    assert_equal(stats.item(1, "rows").int64(), Int64(1))
    var empty = _path("empty.parquet")
    var empty_sink = FileSink(source.clear(), empty, "parquet")
    empty_sink.finish()
    assert_true(read_parquet(empty).equals(source.clear()))


def test_parquet_read_source_can_remain_open_until_output_is_published() raises:
    var source = read_parquet("tests/fixtures/parquet/typed.parquet")
    var path = _path("replace.parquet")
    var first = FileSink(source.clear(), path, "parquet")
    first.append(source)
    first.finish()
    var replacement = FileSink(source.clear(), path, "parquet")
    replacement.append(source.head(1))
    assert_true(read_parquet(path).equals(source))
    replacement.append(source.slice(1))
    replacement.finish()
    assert_true(read_parquet(path).equals(source))
