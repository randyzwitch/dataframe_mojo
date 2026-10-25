"""IPC interoperability with independent files and actual writer outputs."""
# contract-group: io.ipc
from std.os import getenv
from std.testing import assert_equal, assert_true
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Series,
    StringColumn,
    col,
    concat,
    ArrowArray,
    ArrowSchema,
    export_arrow_series,
    read_ipc,
    read_ipc_stream,
    read_ipc_schema,
    write_ipc,
    write_ipc_stream,
)
from dataframe.io.arrow import _at, _release_imported
from tests.interchange.test_parquet import assert_typed_rows
from tests.support.arrow_foreign import packed_metadata, text_bytes


def fixture(name: String) -> String:
    return "tests/fixtures/ipc/" + name


def read(path: String, stream: Bool) raises -> DataFrame:
    return read_ipc_stream(path) if stream else read_ipc(path)


def check_core(frame: DataFrame, rows: Int, names: List[String]) raises:
    assert_equal(frame.columns(), names)
    assert_equal(frame.height(), rows)
    var integers: List[Int64] = [0, 1, 2, 3, 100, 101, 102, 200, 201]
    var strings: List[String] = [
        "red",
        "",
        "blue",
        "",
        "red",
        "green",
        "red",
        "blue",
        "last",
    ]
    for row in range(rows):
        for name in names:
            if name == "x":
                assert_equal(frame.item(row, name).int64(), integers[row])
            elif name == "s":
                assert_equal(frame.item(row, name).is_null(), row == 3)
                if row != 3:
                    assert_equal(frame.item(row, name).string(), strings[row])
            else:
                assert_equal(frame.item(row, name).is_null(), row == 2)
                if row != 2:
                    assert_equal(
                        frame.item(row, name).float64(), Float64(row + 1)
                    )


def test_ipc_file_stream_codecs_projection_limits_and_record_batch_chunks() raises:
    for stream in [False, True]:
        var suffix = ".stream" if stream else ".arrow"
        for codec in ["uncompressed", "lz4", "zstd"]:
            var path = fixture("core-" + codec + suffix)
            for mapped in [False, True]:
                var full = read_ipc_stream(
                    path, memory_map=mapped
                ) if stream else read_ipc(path, memory_map=mapped)
                check_core(full, 9, ["x", "s", "score"])
                assert_equal(len(full.column("x").chunks()), 3)
                var projected = read_ipc_stream(
                    path, columns=["score", "s"], n_rows=5, memory_map=mapped
                ) if stream else read_ipc(
                    path, columns=["score", "s"], n_rows=5, memory_map=mapped
                )
                check_core(projected, 5, ["score", "s"])
                assert_equal(len(projected.column("s").chunks()), 2)
                var none = read_ipc_stream(
                    path, columns=List[String](), n_rows=4
                ) if stream else read_ipc(
                    path, columns=List[String](), n_rows=4
                )
                assert_equal(none.height(), 4)
                assert_equal(none.width(), 0)
                var empty = read_ipc_stream(
                    path, n_rows=0
                ) if stream else read_ipc(path, n_rows=0)
                check_core(empty, 0, ["x", "s", "score"])


def test_ipc_schema_only_and_row_limit_do_not_read_later_corrupt_bodies() raises:
    for stream in [False, True]:
        var path = fixture("late-corrupt" + (".stream" if stream else ".arrow"))
        var fields = read_ipc_schema(path, stream=stream)
        assert_equal(len(fields), 3)
        assert_equal(fields[0].name, "x")
        assert_equal(fields[0].dtype, DataType.INT64)
        assert_equal(fields[1].dtype, DataType.STRING)
        var first = read_ipc_stream(path, n_rows=3) if stream else read_ipc(
            path, n_rows=3
        )
        check_core(first, 3, ["x", "s", "score"])
        var rejected = False
        try:
            _ = read(path, stream)
        except error:
            assert_true(String(error).find("stream:") >= 0)
            rejected = True
        assert_true(
            rejected,
            "full read must report the independently corrupted later body",
        )
        # A failed stream must not poison subsequent opens.
        check_core(
            read(
                fixture(
                    "core-uncompressed" + (".stream" if stream else ".arrow")
                ),
                stream,
            ),
            9,
            ["x", "s", "score"],
        )


def test_ipc_empty_and_zero_column_external_files_retain_schema_and_rows() raises:
    for stream in [False, True]:
        var suffix = ".stream" if stream else ".arrow"
        check_core(
            read(fixture("empty" + suffix), stream), 0, ["x", "s", "score"]
        )
        var zero = read(fixture("zero-columns" + suffix), stream)
        assert_equal(zero.height(), 9)
        assert_equal(zero.width(), 0)


def check_metadata(series: Series) raises:
    var wanted = packed_metadata(
        [text_bytes("application")], [text_bytes("independent")]
    )
    var array = ArrowArray()
    var schema = ArrowSchema()
    export_arrow_series(series, array, schema)
    assert_true(schema.metadata != 0)
    for i in range(len(wanted)):
        assert_equal(_at[UInt8](schema.metadata).unsafe_offset(i)[], wanted[i])
    _release_imported(array, schema)


def test_ipc_nested_temporal_decimal_widths_and_field_metadata_survive_import_and_write() raises:
    for stream in [False, True]:
        var suffix = ".stream" if stream else ".arrow"
        var full = read(fixture("typed" + suffix), stream)
        assert_typed_rows(full.drop(["narrow_precision_wide_storage"]))
        assert_equal(full.column("decimal").dtype(), DataType.decimal(9, 2, 32))
        assert_equal(
            full.column("narrow_precision_wide_storage").dtype(),
            DataType.decimal(9, 2, 128),
        )
        check_metadata(full.column("i"))
        var path = getenv("COVERAGE_TEST_TMP") + "/ipc-written-typed" + suffix
        if stream:
            write_ipc_stream(full, path, batch_size=2)
        else:
            write_ipc(full, path, batch_size=2)
        var back = read(path, stream)
        assert_true(back.equals(full))
        check_metadata(back.column("i"))
    var views = read_ipc(fixture("views.arrow"))
    assert_equal(views.item(0, "text").string(), "long external string")
    assert_equal(views.item(2, "text").string(), "é")
    var binary: List[UInt8] = [255, 0]
    assert_equal(views.item(0, "binary").bytes(), binary)
    assert_true(views.item(1, "binary").is_null())


def test_ipc_dictionary_deltas_and_replacements_keep_chunk_codes_aligned() raises:
    for name in [
        "dictionary-delta.arrow",
        "dictionary-delta.stream",
        "dictionary-replacement.stream",
    ]:
        var stream = name.endswith(".stream")
        var frame = read(fixture(name), stream)
        assert_equal(frame.height(), 6)
        assert_true(frame.column("category").dtype().is_categorical())
        assert_equal(len(frame.column("category").chunks()), 2)
        var labels: List[String] = ["red", "blue", "", "red", "green", "blue"]
        for row in range(6):
            assert_equal(frame.item(row, "x").int64(), Int64(row))
            assert_equal(frame.item(row, "category").is_null(), row == 2)
            if row != 2:
                assert_equal(frame.item(row, "category").string(), labels[row])
        var grouped = frame.group_by("category").agg(col("x").sum())
        assert_equal(grouped.height(), 4)
        for row in range(4):
            var key = grouped.item(row, "category")
            var expected = Int64(2)
            if not key.is_null():
                expected = Int64(
                    3 if key.string()
                    == "red" else (6 if key.string() == "blue" else 4)
                )
            assert_equal(grouped.item(row, "x").int64(), expected)
        if name == "dictionary-replacement.stream":
            var root = getenv("COVERAGE_TEST_TMP") + "/ipc-written-dictionary"
            write_ipc(frame, root + ".arrow", batch_size=3)
            write_ipc_stream(frame, root + ".stream", batch_size=3)
            assert_true(read_ipc(root + ".arrow").equals(frame))
            assert_true(read_ipc_stream(root + ".stream").equals(frame))


def test_ipc_nested_dictionary_changes_and_empty_dictionary_schemas_preserve_values() raises:
    for name in ["dictionary-delta.arrow", "dictionary-replacement.stream"]:
        var fields = read_ipc_schema(
            fixture(name), stream=name.endswith(".stream")
        )
        assert_true(fields[1].dtype.is_categorical())
        var empty = read_ipc(fixture(name), n_rows=0) if name.endswith(
            ".arrow"
        ) else read_ipc_stream(fixture(name), n_rows=0)
        assert_equal(empty.height(), 0)
        assert_true(empty.column("category").dtype().is_categorical())
    var frame = read_ipc_stream(fixture("nested-dictionary.stream"))
    assert_equal(frame.height(), 4)
    assert_equal(len(frame.column("items").chunks()), 2)
    var labels: List[String] = ["red", "blue", "red", "green"]
    for row in range(4):
        var list = frame.item(row, "items").list()
        assert_equal(list.get(0).string(), labels[row])
        assert_equal(len(list), 1 if row % 2 == 0 else 2)
        if row == 1:
            assert_true(list.get(1).is_null())
        elif row == 3:
            assert_equal(list.get(1).string(), "blue")
        assert_equal(frame.item(row, "record").is_null(), row == 1)
        if row != 1:
            assert_equal(
                frame.item(row, "record").struct_field("category").string(),
                labels[row],
            )
    var root = getenv("COVERAGE_TEST_TMP") + "/ipc-written-nested-dictionary"
    # A normal concat can combine nested chunks with independent dictionaries.
    # The writer must also normalize frames that did not come from one read.
    var independent = concat(
        [
            read_ipc_stream(fixture("nested-dictionary.stream"), n_rows=2),
            frame.slice(2, 2),
        ]
    )
    # Series.equals also compares categorical dictionary layouts. Check cells
    # here; the input deliberately has different dictionaries in its chunks.
    assert_equal(independent.height(), 4)
    for row in range(4):
        assert_equal(
            independent.item(row, "items").list().get(0).string(), labels[row]
        )
    write_ipc(independent, root + ".arrow", batch_size=2)
    write_ipc_stream(independent, root + ".stream", batch_size=2)
    assert_true(read_ipc(root + ".arrow").equals(frame))
    assert_true(read_ipc_stream(root + ".stream").equals(frame))


def test_ipc_writer_preserves_unaligned_chunks_nulls_empty_and_zero_column_frames() raises:
    var x = Series(
        "x",
        Column[Int64](
            [0, 1, 0, 3, 4, 5], [True, True, False, True, True, True]
        ),
    )
    var s = Series(
        "s",
        StringColumn(
            ["red", "", "λ", "", "green", "last"],
            [True, True, True, False, True, True],
        ),
    )
    var source = DataFrame(
        [
            Series._from_chunks([x.slice(0, 2), x.slice(2, 4)]),
            Series._from_chunks([s.slice(0, 1), s.slice(1, 3), s.slice(4, 2)]),
        ]
    )
    for stream in [False, True]:
        var suffix = ".stream" if stream else ".arrow"
        var root = getenv("COVERAGE_TEST_TMP") + "/ipc-written-"
        for codec in ["uncompressed", "lz4", "zstd"]:
            var path = root + codec + suffix
            if stream:
                write_ipc_stream(source, path, compression=codec, batch_size=3)
            else:
                write_ipc(source, path, compression=codec, batch_size=3)
            var back = read(path, stream)
            assert_true(back.equals(source))
            assert_equal(len(back.column("x").chunks()), 4)
        var empty = root + "empty" + suffix
        var zero = root + "zero-columns" + suffix
        if stream:
            write_ipc_stream(source.clear(), empty)
            write_ipc_stream(
                DataFrame(List[Series](), height=7), zero, batch_size=3
            )
        else:
            write_ipc(source.clear(), empty)
            write_ipc(DataFrame(List[Series](), height=7), zero, batch_size=3)
        assert_true(read(empty, stream).equals(source.clear()))
        assert_equal(read(zero, stream).height(), 7)
        assert_equal(read(zero, stream).width(), 0)


def test_ipc_invalid_inputs_and_writer_options_raise_and_keep_existing_output() raises:
    var source = read_ipc(fixture("core-uncompressed.arrow"))
    var existing = getenv("COVERAGE_TEST_TMP") + "/ipc-existing.arrow"
    write_ipc(source, existing)
    for mode in range(11):
        var rejected = False
        try:
            if mode == 0:
                _ = read_ipc("")
            elif mode == 1:
                _ = read_ipc(fixture("missing.arrow"))
            elif mode == 2:
                _ = read_ipc(fixture("invalid.arrow"))
            elif mode == 3:
                _ = read_ipc_stream(fixture("truncated.stream"))
            elif mode == 4:
                _ = read_ipc(
                    fixture("core-uncompressed.arrow"), columns=["s", "s"]
                )
            elif mode == 5:
                _ = read_ipc(
                    fixture("core-uncompressed.arrow"), columns=["absent"]
                )
            elif mode == 6:
                _ = read_ipc(fixture("core-uncompressed.arrow"), n_rows=-1)
            elif mode == 7:
                _ = read_ipc("bad\x00path")
            elif mode == 8:
                write_ipc(source, existing, compression="snappy")
            elif mode == 9:
                write_ipc(source, existing, batch_size=0)
            else:
                write_ipc(
                    source,
                    getenv("COVERAGE_TEST_TMP") + "/missing-folder/fail.arrow",
                )
        except:
            rejected = True
        assert_true(rejected, "invalid IPC case must raise: " + String(mode))
    assert_true(read_ipc(existing).equals(source))
