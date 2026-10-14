"""Decoded source staging retains only projected physical windows and metadata."""

from std.memory import ArcPointer
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from dataframe import (
    BoolColumn,
    Column,
    CsvField,
    CsvSchema,
    DataFrame,
    DataType,
    ListColumn,
    Series,
    StringColumn,
    col,
    read_csv,
    scan_csv,
)
from dataframe.dtype import CategoricalDictionary, NUMERIC_DTYPES
from dataframe.lazy import LazyFrame, SELECT, _plan_node
from dataframe.string_column import StringCodes
from dataframe.accelerator.source_compaction import compact_source
from dataframe.accelerator.sources import (
    file_source_nodes,
    source_payload_bytes,
    stage_file_sources,
)
from tests.tooling.contract_runner import (
    temporary_directory,
    write_text,
    run_command,
)


def _fixed[
    T: Copyable & Deinitable & Equatable & Writable
](values: List[T], dtype: DataType) raises:
    var source = Series._wrap(
        "value", Column[T](values.copy(), [True, False, True, False, True])
    ).with_dtype(dtype)
    # One packed Arrow metadata pair k=v; retain this exact immutable owner.
    var metadata: List[UInt8] = [1, 0, 0, 0, 1, 0, 0, 0, 107, 1, 0, 0, 0, 118]
    source._field_metadata = ArcPointer(metadata.copy())
    for rows in [0, 1, 3]:
        var window = source.slice(1, rows)
        var compact = window._compact_input()
        assert_true(compact.equals(window))
        assert_equal(compact.dtype(), dtype)
        assert_equal(compact.name(), "value")
        assert_false(compact.is_chunked())
        ref storage = compact._data[Column[T]]
        assert_equal(storage._offset, 0)
        assert_equal(len(storage._data[]), rows)
        assert_equal(len(storage._bits[]), (rows + 7) // 8)
        assert_true(storage._data.ptr() != source._data[Column[T]]._data.ptr())
        assert_true(storage._bits.ptr() != source._data[Column[T]]._bits.ptr())
        assert_equal(compact._field_metadata.value()[], metadata)
        assert_true(
            compact._field_metadata.value().ptr()
            == source._field_metadata.value().ptr()
        )
        for row in range(rows):
            assert_equal(storage._get(row), values[row + 1])
    assert_equal(len(source._data[Column[T]]._data[]), 5)


def test_file_source_compaction_fixed_widths_logical_types_and_field_metadata() raises:
    comptime for i in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[i]
        _fixed[Scalar[D]](
            [
                Scalar[D](1),
                Scalar[D](2),
                Scalar[D](3),
                Scalar[D](4),
                Scalar[D](5),
            ],
            DataType.of(D),
        )
    _fixed[Int128]([1, 2, 3, 4, 5], DataType.decimal(38, 2))
    _fixed[Int32]([1, 2, 3, 4, 5], DataType.decimal(8, 2, 32))
    _fixed[Int64]([1, 2, 3, 4, 5], DataType.decimal(18, 2, 64))
    _fixed[Int64]([1, 2, 3, 4, 5], DataType.datetime("ns", "+05:30"))
    _fixed[Int64]([1, 2, 3, 4, 5], DataType.duration("ms"))
    _fixed[Int64]([1, 2, 3, 4, 5], DataType.DATE)


def test_file_source_compaction_csv_views_and_zero_row_head_release_backing_owners() raises:
    var work = temporary_directory("/tmp/mojo-file-source-owners")
    var path = work + "/values.csv"
    write_text(path, 'id,s\n0,""\n1,a\n2,abcdefghijklmnop\n3,\n')
    var schema = CsvSchema(
        [CsvField("id", DataType.INT64), CsvField("s", DataType.STRING)]
    )
    var decoded = read_csv(path, schema)
    assert_equal(decoded.height(), 4)
    assert_true(decoded.column("s")._data[StringColumn]._is_view_storage())
    var strings = compact_source(decoded.select(["s"]))
    ref storage = strings._columns[0]._data[StringColumn]
    assert_false(storage._is_view_storage())
    assert_false(Bool(storage._codes))
    assert_equal(storage._offset, 0)
    assert_equal(len(storage._bytes[]), 17)
    assert_equal(len(storage._offsets[]), 5)
    assert_equal(len(storage._bits[]), 1)
    assert_equal(source_payload_bytes(strings), 58)
    assert_equal(strings.item(0, "s").string(), "")
    assert_equal(strings.item(1, "s").string(), "a")
    assert_equal(strings.item(2, "s").string(), "abcdefghijklmnop")
    assert_true(strings.item(3, "s").is_null())
    for rows in [0, 1]:
        var limited = read_csv(path, schema, n_rows=rows)
        assert_equal(limited.height(), rows)
        # Eager reader limits are zero-copy windows over a decoded chunk.
        assert_equal(len(limited.column("id")._data[Column[Int64]]._data[]), 4)
        var compact = compact_source(limited)
        assert_equal(compact.height(), rows)
        assert_equal(
            len(compact.column("id")._data[Column[Int64]]._data[]), rows
        )
        ref text = compact._columns[1]._data[StringColumn]
        assert_false(text._is_view_storage())
        assert_equal(len(text._bytes[]), 0)
        assert_equal(len(text._offsets[]), rows + 1)
        assert_equal(text._offset, 0)
        assert_true(compact.equals(limited))
    write_text(path, "s\n" + String('""\n') * 64)
    var empty_values = read_csv(
        path, CsvSchema([CsvField("s", DataType.STRING)])
    )
    assert_true(empty_values._columns[0]._data[StringColumn]._is_view_storage())
    var compact_empty = compact_source(empty_values)
    assert_equal(compact_empty.height(), 64)
    assert_equal(len(compact_empty._columns[0]._data[StringColumn]._bytes[]), 0)
    assert_equal(
        len(compact_empty._columns[0]._data[StringColumn]._offsets[]), 65
    )
    assert_true(source_payload_bytes(compact_empty) >= 65 * 8)
    run_command(["rm", "-r", work])


def test_file_source_compaction_categorical_domain_and_parquet_string_code_cache() raises:
    var dictionary = CategoricalDictionary()
    for value in ["z", "a", "é🙂", "", "unused"]:
        dictionary.append(value)
    var dtype = DataType.categorical(dictionary^)
    var category = Series(
        "k", Column[UInt32]([3, 0, UInt32.MAX, 2], [True, True, False, True])
    ).with_dtype(dtype)
    var compact = compact_source(DataFrame([category.slice(1, 3)]))
    assert_equal(compact._columns[0].dtype(), dtype)
    assert_true(
        compact._columns[0].dtype().dictionary().ptr()
        == dtype.dictionary().ptr()
    )
    assert_equal(len(compact._columns[0].dtype().dictionary()[]), 5)
    assert_equal(len(compact._columns[0]._data[Column[UInt32]]._data[]), 3)
    assert_equal(compact._columns[0]._data[Column[UInt32]]._get(0), UInt32(0))
    assert_equal(compact._columns[0]._data[Column[UInt32]]._get(1), UInt32.MAX)
    assert_equal(compact._columns[0]._data[Column[UInt32]]._get(2), UInt32(2))
    assert_equal(source_payload_bytes(compact), 75)
    assert_true(compact._columns[0].equals(category.slice(1, 3)))

    # Match Parquet's physical StringCodes cache without loading a native
    # decoder: it is an optimization, while the public logical type is String.
    var texts = StringColumn(["z", "a", "z", "é🙂"])
    texts._codes = ArcPointer(
        StringCodes(
            ArcPointer(
                List[UInt32]([UInt32(0), UInt32(1), UInt32(0), UInt32(2)])
            ),
            dtype,
        )
    )
    var cached = Series("s", texts^)
    var window = Series._from_chunks([cached.slice(1, 1), cached.slice(2, 2)])
    var plain = window._compact_input()
    assert_equal(plain.dtype(), DataType.STRING)
    assert_false(plain.is_chunked())
    assert_false(Bool(plain._data[StringColumn]._codes))
    assert_false(plain._data[StringColumn]._is_view_storage())
    assert_equal(len(plain._data[StringColumn]._offsets[]), 4)
    assert_equal(len(plain._data[StringColumn]._bytes[]), 8)
    assert_true(plain.equals(window))
    assert_true(Bool(cached._data[StringColumn]._codes))
    assert_equal(plain.get(0).string(), "a")
    assert_equal(plain.get(1).string(), "z")
    assert_equal(plain.get(2).string(), "é🙂")


def test_file_source_compaction_chunked_bool_zero_columns_and_nested_refusal() raises:
    var source = Series(
        "b",
        BoolColumn(
            [True, False, True, False, True], [True, False, True, True, True]
        ),
    )
    var chunks = Series._from_chunks([source.slice(1, 2), source.slice(3, 2)])
    var compact = chunks._compact_input()
    assert_true(compact.equals(chunks))
    assert_false(compact.is_chunked())
    assert_equal(compact._data[BoolColumn]._offset, 0)
    assert_equal(len(compact._data[BoolColumn]._data[]), 1)
    assert_equal(len(compact._data[BoolColumn]._bits[]), 1)
    assert_true(
        compact._data[BoolColumn]._data.ptr()
        != source._data[BoolColumn]._data.ptr()
    )
    for rows in [0, 7]:
        var empty = compact_source(DataFrame(List[Series](), height=rows))
        assert_equal(empty.height(), rows)
        assert_equal(empty.width(), 0)
        assert_equal(source_payload_bytes(empty), 0)
    var nested = Series(
        "nested", ListColumn([0, 1], Series("child", Column[Int64]([1])))
    )
    with assert_raises(contains="Metal unsupported [source]: nested"):
        _ = nested._compact_input()


def test_file_source_staging_projects_and_pushes_reader_heads_before_compacting() raises:
    var work = temporary_directory("/tmp/mojo-file-source-stage")
    var path = work + "/values.csv"
    write_text(path, "id,unused\n0,broken\n1,broken\n2,broken\n3,broken\n")
    var schema = CsvSchema(
        [CsvField("id", DataType.INT64), CsvField("unused", DataType.INT64)]
    )
    for rows in [0, 1]:
        var query = scan_csv(path, schema).select(col("id")).head(rows)
        var staged = stage_file_sources(query)
        assert_equal(staged[2], 1)
        var frame = staged[0]._frames[len(staged[0]._frames) - 1].copy()
        assert_equal(frame.columns(), List[String](["id"]))
        assert_equal(frame.height(), rows)
        assert_equal(len(frame._columns[0]._data[Column[Int64]]._data[]), rows)
        assert_equal(staged[1], 0 if rows == 0 else 9)
        assert_true(
            staged[0]
            .collect(engine="cpu", streaming=False)
            .equals(query.collect(engine="cpu", streaming=False))
        )
    run_command(["rm", "-r", work])


def test_file_source_staging_preserves_decode_failure_on_typed_empty_scan() raises:
    var work = temporary_directory("/tmp/mojo-file-source-failure")
    var path = work + "/broken.csv"
    write_text(path, "x\nbroken\n")
    var schema = CsvSchema([CsvField("x", DataType.INT64)])
    var query = scan_csv(path, schema).select(col("x").sum())
    var metadata = stage_file_sources(query, empty=True)
    assert_equal(metadata[0]._nodes[0].source_failure, "")
    var staged = stage_file_sources(query)
    assert_true("CSV integer" in staged[0]._nodes[0].source_failure)
    assert_equal(staged[0]._frames[len(staged[0]._frames) - 1].height(), 0)
    assert_equal(
        staged[0]._frames[len(staged[0]._frames) - 1]._columns[0].dtype(),
        DataType.INT64,
    )
    # An actual empty source permits SUM's identity; a failed source with a
    # readable schema must raise instead of exposing the placeholder identity.
    with assert_raises(contains="CSV integer"):
        _ = staged[0].collect(engine="cpu", streaming=False)
    write_text(path, "x\n")
    var empty = stage_file_sources(query)
    assert_equal(empty[0]._nodes[0].source_failure, "")
    assert_equal(
        empty[0].collect(engine="cpu", streaming=False).item(0, "x").int64(),
        Int64(0),
    )
    run_command(["rm", "-r", work])


def test_file_source_staging_does_not_probe_unreachable_missing_csv() raises:
    var work = temporary_directory("/tmp/mojo-file-source-reachability")
    var path = work + "/valid.csv"
    write_text(path, "x\n7\n9\n")
    var schema = CsvSchema([CsvField("x", DataType.INT64)])
    var missing = scan_csv(work + "/missing.csv", schema)
    var valid = scan_csv(path, schema)
    var root = valid._nodes[0].copy()
    root.offset = len(missing._frames)
    var frames = missing._frames.copy()
    frames.extend(valid._frames.copy())
    var schemas = missing._schemas.copy()
    schemas.extend(valid._schemas.copy())
    var query = LazyFrame(
        [
            missing._nodes[0].copy(),
            _plan_node(SELECT, 0, exprs=[col("x")]),
            root^,
        ],
        frames^,
        schemas^,
    )
    var sources = file_source_nodes(query)
    assert_equal(len(sources), 1)
    assert_equal(sources[0], 2)
    var staged = stage_file_sources(query)
    assert_equal(staged[2], 1)
    assert_equal(len(file_source_nodes(staged[0])), 0)
    # The dead SELECT references x, but must not make projection analysis
    # discover the dead source's schema or touch its nonexistent path.
    var result = staged[0].collect(
        engine="cpu", optimize=False, streaming=False
    )
    assert_equal(result.height(), 2)
    assert_equal(result.columns(), List[String](["x"]))
    assert_equal(result.item(0, "x").int64(), Int64(7))
    assert_equal(result.item(1, "x").int64(), Int64(9))
    run_command(["rm", "-r", work])
