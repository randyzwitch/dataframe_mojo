"""Table JSON readers: schema, batches, framing and observed-row errors."""
# contract-group: io.json
from std.os import getenv
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    Column,
    DataFrame,
    DataType,
    ListColumn,
    StructColumn,
    Series,
)
from dataframe.io.json import read_ndjson, read_json, NdjsonReader
from tests.support.coverage_compare import assert_frame


def _path(name: String) -> String:
    return getenv("COVERAGE_TEST_TMP") + "/" + name


def _write(path: String, text: String) raises:
    with open(path, "w") as file:
        file.write(text)


def test_ndjson_tables_infer_nested_fields_promote_numbers_and_keep_null_schema() raises:
    var path = _path("json-nested.ndjson")
    _write(
        path,
        "\uFEFF\n"
        + '{"a":1,"label":"é\\n\\uD83D\\uDE00","items":[null],"sub":{"left":1}}\r\n'
        + '{"a":2.5,"label":null,"items":[1.5,null],"sub":{"right":true},"later":null}\n'
        + '{"a":null,"label":"tail","items":[],"sub":null}',
    )
    var expected = DataFrame(
        [
            Series("a", Column[Float64]([1, 2.5, 0], [True, True, False])),
            Series(
                "label",
                Column[String](["é\n😀", "", "tail"], [True, False, True]),
            ),
            Series(
                "items",
                ListColumn(
                    [Int64(0), Int64(1), Int64(3), Int64(3)],
                    Series(
                        "item",
                        Column[Float64]([0, 1.5, 0], [False, True, False]),
                    ),
                ),
            ),
            Series(
                "sub",
                StructColumn(
                    [
                        Series(
                            "left",
                            Column[Int64]([1, 0, 0], [True, False, False]),
                        ),
                        Series(
                            "right",
                            Column[Bool](
                                [False, True, False], [False, True, False]
                            ),
                        ),
                    ],
                    [UInt8(3)],
                    length=3,
                ),
            ),
            Series.full_null("later", DataType.NULL, 3),
        ]
    )
    for size in [1, 2, 64]:
        assert_frame(read_ndjson(path, batch_size=size), expected)
    var nulls = _path("json-null-lists.ndjson")
    _write(nulls, '{"n":null,"items":[]}\n{"n":null,"items":[null]}')
    var frame = read_ndjson(nulls, batch_size=1)
    assert_equal(frame.column("n").dtype(), DataType.NULL)
    assert_equal(frame.column("items").dtype(), DataType.list(DataType.NULL))


def test_ndjson_strict_late_fields_ignored_cells_projection_and_row_limits() raises:
    var path = _path("json-late.ndjson")
    _write(path, '{"a":1}\n{"a":2,"late":3}')
    with assert_raises(contains="outside"):
        _ = read_ndjson(path, infer_schema_length=1, batch_size=1)
    var kept = read_ndjson(
        path, infer_schema_length=1, ignore_errors=True, batch_size=1
    )
    assert_frame(kept, DataFrame([Series("a", Column[Int64]([1, 2]))]))
    var schema = DataType.struct(
        ["a", "skip", "items"],
        [DataType.INT64, DataType.INT64, DataType.list(DataType.INT64)],
    )
    _write(
        path,
        '{"a":true,"skip":"bad","items":[1,"bad",null]}\n{"a":2,"items":2}',
    )
    var ignored = read_ndjson(path, schema, ignore_errors=True, batch_size=1)
    assert_equal(ignored.column("a").null_count(), 1)
    assert_equal(ignored.column("skip").null_count(), 2)
    assert_equal(ignored.column("items").null_count(), 1)
    assert_equal(ignored.column("items").list_column().child().null_count(), 2)
    _write(path, '{"a":1,"skip":"bad"}\n{bad}')
    var names: List[String] = ["a"]
    var prefix = read_ndjson(path, schema, columns=names, n_rows=1)
    assert_frame(prefix, DataFrame([Series("a", Column[Int64]([1]))]))
    with assert_raises(contains="line 2"):
        _ = read_ndjson(path, schema, columns=names, ignore_errors=True)
    _write(path, "null")
    with assert_raises(contains="objects"):
        _ = read_ndjson(path, schema, ignore_errors=True)


def test_json_array_tables_empty_objects_and_strict_container_framing() raises:
    var path = _path("json-array.json")
    _write(path, '[\n{"a":1},\n{"a":2,"s":"x"}\n]')
    for size in [1, 2, 64]:
        var frame = read_json(path, batch_size=size)
        assert_equal(frame.height(), 2)
        assert_equal(frame.column("a").get(1).int64(), Int64(2))
        assert_equal(frame.column("s").null_count(), 1)
    _write(path, "[{},{}]")
    assert_equal(read_json(path, batch_size=1).height(), 2)
    assert_equal(read_json(path).width(), 0)
    _write(path, "[]")
    assert_equal(read_json(path).height(), 0)
    var schema = DataType.struct(["a"], [DataType.INT64])
    assert_equal(read_json(path, schema).column("a").dtype(), DataType.INT64)
    for bad in [
        String("[{},]"),
        String("[] trailing"),
        String("[null]"),
        String("{}"),
        String("["),
    ]:
        _write(path, bad)
        with assert_raises(contains="JSON"):
            _ = read_json(path)
    _write(path, "{}\n{}")
    assert_equal(read_ndjson(path, batch_size=1).height(), 2)
    assert_equal(read_ndjson(path).width(), 0)
    _write(path, "")
    with assert_raises(contains="empty"):
        _ = read_ndjson(path)
    assert_equal(read_ndjson(path, schema).column("a").dtype(), DataType.INT64)


def _first_batch(path: String) raises -> DataFrame:
    var reader = NdjsonReader(path, batch_size=1)
    var batch = reader.next()
    return batch.take()


def test_ndjson_batches_own_values_after_source_and_previous_decoder_state_die() raises:
    var path = _path("json-owned.ndjson")
    _write(
        path,
        '{"s":"long value é 😀 that outlives its source","x":1}\n{"s":"second","x":2}',
    )
    var first = _first_batch(path)
    _write(path, "overwritten")
    assert_equal(
        first.column("s").get(0).string(),
        "long value é 😀 that outlives its source",
    )
    assert_equal(first.column("x").get(0).int64(), Int64(1))
