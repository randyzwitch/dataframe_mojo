"""Prepared table decoding keeps batches independent and errors local."""
# contract-group: kernel.json
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import DataType, Series
from dataframe.json_reader import JsonReader, json_bytes
from dataframe.json_number import validate_json_numbers
from dataframe.json_decode import JsonDecoder, JsonDecodeOptions


def _decode(
    text: String, dtype: DataType, options: JsonDecodeOptions
) raises -> Series:
    var reader = JsonReader(json_bytes(text))
    reader.parse(json_bytes(text))
    validate_json_numbers(reader)
    var decoder = JsonDecoder(dtype, 1, options)
    decoder.append(reader)
    return decoder.finish()


def test_prepared_json_decoder_moves_batches_without_losing_previous_buffers() raises:
    var dtype = DataType.struct(
        ["x", "s", "items", "n"],
        [
            DataType.INT64,
            DataType.STRING,
            DataType.list(DataType.INT64),
            DataType.NULL,
        ],
    )
    var decoder = JsonDecoder(dtype, 2, JsonDecodeOptions(io=True))
    var empty = String("")
    var reader = JsonReader(json_bytes(empty))
    var documents: List[String] = [
        '{"x":1,"s":"long external string é 😀","items":[1,null]}',
        '{"x":2,"s":"second long string","items":[]}',
        '{"x":3,"s":"third","items":[3]}',
    ]
    for i in range(2):
        reader.parse(json_bytes(documents[i]))
        decoder.append(reader)
    assert_equal(decoder.rows(), 2)
    var first = decoder.finish()
    assert_equal(decoder.rows(), 0)
    reader.parse(json_bytes(documents[2]))
    decoder.append(reader)
    var second = decoder.finish()
    assert_equal(first.dtype(), dtype)
    assert_equal(second.dtype(), dtype)
    assert_equal(len(first), 2)
    assert_equal(len(second), 1)
    assert_equal(first.struct_column().field("x").get(1).int64(), Int64(2))
    assert_equal(
        first.struct_column().field("s").get(0).string(),
        "long external string é 😀",
    )
    assert_equal(second.struct_column().field("x").get(0).int64(), Int64(3))
    assert_equal(
        first.struct_column().field("items").list_column().child().null_count(),
        1,
    )
    assert_equal(
        second.struct_column()
        .field("items")
        .list_column()
        .child()
        .get(0)
        .int64(),
        Int64(3),
    )
    assert_equal(first.struct_column().field("n").null_count(), 2)
    assert_equal(len(decoder.finish()), 0)


def test_prepared_json_io_policy_ignores_bad_cells_and_rejects_unknown_fields() raises:
    var dtype = DataType.struct(
        ["a", "items", "sub"],
        [
            DataType.INT64,
            DataType.list(DataType.INT64),
            DataType.struct(["ok"], [DataType.INT64]),
        ],
    )
    var text = String(
        '{"a":true,"items":[1,"bad",3.5,null],"sub":{"ok":"bad","extra":3},"extra":9}'
    )
    var result = _decode(
        text,
        dtype,
        JsonDecodeOptions(
            io=True, ignore_errors=True, reject_unknown_fields=True
        ),
    )
    var fields = result.struct_column()
    assert_equal(fields.field("a").null_count(), 1)
    var items = fields.field("items").list_column().child()
    assert_equal(len(items), 4)
    assert_equal(items.get(0).int64(), Int64(1))
    assert_equal(items.null_count(), 3)
    assert_equal(
        fields.field("sub").struct_column().field("ok").null_count(), 1
    )
    var unexpected = String('{"a":1,"items":[],"sub":{"ok":1,"unknown":0}}')
    with assert_raises(contains="outside"):
        _ = _decode(
            unexpected,
            dtype,
            JsonDecodeOptions(io=True, reject_unknown_fields=True),
        )
    for value in [
        String("true"),
        String("1.5"),
        String('"bad"'),
        String("128"),
    ]:
        with assert_raises(contains="JSON"):
            _ = _decode(value, DataType.INT8, JsonDecodeOptions(io=True))
    var scalar = String("2")
    var ignored = _decode(
        scalar,
        DataType.list(DataType.INT64),
        JsonDecodeOptions(io=True, ignore_errors=True),
    )
    assert_equal(ignored.null_count(), 1)
    var root_null = String("null")
    with assert_raises(contains="objects"):
        _ = _decode(
            root_null, dtype, JsonDecodeOptions(io=True, ignore_errors=True)
        )


def test_prepared_json_projection_skips_unselected_conversions() raises:
    var dtype = DataType.struct(
        ["keep", "skip"], [DataType.INT64, DataType.INT64]
    )
    var columns: List[String] = ["keep"]
    var decoder = JsonDecoder(
        dtype,
        4,
        JsonDecodeOptions(io=True, reject_unknown_fields=True),
        columns,
    )
    var text = String('{"keep":1,"skip":"invalid unselected integer"}')
    var reader = JsonReader(json_bytes(text))
    reader.parse(json_bytes(text))
    decoder.append(reader)
    var output = decoder.finish()
    assert_equal(output.dtype(), DataType.struct(["keep"], [DataType.INT64]))
    assert_equal(output.struct_column().field("keep").get(0).int64(), Int64(1))
    assert_equal(decoder.output_dtype(), output.dtype())
    var missing: List[String] = ["missing"]
    with assert_raises(contains="not found"):
        _ = JsonDecoder(dtype, 1, JsonDecodeOptions(io=True), missing)
