"""Independent schema examples for sampled JSON object inference."""
# contract-group: kernel.json_schema
from std.testing import assert_equal, assert_raises
from dataframe.dtype import DataType
from dataframe.json_reader import JsonReader, json_bytes
from dataframe.io.json_schema import JsonSchema


def _infer(documents: List[String]) raises -> DataType:
    var schema = JsonSchema()
    var empty = String("")
    var reader = JsonReader(json_bytes(empty))
    for text in documents:
        reader.parse(json_bytes(text))
        schema.observe(reader)
    return schema.finish()


def test_json_schema_unions_fields_promotes_numbers_and_retains_nulls() raises:
    var documents: List[String] = [
        '{"a":null,"items":[],"sub":{"x":1}}',
        '{"a":1,"items":[null],"sub":{"y":"é"},"later":null}',
        '{"a":2.5,"items":[1,2.0],"sub":{"x":null}}',
    ]
    var expected = DataType.struct(
        ["a", "items", "sub", "later"],
        [
            DataType.FLOAT64,
            DataType.list(DataType.FLOAT64),
            DataType.struct(["x", "y"], [DataType.INT64, DataType.STRING]),
            DataType.NULL,
        ],
    )
    assert_equal(_infer(documents), expected)
    assert_equal(DataType.parse("list[null]"), DataType.list(DataType.NULL))
    var nulls: List[String] = ['{"a":null,"b":[]}', '{"a":null,"b":[null]}']
    assert_equal(
        _infer(nulls),
        DataType.struct(
            ["a", "b"], [DataType.NULL, DataType.list(DataType.NULL)]
        ),
    )
    var objects: List[String] = ["{}", "{}"]
    assert_equal(
        _infer(objects), DataType.struct(List[String](), List[DataType]())
    )


def test_json_schema_mixed_scalars_names_duplicates_and_invalid_shapes() raises:
    var documents: List[String] = [
        '{"s":1,"b":true,"quote\\"é":1,"duplicate":1,"duplicate":{}}',
        '{"s":"text","b":2,"quote\\"é":2}',
    ]
    assert_equal(
        _infer(documents),
        DataType.struct(
            ["s", "b", 'quote"é', "duplicate"],
            [DataType.STRING, DataType.STRING, DataType.INT64, DataType.INT64],
        ),
    )
    var incompatible: List[String] = ['{"a":[1]}', '{"a":1}']
    with assert_raises(contains="Cannot infer"):
        _ = _infer(incompatible)
    var scalar: List[String] = ["null"]
    with assert_raises(contains="objects"):
        _ = _infer(scalar)
    var empty = List[String]()
    with assert_raises(contains="empty"):
        _ = _infer(empty)
