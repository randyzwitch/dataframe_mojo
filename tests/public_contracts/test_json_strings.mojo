"""JSON string decoding: independently specified cells and parser failures."""
# contract-group: expr.json
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Series,
    StructColumn,
    ListColumn,
    col,
    Expr,
)
from tests.public_contracts.test_text_functions import _source
from tests.support.coverage_compare import assert_frame
from dataframe.dtype import CategoricalDictionary


def _bits(valid: List[Bool]) -> List[UInt8]:
    var result = List[UInt8](length=(len(valid) + 7) // 8, fill=0)
    for i in range(len(valid)):
        if valid[i]:
            result[i // 8] |= UInt8(1) << UInt8(i % 8)
    return result^


def test_json_decode_nested_schema_layouts_nulls_missing_and_duplicate_fields() raises:
    var dtype = DataType.struct(
        ["a", "b", "text", "items", "sub"],
        [
            DataType.INT64,
            DataType.BOOL,
            DataType.STRING,
            DataType.list(DataType.INT64),
            DataType.struct(["x"], [DataType.FLOAT64]),
        ],
    )
    var values: List[String] = [
        '{"a":1,"b":true,"text":"é\\uD83D\\uDE00","items":[1,null,2],"sub":{"x":1.5}}',
        '{"a":2,"items":[],"extra":[0]}',
        "not JSON; null payload must not be parsed",
        "null",
        '{"a":3,"a":9,"b":false,"text":"","items":4,"sub":{}}',
        "{}",
    ]
    var valid: List[Bool] = [True, True, False, True, True, True]
    var children = List[Series]()
    children.append(
        Series(
            "a",
            Column[Int64](
                [1, 2, 0, 0, 3, 0], [True, True, False, False, True, False]
            ),
        )
    )
    children.append(
        Series(
            "b",
            Column[Bool](
                [True, False, False, False, False, False],
                [True, False, False, False, True, False],
            ),
        )
    )
    children.append(
        Series(
            "text",
            Column[String](
                ["é😀", "", "", "", "", ""],
                [True, False, False, False, True, False],
            ),
        )
    )
    var offsets: List[Int64] = [0, 3, 3, 3, 3, 4, 4]
    children.append(
        Series(
            "items",
            ListColumn(
                offsets^,
                Series(
                    "item",
                    Column[Int64]([1, 0, 2, 4], [True, False, True, True]),
                ),
                _bits([True, True, False, False, True, False]),
            ),
        )
    )
    children.append(
        Series(
            "sub",
            StructColumn(
                [
                    Series(
                        "x",
                        Column[Float64](
                            [1.5, 0, 0, 0, 0, 0],
                            [True, False, False, False, False, False],
                        ),
                    )
                ],
                _bits([True, False, False, False, True, False]),
            ),
        )
    )
    var expected = DataFrame(
        [
            Series(
                "j",
                StructColumn(
                    children^, _bits([True, True, False, False, True, True])
                ),
            )
        ]
    )
    for layout in range(4):
        var frame = _source(values, valid, layout)
        var expr = col("s").str().json_decode(dtype).alias("j")
        assert_frame(frame.select(expr), expected)
        assert_frame(frame.lazy().select(expr).collect(engine="cpu"), expected)
        assert_equal(
            frame.lazy().select(expr).collect_schema(),
            [String("j: ") + dtype.name()],
        )


def test_json_decode_numeric_coercion_null_overflow_and_wrong_type() raises:
    var frame = DataFrame(
        [
            Series(
                "s",
                Column[String](["1", "127.9", "128", "-128.9", "true", "null"]),
            )
        ]
    )
    assert_frame(
        frame.select(col("s").str().json_decode(DataType.INT8).alias("v")),
        DataFrame(
            [
                Series(
                    "v",
                    Column[Int8](
                        [1, 127, 0, -128, 1, 0],
                        [True, True, False, True, True, False],
                    ),
                )
            ]
        ),
    )
    with assert_raises(contains="numeric"):
        _ = DataFrame([Series("s", Column[String](['"2"']))]).select(
            col("s").str().json_decode(DataType.INT64)
        )
    with assert_raises():
        _ = DataFrame(
            [Series("s", Column[String](['{"a":1,"ignored":1e400}']))]
        ).select(
            col("s").str().json_decode(DataType.struct(["a"], [DataType.INT64]))
        )
    with assert_raises(contains="does not support"):
        _ = (
            frame.lazy()
            .select(col("s").str().json_decode(DataType.BINARY))
            .collect_schema()
        )


def test_json_decode_strings_bool_temporal_and_decimal() raises:
    var strings = DataFrame(
        [
            Series(
                "s",
                Column[String](['"a\\n\\u00e9"', "1", "true", "null", "1e2"]),
            )
        ]
    )
    assert_frame(
        strings.select(col("s").str().json_decode(DataType.STRING).alias("v")),
        DataFrame(
            [
                Series(
                    "v",
                    Column[String](
                        ["a\né", "1", "true", "", "100"],
                        [True, True, True, False, True],
                    ),
                )
            ]
        ),
    )
    var date = DataFrame(
        [
            Series(
                "s",
                Column[String](
                    ['"1970-01-02"', "1.9", '"bad"', "2147483648", "null"]
                ),
            )
        ]
    )
    assert_frame(
        date.select(col("s").str().json_decode(DataType.DATE).alias("v")),
        DataFrame(
            [
                Series(
                    "v",
                    Column[Int64](
                        [1, 1, 0, 0, 0], [True, True, False, False, False]
                    ),
                ).with_dtype(DataType.DATE)
            ]
        ),
    )
    var dtype = DataType.decimal(8, 2)
    var decimal = DataFrame(
        [
            Series(
                "s",
                Column[String](
                    ['"1.245"', '"1.255"', "1.245", "1.255", "-1.239", "1e2"]
                ),
            )
        ]
    )
    var expected = Series(
        "v",
        Column[Int128](
            [
                Int128(124),
                Int128(126),
                Int128(125),
                Int128(125),
                Int128(-124),
                Int128(10000),
            ]
        ),
    ).with_dtype(dtype)
    assert_frame(
        decimal.select(col("s").str().json_decode(dtype).alias("v")),
        DataFrame([expected^]),
    )


def test_json_path_layouts_first_null_duplicate_order_and_invalid_rows() raises:
    var values: List[String] = [
        '{"a":"é a long external string 😀","list":[null,2]}',
        '{"a":2.0,"list":[1,2]}',
        '{"a":null,"list":[]}',
        '{"a":"first","a":"last","list":[3],"b":{"a":4}}',
        "[1,2]",
        "malformed JSON",
        "null",
        '{"a":{"b":1.0,"a":"é"},"list":[0]}',
    ]
    var selected: List[String] = [
        "é a long external string 😀",
        "2.0",
        "",
        "last",
        "",
        "",
        "",
        '{"b":1.0,"a":"é"}',
    ]
    var first: List[String] = ["", "1", "", "3", "", "", "", "0"]
    for mode in range(3):
        var valid = List[Bool](length=len(values), fill=mode != 2)
        if mode == 1:
            valid[1] = False
            valid[7] = False
        var field_valid = List[Bool](length=len(values), fill=False)
        var first_valid = List[Bool](length=len(values), fill=False)
        for i in [0, 1, 3, 7]:
            field_valid[i] = valid[i]
        for i in [1, 3, 7]:
            first_valid[i] = valid[i]
        var expected = DataFrame(
            [
                Series(
                    "a", Column[String](selected.copy(), field_valid.copy())
                ),
                Series(
                    "recursive",
                    Column[String](selected.copy(), field_valid.copy()),
                ),
                Series(
                    "first", Column[String](first.copy(), first_valid.copy())
                ),
            ]
        )
        var exprs: List[Expr] = [
            col("s").str().json_path_match("$.a").alias("a"),
            col("s").str().json_path_match("$..a").alias("recursive"),
            col("s").str().json_path_match("$.list[*]").alias("first"),
        ]
        for layout in range(4):
            var source = _source(values, valid, layout)
            assert_frame(source.select_exprs(exprs, batch_size=1), expected)
            for optimize in [False, True]:
                for streaming in [False, True]:
                    assert_frame(
                        source.lazy()
                        .select_exprs(exprs)
                        .collect(
                            optimize=optimize, streaming=streaming, batch_size=1
                        ),
                        expected,
                    )
            assert_frame(source.clear().select_exprs(exprs), expected.clear())
            assert_equal(
                source.lazy().select_exprs(exprs).collect_schema(),
                [
                    String("a: ") + DataType.STRING.name(),
                    String("recursive: ") + DataType.STRING.name(),
                    String("first: ") + DataType.STRING.name(),
                ],
            )

    # Scalar filter results retain the current selection; mixed expressions
    # compare Boolean values rather than treating a selection as truthy.
    var documents: List[String] = [
        "[1,2]",
        '[{"x":1},{"x":2}]',
        "[false,true]",
        '{"x":false}',
        "null",
        "{broken",
    ]
    var exprs: List[Expr] = [
        col("s").str().json_path_match("$[?(1 == 1)]").alias("whole"),
        col("s").str().json_path_match("$[?(1 == 2)]").alias("false"),
        col("s").str().json_path_match("$[?(@.x == 1 && 1 == 1)]").alias("and"),
        col("s")
        .str()
        .json_path_match("$[?(@.x || (true == true))]")
        .alias("or"),
    ]
    for nullable in [False, True]:
        var valid = List[Bool](length=6, fill=not nullable)
        var whole_valid: List[Bool] = [True, True, True, True, False, False]
        var or_valid: List[Bool] = [False, False, False, True, False, False]
        if nullable:
            whole_valid = List[Bool](length=6, fill=False)
            or_valid = List[Bool](length=6, fill=False)
        var expected = DataFrame(
            [
                Series("whole", Column[String](documents.copy(), whole_valid)),
                Series(
                    "false",
                    Column[String](
                        documents.copy(), List[Bool](length=6, fill=False)
                    ),
                ),
                Series(
                    "and",
                    Column[String](
                        documents.copy(), List[Bool](length=6, fill=False)
                    ),
                ),
                Series("or", Column[String](documents.copy(), or_valid)),
            ]
        )
        for layout in range(4):
            var source = _source(documents, valid, layout)
            assert_frame(source.select_exprs(exprs, batch_size=1), expected)
            for streaming in [False, True]:
                assert_frame(
                    source.lazy()
                    .select_exprs(exprs)
                    .collect(streaming=streaming, batch_size=1),
                    expected,
                )


def test_json_arguments_validate_on_empty_and_all_null_inputs() raises:
    for rows in [0, 2]:
        var values = List[String](length=rows, fill="invalid payload")
        var valid = List[Bool](length=rows, fill=False)
        var source = _source(values, valid, 0)
        for path in [String(""), String("a"), String("$["), String("$[::0]")]:
            with assert_raises(contains="JSONPath"):
                _ = (
                    source.lazy()
                    .select(col("s").str().json_path_match(path))
                    .collect_schema()
                )
        with assert_raises(contains="does not support"):
            _ = (
                source.lazy()
                .select(col("s").str().json_decode(DataType.BINARY))
                .collect_schema()
            )
        var nested = DataType.struct(["value"], [DataType.list(DataType.INT64)])
        var decoded = source.select(col("s").str().json_decode(nested)).column(
            "s"
        )
        assert_true(decoded.dtype() == nested)
        assert_equal(len(decoded), rows)
        assert_equal(decoded.null_count(), rows)


def test_json_decode_group_window_streaming_and_dictionary() raises:
    var source = DataFrame(
        [
            Series("g", Column[Int64]([1, 2, 1, 2])),
            Series(
                "s",
                Column[String](['{"x":1}', '{"x":2}', '{"x":3}', '{"x":4}']),
            ),
        ]
    )
    var value = (
        col("s")
        .str()
        .json_decode(DataType.struct(["x"], [DataType.INT64]))
        .field("x")
    )
    var exprs: List[Expr] = [
        value.alias("x"),
        value.sum().over("g").alias("group_sum"),
    ]
    var expected = DataFrame(
        [
            Series("x", Column[Int64]([1, 2, 3, 4])),
            Series("group_sum", Column[Int64]([4, 6, 4, 6])),
        ]
    )
    assert_frame(source.select_exprs(exprs), expected)
    for streaming in [False, True]:
        assert_frame(
            source.lazy()
            .select_exprs(exprs)
            .collect(streaming=streaming, batch_size=1),
            expected,
        )
    var grouped = (
        source.lazy()
        .group_by("g")
        .agg([value.sum().alias("x")])
        .sort("g")
        .collect()
    )
    assert_frame(
        grouped,
        DataFrame(
            [
                Series("g", Column[Int64]([1, 2])),
                Series("x", Column[Int64]([4, 6])),
            ]
        ),
    )
    var strings: List[String] = ['"z"', '"é"', "null", '"z"']
    var valid: List[Bool] = [True, True, True, True]
    for layout in range(4):
        var frame = _source(strings, valid, layout)
        var actual = frame.select(
            col("s").str().json_decode(DataType.CATEGORICAL)
        ).column("s")
        assert_true(actual.dtype().is_categorical())
        assert_equal(actual.get(0).string(), "z")
        assert_equal(actual.get(1).string(), "é")
        assert_true(actual.get(2).is_null())
        assert_equal(actual.get(3).string(), "z")


def test_json_categorical_dictionaries_survive_nested_batches() raises:
    var dictionary = CategoricalDictionary()
    dictionary.append("é")
    dictionary.append("z")
    var dtype = DataType.categorical(dictionary^)
    var source = DataFrame(
        [Series("s", Column[String](['"z"', '"é"', "null", '"z"']))]
    )
    var category = source.select(
        col("s").str().json_decode(dtype), batch_size=1
    ).column("s")
    assert_true(category.dtype() == dtype)
    assert_equal(String(category.dtype().dictionary()[].get(0)), "é")
    assert_equal(category.get(0).string(), "z")
    with assert_raises():
        _ = DataFrame([Series("s", Column[String](['"unknown"']))]).select(
            col("s").str().json_decode(dtype)
        )
    var values: List[String] = [
        '{"cat":"é","tags":["z","é"],"records":[{"v":"z"}]}',
        '{"cat":"z","tags":["é"],"records":[{"v":"é"},{"v":"z"}]}',
        "null",
        "{}",
    ]
    var valid: List[Bool] = [True, True, True, True]
    var nested = DataType.struct(
        ["cat", "tags", "records"],
        [
            DataType.CATEGORICAL,
            DataType.list(DataType.CATEGORICAL),
            DataType.list(DataType.struct(["v"], [DataType.CATEGORICAL])),
        ],
    )
    for layout in range(4):
        var frame = _source(values, valid, layout)
        for streaming in [False, True]:
            var result = (
                frame.lazy()
                .select(col("s").str().json_decode(nested))
                .collect(streaming=streaming, batch_size=1)
                .column("s")
            )
            assert_equal(result.get(0).struct_field("cat").string(), "é")
            assert_equal(result.get(1).struct_field("cat").string(), "z")
            assert_true(result.get(2).is_null())
            assert_true(result.get(3).struct_field("cat").is_null())
            var tags = result.get(0).struct_field("tags").list()
            assert_true(tags.dtype().is_categorical())
            assert_equal(tags.get(0).string(), "z")
            assert_equal(tags.get(1).string(), "é")
            var records = result.get(1).struct_field("records").list()
            assert_equal(records.get(0).struct_field("v").string(), "é")
            assert_equal(records.get(1).struct_field("v").string(), "z")
    var arrays = DataFrame(
        [Series("s", Column[String](['["z"]', '["é","z"]', "null", "[]"]))]
    )
    var result = arrays.select(
        col("s").str().json_decode(DataType.list(DataType.CATEGORICAL)),
        batch_size=1,
    ).column("s")
    assert_true(result.get(0).list().dtype().is_categorical())
    assert_equal(result.get(1).list().get(0).string(), "é")
    assert_equal(result.get(1).list().get(1).string(), "z")
    assert_true(result.get(2).is_null())
    assert_equal(len(result.get(3).list()), 0)


def test_json_int128_expression_and_nested_empty_seeds() raises:
    var maximum = Int128(170141183460469231731687303715884105727)
    var dtype = DataType.of(DType.int128)
    var source = DataFrame(
        [
            Series(
                "s",
                Column[String]([String(maximum), String(-maximum - 1), "null"]),
            )
        ]
    )
    var result = source.select(
        col("s").str().json_decode(dtype), batch_size=1
    ).column("s")
    assert_true(result.dtype() == dtype)
    assert_equal(result._data[Column[Int128]]._get(0), maximum)
    assert_equal(result._data[Column[Int128]]._get(1), -maximum - 1)
    assert_equal(result.get(0).numeric[DType.int128](), maximum)
    assert_equal(result.get(1).numeric[DType.int128](), -maximum - 1)
    assert_equal(String(result.get(0)), String(maximum))
    assert_true(result.to_string().find(String(maximum)) >= 0)
    assert_true(result.get(0) != result.get(1))
    assert_true(result._data[Column[Int128]].is_null(2))
    var arrays = DataFrame(
        [
            Series(
                "s", Column[String](["[" + String(maximum) + "]", "[]", "null"])
            )
        ]
    )
    var nested = arrays.select(
        col("s").str().json_decode(DataType.list(dtype)), batch_size=1
    ).column("s")
    assert_equal(
        nested.list_column().child()._data[Column[Int128]]._get(0), maximum
    )
    assert_equal(len(nested.get(1).list()), 0)
    assert_true(nested.get(2).is_null())
