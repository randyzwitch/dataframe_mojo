"""Nested JSON expressions retain all values across serial and parallel batches."""
# contract-group: expr.json
from std.testing import assert_equal, assert_true
from dataframe import (
    DataFrame,
    DataType,
    Series,
    Column,
    StringColumn,
    col,
    as_struct,
)
from dataframe.string_column import StringBuilder
from tests.support.group_strategy_reference import ScopedEnvironment
from tests.public_contracts.test_text_functions import _source


def test_json_decode_nested_many_serial_and_parallel_batches() raises:
    var n = 131_073
    var strings = StringBuilder(n, n * 72)
    for row in range(n):
        if row % 17 == 0:
            strings.append_null()
        else:
            var integer = String(row)
            strings.append(
                '{"i":'
                + integer
                + ',"items":['
                + integer
                + ',null],"label":"é"}'
            )
    var source = DataFrame([Series("s", strings^.finish())])
    var dtype = DataType.struct(
        ["i", "items", "label"],
        [DataType.INT64, DataType.list(DataType.INT64), DataType.STRING],
    )
    for workers in [1, 2]:
        var environment = ScopedEnvironment(
            "DATAFRAME_THREADS", String(workers)
        )
        var result = source.select(
            col("s").str().json_decode(dtype), batch_size=127
        ).column("s")
        var output = result.struct_column()
        var integers = output.field("i")
        var lists = output.field("items").list_column()
        var values = lists.child()
        var labels = output.field("label")
        assert_equal(len(result), n)
        for row in range(n):
            var valid = row % 17 != 0
            assert_equal(output._valid(row), valid)
            assert_equal(integers._data[Column[Int64]]._valid(row), valid)
            assert_equal(lists._valid(row), valid)
            assert_equal(labels._data[StringColumn]._valid(row), valid)
            if valid:
                assert_equal(
                    integers._data[Column[Int64]]._get(row), Int64(row)
                )
                assert_equal(lists._physical_count(row), 2)
                var start = lists._start(row)
                assert_equal(
                    values._data[Column[Int64]]._get(start), Int64(row)
                )
                assert_true(values._data[Column[Int64]]._valid(start))
                assert_true(not values._data[Column[Int64]]._valid(start + 1))
                assert_equal(String(labels._data[StringColumn]._get(row)), "é")
        _ = environment^


def test_json_nested_categorical_field_get_and_repack_across_batches() raises:
    var text: List[String] = [
        '{"c":"z","items":["é","z"]}',
        '{"c":"é","items":["z"]}',
        "null",
        '{"c":"z","items":[]}',
        "{}",
    ]
    var valid: List[Bool] = [True, True, True, True, True]
    var dtype = DataType.struct(
        ["c", "items"],
        [DataType.CATEGORICAL, DataType.list(DataType.CATEGORICAL)],
    )
    var value = col("s").str().json_decode(dtype)
    var field = value.field("c")
    var element = value.field("items").list().get(0)
    var expected: List[String] = ["z", "é", "", "z", ""]
    var expected_element: List[String] = ["é", "z", "", "", ""]
    var present: List[Bool] = [True, True, False, True, False]
    for layout in range(4):
        var frame = _source(text, valid, layout)
        for streaming in [False, True]:
            var result = (
                frame.lazy()
                .select_exprs(
                    [
                        field.alias("c"),
                        element.alias("element"),
                        as_struct(
                            [field.alias("c"), element.alias("element")]
                        ).alias("packed"),
                    ]
                )
                .collect(streaming=streaming, batch_size=1)
            )
            assert_true(result.column("c").dtype().is_categorical())
            assert_true(result.column("element").dtype().is_categorical())
            for row in range(len(text)):
                assert_equal(result.item(row, "c").is_null(), not present[row])
                assert_equal(result.item(row, "element").is_null(), row >= 2)
                assert_equal(
                    result.item(row, "packed").struct_field("c").is_null(),
                    not present[row],
                )
                if present[row]:
                    assert_equal(result.item(row, "c").string(), expected[row])
                    assert_equal(
                        result.item(row, "packed").struct_field("c").string(),
                        expected[row],
                    )
                if row < 2:
                    assert_equal(
                        result.item(row, "element").string(),
                        expected_element[row],
                    )
                    assert_equal(
                        result.item(row, "packed")
                        .struct_field("element")
                        .string(),
                        expected_element[row],
                    )
        var empty = frame.clear().select(
            as_struct([field.alias("c")]).alias("packed")
        )
        assert_true(
            empty.column("packed").dtype().field_dtype(0).is_categorical()
        )
