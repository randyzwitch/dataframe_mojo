"""Explicit categorical CSV decoding agrees with String followed by cast."""
from std.collections import Dict
from std.testing import TestSuite, assert_equal, assert_true, assert_raises
from dataframe import (
    CsvField,
    CsvSchema,
    DataFrame,
    DataType,
    read_csv,
    scan_csv,
    col,
)
from test_csv_borrowed import set_threads

comptime PATH = "/tmp/dataframe_mojo_csv_categorical.csv"


def write(text: String) raises:
    with open(PATH, "w") as file:
        file.write(text)


def schema(categorical: Bool) raises -> CsvSchema:
    return CsvSchema(
        [
            CsvField.int64("id"),
            CsvField.categorical("label") if categorical else CsvField.string(
                "label"
            ),
        ]
    )


def same(actual: DataFrame, strings: DataFrame) raises:
    var got = actual.column("label")
    var expected = strings.column("label").cast("categorical")
    assert_true(got.dtype().is_categorical())
    assert_true(got.cast("string").equals(expected.cast("string")))
    assert_equal(got.null_count(), expected.null_count())
    assert_equal(actual.height(), strings.height())


def test_values_nulls_escaping_and_schema_forms() raises:
    write(
        'id,label\r\n1,pear\r\n2,\r\n3,""\r\n4,"say ""hi"""\r\n5,"雪\nline"\r\n6,pear\r\n7,NA\r\n8,"NA"\r\n9, leading \r\n'
    )
    var strings = read_csv(PATH, schema(False), null_values=["NA"])
    var actual = read_csv(PATH, schema(True), null_values=["NA"])
    same(actual, strings)
    assert_equal(actual.column("label").null_count(), 3)
    assert_equal(actual.column("label").get(2).string(), "")
    assert_equal(len(actual.column("label").dtype().dictionary()[]), 5)
    var pairs: List[Tuple[String, DataType]] = [
        ("id", DataType.INT64),
        ("label", DataType.parse("categorical")),
    ]
    same(read_csv(PATH, pairs, null_values=["NA"]), strings)
    var overrides = Dict[String, String]()
    overrides["label"] = "categorical"
    same(
        read_csv(PATH, schema_overrides=overrides, null_values=["NA"]), strings
    )
    assert_equal(read_csv(PATH).column("label").dtype(), DataType.STRING)


def test_empty_null_only_custom_quote_and_lazy() raises:
    write("id,label\n")
    same(read_csv(PATH, schema(True)), read_csv(PATH, schema(False)))
    assert_equal(
        len(
            read_csv(PATH, schema(True)).column("label").dtype().dictionary()[]
        ),
        0,
    )
    write("id,label\n1,\n2,\n")
    same(read_csv(PATH, schema(True)), read_csv(PATH, schema(False)))
    write("id,label\n1,'it''s quoted'\n2,''\n")
    same(
        read_csv(PATH, schema(True), quote_char="'"),
        read_csv(PATH, schema(False), quote_char="'"),
    )
    write('id,label\n1,pear\n2,""\n3,\n4,apple\n')
    var expected = read_csv(PATH, schema(False)).head(3)
    for optimize in [False, True]:
        same(
            scan_csv(PATH, schema(True)).head(3).collect(optimize=optimize),
            expected,
        )


def test_utf8_lossy_errors_and_projection() raises:
    var bytes = List[UInt8]()
    bytes.extend("id,label\n1,".as_bytes())
    bytes.append(255)
    bytes.extend("\n2,ok\n".as_bytes())
    with open(PATH, "w") as file:
        file.write_bytes(bytes)
    for ignore in [False, True]:
        with assert_raises(contains="not valid UTF-8"):
            _ = read_csv(
                PATH, schema(True), columns=["id"], ignore_errors=ignore
            )
    same(
        read_csv(PATH, schema(True), encoding="utf8-lossy"),
        read_csv(PATH, schema(False), encoding="utf8-lossy"),
    )


def test_parallel_dictionaries_limits_and_projection() raises:
    var text = String("id,label\n")
    for i in range(90000):
        var label = "a long repeating value " + String((i // 15000) * 7 + i % 5)
        text += String(i) + "," + ("" if i % 31 == 0 else label) + "\n"
    write(text)
    set_threads(1)
    var strings = read_csv(PATH, schema(False))
    var one = read_csv(PATH, schema(True))
    same(one, strings)
    set_threads(8)
    var many = read_csv(PATH, schema(True))
    same(many, strings)
    assert_true(
        one.column("label")
        .cast("string")
        .equals(many.column("label").cast("string"))
    )
    same(read_csv(PATH, schema(True), n_rows=45001), strings.head(45001))
    var projected = read_csv(
        PATH, schema(True), columns=["label"], n_rows=45001
    )
    assert_true(
        projected.column("label")
        .cast("string")
        .equals(strings.head(45001).column("label"))
    )
    set_threads(1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
