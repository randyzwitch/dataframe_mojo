"""Public Polars-derived CSV parsing and expression integration."""
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)

from dataframe import DataType, CsvField, CsvSchema, col, lit, read_csv

comptime CSV_PATH = "/tmp/dataframe_mojo_test.csv"


def _write(text: String) raises:
    with open(CSV_PATH, "w") as file:
        file.write(text)


def _write_bytes(bytes: List[UInt8]) raises:
    with open(CSV_PATH, "w") as file:
        file.write_bytes(bytes)


def _schema() raises -> CsvSchema:
    return CsvSchema(
        [
            CsvField.int64("id", False),
            CsvField.float64("value"),
            CsvField.bool("active"),
            CsvField.string("label"),
        ]
    )


def test_all_dtypes_nulls_quotes_unicode_and_expression_pipeline() raises:
    _write(
        "\ufeffid,value,active,label\r\n"
        '9007199254740993,1.5,true,"hello, 🔥"\r\n'
        '2,,false,""\r\n'
        '3,nan,true,"line one\r\nline two"\r\n'
        '4,-inf,false,"say ""hello"""'
    )
    var frame = read_csv(CSV_PATH, _schema(), buffer_size=1)
    assert_equal(frame.height(), 4)
    assert_equal(frame.width(), 4)
    assert_equal(frame.column("id").int64().value(0), Int64(9007199254740993))
    assert_almost_equal(frame.column("value").float64().value(0), 1.5)
    assert_true(frame.column("value").float64().is_null(1))
    var nan = frame.column("value").float64().value(2)
    assert_true(nan != nan)
    assert_true(frame.column("value").float64().value(3) < -1e300)
    assert_true(frame.column("active").bool().value(0))
    assert_false(frame.column("active").bool().value(1))
    assert_equal(frame.column("label").string().value(0), "hello, 🔥")
    assert_equal(frame.column("label").string().value(1), "")
    assert_equal(
        frame.column("label").string().value(2), "line one\r\nline two"
    )
    assert_equal(frame.column("label").string().value(3), 'say "hello"')

    var result = (
        frame.filter(col("active").eq(lit(True)))
        .with_columns((col("value") * lit(Float64(2))).alias("doubled"))
        .select_exprs([col("id"), col("doubled")])
    )
    assert_equal(result.height(), 2)
    assert_equal(result.column("id").int64().value(0), Int64(9007199254740993))
    assert_almost_equal(result.column("doubled").float64().value(0), 3.0)
    assert_true(
        result.column("doubled").float64().value(1)
        != result.column("doubled").float64().value(1)
    )


def test_buffer_size_compatibility_preserves_public_results() raises:
    var text = String(
        'id,value,active,label\n1,2.5,true,"a,b"\n2,3.5,false,"x\n""y"""\n'
    )
    _write(text)
    var expected = read_csv(CSV_PATH, _schema())
    # buffer_size remains accepted for API compatibility; range discovery is
    # now CountLines-driven rather than a separate streaming tokenizer.
    for size in [1, 64, 65, 65536]:
        var actual = read_csv(CSV_PATH, _schema(), buffer_size=size)
        assert_equal(actual.height(), expected.height())
        for row in range(actual.height()):
            assert_equal(
                actual.column("id").int64().value(row),
                expected.column("id").int64().value(row),
            )
            assert_equal(
                actual.column("label").string().value(row),
                expected.column("label").string().value(row),
            )


def test_headerless_empty_and_header_only_files() raises:
    _write("1,2.0,true,x\n2,3.0,false,y")
    var headerless = read_csv(CSV_PATH, _schema(), has_header=False)
    assert_equal(headerless.height(), 2)
    assert_equal(headerless.column("label").string().value(1), "y")

    _write("")
    var empty = read_csv(CSV_PATH, _schema())
    assert_equal(empty.height(), 0)
    assert_equal(empty.width(), 4)

    _write("id,value,active,label\n")
    var header_only = read_csv(CSV_PATH, _schema())
    assert_equal(header_only.height(), 0)
    assert_equal(header_only.schema()[0].dtype, DataType.INT64)


def test_quoted_empty_is_value_but_unquoted_empty_is_null() raises:
    var schema = CsvSchema([CsvField.string("s")])
    _write('s\n\n""\nvalue')
    var frame = read_csv(CSV_PATH, schema)
    assert_equal(frame.height(), 3)
    assert_true(frame.column("s").string().is_null(0))
    assert_false(frame.column("s").string().is_null(1))
    assert_equal(frame.column("s").string().value(1), "")
    assert_equal(frame.column("s").string().value(2), "value")


def test_schema_and_option_validation() raises:
    with assert_raises():
        _ = CsvSchema([])
    with assert_raises():
        _ = CsvSchema([CsvField.int64("x"), CsvField.string("x")])
    _write("id,value,active,label\n")
    with assert_raises():
        _ = read_csv(CSV_PATH, _schema(), buffer_size=0)


def test_conversion_errors_and_nullable_metadata() raises:
    var one_int = CsvSchema([CsvField.int64("x", False)])
    # CsvField.nullable describes schema metadata. Polars' parser emits null
    # for empty primitive fields regardless of that metadata flag.
    for empty in ["x\n\n", 'x\n""']:
        _write(empty)
        var parsed = read_csv(CSV_PATH, one_int)
        assert_true(parsed.column("x").int64().is_null(0))
    for bad in ["x\n9223372036854775808", "x\n1.5"]:
        _write(bad)
        with assert_raises():
            _ = read_csv(CSV_PATH, one_int)

    var one_float = CsvSchema([CsvField.float64("x")])
    _write("x\n1e9999")
    # Polars' float parser accepts exponent overflow as IEEE positive infinity.
    var overflow = read_csv(CSV_PATH, one_float)
    assert_true(overflow.column("x").float64().value(0) > 1e300)

    var one_bool = CsvSchema([CsvField.bool("x")])
    for valid in ["x\nTrue", "x\nTRUE", "x\nfAlSe"]:
        _write(valid)
        _ = read_csv(CSV_PATH, one_bool)
    for bad in ["x\n1", "x\nyes"]:
        _write(bad)
        with assert_raises():
            _ = read_csv(CSV_PATH, one_bool)


def test_conversion_errors_keep_record_and_field_order_across_decode_batches() raises:
    var schema = CsvSchema([CsvField.int64("a"), CsvField.int64("b")])
    var source = String("a,b\n")
    for row in range(300):
        if row == 254:
            source += "1,bad\n"
        elif row == 255:
            source += "bad,2\n"
        else:
            source += "1,2\n"
    _write(source)
    try:
        _ = read_csv(CSV_PATH, schema)
        assert_true(False, "expected the first conversion error")
    except error:
        assert_true(String(error).find("field 'b'") >= 0)


def test_malformed_csv_and_positional_explicit_schema() raises:
    var schema = CsvSchema([CsvField.string("a"), CsvField.string("b")])
    for bad in [
        "a,b\n1,2,3",
        'a,b\n"unterminated,2',
        'a,b\n"x"tail,2',
        "a,b\n1,2\r3,4",
    ]:
        _write(bad)
        with assert_raises():
            _ = read_csv(CSV_PATH, schema, buffer_size=1)

    # Missing trailing fields null-fill; an interior quote is raw data unless
    # the field started quoted. Both cases follow Polars splitfields rules.
    _write('a,b\n1\n1"quote,2\n')
    var short = read_csv(CSV_PATH, schema)
    assert_true(short.column("b").string().is_null(0))
    assert_equal(short.column("a").string().value(1), '1"quote')

    # Explicit schema names label positions and do not validate header names.
    _write("b,a\n1,2")
    var positional = read_csv(CSV_PATH, schema)
    assert_equal(positional.column("a").string().value(0), "1")


def test_invalid_utf8_is_rejected_across_buffer_boundaries() raises:
    var bytes: List[UInt8] = [115, 10, 240, 40, 140, 40]
    _write_bytes(bytes^)
    var schema = CsvSchema([CsvField.string("s")])
    for size in range(1, 8):
        with assert_raises():
            _ = read_csv(CSV_PATH, schema, buffer_size=size)


def test_bom_is_only_recognized_at_start() raises:
    var schema = CsvSchema([CsvField.string("s")])
    _write("\ufeffs\n\ufeffvalue")
    var frame = read_csv(CSV_PATH, schema, buffer_size=1)
    assert_equal(frame.column("s").string().value(0), "\ufeffvalue")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
