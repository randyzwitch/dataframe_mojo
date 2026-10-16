"""CSV conversion boundaries use literal text and independent exact values."""
# contract-group: io.csv
from std.os import getenv
from std.memory import bitcast
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    Column,
    CsvField,
    CsvSchema,
    DataFrame,
    DataType,
    Series,
    read_csv,
    to_csv_string,
    write_csv,
)
from tests.support.numeric_model import (
    INTEGER_DTYPES,
    FLOAT_DTYPES,
    integer_bounds,
)
from tests.support.coverage_compare import assert_float
from tests.support.csv_reference import csv_records, csv_float
from tests.support.coverage_generators import offset_layout, chunk_layout


def _path() -> String:
    return getenv("COVERAGE_TEST_TMP") + "/csv-types.csv"


def _write(text: String) raises:
    with open(_path(), "w") as file:
        file.write(text)


def _integers[D: DType]() raises:
    var bounds = integer_bounds[D]()
    var values: List[Int128] = [bounds[0], bounds[1], 0, 1]
    var text = String("v\n")
    var payload = List[Scalar[D]]()
    for value in values:
        text += String(value) + "\n"
        payload.append(value.cast[D]())
    text += '""\n'
    _write(text)
    var schema = CsvSchema([CsvField("v", DataType.of(D))])
    var actual = read_csv(_path(), schema)
    assert_equal(actual.height(), 5)
    assert_true(actual.column("v").dtype() == DataType.of(D))
    for row in range(4):
        assert_equal(
            actual.item(row, "v").numeric[D]().cast[DType.int128](), values[row]
        )
    assert_true(actual.item(4, "v").is_null())
    var source = DataFrame([Series("v", Column[Scalar[D]](payload^))])
    for layout in range(3):
        var frame = (
            source.copy() if layout
            == 0 else offset_layout(source) if layout
            == 1 else chunk_layout(source)
        )
        var records = csv_records(to_csv_string(frame))
        assert_equal(len(records), 5)
        for row in range(4):
            assert_equal(records[row + 1][0].text, String(values[row]))
    for invalid in [
        String(bounds[0] - 1),
        String(bounds[1] + 1),
        "1.0",
        "1e2",
        "9junk",
    ]:
        _write("v\n" + invalid + "\n7\n")
        with assert_raises():
            _ = read_csv(_path(), schema)
        var relaxed = read_csv(_path(), schema, ignore_errors=True)
        assert_equal(relaxed.height(), 2)
        assert_true(relaxed.item(0, "v").is_null())
        assert_equal(
            relaxed.item(1, "v").numeric[D]().cast[DType.int128](), Int128(7)
        )
    comptime if not D.is_signed():
        _write("v\n-0\n")
        with assert_raises():
            _ = read_csv(_path(), schema)


def test_csv_every_integer_width_checks_literal_limits_and_invalid_rows() raises:
    comptime for index in range(len(INTEGER_DTYPES)):
        _integers[INTEGER_DTYPES[index]]()


def _floats[D: DType]() raises:
    _write('v\n-0.0\nNaN\ninf\n-inf\n1.25\n""\n')
    var schema = CsvSchema([CsvField("v", DataType.of(D))])
    var actual = read_csv(_path(), schema)
    var values: List[Float64] = [
        -0.0,
        Float64("nan"),
        Float64("inf"),
        -Float64("inf"),
        1.25,
    ]
    assert_equal(actual.height(), 6)
    assert_true(actual.column("v").dtype() == DataType.of(D))
    for row in range(5):
        assert_float(Float64(actual.item(row, "v").numeric[D]()), values[row])
    assert_true(actual.item(5, "v").is_null())
    var records = csv_records(to_csv_string(actual, quote_style="always"))
    assert_equal(len(records), 7)
    for row in range(5):
        assert_float(csv_float(records[row + 1][0].text), values[row])
    assert_true(not records[6][0].quoted and not records[6][0].text)
    _write("v\nnot-a-float\n1.5\n")
    with assert_raises():
        _ = read_csv(_path(), schema)
    var relaxed = read_csv(_path(), schema, ignore_errors=True)
    assert_true(relaxed.item(0, "v").is_null())
    assert_float(Float64(relaxed.item(1, "v").numeric[D]()), Float64(1.5))


def test_csv_float_specials_signed_zero_and_writer_output_match_independent_consumer() raises:
    comptime for index in range(len(FLOAT_DTYPES)):
        _floats[FLOAT_DTYPES[index]]()


def test_csv_float32_decimal_rounding_uses_target_precision_and_nullable_is_metadata() raises:
    # Exact halfway point between 1 and its Float32 successor is
    # 1 + 2^-24. A positive decimal residual above it must round upward;
    # rounding first to Float64 erases that residual and picks the even 1.
    var texts: List[String] = [
        "1.000000059604644775390625",
        "1.000000059604644775390625000001",
        "1e-46",
        "1e-45",
        "3.40282346638528859811704183484516925440e38",
        "1e400",
        "-0",
    ]
    var expected: List[UInt32] = [
        0x3F800000,
        0x3F800001,
        0,
        1,
        0x7F7FFFFF,
        0x7F800000,
        0x80000000,
    ]
    var text = String("x\n")
    for value in texts:
        text += value + "\n"
    text += '"NULL"\n\n'
    _write(text)
    var schema = CsvSchema([CsvField("x", DataType.FLOAT32, nullable=False)])
    var actual = read_csv(_path(), schema, null_values=["NULL"])
    assert_equal(actual.height(), len(texts) + 2)
    assert_equal(actual.column("x").dtype(), DataType.FLOAT32)
    for row in range(len(texts)):
        assert_equal(
            bitcast[DType.uint32](actual.item(row, "x").float32()),
            expected[row],
        )
    for row in [len(texts), len(texts) + 1]:
        assert_true(actual.item(row, "x").is_null())
    _write("x\n1e400\n-1e400\n")
    var wide = read_csv(_path(), CsvSchema([CsvField.float64("x")]))
    assert_float(wide.item(0, "x").float64(), Float64("inf"))
    assert_float(wide.item(1, "x").float64(), -Float64("inf"))


def test_csv_temporal_fields_parse_iso_instants_and_custom_civil_formats() raises:
    _write(
        "date,time,stamp\n29/02/2024,23:59:59.999999999,1969-12-31T23:59:59.999999Z\n01/01/1970,00:00:00,1970-01-01T00:00:00.123456Z\n,,\n"
    )
    var schema = CsvSchema(
        [
            CsvField.date("d", "%d/%m/%Y"),
            CsvField.time("t"),
            CsvField.datetime("ts", "us", time_zone="UTC"),
        ]
    )
    var actual = read_csv(_path(), schema)
    assert_equal(actual.columns(), ["d", "t", "ts"])
    assert_equal(actual.height(), 3)
    assert_equal(
        actual.dtypes(),
        [DataType.DATE, DataType.TIME, DataType.datetime("us", "UTC")],
    )
    var days: List[Int64] = [19782, 0]
    var times: List[Int64] = [86399999999999, 0]
    var instants: List[Int64] = [-1, 123456]
    for row in range(2):
        assert_equal(actual.item(row, "d").to_physical(), days[row])
        assert_equal(actual.item(row, "t").to_physical(), times[row])
        assert_equal(actual.item(row, "ts").to_physical(), instants[row])
    for name in ["d", "t", "ts"]:
        assert_true(actual.item(2, name).is_null())
    var records = csv_records(to_csv_string(actual))
    assert_equal(records[1][0].text, "2024-02-29")
    assert_equal(records[1][1].text, "23:59:59.999999999")
    assert_equal(records[1][2].text, "1969-12-31 23:59:59.999999+00:00")
    assert_equal(records[2][2].text, "1970-01-01 00:00:00.123456+00:00")


def test_csv_categorical_values_keep_quoted_empty_null_tokens_and_unicode() raises:
    _write(
        'v,id\n"",0\n"é,quote""",1\n,2\nlong-external-value,3\n"",4\nNULL,5\n'
    )
    var schema = CsvSchema([CsvField.categorical("v"), CsvField.int64("id")])
    var actual = read_csv(_path(), schema, null_values=["NULL"])
    assert_equal(actual.height(), 6)
    assert_true(actual.column("v").dtype().is_categorical())
    var expected: List[String] = [
        "",
        'é,quote"',
        "",
        "long-external-value",
        "",
        "",
    ]
    for row in range(6):
        assert_equal(actual.item(row, "id").int64(), Int64(row))
        assert_equal(actual.item(row, "v").is_null(), row == 2 or row == 5)
        if row != 2 and row != 5:
            assert_equal(actual.item(row, "v").string(), expected[row])
    # Quoting a token does not protect a value from a configured null token.
    var colliding = read_csv(_path(), schema, null_values=["", "NULL"])
    for row in [0, 2, 4, 5]:
        assert_true(colliding.item(row, "v").is_null())


# expected-failure: Verified to_csv_string emits UInt32 dictionary codes while
# expected-failure: write_csv decodes categorical values; public same-text
# expected-failure: contract requires decoded values. Later layouts await
# expected-failure: first failure repair.
# expected-diagnostic: CSV writer must emit categorical values rather than
# expected-diagnostic: dictionary codes
def test_csv_writer_emits_categorical_values_instead_of_dictionary_codes() raises:
    var raw = Series(
        "v",
        Column[String](
            ["", 'é,quote"', "poison", "long-external-value", ""],
            [True, True, False, True, True],
        ),
    ).cast(DataType.CATEGORICAL)
    var expected: List[String] = ["", 'é,quote"', "", "long-external-value", ""]
    var source = DataFrame([raw^])
    for layout in range(3):
        var frame = (
            source.copy() if layout
            == 0 else offset_layout(source) if layout
            == 1 else chunk_layout(source)
        )
        var records = csv_records(to_csv_string(frame))
        assert_equal(len(records), 6)
        for row in range(5):
            assert_true(
                records[row + 1][0].text == expected[row],
                "CSV writer must emit categorical values rather than dictionary codes",
            )
            assert_equal(records[row + 1][0].quoted, row != 2 and row != 3)


def test_csv_file_writer_decodes_categorical_dictionaries_across_layouts_and_buffers() raises:
    var values: List[String] = [
        "",
        'é,quote"',
        "poison",
        "long-external-value",
        "",
    ]
    var valid: List[Bool] = [True, True, False, True, True]
    var source = DataFrame(
        [
            Series("v", Column[String](values.copy(), valid.copy())).cast(
                DataType.CATEGORICAL
            )
        ]
    )
    for layout in range(3):
        var frame = (
            source.copy() if layout
            == 0 else offset_layout(source) if layout
            == 1 else chunk_layout(source)
        )
        for buffer in [1, 7, 64]:
            write_csv(frame, _path(), buffer_size=buffer)
            var text: String
            with open(_path(), "r") as file:
                text = file.read()
            var records = csv_records(text)
            assert_equal(len(records), 6)
            for row in range(5):
                assert_equal(
                    records[row + 1][0].text, values[row] if valid[row] else ""
                )
                assert_equal(records[row + 1][0].quoted, row != 2 and row != 3)
            var actual = read_csv(
                _path(), CsvSchema([CsvField.categorical("v")])
            )
            for row in range(5):
                assert_equal(actual.item(row, "v").is_null(), not valid[row])
                if valid[row]:
                    assert_equal(actual.item(row, "v").string(), values[row])


def test_csv_unsupported_binary_input_and_writer_types_reject_even_empty_columns() raises:
    with assert_raises(contains="binary"):
        _ = CsvSchema([CsvField("v", DataType.BINARY)])
    for n in [0, 1]:
        var values = List[List[UInt8]]()
        if n:
            values.append([0, 255])
        var frame = DataFrame([Series.binary("v", values^)])
        with assert_raises(contains="binary"):
            _ = to_csv_string(frame)
