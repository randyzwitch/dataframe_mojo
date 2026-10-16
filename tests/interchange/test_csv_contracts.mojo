"""CSV interoperability uses independent Mojo producers and consumers."""
# contract-group: io.csv
from std.collections import Dict
from std.math import isnan
from std.os import getenv
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    Column,
    CsvField,
    CsvSchema,
    DataFrame,
    DataType,
    Series,
    read_csv,
    write_csv,
    to_csv_string,
)
from tests.support.csv_reference import (
    csv_records,
    csv_produce_model,
    csv_float,
)
from tests.support.reference_model import ReferenceTable, generated_model
from tests.support.coverage_generators import frame_from_model, chunk_layout
from tests.support.coverage_compare import assert_model, assert_float


def _path(name: String) -> String:
    return getenv("COVERAGE_TEST_TMP") + "/csv-" + name + ".csv"


def _write(path: String, text: String) raises:
    with open(path, "w") as file:
        file.write(text)


def _schema() raises -> CsvSchema:
    return CsvSchema(
        [
            CsvField.int64("i"),
            CsvField.float64("f"),
            CsvField.string("s"),
            CsvField.bool("b"),
        ]
    )


def _consume(
    text: String,
    model: ReferenceTable,
    separator: UInt8,
    header: Bool,
    style: String,
) raises:
    var records = csv_records(text, separator)
    assert_equal(len(records), len(model.rows) + Int(header))
    if header:
        assert_equal(len(records[0]), 4)
        for column in range(4):
            assert_equal(records[0][column].text, model.names[column])
            assert_equal(
                records[0][column].quoted, style in ["always", "non_numeric"]
            )
    for row in range(len(model.rows)):
        ref fields = records[row + Int(header)]
        ref expected = model.rows[row]
        assert_equal(len(fields), 4)
        var validity: List[Bool] = [
            expected.iv,
            expected.fv,
            expected.sv,
            expected.bv,
        ]
        for column in range(4):
            var is_null = not fields[column].quoted and not fields[column].text
            assert_equal(not is_null, validity[column])
            if validity[column]:
                var quote = (
                    style == "always" or style == "non_numeric" and column >= 2
                )
                if style == "necessary" and column == 2:
                    quote = not expected.s
                    for byte in expected.s.as_bytes():
                        quote = (
                            quote
                            or byte == separator
                            or byte == 34
                            or byte == 10
                            or byte == 13
                        )
                assert_equal(fields[column].quoted, quote)
        if expected.iv:
            assert_equal(Int64(Int(fields[0].text)), expected.i)
        if expected.fv:
            assert_float(csv_float(fields[1].text), expected.f)
        if expected.sv:
            assert_equal(fields[2].text, expected.s)
        if expected.bv:
            assert_equal(fields[3].text, "true" if expected.b else "false")


def test_csv_reference_has_literal_quote_record_null_and_numeric_anchors() raises:
    var records = csv_records('a,"",,"line\n""quote"""\r\n"x,y",last,,')
    assert_equal(len(records), 2)
    assert_equal(len(records[0]), 4)
    assert_equal(records[0][1].text, "")
    assert_true(records[0][1].quoted)
    assert_true(not records[0][2].quoted)
    assert_equal(records[0][3].text, 'line\n"quote"')
    assert_equal(records[1][0].text, "x,y")
    assert_equal(len(records[1]), 4)
    assert_equal(len(csv_records("")), 0)
    assert_equal(csv_float("1.25"), Float64(1.25))
    assert_float(csv_float("-0.0"), Float64(-0.0))
    assert_true(isnan(csv_float("NaN")))
    with assert_raises():
        _ = csv_float("1junk")
    with assert_raises():
        _ = csv_records('"unterminated')
    with assert_raises():
        _ = csv_records('"closed"junk')


def test_csv_reader_accepts_independent_records_across_delimiters_endings_and_nulls() raises:
    var path = _path("independent")
    for n in [0, 1, 9, 65]:
        for mode in range(3):
            var model = generated_model(n, 3, mode)
            for separator in [",", "|", "\t"]:
                for newline in ["\n", "\r\n"]:
                    for header in [False, True]:
                        var text = csv_produce_model(
                            model, separator, newline, header
                        )
                        for final_newline in [False, True]:
                            var source = text.copy()
                            if not final_newline and source:
                                var trimmed = String(
                                    source[
                                        byte = 0 : source.byte_length()
                                        - newline.byte_length()
                                    ]
                                )
                                source = trimmed^
                            # Bare trailing nulls at EOF have their own focused
                            # contract below, so a failure there cannot stop the
                            # ordinary record/quoting matrix.
                            if not final_newline and source.endswith(separator):
                                continue
                            _write(path, source)
                            for buffer in [1, 7, 64]:
                                try:
                                    assert_model(
                                        read_csv(
                                            path,
                                            _schema(),
                                            has_header=header,
                                            separator=separator,
                                            buffer_size=buffer,
                                        ),
                                        model,
                                    )
                                except error:
                                    print(
                                        "CSV fixture:",
                                        n,
                                        mode,
                                        separator,
                                        newline.byte_length(),
                                        header,
                                        final_newline,
                                        buffer,
                                    )
                                    raise error


# expected-failure: Verified independently authored final record ending in an
# expected-failure: unquoted empty field without a newline raises Explicit
# expected-failure: height does not match columns; valid CSV must preserve the
# expected-failure: final field and row. Focused case stops at first affected
# expected-failure: fixture; later dimensions await repair.
# expected-diagnostic: unterminated trailing null CSV field must preserve
# expected-diagnostic: record width
def test_csv_unterminated_trailing_null_field_preserves_record_width_and_height() raises:
    var path = _path("trailing-null")
    for n in [1, 9, 65]:
        for mode in [1, 2]:
            var model = generated_model(n, 3, mode)
            for header in [False, True]:
                for separator in [",", "|", "\t"]:
                    var text = csv_produce_model(model, separator, "\n", header)
                    var trimmed = String(
                        text[byte = 0 : text.byte_length() - 1]
                    )
                    if not trimmed.endswith(separator):
                        continue
                    _write(path, trimmed)
                    try:
                        assert_model(
                            read_csv(
                                path,
                                _schema(),
                                has_header=header,
                                separator=separator,
                            ),
                            model,
                        )
                    except error:
                        if (
                            String(error).find(
                                "Explicit height does not match columns"
                            )
                            < 0
                        ):
                            raise error
                        assert_true(
                            False,
                            "unterminated trailing null CSV field must preserve record width",
                        )


def test_csv_writer_outputs_are_consumed_by_independent_field_parser() raises:
    var path = _path("writer")
    for n in [0, 1, 9, 65]:
        for mode in range(3):
            var model = generated_model(n, 3, mode)
            for layout in range(3):
                var frame = frame_from_model(model, 7 if layout == 1 else 0)
                if layout == 2:
                    frame = chunk_layout(frame)
                for style in ["necessary", "always", "non_numeric"]:
                    for separator in [",", "|"]:
                        for newline in ["\n", "\r\n"]:
                            for header in [False, True]:
                                var text = to_csv_string(
                                    frame,
                                    has_header=header,
                                    separator=separator,
                                    quote_style=style,
                                    line_terminator=newline,
                                )
                                _consume(
                                    text,
                                    model,
                                    separator.as_bytes()[0],
                                    header,
                                    style,
                                )
                                write_csv(
                                    frame,
                                    path,
                                    has_header=header,
                                    separator=separator,
                                    quote_style=style,
                                    line_terminator=newline,
                                    buffer_size=7,
                                )
                                with open(path, "r") as file:
                                    assert_equal(file.read(), text)
    # Raw quoting is deliberately observable, even when the text is not valid
    # input for the independent well-formed CSV consumer.
    var raw = DataFrame([Series("s", Column[String](['a"b', "x,y", ""]))])
    var text = to_csv_string(raw, quote_style="never", has_header=False)
    assert_equal(text, 'a"b\nx,y\n\n')


def test_csv_writer_null_token_collisions_have_literal_quoted_output() raises:
    var source = DataFrame(
        [Series("s", Column[String](["NA", "", "poison"], [True, True, False]))]
    )
    var text = to_csv_string(source, has_header=False, null_value="NA")
    assert_equal(text, '"NA"\n""\nNA\n')
    var path = _path("null-token-collision")
    _write(path, text)
    var actual = read_csv(
        path,
        CsvSchema([CsvField.string("s")]),
        has_header=False,
        null_values=["NA"],
    )
    assert_equal(actual.height(), 3)
    assert_true(actual.item(0, "s").is_null())
    assert_equal(actual.item(1, "s").string(), "")
    assert_true(actual.item(2, "s").is_null())


def test_csv_schema_labels_projection_limits_skips_and_disabled_quotes() raises:
    var path = _path("options")
    _write(
        path,
        '"skip\nrecord",0\noriginal_a,original_b\n1,"a,b"\n2,""\n3,\n4,last',
    )
    var schema = CsvSchema([CsvField.int64("x"), CsvField.string("s")])
    for limit in [0, 1, 3, 9]:
        var actual = read_csv(path, schema, skip_rows=1, n_rows=limit)
        assert_equal(actual.columns(), ["x", "s"])
        assert_equal(actual.height(), min(limit, 4))
        for row in range(actual.height()):
            assert_equal(actual.item(row, "x").int64(), Int64(row + 1))
            assert_equal(actual.item(row, "s").is_null(), row == 2)
            if row != 2:
                assert_equal(
                    actual.item(row, "s").string(),
                    "a,b" if row == 0 else "" if row == 1 else "last",
                )
    var projected = read_csv(path, schema, skip_rows=1, columns=["s"])
    assert_equal(projected.columns(), ["s"])
    assert_equal(projected.height(), 4)
    _write(path, '\ufeff# comment\nx,s\n1,"literal"\n# another\n2,end\n')
    var plain = read_csv(path, schema, comment_prefix="#", quote_char="")
    assert_equal(plain.height(), 2)
    assert_equal(plain.item(0, "s").string(), '"literal"')
    assert_equal(plain.item(1, "s").string(), "end")
    _write(path, "")
    assert_equal(read_csv(path, schema).height(), 0)
    _write(path, "x,s\n")
    assert_equal(
        read_csv(path, schema).dtypes(), [DataType.INT64, DataType.STRING]
    )


def test_csv_null_tokens_padding_and_conversion_errors_preserve_other_fields() raises:
    var path = _path("nulls-errors")
    _write(path, 'i,f,s,b\n  +7,\t1.5,"",TrUe\n,,"NULL",FALSE\n,,NULL,\n8\n')
    var actual = read_csv(path, _schema(), null_values=["NULL"])
    assert_equal(actual.height(), 4)
    assert_equal(actual.item(0, "i").int64(), Int64(7))
    assert_float(actual.item(0, "f").float64(), Float64(1.5))
    assert_equal(actual.item(0, "s").string(), "")
    assert_equal(actual.item(0, "b").bool(), True)
    for row in [1, 2]:
        assert_true(actual.item(row, "i").is_null())
        assert_true(actual.item(row, "f").is_null())
        assert_true(actual.item(row, "s").is_null())
    assert_equal(actual.item(1, "b").bool(), False)
    assert_true(actual.item(2, "b").is_null())
    assert_equal(actual.item(3, "i").int64(), Int64(8))
    for name in ["f", "s", "b"]:
        assert_true(actual.item(3, name).is_null())
    _write(path, "i,f,s,b\nno,1.5,keep,true\n8,bad,also,wrong\n")
    with assert_raises():
        _ = read_csv(path, _schema())
    var relaxed = read_csv(path, _schema(), ignore_errors=True)
    assert_equal(relaxed.height(), 2)
    assert_true(relaxed.item(0, "i").is_null())
    assert_float(relaxed.item(0, "f").float64(), Float64(1.5))
    assert_equal(relaxed.item(0, "s").string(), "keep")
    assert_equal(relaxed.item(0, "b").bool(), True)
    assert_equal(relaxed.item(1, "i").int64(), Int64(8))
    assert_true(relaxed.item(1, "f").is_null())
    assert_equal(relaxed.item(1, "s").string(), "also")
    assert_true(relaxed.item(1, "b").is_null())


def test_csv_inference_sampling_overrides_and_temporal_opt_in_follow_literal_types() raises:
    var path = _path("inference")
    _write(path, 'a,b,c,d,e\ntrue,001,1.5,,hello\nfalse,-3,2,,""\n')
    var inferred = read_csv(path)
    assert_equal(inferred.columns(), ["a", "b", "c", "d", "e"])
    assert_equal(
        inferred.dtypes(),
        [
            DataType.BOOL,
            DataType.INT64,
            DataType.FLOAT64,
            DataType.STRING,
            DataType.STRING,
        ],
    )
    assert_equal(inferred.item(0, "b").int64(), Int64(1))
    assert_float(inferred.item(1, "c").float64(), Float64(2))
    assert_true(inferred.item(0, "d").is_null())
    assert_equal(inferred.item(1, "e").string(), "")
    var overrides = Dict[String, String]()
    overrides["b"] = "float32"
    overrides["e"] = "categorical"
    overrides["unknown"] = "invalid-ignored"
    var explicit = read_csv(path, schema_overrides=overrides)
    assert_true(explicit.column("e").dtype().is_categorical())
    assert_equal(explicit.column("b").dtype(), DataType.FLOAT32)
    assert_equal(explicit.item(0, "b").float32(), Float32(1))
    _write(path, "x\n1\nword\n")
    with assert_raises():
        _ = read_csv(path, infer_schema_length=1)
    var whole = read_csv(path, infer_schema_length=-1)
    assert_equal(whole.column("x").dtype(), DataType.STRING)
    assert_equal(whole.item(0, "x").string(), "1")
    assert_equal(whole.item(1, "x").string(), "word")
    _write(path, "day\n2024-02-29\n1970-01-01\n")
    assert_equal(read_csv(path).column("day").dtype(), DataType.STRING)
    var dates = read_csv(path, try_parse_dates=True)
    assert_equal(dates.column("day").dtype(), DataType.DATE)
    assert_equal(dates.item(0, "day").to_physical(), Int64(19782))
    assert_equal(dates.item(1, "day").to_physical(), Int64(0))
    _write(path, "")
    with assert_raises():
        _ = read_csv(path)


def test_csv_utf8_validation_includes_unprojected_text_and_lossy_replacement() raises:
    var path = _path("encoding")
    var bytes: List[UInt8] = [120, 44, 115, 10, 49, 44, 255, 10]
    _write(path, String(unsafe_from_utf8=bytes^))
    var schema = CsvSchema([CsvField.int64("x"), CsvField.string("s")])
    with assert_raises():
        _ = read_csv(path, schema)
    with assert_raises():
        _ = read_csv(path, schema, columns=["x"])
    var lossy = read_csv(path, schema, encoding="utf8-lossy")
    assert_equal(lossy.item(0, "x").int64(), Int64(1))
    assert_equal(lossy.item(0, "s").string(), "�")


def test_csv_default_inference_boundary_and_duplicate_header_names_have_literal_results() raises:
    var path = _path("inference-boundary")
    for numeric_rows in [99, 100, 101]:
        var text = String("x\n")
        for row in range(numeric_rows):
            text += String(row) + "\n"
        text += "word\n"
        _write(path, text)
        if numeric_rows < 100:
            var actual = read_csv(path)
            assert_equal(actual.column("x").dtype(), DataType.STRING)
            assert_equal(actual.item(numeric_rows, "x").string(), "word")
        else:
            with assert_raises():
                _ = read_csv(path)
        var whole = read_csv(path, infer_schema_length=-1)
        assert_equal(whole.height(), numeric_rows + 1)
        assert_equal(whole.column("x").dtype(), DataType.STRING)
        for row in range(numeric_rows):
            assert_equal(whole.item(row, "x").string(), String(row))
        assert_equal(whole.item(numeric_rows, "x").string(), "word")
    _write(path, "a,a,a\n1,2,3\n")
    var named = read_csv(path)
    assert_equal(named.columns(), ["a", "a_duplicated_0", "a_duplicated_1"])
    for index in range(3):
        assert_equal(
            named.item(0, named.columns()[index]).int64(), Int64(index + 1)
        )
    var anonymous = read_csv(path, has_header=False)
    assert_equal(anonymous.columns(), ["column_1", "column_2", "column_3"])
    assert_equal(anonymous.height(), 2)
    var overrides = Dict[String, String]()
    overrides["a"] = "not-a-dtype"
    with assert_raises():
        _ = read_csv(path, schema_overrides=overrides)


def test_csv_custom_quote_and_writer_invalid_options_follow_explicit_grammar() raises:
    var path = _path("custom-quote")
    _write(path, "x|s\n1|'a|b'\n2|'it''s\nmultiline'\n3|''\n4|\n")
    var schema = CsvSchema([CsvField.int64("x"), CsvField.string("s")])
    var actual = read_csv(path, schema, separator="|", quote_char="'")
    assert_equal(actual.height(), 4)
    for row in range(4):
        assert_equal(actual.item(row, "x").int64(), Int64(row + 1))
    assert_equal(actual.item(0, "s").string(), "a|b")
    assert_equal(actual.item(1, "s").string(), "it's\nmultiline")
    assert_equal(actual.item(2, "s").string(), "")
    assert_true(actual.item(3, "s").is_null())
    for separator in ["", "::", "\r", "\n"]:
        with assert_raises():
            _ = to_csv_string(actual, separator=separator)
    for style in ["", "sometimes"]:
        with assert_raises():
            _ = to_csv_string(actual, quote_style=style)
    for ending in ["", "\r", "end"]:
        with assert_raises():
            _ = to_csv_string(actual, line_terminator=ending)
    with assert_raises():
        write_csv(actual, path, buffer_size=0)


def test_csv_invalid_options_schema_and_extra_fields_fail_without_changing_files() raises:
    var path = _path("invalid")
    _write(path, "i,f,s,b\n1,2,text,true,extra\n")
    with assert_raises():
        _ = read_csv(path, _schema())
    var truncated = read_csv(path, _schema(), truncate_ragged_lines=True)
    assert_equal(truncated.height(), 1)
    assert_equal(truncated.item(0, "s").string(), "text")
    assert_equal(
        read_csv(path, _schema(), columns=["i"]).item(0, "i").int64(), Int64(1)
    )
    for separator in ["", "::", "\n", "\r"]:
        with assert_raises():
            _ = read_csv(path, _schema(), separator=separator)
    for quote in ["xx", ",", "\n"]:
        with assert_raises():
            _ = read_csv(path, _schema(), quote_char=quote)
    with assert_raises():
        _ = read_csv(path, _schema(), skip_rows=-1)
    with assert_raises():
        _ = read_csv(path, _schema(), n_rows=-2)
    with assert_raises():
        _ = read_csv(path, _schema(), columns=["i", "i"])
    with assert_raises():
        _ = read_csv(path, _schema(), columns=["unknown"])
    with assert_raises():
        _ = read_csv(path, _schema(), encoding="ascii")
    with assert_raises():
        _ = read_csv(path, _schema(), buffer_size=0)
    with assert_raises():
        _ = CsvSchema(List[CsvField]())
    with assert_raises():
        _ = CsvSchema([CsvField.int64("x"), CsvField.string("x")])
    with open(path, "r") as file:
        assert_equal(file.read(), "i,f,s,b\n1,2,text,true,extra\n")
