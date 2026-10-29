"""CSV memory sources share parsing rules and own all returned values."""
# contract-group: io.csv_buffer
from std.collections import Dict
from std.testing import assert_true
from dataframe import (
    Column,
    CsvBuffer,
    CsvField,
    CsvSchema,
    DataFrame,
    DataType,
    Series,
    StringColumn,
    read_csv,
)
from tests.support.coverage_compare import assert_frame


def test_text_buffer_inference_preserves_bom_quotes_utf8_comments_and_nulls() raises:
    var source = CsvBuffer(
        "\ufeff# before header\nid;text\n1;'α,\nβ'\n# between rows\n2;NA\n3;''\n"
    )
    var result = read_csv(
        source,
        separator=";",
        quote_char="'",
        comment_prefix="#",
        null_values=["NA"],
        infer_schema_length=-1,
    )
    var expected = DataFrame(
        [
            Series("id", Column[Int64]([1, 2, 3])),
            Series(
                "text", StringColumn(["α,\nβ", "", ""], [True, False, True])
            ),
        ]
    )
    assert_frame(result, expected)
    # The same owner can be decoded repeatedly without consuming it.
    assert_frame(
        read_csv(
            source,
            separator=";",
            quote_char="'",
            comment_prefix="#",
            null_values=["NA"],
        ),
        expected,
    )


def test_buffer_explicit_schema_projection_skips_and_row_limit_match_literal_cells() raises:
    var schema = CsvSchema(
        [CsvField.int64("id"), CsvField.float64("amount"), CsvField.string("s")]
    )
    var result = read_csv(
        CsvBuffer("skip me\n1|10.5|keep\n2|NA|other\n3|30.5|last\n"),
        schema,
        has_header=False,
        skip_rows=1,
        separator="|",
        null_values=["NA"],
        columns=["id", "amount"],
        n_rows=2,
    )
    assert_frame(
        result,
        DataFrame(
            [
                Series("id", Column[Int64]([1, 2])),
                Series("amount", Column[Float64]([10.5, 0], [True, False])),
            ]
        ),
    )
    var pairs: List[Tuple[String, DataType]] = [("x", DataType.INT16)]
    assert_frame(
        read_csv(CsvBuffer("x\n1\n2\n"), pairs),
        DataFrame([Series("x", Column[Int16]([1, 2]))]),
    )


def _local_buffer_result() raises -> DataFrame:
    return read_csv(CsvBuffer("id,text\n1,abcdefghijklmnopqrstuvwxyz\n"))


def test_byte_lists_and_destroyed_local_buffers_leave_owned_output_columns() raises:
    var text = (
        "id,text\n1,abcdefghijklmnopqrstuvwxyz\n2,another long string value\n"
    )
    var bytes = List[UInt8]()
    for byte in text.as_bytes():
        bytes.append(byte)
    var result = read_csv(bytes)
    for k in range(len(bytes)):
        bytes[k] = 255
    assert_frame(
        result,
        DataFrame(
            [
                Series("id", Column[Int64]([1, 2])),
                Series(
                    "text",
                    StringColumn(
                        [
                            "abcdefghijklmnopqrstuvwxyz",
                            "another long string value",
                        ]
                    ),
                ),
            ]
        ),
    )
    assert_frame(
        _local_buffer_result(),
        DataFrame(
            [
                Series("id", Column[Int64]([1])),
                Series("text", StringColumn(["abcdefghijklmnopqrstuvwxyz"])),
            ]
        ),
    )


def test_buffer_inference_overrides_and_strict_errors_use_reader_rules() raises:
    var overrides = Dict[String, String]()
    overrides["x"] = "int32"
    assert_frame(
        read_csv(CsvBuffer("x\n1\n2\n"), schema_overrides=overrides),
        DataFrame([Series("x", Column[Int32]([1, 2]))]),
    )
    var schema = CsvSchema([CsvField.int64("x")])
    var failed = False
    try:
        _ = read_csv(CsvBuffer("x\nbad\n"), schema)
    except:
        failed = True
    assert_true(failed)
    assert_frame(
        read_csv(CsvBuffer("x\nbad\n"), schema, ignore_errors=True),
        DataFrame([Series("x", Column[Int64]([0], [False]))]),
    )
    failed = False
    try:
        _ = read_csv(CsvBuffer("x\n1\n"), separator="||")
    except:
        failed = True
    assert_true(failed)


def test_buffer_bytes_keep_encoding_rules_and_empty_explicit_schemas() raises:
    var bytes: List[UInt8] = [120, 10, 255, 10]
    var schema = CsvSchema([CsvField.string("x")])
    var failed = False
    try:
        _ = read_csv(bytes, schema)
    except:
        failed = True
    assert_true(failed)
    assert_frame(
        read_csv(bytes, schema, encoding="utf8-lossy"),
        DataFrame([Series("x", StringColumn(["�"]))]),
    )
    assert_frame(
        read_csv(CsvBuffer("x\n"), CsvSchema([CsvField.int64("x")])),
        DataFrame([Series("x", Column[Int64](List[Int64]()))]),
    )
