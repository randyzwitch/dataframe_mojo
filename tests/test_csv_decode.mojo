"""Differential coverage for the clean chunk decoder.

These records have no header because ``decode_chunk`` is the post-prelude
primitive.  Shared cases compare its typed result with the legacy public
reader; projection-tail cases pin Polars' `skip_this_line` contract.
"""
from std.testing import TestSuite, assert_equal, assert_raises, assert_true
from dataframe.csv import CsvField, CsvOptions, CsvSchema, read_csv
from dataframe.csv_decode import decode_chunk
from dataframe.frame import DataFrame


comptime PATH = "/tmp/dataframe_mojo_csv_decode_test.csv"


def typed_schema() raises -> CsvSchema:
    return CsvSchema(
        [
            CsvField.int64("id", False),
            CsvField.float64("score"),
            CsvField.bool("active"),
            CsvField.string("label"),
        ]
    )


def three_schema() raises -> CsvSchema:
    return CsvSchema(
        [
            CsvField.int64("id"),
            CsvField.string("name"),
            CsvField.float64("score"),
        ]
    )


def options(
    comment_prefix: String = "",
    skip_rows: Int = 0,
    n_rows: Int = -1,
    null_values: List[String] = List[String](),
    ignore_errors: Bool = False,
    truncate_ragged_lines: Bool = False,
    encoding: String = "utf8",
) -> CsvOptions:
    return CsvOptions(
        ",",
        '"',
        comment_prefix,
        skip_rows,
        n_rows,
        null_values.copy(),
        ignore_errors,
        truncate_ragged_lines,
        encoding,
    )


def decode(
    text: String,
    schema: CsvSchema,
    options: CsvOptions,
    keep: List[Bool],
    rows: Int = 32,
) raises -> DataFrame:
    var owned = List[UInt8]()
    owned.extend(text.as_bytes())
    var input = Span[UInt8, ImmutAnyOrigin](
        unsafe_ptr=owned.unsafe_ptr()
        .unsafe_mut_cast[False]()
        .unsafe_origin_cast[ImmutAnyOrigin](),
        length=len(owned),
    )
    var result = decode_chunk(input, schema, options, keep, rows)
    _ = owned^
    return result^


def legacy(text: String, schema: CsvSchema) raises -> DataFrame:
    with open(PATH, "w") as file:
        file.write(text)
    return read_csv(PATH, schema, has_header=False, buffer_size=1)


def test_typed_borrowed_and_escaped_fields_match_legacy() raises:
    var text = String(
        '1,2.5,true,"hello, 🔥"\r\n'
        '2,,false,""\r\n'
        '3,nan,true,"line one\r\nline two"\r\n'
        '4,-inf,false,"say ""hello"""\n'
    )
    var actual = decode(
        text, typed_schema(), options(), [True, True, True, True]
    )
    var expected = legacy(text, typed_schema())
    assert_true(actual.equals(expected))


def test_comments_and_nulls_match_legacy_after_prelude() raises:
    var schema = three_schema()
    var text = "# generated\n1,NA,1\n#more\n2,ok,2\n3,last,3\n"
    var actual = decode(
        text,
        schema,
        options("#", 0, -1, ["NA"]),
        [True, True, True],
    )
    with open(PATH, "w") as file:
        file.write(text)
    var expected = read_csv(
        PATH,
        schema,
        has_header=False,
        buffer_size=1,
        comment_prefix="#",
        null_values=["NA"],
    )
    assert_true(actual.equals(expected))


def test_ignore_errors_is_polars_null_fill_not_legacy_row_drop() raises:
    var schema = three_schema()
    var bad = "1,a,1\nbad,b,2\n3,c\n5,e,5\n"
    var actual = decode(
        bad, schema, options(ignore_errors=True), [True, True, True]
    )
    # Polars PrimitiveChunkedBuilder::parse_bytes appends null on an invalid
    # primitive. The legacy Mojo reader drops that whole record, so this is a
    # deliberate direct-port behavior change.
    assert_equal(actual.height(), 4)
    assert_true(actual.column("id").int64().is_null(1))
    assert_true(actual.column("score").float64().is_null(2))


def test_ragged_truncation_matches_legacy() raises:
    var schema = three_schema()
    var ragged = "1,a,1\n3,c\n4,d,4,extra\n"
    var actual = decode(
        ragged, schema, options(truncate_ragged_lines=True), [True, True, True]
    )
    with open(PATH, "w") as file:
        file.write(ragged)
    var expected = read_csv(
        PATH,
        schema,
        has_header=False,
        buffer_size=1,
        truncate_ragged_lines=True,
    )
    assert_true(actual.equals(expected))


def test_projection_tail_has_polars_quote_aware_skip_semantics() raises:
    var text = '1,"not-used\nstill",x,extra\n2,ok,2\n3,"unterminated'
    var actual = decode(text, three_schema(), options(), [True, False, False])
    # Polars parse_lines calls skip_this_line once it has the final requested
    # field, so malformed and ragged omitted tails never become decode errors.
    assert_equal(actual.height(), 3)
    assert_equal(actual.width(), 1)
    assert_equal(actual.column("id").int64().value(0), Int64(1))
    assert_equal(actual.column("id").int64().value(1), Int64(2))
    assert_equal(actual.column("id").int64().value(2), Int64(3))


def test_strict_utf8_once_per_chunk_uses_full_source_schema() raises:
    var bytes: List[UInt8] = [49, 44, 255, 10]
    var input = Span[UInt8, ImmutAnyOrigin](
        unsafe_ptr=bytes.unsafe_ptr()
        .unsafe_mut_cast[False]()
        .unsafe_origin_cast[ImmutAnyOrigin](),
        length=len(bytes),
    )
    var schema = CsvSchema([CsvField.int64("id"), CsvField.string("name")])
    with assert_raises(contains="not valid UTF-8"):
        _ = decode_chunk(input, schema, options(), [True, True], 1)
    # Polars read_impl computes check_utf8 from the full source schema before
    # projection, so omitting the String output does not bypass this check.
    with assert_raises(contains="not valid UTF-8"):
        _ = decode_chunk(input, schema, options(), [True, False], 1)
    _ = bytes^


def test_direct_splitfield_grammar_matches_polars_builder_rules() raises:
    var schema = CsvSchema([CsvField.string("a"), CsvField.string("b")])
    # SplitFields only enters quote parity for a quote at field start. The
    # parser therefore hands bare CR and interior quotes to the Utf8 builder.
    var plain = decode('1"quote,2\r3\n', schema, options(), [True, True])
    assert_equal(plain.column("a").string().value(0), '1"quote')
    assert_equal(plain.column("b").string().value(0), "2\r3")
    # Utf8Field itself verifies a quoted field's closing quote before
    # escape_field, exactly as builder.rs does.
    with assert_raises():
        _ = decode('"x"tail,2\n', schema, options(), [True, True])


def test_chunk_decoder_does_not_apply_global_n_rows() raises:
    var actual = decode(
        "1,a,1\n2,b,2\n3,c,3\n",
        three_schema(),
        options(n_rows=1),
        [True, True, True],
    )
    # read_impl limits the concatenated frame. Enforcing this per chunk would
    # make parallel decode depend on chunk boundaries.
    assert_equal(actual.height(), 3)


def test_primitive_whitespace_follows_polars_builder() raises:
    var schema = CsvSchema([CsvField.int64("number")])
    var actual = decode('" \t12"\n', schema, options(), [True])
    assert_equal(actual.column("number").int64().value(0), 12)
    with assert_raises():
        _ = decode('"\n12"\n', schema, options(), [True])
    with assert_raises():
        _ = decode('"\r12"\n', schema, options(), [True])
    with assert_raises():
        _ = decode('"12 "\n', schema, options(), [True])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
