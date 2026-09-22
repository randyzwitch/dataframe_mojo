"""Differential tests for plain and simple-quoted CSV borrowing.

The one-byte read is the scalar reference. A finite, oversized row limit
keeps every read on the serial streaming path, including the large buffers
that exercise `_feed_borrowed`.
"""
from std.testing import TestSuite, assert_equal, assert_true, assert_raises

from dataframe import CsvField, CsvSchema, DataFrame, read_csv


comptime PATH = "/tmp/dataframe_mojo_csv_borrowed_148.csv"
comptime SERIAL_LIMIT = 1000000


def write(text: String) raises:
    with open(PATH, "w") as file:
        file.write(text)


def write_bytes(bytes: List[UInt8]) raises:
    with open(PATH, "w") as file:
        file.write_bytes(bytes)


def text_schema() raises -> CsvSchema:
    return CsvSchema(
        [
            CsvField.string("a"),
            CsvField.string("b"),
            CsvField.string("c"),
        ]
    )


def typed_schema() raises -> CsvSchema:
    return CsvSchema(
        [
            CsvField.int64("id", False),
            CsvField.string("name"),
            CsvField.float64("score"),
        ]
    )


def read_text(size: Int) raises -> DataFrame:
    return read_csv(PATH, text_schema(), n_rows=SERIAL_LIMIT, buffer_size=size)


def read_typed(size: Int, n_rows: Int = SERIAL_LIMIT) raises -> DataFrame:
    return read_csv(PATH, typed_schema(), n_rows=n_rows, buffer_size=size)


def assert_text_matches_scalar() raises:
    var reference = read_text(1)
    for size in [64, 65, 128, 4096]:
        assert_true(
            read_text(size).equals(reference),
            "plain borrowed frame differs at buffer size " + String(size),
        )


def assert_typed_matches_scalar() raises:
    var reference = read_typed(1)
    for size in [64, 65, 128, 4096]:
        assert_true(
            read_typed(size).equals(reference),
            "typed borrowed frame differs at buffer size " + String(size),
        )


def error_text(size: Int, projected: Bool = False) -> String:
    try:
        if projected:
            var projected_schema = CsvSchema(
                [CsvField.int64("id"), CsvField.string("hidden")]
            )
            _ = read_csv(
                PATH,
                projected_schema,
                columns=["id"],
                n_rows=SERIAL_LIMIT,
                buffer_size=size,
            )
        else:
            _ = read_typed(size)
    except e:
        return String(e)
    return "ok"


def assert_error_matches_scalar(projected: Bool = False) raises:
    var reference = error_text(1, projected)
    assert_true(reference != "ok", "the scalar reference must fail")
    for size in [64, 65, 128, 4096]:
        assert_equal(
            error_text(size, projected),
            reference,
            "error text differs at buffer size " + String(size),
        )


def test_plain_structures_at_every_offset_in_two_blocks() raises:
    # `,,\n` makes every byte structural while still being a valid three-field
    # row. The first 128 data bytes therefore exercise a separator or LF at
    # every mask offset without making the one-byte reference expensive.
    var text = String("a,b,c\n")
    for _ in range(44):
        text += ",,\n"
    var ordinary = String()
    for _ in range(67):
        ordinary += "x"
    text += ordinary + ",plain,value\n"
    write(text)
    assert_text_matches_scalar()


def test_partial_rows_nulls_and_quote_crlf_fallbacks_resume_cleanly() raises:
    var long_name = String()
    for _ in range(95):
        long_name += "p"
    write(
        "id,name,score\n"
        + "1,"
        + long_name
        + ",1\n"
        + "2,,2\n"
        + "3,NA,3\n"
        + '4,"quoted, value",4\n'
        + '5,"say ""hello""",5\r\n'
        + "6,plain,6\n"
    )
    var reference = read_csv(
        PATH,
        typed_schema(),
        null_values=["NA"],
        n_rows=SERIAL_LIMIT,
        buffer_size=1,
    )
    for size in [64, 65, 128, 4096]:
        var actual = read_csv(
            PATH,
            typed_schema(),
            null_values=["NA"],
            n_rows=SERIAL_LIMIT,
            buffer_size=size,
        )
        assert_true(
            actual.equals(reference), "fallback mismatch " + String(size)
        )


def test_row_limit_stops_before_later_invalid_bytes() raises:
    var bytes: List[UInt8] = []
    bytes.extend("id,name,score\n1,first,1\nbad,".as_bytes())
    bytes.append(255)
    bytes.extend(",not-a-float\n".as_bytes())
    write_bytes(bytes)
    var reference = read_typed(1, n_rows=1)
    assert_equal(reference.height(), 1)
    for size in [64, 65, 128, 4096]:
        assert_true(
            read_typed(size, n_rows=1).equals(reference),
            "n_rows stopped differently at " + String(size),
        )


def test_full_error_text_for_plain_malformed_records() raises:
    # Each row starts plain so borrowing must hand it back to the scalar
    # state machine without changing strict error selection or location.
    for bad in [
        "id,name,score\n1,plain,1\n2,plain,2\r3,next,3\n",
        'id,name,score\n1,plain,1\n2"quote,2\n',
        "id,name,score\n1,plain,1\n2,plain,2,extra\n",
        "id,name,score\n1,plain\n",
    ]:
        write(bad)
        assert_error_matches_scalar()


def test_invalid_unprojected_string_matches_scalar_error_priority() raises:
    # An invalid first text field plus an extra field must still be replayed
    # through scalar parsing, where UTF-8 is reported before row width.
    var bytes: List[UInt8] = []
    bytes.extend("id,name,score\n1,plain,1\n2,".as_bytes())
    bytes.append(255)
    bytes.extend(",2,extra\n".as_bytes())
    write_bytes(bytes)
    assert_error_matches_scalar()


def test_utf8_validation_follows_projected_string_columns() raises:
    var bytes: List[UInt8] = []
    bytes.extend("id,hidden\n1,".as_bytes())
    bytes.append(255)
    bytes.extend("\n".as_bytes())
    write_bytes(bytes)
    # No String column is projected, so the omitted field is not decoded or
    # validated. This is the same projection rule Polars applies.
    var projected_schema = CsvSchema(
        [CsvField.int64("id"), CsvField.string("hidden")]
    )
    for size in [1, 64, 65, 128, 4096]:
        var frame = read_csv(
            PATH,
            projected_schema,
            columns=["id"],
            n_rows=SERIAL_LIMIT,
            buffer_size=size,
        )
        assert_equal(frame.height(), 1)

    # Selecting a String column validates the whole input chunk, including an
    # invalid byte in a projected-out field.
    bytes = []
    bytes.extend("id,name,score\n1,kept,".as_bytes())
    bytes.append(255)
    bytes.extend("\n".as_bytes())
    write_bytes(bytes)
    for size in [1, 64, 65, 128, 4096]:
        with assert_raises(contains="not valid UTF-8"):
            _ = read_csv(
                PATH,
                typed_schema(),
                columns=["name"],
                buffer_size=size,
            )


def test_simple_quotes_null_tokens_and_mask_boundaries() raises:
    var text = String("a,b,c\n")
    var padding = String()
    for _ in range(128):
        text += '"' + padding + '","a,b",""\n'
        padding += "x"
    text += 'NA,"NA",\n"",plain,"end"\n'
    write(text)
    var reference = read_csv(
        PATH,
        text_schema(),
        null_values=["NA"],
        n_rows=SERIAL_LIMIT,
        buffer_size=1,
    )
    for size in [64, 65, 128, 4096]:
        var actual = read_csv(
            PATH,
            text_schema(),
            null_values=["NA"],
            n_rows=SERIAL_LIMIT,
            buffer_size=size,
        )
        assert_true(actual.equals(reference))


def test_simple_quote_fallback_error_precedence() raises:
    for row in [
        '2,"name"junk,2\n',
        '2,"a""b",bad\n',
        'bad,"name",2,extra\n',
        '2,"unterminated,2',
        '2,"name",2\r3,next,3\n',
    ]:
        write('id,name,score\n1,"plain",1\n' + row)
        assert_error_matches_scalar()


def test_invalid_quoted_text_precedes_numeric_conversion() raises:
    var bytes: List[UInt8] = []
    bytes.extend('id,name,score\n1,"plain",1\nbad,"'.as_bytes())
    bytes.append(255)
    bytes.extend('",2\n'.as_bytes())
    write_bytes(bytes)
    assert_error_matches_scalar()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
