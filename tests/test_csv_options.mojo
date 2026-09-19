"""Options for read_csv: dialect, skipping, projection, nulls, permissive modes.

Every scenario is read with every buffer size from 1 byte to the whole file,
so results cannot depend on where the input is split.
"""
from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
    assert_false,
    assert_raises,
)
from dataframe import CsvField, CsvSchema, DataFrame, read_csv

comptime PATH = "/tmp/dataframe_mojo_options_test.csv"


def write(text: String) raises -> Int:
    with open(PATH, "w") as file:
        file.write(text)
    return text.byte_length()


def schema() raises -> CsvSchema:
    return CsvSchema(
        [
            CsvField.int64("id"),
            CsvField.string("name"),
            CsvField.float64("score"),
        ]
    )


def cells(frame: DataFrame) raises -> String:
    var out = String()
    for r in range(frame.height()):
        for c in range(frame.width()):
            out += String(frame._columns[c].get(r)) + (
                "|" if c + 1 < frame.width() else "\n"
            )
    return out^


def read_all_splits(
    length: Int,
    has_header: Bool = True,
    separator: String = ",",
    quote_char: String = '"',
    comment_prefix: String = "",
    skip_rows: Int = 0,
    n_rows: Int = -1,
    columns: List[String] = List[String](),
    null_values: List[String] = List[String](),
    ignore_errors: Bool = False,
    truncate_ragged_lines: Bool = False,
    encoding: String = "utf8",
) raises -> String:
    """Read with every buffer size and require identical results."""
    var expected = String()
    for size in range(1, length + 2):
        var frame = read_csv(
            PATH,
            schema(),
            has_header=has_header,
            separator=separator,
            quote_char=quote_char,
            comment_prefix=comment_prefix,
            skip_rows=skip_rows,
            n_rows=n_rows,
            columns=columns,
            null_values=null_values,
            ignore_errors=ignore_errors,
            truncate_ragged_lines=truncate_ragged_lines,
            encoding=encoding,
            buffer_size=size,
        )
        var text = String(frame.width()) + ":" + cells(frame)
        if size == 1:
            expected = text
        else:
            assert_equal(text, expected, msg="buffer size " + String(size))
    return expected


def test_separator_and_quote_char() raises:
    var n = write("id;name;score\r\n1;'a;b';1.5\r\n2;'it''s';\r\n")
    assert_equal(
        read_all_splits(n, separator=";", quote_char="'"),
        "3:1|a;b|1.5\n2|it's|null\n",
    )
    n = write('id\tname\tscore\n1\t"x\ty"\t2\n')
    assert_equal(read_all_splits(n, separator="\t"), "3:1|x\ty|2.0\n")
    # Quoting disabled: quote bytes are ordinary data.
    n = write('id,name,score\n1,"raw",3\n')
    assert_equal(read_all_splits(n, quote_char=""), '3:1|"raw"|3.0\n')


def test_comments_and_skip_rows() raises:
    var n = write(
        "# generated\nid,name,score\n#x,skipped,1\n1,a,1\n## more\n2,#b,2\n#"
    )
    assert_equal(
        read_all_splits(n, comment_prefix="#"), "3:1|a|1.0\n2|#b|2.0\n"
    )
    # A partial prefix match ("-" of "--") is replayed as ordinary data.
    n = write("--a\nid,name,score\n-1,-y,3\n-- note\n")
    assert_equal(read_all_splits(n, comment_prefix="--"), "3:-1|-y|3.0\n")
    n = write('junk "line\nmore junk\nid,name,score\n5,e,5\n')
    assert_equal(read_all_splits(n, skip_rows=2), "3:5|e|5.0\n")


def test_n_rows_and_projection() raises:
    var n = write("id,name,score\n1,a,1\n2,b,2\n3,c,x\n")
    assert_equal(read_all_splits(n, n_rows=2), "3:1|a|1.0\n2|b|2.0\n")
    assert_equal(read_all_splits(n, n_rows=0), "3:")
    # Projection never decodes the bad score in the third row.
    assert_equal(
        read_all_splits(n, columns=["name", "id"]), "2:1|a\n2|b\n3|c\n"
    )
    with assert_raises(contains="Unknown CSV column: zzz"):
        _ = read_csv(PATH, schema(), columns=["zzz"])
    with assert_raises(contains="listed twice"):
        _ = read_csv(PATH, schema(), columns=["id", "id"])


def test_null_values() raises:
    var n = write('id,name,score\nNA,NA,NA\n1,"NA",-\n')
    assert_equal(
        read_all_splits(n, null_values=["NA", "-"]),
        "3:null|null|null\n1|NA|null\n",
    )
    with assert_raises(contains="invalid Int64 value 'NA'"):
        _ = read_csv(PATH, schema())


def test_permissive_modes() raises:
    var n = write("id,name,score\n1,a,1\nbad,b,2\n3,c\n4,d,4,extra\n5,e,5\n")
    assert_equal(read_all_splits(n, ignore_errors=True), "3:1|a|1.0\n5|e|5.0\n")
    n = write("id,name,score\n1,a,1\n3,c\n4,d,4,extra\n")
    assert_equal(
        read_all_splits(n, truncate_ragged_lines=True),
        "3:1|a|1.0\n3|c|null\n4|d|4.0\n",
    )
    with assert_raises(contains="expected 3 fields, found 2"):
        _ = read_csv(PATH, schema())
    # Header errors are never skipped.
    n = write("id,wrong,score\n1,a,1\n")
    with assert_raises(contains="CSV header field 2"):
        _ = read_csv(PATH, schema(), ignore_errors=True)


def test_lossy_encoding() raises:
    with open(PATH, "w") as file:
        file.write("id,name,score\n1,")
    var bytes: List[UInt8] = [104, 255, 105, 44, 50, 10]
    with open(PATH, "a") as file:
        file.write_bytes(bytes)
    assert_equal(read_all_splits(22, encoding="utf8-lossy"), "3:1|h�i|2.0\n")
    with assert_raises(contains="not valid UTF-8"):
        _ = read_csv(PATH, schema())


def test_option_validation() raises:
    _ = write("id,name,score\n")
    with assert_raises(contains="separator must be a single byte"):
        _ = read_csv(PATH, schema(), separator="")
    with assert_raises(contains="quote_char cannot be the separator"):
        _ = read_csv(PATH, schema(), quote_char=",")
    with assert_raises(contains="quote_char must be one byte"):
        _ = read_csv(PATH, schema(), quote_char="''")
    with assert_raises(contains="skip_rows"):
        _ = read_csv(PATH, schema(), skip_rows=-1)
    with assert_raises(contains="n_rows"):
        _ = read_csv(PATH, schema(), n_rows=-2)
    with assert_raises(contains="encoding"):
        _ = read_csv(PATH, schema(), encoding="latin1")
    with assert_raises(contains="comment_prefix"):
        _ = read_csv(PATH, schema(), comment_prefix="\n")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
