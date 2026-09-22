"""Public CSV dialect, projection, null, and permissive-mode semantics.

Every successful case is decoded by the same CountLines/chunk pipeline at one
and 32 workers. Expected permissive and null behavior is pinned to Polars
1.44.2 rather than the retired scalar reader.
"""
from std.ffi import external_call
from std.testing import TestSuite, assert_equal, assert_raises, assert_true
from dataframe import CsvField, CsvSchema, DataFrame, read_csv

comptime PATH = "/tmp/dataframe_mojo_options_test.csv"


def c_string(text: String) -> List[UInt8]:
    var bytes = List[UInt8]()
    bytes.extend(text.as_bytes())
    bytes.append(0)
    return bytes^


def set_threads(n: Int):
    var name = c_string("DATAFRAME_THREADS")
    var value = c_string(String(n))
    _ = external_call["setenv", Int32](
        Int(name.unsafe_ptr()), Int(value.unsafe_ptr()), Int32(1)
    )
    _ = name^
    _ = value^


def write(text: String) raises:
    with open(PATH, "w") as file:
        file.write(text)


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


def read_at_workers(
    workers: Int,
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
) raises -> DataFrame:
    set_threads(workers)
    return read_csv(
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
        buffer_size=1 if workers == 1 else 65536,
    )


def read_all_workers(
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
    var one = read_at_workers(
        1,
        has_header,
        separator,
        quote_char,
        comment_prefix,
        skip_rows,
        n_rows,
        columns,
        null_values,
        ignore_errors,
        truncate_ragged_lines,
        encoding,
    )
    var many = read_at_workers(
        32,
        has_header,
        separator,
        quote_char,
        comment_prefix,
        skip_rows,
        n_rows,
        columns,
        null_values,
        ignore_errors,
        truncate_ragged_lines,
        encoding,
    )
    assert_true(one.equals(many), "one and 32 workers differ")
    return String(many.width()) + ":" + cells(many)


def test_separator_and_quote_char() raises:
    write("id;name;score\r\n1;'a;b';1.5\r\n2;'it''s';\r\n")
    assert_equal(
        read_all_workers(separator=";", quote_char="'"),
        "3:1|a;b|1.5\n2|it's|null\n",
    )
    write('id\tname\tscore\n1\t"x\ty"\t2\n')
    assert_equal(read_all_workers(separator="\t"), "3:1|x\ty|2.0\n")
    write('id,name,score\n1,"raw",3\n')
    assert_equal(read_all_workers(quote_char=""), '3:1|"raw"|3.0\n')


def test_comments_and_skip_rows() raises:
    write("# generated\nid,name,score\n#x,skipped,1\n1,a,1\n## more\n2,#b,2\n#")
    assert_equal(read_all_workers(comment_prefix="#"), "3:1|a|1.0\n2|#b|2.0\n")
    write("--a\nid,name,score\n-1,-y,3\n-- note\n")
    assert_equal(read_all_workers(comment_prefix="--"), "3:-1|-y|3.0\n")
    write('junk "line\nmore junk\nid,name,score\n5,e,5\n')
    assert_equal(read_all_workers(skip_rows=2), "3:5|e|5.0\n")


def test_n_rows_and_projection() raises:
    write("id,name,score\n1,a,1\n2,b,2\n3,c,x\n")
    # n_rows limits the returned frame after a source chunk has been parsed;
    # Polars still reports an invalid selected value later in that chunk.
    with assert_raises():
        _ = read_csv(PATH, schema(), n_rows=2)
    with assert_raises():
        _ = read_csv(PATH, schema(), n_rows=0)
    # The score is unselected here, so parse_lines skips its tail entirely.
    assert_equal(read_all_workers(columns=["name", "id"]), "2:1|a\n2|b\n3|c\n")
    write("id,name,score\n1,a,1\n2,b,2\n3,c,3\n")
    assert_equal(read_all_workers(n_rows=2), "3:1|a|1.0\n2|b|2.0\n")
    assert_equal(read_all_workers(n_rows=0), "3:")
    with assert_raises(contains="Unknown CSV column: zzz"):
        _ = read_csv(PATH, schema(), columns=["zzz"])
    with assert_raises(contains="listed twice"):
        _ = read_csv(PATH, schema(), columns=["id", "id"])


def test_null_values_match_quoted_and_unquoted_markers() raises:
    write('id,name,score\nNA,NA,NA\n1,"NA",-\n')
    assert_equal(
        read_all_workers(null_values=["NA", "-"]),
        "3:null|null|null\n1|null|null\n",
    )
    with assert_raises():
        _ = read_csv(PATH, schema())


def test_permissive_conversion_null_fill_and_ragged_policy() raises:
    # Polars ignore_errors null-fills a failed primitive; it does not drop its
    # record as the previous reader did.
    write("id,name,score\n1,a,1\nbad,b,2\n3,c\n5,e,5\n")
    assert_equal(
        read_all_workers(ignore_errors=True),
        "3:1|a|1.0\nnull|b|2.0\n3|c|null\n5|e|5.0\n",
    )
    # A surplus field remains structural and needs truncate_ragged_lines;
    # ignore_errors only handles conversion failures.
    write("id,name,score\n1,a,1\n3,c\n4,d,4,extra\n")
    with assert_raises():
        _ = read_csv(PATH, schema(), ignore_errors=True)
    assert_equal(
        read_all_workers(truncate_ragged_lines=True),
        "3:1|a|1.0\n3|c|null\n4|d|4.0\n",
    )
    # Schema field names label positions. Polars does not require the input
    # header to duplicate the explicit schema names.
    write("id,wrong,score\n1,a,1\n")
    assert_equal(read_all_workers(), "3:1|a|1.0\n")


def test_numeric_leading_whitespace_and_case_insensitive_bool() raises:
    write("id,name,score\n \t12,plain,\t1.5\n")
    assert_equal(read_all_workers(), "3:12|plain|1.5\n")
    write("id,name,score\n12 ,plain,1\n")
    with assert_raises():
        _ = read_csv(PATH, schema())
    var booleans = CsvSchema([CsvField.bool("active")])
    write("active\nTRUE\nfAlSe\n")
    for workers in [1, 32]:
        set_threads(workers)
        var frame = read_csv(PATH, booleans)
        assert_true(frame.column("active").bool().value(0))
        assert_true(not frame.column("active").bool().value(1))


def test_lossy_encoding() raises:
    with open(PATH, "w") as file:
        file.write("id,name,score\n1,")
    var bytes: List[UInt8] = [104, 255, 105, 44, 50, 10]
    with open(PATH, "a") as file:
        file.write_bytes(bytes)
    assert_equal(read_all_workers(encoding="utf8-lossy"), "3:1|h�i|2.0\n")
    with assert_raises(contains="not valid UTF-8"):
        _ = read_csv(PATH, schema())


def test_option_validation() raises:
    write("id,name,score\n")
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
