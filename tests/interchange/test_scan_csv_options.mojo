"""scan_csv takes read_csv's reader options, and they reach the reader
however the plan runs: eagerly, streaming in batches, with a filter run per
decoded range, and for the schema alone. Each scan gives read_csv's frame."""
# contract-group: io.scan_csv_options
from std.collections import Dict
from std.os import getenv
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    CsvField,
    CsvSchema,
    DataFrame,
    LazyFrame,
    col,
    lit,
    read_csv,
    scan_csv,
)
from tests.support.coverage_compare import assert_frame


def _path(name: String) -> String:
    return getenv("COVERAGE_TEST_TMP") + "/scan-options-" + name + ".csv"


def _write(name: String, text: String) raises -> String:
    var path = _path(name)
    with open(path, "w") as file:
        file.write(text)
    return path


def check(plan: LazyFrame, expected: DataFrame) raises:
    """Every executor setting gives `expected`, and the schema matches."""
    for optimize in [True, False]:
        for streaming in [True, False]:
            for batch_size in [65536, 2]:
                assert_frame(
                    plan.collect(
                        optimize=optimize,
                        streaming=streaming,
                        batch_size=batch_size,
                    ),
                    expected,
                )
    var schema = List[String]()
    for field in expected.schema():
        schema.append(field.name + ": " + field.dtype.name())
    assert_equal(plan.collect_schema(), schema)


def check_queries(plan: LazyFrame, expected: DataFrame, column: String) raises:
    """The scan alone, under a filter (run per decoded range), a column
    selection and a head."""
    check(plan, expected)
    check(
        plan.filter(col(column).is_not_null()),
        expected.filter(col(column).is_not_null()),
    )
    check(plan.select([column]), expected.select([column]))
    check(plan.head(2), expected.head(2))


def test_separator_header_and_skipped_rows() raises:
    var tsv = _write(
        "tsv", "junk line\nmore junk\na\tb\tc\n1\tx y\t2.5\n2\t\t3.5\n3\tz\t\n"
    )
    check_queries(
        scan_csv(tsv, separator="\t", skip_rows=2),
        read_csv(tsv, separator="\t", skip_rows=2),
        "b",
    )
    var bare = _write("bare", "1;x;2.5\n2;y;3.5\n3;;4.5\n")
    var plan = scan_csv(bare, separator=";", has_header=False)
    var expected = read_csv(bare, separator=";", has_header=False)
    check_queries(plan, expected, "column_2")
    assert_equal(expected.columns(), ["column_1", "column_2", "column_3"])
    assert_equal(expected.height(), 3)


def test_quotes_comments_and_nulls() raises:
    var quoted = _write(
        "quoted",
        "# a comment\nid,name,score\n1,'a,b',NA\n# inside\n2,'c',-\n3,'d',4\n"
        + "4,'e',5\n",
    )
    var nulls: List[String] = ["NA", "-"]
    var plan = scan_csv(
        quoted, quote_char="'", comment_prefix="#", null_values=nulls
    )
    var expected = read_csv(
        quoted, quote_char="'", comment_prefix="#", null_values=nulls
    )
    check_queries(plan, expected, "score")
    assert_equal(expected.height(), 4)
    assert_equal(expected.item(0, "name").string(), "a,b")
    assert_true(expected.item(1, "score").is_null())
    # A head over a file with comment lines still gives its first rows.
    check(plan.head(3), expected.head(3))


def test_inference_settings() raises:
    var text = _write(
        "infer",
        "a,b,when\n1,2,2024-01-02\n2,3,2024-02-03\n3,x,2024-03-04\n",
    )
    var overrides = Dict[String, String]()
    overrides["a"] = "string"
    check_queries(
        scan_csv(text, schema_overrides=overrides, try_parse_dates=True),
        read_csv(text, schema_overrides=overrides, try_parse_dates=True),
        "a",
    )
    # Inferring from the first row only, b is Int64 and "x" fails to parse:
    # ignore_errors makes it null, as read_csv does.
    check_queries(
        scan_csv(text, infer_schema_length=1, ignore_errors=True),
        read_csv(text, infer_schema_length=1, ignore_errors=True),
        "b",
    )
    var ragged = _write("ragged", "a,b\n1,2\n3,4,5\n6,7\n")
    check(
        scan_csv(ragged, truncate_ragged_lines=True),
        read_csv(ragged, truncate_ragged_lines=True),
    )


def test_explicit_schema_with_options() raises:
    var path = _write("explicit", "skip me\n1|x\n2|\n3|z\n")
    var schema = CsvSchema([CsvField.int64("i"), CsvField.string("s")])
    var plan = scan_csv(
        path, schema, separator="|", has_header=False, skip_rows=1
    )
    var expected = read_csv(
        path, schema, separator="|", has_header=False, skip_rows=1
    )
    check_queries(plan, expected, "s")
    assert_equal(expected.height(), 3)


def test_invalid_options_raise_when_scanning() raises:
    var path = _write("plain", "a\n1\n")
    with assert_raises(contains="separator"):
        _ = scan_csv(path, separator="ab")
    with assert_raises(contains="skip_rows"):
        _ = scan_csv(path, skip_rows=-1)
    with assert_raises(contains="encoding"):
        _ = scan_csv(path, encoding="latin1")
    # Defaults build the plan they always did.
    var plan = scan_csv(path)
    assert_equal(len(plan._plan._nodes[0].names2), 0)
    check(plan, read_csv(path))
