"""Independent Polars corpus, also emitted for external writer validation."""
# contract-group: io.json
from std.os import getenv
from std.testing import assert_equal
from dataframe import (
    read_json,
    read_ndjson,
    write_json,
    write_ndjson,
    scan_ndjson,
)
from tests.support.coverage_compare import assert_frame


def test_json_polars_corpus_covers_escaped_unicode_missing_fields_and_nested_nulls() raises:
    var actual = read_ndjson(
        "tests/fixtures/json_tables/mixed.ndjson", batch_size=17
    )
    var expected = read_ndjson(
        "tests/fixtures/json_tables/mixed.expected.ndjson", batch_size=13
    )
    assert_frame(actual, expected)
    assert_frame(
        read_json(
            "tests/fixtures/json_tables/mixed.expected.json", batch_size=11
        ),
        expected,
    )
    assert_equal(actual.height(), 257)
    assert_equal(
        actual.column("text").get(9).string(), 'é😀\n\r\t\b\f\u0000"\\/'
    )
    assert_frame(
        scan_ndjson("tests/fixtures/json_tables/mixed.ndjson").collect(
            engine="cpu", batch_size=19
        ),
        expected,
    )
    var root = getenv("COVERAGE_TEST_TMP")
    write_ndjson(actual, root + "/json-written-oracle.ndjson", buffer_size=127)
    write_json(actual, root + "/json-written-oracle.json", buffer_size=127)
