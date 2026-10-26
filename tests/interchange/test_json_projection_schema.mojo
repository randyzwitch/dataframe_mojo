"""Projection skips conversions while retaining strict unknown-key checks."""
# contract-group: io.json
from std.os import getenv
from std.testing import assert_equal, assert_raises
from dataframe import read_ndjson, scan_ndjson


def test_json_projection_rejects_late_unknown_keys_inside_unselected_nested_fields() raises:
    var path = getenv("COVERAGE_TEST_TMP") + "/json-projected-unknown.ndjson"
    for nested in [
        String('{"sub":{"known":2,"late":3},"items":[]}'),
        String('{"sub":null,"items":[{"name":"x","late":true}]}'),
    ]:
        with open(path, "w") as file:
            file.write('{"keep":1,"sub":{"known":1},"items":[{"name":"a"}]}\n')
            file.write(nested)
        for rows in [-1, 2]:
            with assert_raises(contains="outside"):
                _ = read_ndjson(
                    path, infer_schema_length=1, columns=["keep"], n_rows=rows
                )
        for streaming in [False, True]:
            with assert_raises(contains="outside"):
                _ = (
                    scan_ndjson(path, infer_schema_length=1)
                    .select(["keep"])
                    .collect(engine="cpu", streaming=streaming, batch_size=1)
                )
            var ignored = (
                scan_ndjson(path, infer_schema_length=1, ignore_errors=True)
                .select(["keep"])
                .collect(engine="cpu", streaming=streaming, batch_size=1)
            )
            assert_equal(ignored.height(), 2)
            assert_equal(ignored.column("keep").null_count(), 1)
