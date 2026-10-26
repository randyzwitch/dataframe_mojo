"""Lazy NDJSON schema, projection, streaming and source-identity contracts."""
# contract-group: io.json
from std.os import getenv
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Series,
    col,
    scan_ndjson,
    write_ndjson,
)
from tests.support.coverage_compare import assert_frame
from tests.support.group_strategy_reference import ScopedEnvironment


def _path(name: String) -> String:
    return getenv("COVERAGE_TEST_TMP") + "/" + name


def test_ndjson_scan_defers_opens_reuses_plans_and_samples_only_schema() raises:
    var path = _path("json-lazy-deferred.ndjson")
    var plan = scan_ndjson(path, infer_schema_length=1)
    with open(path, "w") as file:
        file.write('{"x":1,"s":"first"}\n{bad}')
    assert_equal(plan.collect_schema(), List[String](["x: int64", "s: string"]))
    assert_true("SCAN NDJSON" in plan.explain())
    var full = DataFrame(
        [
            Series("x", Column[Int64]([1, 2, 3])),
            Series("s", Column[String](["first", "é😀", "last"])),
        ]
    )
    write_ndjson(full, path)
    for _ in range(2):
        assert_frame(plan.collect(engine="cpu", batch_size=2), full)
    var missing = scan_ndjson(_path("json-lazy-never-created.ndjson"))
    with assert_raises():
        _ = missing.collect(engine="cpu")


def test_ndjson_scan_streamed_filters_groups_projections_and_heads_match_eager() raises:
    var path = _path("json-lazy-groups.ndjson")
    var keys = List[Int64]()
    var values = List[Int64]()
    var valid = List[Bool]()
    for row in range(513):
        keys.append(Int64(row % 7))
        values.append(Int64(row))
        valid.append(row % 11 != 0)
    var frame = DataFrame(
        [
            Series("k", Column[Int64](keys^)),
            Series("v", Column[Int64](values^, valid)),
        ]
    )
    write_ndjson(frame, path)
    for threads in [1, 2]:
        var workers = ScopedEnvironment("DATAFRAME_THREADS", String(threads))
        for optimize in [False, True]:
            for streaming in [False, True]:
                for size in [1, 17, 512]:
                    var result = (
                        scan_ndjson(path)
                        .filter(col("v") > 100)
                        .group_by("k")
                        .agg(col("v").sum())
                        .sort("k")
                        .collect(
                            engine="cpu",
                            optimize=optimize,
                            streaming=streaming,
                            batch_size=size,
                        )
                    )
                    assert_frame(
                        result,
                        frame.filter(col("v") > 100)
                        .group_by("k")
                        .agg(col("v").sum())
                        .sort("k"),
                    )
                    var head = (
                        scan_ndjson(path)
                        .select(["v"])
                        .head(9)
                        .collect(
                            engine="cpu",
                            optimize=optimize,
                            streaming=streaming,
                            batch_size=size,
                        )
                    )
                    assert_frame(head, frame.select(["v"]).head(9))
        _ = workers^


def test_ndjson_scan_projection_limits_and_explicit_schemas_keep_error_semantics() raises:
    var path = _path("json-lazy-errors.ndjson")
    var schema = DataType.struct(
        ["x", "unused"], [DataType.INT64, DataType.INT64]
    )
    with open(path, "w") as file:
        file.write('{"x":1,"unused":"bad"}\n{bad}')
    var plan = scan_ndjson(path, schema).select(["x"]).head(1)
    assert_true("project x" in plan.explain())
    for streaming in [False, True]:
        var result = plan.collect(
            engine="cpu", streaming=streaming, batch_size=1
        )
        assert_frame(result, DataFrame([Series("x", Column[Int64]([1]))]))
        assert_equal(
            scan_ndjson(path, schema)
            .head(0)
            .collect(engine="cpu", streaming=streaming)
            .height(),
            0,
        )
    with assert_raises(contains="line 2"):
        _ = (
            scan_ndjson(path, schema, ignore_errors=True)
            .select(["x"])
            .collect(engine="cpu")
        )
    with open(path, "w") as file:
        file.write('{"x":1}\n{"x":2,"late":true}')
    with assert_raises(contains="outside"):
        _ = scan_ndjson(path, infer_schema_length=1).collect(engine="cpu")
    assert_equal(
        scan_ndjson(path, infer_schema_length=1, ignore_errors=True)
        .collect(engine="cpu")
        .height(),
        2,
    )


def test_ndjson_scan_shared_sources_with_distinct_schemas_are_not_conflated() raises:
    var path = _path("json-lazy-schema-identity.ndjson")
    with open(path, "w") as file:
        file.write('{"x":1}\n{"x":2}')
    var integer = scan_ndjson(path, DataType.struct(["x"], [DataType.INT64]))
    var strings = scan_ndjson(
        path, DataType.struct(["x"], [DataType.STRING])
    ).rename({"x": "s"})
    var joined = integer.join(strings, left_on=["x"], right_on=["s"])
    # Type checking must still see the two different source schemas.
    with assert_raises():
        _ = joined.collect(engine="cpu")
    var zero = _path("json-lazy-zero.ndjson")
    with open(zero, "w") as file:
        file.write("{}\n{}")
    assert_equal(
        scan_ndjson(zero).collect(engine="cpu", batch_size=1).height(), 2
    )
