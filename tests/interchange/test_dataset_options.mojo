"""Path metadata is inferred without attempting to parse source contents."""
# contract-group: io.sources.metadata
from std.collections import Dict
from std.os import getenv, mkdir
from std.os.path import exists
from std.testing import assert_equal, assert_true, assert_raises
from dataframe.dtype import DataType
from dataframe.io.dataset_options import (
    DatasetOptions,
    DatasetSpec,
    HIVE_NULL,
    hive_encode,
    hive_decode,
    hive_int64,
)


def _metadata_file(name: String) raises -> String:
    var root = getenv("COVERAGE_TEST_TMP") + "/dataset-metadata"
    if not exists(root):
        mkdir(root)
    var parent = root.copy()
    var parts = name.split("/")
    for p in range(len(parts) - 1):
        parent += "/" + String(parts[p])
        if not exists(parent):
            mkdir(parent)
    var path = root + "/" + name
    with open(path, "w") as file:
        file.write("deliberately invalid source contents")
    return path


def test_hive_percent_encoding_roundtrips_utf8_reserved_and_null_literal() raises:
    for value in [String("plain"), "a/b=c%20", "café", "", HIVE_NULL]:
        assert_equal(hive_decode(hive_encode(value)), value)
    assert_equal(hive_decode("a%252Fb"), "a%2Fb")
    assert_true(hive_encode(HIVE_NULL) != HIVE_NULL)
    for invalid in [String("%"), "%2", "%GG", "%FF"]:
        with assert_raises():
            _ = hive_decode(invalid)


def test_hive_integer_conversion_checks_signed_boundaries() raises:
    assert_equal(hive_int64("9223372036854775807"), Int64.MAX)
    assert_equal(hive_int64("-9223372036854775808"), Int64.MIN)
    assert_equal(hive_int64("+001"), Int64(1))
    for invalid in [
        String("9223372036854775808"),
        "-9223372036854775809",
        "",
        "+",
        "1.0",
        " 1",
    ]:
        with assert_raises():
            _ = hive_int64(invalid)


def test_dataset_manifest_sorts_preserves_repetitions_and_infers_all_paths() raises:
    var a = _metadata_file("year=2024/label=a%2Fb/a.parquet")
    var b = _metadata_file("year=2025/label=word/b.parquet")
    var spec = DatasetSpec([b, a, a], DatasetOptions(hive_partitioning=True))
    assert_equal(spec.paths, List[String]([a, a, b]))
    assert_equal(spec.hive_names, List[String](["year", "label"]))
    assert_equal(spec.hive_dtypes[0], DataType.INT64)
    assert_equal(spec.hive_dtypes[1], DataType.STRING)
    assert_equal(spec.hive_values[0][1], "a/b")
    assert_equal(spec.hive_values[2][0], "2025")
    assert_true(spec.active())
    var text = _metadata_file("year=unknown/label=z/c.parquet")
    var mixed = DatasetSpec([a, text], DatasetOptions(hive_partitioning=True))
    assert_equal(mixed.hive_dtypes[mixed.hive_index("year")], DataType.STRING)
    var overrides = Dict[String, String]()
    overrides["year"] = "string"
    var forced = DatasetSpec(
        [a, b], DatasetOptions(hive_partitioning=True, hive_schema=overrides)
    )
    assert_equal(forced.hive_dtypes[forced.hive_index("year")], DataType.STRING)


def test_hive_null_missing_and_encoded_sentinel_are_distinct() raises:
    var null_path = _metadata_file("key=" + HIVE_NULL + "/null.parquet")
    var literal_path = _metadata_file(
        "key=" + hive_encode(HIVE_NULL) + "/literal.parquet"
    )
    var absent = _metadata_file("unpartitioned/absent.parquet")
    var spec = DatasetSpec(
        [null_path, literal_path, absent],
        DatasetOptions(hive_partitioning=True),
    )
    var key = spec.hive_index("key")
    for f in range(len(spec.paths)):
        if spec.paths[f] == literal_path:
            assert_true(not spec.hive_nulls[f][key])
            assert_equal(spec.hive_values[f][key], HIVE_NULL)
        else:
            assert_true(spec.hive_nulls[f][key])


def test_dataset_options_and_hive_paths_reject_invalid_metadata() raises:
    assert_true(not DatasetOptions().active())
    with assert_raises(contains="missing_columns"):
        _ = DatasetOptions(missing_columns="drop")
    with assert_raises(contains="extra_columns"):
        _ = DatasetOptions(extra_columns="insert")
    var overrides = Dict[String, String]()
    overrides["key"] = "not-a-dtype"
    with assert_raises(contains="hive_schema"):
        _ = DatasetOptions(hive_schema=overrides)
    var duplicate = _metadata_file("same=1/same=2/duplicate.parquet")
    with assert_raises(contains="duplicate Hive"):
        _ = DatasetSpec([duplicate], DatasetOptions(hive_partitioning=True))
    var overflow = _metadata_file("number=9223372036854775808/overflow.parquet")
    var inferred = DatasetSpec(
        [overflow], DatasetOptions(hive_partitioning=True)
    )
    assert_equal(inferred.hive_dtypes[0], DataType.STRING)
    overrides = Dict[String, String]()
    overrides["number"] = "int64"
    with assert_raises(contains="overflow"):
        _ = DatasetSpec(
            [overflow],
            DatasetOptions(hive_partitioning=True, hive_schema=overrides),
        )
