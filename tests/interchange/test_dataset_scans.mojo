"""Dataset batches read one source at a time and align public column values."""
# contract-group: io.sources.dataset
from std.collections import Optional
from std.os import getenv, mkdir
from std.os.path import exists
from std.testing import assert_equal, assert_true, assert_raises
from dataframe.dtype import DataType
from dataframe.expr import col
from dataframe.frame import DataFrame, concat
from dataframe.series import Series
from dataframe.storage.string import StringColumn
from dataframe.categorical import encode, decode
from dataframe.io.dataset_options import DatasetOptions, DatasetSpec
from dataframe.io.csv_types import _CsvScanOptions
from dataframe.io.dataset_sources import dataset_schema, align_dataset_frame
from dataframe.io.dataset_scan import (
    _DatasetBatches,
    read_dataset,
    dataset_schema_frame,
)


def _csv(name: String, contents: String) raises -> String:
    var root = getenv("COVERAGE_TEST_TMP") + "/dataset-scans"
    if not exists(root):
        mkdir(root)
    var parts = name.split("/")
    var directory = root.copy()
    for p in range(len(parts) - 1):
        directory += "/" + String(parts[p])
        if not exists(directory):
            mkdir(directory)
    var path = root + "/" + name
    with open(path, "w") as file:
        file.write(contents)
    return path


def test_categorical_alignment_retains_labels_across_empty_and_reordered_dictionaries() raises:
    var path = _csv("categorical/metadata.csv", "unused\n")
    var spec = DatasetSpec([path, path, path], DatasetOptions())
    var empty = DataFrame(
        [encode(Series("label", StringColumn(List[String]())))]
    )
    var first = DataFrame(
        [encode(Series("label", StringColumn(["red", "blue", "red"])))]
    )
    var second = DataFrame(
        [encode(Series("label", StringColumn(["blue", "red"])))]
    )
    var target = dataset_schema(spec, [empty^, first.clear(), second.clear()])
    assert_true(target.column("label").dtype().is_categorical())
    var aligned_first = align_dataset_frame(spec, 1, first, target)
    var aligned_second = align_dataset_frame(spec, 2, second, target)
    var actual = concat([aligned_first^, aligned_second^])
    var labels = decode(actual.column("label"))
    var expected: List[String] = ["red", "blue", "red", "blue", "red"]
    for row in range(len(expected)):
        assert_equal(labels.get(row).string(), expected[row])
    var plain = DataFrame([Series("label", StringColumn(["red"]))])
    with assert_raises(contains="dtype mismatch"):
        _ = align_dataset_frame(spec, 1, plain, target)


def test_csv_dataset_batches_align_union_hive_values_and_global_limit() raises:
    var a = _csv(
        "union/year=2024/a.csv", "id,year,left\n1,999,alpha\n2,999,beta\n"
    )
    var b = _csv("union/year=2025/b.csv", "id,right\n3,30\n4,40\n")
    var spec = DatasetSpec(
        [b, a, a],
        DatasetOptions(
            hive_partitioning=True,
            union_by_name=True,
            include_file_paths="path",
        ),
    )
    var batches = _DatasetBatches(spec, False, batch_size=1, limit=5)
    assert_equal(
        batches.schema.columns(),
        List[String](["id", "year", "left", "path", "right"]),
    )
    var ids = List[Int64]()
    var years = List[Int64]()
    var paths = List[String]()
    while True:
        var next = batches.next()
        if not next:
            break
        var frame = next.take()
        assert_equal(frame.height(), 1)
        ids.append(frame.column("id").get(0).int64())
        years.append(frame.column("year").get(0).int64())
        paths.append(frame.column("path").get(0).string())
        if ids[len(ids) - 1] <= 2:
            assert_true(frame.column("right").get(0).is_null())
        else:
            assert_true(frame.column("left").get(0).is_null())
    batches.finish()
    var expected_ids: List[Int64] = [1, 2, 1, 2, 3]
    var expected_years: List[Int64] = [2024, 2024, 2024, 2024, 2025]
    assert_equal(ids, expected_ids)
    assert_equal(years, expected_years)
    assert_equal(paths, List[String]([a, a, a, a, b]))


def test_csv_partition_pruning_skips_bad_payload_and_dtype_validation() raises:
    var a = _csv("pruning/key=1/a.csv", "id,value\n1,good\n")
    var b = _csv("pruning/key=2/b.csv", "id,value\nbad,other\n")
    var spec = DatasetSpec([a, b], DatasetOptions(hive_partitioning=True))
    var target = dataset_schema_frame(spec, False)
    assert_equal(target.column("id").dtype(), DataType.INT64)
    var kept = read_dataset(
        spec, False, predicate=Optional(col("key") == 1), batch_size=1
    )
    assert_equal(kept.height(), 1)
    assert_equal(kept.column("id").get(0).int64(), Int64(1))
    var empty = read_dataset(spec, False, predicate=Optional(col("key") == 999))
    assert_equal(empty.height(), 0)
    assert_equal(empty.columns(), target.columns())
    assert_equal(empty.column("id").dtype(), DataType.INT64)
    with assert_raises(contains="dtype mismatch"):
        _ = read_dataset(spec, False, predicate=Optional(col("key") == 2))
    # An unselected mismatch must not reject a kept file.
    var projected = read_dataset(
        spec, False, columns=Optional(List[String](["value"]))
    )
    assert_equal(projected.height(), 2)


def test_csv_virtual_projection_counts_rows_without_converting_unused_fields() raises:
    var a = _csv("virtual/key=7/a.csv", "id\n1\nnot-an-integer\n3\n")
    var spec = DatasetSpec([a], DatasetOptions(hive_partitioning=True))
    var scan = _CsvScanOptions.defaults()
    scan.schema_overrides["id"] = "int64"
    var result = read_dataset(
        spec,
        False,
        scan=scan,
        columns=Optional(List[String](["key"])),
        batch_size=1,
    )
    assert_equal(result.height(), 3)
    assert_equal(result.columns(), List[String](["key"]))
    for row in range(3):
        assert_equal(result.column("key").get(row).int64(), Int64(7))


def test_parquet_virtual_projection_uses_footer_counts_not_corrupt_payload() raises:
    var path = "tests/fixtures/parquet/late-corrupt.parquet"
    var spec = DatasetSpec([path], DatasetOptions(include_file_paths="path"))
    var batches = _DatasetBatches(
        spec,
        True,
        columns=Optional(List[String](["path"])),
        batch_size=2,
        row_groups=Optional(List[Int]([3, 0, 3])),
    )
    var rows = 0
    while True:
        var next = batches.next()
        if not next:
            break
        var frame = next.take()
        assert_true(frame.height() <= 2)
        assert_equal(frame.width(), 1)
        for row in range(frame.height()):
            assert_equal(frame.column("path").get(row).string(), path)
        rows += frame.height()
    batches.finish()
    assert_equal(rows, 8)
    with assert_raises():
        _ = read_dataset(
            spec,
            True,
            columns=Optional(List[String](["x"])),
            row_groups=Optional(List[Int]([3])),
        )


def test_parquet_physical_projection_row_groups_repetitions_and_empty_target() raises:
    var path = "tests/fixtures/parquet/plain.parquet"
    var spec = DatasetSpec([path, path], DatasetOptions())
    var result = read_dataset(
        spec,
        True,
        columns=Optional(List[String](["x"])),
        row_groups=Optional(List[Int]([2, 0, 2])),
        batch_size=1,
        limit=11,
    )
    assert_equal(result.height(), 11)
    assert_equal(result.columns(), List[String](["x"]))
    assert_equal(result.column("x").get(0).int64(), Int64(100))
    assert_equal(result.column("x").get(3).int64(), Int64(0))
    assert_equal(result.column("x").get(10).int64(), Int64(100))
    var zero = read_dataset(
        spec,
        True,
        columns=Optional(List[String]()),
        row_groups=Optional(List[Int]([0])),
    )
    assert_equal(zero.height(), 8)
    assert_equal(zero.width(), 0)
    var empty = read_dataset(spec, True, limit=0)
    assert_equal(empty.height(), 0)
    assert_equal(empty.column("x").dtype(), DataType.INT64)


def test_legacy_projection_ignores_unselected_name_differences() raises:
    var a = _csv("legacy/a.csv", "id,left\n1,a\n")
    var b = _csv("legacy/b.csv", "id,right\n2,b\n")
    var spec = DatasetSpec([a, b], DatasetOptions())
    var result = read_dataset(
        spec, False, columns=Optional(List[String](["id"]))
    )
    assert_equal(result.height(), 2)
    assert_equal(result.width(), 1)
    with assert_raises(contains="missing column"):
        _ = read_dataset(spec, False)
