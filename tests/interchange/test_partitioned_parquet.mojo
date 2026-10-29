"""Hive directory writes preserve scalar keys, row order, and atomic roots."""
# contract-group: io.parquet.partitioned
from std.os import getenv, listdir, mkdir
from std.os.path import exists, isdir
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, Series, StringColumn, read_parquet
from dataframe.io.dataset_options import (
    HIVE_NULL,
    hive_encode,
    DatasetOptions,
    DatasetSpec,
)
from dataframe.io.partitioned_parquet import (
    write_partitioned_parquet,
    _DatasetOutput,
)


def _root(name: String) -> String:
    return getenv("COVERAGE_TEST_TMP") + "/partitioned-parquet-" + name


def _source() raises -> DataFrame:
    return DataFrame(
        [
            Series(
                "group/name",
                StringColumn(
                    ["a/b", "a/b", "", HIVE_NULL, "ignored", "é", "a=b", "a/b"],
                    [True, True, True, True, False, True, True, True],
                ),
            ),
            Series("j", Column[Int64]([1, 1, 2, 2, 3, 3, 4, 5])),
            Series("v", Column[Int64]([0, 1, 2, 3, 4, 5, 6, 7])),
        ]
    )


def test_partition_directories_escape_names_values_and_distinguish_literal_null() raises:
    var frame = _source()
    var root = _root("escaped")
    write_partitioned_parquet(
        frame, root, partition_by=["group/name", "j"], row_group_size=1
    )
    var relatives: List[String] = [
        "group%2Fname=a%2Fb/j=1",
        "group%2Fname=/j=2",
        "group%2Fname=" + hive_encode(HIVE_NULL) + "/j=2",
        "group%2Fname=__HIVE_DEFAULT_PARTITION__/j=3",
        "group%2Fname=%C3%A9/j=3",
        "group%2Fname=a%3Db/j=4",
        "group%2Fname=a%2Fb/j=5",
    ]
    var files = List[String]()
    for relative in relatives:
        var file = root + "/" + relative + "/part-00000.parquet"
        assert_true(exists(file))
        files.append(file)
    assert_true(read_parquet(files[0]).equals(frame.take([0, 1])))
    for k in range(1, len(files)):
        assert_true(read_parquet(files[k]).equals(frame.slice(k + 1, 1)))
    # Decode the actual written paths separately from their retained file keys.
    var spec = DatasetSpec(files, DatasetOptions(hive_partitioning=True))
    assert_equal(spec.hive_names, List[String](["group/name", "j"]))
    assert_equal(spec.hive_values[0][0], "a/b")
    assert_equal(spec.hive_values[2][0], HIVE_NULL)
    assert_true(not spec.hive_nulls[2][0])
    assert_true(spec.hive_nulls[3][0])


def test_categorical_partition_values_are_decoded_and_rows_remain_in_input_order() raises:
    var frame = DataFrame(
        [
            Series("k", StringColumn(["red", "blue", "red"])).cast(
                "categorical"
            ),
            Series("v", Column[Int64]([4, 3, 2])),
        ]
    )
    var root = _root("categorical")
    write_partitioned_parquet(frame, root, partition_by=["k"])
    var red = read_parquet(root + "/k=red/part-00000.parquet")
    assert_equal(red.columns(), frame.columns())
    assert_equal(red.column("k").get(0).string(), "red")
    assert_equal(red.column("k").get(1).string(), "red")
    assert_equal(red.item(0, "v").int64(), Int64(4))
    assert_equal(red.item(1, "v").int64(), Int64(2))


def test_empty_partition_dataset_creates_empty_root_and_accepts_empty_destination() raises:
    var source = _source().clear()
    for existing in [False, True]:
        var root = _root("empty-existing" if existing else "empty")
        if existing:
            mkdir(root)
        write_partitioned_parquet(source, root + "/", partition_by=["j"])
        assert_true(isdir(root))
        assert_equal(len(listdir(root)), 0)


def test_invalid_partition_options_do_not_create_output_or_replace_existing_root() raises:
    var frame = _source()
    var root = _root("invalid")
    for keys in [
        List[String](),
        List[String](["missing"]),
        List[String](["j", "j"]),
    ]:
        with assert_raises():
            write_partitioned_parquet(frame, root, partition_by=keys)
        assert_true(not exists(root))
    with assert_raises():
        write_partitioned_parquet(
            frame, root, partition_by=["j"], compression="invalid"
        )
    with assert_raises():
        write_partitioned_parquet(
            frame, root, partition_by=["j"], row_group_size=0
        )
    assert_true(not exists(root))
    mkdir(root)
    with open(root + "/keep", "w") as file:
        file.write("original")
    with assert_raises():
        write_partitioned_parquet(frame, root, partition_by=["j"])
    with open(root + "/keep", "r") as file:
        assert_equal(file.read(), "original")
    assert_equal(len(listdir(root)), 1)


def test_binary_partition_keys_and_nul_field_names_reject_before_creating_root() raises:
    var values: List[List[UInt8]] = [[0, 255]]
    var binary = DataFrame([Series.binary("k", values)])
    var binary_root = _root("binary-rejected")
    with assert_raises():
        write_partitioned_parquet(binary, binary_root, partition_by=["k"])
    assert_true(not exists(binary_root))
    var nul = DataFrame([Series("bad\x00name", Column[Int64]([1]))])
    var nul_root = _root("nul-rejected")
    with assert_raises():
        write_partitioned_parquet(nul, nul_root, partition_by=["bad\x00name"])
    assert_true(not exists(nul_root))


def _conflicting_publish(root: String) raises -> String:
    var output = _DatasetOutput(root)
    var temporary = output.temporary
    var path = output.file_path(["k=a"])
    with open(path, "w") as file:
        file.write("partial")
    mkdir(root)
    with open(root + "/keep", "w") as file:
        file.write("original")
    with assert_raises():
        output.publish()
    return temporary


def test_failed_directory_publication_cleans_only_owned_staging_paths() raises:
    var root = _root("publish-conflict")
    var temporary = _conflicting_publish(root)
    assert_true(not exists(temporary))
    assert_equal(len(listdir(root)), 1)
    with open(root + "/keep", "r") as file:
        assert_equal(file.read(), "original")
