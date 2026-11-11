"""Native calls retain short, heap-backed, and newly terminated strings."""
# contract-group: io.native_strings
from std.os import getenv, mkdir, symlink
from std.testing import assert_equal, assert_raises, assert_true
from dataframe import (
    Column,
    DataFrame,
    Series,
    read_ipc,
    read_ipc_schema,
    read_ipc_stream,
    read_parquet,
    scan_ipc,
    scan_parquet,
    write_ipc,
    write_ipc_stream,
    write_parquet,
)
from dataframe.io.file_sources import _matches, resolve_paths
from dataframe.io.parquet import parquet_row_group_statistics


def test_native_paths_and_projections_preserve_utf8_and_string_owners() raises:
    # Appending exercises strings whose C terminator must be installed; the
    # long name and path exercise COW heap storage as well as inline strings.
    var name = String("long_column_")
    for _ in range(8):
        name += "é_"
    var names: List[String] = ["x", name]
    var source = DataFrame(
        [
            Series(names[0], Column[Int64]([4, 7, 9])),
            Series(names[1], Column[String](["café", "", "last"])),
        ]
    )
    var stem = getenv("COVERAGE_TEST_TMP") + "/native-strings-"
    stem += "é-long-path"
    var path = stem + ".parquet"
    write_parquet(source, path, row_group_size=2)
    var selected = read_parquet(path, columns=names)
    assert_equal(selected.columns(), names)
    assert_equal(selected.item(2, name).string(), "last")
    assert_equal(parquet_row_group_statistics(path).height(), 2)
    assert_equal(scan_parquet(path).select(names).collect().height(), 3)
    for stream in [False, True]:
        path = stem + (".stream" if stream else ".arrow")
        if stream:
            write_ipc_stream(source, path)
            selected = read_ipc_stream(path, columns=names)
        else:
            write_ipc(source, path)
            selected = read_ipc(path, columns=names)
            assert_equal(scan_ipc(path).select(names).collect().height(), 3)
        assert_equal(selected.columns(), names)
        assert_equal(selected.item(0, name).string(), "café")
        assert_equal(read_ipc_schema(path, stream=stream)[1].name, name)
    # Native borrowing must not consume or modify the caller's inputs.
    assert_equal(names[0], "x")
    assert_equal(names[1], name)
    assert_equal(source.item(1, name).string(), "")


def test_file_source_native_calls_validate_nuls_and_match_utf8() raises:
    assert_true(_matches("*.csv", "é.csv"))
    assert_true(_matches("a?", "ab"))
    assert_true(not _matches("a?", "abc"))
    for invalid in ["\x00", "valid\x00ignored"]:
        with assert_raises(contains="must not contain NUL"):
            _ = _matches(invalid, "valid")
        with assert_raises(contains="must not contain NUL"):
            _ = _matches("*", invalid)
        with assert_raises(contains="must not contain NUL"):
            _ = resolve_paths([invalid])


def test_recursive_globs_skip_directory_symlinks_and_cycles() raises:
    var root = getenv("COVERAGE_TEST_TMP") + "/symlink-sources"
    mkdir(root)
    mkdir(root + "/real")
    with open(root + "/real/é.csv", "w") as file:
        file.write("x\n1\n")
    symlink("real", root + "/alias")
    symlink("..", root + "/real/cycle")
    symlink("missing", root + "/broken")
    assert_equal(
        resolve_paths([root + "/**/*.csv"]),
        List[String]([root + "/real/é.csv"]),
    )
