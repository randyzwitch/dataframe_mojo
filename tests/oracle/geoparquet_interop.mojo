"""Read independent GeoParquet files and produce files for both GIS engines."""
from std.sys import argv
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    DataType,
    read_parquet,
    scan_parquet,
    write_parquet,
    write_geoparquet,
    to_wkb,
    col,
)


def main() raises:
    var args = argv()
    var directory = String(args[1])
    for i in range(2, len(args)):
        var path = directory + "/" + String(args[i])
        print("Checking", String(args[i]))
        var frame = read_parquet(path)
        var expected = read_parquet(path + ".expected")
        var geometry = frame.column("geometry")
        assert_true(geometry.dtype().is_geometry())
        assert_true(to_wkb(geometry).equals(expected.column("geometry")))
        assert_true(
            geometry.geometry_type()
            .renamed("types")
            .equals(expected.column("types"))
        )
        var ids = frame.select(["id"])
        assert_true(read_parquet(path, columns=["id"]).equals(ids))
        assert_true(
            read_parquet(path, columns=["geometry"]).equals(
                frame.select(["geometry"])
            )
        )
        var first_group = read_parquet(path + ".group")
        assert_true(
            read_parquet(path, row_groups=List[Int]([0])).equals(
                frame.head(first_group.height())
            )
        )
        assert_true(
            read_parquet(path, row_groups=List[Int]()).equals(frame.head(0))
        )
        for streaming in [False, True]:
            assert_true(
                scan_parquet(path).collect(streaming=streaming).equals(frame)
            )
            assert_true(
                scan_parquet(path)
                .filter(col("id") >= 3)
                .select(["geometry"])
                .collect(streaming=streaming)
                .equals(frame.filter(col("id") >= 3).select(["geometry"]))
            )
        write_geoparquet(frame, path + ".out", row_group_size=7)
        write_geoparquet(frame.head(0), path + ".empty")
        write_geoparquet(frame.filter(col("id") < 1), path + ".point")
        write_geoparquet(frame.filter(col("id") == 1), path + ".empty_geometry")
        write_geoparquet(frame.tail(1), path + ".null")
        assert_true(read_parquet(path + ".out").equals(frame))

    for name in ["invalid_wkb", "corrupt_pages"]:
        var path = directory + "/" + name + ".parquet"
        with assert_raises():
            _ = read_parquet(path)
        var ids = read_parquet(directory + "/ids.parquet")
        assert_true(read_parquet(path, columns=["id"]).equals(ids))
        for streaming in [False, True]:
            assert_true(
                scan_parquet(path)
                .select(["id"])
                .collect(streaming=streaming)
                .equals(ids)
            )
            assert_true(
                scan_parquet(path)
                .filter(col("id") >= 3)
                .select(["id"])
                .collect(streaming=streaming)
                .equals(ids.filter(col("id") >= 3))
            )
    var partial = directory + "/corrupt_later_groups.parquet"
    with assert_raises():
        _ = read_parquet(partial)
    var first = read_parquet(directory + "/" + String(args[2])).head(7)
    assert_true(read_parquet(partial, row_groups=List[Int]([0])).equals(first))
    for streaming in [False, True]:
        assert_true(
            scan_parquet(partial)
            .filter(col("id") < 7)
            .collect(streaming=streaming)
            .equals(first)
        )
    with assert_raises(contains="Unsupported GeoParquet version: 9.0.0"):
        _ = read_parquet(directory + "/future.parquet")
    var plain = read_parquet(directory + "/plain.parquet")
    assert_equal(plain.column("geometry").dtype(), DataType.BINARY)
    write_parquet(plain, directory + "/plain.out")
    assert_true(read_parquet(directory + "/plain.out").equals(plain))
    print(
        "GeoParquet external bytes/types, projection, row groups and lazy checks passed"
    )
