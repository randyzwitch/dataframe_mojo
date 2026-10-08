"""External GeoParquet fixtures, CRS semantics, row groups and round-trips."""
from std.testing import TestSuite, assert_equal, assert_true, assert_raises
from dataframe import (
    DataType,
    DataFrame,
    Series,
    from_geojson,
    from_wkb,
    to_wkb,
    read_parquet,
    write_parquet,
    write_geoparquet,
    scan_parquet,
    parquet_backend_version,
    col,
    concat,
)
from dataframe.json_value import JsonValue
from dataframe.geospatial_metadata import (
    apply_geoparquet_metadata,
    _geoparquet_crs_equal,
)

comptime ROOT = "tests/fixtures/geospatial/"
comptime OUTPUT = "/tmp/dataframe_geoparquet_test.parquet"


def test_external_geoparquet() raises:
    var frame = read_parquet(ROOT + "default_crs.parquet")
    var g = frame.column("geom")
    assert_true(g.dtype().is_geometry())
    assert_equal(g.dtype().geometry_metadata(), '{"crs":"OGC:CRS84"}')
    assert_equal(g.null_count(), 1)
    assert_equal(g.get(2).bytes()[0], UInt8(0))  # big-endian bytes preserved
    assert_equal(len(g.get(3).bytes()), 9)  # empty polygon, not null
    assert_equal(frame.column("label").get(2).string(), "a")
    var unknown = read_parquet(ROOT + "unknown_crs.parquet")
    assert_equal(unknown.column("geom").dtype(), DataType.geometry())
    var projected = read_parquet(ROOT + "projjson.parquet")
    var metadata = JsonValue(
        projected.column("geom").dtype().geometry_metadata()
    )
    assert_equal(metadata.get("crs").get("id").get("code").string(), "CRS84")
    assert_equal(metadata.get("edges").string(), "spherical")
    assert_equal(metadata.get("epoch").text, "2021.0")
    assert_equal(
        read_parquet(ROOT + "plain_binary.parquet").column("geom").dtype(),
        DataType.BINARY,
    )


def test_projection_rows_empty_and_lazy() raises:
    var path = ROOT + "default_crs.parquet"
    var source = read_parquet(path)
    var geometry = read_parquet(path, columns=["geom"])
    assert_true(geometry.equals(source.select(["geom"])))
    assert_true(
        read_parquet(path, columns=["id"]).equals(source.select(["id"]))
    )
    assert_true(
        read_parquet(path, row_groups=List[Int]([1])).equals(source.slice(2, 2))
    )
    assert_true(
        read_parquet(path, row_groups=List[Int]()).equals(source.head(0))
    )
    var expected = source.filter(col("id") >= 2).select(["geom"])
    assert_true(
        scan_parquet(path)
        .filter(col("id") >= 2)
        .select(["geom"])
        .collect()
        .equals(expected)
    )
    assert_true(scan_parquet(path).select(["geom"]).collect().equals(geometry))


def test_geoparquet_write_and_plain_parquet_metadata() raises:
    for fixture in [
        "default_crs.parquet",
        "unknown_crs.parquet",
        "projjson.parquet",
    ]:
        var frame = read_parquet(ROOT + fixture)
        for codec in ["uncompressed", "snappy", "zstd"]:
            write_geoparquet(frame, OUTPUT, compression=codec, row_group_size=1)
            assert_true(read_parquet(OUTPUT).equals(frame))
        write_geoparquet(frame.head(0), OUTPUT)
        assert_true(read_parquet(OUTPUT).equals(frame.head(0)))
    var frame = from_geojson('{"type":"Point","coordinates":[1,2]}')
    var other = from_wkb(
        to_wkb(frame.column("geometry")), '{"crs":"EPSG:3857"}'
    ).renamed("other")
    var two = DataFrame([frame.column("geometry"), other.copy()])
    with assert_raises(contains="PROJJSON"):
        write_geoparquet(two, OUTPUT)
    # Ordinary Parquet retains GeoArrow metadata even with a string CRS.
    write_parquet(two, OUTPUT)
    assert_true(read_parquet(OUTPUT).equals(two))
    with assert_raises(contains="primary"):
        write_geoparquet(frame, OUTPUT, primary_column="absent")
    var unknown = from_wkb(to_wkb(other)).renamed("unknown")
    two = DataFrame([frame.column("geometry"), unknown^])
    write_geoparquet(two, OUTPUT, primary_column="unknown")
    assert_true(read_parquet(OUTPUT).equals(two))


def test_invalid_geoparquet() raises:
    for fixture in [
        "conflicting_crs.parquet",
        "unsupported_encoding.parquet",
        "invalid_metadata.parquet",
        "invalid_wkb.parquet",
    ]:
        with assert_raises():
            _ = read_parquet(ROOT + fixture)
    # A projection excluding unsupported geometry remains useful.
    assert_equal(
        read_parquet(
            ROOT + "unsupported_encoding.parquet", columns=["id"]
        ).height(),
        4,
    )


def test_unsupported_m_and_nested_geometry_writes() raises:
    # ISO Point M is valid geometry storage but not GeoParquet 1.x.
    var data = List[UInt8](length=29, fill=0)
    data[0] = 1
    data[1] = 209
    data[2] = 7
    var g = from_wkb(Series.binary("g", [data^]))
    with assert_raises(contains="M coordinates"):
        write_geoparquet(DataFrame([g^]), OUTPUT)
    var source = from_geojson('{"type":"Point","coordinates":[1,2]}')
    var nested = source.pack_struct("nested", ["geometry"])
    with assert_raises(contains="top-level"):
        write_geoparquet(nested, OUTPUT)


def test_geopandas_datum_member_ids() raises:
    # GeoPandas strips only datum-ensemble member IDs in its geo document.
    # Preserve the richer Arrow CRS while accepting this redundant metadata.
    var member = (
        '{"name":"WGS 84 (Transit)","id":{"authority":"EPSG","code":1166}}'
    )
    var bare = '{"name":"WGS 84 (Transit)"}'
    var prefix = '{"type":"GeographicCRS","datum_ensemble":{"members":['
    var suffix = '],"ellipsoid":{"semi_major_axis":6378137}},"id":{"authority":"OGC","code":"CRS84"}}'
    var rich = prefix + member + suffix
    var sparse = prefix + bare + suffix
    for nested in [False, True]:
        var lhs = '{"base_crs":' + rich + "}" if nested else rich
        var rhs = '{"base_crs":' + sparse + "}" if nested else sparse
        assert_true(_geoparquet_crs_equal(JsonValue(lhs), JsonValue(rhs)))
        assert_true(_geoparquet_crs_equal(JsonValue(rhs), JsonValue(lhs)))
    var frame = from_geojson('{"type":"Point","coordinates":[1,2]}')
    var geometry = from_wkb(
        to_wkb(frame.column("geometry")), '{"crs":' + rich + "}"
    )
    frame = DataFrame([geometry^])
    var metadata = (
        '{"version":"1.1.0","primary_column":"geometry","columns":{"geometry":{"encoding":"WKB","geometry_types":["Point"],"crs":'
        + sparse
        + "}}}"
    )
    assert_true(apply_geoparquet_metadata(frame, metadata).equals(frame))
    # IDs that are present in both copies must match. Missing IDs elsewhere
    # and changed member names or ellipsoid parameters must still conflict.
    for conflict in [
        prefix
        + '{"name":"WGS 84 (Transit)","id":{"authority":"EPSG","code":9999}}'
        + suffix,
        prefix + '{"name":"different datum"}' + suffix,
        '{"type":"GeographicCRS","datum_ensemble":{"members":['
        + bare
        + '],"ellipsoid":{"semi_major_axis":1}},"id":{"authority":"OGC","code":"CRS84"}}',
        '{"type":"GeographicCRS","datum_ensemble":{"members":['
        + bare
        + '],"ellipsoid":{"semi_major_axis":6378137}}}',
    ]:
        assert_true(
            not _geoparquet_crs_equal(JsonValue(rich), JsonValue(conflict))
        )
        assert_true(
            not _geoparquet_crs_equal(JsonValue(conflict), JsonValue(rich))
        )
        var bad = (
            '{"version":"1.1.0","primary_column":"geometry","columns":{"geometry":{"encoding":"WKB","geometry_types":["Point"],"crs":'
            + conflict
            + "}}}"
        )
        with assert_raises(contains="Conflicting GeoParquet/GeoArrow metadata"):
            _ = apply_geoparquet_metadata(frame, bad)
    assert_true(
        not _geoparquet_crs_equal(
            JsonValue('{"members":[' + member + "]}"),
            JsonValue('{"members":[' + bare + "]}"),
        )
    )


def main() raises:
    try:
        print("libdfparquet: arrow", parquet_backend_version())
    except e:
        print("skipped:", e)
        return
    TestSuite.discover_tests[__functions_in_module()]().run()
