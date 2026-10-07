"""Produce GeoParquet for a separate-process PyArrow consumer."""
from std.sys import argv
from std.testing import assert_true
from dataframe import from_geojson, read_geojson, read_parquet, write_geoparquet


def main() raises:
    var directory = String(argv()[1])
    var frame = read_geojson("tests/fixtures/geospatial/features.geojson")
    write_geoparquet(frame, directory + "/features.parquet", row_group_size=1)
    assert_true(read_parquet(directory + "/features.parquet").equals(frame))
    var xyz = from_geojson(
        '{"type":"FeatureCollection","features":[{"type":"Feature","properties":{},"geometry":{"type":"Point","coordinates":[1,2,3]}},{"type":"Feature","properties":{},"geometry":{"type":"LineString","coordinates":[[-4,5,-6],[7,-8,9]]}}]}'
    )
    write_geoparquet(xyz, directory + "/xyz.parquet")
    for fixture in [
        "default_crs.parquet",
        "unknown_crs.parquet",
        "projjson.parquet",
    ]:
        var external = read_parquet("tests/fixtures/geospatial/" + fixture)
        write_geoparquet(
            external, directory + "/" + fixture, primary_column="geom"
        )
