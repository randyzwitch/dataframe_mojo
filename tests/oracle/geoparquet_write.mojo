"""Produce GeoParquet for a separate-process PyArrow consumer."""
from std.sys import argv
from std.testing import assert_true
from dataframe import read_geojson, read_parquet, write_geoparquet


def main() raises:
    var directory = String(argv()[1])
    var frame = read_geojson("tests/fixtures/geospatial/features.geojson")
    write_geoparquet(frame, directory + "/features.parquet", row_group_size=1)
    assert_true(read_parquet(directory + "/features.parquet").equals(frame))
    for fixture in [
        "default_crs.parquet",
        "unknown_crs.parquet",
        "projjson.parquet",
    ]:
        var external = read_parquet("tests/fixtures/geospatial/" + fixture)
        write_geoparquet(
            external, directory + "/" + fixture, primary_column="geom"
        )
