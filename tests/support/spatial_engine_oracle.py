"""Optional installed GIS consumers for actual Mojo GeoParquet writer outputs.

No extension installation or network access is attempted. Missing imports or a
missing cached spatial extension are reported as unavailable capabilities;
once a capability is loaded, every semantic assertion is mandatory.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import warnings


def check_geopandas(directory):
    import geopandas as gpd
    import shapely

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        frame = gpd.read_parquet(directory / "spatial-written.parquet")
        assert frame.geometry.name == "other"
        assert frame.crs is None
        assert frame["shape"].crs.to_authority() == ("OGC", "CRS84")
        assert frame["id"].tolist() == [4, 0, 4]
        assert shapely.get_coordinates(frame["shape"]).tolist() == [
            [5, -7],
            [-2, 3],
            [5, -7],
        ]
        assert (
            shapely.get_coordinates(frame["other"]).tolist() == [[10, 20]] * 3
        )
        empty = gpd.read_parquet(directory / "spatial-written-empty.parquet")
        assert len(empty) == 0
        assert empty.geometry.name == "shape"
        assert empty.crs.to_authority() == ("OGC", "CRS84")
        xyz = gpd.read_parquet(directory / "spatial-written-xyz.parquet")
        assert xyz.geometry.geom_type.tolist() == ["Point"] * 3
        assert xyz.geometry.is_empty.tolist() == [False, False, True]
        assert shapely.get_coordinates(
            xyz.geometry, include_z=True
        ).tolist() == [[-2, 3, 10], [5, -7, 20]]
    print(
        json.dumps(
            {
                "capability": "geopandas",
                "status": "passed",
                "version": gpd.__version__,
                "evidence": "actual public writer outputs recognized as geometry with exact CRS, primary column, coordinates, XYZ and empty schema; warnings are errors",
            }
        )
    )


def check_duckdb(directory, connection):
    import duckdb

    path = str(directory / "spatial-written.parquet")
    description = connection.execute(
        "DESCRIBE SELECT shape, other FROM read_parquet(?)", [path]
    ).fetchall()
    assert all(
        row[1].startswith("GEOMETRY") for row in description
    ), description
    rows = connection.execute(
        "SELECT id, ST_AsText(shape), ST_AsText(other), ST_CRS(shape), ST_CRS(other) FROM read_parquet(?)",
        [path],
    ).fetchall()
    assert rows == [
        (4, "POINT (5 -7)", "POINT (10 20)", "OGC:CRS84", None),
        (0, "POINT (-2 3)", "POINT (10 20)", "OGC:CRS84", None),
        (4, "POINT (5 -7)", "POINT (10 20)", "OGC:CRS84", None),
    ], rows
    assert connection.execute(
        "SELECT count(*) FROM read_parquet(?)",
        [str(directory / "spatial-written-empty.parquet")],
    ).fetchone() == (0,)
    print(
        json.dumps(
            {
                "capability": "duckdb_spatial",
                "status": "passed",
                "version": duckdb.__version__,
                "evidence": "actual public writer outputs consumed as GEOMETRY with exact coordinates, CRS and empty schema",
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-written", required=True, type=Path)
    args = parser.parse_args()
    if (
        importlib.util.find_spec("geopandas") is None
        or importlib.util.find_spec("shapely") is None
    ):
        print(
            json.dumps(
                {
                    "capability": "geopandas",
                    "status": "unavailable",
                    "reason": "optional GeoPandas/Shapely dependency is not installed",
                }
            )
        )
    else:
        check_geopandas(args.check_written)
    if importlib.util.find_spec("duckdb") is None:
        print(
            json.dumps(
                {
                    "capability": "duckdb_spatial",
                    "status": "unavailable",
                    "reason": "optional DuckDB dependency is not installed",
                }
            )
        )
        return
    import duckdb

    connection = duckdb.connect(
        config={
            "autoinstall_known_extensions": False,
            "autoload_known_extensions": False,
        }
    )
    try:
        connection.execute("LOAD spatial")
    except duckdb.IOException as error:
        if "not found" not in str(error):
            raise
        print(
            json.dumps(
                {
                    "capability": "duckdb_spatial",
                    "status": "unavailable",
                    "reason": "no installed compatible Spatial extension; no download attempted",
                    "version": duckdb.__version__,
                }
            )
        )
        return
    check_duckdb(args.check_written, connection)


if __name__ == "__main__":
    main()
