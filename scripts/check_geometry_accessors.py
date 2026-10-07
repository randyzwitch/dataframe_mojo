"""Development-only GeoPandas/GEOS oracle: pixi run -e oracle oracle-geometry."""
import json
from pathlib import Path
import subprocess
import tempfile

import geopandas as gpd
import numpy as np
import pyarrow.parquet as pq
import shapely

from check_duckdb_spatial import main as check_duckdb_spatial

ROOT = Path(__file__).resolve().parents[1]
check_duckdb_spatial()
subprocess.run(
    ["mojo", "run", "-I", ".", "tests/oracle/geometry_accessors.mojo"],
    cwd=ROOT,
    check=True,
)
# Keep the native Parquet loader and embedded Python in separate Mojo processes.
with tempfile.TemporaryDirectory() as directory:
    subprocess.run(
        [
            "mojo",
            "run",
            "-I",
            ".",
            "tests/oracle/geoparquet_write.mojo",
            directory,
        ],
        cwd=ROOT,
        check=True,
    )
    for filename, expected_types in [
        ("features.parquet", ["Point", "Polygon"]),
        ("xyz.parquet", ["Point Z", "LineString Z"]),
    ]:
        path = Path(directory) / filename
        frame = gpd.read_parquet(path)
        assert frame.crs.to_authority() == ("OGC", "CRS84")
        spec = json.loads(pq.read_schema(path).metadata[b"geo"])["columns"][
            "geometry"
        ]
        assert spec["geometry_types"] == expected_types
        if filename == "xyz.parquet":
            xyz = shapely.get_coordinates(frame.geometry.array, include_z=True)
            expected_bounds = np.concatenate([xyz.min(axis=0), xyz.max(axis=0)])
        else:
            expected_bounds = frame.total_bounds
        np.testing.assert_allclose(spec["bbox"], expected_bounds)
    print(
        "GeoPandas GeoParquet reader, observed type inventory and 2D/3D bounds oracle passed"
    )
