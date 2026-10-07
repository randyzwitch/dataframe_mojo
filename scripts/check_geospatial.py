"""GeoArrow/GeoParquet differential checks (development-only PyArrow).

Keep Python and the Parquet writer in separate Mojo processes: std.python
and the native loader currently declare incompatible dlopen return types.
Run with pixi run -e oracle oracle-geospatial.
"""
import json
from pathlib import Path
import struct
import subprocess
import tempfile

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
subprocess.run(["mojo", "run", "-I", ".", "tests/oracle/geospatial.mojo"], cwd=ROOT, check=True)
with tempfile.TemporaryDirectory() as directory:
    subprocess.run(["mojo", "run", "-I", ".", "tests/oracle/geoparquet_write.mojo", directory], cwd=ROOT, check=True)
    output = Path(directory)
    table = pq.read_table(output / "features.parquet")
    table.validate(full=True)
    meta = json.loads(table.schema.metadata[b"geo"])
    assert meta["version"] == "1.1.0"
    assert meta["primary_column"] == "geometry"
    assert meta["columns"]["geometry"]["encoding"] == "WKB"
    assert "crs" not in meta["columns"]["geometry"]  # CRS84 default, not null
    assert table.column("geometry").to_pylist() == [
        struct.pack("<BIdd", 1, 1, 1.0, 2.0), None, struct.pack("<BII", 1, 3, 0)
    ]
    assert table.column("count").to_pylist() == [2, 3, None]
    assert table.column("feature_id").to_pylist() == ['"α"', "7", None]
    assert pq.ParquetFile(output / "features.parquet").metadata.num_row_groups == 3
    for fixture in ["default_crs.parquet", "unknown_crs.parquet", "projjson.parquet"]:
        expected = pq.read_table(ROOT / "tests" / "fixtures" / "geospatial" / fixture)
        actual = pq.read_table(output / fixture)
        assert actual.column_names == expected.column_names
        assert actual.to_pydict() == expected.to_pydict()
        before = json.loads(expected.schema.metadata[b"geo"])["columns"]["geom"]
        after = json.loads(actual.schema.metadata[b"geo"])["columns"]["geom"]
        for key in ["crs", "edges", "epoch", "orientation"]:
            assert after.get(key) == before.get(key), (fixture, key)
        assert ("crs" in after) == ("crs" in before)
    print("GeoParquet PyArrow metadata and byte-for-byte value oracle passed")
