"""Generate small external geospatial fixtures with PyArrow (oracle environment)."""
import json
import struct
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "geospatial"
ROOT.mkdir(exist_ok=True)


def point(x, y, endian="<"):
    return struct.pack(endian + "BIdd", 1 if endian == "<" else 0, 1, x, y)


rows = [point(1, 2), None, point(-3, 4, ">"), struct.pack("<BII", 1, 3, 0)]
base = pa.table({"id": [0, 1, 2, 3], "label": ["a", "b", "a", "b"], "geom": pa.array(rows, pa.binary())})


def write(name, column, *, arrow_metadata=None, version="1.1.0"):
    metadata = {"version": version, "primary_column": "geom", "columns": {"geom": column}}
    table = base
    if arrow_metadata is not None:
        index = table.schema.get_field_index("geom")
        field = table.schema.field(index).with_metadata({
            "ARROW:extension:name": "geoarrow.wkb",
            "ARROW:extension:metadata": json.dumps(arrow_metadata),
        })
        table = table.set_column(index, field, table.column(index))
    table = table.replace_schema_metadata({"geo": json.dumps(metadata)})
    pq.write_table(table, ROOT / name, row_group_size=2, use_dictionary=["label"])


spec = {"encoding": "WKB", "geometry_types": ["Point", "Polygon"]}
write("default_crs.parquet", spec)
write("unknown_crs.parquet", {**spec, "crs": None}, version="1.0.0")
# Complete PROJJSON from the published OGC:CRS84 definition.
projjson = {
    "$schema": "https://proj.org/schemas/v0.7/projjson.schema.json",
    "type": "GeographicCRS", "name": "WGS 84 (CRS84)",
    "datum": {"type": "GeodeticReferenceFrame", "name": "World Geodetic System 1984",
              "ellipsoid": {"name": "WGS 84", "semi_major_axis": 6378137,
                            "inverse_flattening": 298.257223563}},
    "coordinate_system": {"subtype": "ellipsoidal", "axis": [
        {"name": "Geodetic longitude", "abbreviation": "Lon", "direction": "east", "unit": "degree"},
        {"name": "Geodetic latitude", "abbreviation": "Lat", "direction": "north", "unit": "degree"}]},
    "id": {"authority": "OGC", "code": "CRS84"},
}
write("projjson.parquet", {**spec, "crs": projjson, "edges": "spherical", "epoch": 2021.0})
write("conflicting_crs.parquet", spec, arrow_metadata={"crs": "EPSG:3857"})
write("unsupported_encoding.parquet", {**spec, "encoding": "point"})
write("invalid_metadata.parquet", {**spec, "crs": 42})
# WKB framing is validated on import.
bad = base.set_column(2, "geom", pa.array([b"\x01", None, rows[2], rows[3]], pa.binary()))
bad = bad.replace_schema_metadata({"geo": json.dumps({"version": "1.1.0", "primary_column": "geom", "columns": {"geom": spec}})})
pq.write_table(bad, ROOT / "invalid_wkb.parquet")
# Plain binary must remain plain binary without any geospatial metadata.
pq.write_table(base, ROOT / "plain_binary.parquet")

features = {"type": "FeatureCollection", "features": [
    {"type": "Feature", "id": "α", "geometry": {"type": "Point", "coordinates": [1, 2]},
     "properties": {"name": "east", "count": 2, "active": True, "nested": {"tags": ["a", "b"]}}},
    {"type": "Feature", "id": 7, "geometry": None, "properties": {"name": None, "count": 3, "active": False}},
    {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": []}, "properties": None},
]}
(ROOT / "features.geojson").write_text(json.dumps(features, ensure_ascii=False) + "\n")
