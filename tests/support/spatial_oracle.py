"""Independent PyArrow GeoParquet producer/consumer; runtime semantics are Mojo."""
import argparse
import json
from pathlib import Path
import struct

import pyarrow as pa
import pyarrow.parquet as pq

FIXTURES = Path(__file__).parents[1] / "fixtures/spatial"


def point(x, y):
    return struct.pack("<BI2d", 1, 1, x, y)


def polygon():
    xy = [(0, 0), (0, 4), (4, 4), (4, 0), (0, 0)]
    return struct.pack("<BIII", 1, 3, 1, 5) + b"".join(
        struct.pack("<2d", *p) for p in xy
    )


def geometries():
    return [
        point(-2, 3),
        point(float("nan"), float("nan")),
        polygon(),
        None,
        point(5, -7),
    ]


def generate():
    FIXTURES.mkdir(parents=True, exist_ok=True)
    base = pa.table(
        {
            "id": pa.array(range(5), type=pa.int64()),
            "shape": pa.array(geometries(), type=pa.binary()),
            "other": pa.array([point(10, 20)] * 5, type=pa.binary()),
        }
    )
    for version, name, crs in [
        ("1.0.0", "default", "omitted"),
        ("1.1.0", "unknown", None),
        (
            "1.1.0",
            "object",
            {"type": "GeographicCRS", "name": "independent fixture"},
        ),
    ]:
        column = {
            "encoding": "WKB",
            "geometry_types": ["Point", "Polygon"],
            "bbox": [-999, -999, 999, 999],
        }
        if crs != "omitted":
            column["crs"] = crs
        if name == "object":
            column.update(
                edges="spherical", orientation="counterclockwise", epoch=2020.5
            )
        geo = {
            "version": version,
            "primary_column": "shape",
            "columns": {
                "shape": column,
                "other": {
                    "encoding": "WKB",
                    "geometry_types": ["Point"],
                    "crs": None,
                },
            },
        }
        table = base.replace_schema_metadata({b"geo": json.dumps(geo).encode()})
        pq.write_table(
            table,
            FIXTURES / f"{name}.parquet",
            row_group_size=2,
            compression="NONE",
        )
        assert (
            pq.read_table(FIXTURES / f"{name}.parquet")
            .column("shape")
            .to_pylist()
            == geometries()
        )
    good = pq.read_table(FIXTURES / "default.parquet")
    geo = json.loads(good.schema.metadata[b"geo"])
    for scenario in [
        "unsupported-version",
        "missing-primary",
        "unsupported-encoding",
    ]:
        mutated = json.loads(json.dumps(geo))
        if scenario == "unsupported-version":
            mutated["version"] = "2.0.0"
        elif scenario == "missing-primary":
            mutated["primary_column"] = "absent"
        elif scenario == "unsupported-encoding":
            mutated["columns"]["shape"]["encoding"] = "point"
        pq.write_table(
            good.replace_schema_metadata(
                {b"geo": json.dumps(mutated).encode()}
            ),
            FIXTURES / f"{scenario}.parquet",
            compression="NONE",
        )
    fields = list(good.schema)
    fields[1] = fields[1].with_metadata(
        {
            b"ARROW:extension:name": b"geoarrow.wkb",
            b"ARROW:extension:metadata": b'{"crs":"EPSG:3857"}',
        }
    )
    conflict = pa.Table.from_arrays(
        good.columns, schema=pa.schema(fields, metadata=good.schema.metadata)
    )
    pq.write_table(
        conflict, FIXTURES / "conflicting-crs.parquet", compression="NONE"
    )
    arrays = list(good.columns)
    arrays[1] = pa.chunked_array([pa.array([b"bad WKB"] * 5, type=pa.binary())])
    invalid = pa.Table.from_arrays(arrays, schema=good.schema)
    pq.write_table(
        invalid,
        FIXTURES / "invalid-wkb.parquet",
        compression="NONE",
        row_group_size=2,
    )
    member_crs = {
        "type": "GeographicCRS",
        "name": "fixture",
        "datum_ensemble": {
            "name": "fixture ensemble",
            "members": [
                {"name": "member", "id": {"authority": "EPSG", "code": 123}}
            ],
            "accuracy": "2.0",
        },
    }
    for name in ["datum-id-omitted", "datum-id-conflict"]:
        arrow_crs = json.loads(json.dumps(member_crs))
        member = arrow_crs["datum_ensemble"]["members"][0]
        if name == "datum-id-omitted":
            del member["id"]
        else:
            member["id"]["code"] = 124
        fields = list(good.schema)
        fields[1] = fields[1].with_metadata(
            {
                b"ARROW:extension:name": b"geoarrow.wkb",
                b"ARROW:extension:metadata": json.dumps(
                    {"crs": arrow_crs}
                ).encode(),
            }
        )
        declaration = json.loads(json.dumps(geo))
        declaration["columns"]["shape"]["crs"] = member_crs
        table = pa.Table.from_arrays(
            good.columns,
            schema=pa.schema(
                fields, metadata={b"geo": json.dumps(declaration).encode()}
            ),
        )
        pq.write_table(table, FIXTURES / f"{name}.parquet", compression="NONE")
    print(
        "Verified independent GeoParquet fixtures, versions/CRS/empty/null/mixed geometry and malformed metadata"
    )


def check_written(directory):
    file = pq.ParquetFile(directory / "spatial-written.parquet")
    table = file.read()
    assert table.column_names == ["id", "shape", "other"]
    assert table.column("id").to_pylist() == [4, 0, 4]
    assert table.column("shape").to_pylist() == [
        point(5, -7),
        point(-2, 3),
        point(5, -7),
    ]
    geo = json.loads(file.metadata.metadata[b"geo"])
    assert geo["version"] == "1.1.0"
    assert geo["primary_column"] == "other"
    assert geo["columns"]["shape"]["geometry_types"] == ["Point"]
    assert geo["columns"]["shape"]["bbox"] == [-2, -7, 5, 3]
    assert "crs" not in geo["columns"]["shape"]
    assert geo["columns"]["other"]["bbox"] == [10, 20, 10, 20]
    assert geo["columns"]["other"]["crs"] is None
    assert all(
        file.metadata.row_group(i).num_rows <= 1
        for i in range(file.metadata.num_row_groups)
    )
    for field in table.schema:
        if field.name in ("shape", "other"):
            assert field.metadata[b"ARROW:extension:name"] == b"geoarrow.wkb"
            extension = json.loads(field.metadata[b"ARROW:extension:metadata"])
            assert extension.get("crs") == (
                "OGC:CRS84" if field.name == "shape" else None
            )
    assert all(
        file.metadata.row_group(i).column(c).compression == "ZSTD"
        for i in range(file.metadata.num_row_groups)
        for c in range(file.metadata.num_columns)
    )
    empty = pq.ParquetFile(directory / "spatial-written-empty.parquet")
    empty_geo = json.loads(empty.metadata.metadata[b"geo"])
    assert empty.read().num_rows == 0
    assert "bbox" not in empty_geo["columns"]["shape"]
    assert empty_geo["columns"]["shape"]["geometry_types"] == []
    xyz = pq.ParquetFile(directory / "spatial-written-xyz.parquet")
    xyz_geo = json.loads(xyz.metadata.metadata[b"geo"])
    assert xyz_geo["columns"]["shape"]["bbox"] == [-2, -7, 10, 5, 3, 20]
    assert xyz_geo["columns"]["shape"]["geometry_types"] == ["Point Z"]
    assert xyz.read().column("shape").to_pylist() == [
        struct.pack("<BI3d", 1, 1001, -2, 3, 10),
        struct.pack("<BI3d", 1, 1001, 5, -7, 20),
        struct.pack("<BI3d", 1, 1001, float("nan"), float("nan"), float("nan")),
    ]
    for name in ["spherical", "nonfinite-z"]:
        file = pq.ParquetFile(directory / f"spatial-written-{name}.parquet")
        meta = json.loads(file.metadata.metadata[b"geo"])
        assert "bbox" not in meta["columns"]["shape"]
        assert meta["columns"]["shape"]["geometry_types"] == ["Point Z"]
        if name == "spherical":
            assert meta["columns"]["shape"]["edges"] == "spherical"
    print(
        "Independent PyArrow consumer verified actual public GeoParquet bytes, IDs, fresh type/bounds inventories, CRS, primary geometry, empty schema and row groups"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-written", type=Path)
    args = parser.parse_args()
    if args.check_written:
        check_written(args.check_written)
    else:
        generate()
