"""Independent GeoPandas/DuckDB GeoParquet oracle for #123.

Run with pixi run -e oracle oracle-geoparquet after building libdfparquet.
Python producers/consumers and the Mojo native loader run in separate processes.
"""
import json
from pathlib import Path
import subprocess
import struct
import sys
import tempfile
import warnings

import geopandas as gpd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pyproj import CRS
import shapely

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "oracle"))
from duckdb_spatial_fixtures import connection, install_spatial, wkts


def wkb_type(value):
    if value is None:
        return None
    # GEOS drops Z on empty multi-geometries/collections. Read the ISO type
    # header so valid DuckDB XYZ empties retain their declared dimensionality.
    code = struct.unpack("<I" if value[0] else ">I", value[1:5])[0]
    assert 1 <= code % 1000 <= 7 and code // 1000 in (0, 1), code
    return shapely.from_wkb(value).geom_type + (" Z" if code >= 1000 else "")


def geo(path):
    return json.loads(pq.read_metadata(path).metadata[b"geo"])


def crs(spec):
    value = spec.get("crs", "OGC:CRS84")
    return None if value is None else CRS.from_user_input(value)


def write_variant(table, path, metadata):
    schema_metadata = dict(table.schema.metadata or {})
    schema_metadata[b"geo"] = json.dumps(metadata).encode()
    pq.write_table(
        table.replace_schema_metadata(schema_metadata), path, row_group_size=7
    )


def produce(directory):
    con = connection()
    # GeoParquet 1.x permits XY and XYZ, not M or ZM.
    values = list(wkts())[:28] + ["LINESTRING (179 5, -179 -4)", None]
    cases = []
    for index, coordinate_system in enumerate((None, "OGC:CRS84", "EPSG:3857")):
        frame = gpd.GeoDataFrame(
            {"id": range(len(values))},
            geometry=gpd.GeoSeries.from_wkt(values, crs=coordinate_system),
        )
        for version in ("1.0.0", "1.1.0"):
            path = directory / f"geopandas-{index}-{version}.parquet"
            frame.to_parquet(
                path, index=False, schema_version=version, row_group_size=7
            )
            assert geo(path)["version"] == version
            cases.append(path)
        con.register(
            "wkt_input", pa.table({"id": range(len(values)), "wkt": values})
        )
        expression = "ST_GeomFromText(wkt)"
        params = []
        if coordinate_system is not None:
            expression = f"ST_SetCRS({expression}, ?)"
            params.append(coordinate_system)
        con.execute(
            f"CREATE OR REPLACE TABLE source AS SELECT id, {expression} AS geometry FROM wkt_input",
            params,
        )
        path = directory / f"duckdb-{index}.parquet"
        con.execute(
            "COPY source TO ? (FORMAT PARQUET, GEOPARQUET_VERSION 'V1')",
            [str(path)],
        )
        assert geo(path)["version"] == "1.0.0"
        # DuckDB V1 omits CRS for untagged geometry, declaring CRS84.
        # Unknown CRS (explicit null) is independently covered by GeoPandas.
        assert crs(geo(path)["columns"]["geometry"]) == (
            CRS.from_user_input(coordinate_system or "OGC:CRS84")
        )
        cases.append(path)

    for path in cases:
        table = pq.read_table(path)
        values = table.column("geometry").to_pylist()
        types = [wkb_type(value) for value in values]
        pq.write_table(
            pa.table(
                {
                    "geometry": pa.array(values, type=pa.binary()),
                    "types": pa.array(types, type=pa.string()),
                }
            ),
            str(path) + ".expected",
        )
        pq.write_table(
            pq.ParquetFile(path).read_row_group(0).select(["id"]),
            str(path) + ".group",
        )

    source = cases[0]
    table = pq.read_table(source)
    spec = geo(source)
    # Strip field/schema metadata so this is genuinely plain binary Parquet.
    plain = pa.table({name: table.column(name) for name in table.column_names})
    pq.write_table(plain, directory / "plain.parquet")
    pq.write_table(plain.select(["id"]), directory / "ids.parquet")
    values = table.column("geometry").to_pylist()
    values[0] = b"invalid WKB"
    invalid = pa.table(
        {
            "id": table.column("id"),
            "geometry": pa.array(values, type=pa.binary()),
        }
    )
    write_variant(invalid, directory / "invalid_wkb.parquet", spec)
    # Destroy only the geometry's compressed column chunks. Successful ID
    # projections prove the native reader never decodes those physical pages.
    path = directory / "corrupt_pages.parquet"
    write_variant(plain, path, spec)
    parquet = pq.ParquetFile(path)
    data = bytearray(path.read_bytes())
    later = bytearray(data)
    for group in range(parquet.num_row_groups):
        column = parquet.metadata.row_group(group).column(1)
        offset = column.dictionary_page_offset or column.data_page_offset
        data[offset : offset + column.total_compressed_size] = bytes(
            column.total_compressed_size
        )
        if group > 0:
            later[offset : offset + column.total_compressed_size] = bytes(
                column.total_compressed_size
            )
    path.write_bytes(data)
    (directory / "corrupt_later_groups.parquet").write_bytes(later)
    future = dict(spec, version="9.0.0")
    write_variant(plain, directory / "future.parquet", future)
    return cases


def consume_duckdb(path):
    con = connection()
    dtype = con.execute(
        "DESCRIBE SELECT geometry FROM read_parquet(?)", [str(path)]
    ).fetchone()[1]
    assert dtype.startswith("GEOMETRY"), dtype
    return con.execute(
        """SELECT id, ST_AsWKB(geometry), ST_GeometryType(geometry),
                  ST_ZMFlag(geometry), ST_IsEmpty(geometry), ST_CRS(geometry)
           FROM read_parquet(?) ORDER BY id""",
        [str(path)],
    ).fetchall()


def check(source, output, row_ids):
    before = pq.read_table(source).take(pa.array(row_ids, type=pa.int64()))
    after = pq.read_table(output)
    assert after.to_pydict() == before.to_pydict(), output
    original = geo(source)["columns"]["geometry"]
    metadata = geo(output)
    assert metadata["version"] == "1.1.0"
    assert metadata["primary_column"] == "geometry"
    spec = metadata["columns"]["geometry"]
    assert spec["encoding"] == "WKB"
    assert crs(spec) == crs(original)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        frame = gpd.read_parquet(output)
        baseline = gpd.read_parquet(source).iloc[row_ids]
        assert frame.crs == baseline.crs
        # ISO output preserves dimensionality and catches changes in geometry
        # semantics independently of the raw Parquet binary comparison above.
        assert (
            shapely.to_wkb(frame.geometry, flavor="iso").tolist()
            == shapely.to_wkb(baseline.geometry, flavor="iso").tolist()
        )
        observed = {
            wkb_type(value)
            for value in after.column("geometry").to_pylist()
            if value is not None
        }
        assert set(spec["geometry_types"]) == observed, (output, spec, observed)
        coordinates = shapely.get_coordinates(
            frame.geometry.array, include_z=True
        )
        if len(coordinates):
            z = coordinates[:, 2]
            z = z[np.isfinite(z)]
            low, high = coordinates[:, :2].min(axis=0), coordinates[:, :2].max(
                axis=0
            )
            expected = [*low, z.min(), *high, z.max()] if len(z) else [
                *low,
                *high,
            ]
            np.testing.assert_allclose(spec["bbox"], expected)
        else:
            assert "bbox" not in spec
        expected_rows = consume_duckdb(source)
        assert consume_duckdb(output) == [expected_rows[i] for i in row_ids]


def check_mutation_detection(source, directory):
    """Prove that the consumer assertions reject meaningful output changes."""
    output = Path(str(source) + ".out")
    table = pq.read_table(output)
    rows = list(range(table.num_rows))
    for mutation in ("crs", "types", "bbox", "wkb", "missing_geo"):
        metadata = geo(output)
        spec = metadata["columns"]["geometry"]
        changed = table
        if mutation == "crs":
            spec["crs"] = CRS.from_epsg(3857).to_json_dict()
        elif mutation == "types":
            spec["geometry_types"] = ["Point"]
        elif mutation == "bbox":
            spec["bbox"][0] += 100
        elif mutation == "wkb":
            values = table.column("geometry").to_pylist()
            values[0] = values[2]
            changed = table.set_column(
                table.schema.get_field_index("geometry"),
                table.schema.field("geometry"),
                pa.array(values, type=table.column("geometry").type),
            )
        path = directory / (mutation + ".parquet")
        if mutation == "missing_geo":
            pq.write_table(changed.replace_schema_metadata({}), path)
        else:
            write_variant(changed, path, metadata)
        try:
            check(source, path, rows)
        except (AssertionError, KeyError):
            continue
        raise AssertionError(f"GeoParquet oracle accepted {mutation} mutation")
    print(
        "GeoParquet oracle rejects CRS, type, bbox, WKB and metadata mutations",
        flush=True,
    )


def main():
    install_spatial()
    with tempfile.TemporaryDirectory(prefix="geoparquet-oracle-") as name:
        directory = Path(name)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cases = produce(directory)
        subprocess.run(
            [
                "mojo",
                "run",
                "-I",
                ".",
                "tests/oracle/geoparquet_interop.mojo",
                name,
                *(path.name for path in cases),
            ],
            cwd=ROOT,
            check=True,
        )
        for path in cases:
            count = pq.read_metadata(path).num_rows
            for suffix, rows in (
                ("out", list(range(count))),
                ("empty", []),
                ("point", [0]),
                ("empty_geometry", [1]),
                ("null", [count - 1]),
            ):
                check(path, Path(str(path) + "." + suffix), rows)
            assert (
                pq.read_metadata(str(path) + ".out").num_row_groups
                == (count + 6) // 7
            )
        plain = directory / "plain.out"
        assert b"geo" not in (pq.read_metadata(plain).metadata or {})
        assert (
            pq.read_table(plain).to_pydict()
            == pq.read_table(directory / "plain.parquet").to_pydict()
        )
        check_mutation_detection(cases[0], directory)
        print(
            f"GeoPandas/DuckDB GeoParquet oracle passed: {len(cases)} producer cases, full/empty/filtered outputs, exact WKB/CRS/types/bounds, no consumer warnings",
            flush=True,
        )


if __name__ == "__main__":
    main()
