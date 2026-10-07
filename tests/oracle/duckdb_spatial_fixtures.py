"""DuckDB Spatial producer/consumer for the #122 Arrow C interface oracle."""
import json
from pathlib import Path

import duckdb
import pyarrow as pa
from pyproj import CRS

EXTENSIONS = Path(__file__).resolve().parents[2] / "build" / "duckdb_extensions"
CRS_CASES = (None, "EPSG:4326", "EPSG:3857")
_CONNECTION = None


def install_spatial():
    """Explicit prerequisite; installation/load failures must fail the oracle."""
    EXTENSIONS.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(
        config={"extension_directory": str(EXTENSIONS)}
    ) as connection:
        connection.execute(
            "INSTALL spatial FROM 'https://extensions.duckdb.org'"
        )
        connection.execute("LOAD spatial")
    print(
        f"DuckDB {duckdb.__version__}: Spatial extension installed", flush=True
    )


def connection():
    global _CONNECTION
    if _CONNECTION is None:
        _CONNECTION = duckdb.connect(
            config={"extension_directory": str(EXTENSIONS)}
        )
        _CONNECTION.execute("LOAD spatial")
        # Exercise the offset layouts supported by this C adapter, not views.
        _CONNECTION.execute("SET arrow_output_version = '1.0'")
    return _CONNECTION


def wkts():
    # DuckDB constructs and serializes every geometry; no GeoPandas/Shapely
    # producer or manually attached extension metadata participates here.
    for suffix in ("", " Z", " M", " ZM"):
        extra = "" if not suffix else " 7" if suffix != " ZM" else " 7 11"
        a, b, c = f"-2 -3{extra}", f"4 -3{extra}", f"4 5{extra}"
        ring = f"{a}, {b}, {c}, {a}"
        for family, body in (
            ("POINT", f"({a})"),
            ("LINESTRING", f"({a}, {c})"),
            ("POLYGON", f"(({ring}))"),
            ("MULTIPOINT", f"(({a}), ({c}))"),
            ("MULTILINESTRING", f"(({a}, {c}))"),
            ("MULTIPOLYGON", f"((({ring})))"),
            (
                "GEOMETRYCOLLECTION",
                f"(POINT{suffix} ({a}), LINESTRING{suffix} ({a}, {c}))",
            ),
        ):
            yield f"{family}{suffix} {body}"
            yield f"{family}{suffix} EMPTY"
    yield "LINESTRING (179 5, -179 -4)"
    yield None


def make(case):
    con = connection()
    large = bool(case % 2)
    crs = CRS_CASES[case // 2]
    con.execute(f"SET arrow_large_buffer_size = {'true' if large else 'false'}")
    values = list(wkts())
    con.register(
        "wkt_input", pa.table({"id": range(len(values)), "wkt": values})
    )
    expression = "ST_GeomFromText(wkt)"
    params = []
    if crs is not None:
        expression = f"ST_SetCRS({expression}, ?)"
        params.append(crs)
    batch = (
        con.execute(
            f"SELECT {expression} AS geometry FROM wkt_input ORDER BY id",
            params,
        )
        .to_arrow_table()
        .combine_chunks()
        .to_batches()[0]
    )
    field = batch.schema.field(0)
    assert field.type == (pa.large_binary() if large else pa.binary())
    assert field.metadata[b"ARROW:extension:name"] == b"geoarrow.wkb"
    _check_crs(field, crs)
    assert batch.column(0).null_count == 1
    rows = _consume(batch, "producer_coverage", crs)
    families = {
        "POINT",
        "LINESTRING",
        "POLYGON",
        "MULTIPOINT",
        "MULTILINESTRING",
        "MULTIPOLYGON",
        "GEOMETRYCOLLECTION",
    }
    assert {
        (row[1], row[2]) for row in rows if row[0] is not None and not row[3]
    } == {(family, flag) for family in families for flag in range(4)}
    assert sum(row[3] is True for row in rows) == 28
    return batch


def _check_crs(field, expected):
    crs = json.loads(field.metadata[b"ARROW:extension:metadata"]).get("crs")
    if expected is None:
        assert crs is None, crs
    else:
        assert CRS.from_user_input(crs) == CRS.from_user_input(expected), crs


def _consume(batch, name, expected_crs):
    con = connection()
    con.register(name, batch)
    dtype = con.execute(f"DESCRIBE SELECT geometry FROM {name}").fetchone()[1]
    assert dtype.startswith(
        "GEOMETRY"
    ), dtype  # Metadata loss must not pass as BLOB.
    rows = con.execute(
        f"""
        SELECT ST_AsWKB(geometry), ST_GeometryType(geometry),
               ST_ZMFlag(geometry), ST_IsEmpty(geometry), ST_NPoints(geometry),
               ST_CRS(geometry)
        FROM {name}
    """
    ).fetchall()
    assert all(
        row[-1] == expected_crs for row in rows if row[0] is not None
    ), rows
    return rows


def check(original, back, case, rows):
    back.validate(full=True)
    expected = original.take(pa.array(rows, type=pa.int64()))
    assert back.column(0).to_pylist() == expected.column(0).to_pylist()
    assert back.schema.field(0).metadata == original.schema.field(0).metadata
    crs = CRS_CASES[case // 2]
    _check_crs(back.schema.field(0), crs)
    before = _consume(expected, "before_mojo", crs)
    after = _consume(back, "after_mojo", crs)
    assert after == before
    # Require a real DuckDB geometry scan and fresh GeoArrow export as well.
    again = (
        connection().execute("SELECT geometry FROM after_mojo").to_arrow_table()
    )
    assert again.column(0).to_pylist() == expected.column(0).to_pylist()
    _check_crs(again.schema.field(0), crs)
    return True


def check_native(batch):
    rows = _consume(batch, "native_mojo", "OGC:CRS84")
    assert [row[0] for row in rows] == batch.column(0).to_pylist()
    assert [row[3] for row in rows] == [False, True, None]
    result = (
        connection()
        .execute("SELECT geometry FROM native_mojo")
        .to_arrow_table()
    )
    _check_crs(result.schema.field(0), "OGC:CRS84")
    return result.combine_chunks().to_batches()[0]


def check_mutation_detection():
    """Ensure the oracle rejects dropped metadata, changed CRS and changed WKB."""
    source = make(2)
    assert check(source, source, 2, list(range(source.num_rows)))
    altered_crs = dict(source.schema.field(0).metadata)
    altered_crs[b"ARROW:extension:metadata"] = b'{"crs":"EPSG:3857"}'
    values = source.column(0).to_pylist()
    values[0] = values[
        2
    ]  # Replace a point with a valid line, not corrupt bytes.
    for bad in (
        pa.RecordBatch.from_arrays([source.column(0)], names=["geometry"]),
        pa.RecordBatch.from_arrays(
            [source.column(0)],
            schema=pa.schema(
                [source.schema.field(0).with_metadata(altered_crs)]
            ),
        ),
        pa.RecordBatch.from_arrays(
            [pa.array(values, type=source.column(0).type)], schema=source.schema
        ),
    ):
        try:
            check(source, bad, 2, list(range(source.num_rows)))
        except (AssertionError, TypeError, KeyError):
            continue
        raise AssertionError("DuckDB oracle accepted a mutated round trip")
    print("DuckDB oracle rejects metadata, CRS and WKB mutations", flush=True)
