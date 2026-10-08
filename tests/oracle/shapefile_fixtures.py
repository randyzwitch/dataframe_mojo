"""Independent shapefile producers and GeoPandas/Shapely checks."""
from datetime import date
import json
from pathlib import Path
import shutil
import struct
import warnings

import geopandas as gpd
import numpy as np
import pyarrow as pa
import shapely
from pyproj import CRS

TYPES = [1, 3, 5, 8, 11, 13, 15, 18, 21, 23, 25, 28]


def header(kind, size):
    return struct.pack(">7i", 9994, 0, 0, 0, 0, 0, size // 2) + struct.pack(
        "<2i8d", 1000, kind, *([0.0] * 8)
    )


def record(kind, parts):
    z, m = kind in [11, 13, 15, 18], kind >= 21
    base = kind - (10 if z else 20 if m else 0)
    points = [p for part in parts for p in part]
    out = struct.pack("<i", kind)
    if base != 1:
        out += struct.pack("<4d", 0, 0, 100, 100)
        if base != 8:
            starts, n = [], 0
            for part in parts:
                starts.append(n)
                n += len(part)
            out += struct.pack("<2i", len(parts), len(points))
            out += struct.pack("<" + "i" * len(parts), *starts)
        else:
            out += struct.pack("<i", len(points))
    for p in points:
        out += struct.pack("<2d", *p[:2])
    if z:
        if base != 1:
            out += struct.pack("<2d", 1, 9)
        out += struct.pack("<" + "d" * len(points), *[p[2] for p in points])
    if m or z:
        if base != 1:
            out += struct.pack("<2d", 10, 20)
        # Z fixtures carry actual measures to exercise XYZM too.
        out += struct.pack("<" + "d" * len(points), *[p[3] for p in points])
    return out


def dbf(path, count, codec="utf-8", deleted=False):
    fields = [
        ("label", "C", 24, 0),
        ("count", "N", 12, 0),
        ("value", "F", 18, 4),
        ("day", "D", 8, 0),
        ("flag", "L", 1, 0),
    ]
    width = 1 + sum(f[2] for f in fields)
    hdr = bytearray(32)
    hdr[0] = 3

    struct.pack_into("<IHH", hdr, 4, count, 33 + 32 * len(fields), width)

    for name, kind, size, scale in fields:
        desc = bytearray(32)
        desc[: len(name)] = name.encode()
        desc[11], desc[16], desc[17] = ord(kind), size, scale
        hdr += desc
    hdr += b"\r"
    for i in range(count):
        null = i == 1
        name = "café" if codec != "utf-8" else "café 雪"
        values = [
            b"" if null else name.encode(codec),
            b"" if null else str(12 + i).encode(),
            b"" if null else b"-3.1250",
            b"00000000" if null else b"20240229",
            b"?" if null else (b"T" if i % 2 == 0 else b"F"),
        ]
        hdr += b"*" if deleted and i == 0 else b" "
        for (_, kind, size, _), val in zip(fields, values):
            hdr += val.ljust(size, b" ") if kind == "C" else val.rjust(
                size, b" "
            )
    path.write_bytes(hdr + b"\x1a")


def write_set(path, kind, records, codec="utf-8", deleted=False, crs=True):
    body, index = bytearray(), bytearray()
    for i, data in enumerate(records):
        index += struct.pack(">2i", (100 + len(body)) // 2, len(data) // 2)
        body += struct.pack(">2i", i + 1, len(data) // 2) + data
    path.write_bytes(header(kind, 100 + len(body)) + body)
    path.with_suffix(".shx").write_bytes(header(kind, 100 + len(index)) + index)
    dbf(path.with_suffix(".dbf"), len(records), codec, deleted)
    if crs:
        path.with_suffix(".prj").write_text(
            CRS.from_epsg(3857).to_wkt("WKT1_ESRI")
        )
    path.with_suffix(".cpg").write_text(codec)


def point(x, y):
    return (x, y, 3.0, 12.0)


def parts_for(base):
    if base == 1:
        return [[point(2, 3)]]
    if base == 8:
        return [[point(2, 3), point(4, 5)]]
    if base == 3:
        return [
            [point(0, 0), point(1, 1)],
            [point(3, 3), point(4, 4), point(5, 4)],
        ]
    # Hole comes first, then two clockwise shells: physical ring order must
    # not be confused with WKB's explicit shell/hole grouping.
    return [
        [point(1, 1), point(2, 1), point(2, 2), point(1, 2), point(1, 1)],
        [point(0, 0), point(0, 4), point(4, 4), point(4, 0), point(0, 0)],
        [
            point(10, 10),
            point(10, 12),
            point(12, 12),
            point(12, 10),
            point(10, 10),
        ],
    ]


def expected_geometry(kind, parts):
    z, m = kind in [11, 13, 15, 18], kind >= 21
    base = kind - (10 if z else 20 if m else 0)
    suffix = " ZM" if z else " M" if m else ""

    def coords(part):
        return ",".join(
            " ".join(map(str, p if z else (p[0], p[1], p[3]) if m else p[:2]))
            for p in part
        )

    if base == 1:
        wkt = "POINT" + suffix + "(" + coords(parts[0]) + ")"
    elif base == 8:
        wkt = "MULTIPOINT" + suffix + "(" + coords(parts[0]) + ")"
    elif base == 3:
        wkt = (
            "MULTILINESTRING"
            + suffix
            + "("
            + ",".join("(" + coords(p) + ")" for p in parts)
            + ")"
        )
    else:
        wkt = (
            "MULTIPOLYGON"
            + suffix
            + "((("
            + coords(parts[1])
            + "),("
            + coords(parts[0])
            + ")),(("
            + coords(parts[2])
            + ")))"
        )
    return shapely.from_wkt(wkt)


def reference(path, codec):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Measured .*")
        return gpd.read_file(path, encoding=codec)


def cases(directory):
    root = Path(directory)
    out = []
    for kind in TYPES:
        base = kind % 10
        parts = parts_for(base)
        path = root / f'type{kind}.shp'
        write_set(
            path,
            kind,
            [record(kind, parts), struct.pack("<i", 0), record(kind, parts)],
        )
        out.append(
            dict(
                path=str(path),
                encoding="utf-8",
                frame=reference(path, "utf-8"),
                geometry=[
                    expected_geometry(kind, parts),
                    None,
                    expected_geometry(kind, parts),
                ],
                unknown=False,
            )
        )
    for kind in [11, 13, 15, 18]:
        parts = parts_for(kind % 10)
        size = sum(map(len, parts)) * 8 + (0 if kind == 11 else 16)
        payload = record(kind, parts)[:-size]
        path = root / f'z_only_{kind}.shp'
        write_set(path, kind, [payload])
        expected = shapely.from_wkb(
            shapely.to_wkb(expected_geometry(kind, parts), output_dimension=3)
        )
        out.append(
            dict(
                path=str(path),
                encoding="utf-8",
                frame=reference(path, "utf-8"),
                geometry=[expected],
                unknown=False,
            )
        )
    for name, codec, deleted, crs in [
        ("latin1", "latin1", False, True),
        ("cp1252", "cp1252", False, True),
        ("deleted", "utf-8", True, True),
        ("unknown", "utf-8", False, False),
        ("no_index", "utf-8", False, True),
    ]:
        path = root / (name + ".shp")
        parts = parts_for(1)
        write_set(
            path,
            1,
            [record(1, parts), struct.pack("<i", 0), record(1, parts)],
            codec,
            deleted,
            crs,
        )
        expected = [
            expected_geometry(1, parts),
            None,
            expected_geometry(1, parts),
        ]
        if deleted:
            expected = expected[1:]
        out.append(
            dict(
                path=str(path),
                encoding=codec,
                frame=reference(path, codec),
                geometry=expected,
                unknown=not crs,
            )
        )
        if name == "no_index":
            path.with_suffix(".shx").unlink()
    # A real third-party producer, in addition to binary-spec fixtures.
    path = root / "geopandas.shp"
    source = gpd.GeoDataFrame(
        {"label": ["one", "two"]},
        geometry=[shapely.from_wkt("POINT Z (1 2 3)"), None],
        crs="EPSG:4326",
    )
    source.to_file(path, encoding="utf-8")
    out.append(
        dict(
            path=str(path),
            encoding="utf-8",
            frame=reference(path, "utf-8"),
            geometry=list(source.geometry),
            unknown=False,
        )
    )
    return out


def check(case, batch):
    reference = case["frame"]
    assert batch.num_rows == len(reference)
    actual = batch.to_pydict()
    if "count" in actual:
        assert batch.schema.field("count").type == pa.int64()
        assert batch.schema.field("value").type == pa.float64()
        assert batch.schema.field("day").type == pa.date32()
        assert batch.schema.field("flag").type == pa.bool_()
    for name in reference.columns:
        if name == "geometry":
            continue
        expect = list(reference[name])
        for i, (a, b) in enumerate(zip(actual[name], expect)):
            if b is None or (
                not isinstance(b, str) and bool(__import__("pandas").isna(b))
            ):
                assert a is None, (name, i, a, b)
            elif isinstance(a, date):
                assert str(a) == str(b)[:10], (name, i, a, b)
            else:
                assert a == b, (name, i, a, b)
    geometries = shapely.from_wkb(actual["geometry"])

    def canonical(g):
        return shapely.to_wkb(shapely.normalize(g), flavor="iso", byte_order=1)

    for a, b, c in zip(geometries, case["geometry"], reference.geometry):
        assert canonical(a) == canonical(b), (
            shapely.to_wkt(a),
            shapely.to_wkt(b),
        )
        # GDAL/Pyogrio currently drops M when constructing GeoPandas objects.
        assert canonical(shapely.force_2d(a)) == canonical(shapely.force_2d(c))
    metadata = json.loads(
        batch.schema.field("geometry").metadata[b"ARROW:extension:metadata"]
    )
    if case["unknown"]:
        assert metadata.get("crs") is None
    else:
        original = Path(case["path"]).with_suffix(".prj").read_text().strip()
        assert metadata["crs"] == original
        assert CRS.from_wkt(metadata["crs"]).equals(
            reference.crs, ignore_axis_order=True
        )
    return True


def malformed(directory):
    root = Path(directory)
    result = []

    def clone(name, suffix, mutate, message, source="type1"):
        path = root / (name + ".shp")
        for ext in [".shp", ".shx", ".dbf", ".prj"]:
            shutil.copyfile(root / (source + ext), path.with_suffix(ext))
        target = path.with_suffix(suffix)
        data = bytearray(target.read_bytes())
        mutate(data)
        target.write_bytes(data)
        result.append(dict(path=str(path), message=message))

    clone(
        "bad_index",
        ".shx",
        lambda b: struct.pack_into(">i", b, 100, 51),
        "SHX offset/length",
    )
    clone(
        "bad_dbf_count",
        ".dbf",
        lambda b: struct.pack_into("<i", b, 4, 1),
        "DBF record count",
    )
    clone(
        "bad_file_length",
        ".shp",
        lambda b: struct.pack_into(">i", b, 24, 50),
        "header length",
    )
    clone(
        "bad_record_number",
        ".shp",
        lambda b: struct.pack_into(">i", b, 100, 9),
        "record 1",
    )
    clone(
        "multipatch",
        ".shp",
        lambda b: struct.pack_into("<i", b, 108, 31),
        "record 1: MultiPatch",
    )
    clone(
        "unknown_type",
        ".shp",
        lambda b: struct.pack_into("<i", b, 108, 99),
        "record 1: Unsupported shape",
    )
    clone(
        "invalid_utf8",
        ".dbf",
        lambda b: b.__setitem__(194, 255),
        "DBF record 1",
    )
    clone(
        "bad_date",
        ".dbf",
        lambda b: b.__setitem__(slice(248, 256), b"20240230"),
        "field",
    )
    clone(
        "bad_xy",
        ".shp",
        lambda b: struct.pack_into("<d", b, 112, float("nan")),
        "record 1: Non-finite",
    )
    clone(
        "bad_parts",
        ".shp",
        lambda b: struct.pack_into("<i", b, 152, 99),
        "record 1: Invalid part offsets",
        source="type3",
    )
    clone(
        "short_index",
        ".shx",
        lambda b: (
            b.__delitem__(slice(100, None)),
            struct.pack_into(">i", b, 24, 50),
        ),
        "SHX record count",
    )
    return result
