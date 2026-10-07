"""GeoPandas/GEOS producer and independent coordinate/type/bounds oracle."""
import geopandas as gpd
import numpy as np
import pyarrow as pa
import shapely


def _wkts():
    for suffix in ("", " Z", " M", " ZM"):
        extra = "" if not suffix else " 7" if suffix != " ZM" else " 7 11"
        a, b, c = f"-2 -3{extra}", f"4 -3{extra}", f"4 5{extra}"
        ring = f"{a}, {b}, {c}, {a}"
        bodies = [
            ("POINT", f"({a})"),
            ("LINESTRING", f"({a}, {c})"),
            ("POLYGON", f"(({ring}))"),
            ("MULTIPOINT", f"(({a}), ({c}))"),
            ("MULTILINESTRING", f"(({a}, {c}))"),
            ("MULTIPOLYGON", f"((({ring})))"),
            ("GEOMETRYCOLLECTION", f"(POINT{suffix} ({a}), LINESTRING{suffix} ({a}, {c}))"),
        ]
        for family, body in bodies:
            yield f"{family}{suffix} {body}"
            yield f"{family}{suffix} EMPTY"
    yield "LINESTRING (179 5, -179 -4)"
    yield None


GEOMETRIES = gpd.GeoSeries(list(shapely.from_wkt(list(_wkts()))) + [shapely.GeometryCollection([shapely.Point(8, 9, 10), shapely.LineString([(1, 2), (3, 4)])])], crs="EPSG:4326")
ORIGINAL = gpd.GeoDataFrame({"geometry": GEOMETRIES})


def make(order):
    # Start with GeoPandas' own Arrow schema/extension/CRS producer.
    batch = pa.table(ORIGINAL.to_arrow(index=False)).combine_chunks().to_batches()[0]
    wkb = shapely.to_wkb(GEOMETRIES.array, byte_order=order, flavor="iso", output_dimension=4)
    return pa.RecordBatch.from_arrays([pa.array(wkb, type=pa.binary())], schema=batch.schema)


def check(original, back, bounds, types, total):
    back.validate(full=True)
    assert back.column(0).to_pylist() == original.column(0).to_pylist()
    assert back.schema.field(0).metadata == original.schema.field(0).metadata
    restored = gpd.GeoDataFrame.from_arrow(back)
    assert restored.crs == ORIGINAL.crs
    for a, b in zip(restored.geometry, GEOMETRIES):
        if a is None:
            assert b is None
        else:
            assert a.equals_exact(b, 0), (a, b)
    expected_bounds = GEOMETRIES.bounds.to_numpy()
    actual_bounds = np.array([bounds.column(i).to_numpy(zero_copy_only=False) for i in range(4)]).T
    np.testing.assert_allclose(actual_bounds, expected_bounds, equal_nan=True)
    np.testing.assert_allclose(total, GEOMETRIES.total_bounds)
    expected_types = [None if value is None else value.geom_type + (" ZM" if shapely.has_z(value) and shapely.has_m(value) else " Z" if shapely.has_z(value) else " M" if shapely.has_m(value) else "") for value in GEOMETRIES]
    # GEOS may drop empty collection dimensionality when serializing WKB;
    # derive flags from the actual header rather than its in-memory geometry.
    import struct
    for i, blob in enumerate(original.column(0).to_pylist()):
        if blob is not None:
            code = struct.unpack(("<" if blob[0] else ">") + "I", blob[1:5])[0]
            expected_types[i] = GEOMETRIES.iloc[i].geom_type + ("", " Z", " M", " ZM")[code // 1000]
    assert types.column(0).to_pylist() == expected_types
    return True
