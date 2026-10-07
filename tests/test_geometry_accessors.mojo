"""Coordinate inspection across every ISO WKB family, dimension and byte order."""
from std.memory import bitcast
from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
    assert_false,
    assert_raises,
)
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Series,
    col,
    concat,
    from_wkb,
    from_geojson,
    to_wkb,
)
from dataframe.json_value import JsonValue
from dataframe.geospatial_metadata import geoparquet_metadata


def uint(mut data: List[UInt8], value: Int, little: Bool):
    for i in range(4):
        data.append(UInt8((value >> (8 * (i if little else 3 - i))) & 255))


def number(mut data: List[UInt8], value: Float64, little: Bool):
    var bits = bitcast[DType.uint64](value)
    for i in range(8):
        data.append(UInt8((bits >> UInt64(8 * (i if little else 7 - i))) & 255))


def coordinate(
    mut data: List[UInt8], x: Float64, y: Float64, dim: Int, little: Bool
):
    number(data, x, little)
    number(data, y, little)
    if dim > 0:
        number(data, 7, little)
    if dim == 3:
        number(data, 11, little)


def wkb(
    kind: Int, dim: Int, little: Bool, empty: Bool = False
) raises -> List[UInt8]:
    var data = List[UInt8]()
    data.append(UInt8(1 if little else 0))
    uint(data, kind + dim * 1000, little)
    if kind == 1:
        var x = Float64("nan") if empty else -2.0
        var y = Float64("nan") if empty else -3.0
        coordinate(data, x, y, dim, little)
    elif empty:
        uint(data, 0, little)
    elif kind == 2:
        uint(data, 2, little)
        coordinate(data, -2, -3, dim, little)
        coordinate(data, 4, 5, dim, little)
    elif kind == 3:
        uint(data, 1, little)
        uint(data, 4, little)
        coordinate(data, -2, -3, dim, little)
        coordinate(data, 4, -3, dim, little)
        coordinate(data, 4, 5, dim, little)
        coordinate(data, -2, -3, dim, little)
    else:
        uint(data, 1, little)
        data.extend(wkb(kind - 3 if kind < 7 else 2, dim, not little))
    return data^


def test_all_families_dimensions_and_orders() raises:
    var families = List[String](
        [
            "Point",
            "LineString",
            "Polygon",
            "MultiPoint",
            "MultiLineString",
            "MultiPolygon",
            "GeometryCollection",
        ]
    )
    var suffixes = List[String](["", " Z", " M", " ZM"])
    for kind in range(1, 8):
        for dim in range(4):
            for order in range(2):
                var geo = from_wkb(
                    Series.binary(
                        "g",
                        [
                            wkb(kind, dim, order == 1),
                            wkb(kind, dim, order == 1, True),
                            List[UInt8](),
                        ],
                        [True, True, False],
                    )
                )
                var types = geo.geometry_type()
                assert_equal(
                    types.get(0).string(), families[kind - 1] + suffixes[dim]
                )
                assert_equal(
                    types.get(1).string(), families[kind - 1] + suffixes[dim]
                )
                assert_true(types.get(2).is_null())
                var bounds = geo.bounding_box()
                assert_equal(bounds.column("xmin").get(0).float64(), -2.0)
                assert_equal(bounds.column("ymin").get(0).float64(), -3.0)
                assert_equal(
                    bounds.column("xmax").get(0).float64(),
                    -2.0 if kind == 1 or kind == 4 else 4.0,
                )
                assert_equal(
                    bounds.column("ymax").get(0).float64(),
                    -3.0 if kind == 1 or kind == 4 else 5.0,
                )
                assert_true(bounds.column("xmin").get(1).is_null())
                assert_true(bounds.column("xmin").get(2).is_null())
                assert_true(not geo.get(1).is_null())
                assert_equal(geo.total_bounds().value().xmin, -2.0)
                assert_false(Bool(geo.slice(1, 2).total_bounds()))
                assert_true("0 coordinates" in String(geo.get(1)))


def test_chunked_bounds_crs_and_display() raises:
    var frame = from_geojson(
        '{"type":"FeatureCollection","features":[{"type":"Feature","properties":{"key":1},"geometry":{"type":"Point","coordinates":[179,5]}},{"type":"Feature","properties":{"key":1},"geometry":{"type":"LineString","coordinates":[[179,5],[-179,-4]]}}]}'
    )
    var geo = frame.column("geometry")
    assert_equal(geo.crs(), '"OGC:CRS84"')
    assert_equal(geo.total_bounds().value().xmin, -179.0)
    assert_equal(geo.total_bounds().value().xmax, 179.0)
    var chunks = Series._from_chunks([geo.slice(1, 1), geo.slice(0, 1)])
    assert_equal(chunks.bounding_box().column("xmin").get(1).float64(), 179.0)
    assert_equal(chunks.total_bounds().value().ymin, -4.0)
    assert_equal(chunks.crs(), geo.crs())
    assert_true(
        "LineString (2 coordinates)" in geo.to_string(max_string_length=80)
    )
    assert_equal(String(geo.get(0)), "Point (1 coordinate)")
    var first = (
        frame.group_by("key").agg(col("geometry").first()).column("geometry")
    )
    assert_equal(first.crs(), geo.crs())
    assert_equal(first.get(0).bytes(), geo.get(0).bytes())
    assert_equal(frame.select(["geometry"]).column("geometry").crs(), geo.crs())
    assert_equal(
        frame.filter(col("key") == 1).column("geometry").crs(), geo.crs()
    )
    assert_equal(frame.sort("key").column("geometry").crs(), geo.crs())
    assert_equal(geo.take([1, 0]).crs(), geo.crs())
    assert_equal(
        concat([frame.copy(), frame.copy()]).column("geometry").crs(), geo.crs()
    )
    assert_equal(from_wkb(to_wkb(geo)).crs(), "null")
    assert_equal(
        from_wkb(
            to_wkb(geo), '{"crs":{"name":"test","type":"GeographicCRS"}}'
        ).crs(),
        '{"name":"test","type":"GeographicCRS"}',
    )


def test_malformed_and_nonfinite_coordinates() raises:
    var good = Series.binary("g", [wkb(1, 0, True)])
    var bad = Series.binary("g", [[UInt8(1)]])
    var chunks = Series._from_chunks([good.copy(), good.copy(), bad.copy()])
    with assert_raises(contains="row 2"):
        _ = from_wkb(chunks)
    # Low-level dtype retagging can bypass from_wkb; accessors still validate.
    with assert_raises(contains="row 2"):
        _ = chunks.with_dtype(DataType.geometry()).bounding_box()
    var infinite = wkb(1, 0, True)
    infinite.resize(5, UInt8(0))
    coordinate(infinite, Float64("inf"), 1, 0, True)
    var geo = from_wkb(Series.binary("g", [infinite^]))
    assert_equal(geo.geometry_type().get(0).string(), "Point")
    with assert_raises(contains="Non-finite WKB XY"):
        _ = geo.bounding_box()
    with assert_raises(contains="geometry series"):
        _ = good.geometry_type()
    with assert_raises(contains="geometry series"):
        _ = good.crs()
    with assert_raises(contains="geometry series"):
        _ = good.total_bounds()


def test_geoparquet_observed_types_and_bounds() raises:
    var geo = from_wkb(
        Series.binary(
            "g", [wkb(1, 0, True), wkb(2, 1, False), wkb(1, 0, False, True)]
        )
    )
    var spec = (
        JsonValue(geoparquet_metadata(DataFrame([geo.copy()])))
        .get("columns")
        .get("g")
    )
    assert_equal(
        spec.get("geometry_types").canonical(), '["Point","LineString Z"]'
    )
    assert_equal(spec.get("bbox").items().__len__(), 6)
    assert_equal(Float64(spec.get("bbox").items()[0].text), -2.0)
    assert_equal(Float64(spec.get("bbox").items()[2].text), 7.0)
    assert_equal(Float64(spec.get("bbox").items()[3].text), 4.0)
    var empty = from_wkb(Series.binary("g", [wkb(3, 0, True, True)]))
    var empty_spec = (
        JsonValue(geoparquet_metadata(DataFrame([empty^])))
        .get("columns")
        .get("g")
    )
    assert_false(empty_spec.has("bbox"))
    assert_equal(empty_spec.get("geometry_types").canonical(), '["Polygon"]')
    var spherical = from_wkb(to_wkb(geo), '{"edges":"spherical"}')
    assert_false(
        JsonValue(geoparquet_metadata(DataFrame([spherical^])))
        .get("columns")
        .get("g")
        .has("bbox")
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
