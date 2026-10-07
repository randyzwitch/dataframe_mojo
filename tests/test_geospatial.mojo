"""Geometry storage, GeoJSON ingestion, GeoArrow metadata and preservation."""
from std.testing import TestSuite, assert_equal, assert_true, assert_raises
from dataframe import (
    DataType,
    Series,
    DataFrame,
    Column,
    col,
    concat,
    when,
    from_wkb,
    to_wkb,
    from_geojson,
    read_geojson,
    ArrowArray,
    ArrowSchema,
    export_arrow_series,
    import_arrow_series,
    export_arrow,
    import_arrow,
)
from dataframe.json_value import JsonValue, json_quote
from dataframe.arrow import _metadata_value, _set_metadata


def point() raises -> Series:
    return from_geojson('{"type":"Point","coordinates":[1,2]}').column(
        "geometry"
    )


def sample() raises -> DataFrame:
    return read_geojson("tests/fixtures/geospatial/features.geojson")


def test_json_validation_and_unicode() raises:
    assert_equal(
        JsonValue('{"a":"\\uD83D\\uDE00","b":1}').get("a").string(), "😀"
    )
    assert_equal(
        JsonValue(' { "b": 2, "a": [true, null] } ').canonical(),
        '{"a":[true,null],"b":2}',
    )
    assert_equal(
        JsonValue(json_quote('a\n\t\x01\\"😀')).string(), 'a\n\t\x01\\"😀'
    )
    for text in [
        "",
        "01",
        "1.",
        "1e",
        "[1,]",
        '{"a":1,"a":2}',
        '"\\uD800"',
        '"\\uDC00"',
        '"\\x"',
        "true false",
        "NaN",
        "[",
        '{"x" 1}',
        '"\n"',
    ]:
        with assert_raises():
            _ = JsonValue(text)
    with assert_raises(contains="nesting"):
        _ = JsonValue(String("[") * 66 + "0" + String("]") * 66)


def test_geometry_dtype_and_wkb() raises:
    var g = point()
    assert_true(g.dtype().is_geometry())
    assert_true(not g.dtype().is_binary())
    assert_equal(g.dtype().geometry_metadata(), '{"crs":"OGC:CRS84"}')
    assert_equal(DataType.parse(g.dtype().name()), g.dtype())
    assert_equal(
        DataType.geometry('{"edges":"planar","crs":null}'), DataType.geometry()
    )
    assert_equal(
        DataType.geometry(' { "edges": "spherical", "crs": "X" }'),
        DataType.geometry('{"crs":"X","edges":"spherical"}'),
    )
    var expected = List[UInt8](length=21, fill=0)
    expected[0] = 1
    expected[1] = 1
    expected[11] = 240
    expected[12] = 63
    expected[20] = 64
    assert_equal(g.get(0).bytes(), expected)
    assert_true(from_wkb(to_wkb(g), g.dtype().geometry_metadata()).equals(g))
    assert_equal(g.get(0).dtype(), g.dtype())
    with assert_raises():
        _ = g.cast(DataType.STRING)
    with assert_raises():
        _ = from_wkb(Series("text", Column[String](["abc"])))
    for meta in ["[]", '{"crs":42}', '{"edges":"bad"}']:
        with assert_raises():
            _ = DataType.geometry(meta)
    var invalid: List[List[UInt8]] = [
        [],
        [1],
        [2, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 8, 0, 0, 0],
        [1, 2, 0, 0, 0, 255, 255, 255, 255],
    ]
    for data in invalid:
        with assert_raises():
            _ = from_wkb(Series.binary("g", [data.copy()]))
    expected.append(0)
    with assert_raises(contains="Trailing"):
        _ = from_wkb(Series.binary("g", [expected.copy()]))
    var nulls = from_wkb(Series.binary("g", [[1]], [False]))
    assert_equal(nulls.null_count(), 1)


def test_geojson_attributes_and_empty_null() raises:
    var frame = sample()
    assert_equal(frame.height(), 3)
    assert_equal(frame.column("count").dtype(), DataType.INT64)
    assert_equal(frame.column("active").dtype(), DataType.BOOL)
    assert_equal(frame.column("name").get(0).string(), "east")
    assert_equal(frame.column("feature_id").get(0).string(), '"α"')
    assert_equal(frame.column("feature_id").get(1).string(), "7")
    assert_equal(
        JsonValue(frame.column("nested").get(0).string())
        .get("tags")
        .items()[1]
        .string(),
        "b",
    )
    assert_equal(frame.column("geometry").null_count(), 1)
    assert_true(not frame.column("geometry").get(2).is_null())
    assert_equal(len(frame.column("geometry").get(2).bytes()), 9)
    var empty = from_geojson('{"type":"FeatureCollection","features":[]}')
    assert_equal(empty.height(), 0)
    assert_true(empty.column("geometry").dtype().is_geometry())
    var mixed = from_geojson(
        '{"type":"FeatureCollection","features":[{"type":"Feature","geometry":null,"properties":{"x":1.5,"n":9223372036854775808}},{"type":"Feature","geometry":null,"properties":{"x":2,"n":1}}]}'
    )
    assert_equal(mixed.column("x").dtype(), DataType.FLOAT64)
    assert_equal(mixed.column("n").get(0).string(), "9223372036854775808")


def test_geojson_geometry_families() raises:
    for text in [
        '{"type":"Point","coordinates":[]}',
        '{"type":"Point","coordinates":[1,2,3]}',
        '{"type":"LineString","coordinates":[[1,2],[3,4]]}',
        '{"type":"Polygon","coordinates":[[[0,0],[1,0],[1,1],[0,0]]]}',
        '{"type":"MultiPoint","coordinates":[[1,2],[3,4]]}',
        '{"type":"MultiLineString","coordinates":[[[1,2,3],[4,5,6]]]}',
        '{"type":"MultiPolygon","coordinates":[[[[0,0],[1,0],[1,1],[0,0]]]]}',
        '{"type":"GeometryCollection","geometries":[{"type":"Point","coordinates":[1,2]},{"type":"LineString","coordinates":[]}]}',
    ]:
        var frame = from_geojson(text)
        var g = frame.column("geometry")
        assert_true(
            from_wkb(to_wkb(g), g.dtype().geometry_metadata()).equals(g)
        )
    var collection = from_geojson(
        '{"type":"GeometryCollection","geometries":[{"type":"Point","coordinates":[1,2,3]}]}'
    )
    var collection_bytes = collection.column("geometry").get(0).bytes()
    assert_equal(
        collection_bytes[1], UInt8(239)
    )  # ISO GeometryCollection Z: 1007
    assert_equal(collection_bytes[2], UInt8(3))
    for text in [
        '{"type":"Point","coordinates":[1]}',
        '{"type":"Point","coordinates":[1,2,3,4]}',
        '{"type":"Point","coordinates":["1",2]}',
        '{"type":"Point","coordinates":[1e9999,2]}',
        '{"type":"Point","coordinates":[1,2],"crs":{}}',
        '{"type":"LineString","coordinates":[[1,2],[1,2,3]]}',
        '{"type":"Feature"}',
        '{"type":"FeatureCollection","features":[{"type":"Point","coordinates":[]}]}',
        '{"type":"Feature","geometry":null,"properties":{"geometry":1}}',
    ]:
        with assert_raises():
            _ = from_geojson(text)


def test_frame_operations_preserve_geometry() raises:
    var frame = sample()
    var dtype = frame.column("geometry").dtype()
    var filtered = frame.filter(col("count") > 1).select(["geometry"])
    assert_equal(filtered.column("geometry").dtype(), dtype)
    assert_equal(frame.take([2, 0, 1]).column("geometry").dtype(), dtype)
    assert_equal(frame.head(0).column("geometry").dtype(), dtype)
    assert_equal(
        concat([frame.copy(), frame.copy()]).column("geometry").dtype(), dtype
    )
    assert_equal(
        frame.lazy()
        .filter(col("count") > 1)
        .collect()
        .column("geometry")
        .dtype(),
        dtype,
    )
    assert_equal(Series.full_null("g", dtype, 2).dtype(), dtype)
    var joined = DataFrame([Series("key", Column[Int64]([1, 2]))]).join(
        DataFrame([Series("key", Column[Int64]([1])), point()]),
        on="key",
        how="left",
    )
    assert_equal(joined.column("geometry").dtype(), dtype)
    assert_true(joined.column("geometry").get(1).is_null())
    var diagonal = concat(
        [frame.copy(), DataFrame([Series("extra", Column[Int64]([1]))])],
        how="diagonal",
    )
    assert_equal(diagonal.column("geometry").dtype(), dtype)
    var foreign = to_wkb(point()).with_dtype(
        DataType.geometry('{"crs":"EPSG:3857"}')
    )
    with assert_raises():
        _ = concat([DataFrame([point()]), DataFrame([foreign.copy()])])


def test_arrow_roundtrip_and_metadata_validation() raises:
    var frame = sample()
    var array = ArrowArray()
    var schema = ArrowSchema()
    export_arrow(frame, array, schema)
    assert_true(import_arrow(array, schema).equals(frame))
    var g = frame.column("geometry").slice(1, 2)
    export_arrow_series(g, array, schema)
    assert_equal(
        _metadata_value(schema, "ARROW:extension:name"), "geoarrow.wkb"
    )
    assert_true(import_arrow_series(array, schema).equals(g))
    assert_equal(array.release, 0)
    assert_equal(schema.release, 0)
    for extension in ["geoarrow.point", "geoarrow.wkt"]:
        export_arrow_series(to_wkb(g), array, schema)
        _set_metadata(schema, ["ARROW:extension:name"], [extension])
        with assert_raises(contains="Unsupported GeoArrow"):
            _ = import_arrow_series(array, schema)
        assert_equal(array.release, 0)
        assert_equal(schema.release, 0)


def test_more_storage_paths_and_rejected_text_ops() raises:
    var frame = sample()
    var dtype = frame.column("geometry").dtype()
    var first = frame.select(col("geometry").first()).column("geometry")
    assert_equal(first.dtype(), dtype)
    assert_equal(first.get(0).bytes(), frame.column("geometry").get(0).bytes())
    var filled = frame.with_columns(
        when(col("geometry").is_null())
        .then(col("geometry"))
        .otherwise(col("geometry"))
        .alias("g")
    )
    assert_equal(filled.column("g").dtype(), dtype)
    var chunked = concat(
        [frame.select(["geometry"]), frame.select(["geometry"])]
    )
    assert_equal(chunked.column("geometry").rechunk().dtype(), dtype)
    var binary_chunks = to_wkb(chunked.column("geometry"))
    assert_true(
        from_wkb(binary_chunks, dtype.geometry_metadata()).equals(
            chunked.column("geometry")
        )
    )
    var nested = frame.pack_struct("location", ["geometry"])
    var array = ArrowArray()
    var schema = ArrowSchema()
    export_arrow(nested, array, schema)
    assert_true(import_arrow(array, schema).equals(nested))
    for target in [
        DataType.STRING,
        DataType.BINARY,
        DataType.CATEGORICAL,
        DataType.INT64,
    ]:
        with assert_raises():
            _ = point().cast(target)
    with assert_raises(contains="string expression"):
        _ = frame.select(col("geometry").str().len_bytes())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
