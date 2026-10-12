"""Optional SHP measures, DBF decoding and geometry extension storage edges."""
from std.os import getenv
from std.os.path import exists
from std.ffi import external_call
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    ArrowArray,
    ArrowSchema,
    Column,
    DataFrame,
    DataType,
    Series,
    col,
    concat,
    from_geojson,
    from_wkb,
    to_wkb,
    read_shapefile,
    write_geoparquet,
    write_parquet,
    read_parquet,
    export_arrow,
    import_arrow,
)
from tests.support.spatial_reference import (
    uint,
    number,
    shape_header,
    shape_record,
    shape_expected,
    write_bytes,
    nan_ordinate,
    polygon,
    compound,
    point,
    line,
)


def test_wkb_collection_child_dimensions_and_byte_orders_remain_independent() raises:
    # A collection's dimensional header does not impose its dimension or byte
    # order on each self-describing child geometry.
    for big in [False, True]:
        var raw = compound(
            7,
            [
                point([8, 9, 10], 1, not big),
                line([[1, 2], [3, 4]], 0, big),
            ],
            1,
            big,
        )
        var geometry = from_wkb(
            Series.binary("geometry", [raw.copy(), [255]], [True, False]),
            '{"crs":"OGC:CRS84"}',
        )
        assert_equal(
            geometry.geometry_type().get(0).string(), "GeometryCollection Z"
        )
        assert_equal(to_wkb(geometry).get(0).bytes(), raw)
        assert_true(to_wkb(geometry).get(1).is_null())
        var bounds = geometry.total_bounds().value().copy()
        assert_equal(bounds.xmin, Float64(1))
        assert_equal(bounds.ymin, Float64(2))
        assert_equal(bounds.xmax, Float64(8))
        assert_equal(bounds.ymax, Float64(9))


def single_shape(
    stem: String, kind: Int, record: List[UInt8], label: List[UInt8] = [65]
) raises:
    if exists(stem + ".shx"):
        var index_path = stem + ".shx\0"
        assert_equal(
            external_call["unlink", Int32](index_path.unsafe_ptr()), Int32(0)
        )
        _ = index_path^
    var shp = shape_header(108 + len(record), kind)
    uint(shp, 1, big=True)
    uint(shp, UInt64(len(record) // 2), big=True)
    shp.extend(record.copy())
    write_bytes(stem + ".shp", shp)
    var dbf: List[UInt8] = [3, 126, 10, 11]
    uint(dbf, 1)
    uint(dbf, 65, 2)
    uint(dbf, 9, 2)
    for _ in range(20):
        dbf.append(0)
    dbf.extend("label".as_bytes())
    for _ in range(6):
        dbf.append(0)
    dbf.append(67)
    uint(dbf, 0)
    dbf.append(8)
    for _ in range(15):
        dbf.append(0)
    dbf.append(13)
    dbf.append(32)
    dbf.extend(label.copy())
    for _ in range(8 - len(label)):
        dbf.append(32)
    dbf.append(26)
    write_bytes(stem + ".dbf", dbf)


def test_shapefile_omitted_measures_and_no_data_sentinels_keep_declared_dimensions() raises:
    for base in [1, 3, 5, 8]:
        var full = shape_record(base, 2)
        var count = (
            1 if base == 1 else 2 if base == 8 else 4 if base == 3 else 15
        )
        var omitted = full.copy()
        omitted.resize(
            len(full) - count * 8 - (0 if base == 1 else 16), UInt8(0)
        )
        var stem = (
            getenv("COVERAGE_TEST_TMP") + "/spatial-omitted-m-" + String(base)
        )
        single_shape(stem, base + 20, omitted)
        var frame = read_shapefile(stem + ".shp")
        var typename = (
            String("Point M") if base
            == 1 else String("MultiLineString M") if base
            == 3 else String("MultiPolygon M") if base
            == 5 else String("MultiPoint M")
        )
        assert_equal(
            frame.column("geometry").geometry_type().get(0).string(), typename
        )
        assert_true(not frame.item(0, "geometry").is_null())
        assert_true(Bool(frame.column("geometry").total_bounds()))
        # Explicit no-data measures have the same semantic dimension as absent
        # measured data; compare NaN classifications instead of NaN payloads.
        var sentinel = full.copy()
        sentinel.resize(len(full) - count * 8, UInt8(0))
        for _ in range(count):
            number(sentinel, -1e39)
        single_shape(stem, base + 20, sentinel)
        var measured = read_shapefile(stem + ".shp")
        assert_equal(
            measured.column("geometry").geometry_type().get(0).string(),
            typename,
        )
        if base == 1:
            var a = to_wkb(frame.column("geometry")).get(0).bytes()
            var b = to_wkb(measured.column("geometry")).get(0).bytes()
            assert_equal(len(a), 29)
            assert_equal(len(b), 29)
            assert_true(nan_ordinate(a, 21))
            assert_true(nan_ordinate(b, 21))
    var z = shape_record(1, 1)
    number(z, -1e39)
    var stem = getenv("COVERAGE_TEST_TMP") + "/spatial-no-data-z-measure"
    single_shape(stem, 11, z)
    var frame = read_shapefile(stem + ".shp")
    assert_equal(
        to_wkb(frame.column("geometry")).get(0).bytes(), shape_expected(1, 1)
    )


def test_shapefile_dbf_supported_explicit_encodings_and_utf8_default_decode_exact_values() raises:
    var stem = getenv("COVERAGE_TEST_TMP") + "/spatial-encoded-label"
    var record = shape_record(1)
    single_shape(stem, 1, record, [99, 97, 102, 233])
    assert_equal(
        read_shapefile(stem + ".shp", encoding="latin1")
        .item(0, "label")
        .string(),
        "café",
    )
    single_shape(stem, 1, record, [128])
    with open(stem + ".cpg", "w") as file:
        file.write("1252\n")
    # The public signature documents explicit encoding and UTF-8 by default;
    # a sidecar does not silently replace the caller's selected decoder.
    with assert_raises(contains="not valid UTF-8"):
        _ = read_shapefile(stem + ".shp")
    assert_equal(
        read_shapefile(stem + ".shp", encoding="windows-1252")
        .item(0, "label")
        .string(),
        "€",
    )
    assert_equal(
        read_shapefile(stem + ".shp", encoding="latin1")
        .item(0, "label")
        .string(),
        "\u0080",
    )
    with assert_raises(contains="ASCII"):
        _ = read_shapefile(stem + ".shp", encoding="ascii")
    with assert_raises():
        _ = read_shapefile(stem + ".shp", encoding="utf8")
    single_shape(stem, 1, record, [129])
    with assert_raises(contains="Undefined"):
        _ = read_shapefile(stem + ".shp", encoding="cp1252")


def test_geometry_nested_arrow_and_plain_parquet_keep_intrinsic_metadata_and_reject_text_ops() raises:
    var frame = from_geojson(
        '{"type":"FeatureCollection","features":[{"type":"Feature","properties":{},"geometry":{"type":"Point","coordinates":[1,2]}},{"type":"Feature","properties":{},"geometry":null}]}'
    )
    var nested = frame.pack_struct("location", ["geometry"])
    var array = ArrowArray()
    var schema = ArrowSchema()
    export_arrow(nested, array, schema)
    var back = import_arrow(array, schema)
    assert_equal(
        back.item(0, "location").struct_field("geometry").dtype(),
        frame.column("geometry").dtype(),
    )
    assert_equal(
        back.item(0, "location").struct_field("geometry").bytes(),
        frame.item(0, "geometry").bytes(),
    )
    assert_true(back.item(1, "location").struct_field("geometry").is_null())
    var output = (
        getenv("COVERAGE_TEST_TMP") + "/spatial-nested-rejected.parquet"
    )
    with assert_raises(contains="top-level"):
        write_geoparquet(nested, output)
    for target in [DataType.STRING, DataType.BINARY, DataType.INT64]:
        with assert_raises():
            _ = frame.column("geometry").cast(target)
    with assert_raises(contains="string expression"):
        _ = frame.select(col("geometry").str().len_bytes())
    var foreign = from_wkb(
        to_wkb(frame.column("geometry")), '{"crs":"EPSG:3857"}'
    )
    var plain = DataFrame([foreign^])
    output = getenv("COVERAGE_TEST_TMP") + "/spatial-plain-geoarrow.parquet"
    write_parquet(plain, output)
    var restored = read_parquet(output)
    assert_equal(
        restored.column("geometry").dtype(), plain.column("geometry").dtype()
    )
    assert_equal(
        restored.item(0, "geometry").bytes(), plain.item(0, "geometry").bytes()
    )
    assert_true(restored.item(1, "geometry").is_null())


def test_shapefile_polygon_holes_are_assigned_by_containment_when_hole_records_precede_shells() raises:
    var hole: List[List[Float64]] = [[1, 1], [3, 1], [3, 3], [1, 3], [1, 1]]
    var outer: List[List[Float64]] = [[0, 0], [0, 4], [4, 4], [4, 0], [0, 0]]
    var island: List[List[Float64]] = [
        [10, 10],
        [10, 12],
        [12, 12],
        [12, 10],
        [10, 10],
    ]
    var record = List[UInt8]()
    uint(record, 5)
    for _ in range(4):
        number(record, 0)
    uint(record, 3)
    uint(record, 15)
    for start in [0, 5, 10]:
        uint(record, UInt64(start))
    for ring in [hole.copy(), island.copy(), outer.copy()]:
        for xy in ring:
            number(record, xy[0])
            number(record, xy[1])
    var stem = getenv("COVERAGE_TEST_TMP") + "/spatial-hole-before-shell"
    single_shape(stem, 5, record)
    var frame = read_shapefile(stem + ".shp")
    var expected = compound(
        6, [polygon([island.copy()]), polygon([outer.copy(), hole.copy()])]
    )
    assert_equal(to_wkb(frame.column("geometry")).get(0).bytes(), expected)
    assert_equal(
        frame.column("geometry").geometry_type().get(0).string(), "MultiPolygon"
    )


def test_shapefile_record_parts_dimensions_types_and_live_dbf_dates_validate_before_access() raises:
    var messages: List[String] = [
        "part offsets",
        "part offsets",
        "truncated",
        "MultiPatch",
        "Unsupported shape",
        "differs from SHP header",
        "Non-finite shape Z",
        "Non-finite shape M",
        "Null shape has trailing",
        "cannot parse",
        "SHX record count",
    ]
    for scenario in range(len(messages)):
        var base = 3 if scenario < 2 else 8 if scenario == 2 else 1
        var dim = 1 if scenario == 6 else 2 if scenario == 7 else 0
        var kind = base + (10 if dim == 1 else 20 if dim == 2 else 0)
        var record = shape_record(base, dim)
        var stem = (
            getenv("COVERAGE_TEST_TMP")
            + "/spatial-checked-record-"
            + String(scenario)
        )
        single_shape(stem, kind, record)
        assert_equal(read_shapefile(stem + ".shp").height(), 1)
        if scenario == 0:
            record[44] = 3
        elif scenario == 1:
            record[48] = 0
        elif scenario == 2:
            for byte in range(4):
                record[36 + byte] = 127
        elif scenario == 3 or scenario == 4:
            record[0] = UInt8(31 if scenario == 3 else 99)
        elif scenario == 5:
            kind = 11
        elif scenario == 6 or scenario == 7:
            record.resize(20, UInt8(0))
            number(record, Float64("nan") if scenario == 6 else Float64("inf"))
        elif scenario == 8:
            record = [0, 0, 0, 0, 0, 0]
        single_shape(stem, kind, record)
        if scenario == 9:
            var dbf: List[UInt8]
            with open(stem + ".dbf", "r") as file:
                dbf = file.read_bytes()
            dbf[43] = 68
            var bad_date = String("20240230")
            for byte in range(8):
                dbf[66 + byte] = bad_date.as_bytes()[byte]
            write_bytes(stem + ".dbf", dbf)
        elif scenario == 10:
            write_bytes(stem + ".shx", shape_header(100, kind))
        with assert_raises(contains=messages[scenario]):
            _ = read_shapefile(stem + ".shp")
