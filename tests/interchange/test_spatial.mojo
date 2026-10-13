"""Supported spatial formats against independent bytes and coordinate meaning."""
from std.os import getenv
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    DataFrame,
    DataType,
    Series,
    from_wkb,
    to_wkb,
    from_geojson,
    read_geojson,
    col,
)
from tests.support.spatial_reference import (
    coordinate,
    header,
    point,
    line,
    polygon,
    compound,
    empty,
    family,
    expected_bounds,
    nan_ordinate,
)


def test_wkb_all_families_dimensions_byte_orders_keep_exact_bytes_nulls_and_xy_bounds() raises:
    var names: List[String] = [
        "Point",
        "LineString",
        "Polygon",
        "MultiPoint",
        "MultiLineString",
        "MultiPolygon",
        "GeometryCollection",
    ]
    var suffixes: List[String] = ["", " Z", " M", " ZM"]
    for big in [False, True]:
        for dim in range(4):
            for kind in range(1, 8):
                var raw = family(kind, dim, big)
                var missing: List[UInt8] = [
                    255,
                    255,
                ]  # Invalid bytes hidden by null validity.
                var bytes: List[List[UInt8]] = [
                    raw.copy(),
                    empty(kind, dim, big),
                    missing^,
                ]
                var shape = from_wkb(
                    Series.binary("shape", bytes, [True, True, False]),
                    '{"crs":"OGC:CRS84"}',
                )
                assert_true(shape.dtype().is_geometry())
                assert_equal(shape.crs(), '"OGC:CRS84"')
                var types = shape.geometry_type()
                assert_equal(
                    types.get(0).string(), names[kind - 1] + suffixes[dim]
                )
                assert_equal(
                    types.get(1).string(), names[kind - 1] + suffixes[dim]
                )
                assert_true(types.get(2).is_null())
                assert_equal(to_wkb(shape).dtype(), DataType.BINARY)
                assert_equal(to_wkb(shape).get(0).bytes(), raw)
                assert_equal(
                    to_wkb(shape).get(1).bytes(), empty(kind, dim, big)
                )
                assert_true(to_wkb(shape).get(2).is_null())
                var bounds = shape.bounding_box()
                var expected = expected_bounds(kind)
                for at in range(4):
                    assert_equal(
                        bounds.item(0, bounds.columns()[at]).float64(),
                        expected[at],
                    )
                    assert_true(bounds.item(1, bounds.columns()[at]).is_null())
                    assert_true(bounds.item(2, bounds.columns()[at]).is_null())
                var total = shape.total_bounds()
                assert_true(Bool(total))
                assert_equal(total.value().xmin, expected[0])
                assert_equal(total.value().ymin, expected[1])
                assert_equal(total.value().xmax, expected[2])
                assert_equal(total.value().ymax, expected[3])
                assert_true(not shape.slice(1, 2).total_bounds())


def test_wkb_metadata_and_row_operations_preserve_crs_bytes_and_reject_incompatibility() raises:
    var dtype = DataType.geometry(' {"edges":"planar", "crs":"EPSG:4326"} ')
    assert_equal(dtype, DataType.geometry('{"crs":"EPSG:4326"}'))
    assert_equal(DataType.geometry(), DataType.geometry('{"crs":null}'))
    var raw: List[List[UInt8]] = [family(1), family(2), family(3)]
    var shape = from_wkb(Series.binary("shape", raw), '{"crs":"EPSG:4326"}')
    var selected = shape.take([2, 0, 2]).renamed("renamed")
    assert_equal(selected.dtype(), dtype)
    assert_equal(selected.crs(), '"EPSG:4326"')
    for row in range(3):
        assert_equal(
            to_wkb(selected).get(row).bytes(), raw[2 if row != 1 else 0]
        )
    var chunked = shape.slice(0, 1).append(shape.slice(1, 2))
    assert_equal(chunked.dtype(), dtype)
    for row in range(3):
        assert_equal(to_wkb(chunked).get(row).bytes(), raw[row])
    with assert_raises():
        _ = shape.append(
            from_wkb(Series.binary("shape", raw), '{"crs":"EPSG:3857"}')
        )
    for metadata in ["[]", '{"crs":17}', '{"edges":"unknown"}', '{"crs":']:
        with assert_raises():
            _ = DataType.geometry(metadata)
    var table = DataFrame([shape^])
    var reordered = (
        table.lazy()
        .select(col("shape").alias("geometry"))
        .filter(col("geometry").is_not_null())
        .collect()
    )
    assert_equal(reordered.column("geometry").dtype(), dtype)
    assert_equal(to_wkb(reordered.column("geometry")).get(1).bytes(), raw[1])


def test_wkb_malformed_structure_and_nonfinite_bounds_report_global_rows() raises:
    var good = family(1)
    var invalid = List[List[UInt8]]()
    invalid.append([])
    invalid.append([2, 1, 0, 0, 0])  # Invalid byte order.
    invalid.append(header(8))  # Unsupported ISO type.
    invalid.append([1, 1, 0, 0, 32, 230, 16, 0, 0])  # EWKB SRID flag.
    for size in range(len(good)):
        var truncated = List[UInt8]()
        for at in range(size):
            truncated.append(good[at])
        invalid.append(truncated^)
    var trailing = good.copy()
    trailing.append(0)
    invalid.append(trailing^)
    invalid.append(compound(4, [family(2)]))  # Multipoint child has line type.
    var nested = good.copy()
    for _ in range(66):
        nested = compound(7, [nested^])
    invalid.append(nested^)
    for malformed in invalid:
        var raised = False
        try:
            _ = from_wkb(
                Series.binary("shape", [good.copy(), malformed.copy()])
            )
        except error:
            assert_true(String(error).find("Invalid WKB at row 1:") >= 0)
            raised = True
        assert_true(raised)
    var infinity = point([Float64("inf"), Float64(1)])
    var poison = from_wkb(Series.binary("shape", [infinity^]))
    assert_equal(poison.geometry_type().get(0).string(), "Point")
    var chunks = from_wkb(Series.binary("shape", [good.copy()])).append(poison)
    var raised = False
    try:
        _ = chunks.bounding_box()
    except error:
        assert_true(String(error).find("Invalid WKB at row 1:") >= 0)
        raised = True
    assert_true(raised)


def test_geojson_properties_identifiers_empty_null_and_xyz_have_independent_wkb_meaning() raises:
    var text = String(
        '{"type":"FeatureCollection","features":[{"type":"Feature","id":1,"geometry":{"type":"Point","coordinates":[-2,3,98]},"properties":{"name":"café","count":2,"measure":1,"flag":true,"nested":{"x":1},"mixed":"1","only_null":null}},{"type":"Feature","id":"1","geometry":{"type":"Polygon","coordinates":[]},"properties":{"name":"","count":null,"measure":2.5,"flag":false,"nested":[1,2],"mixed":1,"only_null":null}},{"type":"Feature","geometry":null,"properties":{}}]}'
    )
    var frame = from_geojson(text, geometry_name="shape", id_name="id")
    assert_equal(frame.height(), 3)
    assert_equal(frame.column("shape").crs(), '"OGC:CRS84"')
    assert_equal(
        to_wkb(frame.column("shape")).get(0).bytes(), point([-2, 3, 98], 1)
    )
    assert_equal(to_wkb(frame.column("shape")).get(1).bytes(), empty(3))
    assert_true(frame.item(2, "shape").is_null())
    assert_equal(frame.item(0, "id").string(), "1")
    assert_equal(frame.item(1, "id").string(), '"1"')
    assert_true(frame.item(2, "id").is_null())
    assert_equal(frame.item(0, "name").string(), "café")
    assert_equal(frame.item(1, "name").string(), "")
    assert_equal(frame.column("count").dtype(), DataType.INT64)
    assert_equal(frame.item(0, "count").int64(), Int64(2))
    assert_equal(frame.column("measure").dtype(), DataType.FLOAT64)
    assert_equal(frame.item(0, "measure").float64(), Float64(1))
    assert_equal(frame.item(1, "measure").float64(), Float64(2.5))
    assert_true(frame.item(0, "flag").bool())
    assert_true(not frame.item(1, "flag").bool())
    assert_equal(frame.item(0, "nested").string(), '{"x":1}')
    assert_equal(frame.item(1, "nested").string(), "[1,2]")
    assert_equal(frame.item(0, "mixed").string(), '"1"')
    assert_equal(frame.item(1, "mixed").string(), "1")
    assert_equal(frame.column("only_null").dtype(), DataType.STRING)
    for name in [
        "name",
        "count",
        "measure",
        "flag",
        "nested",
        "mixed",
        "only_null",
    ]:
        assert_true(frame.item(2, name).is_null())
    var path = getenv("COVERAGE_TEST_TMP") + "/spatial-properties.geojson"
    with open(path, "w") as file:
        file.write(text)
    var from_file = read_geojson(path, geometry_name="shape", id_name="id")
    for row in range(3):
        for name in frame.columns():
            assert_equal(from_file.item(row, name), frame.item(row, name))
    var bare = from_geojson(
        '{"type":"GeometryCollection","geometries":[{"type":"Point","coordinates":[-2,3]},{"type":"LineString","coordinates":[[-2,3],[5,-7]]}]}'
    )
    assert_equal(
        to_wkb(bare.column("geometry")).get(0).bytes(),
        compound(7, [family(1), family(2)]),
    )


def test_geojson_malformed_records_crs_coordinate_dimensions_and_name_collisions_reject() raises:
    var invalid: List[String] = [
        "{",
        '{"type":"FeatureCollection","features":null}',
        '{"type":"Point","coordinates":[1]}',
        '{"type":"Point","coordinates":[1,2,3,4]}',
        '{"type":"LineString","coordinates":[[1,2],[1,2,3]]}',
        '{"type":"Polygon","coordinates":[[[0,0],[1,0],[1,1],[0,1]]]}',
        '{"type":"Polygon","coordinates":[[[0,0],[1,0],[0,0]]]}',
        '{"type":"Point","coordinates":[1e999,2]}',
        '{"type":"Point","coordinates":[1,2],"crs":null}',
        '{"type":"Feature","geometry":null,"properties":{"geometry":1}}',
        '{"type":"Feature","id":1,"geometry":null,"properties":{"feature_id":2}}',
    ]
    for text in invalid:
        with assert_raises():
            _ = from_geojson(text)
    with assert_raises():
        _ = from_geojson(
            '{"type":"Point","coordinates":[1,2]}',
            geometry_name="same",
            id_name="same",
        )


def test_shapefile_all_shape_families_dimensions_dbf_deletion_nulls_and_sidecars() raises:
    from dataframe import read_shapefile
    from tests.support.spatial_reference import write_shape, shape_expected

    var suffixes: List[String] = ["", " Z", " M", " ZM"]
    for base in [1, 3, 5, 8]:
        for dim in range(4):
            var stem = (
                getenv("COVERAGE_TEST_TMP")
                + "/spatial-shape-"
                + String(base)
                + "-"
                + String(dim)
            )
            write_shape(stem, base, dim)
            var frame = read_shapefile(stem + ".shp", geometry_name="shape")
            assert_equal(
                frame.columns(),
                List[String](
                    ["name", "count", "score", "flag", "day", "shape"]
                ),
            )
            assert_equal(frame.height(), 2)
            assert_equal(frame.item(0, "name").string(), "café")
            assert_equal(frame.item(0, "count").int64(), Int64(7))
            assert_equal(frame.item(0, "score").float64(), Float64(1.25))
            assert_true(frame.item(0, "flag").bool())
            assert_equal(frame.column("day").dtype(), DataType.DATE)
            assert_equal(frame.item(0, "day").to_physical(), Int64(19782))
            for name in frame.columns():
                assert_true(frame.item(1, name).is_null())
            var expected = shape_expected(base, dim)
            assert_equal(to_wkb(frame.column("shape")).get(0).bytes(), expected)
            var kind = String(
                "Point" if base
                == 1 else "MultiLineString" if base
                == 3 else "MultiPolygon" if base
                == 5 else "MultiPoint"
            )
            assert_equal(
                frame.column("shape").geometry_type().get(0).string(),
                kind + suffixes[dim],
            )
            assert_equal(
                frame.column("shape").crs(),
                '"LOCAL_CS[\\"fixture coordinates\\",UNIT[\\"metre\\",1]]"',
            )
            var bounds = frame.column("shape").bounding_box()
            var expected_xy = expected_bounds(
                1 if base == 1 else 5 if base == 3 else 4
            )
            if base == 5:
                expected_xy = [0, 0, 12, 12]
            for at in range(4):
                assert_equal(
                    bounds.item(0, bounds.columns()[at]).float64(),
                    expected_xy[at],
                )
                assert_true(bounds.item(1, bounds.columns()[at]).is_null())
    var no_sidecars = getenv("COVERAGE_TEST_TMP") + "/spatial-shape-no-sidecars"
    write_shape(no_sidecars, 1, with_crs=False, with_index=False)
    var unknown = read_shapefile(no_sidecars + ".shp")
    assert_equal(unknown.column("geometry").crs(), "null")
    assert_equal(to_wkb(unknown.column("geometry")).get(0).bytes(), family(1))


def test_shapefile_malformed_shp_shx_dbf_encoding_and_record_counts_reject() raises:
    from dataframe import read_shapefile
    from tests.support.spatial_reference import write_shape, write_bytes

    for scenario in range(8):
        var stem = (
            getenv("COVERAGE_TEST_TMP")
            + "/spatial-shape-invalid-"
            + String(scenario)
        )
        write_shape(stem, 1)
        assert_equal(read_shapefile(stem + ".shp").height(), 2)
        var component = String(
            ".shp" if scenario <= 2 else ".shx" if scenario == 3 else ".dbf"
        )
        var bytes: List[UInt8]
        with open(stem + component, "r") as file:
            bytes = file.read_bytes()
        if scenario == 0:
            bytes[0] = 255  # Incorrect magic.
        elif scenario == 1:
            bytes[103] = 2  # Wrong first record number.
        elif scenario == 2:
            bytes[104] = 127  # Content length outside actual file.
        elif scenario == 3:
            bytes[103] += 1  # SHX first offset disagrees with SHP.
        elif scenario == 4:
            bytes[4] = 2  # DBF and SHP record counts disagree.
        elif scenario == 5:
            bytes[193] = 33  # Invalid deletion marker.
        elif scenario == 6:
            bytes[194] = 255  # Invalid UTF-8 in live character field.
        else:
            bytes[206] = 88  # Invalid live numeric field.
        write_bytes(stem + component, bytes)
        var diagnostics: List[String] = [
            "Invalid SHP header",
            "SHP record 1:",
            "SHP record 1:",
            "SHP record 1:",
            "DBF record count differs",
            "Invalid DBF deletion marker",
            "DBF record 1",
            "DBF record 1",
        ]
        var rejected = False
        try:
            _ = read_shapefile(stem + ".shp")
        except error:
            assert_true(String(error).find(diagnostics[scenario]) >= 0)
            rejected = True
        assert_true(rejected)
    var stem = getenv("COVERAGE_TEST_TMP") + "/spatial-shape-encoding"
    write_shape(stem, 1)
    with assert_raises():
        _ = read_shapefile(stem + ".shp", encoding="unsupported")
    with assert_raises():
        _ = read_shapefile(stem + ".shp", geometry_name="name")
    with assert_raises():
        _ = read_shapefile(stem + ".dbf")


def spatial_file(name: String) -> String:
    return "tests/fixtures/spatial/" + name + ".parquet"


def test_geoparquet_independent_versions_crs_projection_group_order_and_lazy_values() raises:
    from dataframe import read_parquet, scan_parquet

    for source in ["default", "unknown", "object"]:
        var frame = read_parquet(spatial_file(source))
        assert_equal(frame.height(), 5)
        var metadata = String(
            '{"crs":"OGC:CRS84"}' if source
            == "default" else '{"crs":null}' if source
            == "unknown" else '{"crs":{"type":"GeographicCRS","name":"independent fixture"},"edges":"spherical","orientation":"counterclockwise","epoch":2020.5}'
        )
        assert_equal(frame.column("shape").dtype(), DataType.geometry(metadata))
        assert_equal(frame.column("other").crs(), "null")
        var raw = to_wkb(frame.column("shape"))
        assert_equal(raw.get(0).bytes(), family(1))
        assert_equal(raw.get(1).bytes(), empty(1))
        var ring: List[List[Float64]] = [[0, 0], [0, 4], [4, 4], [4, 0], [0, 0]]
        assert_equal(raw.get(2).bytes(), polygon([ring^]))
        assert_true(raw.get(3).is_null())
        assert_equal(raw.get(4).bytes(), point([5, -7]))
        var selected = read_parquet(
            spatial_file(source),
            columns=["shape", "id"],
            row_groups=List[Int]([2, 0, 2]),
        )
        assert_equal(selected.columns(), List[String](["shape", "id"]))
        var expected_ids: List[Int64] = [4, 0, 1, 4]
        for row in range(4):
            assert_equal(selected.item(row, "id").int64(), expected_ids[row])
            assert_equal(
                to_wkb(selected.column("shape")).get(row).bytes(),
                raw.get(Int(expected_ids[row])).bytes(),
            )
        for streaming in [False, True]:
            var lazy = (
                scan_parquet(spatial_file(source))
                .filter(col("id") >= 3)
                .select(["id", "shape"])
                .collect(streaming=streaming, batch_size=1, engine="cpu")
            )
            assert_equal(lazy.height(), 2)
            assert_equal(
                lazy.column("shape").dtype(), frame.column("shape").dtype()
            )
            assert_equal(lazy.item(0, "id").int64(), Int64(3))
            assert_true(lazy.item(0, "shape").is_null())
            assert_equal(lazy.item(1, "id").int64(), Int64(4))
            assert_equal(
                to_wkb(lazy.column("shape")).get(1).bytes(), point([5, -7])
            )


def test_geoparquet_invalid_metadata_conflicting_crs_and_selected_invalid_wkb_reject() raises:
    from dataframe import read_parquet, scan_parquet

    for source in [
        "unsupported-version",
        "missing-primary",
        "unsupported-encoding",
        "conflicting-crs",
        "invalid-wkb",
    ]:
        with assert_raises():
            _ = read_parquet(spatial_file(source))
    var projected = read_parquet(spatial_file("invalid-wkb"), columns=["id"])
    assert_equal(projected.columns(), List[String](["id"]))
    assert_equal(projected.height(), 5)
    for row in range(5):
        assert_equal(projected.item(row, "id").int64(), Int64(row))
    for streaming in [False, True]:
        var projected_lazy = (
            scan_parquet(spatial_file("invalid-wkb"))
            .select(["id"])
            .collect(streaming=streaming, batch_size=1, engine="cpu")
        )
        assert_equal(projected_lazy.height(), 5)
        for row in range(5):
            assert_equal(projected_lazy.item(row, "id").int64(), Int64(row))


def test_geoparquet_writer_recomputes_filtered_bounds_types_primary_and_empty_schema_for_external_consumer() raises:
    from dataframe import read_parquet, write_geoparquet

    var source = read_parquet(spatial_file("default"))
    var selected = source.take([4, 0, 4])
    var path = getenv("COVERAGE_TEST_TMP") + "/spatial-written.parquet"
    write_geoparquet(selected, path, primary_column="other", row_group_size=1)
    var restored = read_parquet(path)
    assert_equal(restored.columns(), selected.columns())
    assert_equal(restored.height(), 3)
    for row in range(3):
        assert_equal(
            restored.item(row, "id").int64(), selected.item(row, "id").int64()
        )
        for name in ["shape", "other"]:
            assert_equal(
                restored.column(name).dtype(), selected.column(name).dtype()
            )
            assert_equal(
                to_wkb(restored.column(name)).get(row).bytes(),
                to_wkb(selected.column(name)).get(row).bytes(),
            )
    write_geoparquet(
        source.head(0),
        getenv("COVERAGE_TEST_TMP") + "/spatial-written-empty.parquet",
    )
    var empty_frame = read_parquet(
        getenv("COVERAGE_TEST_TMP") + "/spatial-written-empty.parquet"
    )
    assert_equal(empty_frame.height(), 0)
    assert_equal(empty_frame.columns(), source.columns())
    assert_equal(
        empty_frame.column("shape").dtype(), source.column("shape").dtype()
    )
    # spatial_oracle.py --check-written independently consumes these outputs
    # and validates exact WKB, fresh file metadata, CRS, codecs and group sizes.


def test_geoparquet_writer_rejects_m_ordinates_unrepresentable_crs_and_invalid_primary() raises:
    from dataframe import read_parquet, write_geoparquet

    var path = getenv("COVERAGE_TEST_TMP") + "/spatial-invalid-write.parquet"
    for dim in [2, 3]:
        var shape = from_wkb(Series.binary("shape", [family(1, dim)]))
        with assert_raises():
            write_geoparquet(DataFrame([shape^]), path)
    var source = read_parquet(spatial_file("default"))
    with assert_raises():
        write_geoparquet(source, path, primary_column="id")
    with assert_raises():
        write_geoparquet(source, path, primary_column="absent")
    var unsupported = from_wkb(
        Series.binary("shape", [family(1)]), '{"crs":"EPSG:4326"}'
    )
    with assert_raises():
        write_geoparquet(DataFrame([unsupported^]), path)


def test_geojson_all_families_and_empty_shapes_keep_exact_coordinate_order() raises:
    var inputs: List[String] = [
        '{"type":"Point","coordinates":[-2,3]}',
        '{"type":"LineString","coordinates":[[-2,3],[5,-7]]}',
        '{"type":"Polygon","coordinates":[[[-2,-7],[5,-7],[5,3],[-2,3],[-2,-7]]]}',
        '{"type":"MultiPoint","coordinates":[[-2,3],[5,-7]]}',
        '{"type":"MultiLineString","coordinates":[[[-2,3],[5,-7]],[[10,11],[-13,17]]]}',
        '{"type":"MultiPolygon","coordinates":[[[[-2,-7],[5,-7],[5,3],[-2,3],[-2,-7]]],[[[10,11],[12,11],[10,17],[10,11]]]]}',
        '{"type":"GeometryCollection","geometries":[{"type":"Point","coordinates":[-2,3]},{"type":"LineString","coordinates":[[-2,3],[5,-7]]},{"type":"Polygon","coordinates":[]}]}',
    ]
    var names: List[String] = [
        "Point",
        "LineString",
        "Polygon",
        "MultiPoint",
        "MultiLineString",
        "MultiPolygon",
        "GeometryCollection",
    ]
    for kind in range(1, 8):
        var frame = from_geojson(inputs[kind - 1])
        assert_equal(
            to_wkb(frame.column("geometry")).get(0).bytes(), family(kind)
        )
        assert_equal(
            frame.column("geometry").geometry_type().get(0).string(),
            names[kind - 1],
        )
        var text = (
            '{"type":"'
            + names[kind - 1]
            + '","'
            + ("geometries" if kind == 7 else "coordinates")
            + '":[]}'
        )
        var empty_frame = from_geojson(text)
        assert_true(not empty_frame.item(0, "geometry").is_null())
        var empty_bytes = to_wkb(empty_frame.column("geometry")).get(0).bytes()
        if kind == 1:
            # ISO POINT EMPTY accepts any NaN payload; bitwise canonical NaN
            # equality is a representation restriction absent from the contract.
            assert_equal(len(empty_bytes), 21)
            for at in range(5):
                assert_equal(empty_bytes[at], header(1)[at])
            assert_true(nan_ordinate(empty_bytes, 5))
            assert_true(nan_ordinate(empty_bytes, 13))
        else:
            assert_equal(empty_bytes, empty(kind))
        assert_true(not empty_frame.column("geometry").total_bounds())
    var xyz = from_geojson(
        '{"type":"MultiPoint","coordinates":[[-2,3,98],[5,-7,105]]}'
    )
    assert_equal(to_wkb(xyz.column("geometry")).get(0).bytes(), family(4, 1))


def test_geoparquet_writer_xyz_spherical_and_nonfinite_z_bounds_match_external_metadata() raises:
    from dataframe import write_geoparquet, read_parquet

    var bytes: List[List[UInt8]] = [
        point([-2, 3, 10], 1),
        point([5, -7, 20], 1),
        empty(1, 1),
    ]
    var shape = from_wkb(Series.binary("shape", bytes), '{"crs":"OGC:CRS84"}')
    var path = getenv("COVERAGE_TEST_TMP") + "/spatial-written-xyz.parquet"
    write_geoparquet(DataFrame([shape.copy()]), path)
    var restored = read_parquet(path)
    for row in range(3):
        assert_equal(
            to_wkb(restored.column("shape")).get(row).bytes(), bytes[row]
        )
    var spherical = from_wkb(
        Series.binary("shape", bytes), '{"crs":"OGC:CRS84","edges":"spherical"}'
    )
    write_geoparquet(
        DataFrame([spherical^]),
        getenv("COVERAGE_TEST_TMP") + "/spatial-written-spherical.parquet",
    )
    var nonfinite = from_wkb(
        Series.binary("shape", [point([1, 2, Float64("inf")], 1)]),
        '{"crs":"OGC:CRS84"}',
    )
    write_geoparquet(
        DataFrame([nonfinite^]),
        getenv("COVERAGE_TEST_TMP") + "/spatial-written-nonfinite-z.parquet",
    )


def test_geojson_integer_precision_promotion_and_empty_collection_schema() raises:
    var frame = from_geojson(
        '{"type":"FeatureCollection","features":[{"type":"Feature","geometry":null,"properties":{"safe":9007199254740992,"unsafe":9007199254740993,"huge":18446744073709551616}},{"type":"Feature","geometry":null,"properties":{"safe":2.5,"unsafe":2.5,"huge":42}}]}'
    )
    assert_equal(frame.column("safe").dtype(), DataType.FLOAT64)
    assert_equal(frame.item(0, "safe").float64(), Float64(9007199254740992))
    assert_equal(frame.item(1, "safe").float64(), Float64(2.5))
    assert_equal(frame.column("unsafe").dtype(), DataType.STRING)
    assert_equal(frame.item(0, "unsafe").string(), "9007199254740993")
    assert_equal(frame.item(1, "unsafe").string(), "2.5")
    assert_equal(frame.column("huge").dtype(), DataType.STRING)
    assert_equal(frame.item(0, "huge").string(), "18446744073709551616")
    assert_equal(frame.item(1, "huge").string(), "42")
    var empty_frame = from_geojson('{"type":"FeatureCollection","features":[]}')
    assert_equal(empty_frame.height(), 0)
    assert_equal(empty_frame.columns(), List[String](["geometry"]))
    assert_equal(
        empty_frame.column("geometry").dtype(),
        DataType.geometry('{"crs":"OGC:CRS84"}'),
    )
    assert_true(not empty_frame.column("geometry").total_bounds())


def test_geoparquet_optional_datum_member_ids_accept_omission_and_reject_conflicts() raises:
    from dataframe import read_parquet

    var accepted = read_parquet(spatial_file("datum-id-omitted"))
    var expected = DataType.geometry(
        '{"crs":{"type":"GeographicCRS","name":"fixture","datum_ensemble":{"name":"fixture ensemble","members":[{"name":"member"}],"accuracy":"2.0"}}}'
    )
    assert_equal(accepted.column("shape").dtype(), expected)
    assert_equal(to_wkb(accepted.column("shape")).get(0).bytes(), family(1))
    var rejected = False
    try:
        _ = read_parquet(spatial_file("datum-id-conflict"))
    except error:
        if String(error).find("Conflicting GeoParquet/GeoArrow metadata:") < 0:
            raise error^
        rejected = True
    assert_true(rejected)


def test_geometry_key_identity_and_join_output_preserve_bytes_nulls_and_crs() raises:
    from dataframe import Column, concat

    var little = family(1)
    var big = family(1, big=True)
    var shape = from_wkb(
        Series.binary(
            "shape",
            [little.copy(), big.copy(), little.copy(), List[UInt8]()],
            [True, True, True, False],
        ),
        '{"crs":"OGC:CRS84"}',
    )
    var left = DataFrame(
        [Series("id", Column[Int64]([1, 2, 3, 4])), shape.copy()]
    )
    var right = DataFrame(
        [
            from_wkb(
                Series.binary("shape", [big.copy(), little.copy()]),
                '{"crs":"OGC:CRS84"}',
            ),
            Series("hint", Column[String](["big", "little"])),
        ]
    )
    var joined = left.join(right, on="shape", how="left")
    assert_equal(joined.height(), 4)
    assert_equal(joined.column("shape").dtype(), shape.dtype())
    for row in range(4):
        assert_equal(joined.item(row, "id").int64(), Int64(row + 1))
        if row == 3:
            assert_true(joined.item(row, "shape").is_null())
            assert_true(joined.item(row, "hint").is_null())
        else:
            assert_equal(
                to_wkb(joined.column("shape")).get(row).bytes(),
                big if row == 1 else little,
            )
            assert_equal(
                joined.item(row, "hint").string(),
                "big" if row == 1 else "little",
            )
    var grouped = left.group_by("shape").agg(col("id").sum())
    assert_equal(grouped.height(), 3)
    assert_equal(grouped.column("shape").dtype(), shape.dtype())
    var seen = List[String]()
    for row in range(3):
        var key = (
            String("null") if grouped.item(row, "shape")
            .is_null() else "big" if to_wkb(grouped.column("shape"))
            .get(row)
            .bytes()
            == big else "little"
        )
        assert_true(key not in seen)
        seen.append(key)
        assert_equal(
            grouped.item(row, "id").int64(), Int64(2 if key == "big" else 4)
        )
        if key == "little":
            assert_equal(
                to_wkb(grouped.column("shape")).get(row).bytes(), little
            )
    var chunks = concat([left.head(1), left.slice(1, 3)])
    assert_equal(chunks.column("shape").dtype(), shape.dtype())
    for row in range(4):
        assert_equal(chunks.item(row, "id").int64(), Int64(row + 1))
        assert_equal(chunks.item(row, "shape").is_null(), row == 3)
        if row < 3:
            assert_equal(
                to_wkb(chunks.column("shape")).get(row).bytes(),
                big if row == 1 else little,
            )


def test_geoparquet_metadata_import_errors_release_actual_native_exports_and_allow_retry() raises:
    from dataframe import read_parquet
    from tests.support.parquet_runtime import Observer

    var observer = Observer()
    var sources: List[String] = [
        "unsupported-version",
        "missing-primary",
        "unsupported-encoding",
        "conflicting-crs",
        "invalid-wkb",
    ]
    var diagnostics: List[String] = [
        "Unsupported GeoParquet version",
        "GeoParquet primary column",
        "Unsupported GeoParquet encoding",
        "Conflicting GeoParquet/GeoArrow metadata",
        "Invalid WKB at row 0",
    ]
    for scenario in range(len(sources)):
        observer.reset()
        var rejected = False
        try:
            _ = read_parquet(spatial_file(sources[scenario]))
        except error:
            if String(error).find(diagnostics[scenario]) < 0:
                raise error^
            rejected = True
        assert_true(rejected)
        assert_true(observer.get(6) > 0)
        assert_true(observer.get(8) > 0)
        observer.assert_balanced(1)
        observer.reset()
        var retry = read_parquet(spatial_file("default"))
        assert_equal(retry.height(), 5)
        assert_equal(
            retry.column("shape").dtype(),
            DataType.geometry('{"crs":"OGC:CRS84"}'),
        )
        assert_equal(to_wkb(retry.column("shape")).get(0).bytes(), family(1))
        observer.assert_balanced(1)
