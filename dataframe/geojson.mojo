"""RFC 7946 GeoJSON ingestion into WKB geometry and attribute columns."""
from std.memory import Pointer
from .json_value import JsonValue
from .column import Column
from .series import Series
from .frame import DataFrame
from .geometry import from_wkb
from .parse import parse_integer


def _uint(mut out: List[UInt8], value: Int):
    for i in range(4):
        out.append(UInt8((value >> (8 * i)) & 255))


def _double(mut out: List[UInt8], value: Float64):
    var copy = value
    var bits = Pointer(to=copy).unsafe_bitcast[UInt64]()[]
    for i in range(8):
        out.append(UInt8((bits >> UInt64(8 * i)) & 255))


def _dimensions(coords: JsonValue, level: Int) raises -> Int:
    var items = coords.items()
    if len(items) == 0:
        return 0
    if level == 0:
        if len(items) != 2 and len(items) != 3:
            raise Error("GeoJSON positions require two or three ordinates")
        return len(items)
    var dims = 0
    for item in items:
        var child = _dimensions(item, level - 1)
        if child != 0:
            if dims != 0 and dims != child:
                raise Error("GeoJSON geometry has mixed coordinate dimensions")
            dims = child
    return dims


def _position(mut out: List[UInt8], value: JsonValue, dims: Int) raises:
    var coords = value.items()
    if len(coords) != dims:
        raise Error("Empty or inconsistent GeoJSON position")
    for coord in coords:
        if coord.kind() != 45 and (coord.kind() < 48 or coord.kind() > 57):
            raise Error("GeoJSON coordinate must be a number")
        var number = Float64(coord.text)
        if number != number or abs(number) == Float64("inf"):
            raise Error("GeoJSON coordinate must be finite")
        _double(out, number)


def _line(mut out: List[UInt8], value: JsonValue, dims: Int) raises:
    var coords = value.items()
    if len(coords) == 1:
        raise Error("GeoJSON LineString needs at least two positions")
    _uint(out, len(coords))
    for coord in coords:
        _position(out, coord, dims)


def _polygon(mut out: List[UInt8], value: JsonValue, dims: Int) raises:
    var rings = value.items()
    _uint(out, len(rings))
    for ring in rings:
        var positions = ring.items()
        if len(positions) < 4:
            raise Error("GeoJSON polygon ring needs at least four positions")
        var first = positions[0].items()
        var last = positions[len(positions) - 1].items()
        if len(first) != dims or len(last) != dims:
            raise Error("Invalid GeoJSON ring position")
        for d in range(dims):
            if Float64(first[d].text) != Float64(last[d].text):
                raise Error("GeoJSON polygon ring must be closed")
        _line(out, ring, dims)


def _header(mut out: List[UInt8], kind: Int, dims: Int):
    out.append(1)
    _uint(out, kind + (1000 if dims == 3 else 0))


def _geometry(mut out: List[UInt8], value: JsonValue, depth: Int = 0) raises:
    if depth > 64:
        raise Error("GeoJSON geometry nesting exceeds 64")
    if value.has("crs"):
        raise Error("Legacy GeoJSON CRS is unsupported; use RFC 7946 CRS84")
    var name = value.get("type").string()
    if name == "GeometryCollection":
        var geometries = value.get("geometries").items()
        var children = List[List[UInt8]]()
        var dims = 2
        for geometry in geometries:
            var child = List[UInt8]()
            _geometry(child, geometry, depth + 1)
            # This encoder writes little-endian ISO WKB. A Z child (including
            # a nested collection) makes the collection's header three-dimensional.
            if child[2] != 0:
                dims = 3
            children.append(child^)
        _header(out, 7, dims)
        _uint(out, len(children))
        for child in children:
            out.extend(Span(child))
        return
    var kind: Int
    var level = 0
    if name == "Point":
        kind = 1
    elif name == "LineString":
        kind = 2
        level = 1
    elif name == "Polygon":
        kind = 3
        level = 2
    elif name == "MultiPoint":
        kind = 4
        level = 1
    elif name == "MultiLineString":
        kind = 5
        level = 2
    elif name == "MultiPolygon":
        kind = 6
        level = 3
    else:
        raise Error("Unsupported GeoJSON geometry: " + name)
    var coords = value.get("coordinates")
    var dims = _dimensions(coords, level)
    if dims == 0:
        dims = 2
    _header(out, kind, dims)
    if kind == 1:
        if len(coords.items()) == 0:
            for _ in range(dims):
                _double(out, Float64("nan"))
        else:
            _position(out, coords, dims)
    elif kind == 2:
        _line(out, coords, dims)
    elif kind == 3:
        _polygon(out, coords, dims)
    else:
        var children = coords.items()
        _uint(out, len(children))
        for child in children:
            _header(out, kind - 3, dims)
            if kind == 4:
                _position(out, child, dims)
            elif kind == 5:
                _line(out, child, dims)
            else:
                _polygon(out, child, dims)


def _property(name: String, properties: List[JsonValue]) raises -> Series:
    var kind = 0
    var valid = List[Bool]()
    var values = List[JsonValue]()
    for props in properties:
        var value = props.get(name) if props.text != "null" else JsonValue(
            "null"
        )
        valid.append(value.text != "null")
        if value.text != "null":
            var current = 1 if value.kind() == 34 else (
                2 if value.text == "true"
                or value.text
                == "false" else (
                    3 if value.kind() == 45
                    or (value.kind() >= 48 and value.kind() <= 57) else 4
                )
            )
            if kind == 0:
                kind = current
            elif kind != current:
                kind = 4
        values.append(value^)
    if kind == 2:
        var out = List[Bool]()
        for value in values:
            out.append(value.text == "true")
        return Series(name, Column[Bool](out^, valid))
    if kind == 3:
        var ints = List[Int64]()
        var all_int = True
        var oversized_int = False
        for value in values:
            if value.text == "null":
                ints.append(0)
            else:
                try:
                    ints.append(parse_integer[DType.int64](value.text))
                except:
                    all_int = False
                    if not (
                        "." in value.text
                        or "e" in value.text
                        or "E" in value.text
                    ):
                        oversized_int = True
        if all_int:
            return Series(name, Column[Int64](ints^, valid))
        for value in ints:
            if value > 9007199254740992 or value < -9007199254740992:
                oversized_int = True
        if oversized_int:
            var texts = List[String]()
            for value in values:
                texts.append(value.text if value.text != "null" else "")
            return Series(name, Column[String](texts^, valid))
        var floats = List[Float64]()
        for value in values:
            var number = Float64(0) if value.text == "null" else Float64(
                value.text
            )
            if number != number or abs(number) == Float64("inf"):
                raise Error("GeoJSON property exceeds Float64 range: " + name)
            floats.append(number)
        return Series(name, Column[Float64](floats^, valid))
    var strings = List[String]()
    for value in values:
        strings.append(
            value.string() if kind == 1
            and value.text
            != "null" else ("" if value.text == "null" else value.text)
        )
    return Series(name, Column[String](strings^, valid))


def from_geojson(
    text: String,
    *,
    geometry_name: String = "geometry",
    id_name: String = "feature_id",
) raises -> DataFrame:
    """Read a GeoJSON FeatureCollection, Feature, or bare geometry.

    Geometry uses ISO WKB with CRS84 metadata. Scalar properties become
    nullable columns; nested or mixed-type properties retain JSON as text.
    Feature IDs retain JSON text so numeric and string IDs remain distinct.
    Input is parsed eagerly. No coordinate transformation is performed.
    """
    if geometry_name == id_name:
        raise Error("GeoJSON geometry and feature ID column names must differ")
    var root = JsonValue(text)
    if root.has("crs"):
        raise Error("Legacy GeoJSON CRS is unsupported; use RFC 7946 CRS84")
    var kind = root.get("type").string()
    var features = List[JsonValue]()
    if kind == "FeatureCollection":
        features = root.get("features").items()
    else:
        features.append(root.copy())
    var bytes = List[List[UInt8]]()
    var valid = List[Bool]()
    var properties = List[JsonValue]()
    var names = List[String]()
    var ids = List[String]()
    var id_valid = List[Bool]()
    var has_id = False
    for feature in features:
        var is_feature = feature.get("type").string() == "Feature"
        if kind == "FeatureCollection" and not is_feature:
            raise Error("GeoJSON FeatureCollection must contain Features")
        if feature.has("crs"):
            raise Error("Legacy GeoJSON CRS is unsupported; use RFC 7946 CRS84")
        var geom = feature.get("geometry") if is_feature else feature.copy()
        var props = feature.get("properties") if is_feature else JsonValue(
            "null"
        )
        if is_feature and (
            not feature.has("geometry") or not feature.has("properties")
        ):
            raise Error("GeoJSON Feature requires geometry and properties")
        if props.text != "null":
            for name in props.keys():
                if name == geometry_name or name == id_name:
                    raise Error(
                        "GeoJSON property conflicts with output column: " + name
                    )
                var found = False
                for previous in names:
                    found = found or previous == name
                if not found:
                    names.append(name)
        properties.append(props^)
        var id = feature.get("id") if is_feature else JsonValue("null")
        if (
            id.text != "null"
            and id.kind() != 34
            and id.kind() != 45
            and (id.kind() < 48 or id.kind() > 57)
        ):
            raise Error("GeoJSON feature ID must be string or number")
        has_id = has_id or id.text != "null"
        ids.append(id.text)
        id_valid.append(id.text != "null")
        var wkb = List[UInt8]()
        valid.append(geom.text != "null")
        if geom.text != "null":
            _geometry(wkb, geom)
        bytes.append(wkb^)
    var columns = List[Series]()
    for name in names:
        columns.append(_property(name, properties))
    if has_id:
        columns.append(Series(id_name, Column[String](ids^, id_valid)))
    columns.append(
        from_wkb(
            Series.binary(geometry_name, bytes, valid), '{"crs":"OGC:CRS84"}'
        )
    )
    return DataFrame(columns^)


def read_geojson(
    path: String,
    *,
    geometry_name: String = "geometry",
    id_name: String = "feature_id",
) raises -> DataFrame:
    """Read a local UTF-8 GeoJSON file eagerly."""
    with open(path, "r") as file:
        return from_geojson(
            file.read(), geometry_name=geometry_name, id_name=id_name
        )
