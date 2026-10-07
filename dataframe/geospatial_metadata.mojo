"""Translate GeoParquet 1.x metadata to and from geometry column metadata."""
from .json_value import JsonValue, json_quote
from .frame import DataFrame
from .series import Series
from .dtype import DataType
from .geometry import _geometry_summary
from .geometry import from_wkb, _validate_geoparquet_wkb


def apply_geoparquet_metadata(
    frame: DataFrame, text: String
) raises -> DataFrame:
    var meta = JsonValue(text)
    var version = meta.get("version").string()
    if version != "1.0.0" and version != "1.1.0":
        raise Error("Unsupported GeoParquet version: " + version)
    var specs = meta.get("columns")
    var primary = meta.get("primary_column").string()
    if not specs.has(primary):
        raise Error("GeoParquet primary column is not in geometry metadata")
    var columns = List[Series]()
    for name in frame.columns():
        var series = frame.column(name)
        if specs.has(name):
            var spec = specs.get(name)
            if spec.get("encoding").string() != "WKB":
                raise Error(
                    "Unsupported GeoParquet encoding; import WKB: " + name
                )
            var types = spec.get("geometry_types").items()
            var seen = List[String]()
            for item in types:
                var name = item.string()
                var base = String(
                    name[byte = 0 : name.byte_length() - 2]
                ) if name.endswith(" Z") else name
                if (
                    base != "Point"
                    and base != "LineString"
                    and base != "Polygon"
                    and base != "MultiPoint"
                    and base != "MultiLineString"
                    and base != "MultiPolygon"
                    and base != "GeometryCollection"
                ):
                    raise Error("Unknown GeoParquet geometry type: " + name)
                for previous in seen:
                    if name == previous:
                        raise Error(
                            "Duplicate GeoParquet geometry type: " + name
                        )
                seen.append(name)
            if (
                spec.has("edges")
                and spec.get("edges").string() != "planar"
                and spec.get("edges").string() != "spherical"
            ):
                raise Error("GeoParquet supports planar or spherical edges")
            _validate_column_metadata(spec)
            # Missing CRS is CRS84 in GeoParquet, unknown in GeoArrow.
            var crs = spec.get("crs")
            if spec.has("crs") and crs.kind() != 123 and crs.text != "null":
                raise Error("GeoParquet CRS must be PROJJSON or null")
            var json = String("{")
            if not spec.has("crs"):
                json += '"crs":"OGC:CRS84"'
            else:
                json += '"crs":' + crs.text
            for key in ["edges", "orientation", "epoch"]:
                if spec.has(key):
                    json += "," + json_quote(key) + ":" + spec.get(key).text
            json += "}"
            if series.dtype().is_geometry():
                # A file can carry both GeoParquet and Arrow metadata. Do not
                # silently choose one CRS/edge interpretation over the other.
                var existing = JsonValue(series.dtype().geometry_metadata())
                var declared = JsonValue(
                    DataType.geometry(json).geometry_metadata()
                )
                for key in ["crs", "edges", "orientation", "epoch"]:
                    var same = (
                        _geoparquet_crs_equal(
                            existing.get(key), declared.get(key)
                        ) if key
                        == "crs" else existing.get(key).canonical()
                        == declared.get(key).canonical()
                    )
                    if not same:
                        raise Error(
                            "Conflicting GeoParquet/GeoArrow metadata: "
                            + name
                            + ": "
                            + key
                        )
            else:
                series = from_wkb(series, json)
            _validate_geoparquet_wkb(series)
        columns.append(series^)
    return DataFrame(columns^, height=frame.height())


def _geoparquet_crs_equal(
    left: JsonValue, right: JsonValue, context: String = ""
) raises -> Bool:
    # GeoPandas removes optional datum-ensemble member IDs from its geo
    # document for compatibility with older PROJ databases, but retains them
    # in Arrow field metadata. Accept missing IDs only in that exact context.
    # If both copies carry an ID it must agree; all other CRS content remains
    # strict. This is not a general CRS equivalence test or authority lookup.
    if left.kind() != right.kind():
        return False
    if left.kind() == 123:
        for key in left.keys():
            if not right.has(key):
                if context == "member" and key == "id":
                    continue
                return False
            var child_context = String()
            if key == "datum_ensemble":
                child_context = "ensemble"
            elif context == "ensemble" and key == "members":
                child_context = "members"
            if not _geoparquet_crs_equal(
                left.get(key), right.get(key), child_context
            ):
                return False
        for key in right.keys():
            if not left.has(key):
                if context == "member" and key == "id":
                    continue
                return False
        return True
    if left.kind() == 91:
        var lhs = left.items()
        var rhs = right.items()
        if len(lhs) != len(rhs):
            return False
        for i in range(len(lhs)):
            if not _geoparquet_crs_equal(
                lhs[i], rhs[i], "member" if context == "members" else ""
            ):
                return False
        return True
    return left.canonical() == right.canonical()


def geoparquet_metadata(
    frame: DataFrame, primary_column: String = ""
) raises -> String:
    var primary = primary_column
    var specs = String()
    var found_primary = False
    for name in frame.columns():
        var dtype = frame.column(name).dtype()
        if not dtype.is_geometry():
            if _contains_geometry(dtype):
                raise Error("GeoParquet geometry columns must be top-level")
            continue
        var summary = _geometry_summary(frame.column(name), allow_m=False)
        if primary == "":
            primary = name
        found_primary = found_primary or primary == name
        var meta = JsonValue(dtype.geometry_metadata())
        _validate_column_metadata(meta)
        var crs = meta.get("crs")
        var spec = String('{"encoding":"WKB","geometry_types":[')
        for i in range(len(summary.types)):
            if i > 0:
                spec += ","
            spec += json_quote(summary.types[i])
        spec += "]"
        # Vertex extrema are conservative for planar edges. Spherical arcs
        # require a geodesic engine, so omit their optional file bbox.
        if (
            summary.bounds._has_value()
            and meta.get("edges").canonical() != '"spherical"'
            and summary.finite_z
        ):
            ref box = summary.bounds
            spec += ',"bbox":[' + String(box.xmin) + "," + String(box.ymin)
            if summary.has_z:
                spec += "," + String(summary.zmin)
            spec += "," + String(box.xmax) + "," + String(box.ymax)
            if summary.has_z:
                spec += "," + String(summary.zmax)
            spec += "]"
        if crs.kind() == 34:
            if crs.string() != "OGC:CRS84":
                raise Error(
                    "GeoParquet requires PROJJSON CRS metadata (or OGC:CRS84)"
                )
            # Omit to use the format's CRS84 default.
        else:
            spec += ',"crs":' + crs.text
        for key in ["edges", "orientation", "epoch"]:
            if meta.has(key):
                if key == "edges":
                    var edges = meta.get(key).string()
                    if edges != "planar" and edges != "spherical":
                        raise Error(
                            "GeoParquet supports planar or spherical edges"
                        )
                spec += "," + json_quote(key) + ":" + meta.get(key).text
        if specs != "":
            specs += ","
        specs += json_quote(name) + ":" + spec + "}"
    if not found_primary:
        raise Error("GeoParquet requires a geometry primary column")
    return (
        '{"version":"1.1.0","primary_column":'
        + json_quote(primary)
        + ',"columns":{'
        + specs
        + "}}"
    )


def _contains_geometry(dtype: DataType) raises -> Bool:
    if dtype.is_geometry():
        return True
    if dtype.is_list():
        return _contains_geometry(dtype.inner())
    if dtype.is_struct():
        for child in dtype.field_dtypes():
            if _contains_geometry(child):
                return True
    return False


def _validate_column_metadata(meta: JsonValue) raises:
    if (
        meta.has("orientation")
        and meta.get("orientation").string() != "counterclockwise"
    ):
        raise Error("GeoParquet orientation must be counterclockwise")
    if meta.has("epoch"):
        var epoch = meta.get("epoch")
        if epoch.kind() != 45 and (epoch.kind() < 48 or epoch.kind() > 57):
            raise Error("GeoParquet epoch must be a number")
        var value = Float64(epoch.text)
        if value != value or abs(value) == Float64("inf"):
            raise Error("GeoParquet epoch must be finite")
