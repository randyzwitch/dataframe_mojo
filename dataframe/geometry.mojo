"""WKB geometry storage and coordinate accessors; no reprojection."""
from .dtype import DataType
from .field_metadata import _without_extensions
from std.math import inf
from .series import Series
from .string_column import StringColumn
from .wkb import GeometryBounds, _WKBInfo, _wkb_info, validate_wkb
from .column import Column
from .frame import DataFrame
from .json_value import JsonValue


def from_wkb(binary: Series, metadata: String = "{}") raises -> Series:
    """Validate binary WKB and attach GeoArrow metadata, sharing buffers.

    Null rows are not parsed. Metadata describes coordinates already in the
    input; it does not reproject them. Invalid non-null rows raise.
    """
    if binary.dtype() != DataType.BINARY:
        raise Error("from_wkb requires a binary series")
    var dtype = DataType.geometry(metadata)
    var offset = 0
    for chunk in binary.chunks():
        ref values = chunk._data[StringColumn]
        for i in range(len(chunk)):
            if values._valid(i):
                try:
                    validate_wkb(values._row_bytes(i))
                except e:
                    raise Error(
                        "Invalid WKB at row "
                        + String(offset + i)
                        + ": "
                        + String(e)
                    )
        offset += len(chunk)
    var result = binary.with_dtype(dtype)
    result._field_metadata = _without_extensions(binary._field_metadata)
    return result^


def to_wkb(geometry: Series) raises -> Series:
    """Expose a geometry column as binary WKB, sharing buffers."""
    if not geometry.dtype().is_geometry():
        raise Error("to_wkb requires a geometry series")
    var result = geometry.with_dtype(DataType.BINARY)
    result._field_metadata = _without_extensions(geometry._field_metadata)
    return result^


def _validate_geoparquet_wkb(series: Series) raises:
    for chunk in series.chunks():
        ref values = chunk._data[StringColumn]
        for i in range(len(chunk)):
            if values._valid(i):
                validate_wkb(values._row_bytes(i), allow_m=False)


def _require_geometry(series: Series) raises:
    if not series.dtype().is_geometry():
        raise Error("Geometry accessor requires a geometry series")


def _row_info(
    values: StringColumn,
    row: Int,
    offset: Int,
    *,
    analyze: Bool = True,
    allow_m: Bool = True,
) raises -> _WKBInfo:
    try:
        return _wkb_info(
            values._row_bytes(row), analyze=analyze, allow_m=allow_m
        )
    except e:
        raise Error(
            "Invalid WKB at row " + String(offset + row) + ": " + String(e)
        )


def geometry_type(series: Series) raises -> Series:
    """ISO WKB type per row, with Z/M/ZM suffixes and nulls preserved.

    Empty geometries retain their type. Mixed columns report each row's type.
    """
    _require_geometry(series)
    var names = List[String](capacity=len(series))
    var valid = List[Bool](capacity=len(series))
    var offset = 0
    for chunk in series.chunks():
        ref values = chunk._data[StringColumn]
        for i in range(len(chunk)):
            var present = values._valid(i)
            valid.append(present)
            names.append(
                _row_info(
                    values, i, offset, analyze=False
                ).type_name() if present else ""
            )
        offset += len(chunk)
    return Series(series.name(), StringColumn(names, valid))


def crs(series: Series) raises -> String:
    """Return canonical CRS JSON (object or string), or `null` if unknown.

    This is metadata inspection, not a CRS parser or coordinate transform.
    """
    _require_geometry(series)
    return JsonValue(series.dtype().geometry_metadata()).get("crs").canonical()


def bounding_box(series: Series) raises -> DataFrame:
    """Per-row XY bounds as nullable xmin, ymin, xmax, ymax Float64 columns.

    Null and empty geometries yield four nulls; the original geometry validity
    distinguishes them. Z/M are ignored. Coordinates are not reprojected or
    wrapped at the antimeridian; spherical edge extrema are not computed.
    Non-finite XY coordinates raise, except the NaN pair of POINT EMPTY.
    """
    _require_geometry(series)
    var xmin = List[Float64](capacity=len(series))
    var ymin = List[Float64](capacity=len(series))
    var xmax = List[Float64](capacity=len(series))
    var ymax = List[Float64](capacity=len(series))
    var valid = List[Bool](capacity=len(series))
    var offset = 0
    for chunk in series.chunks():
        ref values = chunk._data[StringColumn]
        for i in range(len(chunk)):
            var info = _WKBInfo()
            if values._valid(i):
                info = _row_info(values, i, offset)
            var present = info.bounds._has_value()
            valid.append(present)
            xmin.append(info.bounds.xmin if present else 0)
            ymin.append(info.bounds.ymin if present else 0)
            xmax.append(info.bounds.xmax if present else 0)
            ymax.append(info.bounds.ymax if present else 0)
        offset += len(chunk)
    return DataFrame(
        [
            Series("xmin", Column[Float64](xmin^, valid)),
            Series("ymin", Column[Float64](ymin^, valid)),
            Series("xmax", Column[Float64](xmax^, valid)),
            Series("ymax", Column[Float64](ymax^, valid)),
        ]
    )


struct _GeometrySummary(Movable):
    var types: List[String]
    var bounds: GeometryBounds
    var has_z: Bool
    var finite_z: Bool
    var zmin: Float64
    var zmax: Float64

    def __init__(out self):
        self.types = List[String]()
        self.bounds = GeometryBounds()
        self.has_z = False
        self.finite_z = True
        self.zmin = inf[DType.float64]()
        self.zmax = -inf[DType.float64]()


def _geometry_summary(
    series: Series, *, allow_m: Bool = True
) raises -> _GeometrySummary:
    _require_geometry(series)
    var result = _GeometrySummary()
    var offset = 0
    for chunk in series.chunks():
        ref values = chunk._data[StringColumn]
        for i in range(len(chunk)):
            if not values._valid(i):
                continue
            var info = _row_info(values, i, offset, allow_m=allow_m)
            var name = info.type_name()
            if name not in result.types:
                result.types.append(name^)
            result.bounds._merge(info.bounds)
            result.has_z = result.has_z or info.has_z
            result.finite_z = result.finite_z and info.finite_z
            result.zmin = min(result.zmin, info.zmin)
            result.zmax = max(result.zmax, info.zmax)
        offset += len(chunk)
    return result^


def total_bounds(series: Series) raises -> Optional[GeometryBounds]:
    """XY bounds across all rows; None for an empty/all-null/all-empty column.

    Uses the same coordinate and non-finite policies as `bounding_box`.
    """
    var summary = _geometry_summary(series)
    if not summary.bounds._has_value():
        return None
    return summary.bounds.copy()
