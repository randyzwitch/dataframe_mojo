"""WKB geometry storage. No geometry analysis or coordinate transformation."""
from .dtype import DataType
from .series import Series
from .string_column import StringColumn


def _wkb_uint(
    data: Span[UInt8, ImmutAnyOrigin], mut pos: Int, little: Bool
) raises -> Int:
    if len(data) - pos < 4:
        raise Error("Truncated WKB integer")
    var result = 0
    for i in range(4):
        result |= Int(data[pos + i]) << (8 * (i if little else 3 - i))
    pos += 4
    return result


def _wkb_coords(
    data: Span[UInt8, ImmutAnyOrigin], mut pos: Int, count: Int, dims: Int
) raises:
    var width = dims * 8
    if count > (len(data) - pos) // width:
        raise Error("Truncated WKB coordinates")
    pos += count * width


def _wkb_scan(
    data: Span[UInt8, ImmutAnyOrigin],
    mut pos: Int,
    depth: Int = 0,
    expected: Int = 0,
    parent_dims: Int = 0,
    allow_m: Bool = True,
) raises:
    if depth > 64:
        raise Error("WKB nesting exceeds 64")
    if pos >= len(data):
        raise Error("Truncated WKB geometry")
    var order = data[pos]
    pos += 1
    if order != 0 and order != 1:
        raise Error("Invalid WKB byte order")
    var little = order == 1
    var code = _wkb_uint(data, pos, little)
    var dimensional = code // 1000
    var kind = code % 1000
    if dimensional > 3 or kind < 1 or kind > 7:
        raise Error("Unsupported WKB type (use ISO WKB, not EWKB)")
    if not allow_m and dimensional >= 2:
        raise Error("GeoParquet 1.x does not support M coordinates")
    if expected != 0 and (kind != expected or dimensional != parent_dims):
        raise Error("WKB multi-geometry child type/dimensions differ")
    var dims = 2 + (
        1 if dimensional == 1
        or dimensional == 2 else (2 if dimensional == 3 else 0)
    )
    if kind == 1:
        _wkb_coords(data, pos, 1, dims)
        return
    var count = _wkb_uint(data, pos, little)
    if kind == 2:
        _wkb_coords(data, pos, count, dims)
    elif kind == 3:
        if count > (len(data) - pos) // 4:
            raise Error("Truncated WKB rings")
        for _ in range(count):
            var points = _wkb_uint(data, pos, little)
            _wkb_coords(data, pos, points, dims)
    else:
        if count > (len(data) - pos) // 5:
            raise Error("Truncated WKB children")
        for _ in range(count):
            _wkb_scan(
                data,
                pos,
                depth + 1,
                kind - 3 if kind < 7 else 0,
                dimensional,
                allow_m,
            )


def validate_wkb(
    data: Span[UInt8, ImmutAnyOrigin], *, allow_m: Bool = True
) raises:
    """Check ISO WKB framing, sizes, dimensions and child types.

    Supports Point through GeometryCollection, XY/XYZ/XYM/XYZM, either
    endianness, and empty geometries. Does not validate polygon topology.
    """
    var pos = 0
    _wkb_scan(data, pos, allow_m=allow_m)
    if pos != len(data):
        raise Error("Trailing WKB bytes")


def from_wkb(binary: Series, metadata: String = "{}") raises -> Series:
    """Validate binary WKB and attach GeoArrow metadata, sharing buffers.

    Null rows are not parsed. Metadata describes coordinates already in the
    input; it does not reproject them. Invalid non-null rows raise.
    """
    if binary.dtype() != DataType.BINARY:
        raise Error("from_wkb requires a binary series")
    var dtype = DataType.geometry(metadata)
    for chunk in binary.chunks():
        ref values = chunk._data[StringColumn]
        for i in range(len(chunk)):
            if values._valid(i):
                try:
                    validate_wkb(values._row_bytes(i))
                except e:
                    raise Error(
                        "Invalid WKB at chunk row "
                        + String(i)
                        + ": "
                        + String(e)
                    )
    return binary.with_dtype(dtype)


def to_wkb(geometry: Series) raises -> Series:
    """Expose a geometry column as binary WKB, sharing buffers."""
    if not geometry.dtype().is_geometry():
        raise Error("to_wkb requires a geometry series")
    return geometry.with_dtype(DataType.BINARY)


def _validate_geoparquet_wkb(series: Series) raises:
    for chunk in series.chunks():
        ref values = chunk._data[StringColumn]
        for i in range(len(chunk)):
            if values._valid(i):
                validate_wkb(values._row_bytes(i), allow_m=False)
