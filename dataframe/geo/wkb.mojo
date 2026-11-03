"""Bounded ISO WKB traversal shared by storage validation and geometry accessors."""
from std.memory import bitcast
from std.math import inf


struct GeometryBounds(Copyable, Movable, Writable):
    """Axis-aligned XY coordinate extent, in the geometry's existing CRS."""

    var xmin: Float64
    var ymin: Float64
    var xmax: Float64
    var ymax: Float64

    def __init__(out self):
        self.xmin = inf[DType.float64]()
        self.ymin = inf[DType.float64]()
        self.xmax = -inf[DType.float64]()
        self.ymax = -inf[DType.float64]()

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "[",
            self.xmin,
            ", ",
            self.ymin,
            ", ",
            self.xmax,
            ", ",
            self.ymax,
            "]",
        )

    def _include(mut self, x: Float64, y: Float64):
        self.xmin = min(self.xmin, x)
        self.ymin = min(self.ymin, y)
        self.xmax = max(self.xmax, x)
        self.ymax = max(self.ymax, y)

    def _merge(mut self, other: Self):
        self.xmin = min(self.xmin, other.xmin)
        self.ymin = min(self.ymin, other.ymin)
        self.xmax = max(self.xmax, other.xmax)
        self.ymax = max(self.ymax, other.ymax)

    def _has_value(self) -> Bool:
        return self.xmin <= self.xmax


struct _WKBInfo(Copyable, Movable):
    var kind: Int
    var dimensional: Int
    var coordinate_count: Int
    var bounds: GeometryBounds
    var has_z: Bool
    var finite_z: Bool
    var zmin: Float64
    var zmax: Float64

    def __init__(out self):
        self.kind = 0
        self.dimensional = 0
        self.coordinate_count = 0
        self.bounds = GeometryBounds()
        self.has_z = False
        self.finite_z = True
        self.zmin = inf[DType.float64]()
        self.zmax = -inf[DType.float64]()

    def type_name(self) -> String:
        var names = List[String](
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
        var suffix = List[String](["", " Z", " M", " ZM"])
        return names[self.kind - 1] + suffix[self.dimensional]


def _wkb_float(
    data: Span[UInt8, ImmutAnyOrigin], mut pos: Int, little: Bool
) -> Float64:
    # The caller bounds-checks the complete coordinate sequence first.
    var bits = UInt64(0)
    for i in range(8):
        bits |= UInt64(data[pos + i]) << UInt64(8 * (i if little else 7 - i))
    pos += 8
    return bitcast[DType.float64](bits)


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
    data: Span[UInt8, ImmutAnyOrigin],
    mut pos: Int,
    count: Int,
    dims: Int,
    little: Bool,
    point: Bool,
    dimensional: Int,
    mut info: _WKBInfo,
    analyze: Bool,
) raises:
    var width = dims * 8
    if count > (len(data) - pos) // width:
        raise Error("Truncated WKB coordinates")
    if not analyze:
        pos += count * width
        return
    for _ in range(count):
        var x = _wkb_float(data, pos, little)
        var y = _wkb_float(data, pos, little)
        var z = Float64(0)
        if dims > 2:
            z = _wkb_float(data, pos, little)
        if dims == 4:
            _ = _wkb_float(data, pos, little)
        if point and x != x and y != y:
            continue  # The ISO WKB representation of POINT EMPTY.
        if (
            x != x
            or y != y
            or abs(x) == inf[DType.float64]()
            or abs(y) == inf[DType.float64]()
        ):
            raise Error("Non-finite WKB XY coordinate")
        info.coordinate_count += 1
        info.bounds._include(x, y)
        if dimensional == 1 or dimensional == 3:
            info.has_z = True
            if z != z or abs(z) == inf[DType.float64]():
                info.finite_z = False
            else:
                info.zmin = min(info.zmin, z)
                info.zmax = max(info.zmax, z)


def _wkb_scan(
    data: Span[UInt8, ImmutAnyOrigin],
    mut pos: Int,
    mut info: _WKBInfo,
    analyze: Bool = False,
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
    if depth == 0:
        info.kind = kind
        info.dimensional = dimensional
    var dims = 2 + (
        1 if dimensional == 1
        or dimensional == 2 else (2 if dimensional == 3 else 0)
    )
    if kind == 1:
        _wkb_coords(
            data, pos, 1, dims, little, True, dimensional, info, analyze
        )
        return
    var count = _wkb_uint(data, pos, little)
    if kind == 2:
        _wkb_coords(
            data, pos, count, dims, little, False, dimensional, info, analyze
        )
    elif kind == 3:
        if count > (len(data) - pos) // 4:
            raise Error("Truncated WKB rings")
        for _ in range(count):
            var points = _wkb_uint(data, pos, little)
            _wkb_coords(
                data,
                pos,
                points,
                dims,
                little,
                False,
                dimensional,
                info,
                analyze,
            )
    else:
        if count > (len(data) - pos) // 5:
            raise Error("Truncated WKB children")
        for _ in range(count):
            _wkb_scan(
                data,
                pos,
                info,
                analyze,
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
    var info = _WKBInfo()
    _wkb_scan(data, pos, info, allow_m=allow_m)
    if pos != len(data):
        raise Error("Trailing WKB bytes")


def _wkb_info(
    data: Span[UInt8, ImmutAnyOrigin],
    *,
    allow_m: Bool = True,
    analyze: Bool = True,
) raises -> _WKBInfo:
    var pos = 0
    var result = _WKBInfo()
    _wkb_scan(data, pos, result, analyze, allow_m=allow_m)
    if pos != len(data):
        raise Error("Trailing WKB bytes")
    return result^


def _wkb_display(data: Span[UInt8, ImmutAnyOrigin]) -> String:
    try:
        var info = _wkb_info(data)
        return (
            info.type_name()
            + " ("
            + String(info.coordinate_count)
            + (
                " coordinate)" if info.coordinate_count
                == 1 else " coordinates)"
            )
        )
    except:
        return "Invalid geometry"
