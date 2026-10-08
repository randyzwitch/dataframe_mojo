"""Native eager ESRI shapefile ingestion, with dBASE attributes and WKT CRS."""
from std.os.path import exists
from std.math import isfinite
from .shapefile_binary import (
    read_bytes,
    require,
    uint,
    number,
    put_uint,
    put_number,
    text,
)
from .shapefile_dbf import read_dbf
from .series import Series
from .frame import DataFrame
from .geometry import from_wkb
from .json_value import json_quote


def _header(data: List[UInt8], component: String) raises -> Int:
    require(data, 0, 100)
    if uint(data, 0, 4, True) != 9994 or uint(data, 28, 4) != 1000:
        raise Error("Invalid " + component + " header signature/version")
    if uint(data, 24, 4, True) * 2 != len(data):
        raise Error(component + " header length differs from file size")
    return uint(data, 32, 4)


def _wkb_header(mut out: List[UInt8], kind: Int, z: Bool, m: Bool):
    out.append(1)
    put_uint(out, kind + (1000 if z else 0) + (2000 if m else 0))


@fieldwise_init
struct _Points(Copyable):
    var x: List[Float64]
    var y: List[Float64]
    var z: List[Float64]
    var m: List[Float64]
    var has_z: Bool
    var has_m: Bool

    def write(self, mut out: List[UInt8], at: Int):
        put_number(out, self.x[at])
        put_number(out, self.y[at])
        if self.has_z:
            put_number(out, self.z[at])
        if self.has_m:
            put_number(out, self.m[at])

    def line(self, mut out: List[UInt8], start: Int, end: Int):
        put_uint(out, end - start)
        for i in range(start, end):
            self.write(out, i)

    def area(self, start: Int, end: Int) -> Float64:
        var area = Float64(0)
        # Translate to avoid cancellation in large projected coordinates.
        for i in range(start, end - 1):
            area += (self.x[i] - self.x[start]) * (
                self.y[i + 1] - self.y[start]
            ) - (self.x[i + 1] - self.x[start]) * (self.y[i] - self.y[start])
        return area

    def contains(self, start: Int, end: Int, point: Int) -> Bool:
        var inside = False
        var x = self.x[point]
        var y = self.y[point]
        for i in range(start, end - 1):
            var a = self.y[i]
            var b = self.y[i + 1]
            if (a > y) != (b > y):
                if (
                    x
                    < (self.x[i + 1] - self.x[i]) * (y - a) / (b - a)
                    + self.x[i]
                ):
                    inside = not inside
        return inside


def _coordinates(
    data: List[UInt8], at: Int, count: Int, z: Bool, measured: Bool, point: Bool
) raises -> _Points:
    require(data, at, count * 16)
    var x = List[Float64]()
    var y = List[Float64]()
    var zs = List[Float64](length=count, fill=0)
    var ms = List[Float64](length=count, fill=Float64("nan"))
    for i in range(count):
        x.append(number(data, at + i * 16))
        y.append(number(data, at + i * 16 + 8))
        if not isfinite(x[i]) or not isfinite(y[i]):
            raise Error("Non-finite shape XY coordinate")
    var pos = at + count * 16
    if z:
        require(data, pos, count * 8 + (0 if point else 16))
        if not point:
            pos += 16
        for i in range(count):
            zs[i] = number(data, pos + i * 8)
            if not isfinite(zs[i]):
                raise Error("Non-finite shape Z coordinate")
        pos += count * 8
    var has_m = measured
    if z or measured:
        if len(data) != pos:
            require(data, pos, count * 8 + (0 if point else 16))
            if not point:
                pos += 16
            for i in range(count):
                var value = number(data, pos + i * 8)
                if not isfinite(value):
                    raise Error("Non-finite shape M coordinate")
                if value >= -1e38:
                    ms[i] = value
                    has_m = True
            pos += count * 8
    if pos != len(data):
        raise Error("Unexpected trailing shape record bytes")
    return _Points(x^, y^, zs^, ms^, z, has_m)


def _shape(data: List[UInt8], declared: Int) raises -> List[UInt8]:
    var kind = uint(data, 0, 4)
    if kind == 0:
        if len(data) != 4:
            raise Error("Null shape has trailing bytes")
        return List[UInt8]()
    if kind == 31:
        raise Error("MultiPatch shape type is unsupported")
    if kind not in [1, 3, 5, 8, 11, 13, 15, 18, 21, 23, 25, 28]:
        raise Error("Unsupported shape type: " + String(kind))
    if kind != declared:
        raise Error("Shape type differs from SHP header")
    var z = kind >= 11 and kind <= 18
    var measured = kind >= 21
    var base = kind - (10 if z else 20 if measured else 0)
    var out = List[UInt8]()
    if base == 1:
        var points = _coordinates(data, 4, 1, z, measured, True)
        _wkb_header(out, 1, z, points.has_m)
        points.write(out, 0)
        return out^
    require(data, 0, 40)
    var parts = List[Int]()
    var count: Int
    var at: Int
    if base == 8:
        count = uint(data, 36, 4)
        at = 40
    else:
        var n_parts = uint(data, 36, 4)
        count = uint(data, 40, 4)
        require(data, 44, n_parts * 4)
        at = 44 + n_parts * 4
        for i in range(n_parts):
            var start = uint(data, 44 + i * 4, 4)
            if (
                (i == 0 and start != 0)
                or start >= count
                or (i > 0 and start <= parts[i - 1])
            ):
                raise Error("Invalid part offsets")
            parts.append(start)
        if (n_parts == 0) != (count == 0):
            raise Error("Inconsistent shape part/point counts")
        parts.append(count)
    var points = _coordinates(data, at, count, z, measured, False)
    if base == 8:
        _wkb_header(out, 4, z, points.has_m)
        put_uint(out, count)
        for i in range(count):
            _wkb_header(out, 1, z, points.has_m)
            points.write(out, i)
    elif base == 3:
        var n = len(parts) - 1
        _wkb_header(out, 2 if n == 1 else 5, z, points.has_m)
        if n != 1:
            put_uint(out, n)
        for i in range(n):
            if parts[i + 1] - parts[i] < 2:
                raise Error("Polyline part needs at least two points")
            if n != 1:
                _wkb_header(out, 2, z, points.has_m)
            points.line(out, parts[i], parts[i + 1])
    else:
        var shells = List[Int]()
        var holes = List[Int]()
        var areas = List[Float64]()
        for i in range(len(parts) - 1):
            var first = parts[i]
            var last = parts[i + 1] - 1
            if (
                last - first < 3
                or points.x[first] != points.x[last]
                or points.y[first] != points.y[last]
            ):
                raise Error("Polygon ring must have four points and be closed")
            var area = points.area(first, last + 1)
            if area == 0:
                raise Error("Polygon ring has zero signed area")
            areas.append(abs(area))
            if area < 0:
                shells.append(i)
            else:
                holes.append(i)
        var owners = List[Int](length=len(holes), fill=-1)
        for h in range(len(holes)):
            for shell in shells:
                if points.contains(
                    parts[shell], parts[shell + 1], parts[holes[h]]
                ):
                    if owners[h] < 0 or areas[shell] < areas[owners[h]]:
                        owners[h] = shell
            if owners[h] < 0:
                raise Error("Polygon hole has no containing clockwise shell")
        _wkb_header(out, 3 if len(shells) <= 1 else 6, z, points.has_m)
        if len(shells) != 1:
            put_uint(out, len(shells))
        for shell in shells:
            if len(shells) > 1:
                _wkb_header(out, 3, z, points.has_m)
            var rings = 1
            for owner in owners:
                rings += Int(owner == shell)
            put_uint(out, rings)
            points.line(out, parts[shell], parts[shell + 1])
            for h in range(len(holes)):
                if owners[h] == shell:
                    points.line(out, parts[holes[h]], parts[holes[h] + 1])
    return out^


def _sidecar(stem: String, extension: String) -> String:
    var path = stem + extension
    return path if exists(path) else stem + extension.upper()


def read_shapefile(
    path: String,
    *,
    geometry_name: String = "geometry",
    encoding: String = "utf-8",
) raises -> DataFrame:
    """Read SHP/DBF with optional validated SHX and verbatim WKT PRJ.

    Supports Point, PolyLine, Polygon and MultiPoint in XY, Z and M forms.
    Deleted DBF rows are skipped with their geometries. Missing PRJ means
    unknown CRS. Encoding is explicit (UTF-8 by default); invalid bytes raise.
    Files are read eagerly into memory. No GDAL or Python runtime is needed.
    """
    if not path.lower().endswith(".shp"):
        raise Error("read_shapefile requires a .shp path")
    var stem = String(path[byte = 0 : path.byte_length() - 4])
    var data = read_bytes(path)
    var declared = _header(data, "SHP")
    var index_path = _sidecar(stem, ".shx")
    var index = List[UInt8]()
    if exists(index_path):
        index = read_bytes(index_path)
        if _header(index, "SHX") != declared or (len(index) - 100) % 8 != 0:
            raise Error("SHX header disagrees with SHP")
    var geometries = List[List[UInt8]]()
    var validity = List[Bool]()
    var pos = 100
    while pos < len(data):
        var row = len(geometries)
        try:
            var record = uint(data, pos, 4, True)
            var size = uint(data, pos + 4, 4, True) * 2
            if record != row + 1:
                raise Error("Unexpected record number " + String(record))
            require(data, pos + 8, size)
            if len(index):
                var at = 100 + row * 8
                if at + 8 > len(index):
                    raise Error("SHX record count disagrees with SHP")
                if (
                    uint(index, at, 4, True) * 2 != pos
                    or uint(index, at + 4, 4, True) * 2 != size
                ):
                    raise Error("SHX offset/length disagrees with SHP")
            var bytes = List[UInt8](capacity=size)
            for i in range(size):
                bytes.append(data[pos + 8 + i])
            var wkb = _shape(bytes, declared)
            validity.append(len(wkb) != 0)
            geometries.append(wkb^)
            pos += 8 + size
        except e:
            raise Error("SHP record " + String(row + 1) + ": " + String(e))
    if len(index) and (len(index) - 100) // 8 != len(geometries):
        raise Error("SHX record count disagrees with SHP")
    var attributes = read_dbf(_sidecar(stem, ".dbf"), encoding, len(geometries))
    var metadata = String("{}")
    var prj = _sidecar(stem, ".prj")
    if exists(prj):
        var bytes = read_bytes(prj)
        var wkt = text(bytes, 0, len(bytes), "utf8")
        if String(wkt.strip()) != "":
            metadata = '{"crs":' + json_quote(wkt) + "}"
    var geometry = from_wkb(
        Series.binary(geometry_name, geometries, validity), metadata
    ).take(attributes[1])
    var columns = attributes[0].copy()
    columns.append(geometry^)
    return DataFrame(columns^, height=len(attributes[1]))
