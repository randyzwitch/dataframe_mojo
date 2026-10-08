# Shapefile reader

```mojo
from dataframe import read_shapefile

var parcels = read_shapefile("parcels.shp")
var legacy = read_shapefile("roads.shp", encoding="windows-1252")
```

`read_shapefile(path, geometry_name="geometry", encoding="utf-8")` reads an
ESRI shapefile into ordinary attribute columns and a WKB-backed geometry
column. It is an eager, native reader with no Python or GDAL runtime dependency.
The input components and output are held in memory.

The `.shp` and `.dbf` files are required. A `.shx`, when present, must agree
with every geometry record's offset and length. Header lengths, record
numbers, part offsets, attribute widths and the geometry/attribute record
counts are checked. Lowercase and uppercase sidecar extensions are recognized.
Deleted DBF records are omitted together with their corresponding geometry;
they never shift the remaining attributes onto the wrong geometry.

Supported geometry records are Null, Point, PolyLine, Polygon and MultiPoint,
including Z and M variants. Single-part polylines become LineString; multiple
parts become MultiLineString. Clockwise polygon rings are shells, and each
counterclockwise hole is assigned to its smallest containing shell, independent
of the input ring order. Multiple shells become MultiPolygon. Unclosed,
zero-area and orphan-hole rings raise; the reader does not repair invalid
polygon topology. MultiPatch and unknown record types raise with the record
number.

Null shapes remain null. Z coordinates are retained. M values below `-1e38`
are represented as missing measures (NaN). M shape types retain their measured
dimension even when the optional measure payload is absent. Z shapes gain an
M dimension when their measure array contains usable values; an absent or
entirely no-data measure array leaves them XYZ. Present coordinates must be
finite. GeoParquet 1.x cannot represent M; use Arrow/WKB interchange to retain
those dimensions.

DBF fields map as follows:

| DBF type | Output |
| --- | --- |
| C | String, preserving leading spaces and removing trailing space/NUL padding |
| N, zero decimal places | Int64, with overflow checks |
| N, nonzero decimal places; F | Float64 |
| D | Date, parsed strictly as YYYYMMDD |
| L | Bool (`T/Y`, `F/N`, case-insensitive) |

Blank fields are null. Numeric fields filled with asterisks, date `00000000`,
and logical `?` are also null. Unsupported field types and malformed values
raise; value errors identify the DBF record and field. The reader supports
the dBASE III/IV fixed-width layout, not memo contents or FoxPro extensions.

Encoding defaults to strict UTF-8. Explicit overrides support Latin-1,
Windows-1252 and ASCII. Invalid byte sequences and undefined Windows-1252 bytes
raise. The reader does not infer encoding from `.cpg` or DBF language-driver
bytes; pass the intended encoding explicitly.

A nonempty `.prj` is retained as its original WKT string in the geometry's
GeoArrow CRS metadata. There is no conversion to PROJJSON, coordinate
transformation or CRS guess. Missing or blank `.prj` means unknown CRS.

Run `pixi run -e oracle oracle-shapefile` for fixtures checked against
GeoPandas/GDAL and Shapely, including a GeoPandas-produced shapefile. Pyogrio
currently drops measured coordinates when creating GeoPandas geometries, so
M/ZM values are additionally checked against independent Shapely WKT geometries.
The oracle verifies attributes, dimensions, CRS, optional sidecars, encoding,
deleted records and malformed input. Contract tests run on every CI platform.
