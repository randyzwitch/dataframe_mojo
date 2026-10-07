# Geospatial storage and import

Geometry is a logical column type over the existing nullable binary storage.
Values use ISO Well-Known Binary (WKB); the dtype carries GeoArrow JSON metadata,
including the coordinate reference system (CRS) and edge interpretation.
No Python or additional native library is needed for WKB or GeoJSON ingestion.
Parquet uses the existing optional `libdfparquet` backend, rebuilt from this
revision (`pixi run -e native build-dfparquet`).

## Read GeoJSON and save GeoParquet

```mojo
from dataframe import read_geojson, read_parquet, write_geoparquet, col

var places = read_geojson("places.geojson")
var selected = places.filter(col("population") > 1000)
write_geoparquet(selected, "places.parquet")
var restored = read_parquet("places.parquet")
print(restored.column("geometry").dtype().geometry_metadata())
```

`read_geojson(path, geometry_name="geometry", id_name="feature_id")` reads a
local UTF-8 file. `from_geojson(text, ...)` reads the same format from memory.
Both accept a FeatureCollection, a single Feature, or a bare geometry. They
parse the entire input eagerly; there is no GeoJSON streaming scan yet.

- Point, LineString, Polygon, MultiPoint, MultiLineString, MultiPolygon, and
  GeometryCollection are supported, with two or three coordinate dimensions.
- Coordinates retain their order: longitude/easting first, latitude/northing
  second. GeoJSON gets `{"crs":"OGC:CRS84"}` metadata. Legacy `crs` members
  raise instead of being silently interpreted as WGS84.
- Null geometries remain null. Empty coordinate arrays produce empty geometries
  (empty points use WKB NaN ordinates); an empty polygon is not a missing value.
- Scalar properties become nullable String, Bool, Int64, or Float64 columns.
  Missing properties become null. A property with only null values is String.
  Mixed integer/fractional columns become Float64 when integers can be represented
  safely; oversized integers retain JSON text instead of being rounded.
- Nested and mixed-type properties retain JSON text. In mixed-type columns,
  string values keep their JSON quotes to distinguish `"1"` from `1`.
- Feature IDs retain JSON text in `feature_id`, when present, so numeric and
  string identifiers remain distinct. Property names conflicting with the
  geometry or ID column raise; choose different output names to read them.
- Foreign members and input bounding boxes are not retained. Geometry values,
  properties, and feature IDs are retained; this is not a lossless GeoJSON
  document editor.

Malformed JSON, non-finite coordinates, inconsistent coordinate dimensions,
and unclosed or undersized polygon rings raise. This checks representation and
basic structure, not polygon topology or geographical coordinate ranges.

## Work with WKB directly

```mojo
from dataframe import Series, from_wkb, to_wkb

# raw is List[List[UInt8]], one ISO WKB geometry per valid row.
var binary = Series.binary("shape", raw, valid)
var geometry = from_wkb(binary, metadata='{"crs":"EPSG:4326"}')
var bytes = to_wkb(geometry)
```

`from_wkb` validates non-null rows and shares their buffers. It accepts either
endianness, the seven basic geometry types, XY/XYZ/XYM/XYZM, empty geometries,
and geometry collections. It checks buffer lengths, counts, nesting (up to 64),
and multi-geometry child types. It does not check self-intersections or repair
invalid shapes. EWKB with embedded SRIDs is not supported; convert it to ISO
WKB and supply CRS metadata separately. `to_wkb` removes the logical geometry
tag while sharing storage. Geometry casts to text/numbers are rejected.

`DataType.geometry(metadata="{}")` validates and normalizes the JSON metadata.
`dtype.is_geometry()` identifies the type; `dtype.geometry_metadata()` returns
the normalized JSON. Missing/null CRS means unknown. Explicit planar edges are
equivalent to omitted edges. JSON keys and whitespace are normalized, but CRS
aliases or different PROJJSON definitions are not resolved for equivalence.
Assigning metadata describes existing coordinates; it never transforms them.
Use `from_wkb` for validated construction; `Series.with_dtype` remains the
low-level storage retagging operation.

Geometry dtype and metadata survive projection, renaming, filtering, gathering,
slicing, chunking, concatenation, and join output. Different geometry metadata
makes dtypes incompatible for concatenation. Equality, uniqueness, and any
ordinary dataframe key comparison use WKB bytes, not topological equality.

## Arrow and Parquet

Arrow import/export uses `geoarrow.wkb`, with `ARROW:extension:name` and
`ARROW:extension:metadata` on each geometry field. Import accepts binary and
large_binary storage; export uses large_binary. Unsupported GeoArrow geometry
encodings raise rather than silently losing spatial semantics.

`read_parquet` and `scan_parquet` recognize WKB GeoParquet 1.0.0 and 1.1.0,
including projected columns, empty inputs, row-group selection, and lazy
execution. Files without spatial metadata keep ordinary binary columns.
Metadata rules follow the [GeoParquet 1.1 specification](https://geoparquet.org/releases/v1.1.0/):

- Omitted CRS means OGC:CRS84; explicit `null` means unknown. CRS PROJJSON, edge
  interpretation, orientation, and coordinate epoch are retained per column.
- Conflicting GeoArrow and GeoParquet CRS/edge metadata raises.
- Geometry type inventories and file bounds are not retained as column
  invariants because filtering and concatenation can make them stale.
- The dataframe has no active/primary geometry designation. Reading retains
  every selected geometry column; writing chooses the first geometry column
  unless `primary_column` is supplied.

`write_geoparquet(frame, path, primary_column="", compression="zstd",
row_group_size=1_000_000)` writes GeoParquet 1.1 with WKB columns, replacing an
existing file. Compression options match `write_parquet`. The writer accepts
PROJJSON CRS objects, OGC:CRS84, or unknown CRS. Other CRS strings require the
caller to provide PROJJSON; no projection library or CRS database is bundled.
It writes unknown geometry-type inventories and omits bounds, avoiding stale
claims after dataframe operations. GeoParquet 1.x does not support M ordinates
or nested geometry columns; those are rejected on GeoParquet writing.

`write_parquet` preserves GeoArrow geometry metadata, including arbitrary CRS
strings, but does not add the GeoParquet file contract. Use `write_geoparquet`
for interoperability with GIS file readers.

Spatial predicates, distances, spatial joins/indexes, reprojection, WKT, native
GeoArrow coordinate encodings, GeoParquet 2.x, Shapefile, GeoPackage, and raster
readers remain future work. Existing scalar Parquet pruning still works; there
is no spatial bounding-box pruning yet.

## Validation

`tests/test_geospatial.mojo` covers storage, GeoJSON, malformed inputs,
metadata, dataframe operations, and Arrow round-trips.
`tests/test_geoparquet.mojo` uses external PyArrow-generated fixtures and checks
reads/writes, CRS rules, projections, lazy execution, and empty files.
Regenerate fixtures with `pixi run -e oracle python scripts/make_geospatial_fixtures.py`.
Run the independent producer/consumer check with
`pixi run -e oracle oracle-geospatial` after building `libdfparquet`.
