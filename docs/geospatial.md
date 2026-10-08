# Geospatial storage and coordinate accessors

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

## Inspect geometry and bounds

```mojo
var geometry = places.column("geometry")
print(geometry.geometry_type())  # Point, Polygon Z, …; nulls remain null
print(geometry.crs())            # Canonical CRS JSON, or "null" if unknown
print(geometry.bounding_box())   # xmin, ymin, xmax, ymax Float64 columns
var extent = geometry.total_bounds()
if extent:
    print(extent.value())        # [xmin, ymin, xmax, ymax]
```

The same accessors are also exported as functions taking a Series.
`geometry_type()` includes ` Z`, ` M`, or ` ZM` suffixes and reports the type
of empty geometries. Geometry display shows a type and coordinate count, such
as `Point (1 coordinate)` or `Polygon (0 coordinates)`, rather than raw bytes.
`to_wkb` remains the buffer-sharing binary accessor.

Bounds scan every coordinate, including polygon holes and collection children,
in either WKB byte order. They use XY only, in the existing coordinate system;
Z and M do not affect the result. Null and empty geometries produce four null
bounds, while their original validity and geometry type remain distinct.
`total_bounds()` returns `Optional[GeometryBounds]`, with `xmin`, `ymin`, `xmax`,
and `ymax` fields; it is absent when there are no non-empty geometries.

These are axis-aligned coordinate extrema, matching the planar interpretation
of [GeoPandas bounds](https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoSeries.bounds.html).
A line from longitude 179 to -179 has xmin=-179 and xmax=179. Longitude is not
wrapped, no shortest antimeridian interval is selected, and spherical arc
extrema are not computed. Bounds reject non-finite XY coordinates except the
pair of NaNs representing an empty point. Errors identify the global row,
including for chunked columns. Type inspection still accepts structurally valid
WKB with non-finite coordinates, and Z/M values do not affect XY bounds.

## Arrow and Parquet

Arrow import/export uses `geoarrow.wkb`, with `ARROW:extension:name` and
`ARROW:extension:metadata` on each geometry field. Import accepts binary and
large_binary storage; export uses large_binary. Other GeoArrow extensions retain
their opaque field metadata on supported storage layouts, without geometry
interpretation. See [field metadata and extensions](arrow.md#field-metadata-and-extensions).

`read_parquet` and `scan_parquet` recognize WKB GeoParquet 1.0.0 and 1.1.0,
including projected columns, empty inputs, row-group selection, and lazy
execution. Files without spatial metadata keep ordinary binary columns.
Metadata rules follow the [GeoParquet 1.1 specification](https://geoparquet.org/releases/v1.1.0/):

- Omitted CRS means OGC:CRS84; explicit `null` means unknown. CRS PROJJSON, edge
  interpretation, orientation, and coordinate epoch are retained per column.
- Conflicting GeoArrow and GeoParquet CRS/edge metadata raises. GeoPandas'
  omission of optional datum-ensemble member IDs is accepted when the remaining
  CRS content agrees; IDs present in both copies must match. The Arrow CRS
  is preserved. Other CRS differences still raise without a CRS database
  to establish equivalence.
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
It scans the current rows to write unique observed geometry types, including
` Z` suffixes, and fresh file bounds. Bounds contain six values when non-empty
Z coordinates are present, otherwise four. Mixed XY/XYZ files use the available
Z coordinates for the Z extent. The optional bbox is omitted for all-empty or
all-null data, spherical edges, or non-finite Z values. No per-row bbox covering
columns are added. GeoParquet 1.x does not support M ordinates or nested geometry
columns; those are rejected on GeoParquet writing.

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

`pixi run -e oracle oracle-geoparquet` runs the GeoParquet interoperability
checks used in CI. GeoPandas produces both 1.0 and 1.1 files, and DuckDB Spatial
produces 1.0 files through its native Parquet writer. Both engines read the
Mojo-written full, empty, filtered, empty-geometry and null-only outputs with
warnings treated as errors.
The checks compare exact WKB, CRS, geometry types and fresh file bounds across
all seven geometry families, XY/XYZ, nulls and empties. CRS cases include unknown,
CRS84 and projected coordinates. DuckDB's V1 writer omits CRS for untagged
geometry, which declares CRS84; GeoPandas supplies the explicit-null CRS case.
The suite also checks row-group selection and lazy pruning past corrupted row
groups, eager/lazy projections over invalid WKB and corrupted geometry pages,
unknown-version rejection, and ordinary binary Parquet. Mutation checks reject
changed CRS, type inventories, bounds, WKB and missing metadata.
It uses the same Spatial extension installation/cache as the Arrow
oracle below; installation failures fail the check.

`tests/test_geometry_accessors.mojo` covers all seven WKB families, all four
coordinate layouts, both byte orders (including mixed-order children), empty
and null rows, chunked errors, display, and CRS-preserving operations.
`pixi run -e oracle oracle-geometry` compares WKB, types, per-row and total bounds,
and CRS with GeoPandas/GEOS, then checks GeoParquet output through GeoPandas.
GeoPandas and Shapely are development-only dependencies in the oracle environment.

The same `oracle-geometry` CI task also runs the DuckDB Spatial Arrow C oracle.
Run it alone with `pixi run -e oracle oracle-duckdb-spatial`. Its setup installs
DuckDB's official Spatial extension over HTTPS into `build/duckdb_extensions`
(the first run needs network access). Installation or loading failures fail the
check rather than skipping it. DuckDB itself constructs and exports the WKB
and GeoArrow metadata: seven geometry families, XY/XYZ/XYM/XYZM, nulls, empties,
unknown CRS, EPSG:4326 and EPSG:3857, with binary and large_binary layouts.
Mojo imports/exports through the C interface, then DuckDB consumes the result
as GEOMETRY and checks WKB bytes, CRS, type, dimensions, emptiness and coordinate
counts. Reordered rows and a Mojo-originated GeoJSON round trip are covered.
Mutation checks ensure metadata loss, CRS changes and changed WKB are detected.
