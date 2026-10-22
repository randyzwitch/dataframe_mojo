# Architecture and development direction

## Current layers

- `column.mojo`: typed windows (offset, length) onto reference-counted value
  buffers and LSB-first validity bitmaps, as in Arrow arrays.
- `string_column.mojo`: string columns in the Arrow `large_utf8` layout (one
  UTF-8 byte buffer, Int64 offsets, validity), read as borrowed `StringSlice`s
  and built with `StringBuilder`.
- `bool_column.mojo`: Boolean columns with bit-packed values (one bit per
  row), sharing the bitmap helpers with validity.
- `io/`: CSV, Parquet, GeoJSON, and Shapefile readers and writers, and
  (`io/arrow.mojo`) Arrow C Data Interface export (zero-copy where layouts
  agree) and import.
- `series.mojo`: named, runtime-tagged columns and batch slicing/concatenation.
- `expr.mojo`: flat expression nodes and composition, independent of data.
- `binding.mojo`: schema resolution, dtypes, shape, and aggregate dependencies.
- `execution.mojo`: bounded eager evaluation and grouped accumulator scheduling.
- `expr_kernels.mojo`: specialized binary kernels, including explicit Float64 SIMD.
- `reductions.mojo`: mergeable exact-integer and reassociable floating-point states.
- `frame.mojo`: validated eager dataframe operations and expression entry points.

The only runtime dependency is the Mojo standard library. This is an eager CPU
implementation, not a production query engine. Buffers are immutable once
shared, so copying a column, `select`, `rename`, `drop`, `head`, `slice`,
`column()`, typed extraction, GroupBy snapshots, and expression batch slices are
O(1) per column and share storage. The only in-place growth (batch reassembly)
copies first unless the column owns its buffers outright. Column lookup at binding
is O(schema width), but execution uses resolved source indices.

Sorting is O(n log n) with O(n) index workspace. Hash grouping is expected O(n).
Join index construction is expected O(left + right + output); materialization
also scales with column count. Variable-length string costs depend on their length.

See [expression semantics](expressions.md) for the broadcasting, ordering, and
reduction contracts that deliberately leave room for parallelism and SIMD.
No universal dataframe trait or foreign-backend abstraction is imposed.

## Current state

The milestones this page used to list are in place: lazy plans with
predicate, projection, and slice pushdown ([lazy queries](lazy.md)); parallel
hash grouping, joins, and sorts on worker threads, with `maintain_order` as an
explicit request; chunked columns and Arrow C Data import and export
([Arrow interchange](arrow.md)); CSV and Parquet input ([CSV](csv.md),
[Parquet](parquet.md)); and multi-column sorts with multiple numeric or string
grouping and join keys. Performance work is measured query by query against
Polars and DuckDB on external suites; see [benchmarks](design/benchmarks.md).

Parquet reading goes through Arrow C++ behind a C shim (`native/dfparquet`)
until a Mojo-native reader exists; `dataframe/io/parquet.mojo` is the seam.

Optional GPU execution on Apple silicon is covered in
[Apple Metal](apple-metal.md). Distributed execution, arbitrary Python-object
columns, pandas index compatibility, and cross-engine adapters are outside the
scope.
