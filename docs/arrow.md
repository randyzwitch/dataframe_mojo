# Arrow interchange

`dataframe` speaks the [Arrow C Data Interface][cdi]: two C structs,
`ArrowSchema` and `ArrowArray`, that any Arrow implementation can read and
fill. Polars, DuckDB, pyarrow, nanoarrow, and marrow can exchange data with a
`DataFrame` this way. No Arrow library or Python is involved at run time.

For Arrow IPC/Feather files and IPC streams, see [IPC I/O](ipc.md).

[cdi]: https://arrow.apache.org/docs/format/CDataInterface.html

```mojo
from dataframe import ArrowArray, ArrowSchema, export_arrow, import_arrow

var array = ArrowArray()
var schema = ArrowSchema()
export_arrow(frame, array, schema)       # frame -> Arrow struct array ("+s")
var copy = import_arrow(array, schema)   # Arrow struct array -> frame
```

`import_arrow(..., copy=False)` and `import_arrow_series(..., copy=False)`
can retain compatible fixed-width value buffers instead of copying them.
The input structs are consumed, and their array release callback runs after
the last borrowing column, slice, or exported view is destroyed. Schemas and
field metadata are copied before their schema callbacks run. The producer
must leave borrowed buffers unchanged until release; use the default
`copy=True` for an independent snapshot.

The buffer ownership foundation currently borrows matching numeric, decimal,
nanosecond time, and supported timestamp/duration payloads. Unaligned values
and layout conversions use owned copies. Validity, Boolean, string/binary,
and nested offset buffers still use the existing copy paths while the
remaining borrowing layouts are integrated. IPC readers retain their current
copying policy until that integration is complete.

`export_arrow_series` and `import_arrow_series` do the same for one column.
Each function also has an overload taking the structs' raw addresses (`Int`),
for structs owned by C or Python. From Mojo code, prefer the `mut` struct
overloads so the compiler can see the structs being read and written.

A frame exports as a struct array (format `+s`) with one child per column:
the shape pyarrow imports with `RecordBatch._import_from_c` and Polars with
`pl.from_arrow`.

## Types

| dtype | Arrow format | export | import also accepts |
|---|---|---|---|
| Int8–Int64 | `c` `s` `i` `l` | zero-copy | |
| UInt8–UInt64 | `C` `S` `I` `L` | zero-copy | |
| Float32, Float64 | `f` `g` | zero-copy | |
| String | `U` large_utf8 | zero-copy | `u` utf8 (offsets widened) |
| Binary | `Z` large_binary | zero-copy | `z` binary (offsets widened) |
| Categorical | `I` indices, `U` dictionary | codes zero-copy, dictionary copied | any integer indices; `u` or `U` values |
| Bool | `b` bool | zero-copy | |
| Date | `tdD` date32 | days narrowed to Int32 | `tdm` date64 |
| Datetime(unit, time_zone) | `ts{m,u,n}:<zone>` timestamp | zero-copy | `tss:<zone>` (as ms) |
| Duration(unit) | `tD{m,u,n}` duration | zero-copy | `tDs` (as ms) |
| Time | `ttn` time64[ns] | zero-copy | `ttu`, `ttm`, `tts` |

Validity bitmaps are always shared. A sliced column exports its window offset
as the ArrowArray `offset` instead of copying. Only Date builds a new value
buffer (Int64 days narrowed to Arrow's Int32).

A zone-aware datetime exports its zone after the colon (`tsu:Europe/Paris`,
`tsu:+05:30`) and holds UTC values, as Arrow requires; import reads the zone
back and raises when the zone database does not know it.

A categorical exports as Arrow dictionary encoding: its UInt32 codes are the
index array and its dictionary the `dictionary` array (large_utf8). Import
keeps a producer's dictionary and codes when the values are distinct and
non-null and every index is in range, so a dictionary array round-trips
unchanged; otherwise the values are decoded and encoded again. A Parquet file
written from a categorical reads back as String.

Import rejects, with an error naming the format: nested types other than the
top-level struct, and struct arrays with null rows.

## Lifetimes and ownership

- **Export.** The caller owns the two structs, and the exporter fills them.
  The exported buffers are reference-counted and stay alive until the
  consumer calls each struct's `release` exactly once, even if the source
  frame is destroyed first. The consumer may move child arrays out (as pyarrow
  does) and release the parent early. Each child keeps its own buffers alive
  until its own `release`.
- **Import.** By default, import copies values into new buffers and releases
  both producer structs. With `copy=False`, compatible payload buffers share
  one retained parent-array owner. Each buffer keeps that owner alive, and
  its callback runs exactly once after the last view is gone. Error paths
  release any partial result and every consumed export. Importing an already
  released struct (`release == NULL`) raises.
- Exported buffers must not be modified by the consumer. The same buffers may
  back other columns and frames.

## Cost

At 1,000,000 rows (Threadripper 3970X):

| column | export | import |
|---|---|---|
| Int64 / Float64 / String / Bool | 0.18 ms | 0.7 / 0.7 / 1.7 / 0.2 ms |

Zero-copy export only fills the structs and counts nulls with a popcount.
Import is a bulk memory copy.

## Testing

[`test_arrow.mojo`](../tests/interchange/test_arrow.mojo) and
[`test_arrow_edges.mojo`](../tests/interchange/test_arrow_edges.mojo) exercise typed
cells, offset buffers, nested/dictionary layouts, arbitrary field metadata,
malformed inputs and exactly-once release callbacks. Expected values come from
independent fixtures and typed literals. Run `pixi run test --contracts io.arrow`.

After the interchange suite, `pixi run -e oracle test-interchange-oracles`
validates independent PyArrow fixtures and consumes actual Parquet/GeoParquet
writer artifacts. The [coverage guide](design/test-coverage-plan.md) describes boundary
and lifetime evidence; a roundtrip alone does not establish interoperability.

Large record batches can copy columns in parallel on Linux. Release remains
owned by the caller after all jobs finish; Parquet reuses the pool between
row groups. See [measurements and platform policy](design/arrow-import.md).

## Field metadata and extensions

Arrow field metadata is copied on import and owned independently of the producer.
Keys and values remain arbitrary bytes, including embedded NULs and non-UTF-8
bytes. Exported schemas own their metadata until their release callback runs.
Metadata on supported nested list and struct fields follows their child Series.

Unknown extension types use their supported storage dtype and retain both
`ARROW:extension:name` and `ARROW:extension:metadata`. A consumer with the
extension registered can reconstruct it. `geoarrow.wkb` additionally receives
geometry validation; other extensions, including other GeoArrow encodings,
remain opaque. This does not add support for otherwise unsupported storage
layouts.

Projection, aliases, rename, filter, sort, take (including null extension),
slice, chunk views, and rechunk preserve field metadata. Vertical concatenation
requires identical key/value pairs, independent of their order; conflicting or
missing metadata raises. Known geometry extension declarations are compared
through their normalized logical dtype, so freshly constructed and imported
geometry columns can agree. Diagonal concatenation assigns the existing column's
metadata to inserted null rows. Computed results and dtype-changing conversions
have no general metadata-preservation guarantee; explicit logical retagging
clears metadata when the dtype changes, avoiding stale extension declarations.
`from_wkb` and `to_wkb` replace/remove extension declarations while retaining
unrelated application metadata. Root schema application metadata is outside
this field-metadata contract.

Run registered/unknown extension and field-metadata contracts with
`pixi run test --contracts io.arrow`.
