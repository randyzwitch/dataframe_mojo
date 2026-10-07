# Arrow interchange

`dataframe` speaks the [Arrow C Data Interface][cdi]: two C structs,
`ArrowSchema` and `ArrowArray`, that any Arrow implementation can read and
fill. Polars, DuckDB, pyarrow, nanoarrow, and marrow can exchange data with a
`DataFrame` this way. No Arrow library or Python is involved at run time.

[cdi]: https://arrow.apache.org/docs/format/CDataInterface.html

```mojo
from dataframe import ArrowArray, ArrowSchema, export_arrow, import_arrow

var array = ArrowArray()
var schema = ArrowSchema()
export_arrow(frame, array, schema)       # frame -> Arrow struct array ("+s")
var copy = import_arrow(array, schema)   # Arrow struct array -> frame
```

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
- **Import.** Import copies values into new buffers and then calls the
  producer's `release` on both structs, so no foreign memory is retained. The
  structs are released even when import fails. Importing an already released
  struct (`release == NULL`) raises.
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

`tests/test_arrow.mojo` round-trips every dtype, window offsets, and empty
frames, and checks that every release runs exactly once. `pixi run -e oracle
oracle-arrow` (dev only) checks interop with pyarrow, in both directions:
- exports pass `validate(full=True)`;
- pyarrow reads our buffers in place (buffer addresses match);
- pyarrow-produced arrays, including sliced ones, import correctly;
- unsupported types are rejected and still released.

Large record batches can copy columns in parallel on Linux. Release remains
owned by the caller after all jobs finish; Parquet reuses the pool between
row groups. See [measurements and platform policy](arrow-import.md).

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
missing metadata raises. Diagonal concatenation assigns the existing column's
metadata to inserted null rows. Computed results and dtype-changing conversions
have no general metadata-preservation guarantee; explicit logical retagging
clears metadata when the dtype changes, avoiding stale extension declarations.
Root schema application metadata is outside this field-metadata contract.

Run the independent registered/unknown extension oracle with
`pixi run -e oracle oracle-arrow-metadata`.
