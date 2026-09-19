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
| Int64 | `l` int64 | zero-copy | `c` `s` `i` `C` `S` `I` (widened) |
| Float64 | `g` float64 | zero-copy | `f` float32 (widened) |
| String | `U` large_utf8 | zero-copy | `u` utf8 (offsets widened) |
| Bool | `b` bool | values packed to bits | |
| Date | `tdD` date32 | days narrowed to Int32 | `tdm` date64 |
| Datetime(unit) | `ts{s,m,u,n}:` timestamp | zero-copy | |
| Duration(unit) | `tD{s,m,u,n}` duration | zero-copy | |
| Time | `ttn` time64[ns] | zero-copy | `ttu`, `ttm`, `tts` |

Validity bitmaps are always shared. A sliced column exports its window offset
as the ArrowArray `offset` instead of copying. Bool and Date export build new
value buffers (Bool values are stored one byte per value internally).

Import rejects, with an error naming the format: UInt64 (`L`, not
representable in Int64), timestamps with a time zone, dictionary-encoded
arrays, nested types other than the top-level struct, and struct arrays with
null rows.

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
| Int64 / Float64 / String | 0.18 ms | 0.7 / 0.7 / 1.7 ms |
| Bool | 0.85 ms | 2.0 ms |

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
