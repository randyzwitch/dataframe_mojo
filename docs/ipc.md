# Arrow IPC and Feather v2

`read_ipc` and `write_ipc` read and write local Arrow IPC files (Feather v2).
`read_ipc_stream` and `write_ipc_stream` use the sequential IPC stream format.
`scan_ipc` builds a lazy file scan with projection and slice pushdown.
They use the same optional native library as Parquet, built with Arrow IPC
enabled; Python is not required at runtime.

```mojo
from dataframe import read_ipc, read_ipc_schema, write_ipc

var frame = read_ipc("features.arrow", columns=["id", "score"], n_rows=1000)
write_ipc(frame, "subset.arrow", compression="zstd", batch_size=65536)
var fields = read_ipc_schema("features.arrow")
```

Build the native dependency with `pixi run -e native build-dfparquet`.
Library lookup follows [Parquet](parquet.md), including
`DATAFRAME_PARQUET_LIBRARY`. Older native libraries need rebuilding for the
new IPC entry points.

## Reading

Column selection preserves the requested order. Omit `columns` or pass
`None` to select all fields; an empty list returns zero columns while
retaining the selected row count. `n_rows` must be nonnegative; zero returns
an empty typed frame. A limit stops before reading subsequent record batches,
although the last selected batch may be decoded in full before slicing.

Each record batch becomes a column chunk. Categorical dictionaries changing
between batches are unified and codes remapped to preserve their values.
Integer and float widths, decimal precision/scale/storage widths, temporal
units and timestamp zones follow the Arrow interchange contracts. UTF-8 and
binary views retain their native descriptors and byte blocks. Arbitrary
field metadata is retained. Unsupported Arrow types raise rather than
silently dropping fields.

`read_ipc_schema(path)` reads names and dtypes from the file header/footer
without reading record-batch bodies. Pass `stream=True` for an IPC stream.
Its result is the same `List[Field]` returned by `DataFrame.schema()`.
Dictionary labels live in data messages, so schema-only categorical entries
have an empty dictionary.

`memory_map=True` selects a memory-mapped native input (the default for files;
streams default to false). Compatible uncompressed buffers are shared with
columns, slices and chunks. The mapping stays alive after the reader closes
and is released after the final dependent view or Arrow export. Keep mapped
file contents unchanged while these views exist. With `memory_map=False`,
columns retain native read buffers instead. Compressed buffers must decode;
the importer then retains decoded buffers without another payload copy.

Borrowing retains each complete Arrow buffer. In particular, compressed
string/binary views can retain large decoded byte blocks even when a batch
references only a small part of each block. Such files can use more resident
memory than converting their logical values into compact owned strings.
Avoid assuming that fewer payload copies always means lower peak RSS.

Numeric values, validity and Boolean bitmaps, string/binary bytes and views,
large offsets, nested children, and compatible categorical codes share their
buffers. Int32 offsets widen, incompatible dictionary indices and temporal
units convert, and unaligned typed storage uses aligned copies. Rechunking or
requesting a canonical contiguous Arrow export can also allocate. Mutating an
alias detaches its storage first. See [Arrow ownership](arrow.md).

## Lazy file scans

```mojo
from dataframe import scan_ipc, col

var query = (
    scan_ipc("features.arrow")
    .filter(col("score") > 0.9)
    .select(["id", "score"])
    .head(100)
)
var result = query.collect(engine="cpu", batch_size=65536)
```

Constructing a scan does not open the file. `collect_schema` reads only
schema metadata, and each collection opens a fresh reader. The optional
`memory_map` argument has the same buffer-sharing semantics as `read_ipc`.

Projection includes the columns needed by filters, join keys and aggregates,
even when those columns are omitted from the final result. Nonnegative slices
push their offset and length into the native reader, including through
row-local projections. The reader checks record-batch metadata to skip whole
batches without reading their bodies, then decodes only intersecting batches.
A zero-column projection needs only row-count metadata. Arrow may load
file-level dictionary messages before the selected record batches.

Slices stay above filters, sorts, joins and row-order operations whenever
moving them would change which rows are returned. IPC has no Parquet-style
min/max statistics for pruning batches by a filter. Negative offsets retain
the ordinary materializing slice semantics.

The streaming executor imports one record batch at a time and hands smaller
windows to the existing batch operators. Sorts and other global operators
retain their materialization boundaries, and collection retains the final
output. File dictionary deltas resolve to a consistent dictionary across
batches, including nested categorical children. Early stopping and errors
release the native reader and any pending imported batch.

IPC scans need the native library rebuilt with the `dfq_scan_ipc` entry point.
Lazy scanning of the sequential IPC stream format is separate from file scans.

## Writing

Both writers accept `compression="uncompressed"`, `"lz4"`, or `"zstd"` and
a positive `batch_size`. A batch stops at the nearest input column chunk
boundary or the size limit. Differently chunked columns are aligned without
rechunking the complete frame. Empty frames retain their schema, and
zero-column frames retain their row count. Nested categorical dictionaries
are unified across input chunks before writing.

Writing replaces the destination. Invalid compression and batch size are
rejected before opening it; an I/O or serialization failure can leave a
partial file. Native handles and exported Arrow arrays are released on failure.

## Validation

Run the registered Mojo contracts with `pixi run test-cpu --contracts io.ipc`.
Fixtures are generated independently by
`pixi run -e oracle python3 tests/support/ipc_oracle.py` and include file and
stream codecs, dictionary replacements/deltas, nested and temporal types,
decimal widths, and deliberately corrupted later batch bodies.

After the interchange suite, `pixi run -e oracle test-interchange-oracles`
checks actual Mojo writer outputs using independent PyArrow consumers. The
consumer checks complete cells, physical types, field metadata, and record
batch boundaries. Run these checks on Linux and macOS.

Lazy output sinks depend on #796.
