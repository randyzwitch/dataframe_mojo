# Parquet files

`read_parquet(path, columns=..., row_groups=...)` reads selected fields and
row groups. `write_parquet(frame, path, compression="zstd",
row_group_size=1_000_000)` writes a local file. Both use the optional
`libdfparquet` library built with `pixi run -e native build-dfparquet`.

## Reading local datasets

`read_parquet` and `scan_parquet` accept a path, a list of paths, or glob
patterns. Files are processed in sorted path order. Eager `columns` selects
fields in requested order; `row_groups` applies to each file, preserving
requested row-group order and repetitions. An empty row-group list returns
schema without rows.

Both readers accept `glob=True`, `hive_partitioning=False`,
`hive_schema=Dict[String, String]()`, `include_file_paths=""`,
`missing_columns="raise"`, `extra_columns="raise"`, and
`union_by_name=False`. Hive mode adds `key=value` directory fields; a
nonempty `include_file_paths` names an additional source-path String field.
Use `missing_columns="insert"` for typed nulls and `extra_columns="ignore"`
to discard fields absent from the first file, or `union_by_name=True` to
include every field. Shared required dtypes must remain compatible.
See [multiple local files](datasets.md) for partition types, schema alignment,
path ordering, and file pruning.

## Writing

```mojo
from dataframe import read_parquet, scan_parquet, write_parquet

var frame = read_parquet("input.parquet")
write_parquet(frame, "output.parquet", compression="snappy", row_group_size=250_000)
```

Compression is `zstd`, `snappy` or `uncompressed`. A row-group size must be
positive and is an upper bound, not a required frame length. With the default
`partition_by=[]`, this eager single-file writer replaces existing files
directly. I/O failures raise; a failure after opening the destination can leave
a partial file. The caller's frame remains usable and unchanged.

The writer exports the frame through the Arrow C Data Interface and imports
it into Arrow's Parquet writer. Compatible contiguous buffers are shared;
chunked columns may be rechunked by the existing export path. This is eager
writing, not a bounded-memory streaming sink. The native backend consumes
exports on import; Mojo releases any exports remaining after an error.

Arrow schema metadata preserves signed/unsigned integer widths, floating
widths, strings, booleans, dates, timestamp/duration units, time, and nested
list/struct fields. Nulls and empty typed frames are preserved. Other readers
that ignore Arrow schema metadata may expose durations as their physical
integer representation. Binary columns write as `large_binary`. Timestamp zones cannot be restored after a read that discarded them.

The writer resolves `dfq_write_parquet` only when called. Reader-only use can
still load a library with the existing read ABI; rebuild an older library
before using the new writer.

## Lazy sinks and partitioned output

```mojo
scan_parquet("input/*.parquet").sink_parquet(
    "output.parquet", compression="zstd", row_group_size=250_000, batch_size=65536
)
write_parquet(frame, "sales", partition_by=[String("year"), "month"])
```

`LazyFrame.sink_parquet` takes `compression="zstd"`,
`row_group_size=1_000_000`, `batch_size=65536`, and `optimize=True`. It consumes
execution batches directly; row groups span batch boundaries. It publishes
a temporary sibling file only after successful completion, preserving an
existing destination on error. A scan can sink to its own input path.
Stateful query operators retain their existing memory requirements; see
[lazy file sinks](lazy.md#file-sinks).

A nonempty `partition_by` changes eager `write_parquet` to a Hive directory
writer, producing one `part-00000.parquet` per key tuple. Partition columns
remain in each file, and rows keep input order within their partition. Names
and values are escaped, including literal null markers. The root must be
absent or empty. The entire tree is staged beside it and published after all
files succeed; an empty frame creates an empty root. These publication rules
apply to partitioned output and lazy sinks; eager single-file
`write_parquet` retains the direct-write failure behavior described above.
See [dataset output](datasets.md#output) for details.

Rebuild `libdfparquet` before using sinks: incremental writing requires its
writer open/write/close/abort entry points in addition to the eager writer.

## Validation

[`test_parquet.mojo`](../tests/interchange/test_parquet.mojo) covers numeric widths, temporal
units, nested data, nulls, chunked/sliced input, all three codecs, row-group
sizes, empty typed frames, invalid options, I/O errors and release counts.
After `pixi run test --areas interchange`, run
`pixi run -e oracle test-interchange-oracles`. Its PyArrow consumer reads actual
public writer artifacts and checks schemas, values, codecs and temporal metadata
independently of our reader. Build `libdfparquet` first as described in
[the testing guide](testing.md).
