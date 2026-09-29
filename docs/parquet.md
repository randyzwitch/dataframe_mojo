# Parquet files

`read_parquet(path, columns=..., row_groups=...)` reads selected fields and
row groups. `write_parquet(frame, path, compression="zstd",
row_group_size=1_000_000)` writes a local file. Both use the optional
`libdfparquet` library built with `pixi run -e native build-dfparquet`.

## Writing

```mojo
from dataframe import read_parquet, write_parquet

var frame = read_parquet("input.parquet")
write_parquet(frame, "output.parquet", compression="snappy", row_group_size=250_000)
```

Compression is `zstd`, `snappy` or `uncompressed`. A row-group size must be
positive and is an upper bound, not a required frame length. Existing files
are replaced. I/O failures raise; a failure after opening the file can leave
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

## Validation

`tests/test_write_parquet.mojo` covers all supported numeric widths, temporal
units, nested data, nulls, chunked/sliced input, all three codecs, row-group
sizes, empty typed frames, invalid options, I/O errors and release counts.
`scripts/check_parquet_write.py` reads files produced by the Mojo oracle with PyArrow and checks
schemas, values, codecs and temporal metadata independently of our reader.
