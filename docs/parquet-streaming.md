# Parquet row-group streaming (#277)

`read_parquet` imports one selected row group at a time through Arrow's C
stream interface. Each imported batch becomes a Series chunk. The native
reader retains the file reader and schema, while its current decode buffers
are bounded by a row group. The eager result still retains every selected
row; this is not a constant-memory lazy scan. Projection and requested
row-group order, including duplicates, are preserved.

The legacy whole-file C entry point remains available for ABI compatibility.
The Mojo loader requires the new stream symbol, so existing installations
must rebuild `libdfparquet`.

## Memory and time

Measured on an AMD Threadripper 3970X Linux host with Mojo
1.2.0.dev2026092105 (e9569894). Fixtures have eight Int64 columns, Zstandard
compression, no dictionary encoding, and 250,000 rows per row group. Fixture
generation happens in a separate process. Each read runs in a fresh process;
peak RSS comes from `wait4`, includes the runtime and final eager result, and
is not an allocation count. Three paired runs alternate baseline/candidate
order. Reported values are medians; [raw results](parquet-streaming.csv)
include all runs. The driver waits for detected Mojo compiler activity and
discards interrupted runs; other host activity remains possible.

Both variants use the same native library. The baseline is Mojo source at
`b2013a5`, invoking the retained whole-file entry point; the candidate imports
row-group batches through the stream. Read timing includes materialization;
fixture creation is excluded. Row count and chunk count are checked.

| Rows | Whole-file time | Stream time | Whole-file peak RSS | Stream peak RSS | Result chunks |
|---:|---:|---:|---:|---:|---:|
| 1 million | 71.37 ms | 65.22 ms | 171.50 MiB | 114.18 MiB | 4 |
| 10 million | 548.11 ms | 582.33 ms | 1503.53 MiB | 677.96 MiB | 40 |

At ten million rows, streaming lowers peak RSS by about 55%, with a roughly
6% increase in median read time. Batch import and chunk assembly have a cost;
this change targets peak memory. Wider, nested, differently compressed or
larger-row-group files can behave differently.

## Validation and reproduction

Linux and Apple M1 Mac tests cover all thirteen Parquet integration cases
and explicit stream cleanup on EOF, next-batch errors, schema errors and
Arrow-import errors. The native PyArrow check covers projection, reordered
and duplicate groups, empty selections, repeated EOF, early close, and a
corrupted later group. Series preserve the row-group chunk boundaries.

```bash
pixi run -e native build-dfparquet
pixi run python3 scripts/bench_parquet_stream.py --build
BENCH_WAIT_FOR_COMPILERS=1 pixi run -e oracle python3 scripts/bench_parquet_stream.py --run
pixi run -e oracle python3 scripts/check_parquet_stream.py build/dfparquet/libdfparquet.so
```

The optional Java discovery in bundled Thrift is disabled because Java is
not used by this library. Local Mac validation needed an external compiler
wrapper to point its installed Command Line Tools at the SDK's C++ headers;
that environment workaround is not part of the repository or measurements.
