# Parallel Arrow import (#276)

Independent record-batch columns can be copied on a scoped worker pool.
Workers borrow child structs and never invoke release callbacks. The caller
waits for every job, assembles columns in schema order, and releases the
parent array/schema exactly once, including when a child import fails.
Parquet reuses one pool across row groups. Single-column import stays serial.

## Measurements and policy

Measured with Mojo 1.2.0.dev2026092105 (e9569894), unpinned, on Linux AMD
Threadripper 3970X (32 physical cores) and an Apple M1 Mac mini (4 performance
and 4 efficiency cores, 16 GiB RAM). Sweeps use 4/8/16 configured workers,
65,536/125k/250k/1M/10M rows, and two/eight Int64 columns. Import-only timing
excludes export and input construction; each binary warms once and checks
complete values after every repetition. Two rounds reverse baseline/candidate
order, with five samples each. Read timing uses three alternating fresh
processes and eight-column zstd fixtures with 250k-row groups. Table values
are medians at eight configured workers; [raw data](arrow-import.csv) retain
all settings and the rejected M1 parallel trial.

| Machine / case | Serial baseline | Final policy |
|---|---:|---:|
| Linux, standalone 250k × 2 | 0.093 ms | 0.093 ms |
| Linux, standalone 10M × 8 | 66.03 ms | 31.76 ms |
| Linux, read 1M × 8 | 74.22 ms | 45.52 ms |
| Linux, read 10M × 8 | 581.05 ms | 461.69 ms |
| M1, standalone 10M × 8 | 21.92 ms | 21.78 ms |
| M1, read 10M × 8 | 196.96 ms | 198.21 ms |

Standalone thread startup lost on Linux for 250k × 2: 0.09 → 0.42 ms in the
initial trial, while 1M × 2 and 250k × 8 won. Require at least two million
cells before starting a standalone pool, plus the shared 64k rows/worker
floor. This is a conservative measured guard for these fixed-width copies,
not a universal byte model for strings or nested columns. Reused Parquet
pools do not pay startup per batch and use the existing rows/worker floor.

M1 showed no useful parallel-copy win: four-worker 10M × 8 was about
21.76 → 22.12 ms; eight workers regressed to 28.89 ms. macOS therefore stays
serial pending evidence from another Mac. A small difference in final read
times is noise around unchanged scheduling, not a claimed speedup. Linux
streamed reads improve about 21% at 10M rows, but still perform a copy; the
old whole-file decode timings from the issue are not comparable to these
row-group reads. No zero-copy or decode-only-time claim is made.

The driver waits for detected Mojo compiler activity and discards runs if a
compiler is detected afterward (`BENCH_WAIT_FOR_COMPILERS=1`). This does not
exclude all host activity. Both binaries use stream baseline `eafdd9c` and
the same native library; worker-pool calibration in #292 is separate.

## Reproduction

```bash
pixi run python3 scripts/bench_arrow_import.py --build
# Generate fixtures once using scripts/bench_parquet_stream.py --run.
BENCH_WAIT_FOR_COMPILERS=1 pixi run python3 scripts/bench_arrow_import.py --run
```

Set `DATAFRAME_PARQUET_LIBRARY` when the native library is elsewhere. Run
on both machines; CSVs are written to `build/import/results.csv`. The final
Mac policy is serial; reproducing its rejected trial requires removing the
macOS fallback in `_arrow_import_workers` in an experimental checkout.
