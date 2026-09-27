# Lazy streaming: time and resident memory (#109)

The default lazy executor runs supported row-wise operators as ordered
batches, merges aggregate state and streams supported join probe sides.
`collect(streaming=False)` retains the materializing executor. The
[execution guide](lazy.md) lists supported state, boundaries and limitations.

## Method

Measured on 2026-09-26/27 with Mojo 1.2.0.dev2026092105 (e9569894):

- Linux AMD Threadripper 3970X, 32 physical/64 logical CPUs, 512 KiB L2/core,
  eight 16 MiB L3 domains, schedutil, unpinned.
- Darwin 25.5 Apple M1 Mac mini, four performance/four efficiency cores,
  16 GiB RAM, 12 MiB performance-cluster and 4 MiB efficiency-cluster L2,
  unpinned. Sixteen configured workers deliberately oversubscribe it.

Both modes use the same compiled binary, source, input and optimizer. The
pipeline is `scan_csv -> filter -> select -> group_by.agg`, with 100 integer
keys, an integer value and a 96-byte padding field. Projection excludes the
padding from decoding but still scans its bytes. Filter and projection apply
row-wise expressions, followed by exact sum and count. Each run checks total
row count and the exact integer sum outside timing.

The 1M/10M fixtures occupy 106,788,902 / **1,077,888,902 bytes**. CSV generation
is excluded. Each mode runs in a fresh process, three paired repetitions
with order alternated, at 4/8/16 configured workers and 65,536 rows per batch.
`wait4` reports peak process RSS, including runtime and retained allocator
pages; this is not an allocation count. Files are reused, so these are warm
filesystem-cache measurements, not cold-disk throughput. Medians are below;
[all samples](lazy-streaming.csv) are committed.

The driver waits for detected Mojo compiler activity and discards a run if
it detects compilation afterward. It cannot exclude all background work;
small timing differences should not be treated as precise crossovers.

## Results

Times are milliseconds; peak RSS is MiB. M = materializing, S = streaming.

| Machine | Rows | Workers | M time | S time | M RSS | S RSS |
|---|---:|---:|---:|---:|---:|---:|
| Linux | 1M | 4 | 178.89 | 144.64 | 140.26 | 58.66 |
| Linux | 1M | 8 | 105.99 | 94.78 | 140.34 | 87.77 |
| Linux | 1M | 16 | 69.42 | 66.38 | 140.96 | 147.73 |
| Linux | 10M | 4 | 1455.24 | 1336.06 | 1229.28 | 62.65 |
| Linux | 10M | 8 | 903.21 | 886.85 | 1211.41 | 101.88 |
| Linux | 10M | 16 | 609.87 | 630.37 | 1218.86 | 172.27 |
| M1 | 1M | 4 | 60.43 | 64.78 | 138.77 | 53.52 |
| M1 | 1M | 8 | 52.49 | 59.79 | 137.73 | 82.66 |
| M1 | 1M | 16 | 54.81 | 58.72 | 140.66 | 138.44 |
| M1 | 10M | 4 | 619.43 | 660.63 | 1272.45 | 55.09 |
| M1 | 10M | 8 | 505.00 | 606.49 | 1254.20 | 91.16 |
| M1 | 10M | 16 | 489.49 | 611.27 | 1268.73 | 159.34 |

At four workers, increasing input tenfold leaves streaming RSS nearly flat:
59 → 63 MiB on Linux and 54 → 55 MiB on M1. The 1 GB run uses about 95%
less peak RSS than materialization. Resident input is bounded by an active
wave (workers × batch size × record width), decoded columns, reducer state
and runtime/allocator overhead. Here, four batches cover about 27 MiB of CSV
bytes; total peak RSS is roughly twice that. More workers intentionally
increase the number of live batches. On a 1M-row input, sixteen live batches
nearly cover the entire file, so memory need not improve.

Linux at four workers is 19% faster at 1M and 8% faster at 10M. The gain is
not universal: Linux 10M at sixteen workers is about 3% slower, and M1 is
7–25% slower across these settings. Batch scheduling, state merging and page
release have a cost; use `streaming=False` when throughput on an in-memory
workload matters more than peak memory. This work establishes bounded
pipelines, not a promise that every plan or machine gets faster.

## Two memory fixes found by measurement

Lazy schema discovery previously used eager `n_rows=0`, which deliberately
parses a source chunk before truncation. That made optimization allocate
large temporary frames before streaming began. Lazy metadata probes now
infer/validate schema and decode zero rows directly; the eager CSV contract
is unchanged.

`madvise(DONTNEED)` released consumed mapped pages on Linux but left the M1
near 1 GB RSS. The cursor now unmaps consumed whole pages after all jobs in
a wave have copied them, updating the remaining mapping owned by its
scope. Tests retain quoted Unicode strings across multiple unmapping waves
and check early-stop/error cleanup. The final table uses explicit unmapping
on both machines.

## Reproduction and validation

```bash
pixi run python3 scripts/bench_lazy_streaming.py --build
BENCH_WAIT_FOR_COMPILERS=1 pixi run python3 scripts/bench_lazy_streaming.py --run
```

The script creates fixtures in `build/lazy-streaming`, runs both modes, and
writes `results.csv`. Repeat on the second machine. The committed artifact
adds a machine column to both raw outputs.

Linux validation covers the existing lazy, nested, Parquet, grouping,
partitioned grouping, expression reduction and CSV scanner modules, plus
new streaming tests and eager CSV reader/options regression tests. M1 runs
cover lazy/Parquet behavior and final streaming, scanner and cursor cleanup.
Randomized tests compare aggregate states and all join types against the
materializing executor, with floating-point tolerance where reassociation
is permitted. Exact integer overflow is checked only after final merging.
