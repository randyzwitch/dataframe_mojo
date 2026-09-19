# CPU benchmarks

`pixi run bench` runs `benchmarks/bench_suite.mojo` and prints CSV: one line per
workload with rows, null density, kernel width, group shape, batch size,
repetitions, best and mean nanoseconds, rows per second of the best run, and a
checksum. Inputs are generated from a fixed seed before timing; each workload is
evaluated once untimed, then timed five times, and results are validated outside
the timed region (scalar and SIMD arithmetic must produce identical checksums).

- `BENCH_SMOKE=1` (`pixi run bench-smoke`) uses tiny inputs and one repetition.
  CI runs it as a correctness check; no timing thresholds are enforced on shared
  runners.
- `BENCH_LARGE=1` adds 10,000,000-row inputs.
- Peak memory is measured externally: `/usr/bin/time -v pixi run bench` on
  Linux, `/usr/bin/time -l` on macOS. The suite does not count allocations; that
  needs an allocator hook Mojo does not expose, and timing is not used as a proxy.
- The header line records the configuration. Thread count is reported for later
  comparisons; execution is single-threaded today.

Other focused benchmarks: `bench-csv`, `bench-concat`, `bench-sort`,
`bench-group-by`, and `bench-join`.

## Baseline (Mojo 1.1.0, AMD Ryzen Threadripper 3970X, Linux x86-64)

`mojo=1.1.0 seed=20260918 physical_cores=32 native_f64_simd_width=4 threads=1 batch_size=1024`

| workload | rows | nulls | kernel | groups | best ms | rows/s (M) |
|---|---|---|---|---|---|---|
| arithmetic_chain | 1,000 | none | scalar |  | 0.06 | 16.3 |
| arithmetic_chain | 1,000 | none | simd4 |  | 0.06 | 17.4 |
| nullable_compare | 1,000 | none | simd4 |  | 0.03 | 37.7 |
| filter | 1,000 | none | simd4 |  | 0.03 | 28.7 |
| global_sum | 1,000 | none | scalar |  | 0.01 | 80.5 |
| global_count | 1,000 | none | scalar |  | 0.01 | 105.2 |
| arithmetic_chain | 1,000 | 10pct | scalar |  | 0.06 | 16.8 |
| arithmetic_chain | 1,000 | 10pct | simd4 |  | 0.06 | 18.0 |
| nullable_compare | 1,000 | 10pct | simd4 |  | 0.03 | 38.5 |
| filter | 1,000 | 10pct | simd4 |  | 0.03 | 30.2 |
| global_sum | 1,000 | 10pct | scalar |  | 0.01 | 82.9 |
| global_count | 1,000 | 10pct | scalar |  | 0.01 | 105.9 |
| grouped_sum_count | 1,000 | 10pct | scalar | low | 0.04 | 22.4 |
| grouped_sum_count | 1,000 | 10pct | scalar | high | 0.07 | 14.7 |
| grouped_sum_count | 1,000 | 10pct | scalar | skewed | 0.06 | 15.9 |
| arithmetic_chain | 100,000 | none | scalar |  | 6.92 | 14.4 |
| arithmetic_chain | 100,000 | none | simd4 |  | 6.50 | 15.4 |
| nullable_compare | 100,000 | none | simd4 |  | 2.69 | 37.2 |
| filter | 100,000 | none | simd4 |  | 3.60 | 27.8 |
| global_sum | 100,000 | none | scalar |  | 1.17 | 85.4 |
| global_count | 100,000 | none | scalar |  | 0.88 | 113.6 |
| arithmetic_chain | 100,000 | 10pct | scalar |  | 6.71 | 14.9 |
| arithmetic_chain | 100,000 | 10pct | simd4 |  | 6.30 | 15.9 |
| nullable_compare | 100,000 | 10pct | simd4 |  | 2.61 | 38.3 |
| filter | 100,000 | 10pct | simd4 |  | 3.38 | 29.6 |
| global_sum | 100,000 | 10pct | scalar |  | 1.11 | 90.4 |
| global_count | 100,000 | 10pct | scalar |  | 0.85 | 117.7 |
| grouped_sum_count | 100,000 | 10pct | scalar | low | 3.79 | 26.4 |
| grouped_sum_count | 100,000 | 10pct | scalar | high | 6.98 | 14.3 |
| grouped_sum_count | 100,000 | 10pct | scalar | skewed | 6.42 | 15.6 |
| arithmetic_chain | 1,000,000 | none | scalar |  | 183.28 | 5.5 |
| arithmetic_chain | 1,000,000 | none | simd4 |  | 191.88 | 5.2 |
| nullable_compare | 1,000,000 | none | simd4 |  | 37.14 | 26.9 |
| filter | 1,000,000 | none | simd4 |  | 46.02 | 21.7 |
| global_sum | 1,000,000 | none | scalar |  | 11.66 | 85.8 |
| global_count | 1,000,000 | none | scalar |  | 8.82 | 113.3 |
| arithmetic_chain | 1,000,000 | 10pct | scalar |  | 184.45 | 5.4 |
| arithmetic_chain | 1,000,000 | 10pct | simd4 |  | 198.28 | 5.0 |
| nullable_compare | 1,000,000 | 10pct | simd4 |  | 35.85 | 27.9 |
| filter | 1,000,000 | 10pct | simd4 |  | 44.09 | 22.7 |
| global_sum | 1,000,000 | 10pct | scalar |  | 11.48 | 87.1 |
| global_count | 1,000,000 | 10pct | scalar |  | 8.99 | 111.2 |
| grouped_sum_count | 1,000,000 | 10pct | scalar | low | 42.38 | 23.6 |
| grouped_sum_count | 1,000,000 | 10pct | scalar | high | 55.54 | 18.0 |
| grouped_sum_count | 1,000,000 | 10pct | scalar | skewed | 53.84 | 18.6 |

## After fixing quadratic batch reassembly

The baseline exposed a bug: appending each evaluated batch to the output
reserved exactly the new length, so every append reallocated and copied the
whole column, making reassembly quadratic in the number of batches. With
geometric growth, the 1,000,000-row workloads became:

| workload | nulls | before ms | after ms |
|---|---|---|---|
| arithmetic_chain (simd4) | none | 191.9 | 58.3 |
| arithmetic_chain (scalar) | none | 183.3 | 61.8 |
| nullable_compare | none | 37.1 | 26.7 |
| filter | none | 46.0 | 36.1 |
| global_sum / global_count | none | 11.7 / 8.8 | 11.8 / 9.0 |

The analysis below still holds: per-node materialization and copying dominate
the remaining arithmetic time, and SIMD is only slightly faster than scalar.

## After elementwise fusion (#4)

Fused Float64 subtrees evaluate in one SIMD pass reading source buffers
directly. At 1,000,000 rows the 4-lane arithmetic chain fell from 58.3 ms to
22.3 ms (183 ms at the original baseline), and the nullable comparison from
26.7 ms to 18.8 ms; checksums are unchanged. The 1-lane path gains nothing,
since per-row program interpretation replaces per-node materialization.

## What the baseline shows

- The five-node Float64 arithmetic chain runs at about 5 million rows/s, and the
  4-lane SIMD kernel is no faster than the 1-lane one. Arithmetic is not the
  bottleneck: every node materializes a new batch `Series`, source columns are
  sliced by copying values and rebuilding validity one element at a time, and
  batch results are appended into the output. Borrowed column views (#3) and
  fused elementwise kernels (#4) target exactly this.
- A single comparison runs at about 27 million rows/s and a filter at about 22
  million, both again dominated by per-batch copies and `take`.
- Global sums and counts reach 85-110 million rows/s: they read slices and
  update one state, with little materialization.
- Grouped sum+count is 18-24 million rows/s and slows with cardinality, as
  expected from hashing and per-group state.
- Null density (10%) barely changes any workload, because validity is handled
  per element either way.

No workload here claims a speedup; these numbers are the reference for the
execution changes in #3-#8.
