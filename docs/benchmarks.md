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
- The header line records the configuration, including the worker-thread
  limit (`DATAFRAME_THREADS`, default: physical cores). Run with
  `DATAFRAME_THREADS=1` for single-threaded numbers.

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

## After shared buffers (#34)

Columns became windows onto reference-counted buffers, so per-batch slices and
projections no longer copy. At 1,000,000 rows (no nulls):

| workload | before ms | after ms |
|---|---|---|
| global_sum | 11.8 | 7.2 |
| global_count | 9.0 | 4.5 |
| grouped_sum_count (low cardinality) | 43 | 28 |
| filter | 36.1 | 38.7 |
| arithmetic_chain (simd4) | 22.3 | 24.0 |
| nullable_compare | 18.8 | 20.5 |

Scans that only read slices gain the most. Fused arithmetic, comparisons, and
filters pay roughly 5-8% for the extra indirection (offset plus shared-pointer
dereference) on per-row validity reads; hoisting buffer references in those
kernels, or word-at-a-time validity, is the planned follow-up.

## After contiguous UTF-8 strings (#33)

String columns moved from `List[String]` to one UTF-8 buffer with Int64
offsets (Arrow `large_utf8`). Throughput on the string workloads is unchanged
within noise; the per-row work (hashing, CSV tokenizing, rank merging) dominates,
not string storage:

| workload | before ms | after ms |
|---|---|---|
| CSV read, 100k rows with a quoted string field | 92.7 | 94.7 |
| group_by string + int key, 16 / 10k / 111k groups | 10.7 / 12.5 / 27.2 | 10.6 / 12.6 / 28.6 |
| sort one string key (ranked), 200k rows | 90.0 | 91.5 |
| sort one string key (comparator reference), 200k rows | 120.9 | 81.8 |

The comparator sort gains because comparisons read borrowed slices instead of
copying Strings. Memory falls: peak RSS for a 2M-row column plus a 1M-row
`take` is 104 → 92 MB for ~10-byte keys (which `String` already stores inline)
and 231 → 156 MB for ~30-byte strings, since each row costs its bytes plus an
8-byte offset instead of a 24-byte `String` and a separate heap allocation.

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

## Parallel reductions (#6, #8)

Reductions split rows into one contiguous partition per worker thread; each
worker reduces its partition into private state, and states merge in
partition order. At 1,000,000 rows (32-core Threadripper; default workers =
min(physical cores, rows / 65536) = 15):

| workload | 1 thread | default | |
|---|---|---|---|
| global_sum | 5.2 ms | 1.4 ms | 3.7x |
| global_count | 3.3 ms | 1.7 ms | 1.9x |
| grouped_sum_count, 16 groups | 26.8 ms | 22.0 ms | hashing (serial) dominates |
| grouped_sum_count, ~100k groups | 40.2 ms | 38.2 ms | workers capped by group count |

Grouped reductions give each worker state for every group, so the worker
count is capped at rows / (4 x groups). Hash grouping itself, row-wise
expressions, and filters are still single-threaded (#5, #7).

## Direct SIMD loads in unfused float kernels (#3)

Unfused float kernels (Float32, `%`, `//`, `**`, clip, unary math, and any
expression fusion does not cover) now load contiguous vectors straight from
the shared column buffers instead of gathering lane by lane, and apply
validity as a vector mask. Single-threaded, 1,000,000 rows:

| expression | before | after |
|---|---|---|
| Float64 `x % 7` | 20.0 ms | 13.8 ms |
| Float32 `y * 2 + 1` | 32.6 ms | 22.0 ms |
| Float64 `sqrt(x)` | 15.5 ms | 10.6 ms |

Remaining allocations per batch are the output values, the validity list,
and one intermediate per unfused node; source windows allocate nothing.

## Row-parallel expressions and filters (#5, #7)

Row-shaped expressions evaluate one contiguous, batch-aligned partition per
worker and join the pieces in order. Filters compact in parallel: workers
collect true-row indices per partition, then gather rows into disjoint output
ranges aligned to 8 rows (no shared validity bytes). 1,000,000 rows, no
nulls:

| workload | 1 thread | default (15 workers) |
|---|---|---|
| arithmetic_chain (fused SIMD) | 23.5 ms | 6.3 ms |
| arithmetic_chain (scalar) | 63.7 ms | 15.0 ms |
| nullable_compare | 19.9 ms | 4.7 ms |
| filter (end to end) | 37.8-42.4 ms | 10.7 ms |

Crossover: below 2 x 65,536 rows everything stays single-threaded, which
avoids thread startup (~20-50 us per worker) on small frames. Joining the
per-worker pieces is a serial O(n) copy and is the main remaining cost at
high worker counts.

