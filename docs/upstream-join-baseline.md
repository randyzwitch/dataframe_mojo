# Upstream join workload baseline

This starts the post-merge profiling work at `c8c6498` (PR #296).
The engine is unchanged. The first adapters exercise complete analytical
queries, alongside the existing join-result-materialization matrix.

## Provenance and scope

DuckDB source revision: `d8a1bd4f4fbccdf3d22d8f1fe27a94c11c805d0a`.

- [High cardinality join](https://github.com/duckdb/duckdb/blob/d8a1bd4f4fbccdf3d22d8f1fe27a94c11c805d0a/benchmark/micro/join/hashjoin_highcardinality.benchmark):
  1,000 unique left keys, a variable-size unique right table, an inner join,
  grouping by both payloads, count, ascending left payload and limit five.
  Ten million right rows is the upstream size. The result is checked in full.
- [Duplicate string join](https://github.com/duckdb/duckdb/blob/d8a1bd4f4fbccdf3d22d8f1fe27a94c11c805d0a/benchmark/micro/join/hashjoin_dups_rhs.benchmark):
  131,072 left rows, 32,768 distinct `verylargestring`-prefixed keys, a
  variable-size right table, inner join and count. Every right key has exactly
  four left matches. The upstream right table has 134,217,728 rows; the first
  runs deliberately scale it down. These are scaled adaptations, not a claim
  to have run DuckDB's complete suite or original large case.

`base` retains upstream order. `shuffled` deterministically shuffles only the
right rows. For integer keys, `wide` additionally uses an injective mapping
into a 2^40 domain; payloads and expected results stay identical. Thus the
query is unchanged when an ordered-key or bounded-range algorithm cannot
apply. The mapping is restricted to input sizes with safe Int64 arithmetic.
The source SQL uses `count(*)`; the Mojo expression counts a nonnullable join
key and is equivalent for these particular inputs.

Polars benchmark revision `79294cff3b0cd267e3a3589d6687b2783061bf3f`
was inventoried. Its PDS-H analytical queries are a subsequent adapter target;
this initial driver does not claim to implement that suite. Non-equality/asof
joins, spilling, and wider multi-query coverage are also outside this first
increment.

## Measurement contract

All engines load identical generated CSV files with explicit key types before
timing. DuckDB creates native tables before timing; it does not query an Arrow
replacement scan. Mojo and Polars retain their native input frames. Query
planning/execution and final result materialization are timed. These queries
return one or five rows, unlike the existing full-result join matrix.

Each engine runs in a fresh process with one warmup and all raw timed samples
retained. Every repetition checks the complete answer after timing. Engine
order rotates between paired rounds. Thread limits are equal; actual useful
parallelism is an engine decision. Use at least three rounds and five samples.
Compare medians, including round-to-round variation, rather than one best run.

Peak RSS is the fresh worker's high-water mark, obtained with `wait4` on macOS and a GNU `time` helper on Linux. The
Linux helper prevents the parent driver's pre-exec RSS from inflating the
worker's measurement. It
includes input loading, runtime, warmup, and query intermediates, not just
query allocations. Each engine loads CSV into its native representation. These measurements
must not be described as query-only memory. Inputs are cached files, not a cold-disk test.

The JSON sidecar records the engine/compiler versions, source revision, dirty
state, runner digest, input digests, CPU information, command and timestamp.
The driver rejects detected compiler activity before/after each worker. This
does not exclude every kind of background work. Do not run comparisons
concurrently with other benchmark jobs. `--allow-busy` is for correctness
smoke tests only.
The process timeout makes an unexpectedly expensive query fail explicitly;
there is no silently omitted timeout score.

## Reproduce

```bash
mkdir -p build
pixi run mojo build -O3 -g1 -I . benchmarks/bench_upstream_joins.mojo -o build/bench_upstream_joins
pixi run -e oracle python3 scripts/bench_upstream_joins.py --sizes 1000 --reps 1 --rounds 1
pixi run -e oracle python3 scripts/bench_upstream_joins.py --threads 4
```

On Linux, the optimized binary also supports `perf record` / `perf stat`:

```bash
DATAFRAME_THREADS=4 perf record -e cycles:u -g --call-graph dwarf -o build/upstream.data -- \
  build/bench_upstream_joins duplicate_strings LEFT_CSV RIGHT_CSV 20
perf report --stdio -i build/upstream.data
```

Whole-process profiles include CSV loading and warmup. Increase repetitions
and attribute samples by function/call stack; do not label the complete
process profile as join-only. Counter access was verified on the reference
Linux host. Full Xcode was installed on the Mac during this work; Instruments recording
was subsequently verified.

## Linux CPU profiles after build completion

[Compact profile evidence](benchmarks/join-profile-linux.json) records sampled
self CPU percentages on the merged engine, four workers and 1M wide-key rows.
The optimized runner used `-O3 -g1`; `perf` sampled user cycles at 199 Hz with
DWARF call stacks. These recordings were repeated after the unrelated
Bazel/Clang build completed, with no compiler activity detected before or
after recording. Affinity was not pinned; normal desktop background work
remains possible. Loading and warmup remain in the profile; 20 lazy/40 semi
repetitions increase the query contribution. These summaries supersede the
earlier contended-host diagnostic recordings.
Raw `.data` captures remain in `build/profiles` rather than the repository.

For `lazy_narrow`, `_group_rows` (7.07%), `_joint_key_ids` (14.11%) and
`column_codes` (15.41%) are major costs. Swiss-table resize (17.36%) and
dictionary growth (9.02%) together contribute another 26.38%. Call stacks show `_StreamJob.run -> DataFrame.join`.
Source inspection confirms that every batch calls the complete join operation
with the retained right frame, rather than a retained prepared join index.
A 65,536-row probe batch also has only one worker according to the shared
rows/worker floor, so it misses the eager high-cardinality parallel hash route.
The profiling hypothesis is **repeated preparation plus an unsuitable fallback
for batch-sized probes**, not a Mojo compiler limitation.

The separate `semi_unmatched` profile attributes 41.05% to `_HashProbeJob` and
6.94% to `_HashBuildJob`. Visible gather/take entries contribute about 11.5%.
This does not support assuming output gathering is the dominant remaining
cost on this measured case. Inspect probe memory access, dispatch, bounds
checks and table layout before choosing an implementation change.

The first engine follow-up should separate preparation from probing, so a
streaming join prepares its build side once and shares immutable state among
batch jobs. It must preserve duplicates, null behavior, key equality, output
order and cleanup. Validate small/large probes, multiple batches, strings,
compound keys, skew, and all supported streamed join modes. No production
algorithm or dispatch threshold is changed by this benchmark PR.

## Apple M1 baseline (four workers)

Mojo 1.2.0.dev2026092105 (e9569894), Polars 1.44.2, DuckDB 1.5.5;
macOS 26.5.2, M1 with four performance and four efficiency cores, 16 GiB RAM.
Three rotated rounds, five samples after one warmup per engine/round.
Times are medians of the three per-round medians, in milliseconds.
Both dataframe engines construct their lazy plans inside each timed query.
Every timed result passed the full expected-answer check. The host was
otherwise quiet according to its owner; CPU affinity was not pinned.

[Raw samples](benchmarks/upstream-joins-mac-4.csv) and
[provenance](benchmarks/upstream-joins-mac-4.json) include all values,
whole-worker RSS, source/binary/input hashes and actual invocation.

| Query | Layout | Right rows | Mojo | Polars | DuckDB | Mojo / Polars | Mojo / DuckDB |
|---|---|---:|---:|---:|---:|---:|---:|
| duplicate_strings | base | 1,000,000 | 92.82 | 85.43 | 11.28 | 1.09 | 8.23 |
| duplicate_strings | base | 10,000,000 | 846.51 | 862.40 | 65.62 | 0.98 | 12.90 |
| duplicate_strings | shuffled | 1,000,000 | 90.53 | 112.50 | 13.53 | 0.80 | 6.69 |
| duplicate_strings | shuffled | 10,000,000 | 909.08 | 1142.93 | 80.11 | 0.80 | 11.35 |
| highcardinality | base | 1,000,000 | 3.79 | 1.46 | 0.58 | 2.60 | 6.49 |
| highcardinality | base | 10,000,000 | 33.62 | 7.03 | 0.52 | 4.78 | 64.81 |
| highcardinality | shuffled | 1,000,000 | 15.99 | 1.61 | 0.83 | 9.95 | 19.17 |
| highcardinality | shuffled | 10,000,000 | 416.61 | 7.22 | 2.68 | 57.71 | 155.60 |
| highcardinality | wide | 1,000,000 | 30.32 | 1.45 | 1.63 | 20.88 | 18.64 |
| highcardinality | wide | 10,000,000 | 412.07 | 6.95 | 9.73 | 59.31 | 42.34 |

## Eight-worker sensitivity check

Same method and inputs, with eight configured workers on the eight-core M1.
[Raw samples](benchmarks/upstream-joins-mac-8.csv) and
[provenance](benchmarks/upstream-joins-mac-8.json) also include 1M-row cases.
The table uses medians of round medians for 10M right rows.

| Query | Layout | Mojo ms | Polars ms | DuckDB ms |
|---|---|---:|---:|---:|
| duplicate_strings | base | 955.55 | 903.32 | 52.92 |
| duplicate_strings | shuffled | 1004.82 | 1230.68 | 68.78 |
| highcardinality | base | 34.80 | 6.12 | 0.67 |
| highcardinality | shuffled | 415.41 | 6.06 | 2.72 |
| highcardinality | wide | 410.02 | 5.95 | 7.80 |

The high-cardinality Mojo times barely change, while duplicate-string count
gets slower with eight workers. The M1 has four performance and four
efficiency cores: eight configured workers is not eight equivalent cores.
This supports prioritizing less work and better planning over higher worker
counts. The Linux follow-up below was run after the unrelated build finished.


## Linux Threadripper baseline

AMD Threadripper 3970X (32 physical / 64 logical CPUs), with the same Mojo,
Polars and DuckDB versions as the Mac. The owner confirmed the unrelated
Bazel build had finished; no compiler activity was detected during these runs.
The Linux frequency governor was `schedutil`; CPU affinity was not pinned. Each worker count uses three rotated rounds and
five measured repetitions after one warmup. The tables show medians of round
medians in milliseconds, for 10M right rows. Raw files also include 1M rows.

Source/lockfile hashes and every input-file hash match the Mac baseline.
The benchmark commit is `5f4bc46`; production engine code remains `c8c6498`.

### 4 workers

[Raw samples](benchmarks/upstream-joins-linux-4.csv) / [provenance](benchmarks/upstream-joins-linux-4.json).

| Query | Layout | Mojo ms | Polars ms | DuckDB ms | Mojo / Polars | Mojo / DuckDB |
|---|---|---:|---:|---:|---:|---:|
| duplicate_strings | base | 1592.95 | 1271.93 | 88.96 | 1.25 | 17.91 |
| duplicate_strings | shuffled | 1716.18 | 1851.36 | 110.52 | 0.93 | 15.53 |
| highcardinality | base | 40.34 | 10.79 | 1.55 | 3.74 | 26.08 |
| highcardinality | shuffled | 723.45 | 10.60 | 6.53 | 68.24 | 110.74 |
| highcardinality | wide | 719.86 | 10.41 | 15.29 | 69.15 | 47.08 |

### 8 workers

[Raw samples](benchmarks/upstream-joins-linux-8.csv) / [provenance](benchmarks/upstream-joins-linux-8.json).

| Query | Layout | Mojo ms | Polars ms | DuckDB ms | Mojo / Polars | Mojo / DuckDB |
|---|---|---:|---:|---:|---:|---:|
| duplicate_strings | base | 1546.02 | 1349.40 | 51.81 | 1.15 | 29.84 |
| duplicate_strings | shuffled | 1623.89 | 1813.33 | 65.26 | 0.90 | 24.89 |
| highcardinality | base | 42.00 | 6.25 | 1.68 | 6.72 | 25.05 |
| highcardinality | shuffled | 727.87 | 6.48 | 4.52 | 112.40 | 161.03 |
| highcardinality | wide | 736.47 | 6.45 | 9.64 | 114.26 | 76.38 |

### 32 workers

[Raw samples](benchmarks/upstream-joins-linux-32.csv) / [provenance](benchmarks/upstream-joins-linux-32.json).

| Query | Layout | Mojo ms | Polars ms | DuckDB ms | Mojo / Polars | Mojo / DuckDB |
|---|---|---:|---:|---:|---:|---:|
| duplicate_strings | base | 1484.76 | 1479.09 | 35.92 | 1.00 | 41.34 |
| duplicate_strings | shuffled | 1631.13 | 2296.80 | 39.04 | 0.71 | 41.78 |
| highcardinality | base | 45.15 | 7.16 | 2.08 | 6.31 | 21.71 |
| highcardinality | shuffled | 981.15 | 7.05 | 4.66 | 139.26 | 210.63 |
| highcardinality | wide | 967.25 | 7.19 | 7.56 | 134.56 | 127.92 |


The additional cores do not close the Mojo high-cardinality gap: shuffled
10M-row time rises from 723 ms at four workers to 981 ms at 32. For the
ordered duplicate-string count, Mojo improves only from 1,593 to 1,485 ms,
while DuckDB improves from 89 to 36 ms. At 32 workers Mojo is approximately
level with Polars on that count case, yet remains about 41 times slower than
DuckDB. This supports reducing preparation and intermediate materialization
before further worker-count tuning. Some round-to-round differences are
visible in the raw samples; small deltas are not treated as precise crossovers.

## Uncontended Mac profile and query-plan evidence

Instruments Time Profiler successfully recorded the `highcardinality`, wide,
10M-right-row case at four workers on the M1. Forty repetitions plus one
warmup completed with exit status 0 in a 17.80-second recording. The trace
includes startup/loading. [Exported sample summary](benchmarks/join-profile-mac.json)
contains 73.47 seconds of weighted CPU samples across threads; this is not
wall time. The raw trace and XML remain under `build/profiles`.

Self attribution includes `find_slot_matching` (9.74%), `match_empty` (9.64%),
`_id_range_bucket` (8.66%), dictionary growth (8.36%), table resizing (4.77%)
and dictionary order compaction (4.21%). `check_bounds` accounts for 15.48%,
aggregated across inlined call sites. These samples point to building and
encoding a large key domain; they do not establish that bounds checks should
be removed or that a compiler bug causes the gap.

[DuckDB's explain plan](benchmarks/duckdb-highcard-plan.txt), generated with
DuckDB 1.5.5 on the equivalent ordered inputs, supplies an independent reason
for its advantage: the 10M-row scan has a derived `k <= 999` filter, and the
1,000-row table is placed on the build side of the physical hash join. The
query text has no explicit key filter. The engine avoids work using general
join planning and key-range information. This plan inspection is not a Linux
timing result. Wide-key mappings reduce the usefulness of that particular
range bound, which is why all three layouts are retained.

These findings distinguish two follow-ups:

1. Reuse prepared build state across streamed probe batches (the narrow lazy
   join profile). This does not by itself solve the single-small-probe query.
2. Choose build/probe roles using cardinality and propagate safe join-derived
   filters (the upstream high-cardinality query). Preserve logical output
   order and null/duplicate behavior independently of physical build choice.

A second uncontended [Mac profile](benchmarks/join-profile-mac-duplicate.json)
covers the 10M-row duplicate-string count, four workers, ten repetitions and
one warmup. It contains 44.53 seconds of weighted CPU samples. Visible work
includes byte/column hashing (10.38% self combined), `_id_range_bucket`
(8.28%), string-view gathering (3.83%) and appending validity bits (3.87%).
Inlined bounds checks account for 22.54% across multiple sites. The full
query returns a single count, yet the current executor constructs joined
batch frames before reducing them; this input logically produces 40M joined
rows. A subsequent improvement should examine join/aggregate fusion or
count-only probing to avoid unnecessary row/column materialization. This
profile does not identify which corresponding technique DuckDB uses.

## Remaining coverage

This is the first comparison/profiling increment, not a comprehensive engine
ranking. Next coverage should include nulls, skew, compound keys, different
payload widths and build/probe ratios, followed by the pinned Polars PDS-H
queries. General join-result materialization and aggregate-over-join queries
must remain separate categories. A count query is not interchangeable with
materializing every joined row.

An engine change should include a paired before/after comparison of the same
inputs and compiler, complete correctness checks, profiles showing the work
removed, and tests for semantic edge cases. The current results establish
priorities; they do not claim an optimization has already been implemented.
