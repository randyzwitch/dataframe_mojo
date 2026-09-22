# CPU performance evaluation after #154

Production implementation: `be91deb`, PR #155 (Ubuntu and macOS CI passed).
Follow-up experiments are on `perf/cpu-evaluation-followup-110`, stacked on #155.
Parquet research is excluded at the user's request; it is not part of this follow-up.

Measured on 2026-09-22, Linux x86-64, AMD Threadripper 3970X, pinned Mojo
1.2 nightly and Polars 1.44.2. The unchanged baseline is `7a01b37` (#154),
whose tree is also the current merge commit `db2e1a1`. Thread caps are equal;
internal worker counts can differ. Timings use optimized binaries, warm input,
one warmup and repeated measurements, without concurrent builds or tests.
The full comparison checks result heights and checksums. Stable Mojo sorting
is compared with Polars' faster default unstable sort.

## Changes retained

- Bare numeric SUM, MEAN, COUNT, MIN and MAX read column buffers directly,
  avoiding temporary expression batches and widening copies. Float64 SUM/MEAN
  use four-lane SIMD with packed validity masks; COUNT uses bitmap popcount.
  Integer accumulation remains exact in 128 bits, and NaN/extrema/tie behavior
  is preserved. Float64 SIMD scans use fewer, larger partitions to amortize
  thread creation.
- Inner joins count matches in contiguous left ranges, prefix output sizes,
  then fill disjoint spans while preserving left-major/right-row order.
  Large right sides use stable bucket scatter to construct CSR rows. A
  single physical Int64 key with a bounded value range uses direct IDs;
  sparse/wide ranges and other key shapes use the existing dictionary path.
  Both absolute and relative range caps bound the extra allocation. This
  particularly benefits bounded integer keys, not every possible join.
- Exponent decimals with at most 19 mantissa digits use the existing correctly
  rounded converter after combined validation. Long and ambiguous inputs
  retain the borrowed standard fallback. Packed decimal arithmetic speeds
  generic 16/32/64-bit integer parsing. The helper is deliberately out of line:
  inlining made plain CSV slower despite faster isolated integer parsing.

## Final head-to-head

The 1M global sum is now within 1.1x of Polars at 32 threads (0.34 vs 0.30 ms),
and 10M within 1.3x (2.18 vs 1.61 ms). The 10M inner join improved from 2.66 s to
0.58 s, but still takes 3.7x Polars' time. Plain CSV remains within 1.5–1.9x at 32;
exponent CSV takes 28.93 ms vs Polars 20.93 ms. Filter, large arithmetic outputs,
and general joins are larger remaining gaps. These are fixture-specific
results, especially the bounded integer-key join win.

Final 32-thread runs use best of 5 after warmup; single-thread runs use best of 3,
and exponent CSV best of 7. Baseline measurements used best of 3. Both engines
run sequentially, with result heights/checksums checked before reporting.
See [the baseline report](../experiments/cpu110/baseline.md) and
[the final command log](../experiments/cpu110/results/final_comparison.log).

### 32 threads

`polars=1.44.2 threads=32 reps=5 machine=x86_64 Linux`

| workload | rows | dataframe_mojo ms | polars ms | mojo / polars |
|---|---|---|---|---|
| csv_read | 1,000,000 | 23.71 | 15.37 | 1.5x |
| arithmetic_chain | 1,000,000 | 4.35 | 5.42 | 0.8x |
| nullable_compare | 1,000,000 | 1.96 | 5.16 | 0.4x |
| filter | 1,000,000 | 16.76 | 7.05 | 2.4x |
| global_sum | 1,000,000 | 0.34 | 0.30 | 1.1x |
| grouped_low | 1,000,000 | 22.98 | 13.94 | 1.6x |
| grouped_high | 1,000,000 | 25.82 | 13.79 | 1.9x |
| grouped_skew | 1,000,000 | 23.92 | 15.63 | 1.5x |
| grouped_str | 1,000,000 | 32.20 | 14.66 | 2.2x |
| join_inner | 1,000,000 | 56.21 | 28.84 | 1.9x |
| sort_multi | 1,000,000 | 87.32 | 55.82 | 1.6x |
| csv_read | 10,000,000 | 172.57 | 91.36 | 1.9x |
| arithmetic_chain | 10,000,000 | 32.51 | 9.47 | 3.4x |
| nullable_compare | 10,000,000 | 9.26 | 6.74 | 1.4x |
| filter | 10,000,000 | 103.65 | 28.48 | 3.6x |
| global_sum | 10,000,000 | 2.18 | 1.61 | 1.3x |
| grouped_low | 10,000,000 | 239.00 | 115.43 | 2.1x |
| grouped_high | 10,000,000 | 194.40 | 161.29 | 1.2x |
| grouped_skew | 10,000,000 | 214.88 | 118.80 | 1.8x |
| grouped_str | 10,000,000 | 319.15 | 149.54 | 2.1x |
| join_inner | 10,000,000 | 576.77 | 157.96 | 3.7x |
| sort_multi | 10,000,000 | 1198.14 | 588.74 | 2.0x |

### 1 thread

`polars=1.44.2 threads=1 reps=3 machine=x86_64 Linux`

| workload | rows | dataframe_mojo ms | polars ms | mojo / polars |
|---|---|---|---|---|
| csv_read | 1,000,000 | 322.32 | 210.93 | 1.5x |
| arithmetic_chain | 1,000,000 | 19.91 | 2.29 | 8.7x |
| nullable_compare | 1,000,000 | 17.26 | 0.93 | 18.6x |
| filter | 1,000,000 | 52.33 | 7.81 | 6.7x |
| global_sum | 1,000,000 | 0.77 | 0.63 | 1.2x |
| grouped_low | 1,000,000 | 28.20 | 21.97 | 1.3x |
| grouped_high | 1,000,000 | 45.53 | 59.93 | 0.8x |
| grouped_skew | 1,000,000 | 28.79 | 24.02 | 1.2x |
| grouped_str | 1,000,000 | 38.67 | 30.91 | 1.3x |
| join_inner | 1,000,000 | 105.04 | 139.31 | 0.8x |
| sort_multi | 1,000,000 | 739.14 | 362.81 | 2.0x |
| csv_read | 10,000,000 | 3657.93 | 2157.53 | 1.7x |
| arithmetic_chain | 10,000,000 | 198.12 | 18.63 | 10.6x |
| nullable_compare | 10,000,000 | 169.81 | 7.92 | 21.5x |
| filter | 10,000,000 | 524.70 | 71.26 | 7.4x |
| global_sum | 10,000,000 | 9.67 | 7.71 | 1.3x |
| grouped_low | 10,000,000 | 287.29 | 220.60 | 1.3x |
| grouped_high | 10,000,000 | 1139.63 | 1411.61 | 0.8x |
| grouped_skew | 10,000,000 | 291.76 | 225.30 | 1.3x |
| grouped_str | 10,000,000 | 390.07 | 454.74 | 0.9x |
| join_inner | 10,000,000 | 2708.41 | 2214.96 | 1.2x |
| sort_multi | 10,000,000 | 18059.95 | 6634.58 | 2.7x |

### Exponent CSV, 32 threads

`polars=1.44.2 threads=32 reps=7 machine=x86_64 Linux`

| workload | rows | dataframe_mojo ms | polars ms | mojo / polars |
|---|---|---|---|---|
| csv_read | 1,000,000 | 28.93 | 20.93 | 1.4x |

Reproduce the main comparison from this branch:

```sh
pixi run mojo build -I . benchmarks/bench_vs_polars.mojo -o build/bench_cpu110
pixi run -e oracle bench-polars --sizes 1000000,10000000 --threads 32 --reps 5 --runner build/bench_cpu110
pixi run -e oracle bench-polars --sizes 1000000,10000000 --threads 1 --reps 3 --runner build/bench_cpu110
```

## Experiments and decisions

These are isolated prototypes unless explicitly listed above. Their source,
patches, commands, raw results and limitations are in
[experiments/cpu110](../experiments/cpu110/README.md).

| Idea | Measurement | Decision / remaining work |
|---|---|---|
| Exponent parsing | 1M exponent CSV ~53.2 to 30.1–30.7 ms; plain CSV ~22 ms before and after | Retained; private standard-library converter remains a pinned dependency |
| Packed integers | 20M long Int16/Int32/UInt64 fields ~132/251/535 to 112/165/230 ms | Retained outlined helper; signed Int64 and 8-bit paths unchanged |
| Direct Float32 subset | 2–7% gain for matching fields, ~35% loss on fallback | Rejected; general direct rounding also differs from current Float64-then-narrow semantics |
| Projection-specific append loop | Projected reads ~6–8% slower | Rejected; skipping remaining fields outright would change strict structural/UTF-8 validation |
| CSV range count | 16 chunks/worker helps 10M, hurts 1M; size-adaptive 1GB results mixed | Keep current 8; no universal larger-count win established |
| Chunked CSV result | 1M read ~22 to 15 ms; 10M ~166 to 127 ms; lower RSS | Promising architecture; immediate rechunk pays most savings back. Needs chunk-aware columns/kernels and Arrow stream support |
| Scoped pool reuse | Two-stage 1M pipeline ~2.93 to 0.50 ms | Execution-context reuse is promising. No process-global pool or JIT-lifetime change shipped |
| Parallel CSR/expansion | 10M join ~2.56 to 1.41 s before direct integer IDs | Retained with large-right-side threshold; small CSR stays serial |
| CSR bucket arithmetic | Dictionary-key join 1.419 vs 1.400 s | Insufficient measured benefit to retain another change |
| Serial/parallel radix | Full Int64 sort at32: 1M40.76→76.38ms (regression), 10M513.51→445.62ms; Polars14.75/143.49ms | Reject unconditional dispatch; selective eligibility is needed. String/null/multikey fallback preserved |
| Parallel partial selection | Full Int64 top100 at32: 1M12.97→9.99ms, 10M131.63→81.00ms; stable Polars18.75/212.35ms. 10MString433.31→409.98ms vs Polars283.62ms | Candidate worth retaining for small-k, especially integers; isolated patch only |
| Dictionary grouping | 1M repeated grouping saves ~17–25 ms; encode costs ~28–42 ms | Reuse amortizes encoding; logical payload 4–7x smaller. No categorical dtype shipped |
| Dictionary join reconciliation | 1M,32 threads: strings50.1/78.0 ms vs codes36.5/64.7 ms (100/100k keys); encode/remap29.2/60.0 ms | Approximately 3/5 uses to amortize; independent dictionaries, nulls and right-only keys validated. Arrow dictionary arrays remain unsupported |
| Bounded streaming | 20M rows /1.154GB: eager1.477 s, parallel windows1.190 s; peak RSS ~4.6GiB vs34MiB | Strong memory improvement; slower than Polars. Prototype supports one pipeline, not the full lazy API |
| Parquet bridge | 10M Snappy dictionary: PyArrow+Mojo copy299 ms vs Polars56 ms; ~148 ms is combine/import | A format/import yardstick, **not a native Parquet reader**. Native metadata/codec/encoding support remains substantial work |

For the 1.154GB streaming fixture, separate single-pass Polars processes used
~2.79GiB (eager) and ~1.48GiB (streaming), with internal pipeline timings
0.379 s and 0.566 s respectively. The Mojo prototype retains a 4MiB record-aligned
window plus its longest record. It decodes all ten columns; Polars can prune
unused columns through its lazy optimizer. This is a useful end-to-end
comparison, not a claim of equal physical work. First-pass timings and RSS
are distinct from warmed repeated throughput results.

## Remaining route to parity

1. Reuse a scoped execution pool across eager/lazy operations. New thread
   startup is visible even after kernels get much faster.
2. Write arithmetic/comparison/filter results into their final buffers and
   avoid intermediate batch/Series materialization. Fuse compatible filter,
   projection and reduction stages. The bounded pipeline demonstrates the
   memory benefit, but needs a plan representation, ordered sinks, backpressure
   and complete CSV/error-option semantics.
3. Generalize join improvements beyond bounded integer keys, and eliminate
   redundant key/output copies. Raw-key partitioned joins can avoid global
   dictionary encoding for sparse and string keys.
4. Introduce chunked buffers with chunk-aware consumers, keeping contiguous
   materialization explicit. Copy avoidance helps CSV and Arrow import only
   when downstream operators can retain that layout.
5. Integrate eligible parallel top-k, with the measured full-pipeline benefit.
   Keep radix dispatch selective; rank encoding and gathering remain expensive.
   Add dictionary lifetime/reconciliation APIs where repeated use amortizes
   encoding costs.

This evaluation does not close the full chunked-column, streaming, dictionary,
thread-pool or sorting feature issues. Parquet is excluded from further work. Correctness contracts
and reproducible end-to-end gains determine which prototypes should graduate.

## Validation

- Full suite: 53 modules passed, including new parser, direct-reduction,
  bounded-key join, and parallel CSR regressions.
- Polars oracle: 150/150 cases, seeds 110–259; mutation check caught all
  20 injected defects. PyArrow C Data Interface interoperability passed.
- Direct reductions: 7,216 baseline/candidate output comparisons across
  every numeric dtype, sliced validity, nulls, IEEE cases and overflow;
  100 timed dtype/operation/thread-cap combinations retained equal results.
- Numeric parsing: exhaustive single-byte substitutions at each packed digit
  position and a 72,000-case exponent reference corpus, in addition to the
  regular parser tests.
- Formatting, dtype/version checks, API docs, example builds, benchmark smoke,
  and precompiled-package smoke passed locally.

The expanded CSR test found a real temporary-buffer lifetime defect before
publication. An explicit keepalive now retains the row-order buffer until the
worker barrier; the full suite and oracle were rerun after the fix. Earlier
isolated CSR measurements are historical evidence, while the final integrated
comparison above measures the corrected implementation. PR #155 passed both Ubuntu and macOS CI. The follow-up experiments and all
performance timings remain Linux x86-64 only.

## Follow-up evaluation

The complete stable single-key sort and top-100 matrix (1M/10M, Int64/String,
1/32 threads) is recorded in the [experiment ledger](../experiments/cpu110/README.md).
Its Polars top-k comparator uses the original row as a secondary key to
preserve ties, then sorts only the selected 100 rows. These measurements
include rank encoding, selection and output gathering.

The bounded CSV pipeline also completed a 200M-row, 11.54GB input under an
8GiB process address-space cap, with the expected aggregate. This demonstrates
input larger than the process allocation limit for the tested pipeline; the
OS page cache is not restricted. It is a correctness/memory-bound check, not
a comparative throughput measurement.

Raw-key join phase measurements initially omitted retained result assembly.
Those logs are provisional and superseded by the full-result matrix: the
corrected experiment retains a DataFrame, verifies exact production equality,
and times the entire call externally. Its single-thread path loses to the
production implementation in all twelve tested key/cardinality/size cases.
The [24-case full-result matrix](../experiments/cpu110/results/raw_join_full_matrix.md)
is complete. At32 threads, 10M sparse Int64 improves826→419ms and strings
1002→568ms, versus Polars87/115ms. Dense integers and skewed strings regress.
Retain this only as a candidate for guarded sparse/high-cardinality paths;
it is not a universal replacement and is not integrated into production.

The final pool experiment measures100K filter→sum at173us with the legacy
serial threshold,909us with fresh workers at an8K grain, and113us with a reused
explicit owner. Owner startup is1.23ms, so reuse across stages matters. Nested
inline work/error recovery and compiled/JIT exit checks pass. A hidden
process-global owner is not implemented or validated.

The dictionary follow-up confirms repeated code operations benefit beyond
grouping/joins: at1M/32, lexical sorting drops98→39ms (100keys) and82→35ms
(100Kkeys). Unique/count/equality also improve. The lexical dictionary plus
encoding of two columns costs111/201ms; one-shot conversion is generally a
loss. These are encoded-output operations, not a complete categorical API.

### Current disposition

PR #155 contains the production parser, reduction, and ordered-join changes
and passes Ubuntu/macOS CI. This stacked follow-up contains experiments and
evidence only. The remaining feature integrations are explicit pool ownership,
chunk-aware columns, categorical storage, eligible parallel top-k, selective
raw-key joins, and a general streaming plan. The measurements do not establish
the user's target of twice Polars' speed across workloads. The largest broad
gaps remain output materialization/copies, general joins, and sort-key work.
Parquet was removed from the follow-up scope at the user's request.
