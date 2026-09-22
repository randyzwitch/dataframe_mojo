# CPU performance evaluation

Base: `7a01b37` (#154), now merged into `main`. Retained implementation: `be91deb`.
See [the evaluation report](../../docs/cpu-performance-evaluation.md) for final measurements and decisions.
The user requested the complete remaining CPU workplan, not just CSV.
A proposal is evaluated only after a reproducible experiment or verified
prior experiment; source inspection alone is not a performance result.

| Issue / idea | Evidence so far | Remaining evaluation |
|---|---|---|
| #149 float parsing | Wide mantissas in #154; exponent fallback specialization ~6.5x micro / ~40% CSV improvement; Float32 subset rejected | 53 modules +150 oracle cases pass; long/ambiguous inputs use borrowed standard fallback |
| #149 packed integers | Short/packed generic path wins on 16/32/64-bit fields; 8-bit and signed Int64 retain existing path | Outlined helper retains 1.2–2.3x long-field wins; final oracle passes |
| #151 chunked columns | Prototype avoids final copy: 1M read ~22→15 ms; 10M ~166→127 ms; lower comparable RSS | Full column/consumer/Arrow stream integration remains a separate feature |
| #152 projection | Selected-field append experiment regresses projected reads ~6–8%; rejected | Early-stop changes strict unprojected validation; preserve contract |
| #147/#150 scheduling | 4/8/16/32 chunks per worker measured; 16 helps 10M but hurts 1M | 1.15GB size-adaptive result mixed; keep current8 |
| #148 splitter | Scan/decode/concat and per-chunk tails instrumented on plain and quoted input | Instrumentation perturbs timings; complex quote paths still cost more |
| #103 reusable pool | Explicit scoped reuse: 1M two-stage dispatch ~2.93→0.50 ms | No unsafe process-global lifetime change; execution-context integration needed |
| #105 parallel join | Parallel ordered match expansion and adaptive CSR: corrected final head 1M56.2ms /10M576.8ms at32. A separate raw-key common-partition prototype passes Int64/String exact differentials. | Corrected full-result 1M/10M ×1/32 matrix complete: sparse/string wins at32, dense/skew and all one-thread cases lose; see results/raw_join_full_matrix.md |
| #106 dictionary encoding | Repeated code grouping saves ~17–25 ms/1M; encoding ~28–42 ms; 4–7x smaller logical payload | Reconciled Int32 joins save13–22ms, encode/remap29–60ms; categorical/Arrow API remains unimplemented |
| #108 parallel sort | Co-ranked merge already exists; serial radix wins at1 thread; parallel radix loses; parallel partial selection wins in isolation | Complete stable end-to-end matrix below: top-k wins on integers; radix needs selective dispatch; prototypes not shipped |
| #109 streaming | 20M/1.15GB bounded parallel pipeline validated; ~33MiB vs4.6GiB eager RSS; slower than Polars | Full lazy plan integration and diagnostics/options remain unimplemented |
| #115 bare-column reductions | Direct numeric buffers + bitmap COUNT; Float64 SIMD SUM/MEAN ~5–8x faster in tuned microbench; differential and targeted tests pass | 50 dtype/operation combinations improve at1/32; 1Msum0.34ms vsPolars0.30ms |
| #39 native Parquet | Excluded by the user on 2026-09-22 | No further research or PR changes in this scope |
| #110 final yardstick | Complete 1M/10M, 1/32-thread baseline in baseline.md | Final corrected measurements and recommendations in the report |

Retain exact integer accumulation, strict CSV grammar, stable sorting and
existing row-order contracts. Unshipped prototypes must be labelled, and a
lower bound that skips work must not be presented as an implementation win.

## #108 whole-pipeline top-k candidate

`topk_parallel_108_test.mojo` and `topk_pipeline_108.mojo` exercise an
isolated `dataframe/series.mojo` clone in `/tmp/df_topk_experiment`. The
candidate changes only eligible `smallest_indices` calls: each contiguous
range retains its stable local `k`, a serial final heap selects from at most
`workers * k` candidates, then rank-plus-row ordering restores the exact
stable output. Its threshold leaves large-k selection on the existing serial
heap path. The test checks ties, nulls, top and bottom direction, multikey
ranks, rank encoding, and gather output against stable sort heads; it includes
a 131,072-row input that enters the actual two-range path.

This is a measurement candidate, not a production change. Its end-to-end
harness reports full sort, serial heap selection, and candidate selection with
rank encoding and gather for 1M/10M Int64 and String keys. Timing is deferred
to a quiet window.

`radix_sort_108_test.mojo` and `radix_pipeline_108.mojo` are a separate
isolated candidate in `/tmp/df_radix_experiment`. It routes only one non-null
physical Int64 or temporal encoded rank word through four stable LSD radix
passes at 262,144 rows and above; small inputs, nullable values, strings,
floating values, and multi-key ranks remain on the merge path. The candidate
is therefore safe to compare end-to-end
against merge for Int64 and verify fallback behavior for String, but encoded
rank-word microbenchmarks alone do not decide whether it should ship.

## #109 bounded CSV streaming prototype

`streaming_109_prototype.patch` is an isolated experiment against `7a01b37`,
not a proposed public lazy API. It proves one useful shape: CSV scan,
`x > 0`, `x * 2`, and a global Float64 sum. The serial form carries one
stateful tokenizer across drained record batches. The parallel form holds one
record-aligned input window, decodes its ranges concurrently, and retains only
one scalar partial sum per range. A record larger than the configured window
is deliberately carried until its terminator, so the actual bound is the
window plus the longest record.

The isolated test covers quoted multiline records and malformed input for the
serial and parallel variants. The parallel scanner passes each range's global
record base to its reader, but concurrent jobs can surface an error in worker
completion order rather than the earliest malformed record. It also fixes the
dialect to the standard comma/quote/header/default-null configuration and
does not expose `read_csv` options such as comments, projection, encoding,
row limits, or error suppression. These are prototype limitations, not
contracts to carry into a full implementation.

A production #109 implementation still needs a plan IR that can represent a
scan source and row-preserving streaming transforms, an aggregate-state merge
protocol, ordered sinks for non-reduction consumers, bounded pipeline
backpressure, and optimizer/explain support. It must retain eager semantics
for nullable predicates, all CSV options and diagnostic order. Blocking
operators (sort, joins, global order-sensitive operations, arbitrary gather)
need explicit materialization or a compatible external/stateful algorithm.

The reproducible one-GiB fixture is generated by
`generate_streaming_1gb.py`. It repeats the established one-million-row CSV
body twenty times while appending `extra_a:Int64` and `extra_b:Float64`; the
header is written once. `bench_streaming_1gb.py` runs the same
CSV -> filter -> select -> sum pipeline in Polars eager or streaming mode for
either the original eight-column or generated ten-column schemas. Use one
process per mode when reporting RSS.

Window-five one-process measurements on the generated 20-million-row,
1,153,606,359-byte fixture (32 threads) were:

| Engine / pipeline | Peak RSS KiB | Reported internal time | Wall time |
|---|---:|---:|---:|
| dataframe_mojo eager | 4,786,908 | 1.477 s (rerun) | 1.90 s |
| dataframe_mojo bounded parallel, 4 MiB window | 34,112 | 1.190 s (rerun) | 1.25 s |
| Polars eager | 2,926,700 | 0.379 s | 0.90 s |
| Polars streaming | 1,549,524 | 0.566 s | 0.82 s |

The internal timings cover the pipeline; outer wall time also includes process
startup/teardown (including Python for Polars). Mojo internal durations were
collected in window six after making the harness print its one-pass timing;
RSS and wall times in this table are the window-five runs. RSS is process peak,
not an allocation counter. Both engines produce the same sum; Polars streaming
can prune unused columns, while this Mojo prototype decodes the full schema.

## Dictionary-code join experiment

`dictionary_join_experiment.mojo` evaluates the minimum reconciliation a
categorical join needs when each input assigned codes independently. It uses a
large repeated fact table and a one-row-per-key dimension table, so cardinality
100 does not create a quadratic many-to-many output. The right dictionary is
reverse-discovered and contains one quarter right-only keys; both sides have a
null. Before timing, the harness compares the joined left/right payload pairs
and their order with the string-key baseline. It reports encoding and
right-code remapping separately from repeated joins. Suggested inputs are
`1000000 100` and `1000000 100000`.

Current Arrow support is an explicit blocker for a categorical dtype:
`dataframe/arrow.mojo` rejects any non-null `array.dictionary` or
`schema.dictionary` pointer, and the PyArrow oracle deliberately expects a
dictionary array to fail. Import/export therefore neither preserves nor
materializes Arrow dictionaries today. A production categorical design must
define Arrow dictionary import/export, code width, null handling, dictionary
lifetime, and reconciliation/cache ownership before a code-based join can be
an observable optimization.

## Raw-key partitioned join experiment

`raw_partitioned_join_lib.mojo` is a separate #105 experiment: it does not
call `_joint_key_ids`. Each side receives the existing common raw-key hash
partitioning, every right bucket builds a private direct-key dictionary, and
the paired rows are reconstructed with a per-left count prefix so their order
is left-major with right input order inside each match. The companion
differential covers nullable duplicate Int64 and string values, low/high
cardinality, skew, and sparse Int64 extremes that force the production
bounded-range path to fall back.

`raw_partitioned_join_matrix_bench.mojo` reports baseline join time plus the
prototype's partition, build/probe, order (prefix plus fill), and gather
phases. It accepts `int64|string ROWS REPS WORKERS CARDINALITY
repeat|skew|sparse`; `sparse` starts Int64 keys at `Int64.MIN`. This artifact
has compiled and passed its differential, but has no timing result yet.

## Reproduction and artifact scope

`baseline.md` records the exact baseline and head-to-head commands. Production
changes are in `dataframe/`; files here are independent experiments, not a new
public API or a replacement test suite. Benchmark only after builds and tests
have finished; pin `DATAFRAME_THREADS` and `POLARS_MAX_THREADS` equally.

Use a disposable baseline source tree when applying an experiment patch:

```sh
cpu110_base=$(mktemp -d)
git archive 7a01b37 | tar -x -C "$cpu110_base"
patch -d "$cpu110_base" -p1 < experiments/cpu110/projection.patch
pixi run mojo build -I "$cpu110_base" experiments/cpu110/projection_bench.mojo -o /tmp/cpu110_projection_bench
DATAFRAME_THREADS=32 /tmp/cpu110_projection_bench build/bench_polars/left_1000000.csv x
```

Patches headed `a/dataframe/...` use `-p1`. Join patches headed
`dataframe/frame.mojo` use `-p0`. The CSR patch includes the inner-expansion
patch; do not apply both. Apply `join_parallel_inner_overflow_guard.patch` and
`join_csr_order_lifetime.patch` after the CSR patch. The latter preserves the
local scatter-order buffer until fill jobs finish; measurements from before
that fix are historical; the final-head report uses the corrected code. The dense-range and
fast-bucket patches are separate extensions of the CSR variant. Exponent inline/outline, packed integer
generic/short, and reduction SIMD/extended patches are alternatives, not a
stack. The retained production variants may differ from these intermediate
experiments; the PR's source and focused tests are authoritative.

Standalone sort, dictionary, pool, reduction-type and numeric microbenchmarks
can compile with `-I` set to the desired source tree. Their module docstrings
and raw command logs specify inputs. `results/` preserves historical quiet
windows, including rejected candidates and failed intermediate experiments;
use the report's decisions rather than interpreting every number as a shipped
result. RSS comparisons require separate processes and equal retained inputs.

## Follow-up raw-key partitioned join measurements

The isolated prototype hashes raw keys into common partitions, builds private
dictionaries, probes, restores left-major/right-row order, and gathers the
result. These provisional measurements include all four prototype phases; fixture
construction is excluded. **They are not equivalent full-join timings:** the
prototype gathers both complete inputs (including the duplicate right key),
discards the gathered frames, and does not retain a final joined DataFrame.
The corrected full-result experiment is complete and supersedes these numbers;
see [the full matrix](results/raw_join_full_matrix.md). Production comparison is PR #155. Both use 32 workers; best of five
for 1M rows and three for 10M rows. The right side has one unique row per key,
left keys are null every 127 rows, and cardinality is one tenth of left rows.

| Left rows | Key shape | Production ms | Raw-key prototype ms |
|---|---|---:|---:|
| 1M | Dense Int64 | 28.34 | 53.19 |
| 1M | Sparse Int64 | 62.52 | 53.35 |
| 1M | String | 75.28 | 71.33 |
| 10M | Sparse Int64 | 834.60 | 453.02 |
| 10M | String | 998.07 | 652.80 |

Logs: `results/raw_join_followup_*.log`. These phase results motivate evaluating an adaptive sparse/string path,
but do not yet establish an end-to-end join speedup. Stable order
restoration alone costs 217.7 ms (10M sparse) or 268.8 ms (10M strings), making
it the largest remaining prototype phase. Low-cardinality, skew, one-thread, and matched Polars comparisons are now
recorded in the full-result matrix.

### Whole-pipeline sort matrix, completed follow-up results

All times are milliseconds, best of four, matching stable tie order and
null-last semantics. Inputs contain a key plus an original-row payload.
Integer radix is eligible only for non-null Int64/temporal single-key sorts;
string rows exercise its unchanged merge fallback.

| Rows | Key | Threads | Merge | Radix candidate | Polars sort | Serial top-100 | Parallel top-100 | Polars top-100 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1M | Int64 | 1 | 318.15 | 263.91 | 39.54 | 12.90 | 12.87 | 18.92 |
| 1M | Int64 | 32 | 40.76 | 76.38 | 14.75 | 12.97 | 9.99 | 18.75 |
| 1M | String | 1 | 686.20 | 676.50 | 313.98 | 43.82 | 43.54 | 25.32 |
| 1M | String | 32 | 91.36 | 92.63 | 30.00 | 44.08 | 43.17 | 25.28 |
| 10M | Int64 | 1 | 7498.35 | 5334.82 | 548.97 | 131.89 | 129.20 | 207.38 |
| 10M | Int64 | 32 | 513.51 | 445.62 | 143.49 | 131.63 | 81.00 | 212.35 |
| 10M | String | 1 | 14715.93 | 14777.12 | 5618.43 | 432.80 | 431.78 | 271.61 |
| 10M | String | 32 | 1195.59 | 1212.50 | 399.91 | 433.31 | 409.98 | 283.62 |

The Polars top-k comparator selects on key plus unique original-row payload
(to preserve stable ties), then sorts only the selected 100 rows. Radix and
top-k use different seeded fixtures, so their columns are not directly
comparable to one another. All comparator outputs pass independent row-order
checks before/after timing.

Decision: parallel top-k is worth retaining for eligible small-k work; its
end-to-end benefit is largest for integer keys. Radix should not replace merge
unconditionally: 1M/32 regresses 1.87x even though 10M/32 improves 13%. Rank
encoding/gather costs mean the earlier isolated radix gains do not translate
into similar whole-frame gains. Polars still leads full integer/string sorts.

Commands are in `run_sort_matrix.sh`; fixture and output-order oracle are in
`sort_matrix_polars.py`. Logs end in `followup110.log`. Early 1M/one-thread
Polars top-k logs labelled `stable_sort_nulls_last_head100` are superseded by
the `bounded_followup110.log` results used above.

### Input larger than the process memory limit

`generate_streaming_over_cap.py` repeats the established CSV body ten times,
producing 11,536,063,059 bytes (200M rows). `run_streaming_memory_cap.py` runs
the bounded pipeline with `RLIMIT_AS=8 GiB`, one worker and a 4 MiB window.
It completed and matched the expected aggregate (4,498,907,239.999999),
recorded in `results/streaming_memory_cap_8g_followup110.log`. This verifies
execution with input larger than the process address-space allowance; it
does not constrain the host page cache or prove every lazy operator streams.
Its elapsed time is not used as a comparative benchmark because compilation
was allowed concurrently during this functional check.

The earlier 512 MiB attempt failed at TCMalloc startup: its initial aligned
1 GiB virtual reservation exceeds that allowance. That failure does not
measure the streaming pipeline's resident-memory requirement.

### Corrected full-result join conclusion

The [24-case matrix](results/raw_join_full_matrix.md) times the entire call,
retains the joined DataFrame, and checks exact equality with production outside
the timer. At32 workers, 10M sparse Int64 improves 826→419ms and strings
1002→568ms, but Polars takes87/115ms. Dense Int64 regresses224→437ms;
skewed strings regress448→574ms. Every single-thread case regresses.

Decision: raw-key partitioning is only promising as a guarded sparse/high-
cardinality string path at sufficient parallelism. It is not a universal join
replacement. The prototype still rebuilds bucket dictionaries during fill;
retaining per-bucket match state and reducing stable-order/output-copy costs
are concrete next optimizations. These are suggestions, not measured wins.

## Final pool and dictionary follow-ups

`pool_application_owner_experiment.mojo` measures actual filter→sum stages
with 4K/16K/100K inputs, comparing the existing65,536-row grain against8,192
with fresh workers and a reused explicit application owner. At32 maximum
workers,100K takes173us (legacy serial),909us (fresh12 workers), or113us
(reused12 workers). At16K the figures are35/155/29us. At4K all paths remain
serial at about9us. One-worker runs show no meaningful regression. Owner
startup costs1.23ms and is explicitly excluded from repeated-stage timing.
Nested jobs run inline; nested errors drain and propagate; owner reuse after
an error is checked. Both compiled and `mojo run` correctness modes exit
cleanly. This evaluates reuse and lower-grain economics, not a hidden global
pool or a verified process-global JIT teardown mechanism.

`dictionary_reuse_experiment.mojo` extends the grouping/join work to lexical
sort, unique, n_unique, value_counts, and nullable equality. At1M rows/32
threads,100/100K distinct strings, building a lexical dictionary and encoding
two input columns costs111/201ms. Repeated sort is98→39ms /82→35ms; unique
35→22ms /50→34ms; n_unique3.73→2.33ms /133→68ms; value_counts31→19ms /
58→39ms; equality2.71→1.75ms /2.70→1.75ms. Outputs remain encoded; converting
them back to strings is not included. Exact lexical row order, nulls, counts,
and equality are checked. These are repeated-code operation measurements,
not an implemented public categorical dtype. Codebook construction/encoding
needs amortization; one-shot conversion is generally not a win.

Logs: `results/pool_application_owner_t*.log` and
`results/dictionary_reuse_c*_t*.log`. Each artifact provides its CLI in the
source. No production changes were made by these follow-up experiments.
