# Casts that only need Series metadata

An accepted identity cast does not need to read or merge its input arrays.
`cast_series` now handles these cases before its contiguous conversion adapter:

- Equal non-nested dtypes, including Null, Boolean, numeric, temporal, decimal,
  binary and Geometry with the same complete normalized metadata.
- Categorical input with a dictionary-free target, or a target whose dictionary
  has the same labels in the same order. The result keeps the source dictionary
  owner, including when an equivalent target owns a separate dictionary.
- String to Binary, including offset and view storage. This retags the shared
  arrays and clears field metadata, as the existing `with_dtype` contract requires.

The executor also recognizes these non-categorical casts when their operand
already exists as a complete column or prepared row state. It reuses that whole
operand, including through cast chains, rather than assembling unchanged values
batch by batch. This matters for view storage: appending every retained source
chunk separately can repeatedly copy the growing descriptor output. Categorical
expressions keep their distinct decode/encode and dictionary behavior.

The admitted paths retain the source array windows and owners, keeping their
backing buffers alive with the result. They do not populate a contiguous input
cache. An already populated cache remains available, and callers
that explicitly request `rechunk()` or a typed contiguous column still obtain one.
This changes neither those accessor contracts nor the conversion output contract
for casts that actually convert values.

The direct kernel's existing accepted identity behavior is preserved with strict
and non-strict calls, nonzero error offsets, and empty, inactive or mismatched
masks. Invalid payloads, nulls, names and complete logical dtypes remain intact.
Application field metadata keeps its owner on identities and is cleared on the
String-to-Binary dtype change.

Different categorical dictionaries still use the existing contiguous recode
adapter and validate every source dictionary label, including unused labels.
Binary-to-String still checks UTF-8. Other conversions keep their existing typed
conversion path or explicit contiguous adapter. Geometry casts with different
metadata still require explicit WKB import/export. Direct List/Struct casts
remain rejected before identity handling, including empty inputs; the public
executor's existing zero-row identity seeding remains separate.

## Validation

The primary incremental comparison applies pending [PR #1089](https://github.com/randyzwitch/dataframe_mojo/pull/1089)
(engine `c30afb58`) to both baseline and candidate. The combined candidate adds
exactly this change's cast delta: its patch is byte-for-byte identical to the
independent main-based branch's patch. Existing validity/Boolean gains therefore
remain in both sides. Standalone main-based evidence is retained separately.

The external baseline catalog covers 39 dtype/storage variants, strict and
non-strict calls, four direct-kernel mask shapes, empty and one-row inputs,
nonzero slices, all eight bitmap offsets, uneven and one-row chunks, and direct
versus whole-expression execution. Each complete result is checked independently
with Arrow/NumPy/Decimal values and schema, plus actual logical metadata markers.
Both Linux and the actual M1 baseline pass 2,980 cases: 2,968 verified answers and
12 expected nested-cast rejections. All categorical cases in this catalog are
accepted; direct and whole-expression dictionary expectations remain distinct.

Nine new repository contracts additionally check physical owners, cache state,
source destruction, explicit contiguous access, categorical recoding errors,
metadata retention and removal, arbitrary binary bytes, temporal units/zones,
wide decimals, Geometry metadata, and the direct/public empty nested boundary.
The revised candidate passes the same complete 2,980-case catalog on both Linux
and the M1. All 132 selected contracts pass on each integrated host, including
the nine new cast contracts, #1089's validity cases, and three unchanged
expected-failure declarations. Both runs have no skips or failures and matching
fingerprints for all 633 Mojo sources under `dataframe/` and `tests/`. The
independent main-based candidate also passes all 117 applicable contracts on
both hosts.

On the actual M1, the installed native Metal provider is required by a separate
probe using `collect(engine="accel", batch_size=3)`. The main-based baseline
and candidate pass all 15 literal column results, including values, nulls,
logical dtypes and row order. This is public scalar integration evidence; it makes no GPU performance
claim. A separate integrated probe checks predicates guarding identity casts
and casts over cumulative states fed by fill-null. Both revisions pass its
complete literal results for contiguous and unevenly chunked inputs, on CPU
and through the installed native provider at batch size three. Paired timing
and held-out results are recorded below.

## Allocation observations

The frozen incremental Linux comparison verifies all 468 baseline/candidate
pairs and all 936 complete answers. Newly requested bytes decrease in 333 cells,
remain equal in 135, and increase in none; allocation tracking errors are zero.
The retained matrix includes allocation calls, requested bytes, live bytes on
return, peak live bytes, freed bytes and tracking errors. Fixtures and explicit
warm-cache preparation are outside the observed call.
Cold means an unpopulated contiguous merge cache, rather than flushed CPU caches.
These counters measure newly owned allocations, not retained source buffers or
RSS. Input cache presence is recorded before the call, after the call before
export, and after export separately, since exporting an aliased result may
materialize a shared cache outside the observed call.

For 100,000 nullable rows starting at offset 13 in 64 cold chunks:

| Cast | Layer | Requested bytes, baseline / candidate |
| --- | --- | ---: |
| Int64 identity | Direct kernel | 863,705 / 224 |
| Int64 identity | Whole expression | 1,847,136 / 5,455 |
| StringView identity | Direct kernel | 1,641,740 / 224 |
| StringView identity | Whole expression | 14,464,860 / 19,647 |
| StringView to Binary | Whole expression | 14,463,963 / 19,647 |
| Equivalent categorical dictionary | Direct kernel | 457,509 / 224 |
| Equivalent categorical dictionary | Whole expression | 877,808 / 877,808 |

The direct Int64 case's peak newly owned bytes are 827,286 / 224. Parallel
million-row baseline cases have variable peak overlap between workers; all
observed peaks remain in the retained six-counter matrix.

The categorical case has four labels; equality of separately owned dictionaries
still compares their values. Its whole-expression adapter remains measured and
unchanged. The
first development candidate retained input chunks in the direct kernel but
increased whole-expression view allocations through repeated batch appends.
That candidate and all results are retained outside the repository; the final
whole-input dispatch was selected from this fixed allocation matrix before any
paired M1 timing or held-out query timing was run.

## M1 timings

The frozen incremental matrix completes 468 pairs, 936 workers and 2,808
independently checked full answers on the actual M1, configured for four runtime workers
and three repetitions per case. Baseline/candidate order alternates. The
calibrated process guard observes no compiler/known-worker overlap or monitor
errors. Every sample is retained.

For the same 100K nullable rows, offset 13 and 64 unmerged chunks:

| Cast | Layer | Median microseconds, baseline / candidate |
| --- | --- | ---: |
| Int64 identity | Direct kernel | 75 / 1 |
| Int64 identity | Whole expression | 229 / 8 |
| StringView identity | Direct kernel | 379 / 3 |
| StringView identity | Whole expression | 4,219 / 32 |
| StringView to Binary | Whole expression | 4,193 / 33 |
| Equivalent categorical dictionary | Direct kernel | 66 / 1 |
| Equivalent categorical dictionary | Whole expression | 269 / 279 |

Times cover direct casts or expression evaluation. Complete output export and
independent checking follow each timed call. An explicit contiguous consumer
can materialize the shared cache afterward; that separate phase is recorded.
The timer's observed quantum is one microsecond, with 212 zero-duration
samples. Individual hot/empty calls therefore do not resolve sub-microsecond
overhead. Contiguous direct medians span 0–7 microseconds on the baseline and
0–5 on the candidate; all raw samples remain available.

The retained categorical whole-expression adapter has broad, bimodal samples.
A fixed follow-up runs all three categorical modes at 1M rows, with three
alternating rounds of five repetitions: 18 workers and 90 complete verified
answers, with no observed overlap. Self-dictionary pooled medians are
1.132 / 1.135 ms; bare-target medians are 1.152 / 1.141 ms. All modes retain
roughly 1.1–3.3 ms ranges on both revisions. Median ranks can flip as samples
cross these modes, including the equivalent-dictionary control; the complete
sample distributions are retained rather than reduced to a speedup claim.

## Held-out workload

The unchanged complete quick-tier ClickBench run uses all 43 queries, fresh
DuckDB references, four threads, three rounds and three repetitions. All 86
target cells (baseline and candidate) pass the complete answer/schema oracle;
all 774 target snapshots are hash-verified across 258 target worker-query
records. All 516 worker-query records and reference verdicts are retained,
including the Polars comparison's eight passing and 35 wrong-answer cells.

The candidate/baseline elapsed geometric mean is 1.0007, effectively unchanged.
This is the incremental cast comparison over #1089, using fresh observations
of both sides. The validated baseline binary is reused only after matching its
source, compiler, SDK, native library, input and prior full-answer receipts.
Prior timings and ratios are not reused.

All 43 existing trace presence sets and their literal source call sites are
retained. These traces describe observed paths; ownership tests and allocation
accounting establish the cast-copy mechanism. Benchmark queries, generators,
evaluators, drivers and trace instrumentation are unchanged. No per-query
held-out timings were inspected or used to choose the implementation.

Evidence is retained outside the tracked tree under
`build/1003-cast-metadata-validation-a54c04b4/` on Linux and
`/Users/randyzwitch/benchdata/1003-cast-meta/a54c04b4/` on the M1. The `v2`
directory holds the independent branch comparison; `incremental` holds the
comparison with #1089 applied to both sides. The full remaining-consumer ledger
is source evidence, rather than a claim that the rest of #1003 is complete.
