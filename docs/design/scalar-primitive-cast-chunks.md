# Primitive float and Boolean casts over chunks

This #1003 subset consumes cold host chunks for 38 remaining nonidentity
primitive conversion pairs: Float32/Float64 to numeric targets, Bool to
numeric targets, and numeric sources to Bool. Unmasked Float32 to Float64
keeps its existing infallible path; its masked conversion is eligible here.

The data property is physically homogeneous, host-readable primitive chunks.
Examples include converting Boolean flags to numeric counters and narrowing
floating measurements after concatenating Arrow batches or Parquet row
groups. Admission has no query, row-count or value-distribution threshold.
Every cold physical array and both buffers are checked before row reads.
Published caches and unsupported logical/private/device-preparation inputs
retain their existing adapters. Identity casts remain earlier and preserve
source owners and hidden payloads.

The production branch starts from main `e25cdd0f`, independently of #1096.
The incremental comparison applies #1096 to both sides: actual BASE
`484a34bf0a9d88efb36f9c48a09c383e165adda3` and HEAD
`0809807ee39fbbc8c46107925c5b2874ea896a71`. Its normalized primitive patch and
new test blob match production implementation `02072314`. Complete cast-file
or runtime equivalence across these branches is not assumed while #1096
remains outside the production base. Earlier integer-cast gains are controls.

## Conversion behavior

Float-to-integer casts widen Float32 to Float64, reject NaN/infinity, truncate
toward zero, then check half-open bounds formed from exact powers of two.
Unsigned bounds are [0, 2**bits), and signed bounds are
[-2**(bits-1), 2**(bits-1)). Thus -0.75 becomes unsigned zero and -128.75 fits
Int8 after truncation; checking the original fraction would change behavior.
The typed loop computes bounds once and writes an owned target buffer.

Float-to-Bool rejects NaN only. Infinities and nonzero finite values are
true; both signed zeros are false. Integer-to-Bool compares the native
integer with zero, preserving UInt64 behavior. Bool-to-numeric maps packed
source bits to exact zero/one while respecting the bit offset of both payload
and validity. Bool output adopts packed values and validity directly.

Float64-to-Float32 retains primitive IEEE conversion: finite overflow may
produce infinity, underflow may produce subnormals or signed zero, and NaNs
remain valid floating results. No new finite/range rejection is introduced.
NaN payload mapping follows the pinned SDK primitive conversion.

Nonidentity invalid, inactive and relaxed-failure rows have raw numeric zero
or Bool false. Direct masks apply only when their length matches the input;
nonempty malformed masks observe every row. Strict errors preserve the first
observed global `offset + row`, source/target names, original scalar-width
formatted value and cause. All source chunks are validated before observation
can suppress a row. Accepted identities still ignore masks as before.

Public null-otherwise casts can elide observation masks where the existing
executor proves them safe. Explicit typed nonnull zero/False otherwise
exercises the masked branch. A shared cast in multiple projection contexts
can be hoisted by existing CSE, so qualification queries use separate fresh
frames for those contexts. Optimizer behavior is unchanged.

## Correctness and allocation

Baseline-compatible semantic checks in six focused CPU contracts pass on
unchanged main. The final candidate passes all six with additional assertions
that eligible casts leave cold merge caches unmaterialized. Existing
unmasked Float32-to-Float64 cache behavior, published caches, owner lifetime
and identity aliases are controls.

All 93 functional probe cases pass per revision on Linux and live M1: 69 complete
answers and 24 exact strict errors. The finite catalog covers every pair,
64-bit limits, truncation, nonfinite values, Boolean offsets 0–7, malformed
masks, empty input, identities and public conditionals. Independent exact
rational/IEEE expectations cover integer results and finite floating bits.
NaNs have independent classification checks and a separately labelled SDK
primitive-width witness where payload mapping is unspecified. A separate
124-key source formatter witness qualifies strict diagnostics without using
production cast answers to invent expected numbers.

All 49 allocation pairs pass complete answer checks. Calls, requested bytes,
peak owned bytes and freed bytes are lower in 24 cases and equal in 25;
none increases. Live owned bytes are lower in 17 and equal in 32. Tracking
errors are zero. All flat, published-cache, existing Float32 widening and
prior #1096 Int64-to-Int8 controls are retained and equal.

| Direct cold case | Actual pieces | BASE requested bytes | HEAD requested bytes |
| --- | ---: | ---: | ---: |
| Masked Float64 to Int8, 100,000 rows | 64 | 1,972,114 | 112,980 |
| Masked Bool to Float32, 100,000 rows | 64 | 1,467,216 | 412,980 |
| Unmasked Bool to Float32, 100,000 rows | 64 | 467,216 | 412,980 |
| UInt64 to Bool, 100,000 rows | 64 | 2,697,122 | 25,400 |
| Float64 to Bool, 100,000 rows | 64 | 1,897,122 | 25,400 |
| Float64 to Int8, 1,000 rows | 983 | 539,019 | 1,605 |

The calibrated Linux observer includes output and temporary owned allocations
inside the operation. Fixtures and existing warm caches lie outside its
scope. These measurements are not RSS, GPU allocations or latency evidence.
Large generic-path gains include replacing dtype dispatch, intermediate
lists and caught per-row failures, as well as avoiding input merging. The
smaller unmasked Bool savings retain that already-typed conversion as a
separate comparison.

## M1 timing

The quiet-host grid retains all 49 cases and 294 complete answers. All BASE
workers precede all HEAD workers; each worker performs three consecutive
fresh-fixture repetitions. This is one grid, not alternating rounds. The
unchanged calibrated guard checks known compiler/benchmark activity; it does
not certify all host activity. Instrumented allocation timings are excluded.

Medians are lower in 35 cases, equal in one and higher in 13. Every one of
the 24 cold allocation-improving cases also has a lower timing median.
Representative Float64-to-Int8 medians are 7,228→265 us direct, 7,601→663 us
for the whole expression and 403→10 us for the small 983-piece input.
Unmasked Bool-to-Float32 improves 477 [461–485]→339 [325–339] us despite its
smaller allocation savings.

One slower control has separated observed ranges: published-warm
Float64-to-Bool is 7,861 [7,830–7,865]→7,924 [7,877–7,946] us, +0.80%, with
all six allocation statistics equal. The other twelve slower cases overlap.
The largest relative slower median is the old warm Float32-to-Float64 path,
61 [58–62]→63 [55–64] us (+3.28%, overlapping). The prior Int64-to-Int8
control is 250→238 us with overlapping ranges and remains a control.
All slower cases and samples are retained. Three-sample ranges are descriptive;
no universal latency benefit or causal explanation for control drift is
claimed. No additional grid or source change was warranted by these costs.

## Native composition and checks

Four native-provider CPU/accelerator × flat/chunked cases pass complete fixed
values, nulls, schemas, names, order, finite IEEE bits and NaN classification.
Each cast context uses its own single-expression query on a fresh frame.
An explicit accepted Metal Int64 SUM computes 130, followed by existing host
Float64 preparation and strict, relaxed and masked Float64-to-Int8 casts.
The provider returns a host-materialized aggregate. This verifies actual
Metal query composition; it does not claim GPU cast execution, device-
resident guard qualification or GPU performance.

Thirteen noncompiler repository checks, changed-file formatting and bounded
source review pass with no material finding. The broader CPU cast suite
completes 85 checks with no unexpected failures and three pre-existing
binary/categorical expected failures outside this subset.

## Held-out external validation

The unchanged quick-tier ClickBench run passes all 43 queries per revision
against fresh DuckDB references: 86 target verdicts, 258 target worker
records and 774 retained target snapshots verified by hash. All 516 worker
records complete. The previously admitted #1096 candidate exporter is reused
as the actual BASE at its original path; HEAD is freshly built, and both
roles have fresh timings. All queries are retained; ClickBench has no
upstream variant axis.

Aggregate candidate/BASE elapsed is `1.0020234448353276`, effectively
unchanged. Only the aggregate is inspected; held-out per-query timings do
not select the implementation or thresholds. The mechanism grid remains a
scoped diagnostic rather than an engine-wide speedup claim. The report
retains existing Polars translation discrepancies (35 wrong-answer and eight
successful verdicts); DuckDB and both target revisions pass all 43.

The unchanged trace retains all 43 query path-presence sets and literal
source-site provenance. Its operator-level granularity does not prove this
helper's per-query coverage or count copies. The general storage property
and batch/row-group use cases above justify the path; no query-specific
admission is introduced.

Evidence stays outside version control in
`build/1003-primitive-cast-validation-e25cdd0f` and on the M1 at
`/Users/randyzwitch/benchdata/1003-primitive-cast/484a34bf`.
