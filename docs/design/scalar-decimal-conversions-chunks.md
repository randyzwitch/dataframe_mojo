# Remaining decimal scalar conversions over cold chunks

This #1003 family removes source merging from eligible decimal rescaling,
integer/Boolean/float/String-to-decimal casts, and decimal-to-integer/Boolean/
Float32/String casts. One prepared conversion plan reads owned physical
windows and writes one final typed result, packed Boolean result, or String
builder. It also avoids the generic full-size Int128/Float64 intermediates.

## Behavior

Decimal scale reduction and decimal-to-integer use half-to-even rounding.
Decimal-to-Boolean tests the rounded integer: both +0.5 and -0.5 become false.
Decimal-to-Float32 retains the existing Float64 numerator conversion, Float64
power-of-ten conversion, division, and final Float32 narrowing. UInt64 remains
exact through Int128 when converting to decimal.

Float-to-decimal still formats the original physical float width and parses
that text with the existing decimal grammar. Scientific notation, nonfinite
text, and excess fractional digits retain their original failures; arithmetic
scaling would change that behavior. String inputs use the same parser over
borrowed offset or StringView rows. Decimal output text retains its scale.

Multiplication, rescaling, and parser accumulation retain the existing Int128
operation order, including wrapping before the precision check. Initialized
SDK arithmetic witnesses qualify this behavior on Linux and M1. These casts
do not introduce new mathematical overflow checks.

Generic observation masks apply only when their length equals the whole
input length. Null, inactive, and relaxed-failed rows keep null validity and
zero numeric/Boolean payloads, or empty String payloads. A strict error names
the first observed failing global row plus the caller's offset and renders
the original source value. Conversion does not inherit application field
metadata; it retains the source name and attaches the exact target dtype.

Accepted identities and the specialized decimal-to-Float64 route keep their
existing earlier behavior, including their different mask/null-payload rules.
Existing flat implementations remain unchanged. Published caches retain the
actual cached storage and length through the original adapter.

## Admission and ownership

The shortcut admits canonical decimal metadata and ordinary numeric, Boolean,
or uncoded String sources with matching storage only. Before reading payloads,
it checks every cold array, host buffer, physical offset, payload/bitmap bounds,
and cumulative chunk end. String offsets and StringView descriptors/buffers
receive their own complete bounds checks. Unsupported or malformed storage,
physical empty arrays, encoded strings, and nonhost inputs keep the adapter.
These retained cases are not asserted to be inherently faster or necessary.

Each source window retains its physical owner while prepared numeric/text
readers use its addresses. Results own their final buffers and never publish a
source merge cache on admitted cold inputs. The unmasked integer-to-decimal
route retains the empty bitmap sentinel for an all-valid result; generic
routes retain their packed validity representation.

The generic runtime source readers avoid a full source-by-target template
matrix. Unmasked plain integer-to-decimal conversion keeps the existing typed
arithmetic strategy over cold windows: eight source types and three decimal
widths, with typed pointers and factors prepared once. Byte groups assemble
validity without per-row runtime source callbacks. Only a strict failure
prepares an original-width source renderer. The existing flat functions and
all other generic routes remain byte-identical. This adds 24 small typed loop
instantiations; code generation is an explicit tradeoff.

## Validation provenance

Production started independently from main `c9911860` on
`work/1003-decimal-conversion-chunks` and is rebased onto main `726cd1f3` as
`d0034d15`. The refined incremental comparison uses BASE `de8b6a74` and
HEAD `cc0258b9`. The original generic candidate `b84f494d` and pre-refinement
production contexts remain retained with their original identities. Earlier ready
scalar work is present on both comparison sides. The entire new helper block,
dispatch call, and contract blob match the production patch. Removing only
this family's additions restores each context's own complete baseline cast
file byte for byte. Complete existing source/test-tree identity between the
two contexts is not claimed.

The fixed development catalog has 288 functional configurations, including
192 direction/scale/layer controls and 96 focused mask, strict-error, offset,
boundary, empty/scalar/cache, and conditional checks. Six same-dtype decimal
cases are explicit identity controls. The primary oracle uses independent
integer/rational rounding and decimal parsing. Original-width SDK float text
and dtype/range spellings are separately observed qualifications, not an
independent proof of the SDK formatter. All 42 witness records agree on Linux
and M1. Direct raw-payload checks and public valid-row checks are distinct.

Both revisions pass all 288 configurations on Linux and M1 (279 complete answers and
nine exact strict errors). All nine final focused production contracts pass on both hosts with
explicit cold-cache assertions. The archived baseline-compatible first eight
contracts also pass; the ninth checks the refined typed path. Independent checks of 177 direct cold nonidentity configurations confirm
that the source cache remains unpublished through the call and export. The
protocol parser passes 981 pure checks over all 376 catalog configurations,
rejecting malformed, incomplete, duplicate, and reordered records. Every
retained answer/error artifact is independently verified in addition to the
process grammar.

All 13 repository source checks pass on final production `d0034d15`, with
all 950 tracked source files unchanged before and after. Final broader CPU
validation covers 183 cases across 24 modules: 180 ordinary successes and
three existing expected-failure wrappers, with no unexpected failures or
skips. Those three categorical defects are not fixed here. The earlier
independent-production 170-case and rebased 182-case admissions remain
separate historical evidence.

Forty compatibility probes pass per revision on each host: 25 complete normal
results, 13 exact caught errors, and two observed private Variant failures.
The refined comparison explicitly reuses the qualified `de8b6a74` BASE
observations after source/SDK/binary/artifact verification and runs a fresh
`cc0258b9` HEAD. Both private pairs have identical stdout, without coordinate
projection: actual SIGILL on Linux and SIGTRAP on M1. Cross-host signal
equality is not claimed. Raw ASLR-bearing stderr is retained and hashed,
with byte equality diagnostic only; normal and caught-error stderr is empty.

Native M1 integration explicitly reuses the admitted `de8b6a74` BASE
build/runtime after source/SDK/native/binary and artifact reverification;
`cc0258b9` HEAD is freshly built and run. Each role covers 36 route/context
frames containing 108 fresh single-expression projections, followed by an
actual Metal SUM130 and three host decimal casts (112 query executions, 38
markers). All expected schema/name/order/null/value frames are literal.
Decimal conversion executes on the host; this establishes composition, not
GPU decimal execution, helper-specific coverage, or performance. Every copied
artifact is rehashed against its actual manifest; no historical copy count is
attributed to this refinement. The fresh refined M1 elapsed campaign and unchanged heldout validation are
admitted below.

The refinement adds a ninth contract: 6,144 conversions cover all eight
integer types, three decimal widths, physical source offsets and output byte boundaries, empty/scalar
and byte-boundary lengths, absent/present all-valid bitmaps, nulls, relaxed
precision failures, exact raw payloads and padding. Another 24 checks require
complete strict errors at the first observed global row. Both Linux and M1
production runs pass all nine contracts. Original eight-contract expectations
are unchanged.

The refined verifier caches seven pure scalar expected-value functions. Their
math bodies and complete `expected`/`verify` functions are unchanged; removing
one import and seven decorators restores the original oracle byte for byte.
Differential checks cover 393,560 scalar arguments, 752 complete arrays and
336,990 rows, including matching exception types/messages. No engine answers,
output rows or files are skipped or cached. This optimization is outside
observed kernel calls and is frozen before the fresh comparison.

## Measurements

The fixed 88-case cost catalog spans nine representative route classes at
direct/public layers, flat/cold/warm/large shapes, tiny fragmented cases, and
existing decimal-Float64/nondecimal controls. Every case and sample remains
in the report. Allocation counters describe scoped logical requests and peak
newly owned bytes, not process RSS or storage retained before the call.
Functional/allocation elapsed fields support no timing claim.

The refined comparison in `campaign/linux-allocation-comparison.json`
independently verifies all 88 Linux allocation pairs, complete answers and
declared raw payloads with zero observer tracking errors. Its counts and byte
values match the retained original allocation comparison. Calls, requested
bytes, and freed bytes
fall in 54 cases and match in 34; peak newly owned bytes fall in 52 and match
in 36; live newly owned bytes fall in 21 and match in 67. None increase. Lower
freed bytes alone are not an improvement metric.

The following direct cold nullable cases each have 1M logical rows and 64
physical chunks. The fresh refined allocation comparison confirms these
values; the historical comparison remains separately retained.

| Conversion | Requested bytes BASE → HEAD | Peak newly owned bytes BASE → HEAD |
| --- | ---: | ---: |
| Int64 → Decimal32 | 12,317,232 → 4,125,480 | 12,250,736 → 4,125,480 |
| Boolean → Decimal64 | 25,408,232 → 8,125,480 | 25,250,176 → 8,125,480 |
| Float64 → Decimal128 | 50,260,848 → 41,060,520 | 25,251,984 → 16,125,480 |
| String → Decimal64 | 37,468,729 → 8,125,480 | 37,125,269 → 8,125,480 |
| Decimal64 → Int32 | 30,317,232 → 4,125,480 | 29,125,256 → 4,125,480 |
| Decimal32 → Boolean | 22,574,712 → 250,400 | 22,126,728 → 250,400 |
| Decimal128 → Float32 | 30,536,272 → 4,125,480 | 29,125,256 → 4,125,480 |
| Decimal32 → String | 21,712,956 → 16,514,100 | 18,543,192 → 14,416,464 |
| Decimal64 → Decimal128 | 25,317,232 → 16,125,480 | 25,250,512 → 16,125,480 |

Public warm-parent evaluations can create cold physical evaluation windows.
Parent cache presence does not prove that the callee consumed a ready cache;
the warm-up fixture remains alive outside the measured call. These savings
include retired generic conversion intermediates as well as avoided source
merges. They do not measure process RSS or every possible input representation.

Development cost review precedes heldout execution. Only aggregate heldout
elapsed is subsequently inspected; heldout results do not select an implementation.

## Original candidate and bounded refinement

The original generic candidate's complete 88-cell M1 campaign is preserved:
69 lower, four equal and 15 higher medians, with all 528 complete answers and
130,975,914 declared raw positions independently verified. No sample or case
was discarded. Five large regressions affect the unmasked integer-to-decimal
route; the direct 1M-row median changes 2,916→34,340 µs. This candidate was not
accepted for publication. Source inspection identifies replacement of the
existing typed strategy by the generic reader/plan path; timing alone does
not attribute the slowdown to a particular instruction or call.

All original higher medians remain explicit (microseconds):

| Case | Configuration | BASE median | Original median | Change |
| --- | --- | ---: | ---: | ---: |
| 2 | i64→d32, direct, 100,000 rows, 64 requested chunks, cold | 501 | 3579 | +614.37% |
| 3 | i64→d32, whole, 100,000 rows, 64 requested chunks, cold | 619 | 3675 | +493.70% |
| 5 | i64→d32, whole, 100,000 rows, 64 requested chunks, warm | 638 | 3649 | +471.94% |
| 6 | i64→d32, direct, 1,000,000 rows, 64 requested chunks, cold | 2916 | 34340 | +1077.64% |
| 7 | i64→d32, whole, 1,000,000 rows, 64 requested chunks, cold | 1409 | 6786 | +381.62% |
| 8 | bool→d64, direct, 100,000 rows, 1 requested chunks, cold | 6820 | 7080 | +3.81% |
| 9 | bool→d64, whole, 100,000 rows, 1 requested chunks, cold | 6883 | 7150 | +3.88% |
| 12 | bool→d64, direct, 100,000 rows, 64 requested chunks, warm | 6572 | 6676 | +1.58% |
| 16 | f64→d128, direct, 100,000 rows, 1 requested chunks, cold | 19045 | 19073 | +0.15% |
| 20 | f64→d128, direct, 100,000 rows, 64 requested chunks, warm | 17045 | 17055 | +0.06% |
| 23 | f64→d128, whole, 1,000,000 rows, 64 requested chunks, cold | 48601 | 48686 | +0.17% |
| 40 | d32→bool, direct, 100,000 rows, 1 requested chunks, cold | 12508 | 12586 | +0.62% |
| 52 | d128→f32, direct, 100,000 rows, 64 requested chunks, warm | 12020 | 12075 | +0.46% |
| 78 | d64→i64, direct, 1,000 rows, 1024 requested chunks, warm | 134 | 135 | +0.75% |
| 84 | d64→f64, direct, 100,000 rows, 64 requested chunks, cold | 90 | 93 | +3.33% |

The bounded refinement restores typed integer arithmetic for all 24 eligible
source/width pairs, under the original empty-mask condition. It does not
recognize benchmark cases or change inputs, masks, scalar rules, other routes,
or flat algorithms. The same full 288-case functional and 88-case cost
catalogs are rerun. Fresh BASE/refined elapsed samples are required; historical
elapsed results are not reused. The fresh full-grid results follow; unchanged heldout validation is reported separately.


## Fresh refined M1 elapsed results

The official complete 88-cell campaign admits all 528 answers and 130,975,914
declared raw positions: 85 medians are lower and three higher, with none equal
and no undefined ratios. All three samples per role and every case are
retained in `campaign/m1-cost-copyback-1/m1-cost-time-1/all88-timing-comparison.json`
(SHA256 `96725f479ab6fb4e6b67d782772f2681ecca4e0801fe6b1893b49bbb8371de42`).
The copy admission is `33cb8f88d6772abb281e6ba31997a10a56ef14f99627143878ded5b553befbda`;
`refined-cost-review.json` records the acceptance decision and its limits.

All remaining higher medians are disclosed below. Values are microseconds,
shown as median [minimum, maximum]; overlapping ranges do not erase a measured
increase. These are unchanged flat controls, not evidence of blanket
nonregression or whole-engine speedup.

| Case | Configuration | BASE | Refined HEAD | Change |
| --- | --- | ---: | ---: | ---: |
| 16 | f64→d128, direct, 100K, one chunk, cold | 19,045 [18,940, 19,228] | 19,417 [19,042, 19,525] | +1.95% |
| 64 | d64→d128, direct, 100K, one chunk, cold | 8,172 [8,158, 8,420] | 8,443 [8,347, 8,670] | +3.32% |
| 65 | d64→d128, whole, 100K, one chunk, cold | 8,202 [8,079, 8,556] | 8,511 [8,414, 8,777] | +3.77% |

The five previously severe integer-to-decimal cells now have these fresh
paired results, without combining samples from the original campaign:

| Case | Configuration | Fresh BASE | Refined HEAD | Change |
| --- | --- | ---: | ---: | ---: |
| 2 | i64→d32, direct, 100,000 rows, 64 chunks, cold | 524 [514, 3,177] | 223 [220, 238] | -57.44% |
| 3 | i64→d32, whole, 100,000 rows, 64 chunks, cold | 655 [616, 704] | 329 [325, 331] | -49.77% |
| 5 | i64→d32, whole, 100,000 rows, 64 chunks, warm | 664 [594, 687] | 342 [289, 349] | -48.49% |
| 6 | i64→d32, direct, 1,000,000 rows, 64 chunks, cold | 13,840 [8,849, 17,530] | 2,051 [1,929, 2,451] | -85.18% |
| 7 | i64→d32, whole, 1,000,000 rows, 64 chunks, cold | 2,367 [2,121, 5,587] | 2,233 [2,231, 2,427] | -5.66% |

BASE case 6 itself varies across epochs: original samples were
[3,346, 2,916, 2,835] µs; fresh samples are [13,840, 17,530, 8,849] µs.
The refined samples are [2,451, 1,929, 2,051] µs. Both BASE epochs remain
visible, with no causal attribution and no historical/fresh sample mixing.

Representative direct cold 1M-row/64-chunk medians from the same fresh grid:

| Conversion | BASE µs | Refined HEAD µs |
| --- | ---: | ---: |
| i64→d32 | 13,840 | 2,051 |
| bool→d64 | 69,685 | 33,588 |
| f64→d128 | 230,786 | 146,765 |
| string→d64 | 82,014 | 48,316 |
| d64→i32 | 147,679 | 25,320 |
| d32→bool | 132,104 | 24,035 |
| d128→f32 | 134,697 | 5,281 |
| d32→string | 173,912 | 61,898 |
| d64→d128 | 101,945 | 33,957 |

The review accepts the scoped mechanism benefit and explicit 24-loop codegen
tradeoff, while retaining all three increases below 4%. These development
results did not select heldout queries or alter the frozen heldout protocol.

## Unchanged heldout validation

Both revisions independently match fresh DuckDB answers for all 43 quick-tier
ClickBench queries: all 86 target cells and 774 target snapshots verify
complete ordered values, nulls and schemas. The frozen protocol uses all
queries and variants (this plan has only the base variant), four threads,
three rounds and three repetitions, tracing and fresh independent references.
All 66 benchmark sources remain unchanged. Polars' 35 wrong-answer and eight
successful cells remain in the report.

Only aggregate elapsed is inspected: HEAD/BASE geometric mean
`1.0079038459380767x`, or +0.79%. This is a quick-tier result, not a full-tier
performance claim or proof of universal nonregression. No per-query timings
were inspected or used for tuning. Generic trace presence does not establish
helper-specific coverage.

The actual `de8b6a74` BASE exporter is reused after checking its original
build receipt, source, compiler/SDK, lockfile, native library and binary.
The `cc0258b9` HEAD exporter is freshly built with O3 and two compiler threads;
its SHA256 is
`a944895de8aa84baa3cfe4e8d37c546ceb09b3e6a7c3e86cc1fb98de04108730`.
Both roles use fresh elapsed samples; no historical timings are reused.
The calibrated quiet guard and frozen source/toolchain identities pass before
execution and remain checked afterward.

The complete copied package contains all 1,548 engine snapshots and 129 fresh
full DuckDB reference records with schemas, plus the frozen evaluator,
controllers and provenance. Independent local replay reproduces all 172 verdicts using the frozen
evaluator, verifies every snapshot/reference hash and schema, and rehashes all
3,388 copied transport files. Final admission is
`heldout-preparation/heldout-final-admission.json` (SHA256
`b8aafc8c855675217977da8ace9c59d909a72d90ef74830c7b71163ccc216d88`);
complete readback admission is
`bb52c620129c6dae7e94e7959e7525c6ebdd8a245f8983ba31bc55e24d3a70b6`.

## Evidence locations

Generated inputs, source freezes, controllers, binaries, raw answers, samples,
and independent admissions remain outside the tracked tree under
`build/1003-decimal-conversions-validation-c9911860/integer-fast-refinement/`
on Linux and
`/Users/randyzwitch/benchdata/1003-decimal-conversions/de8b6a74/integer-fast-refinement/`
on M1. The actual old BASE native/qualification/exporter paths are retained
in their reuse receipts rather than relabeled as fresh builds.

Key admissions are `production-equivalence-proof.json`,
`broader-cpu/admission.json`, `qualification/final-qualification-admission.json`,
`m1-source-build/native-copyback-admission.json`,
`campaign/linux-allocation-comparison.json`,
`campaign/m1-cost-copyback-1/m1-cost-time-1/copyback-admission.json`, and
`refined-cost-review.json`, and the `heldout-preparation/` build/runtime/copy
admissions. Original candidate evidence remains in the parent
validation directory. No CI polling was used.

## Remaining scope

This covers the related decimal scalar conversion family. It does not close
the full #1003 inventory, including temporal/calendar, categorical/dictionary,
Binary, text/nested, and execution/storage/pipeline consumers. Decimal widening
accessors and their callers also retain a separate ownership/contiguity audit.
