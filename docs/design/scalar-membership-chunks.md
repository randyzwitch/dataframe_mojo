# Scalar membership over owned input windows

This #1003 subset addresses plain integer `is_in`, String literal-set `is_in`
and the Boolean equality used by general membership expressions. Production
starts independently from main `62706a5f`. Incremental validation starts from
`ef374698`, with ready #1092 applied on both sides so its gains are not counted
again. The first candidate is `e9b79f4f`; its cache refinement is `e71609de`
(production `9354ee1a` and `70fbf0ce`, respectively).

The dispatch properties are physical fragmentation, actual storage type and
whether an immutable merge cache is already published. Batched sensor values,
concatenated frames and Parquet row groups produce these representations.
There are no query-name, value-distribution or row-count exceptions.

## Kernels and retained adapters

Plain integer membership parses its serialized target values and converts them
to the selected integer width once per invocation. It retains SIMD8 comparisons
and scalar tails while reading physical source windows. Packed result bytes
are written at global row offsets across arbitrary chunk boundaries. Validity
is appended separately: the existing kernel compares payloads under nulls too,
so a null output's raw value bit can be true. No Float64 conversion is used;
UInt64 upper-half values and integers above 2^53 retain exact comparisons.

String membership decodes its byte-length-prefixed literals once and builds one
owned lookup per invocation. At most eight entries use linear comparison;
more use the existing hash table, including duplicate entries in the threshold.
Jobs own lookup copies and read owned `_Bytes` descriptors. Each worker writes
disjoint global bitmap bytes even when those bytes cross physical windows.
Empty private String sets produce false values with the input validity without
reading string payloads. Null String payloads are skipped before reading bytes.
Unicode, embedded NUL, commas, colons and regex metacharacters use byte-exact
set equality, without normalization, substring or regex interpretation.

Boolean EQ uses existing aligned bitmap inputs. Validity is `av & bv`, and
values are `~(x ^ y) & valid`, preserving zero invalid payloads and whole-operand
scalar broadcasting. Physical singleton chunks do not broadcast. Observation
masks retain the original actual-Boolean comparison behavior. New routes also
check actual storage variants, so private mismatched physical/logical tags keep
their old adapter, error and mask behavior.

Integer and String contiguous kernels remain unchanged. Published immutable
caches are reused without initiating or waiting for a merge. The refined Bool
EQ route also binds each ready operand to its published cache before using the
bitmap kernel. Cold operands retain owned physical windows. Other Boolean
operators are unchanged. Existing literal starts/ends/contains preparation is
unchanged; only a compile-time IS_IN branch is added to its shared chunk job.

Public general `is_in` remains an EQ/fill-false/OR/KEEP_NULLS expansion. Null
candidates do not match, while null source rows stay null. Consequently direct
Bool EQ and public dynamic membership differ when the candidate is null; their
oracles model that distinction. Public empty/all-null candidate lists retain
those general routes rather than being treated as malformed serialized integer
sets. Logical/temporal/private integer-storage fallbacks, general String
comparisons and categorical dictionary preparation remain explicit boundaries.
This does not complete every membership-like route or the wider #1003 inventory.

## Frozen validation

The unchanged baseline passes 640 functional cases on Linux and M1: 638 complete
answers and two exact length-error controls. The catalog contains 384 integer,
86 Boolean and 170 String cells. It covers all eight integer widths, complete
values/schema/nulls and direct raw payloads, offsets and bitmap tails, different
partitions, singleton/empty canonicalization, scalar/dynamic expressions,
StringView/offset layouts and public categorical controls. A separately frozen
24-case supplement closes dense fragmented String coverage on offset/view/mixed
layouts and empty/eight/nine/many-entry sets. It retains the original catalog.

Expectations use independent mathematical integer wrapping and fixed String
byte literals. Every complete Bool answer and every direct payload bit is
verified, including null slots. Whole-expression raw payloads are constrained
only at valid output positions. Strict errors, type binding on empty/inactive
rows, source/lookup lifetime, ready caches, masks and existing Float membership
NaN/signed-zero semantics are covered by focused contracts.

Performance has 178 fixed cells: 90 integer, 20 Boolean and 68 String. Fixtures,
expression construction, serialization and cache preparation are untimed;
existing target parsing/lookup construction remain inside the call. Actual
fixture lengths/counts and source cache presence are recorded before/after the
call and after export. Original source-cache absence does not prove absence of
local slice, dictionary or execution-preparation copies. Cold means merge-cache
cold, not hardware-cache cold. Warm fixtures publish caches before measurement.
Allocation counters measure newly owned requested/peak bytes, not process RSS.
Elapsed comparisons use actual M1, four execution threads, one BLAS thread and
the unchanged calibrated known-process activity guard. The guard does not
certify every host event. Every sample and complete answer is retained.

## Allocation results

For both candidate revisions, all 178 allocation pairs verify complete answers
with zero tracker errors. Requested and peak newly owned bytes decrease in 118
cells and are equal in 60; none increase. For nullable 100K-row sources from
offset 13, with 64 requested chunks and direct calls:

| Consumer | Requested bytes, baseline → candidate | Peak newly owned bytes, baseline → candidate |
| --- | --- | --- |
| Int64, four targets | 885,110 → 25,976 | 838,332 → 25,576 |
| Bool EQ, true scalar | 292,224 → 25,400 | 250,352 → 25,400 |
| String, nine targets, offsets | 2,042,414 → 35,531 | 1,982,650 → 34,736 |
| String, nine targets, views | 1,668,439 → 35,531 | 1,639,540 → 34,736 |
| String, nine targets, mixed layouts | 2,887,645 → 35,531 | 2,450,246 → 34,736 |

These are call-level mechanism measurements, not external-suite headline gains.
The general/categorical controls remain identified separately in the paired
admission rather than being counted as new dedicated-kernel coverage.

## M1 elapsed results and retained tradeoffs

Each complete adjacent grid has 178 cells, three alternating rounds and one
fresh-worker answer per side: 1,068 verified answers. The initial grid measures
`e9b79f4f`; the second measures the cache refinement `e71609de`. Both retain
every sample, source/cache observation, complete Parquet answer and raw payload
file. There are no discarded samples or guard overlaps.

The initial Bool EQ route still walked physical chunks after a cache had been
published. At 1K rows in 1,000 actual singleton chunks (1,024 requested), direct
scalar equality measured 7→20 µs and vector equality 6→32 µs. The refinement
reuses each published operand, with no shape threshold or cache construction.
In the second full grid these controls measure 7→2 µs and 6→3 µs. Their original
results remain visible; the second grid does not replace them.

Representative medians in microseconds, baseline → refined candidate, for
100K rows at offset 13 (64 requested chunks for nullable fragmented cases):

| Configuration | Direct | Whole expression |
| --- | --- | --- |
| Int64, four targets, dense contiguous | 47 → 47 | — |
| Int64, four targets, nullable fragmented | 149 → 72 | 203 → 156 |
| Bool EQ true scalar, dense contiguous | 649 → 651 | — |
| Bool EQ true scalar, nullable fragmented | 539 → 14 | 812 → 249 |
| String nine targets, dense contiguous offsets | 755 → 755 | — |
| String nine targets, nullable fragmented offsets | 990 → 795 | 1,012 → 911 |
| String nine targets, nullable fragmented views | 1,212 → 805 | — |
| String nine targets, nullable fragmented mixed layouts | 1,580 → 795 | 1,548 → 911 |

Tiny cached integer controls vary by one clock quantum: the refined UInt64
single-target median is 1→2 µs (first grid 2→1 µs), and Int64 seven-entry is
2→3 µs (first 3→2 µs). Clock quantum is 1 µs. All samples remain in the report;
no sub-quantum timing inference or universal speedup is claimed.

Large whole-expression calls show substantial variability and some worse
medians. Selected material cases, with milliseconds throughout:

| Configuration (1M rows, nullable, 64 chunks) | First medians | Refined medians | Refined ranges, baseline / candidate |
| --- | --- | --- | --- |
| Int32 one target | 1.341 → 7.912 | 1.518 → 1.062 | 1.254–1.529 / 0.567–1.089 |
| Int32 four targets | 1.149 → 1.336 | 1.156 → 1.330 | 0.643–1.245 / 0.600–1.615 |
| Int64 one target | 1.252 → 1.219 | 1.367 → 2.165 | 1.232–1.715 / 1.101–10.940 |
| UInt64 seven entries | 1.429 → 1.533 | 1.516 → 1.428 | 1.469–1.532 / 0.972–1.495 |
| String nine targets, offsets | 2.796 → 7.168 | 2.778 → 2.591 | 2.770–2.784 / 2.583–7.265 |
| String nine targets, views | 5.053 → 5.275 | 6.905 → 7.562 | 2.898–8.645 / 5.333–7.569 |
| String nine targets, mixed | 5.241 → 4.456 | 5.589 → 6.014 | 3.336–8.234 / 5.561–6.048 |

Int32 four-target whole-expression medians are about 15–16% worse in both
grids, with overlapping ranges. The other large cases also provide no stable
latency improvement claim. Their memory reductions and complete answers remain
verified; these elapsed tradeoffs are retained rather than hidden by reporting
only direct-call wins. String and integer algorithm bodies are unchanged by
the Bool-cache refinement, and no further thresholds were tuned to these cells.

## Contracts and native integration

The initial candidate passes all eleven selected contracts on Linux and M1,
with no expected failures, skips or process cases. All 637 runner source
fingerprints match its committed integration. The cache refinement reruns all
eleven on Linux and the affected extended Bool cache/scalar/empty/mask contract
on M1. Its focused driver and complete committed source/archive hashes are
bound separately; the earlier M1 full-suite run is not relabeled as refined.

The same frozen composition probe verifies twelve cases per revision:
CPU/native accelerator × contiguous/fragmented × offset/view/categorical String
inputs. It checks complete literal values, schemas, field names/order, nulls,
integer precision, String lookup thresholds, Bool equality versus membership,
then downstream fill/Boolean logic and a Float32 projection. An actual native
Metal device query passes. Native SDK runtime, compiler, source, probe, binary
and logs are pinned. This establishes integration and carries no GPU-launch or
GPU performance claim.

## Held-out comparison

Both revisions independently match fresh DuckDB references for all 43 quick-tier
held-out queries: 86 target cells, 258 target worker/query records and 774
retained target snapshots are verified. Aggregate candidate/baseline elapsed
geomean is 0.99631×, effectively unchanged. Individual query timings were not
inspected. This is a quick-tier regression control, rather than a full-tier
headline or evidence that each new route is used by external queries. Existing
trace presence sets and source-site provenance remain unchanged; no direct
membership-copy counter or GPU performance claim is inferred from them.

The original fresh `ef374698` baseline binary remains at its actual parent
path, with build receipt `6bf05a53`; it was neither rebuilt nor relabeled for
the cache refinement. The fresh `e71609de` candidate and all source/SDK/native,
benchmark, controller and guard hashes are recorded in its own manifest.

## Physical-storage guard follow-up

The original representative-array checks did not fully preserve rejection of
private heterogeneous storage. `_from_chunks` can combine different physical
variants under an unchecked logical tag. Boolean EQ then checked lengths
before encountering a later storage mismatch; integer membership parsed
targets first. String membership rejected during span preparation instead of
the legacy merge. These are malformed private inputs, not new supported data.

The correction checks every active cold physical array before the optimized
shortcuts. A published cache is checked through its actual merged variant in
constant time. The check does not copy payloads, parse targets, wait for a
cache or narrow physical String support to the logical String dtype. An
ineligible input follows the original contiguous adapter and rejection order.

The qualification freezes 25 isolated process cases against pre-optimization
`ef374698`: nine independently specified dense/null/cache controls and sixteen
malformed storage cases. The earlier implementation changes thirteen exact
exit/diagnostic outcomes, including seven changes from a Variant abort to a
catchable parse or length error. The correction restores all 25 legacy
outcomes. A positive contract also retains String-backed Binary storage and
canonical removal of empty wrong-storage pieces, with complete cold/warm
empty-, single- and nine-target answers.

The guard correction is production `6c25584e`, with integration candidate
`fcfbc918`. The incremental performance baseline is `c8b2a6e4`, which already
contains the earlier scalar changes and ordinary comparisons from #1094 on
both measurement sides. The original `ef374698` results above remain the
historical membership comparison; they are not relabeled as this baseline.
The new twenty-check source proof preserves the existing payload algorithms,
lookup construction and ownership. Its final refinement is solely an
explicit no-inline boundary for the existing integer chunk kernels.

Both revisions pass all 640 functional cases and the separate 24 dense String
controls on Linux and M1. In the 178 allocation pairs, calls, requested bytes,
live bytes on return, freed bytes and tracking errors are equal throughout.
Transient peak bytes are lower in five cells, equal in 165 and higher in eight;
the largest increase is 2,794 bytes. These peak changes remain in the report.

All three complete adjacent M1 grids are retained. The first two measure the
guard-only candidate; the third measures the outlined candidate. Outlining
restores standalone integer chunk kernels, but the timings do not establish
that it fixes the adverse parallel case:

| Int32, one target, 1M rows, 64 chunks, whole expression | Baseline median/range | Candidate median/range |
| --- | --- | --- |
| Guard grid 1 | 1,074 µs / 554–1,133 | 1,447 µs / 1,376–1,471 |
| Guard grid 2 | 965 µs / 558–1,459 | 1,171 µs / 973–1,449 |
| Outlined grid 3 | 582 µs / 569–610 | 744 µs / 566–1,472 |

The final median remains 27.8% worse with overlapping ranges. The immutable
baseline itself became substantially faster, so the absolute candidate
improvement cannot be attributed to outlining. Other large parallel cases
vary in both directions. There is no general latency-improvement claim for
this correctness correction.

The final grid has 64 slower medians. Two have separated three-sample ranges:
Int64 all-target warm whole-expression membership at 1K rows measures
468 µs (457–469) → 478 µs (473–489), a 2.1% increase; nine-target view String
membership at 1M rows measures 2,376 µs (2,056–2,562) → 3,005 µs
(2,829–3,523), a 26.5% increase. Every cell, sample and baseline-epoch drift
check remains available; these costs are not discarded as noise.

All twelve final production contracts pass on Linux and M1, with 637
committed source fingerprints verified per host. The unchanged composition
probe passes all twelve CPU/native Metal cases for both incremental revisions,
including the actual device query and downstream Float32 projections. The
original baseline admission remains at its actual path; the final candidate
is separately built and admitted. This carries no GPU-launch or performance
claim.

The final unchanged 43-query quick-tier external suite also passes. Both
incremental revisions independently match fresh DuckDB references; all 86
target verdicts, 258 target worker/query records and 774 target snapshots are
verified. Aggregate candidate/baseline elapsed geomean is 1.000264×, effectively
unchanged. Individual query timings were not inspected. Existing Polars
discrepancies (35 wrong, eight correct) remain visible without skips. The
original `c8b2a6e4` baseline is reused from its fully admitted #1094 candidate
at the original evidence paths; the final `fcfbc918` exporter is separately
built and admitted. No benchmark suite, generator, query or driver changes
are included in this correction.

Follow-up evidence is retained under
`build/1003-membership-private-storage-ef374698/fixed/performance`, including
the `chunk-outline` candidate, and the matching M1 directory under
`/Users/randyzwitch/benchdata/1003-membership-private-storage/ef374698/fixed/performance`.
The unchanged measurement protocol is SHA256
`b1c5900e0125aa3c1fb7f7346982ba845ff72cfd10007fa0297a912faa71a664`.

## Evidence and remaining scope

Artifacts remain under `build/1003-membership-validation-ef374698` on Linux and
`/Users/randyzwitch/benchdata/1003-membership/ef374698` on M1. Initial results stay
in the parent directories; the cache refinement uses `cache-reuse`. Frozen
producers, independent oracles, raw Boolean files, full Parquet answers, build
and SDK/runtime receipts, source manifests, logs and admissions bind each
result to its actual source and binary. Benchmark suites, generators, queries
and drivers are unchanged in the implementation PR.

List membership, multi-substring automata, other comparisons, logical/decimal
conversions, dictionary preparation and the remaining internal consumers still
need measured implementation or retention decisions. They are not marked
complete by this subset.
