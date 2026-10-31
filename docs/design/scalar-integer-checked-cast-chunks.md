# Checked and masked integer casts over chunks

This #1003 subset removes input merging and whole-column intermediates from
checked or masked casts of cold, host-readable plain integer chunks to the
ten numeric storage types. It keeps the existing unmasked infallible cast
path first. Contiguous inputs and published merged caches retain their
existing converters. The detected property is cold, host-readable, physically
homogeneous integer chunks, as produced by concatenating Arrow batches or
Parquet row groups before conditionally narrowing counters or identifier
columns. Eligibility has no row-count, value-distribution or query threshold.

The production branch starts from main `8366b270`. The incremental comparison
uses `effe3b5cb5d191645e9960ce27f3a3871c224e58` as BASE and
`484a34bf0a9d88efb36f9c48a09c383e165adda3` as HEAD. Both comparison revisions
include the temporal binary subset; its earlier gains are excluded here.
The cast source and new contract file are byte-identical between production
implementation `56a190eb` and the incremental HEAD.

## Conversion and eligibility

`_cast_checked_chunks` admits the eight plain integer source types and ten
numeric targets. It checks every cold array's physical storage and both
payload and validity host accessibility before reading any observed values.
Unsupported physical storage and buffers needing host preparation keep the
existing rechunk adapter and error ordering. Logical temporal, categorical,
decimal, nested, String, Bool and floating-source conversions remain outside
this subset. Explicit requests for contiguous storage are unchanged.

The typed loop reads physical windows with their payload and bitmap offsets,
checks integer bounds exactly through Int128, and writes one owned target
buffer plus a packed validity bitmap. Null rows, inactive rows and relaxed
range failures retain zero payloads. Strict failures report the first
observed invalid value with the existing source/target names, full UInt64
text, and `offset + global row`. Chunk boundaries do not reset that row.

Accepted identity casts still return shared source storage before inspecting
masks. Their hidden null payloads and ownership behavior are preserved.

The existing cast API has two rules that this change deliberately preserves:

- A nonempty mask of the wrong length observes every row; it does not
  broadcast or reduce its entries.
- Every nonempty mask selects the general Float64 conversion intermediate,
  even when the wrong-length mask is ignored for observation. Integer to
  Float32 can therefore round differently from the unmasked infallible path.

For example, UInt64 `2**63 + 2**39 + 1` converts directly to Float32 bits
`0x5f000001`, while its Float64 bridge produces `0x5f000000`. The new masked
loop preserves that bridge per scalar without allocating an intermediate
column. A public conditional with a nonnull typed zero otherwise passes the
actual observation mask. A null-otherwise total cast can run unmasked before
overlaying validity, and remains a control for that existing execution path.

## Correctness and allocation results

The five new public contracts pass against the unchanged baseline and the
implementation. They cover exact bounds, null poison, ownership, global
strict rows, malformed masks, identity aliases, published caches and the
existing widening path. The candidate also asserts that a cold checked cast
leaves its source merge cache unmaterialized.

The external finite suite passes all 124 cases per revision on both Linux
and the live M1: 116 complete ordered answers and eight exact strict errors.
It covers all eight source widths against ten numeric targets, mode-specific
IEEE bits, public conditionals, empty inputs and mask controls. Expected
integers and Float32/Float64 bits come from an independent exact-integer
oracle; exported Arrow schemas, values, nulls and applicable raw payloads are
checked. The ten standalone baseline rounding qualifications also pass.

All 52 allocation cases pass complete answer checks for both revisions.
The calibrated Linux observer measures requested owned bytes, allocation
calls, peak/live owned bytes and freed bytes within the operation. It
includes output and temporaries; fixture construction and existing warm
caches lie outside the scope. These numbers are not RSS, device allocation
measurements or latency evidence.

| Int64 to Int8 case | Actual input pieces | BASE requested bytes | HEAD requested bytes |
| --- | ---: | ---: | ---: |
| Masked direct, 100,000 rows | 64 | 2,772,114 | 112,980 |
| Masked whole expression, 100,000 rows | 64 | 3,678,991 | 946,630 |
| Unmasked strict all-fit, 100,000 rows | 64 | 972,114 | 112,980 |
| Masked direct, 1,000 rows, cold | 983 | 547,019 | 1,605 |
| Same small input, published cache | 983 | 19,605 | 19,605 |

Across the full catalog, allocation calls, requested bytes, peak owned bytes
and freed bytes are lower in 24 cases and equal in 28; none increases.
Live owned bytes are lower in 18 cases and equal in 34. Tracking errors are
zero throughout. An all-valid unmasked checked output now carries a packed
validity bitmap; the old all-fit fast path could return bitless storage.
That representation cost is included in the all-fit cases, rather than
being hidden by comparisons with the masked general path. Contiguous, published-cache and existing unmasked
infallible controls remain equal. Whole-expression source-cache state alone
does not prove absence of copying: evaluation can create batch-local chunk
metadata. Its scoped allocations are measured separately above.

## M1 timing comparison

The quiet-host grid retains all 52 cells and all 312 complete answers.
All BASE workers run before all HEAD workers; each worker performs three
consecutive fresh-fixture repetitions. This is one grid, not alternating
rounds. The existing calibrated activity guard excludes known compiler and
benchmark overlap. No allocation-instrumented elapsed samples are used.

Medians are lower in 36 cells, equal in three and higher in 13. Every slower
cell has overlapping observed ranges; all 24 cold cases with lower scoped
allocations also have lower timing medians. Representative Int64-to-Int8
medians are 6,580 to 252 us for masked direct conversion, 7,190 to 687 us for
the corresponding whole expression, 325 to 248 us for strict all-fit direct
conversion, and 406 to 12 us for the small cold fragmented input.

The largest slower median is an unchanged contiguous masked UInt64-to-Int64
control: 6,718 [6,698–7,040] to 7,133 [6,731–7,165] us, or +6.18%. All
controls and slower cells remain in the report. These observations support
the scoped improvement, without establishing a universal latency benefit or
precise control-path regression attribution.

The large masked/relaxed gains combine avoiding input merging with replacing
per-row generic dtype dispatch, intermediate lists and exception handling.
They cannot be attributed solely to copy removal. The strict all-fit path
was already typed before this change and has a smaller measured gain.

## Native composition and repository checks

The broader CPU cast suite completed 84 checks with no unexpected failures.
Three pre-existing declared expected failures concern unsupported
binary-to-categorical conversion and categorical dictionary recoding; those
paths are outside this subset. Thirteen noncompiler repository checks and
formatting checks on both changed Mojo files also pass.

The candidate passes four native-provider composition frames on the M1:
CPU/accelerator engines against flat/chunked input, with complete fixed
values, nulls, schemas, names, order and Float32 bits. Separate queries
exercise the Float32 modes because existing common-subexpression elimination
can hoist a shared cast outside the conditional and select unmasked rounding.
Neither the optimizer nor that execution behavior changes here.

An explicit accepted Metal SUM query computes 130 from [64, 65, 1], then
checks strict and relaxed Int8 conversion and a masked cast of its output
combined with another chunk. That query runs on the actual Metal device.
The provider materializes its aggregate output on the host; unsupported casts
can use CPU execution. This qualifies native composition, without claiming
GPU cast execution, a device-resident cast guard test or GPU performance.

## Held-out external validation

The unchanged quick-tier ClickBench protocol passes all 43 queries per
revision against fresh independent DuckDB results: 86 target verdicts, 258
target worker records and 774 retained target snapshots with verified hashes.
All 516 worker records complete. The prior fully admitted temporal candidate
is reused as the actual BASE exporter at its original path; the checked-cast
candidate is freshly compiled. Both roles have fresh measured runs here.
ClickBench defines no variant axis; all its queries are retained.

The candidate/BASE aggregate elapsed geometric mean is
`0.9993728961794217`, effectively unchanged. Only this aggregate is inspected;
held-out per-query timings do not choose the implementation or thresholds.
The external aggregate supplies held-out validation, while the mechanism
timings above remain scoped diagnostics. Existing Polars reference
translations retain 35 wrong-answer and eight successful verdicts; DuckDB
and both target revisions pass all 43. Those reference discrepancies remain
visible in the raw report.

The unchanged trace retains 43 query path-presence sets and their literal
source-site provenance. Its granularity does not identify this helper's use
or count avoided copies. The eligibility property and real-world batch/row-
group examples are stated above; no query-specific admission is introduced.

Evidence remains outside version control in
`build/1003-integer-cast-validation-8366b270` and on the M1 at
`/Users/randyzwitch/benchdata/1003-integer-cast/effe3b5c`.
