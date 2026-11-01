# Automaton text functions over chunks

This #1003 subset removes source materialization from eligible cold host
`contains_any` and `replace_many` calls. It keeps one matcher per invocation,
reads owned offset/StringView windows in row order and constructs one owned
Boolean or String result. Replacement output bytes remain required.

The new path validates physical arrays, cumulative ends, host buffers,
descriptor bounds and view/validity ownership before decoding patterns.
Published caches, coded inputs, nested logical tags, physical empty arrays
and unsupported storage retain the existing adapter. Flat matching kernels
and the automaton are unchanged. Literal order, ASCII-only folding, duplicate
patterns, standard versus leftmost matching, empty-match UTF-8 advancement,
nulls and replacement bytes retain their existing behavior.

Production starts independently from main `c9911860`. Engine/test commit is
`a01a3272`; production `91a527c4` corrects only a test's `raises` annotation
and explicit Int64 flag conversion. Incremental BASE is `e005cc8b`, with all
preceding ready scalar changes applied to both roles; HEAD is `d49a645e`.
The complete engine blob and engine patch match production. The incremental
profile does not import its earlier test file; corrected production tests
are compiled and run separately. No complete test-tree identity is claimed.

The baseline checkout and all candidate sources/binaries remain preserved.
Producer, shared fixture and catalogs were frozen before compilation.
The independent byte-enumeration oracle, error qualification, controller
and reader were frozen before the 304/40 campaigns (protocol `b9b8d00e`).
All cases remain retained, including contiguous, cold, warm, tiny, view,
mixed-storage, explicit merge and owning-output controls.

## Correctness

Both revisions pass all 304 functional configurations on Linux and M1:
296 complete ordered answers and eight exact qualified binding errors.
Names, schema, values, UTF-8/NUL bytes and nulls are independently verified
from Parquet. The producer additionally checks every direct null Boolean
payload is false and every direct null replacement span is empty; the
independent reader binds the checked count to its expected null count.
There is no independent raw-bits-file claim.

Per revision and host, the qualifier passes 25 literal/Unicode checks,
eight complete Error-payload checks and two SDK error-framing checks. It also
preserves 21 private initialized-storage cases: nineteen private stdout
records match byte for byte and every private exit outcome matches within
its host. Two wrong-cache-variant aborts
retain the same message and signal but point to the getter's new source
location (`functions.mojo:78:29` to `:82:29`). The original strict whole-stdout
rule therefore fails; those failures and raw records remain retained.
A separate, explicitly post-observation assessment verifies each coordinate
against its exact frozen source and compares every other byte unchanged.
It does not normalize user error-row positions or relabel the original rule
as passing. Raw ASLR-bearing stderr is retained without byte-equality claims.

All four focused production contracts pass on Linux and M1. They cover
cold-cache nonmaterialization, offsets zero through seven, offset/view/mixed
storage, matching modes, empty Unicode matches, published cache values and
validity, owner lifetime and empty/inactive argument validation. Broader
production CPU validation passes all 105 selected tests, with no failures,
skips or expected failures. Thirteen repository checks and generated API
documentation coverage pass.

## Allocation results

All 40 paired Linux allocation cases pass with zero tracking errors.
Calls, requested bytes and freed bytes decrease in 25 and match in 15;
none increase. Live bytes on return decrease in 10 and match in 30.
Peak newly owned bytes decrease in 18, match in 21 and increase in one.

| Direct cold input | Requested bytes BASE → HEAD |
| --- | ---: |
| Contains, 100K rows / 64 offset chunks | 2,119,552 → 36,580 |
| Replace, 100K rows / 64 offset chunks | 4,126,066 → 2,043,094 |
| Contains, 100K rows / 64 view chunks | 1,643,724 → 35,912 |
| Replace, 100K rows / 64 view chunks | 3,650,238 → 2,029,926 |
| Contains, 1M rows / 64 mixed chunks | 28,481,686 → 260,912 |
| Replace, 1M rows / 64 mixed chunks | 48,544,450 → 20,323,676 |
| Contains, 1K rows / 1,000 actual chunks | 471,710 → 139,126 |
| Replace, 1K rows / 1,000 actual chunks | 492,003 → 159,419 |

The one peak increase is the public 1M-row mixed-storage guaranteed-miss
replacement control: 23,863,532 → 23,888,772 bytes (+25,240). Its requested
bytes still decrease; no uniform peak improvement or noise explanation is
claimed. Required output storage, preparation and lifetime effects are
reported separately. Merge/output controls are descriptive, not a causal
subtraction of the matcher or replacement cost.

Public warm-parent cases create new evaluation windows and retain the cold
allocation costs of corresponding inputs. Parent cache presence is not proof
that a callee consumed a ready cache. Tiny fixtures request 1,024 pieces and
actually retain 1,000. Public construction removes requested empty pieces;
separate private qualification exercises actual physical empty arrays and
retains their adapter.

## Native and elapsed validation

Corrected native fixture v2 passes all 32 CPU/accelerator by flat/chunked
context queries on both revisions, plus explicit actual Metal SUM130 and
three host text composition queries per revision.
These establish host text compatibility; they make no GPU text or performance
claim. Native v1 built successfully but failed one incorrect expected Boolean
before completing its contexts. That source, binaries and failures remain
retained. V2 corrects only the expected case-insensitive match of `abc` in
`abcd`; all 48 logical text cells are independently checked before v2 builds.

The frozen 40-case M1 elapsed comparison completes three repetitions per
revision, with all 240 answers independently checked and no overlapping builds.
Medians decrease in 27 cases, match in two and increase in 11. For direct cold
1M-row mixed input, contains decreases from 34,769 to 27,986 microseconds and
replacement from 48,129 to 39,006 (about 19% each). At 1,000 rows in 1,000
actual chunks, direct cold contains decreases from 366 to 83 microseconds and
replacement from 392 to 106. Public whole-expression measurements for those
same tiny inputs decrease from 958 to 695 and 989 to 717 respectively.

The unchanged flat direct contains control increases from 2,026 [2,022–2,036]
to 2,115 [2,090–2,171] microseconds: +89 microseconds / 4.39%, with separated
three-sample ranges. This is a retained regression without a noise or codegen
attribution. Eight other increased cases have defined ratios from +0.18% to
+2.61%. The unchanged flat merge control increases from 1 [0–2] to 2 [1–2]
microseconds (+1 microsecond; 2× median), with overlapping sample ranges.
The warm merge control changes from 0 [0–3] to 1 [0–2] microseconds; its ratio
is undefined. No zero samples or controls are removed and no full-grid
geometric mean is reported. Sample ranges are descriptive, not confidence
intervals. The complete 40-case table and all samples remain in the evidence.

Unchanged quick43 heldout validation passes all 43 queries for both revisions
against fresh independent DuckDB references. All 86 target cells and 774
retained target snapshots verify. The aggregate HEAD/BASE elapsed ratio is
0.9904502479 for this quick-tier protocol. Only aggregate elapsed results are
inspected; this is no per-query or larger-scale performance claim and does
not feed further tuning. Compiler/resource observations and functional or
allocation-mode elapsed fields support no engine timing claim.

## Remaining scope

The full #1003 inventory remains open: 109 historical owners, 134 explicit
sites and 13 implicit accessor families, including other text, nested,
logical/decimal/temporal/Binary conversions and execution/storage/pipeline
consumers. This subset covers the two shared-automaton functions only.
