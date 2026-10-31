# Temporal binary arithmetic over owned host windows

This #1003 subset removes mandatory input materialization from the temporal
ADD, SUB, MUL and FLOORDIV dispatcher. The production branch starts from main
`5681a00e`; engine and focused contracts are `98568745`. After #1094 merged,
main `8366b270` was merged into the branch as `b3e762b7`. All 298 runtime
files then match the actual validation candidate byte for byte. Incremental validation
compares the actual prior integration `fcfbc918` with `effe3b5c`, applying the
earlier comparison work to both sides. The historical `6f8e4b55` evidence
directory name does not identify the current production base.

## Input adapters and arithmetic

Cold fragmented inputs use owned Int64 cursors bounded by the intersection of
both operands' physical windows. Only a whole one-row operand broadcasts.
Payload pointers include the column offset; validity preparation applies each
bitmap offset once. The output owns one zero-initialized value array and its
validity. Existing bitmap helpers join input validity and apply the global
mask before checked arithmetic reads a row.

The original contiguous kernel remains byte-identical. Each published immutable
cache is bound to its actual merged storage before dispatch. Two ready operands
reuse the contiguous kernel without scanning original chunks, constructing a
cache or waiting for an in-progress merge.

All active cold physical variants must be Int64 columns. Cold logical List or
Struct tags retain their original nested merge; a physical Int64 tag alone is
insufficient to change that rejection path. Unsupported storage and payload or
validity buffers that still require device-to-host preparation also retain the
original adapter. Inputs with valid host access are not restricted to a new
logical temporal tag: the binder's explicit types still determine arithmetic.

Length rejection precedes empty-result handling. Masks keep their existing
scalar-any-active and mismatched-length first-bit rules. Null and inactive rows
skip checked operations and retain zero payloads. The loop preserves global
row order, checked add/subtract/multiply and MIN/-1 floor-division overflow;
zero divisors produce null. Date-duration scaling, date-date millisecond
differences, declared units and timezone metadata remain unchanged. Calendar,
formatting, parsing, temporal casts and packed children remain separate work.

## Frozen checks and allocation results

The finite catalog has 56 functional cells and 60 allocation/timing cells. It
contains eight direct/public routes, offset and differently partitioned inputs,
scalars, nullable and dense inputs, all-false masks, published caches, exact
ticks above 2^53, date scaling and signed floor division. Sixteen small-input
controls were appended before measurements; the original 44 cells and order
remain intact. The measured producer, expectations and complete catalog are
identical between revisions. Expectations use exact Python integers and full
schema/null/raw-tick checks without Float64 conversion.

Both revisions pass all 56 functional cases on Linux and M1. All 60 Linux
allocation pairs pass, with zero tracking errors. Calls, requested bytes, peak
live bytes and freed bytes decrease in 28 cells and are equal in 32; none
increase. Live bytes on return decrease in 14 cells and are equal in 46.

For identical cold nullable offset inputs, representative duration ADD costs are:

| Shape | Requested bytes, baseline → candidate | Peak live bytes, baseline → candidate |
| --- | --- | --- |
| Direct, 100K rows, 64/65 requested chunks | 2,531,581 → 866,014 | 2,438,492 → 816,864 |
| Whole expression, same 100K-row fixture | 3,646,638 → 1,834,742 | 1,023,103 → 895,943 |
| Direct, 1K rows, 983/984 actual pieces | 1,063,961 → 8,990 | 449,213 → 8,608 |

Direct-call allocation counts fall from 153 to 35 for the 100K-row fixture and
from 1,991 to 21 for the small fragmented fixture. Flat and direct small-input
published-cache controls retain identical allocation statistics; the latter
request 8,605 bytes in six calls. These are mechanism measurements, not elapsed
speedups or external-suite headline gains.

Direct cold calls leave the candidate's source caches unmaterialized, while the
legacy calls publish both. Whole-expression source-cache observations do not
prove absence of batch-slice or execution-preparation copies. Those layers
remain visible in the call-level totals. All-valid bitmap presence and unused
padding can differ from the old merge representation; values, logical validity,
schemas and null payloads are checked independently, and allocated bytes are
measured rather than inferred from cache state.

## Boundary and integration evidence

Four focused production contracts pass on Linux, with all 639 runner source
fingerprints verified against committed blobs. Ten isolated subprocess controls
match legacy outcomes: seven private storage rejections and three independently
specified null/cache controls. Raw exits and streams are retained; only source
line/column locations are normalized for stdout comparison. The qualification
exposed two initially accepted nested-tag inputs, and the final cold nested
adapter restores their legacy rejections. The original outcomes remain visible.

Two unchanged temporal contracts are also reused by the native integration
drivers, covering units and all four observed/masked overflow operations across
offsets, fragmentation, reversed conditionals and CPU streaming. The candidate
driver additionally calls the four committed focused contracts. The same frozen
composition probe checks four complete frames (CPU/accelerator × flat/chunked),
actual Metal device discovery, precise temporal values and schemas, and a
downstream Float32 projection. Both revisions pass the shared probe and the two
existing contracts; the candidate also passes all four new committed contracts.
Builds, drivers, source archives, SDK runtime, binaries, device query and complete
logs are bound separately. There is no GPU-launch or GPU performance claim.

## M1 elapsed results and retained tradeoffs

Two complete adjacent grids retain all 60 cells and three fresh-fixture samples
per cell and revision: 720 complete timing answers in total. The first runs all
baseline workers then all candidate workers; the confirmation reverses that
role order. The samples within each worker are consecutive, not alternating
rounds. Both use the same binaries, producer, catalog and calibrated known-process
activity guard. Fixture/cache preparation and result export are outside the
clock. No samples or cells are discarded, and no statistical-significance or
universal speedup claim is made.

Representative duration ADD medians in microseconds, baseline → candidate:

| Configuration | First grid | Reverse-order grid |
| --- | --- | --- |
| Direct, nullable 100K rows, 64/65 requested chunks | 608 → 280 | 622 → 278 |
| Whole expression, same fixture | 819 → 484 | 819 → 510 |
| Direct, nullable 1K rows, 983/984 actual pieces | 653 → 65 | 667 → 73 |

The first grid has ten slower medians; the confirmation has fourteen. Every
slower cell and its complete samples/ranges remain in the reports. In the
confirmation, the only slower median with separated three-sample ranges is
cached direct duration multiplication at 100K rows: 412 µs (410–414) →
417 µs (416–420), a 1.21% increase. Small flat/cached controls can be slower by
one or two microseconds, with overlapping ranges. No regression-free claim is
made.

A material first-grid whole-expression scalar floor-division result is retained:
546 µs (495–576) → 1,315 µs (1,296–1,348). The complete reverse-order
confirmation measures 562 µs (513–584) → 424 µs (404–456) for the same cell,
with unchanged binaries. Source and native-code inspection found the same
inlined floor arithmetic and no per-row checked-helper call, but did not establish
a cause. The lower second result is not an engine fix, and does not replace the
first result. Both grids, all controls and baseline/candidate epoch drift remain
visible. The workload uses one row worker and thirteen 8,192-row batches at
100K rows; this result is not attributed to parallel row execution.

## Heldout regression control

Both revisions independently match fresh DuckDB answers for all 43 unchanged
quick-tier queries. All 86 target verdicts, 258 target worker/query records and
774 retained target snapshots verify. Aggregate candidate/baseline elapsed
geomean is 1.00235×, effectively unchanged. Individual query timings were not
inspected. Existing Polars discrepancies (35 wrong, eight correct) remain visible
without skips. This is a quick-tier regression control, not a full-tier headline
or evidence that every temporal route is exercised by an external query.
The actual prior admitted `fcfbc918` exporter is reused at its original paths;
its timing observations are fresh. The final candidate exporter is separately
built and admitted. Both heldout exporters use the same CPU-provider override;
the separate native composition probe uses the installed Metal provider.

## Evidence and remaining scope

Artifacts are retained under
`build/1003-temporal-binary-validation-6f8e4b55`, including the separate `final`
candidate directory. M1 evidence is under
`/Users/randyzwitch/benchdata/1003-temporal-binary/fcfbc918`. The original Linux
baseline protocol and receipts remain at their actual paths; a later collector
metadata correction records the actual native-provider command on Darwin.
The producer, oracle, case catalog and controller are byte-identical across
that correction. Initial failed attempts and the superseded candidate build remain retained
without being counted as passes or relabeled as the corrected candidate.

This completes no other calendar, cast, decimal, text, nested, categorical or
execution-adapter family. The full #1003 inventory remains open.
