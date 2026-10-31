# arctan2 over numeric input windows

This subset of #1003 replaces the mandatory merges in `arctan2_series`.
Production was developed from main `a54c04b4` and rebased onto `62706a5f`
after #1089–#1091 merged. Incremental validation starts from
`12914b49`, including pending #1089, #1090 and #1091 on both sides.
The candidate engine is `841da205`; the independent production engine/tests
are `25494d48` before rebase and `53f2a4f6` afterward. The initial engine `9ccb78bf` and test-only correction
`23b5d31c` remain in the evidence tree.
Before rebase, affected function and test bytes match between independent
and combined checkouts, while whole-file blobs and hunk coordinates differ
because pending changes also affect `binary.mojo`. After rebase, the entire
kernel file and new contract file match the validated `841da205` files.

## Preserved behavior and retained adapters

Public result dtype is Float32 exactly when y is Float32; otherwise it is
Float64. Both operands are converted to that type before the scalar system
`atan2f` or `atan2` call. In particular, Float32 y=1 and Float64 x=1e39
produces +0 after x narrows to infinity. Computing Float64 atan2 first and
narrowing the result would change this behavior.

The route detects exact floating dtype and physical fragmentation, such as
x/y sensor vectors accumulated in independent batches before computing
headings. It has no query-name or data-value trigger.

Exact floating identity casts need no input materialization. Other casts
retain the existing strict converters, in y-before-x order and before length
checking. Decimal scaling, computed logical dtype restoration and private
cast errors therefore retain their existing paths. Those converters can still
merge inputs; conversion cells are reported separately from identity cells.
Private unsupported result-type calls retain the old conversion/failure path.

Published immutable merge caches are reused. Empty or building caches do not
initiate or wait for a merge on the identity route. If both converted inputs
are contiguous, the existing scalar column kernel is used. Otherwise, two
owned `_NumericInput` cursors bound each span by both physical boundaries,
then use the existing column validity/value readers and scalar libm calls
into one result array and validity list. Cursor row indices are relative to
the current physical column; the column readers apply its slice offset. No SIMD, threading or validity-packing redesign is included.

Only an entire one-row operand broadcasts; single-row physical chunks do not.
Null payloads are skipped before either numeric payload or libm call is read,
and direct invalid output payloads remain positive zero. Empty inputs and
length mismatches retain their original ordering: 0/1 is empty, while 0/2 is
incompatible after successful casts. Whole-expression shape rejection can
occur earlier in planner preparation and retains its distinct diagnostic.

## Frozen validation catalog

The independent NumPy oracle converts source values to their source dtype,
then both operands to the selected result dtype before evaluating atan2.
Complete ordered values, field names, schemas, nulls, NaN/infinity
classification and expected zero signs are checked. Finite nonzero results
allow at most four target ULPs, including Float32 subnormals; expected zeros
and their signs remain exact. Direct invalid payload bits are checked after
the measured call.

There are 108 functional observations per host and revision: 96 complete
answers across six dtype pairings and eight shapes, plus 12 exact negative
controls. The direct mismatch diagnostic is `Incompatible expression lengths`;
the whole-expression planner reports `Expression operands have different
lengths after dropping values`. Initial verifier expectations and their failed
logs are retained; neither error is accepted as an arbitrary exception.

The 68 performance cells contain 36 identity-only and 32 strict-conversion
cases. They cover Float32/Float64 and representative integer coercion,
contiguous and independently chunked operands, nonzero slices, dense/mixed
nulls, 0/1K/100K/1M rows, cold/published-cache controls, scalar broadcasts and
direct/whole evaluation. Actual fixture lengths, physical counts and effective
operand lengths are recorded after timing. Some tiny requested partitions
canonicalize identically; requested counts alone are not coverage evidence.

Cold fixtures rebuild all owners outside the measured call. Warm fixtures
populate both source caches before the call. The before/after-call/post-export
cache bitmap includes both fixture columns, even when one is unused by a
scalar case. Absence of a source cache cannot prove absence of a temporary
converter or whole-expression copy. Every complete output is exported and
verified outside the observed interval.

## Allocation results

All 68 pairs verify complete outputs with zero tracking errors. Requested
bytes and peak newly owned bytes fall in 48 cells and are equal in 20;
none increase. Identity-only cases have 24 lower and 12 equal totals;
strict-conversion cases have 24 lower and 8 equal totals.

For Float64 operands with mixed nulls, 100K rows from offset 13 and 64/65
requested chunks:

| Layer / cache | Requested bytes, baseline → candidate | Peak newly owned bytes, baseline → candidate |
| --- | --- | --- |
| Direct / cold | 2,631,581 → 912,980 | 2,538,268 → 912,756 |
| Whole / cold | 3,746,638 → 1,887,449 | 1,031,295 → 897,663 |
| Direct / warm | 912,980 → 912,980 | 912,756 → 912,756 |
| Whole / warm | 3,746,638 → 1,887,449 | 1,031,295 → 897,663 |

These counters measure requested buffer/metadata bytes and peak newly owned
memory, not process RSS or allocator reservation. Existing source buffers and
prepopulated caches are outside the interval. Output values, the existing
Boolean validity list and its final packed bitmap are still allocated.

## M1 elapsed results and retained tradeoffs

Each adjacent grid alternates the two actual binaries across three rounds,
all 68 cells and one complete answer per worker. Both grids verify all 408
answers, with no discarded samples or guard overlaps. They use four execution
threads, one BLAS thread and the unchanged calibrated host-activity guard.
That guard detects known competing jobs; it does not certify every host event.

The first implementation used the generic one-lane SIMD validity accessor.
Its nullable fragmented 100K-row Float32/Float64 direct medians were about
1.56/1.60 ms against 0.67/0.93 ms baseline. Replacing that accessor with the
existing scalar column validity/value readers removes this regression.
Those initial results and failed development checks remain in the parent
evidence directory; they are not presented as results for the refined engine.

For 100K rows from offset 13 (64/65 requested chunks when fragmented), medians
in microseconds are:

| Configuration | First grid, baseline → candidate | Full repeat, baseline → candidate |
| --- | --- | --- |
| Float32 dense contiguous, direct | 522 → 508 | 515 → 511 |
| Float64 dense contiguous, direct | 671 → 670 | 671 → 672 |
| Float32 nullable fragmented, direct | 666 → 627 | 667 → 629 |
| Float32 nullable fragmented, whole | 826 → 769 | 828 → 772 |
| Float64 nullable fragmented, direct | 930 → 746 | 924 → 749 |
| Float64 nullable fragmented, whole | 1,042 → 893 | 1,039 → 896 |
| Float32 nullable published cache, direct | 501 → 511 | 505 → 512 |
| Float64 nullable published cache, direct | 648 → 603 | 652 → 607 |
| Float32 scalar y, direct | 651 → 671 | 653 → 666 |
| Float32 scalar x, direct | 633 → 652 | 628 → 651 |
| Float64 scalar x, direct | 735 → 747 | 732 → 744 |

The scalar controls retain roughly 2–4% overhead while avoiding their input
merge. The Float32 warm direct control also has a small overhead, despite
reusing its published cache and allocating the same bytes. At 1K rows its
warm direct median is 11→11 µs in the first grid and 11→12 µs in the repeat;
the measured clock quantum is 1 µs. No universal speedup is claimed.

The full repeat was prompted by two variable strict-conversion whole-expression
1M-row cells, without engine changes or case selection:

| y / x | First medians, ms | First ranges, baseline / candidate, ms | Repeat medians, ms | Repeat ranges, baseline / candidate, ms |
| --- | --- | --- | --- | --- |
| Int64 / Float64 | 4.152 → 8.891 | 4.116–9.390 / 4.597–9.243 | 4.184 → 4.098 | 4.164–7.916 / 4.087–8.174 |
| Float64 / Int64 | 5.015 → 6.477 | 4.873–5.024 / 4.944–6.758 | 8.145 → 7.078 | 8.088–8.501 / 4.979–8.416 |

The large first-grid median regressions are not reproduced in the full repeat.
The variability remains visible; these cells provide no reliable speedup claim
and the first results are not replaced by the repeat. Strict converters retain
their existing behavior, including any converter-specific input merge.

## Contract and native integration checks

For the initial engine, the focused runner selects 109 public and eight
kernel-equivalence contracts per host. All five new atan2 tests pass on Linux and M1. There are 115 ordinary
passes and two unchanged strict expected failures per host, with no unexpected
failures, skips or process cases. Those existing Float32/Float64 infinite-divisor
modulo failures retain their original policy and diagnostics; they are not
counted as successful atan2 coverage. All 635 runner source fingerprints match
the pre-refinement committed source, with the engine subset also matching
that held-out source manifest. The accessor refinement retains this prior
evidence and separately reruns its affected contracts and frozen catalogs.
The revised engine passes three affected contracts plus an explicit empty
versus fragmented two-row rejection check on each host. The same frozen
108-case functional catalog also passes on Linux and M1; the revised
68-pair Linux allocation grid retains 48 lower and 20 equal totals.

The new tests check sliced unequal windows, scalar-vs-single-row chunks,
source lifetime, published cache retention, Float32 precedence and narrowing,
Decimal32/64/128 scaling and computed logical restoration, public type binding
on empty/inactive rows, and strict private cast/error ordering.

The same frozen native composition probe passes eight cases per revision:
CPU/installed accelerator × contiguous/unequal sliced inputs × Float32/Float64
result types. It checks complete literal values, schemas, names, order, nulls,
zero signs and NaNs, including a separate Float32 arithmetic projection after
atan2. The actual Metal device query passes. Native SDK runtime, compiler,
probe, binary, source and log hashes are retained. This is scalar integration
evidence and carries no GPU performance claim.

## Held-out comparison and final branch

In the quick-tier held-out run, both revisions independently match fresh
DuckDB references for all 43 queries:
86 target cells, 258 target worker/query records and 774 retained target answer
snapshots are verified. Aggregate candidate/baseline elapsed geomean is
1.00136×, effectively unchanged. Individual query timings were not inspected. This is not a full-tier
headline or evidence of external-suite atan2 usage or speedup.
Polars reference disagreements remain visible in the retained results.

The measured baseline includes #1089–#1091 from the start. Updated main also
contains the independent #1088 auto-placement changes in `lazy_api.mojo` and
`planning/device_placement.mojo`. Rebase preserves the entire tested binary
kernel file and new contract file byte-for-byte. The exact final branch gets
a separate focused Linux compile/run; the M1 native and elapsed receipts name
their actual `841da205` integration tree, rather than attributing those binaries
to the later rebase. No performance claim is made for #1088 or automatic placement.

## Retained evidence

Linux artifacts are under `build/1003-arctan2-validation-12914b49`; M1 artifacts
under `/Users/randyzwitch/benchdata/1003-arctan2/12914b49`. Refined-engine
evidence is in each root’s `direct-column` directory. Frozen sources,
build/runtime sidecars, full Parquet answers, logs, cache/fixture observations
and admissions bind the measurements to actual binaries and engine revisions.
The prior numerical oracle and verifier versions remain in the evidence tree.
This is one measured scalar adapter; the remaining #1003 inventory stays open.
