# Conditional predicate chunk consumption

This #1003 subset removes predicate materialization in three execution paths:
ordinary conditionals, null-otherwise masked branches, and whole numeric
branches whose values already exist. It reads eligible host Boolean windows
directly and preserves the required selection masks and owned output validity.

## Preparation and ownership

`_ConditionalFlags` retains a typed Boolean column and its actual adapted
length. Streaming inputs additionally retain the original Series owner. Each
consumer creates a fresh owned `_BoolInput` cursor, so repeated selection and
validity passes start at the first logical row. Pointers and offsets are
captured once per physical window, and the cursor advances once per window.
Selection fills the existing three masks. Validity uses masked partial-byte
heads/tails and direct complete-byte stores into its owned result, without
another packed truth buffer. Numeric outputs retain the branch's value owner,
physical offset and logical dtype; only validity changes.

Normal predicate preparation preserves the existing flat and published-cache
adapters. Eligible cold inputs have exact logical BOOL type, Boolean physical
arrays, host-readable buffers, consistent cumulative ends and sufficient
visible bitmap storage. Scalar operands and safe prefixes retain their old
behavior. Unsupported preparation follows the original merge and extraction
path before evaluating the branch.

Whole-column reuse requires a separate rule. `Series.slice` reads original
arrays and ignores a published merge cache. After the unchanged whole-branch
eligibility checks, an existing `_whole` predicate can therefore reuse its
original windows, including an original flat slice. Every original array and
cumulative boundary is validated, even when a cache exists. A published cache
whose initialized values differ remains observable by normal preparation,
while whole-row evaluation continues to observe the originals.

The existing `_whole` recognizers cover source columns, identity casts and
prepared states. Computed predicates such as `p.fill_null(False)` retain
`_rows`; an eligible chunked result can avoid its final merge. Declined lookup
or admission executes the original `_rows` and preserves its subsequent
length/type check and optional fallback.

## Semantics

Selection remains predicate-valid AND raw-true. The selected mask is independent
of outer activity; branch masks additionally restrict observation to active
rows. Hidden NULL-true payloads select otherwise. Checked branches observe only
their selected rows, with the same global error positions and diagnostics.

The wrapper retains the actual predicate length, including scalar length one.
Existing numeric value-alias eligibility is unchanged. Flat/published helpers
remain byte-identical, retaining their original bitmap-padding behavior.
Original-window reuse reads bounded visible bits, matching the old `_rows`
copy. Own-validity pointer preparation still happens at size zero.

Output names, schemas, order, nulls, numeric raw payloads, temporal units and
timezones, and decimal precision/scale remain covered. Required output masks
remain owned. Explicit `Series.rechunk()` retains its contiguous ownership and
cache behavior.

## Comparison and source provenance

Production starts from main `c9911860` on the independent
`work/1003-conditional-predicate-chunks` branch. Initial candidate A uses
production `a223cc7f` and incremental HEAD `ff76f2fb`. Refined candidate B uses
production `a26891b7` and incremental HEAD `e005cc8b`. Both comparisons use
admitted BASE `627b7e0f`, with the preceding String-cast and Boolean/String-
choice changes present on both comparison sides. The new predicate helpers
and contract source match between each production/incremental pair; production
retains its independent main-based context. All A sources and artifacts remain
retained separately from B.

The producer and case catalog were frozen before compilation. The independent
oracle, controller and receipt reader were frozen before the first runtime
observation, with protocol SHA256 `ecb63ecf9520bb92edc3ae4015c7fc904fc06306341e640263616569b3e56059`.
All 56 functional cases and 32 cost cases remain retained. Predeclared
representative groups 0, 1, 4, 5, 20 and 25 execute first; execution indices and
original group IDs are recorded separately. There are no performance-selected
case skips. Development-profile refinements retain the original results and
the complete case catalog; heldout observations do not choose the implementation.

## Correctness and native compatibility

Both Linux and live M1 pass 53 complete functional answers and three exact
error-byte controls per revision. The separate qualifier checks 171 literal
records and six private initialized-storage/empty-input controls per revision
and host. Full positive stdout and all private stdout/exit outcomes match
within each host. Linux SIGILL and Darwin SIGTRAP outcomes are separately
observed; raw ASLR-bearing stderr is retained without a byte-equality claim.

Baseline CPU contracts passed as five unchanged tests in the original suite
plus the corrected Date fixture rerun separately. Date uses this repository's
Int64 temporal storage. The six candidate contracts pass together, including
cold-cache assertions through the reader. Initial fixture compilation and
storage-assumption failures are retained with their actual source versions.

Initial A broader CPU-provider validation on M1 passes all 117 selected tests
across 20 modules, with zero expected failures, unexpected failures or skips.
Refined B passes all 111 tests selected by the same contracts on its
independent main-based Linux production snapshot, with zero expected failures,
unexpected failures or skips. Its six candidate contracts also pass together.

Both A and B live native M1 validation pass CPU/accelerator by flat/chunked
contexts, each with eight retained Boolean/String queries and four numeric
queries on fresh frames. Numeric cases cover null otherwise, explicit
otherwise, checked Int8 observation and conditional SUM34. All four observed numeric outputs shared
their input value owner; offset seven and hidden 10000 payloads survive frame
destruction. This observation adds no GPU alias invariant.

An explicit actual Metal SUM130 followed by host conditional composition also
passes for both revisions. These checks establish host-materialized
compatibility; they make no GPU-predicate or GPU-performance claim. The initial
native fixture typing failure is retained and its literal values/expectations
are unchanged.

## Allocation results

All 32 refined B paired Linux allocation cases pass independent complete-answer
and raw-payload verification, with zero tracking errors. Calls, requested bytes
and freed bytes decrease in 16 and remain equal in 16; none increases. Live
bytes on return are equal in all 32. Peak newly owned bytes decrease in 14,
remain equal in 15 and increase in three.

Examples at 262,153 rows and four actual workers:

| Existing executor route | Requested bytes BASE → HEAD | Calls BASE → HEAD |
| --- | ---: | ---: |
| Whole source predicate, flat | 316,211 → 32,858 | 127 → 2 |
| Whole source predicate, 64 chunks | 418,483 → 32,858 | 505 → 2 |
| Computed fill predicate, flat source | 461,898 → 343,187 | 452 → 442 |
| Computed fill predicate, 64 source chunks | 477,578 → 358,867 | 734 → 724 |
| Explicit owning Bool merge, 64 chunks | 95,410 → 95,410 | 8 → 8 |
| Prepared flat output-mask control | 65,629 → 65,629 | 4 → 4 |

Computed predicates retain evaluation and row assembly; their savings concern
final preparation. Whole-source reuse also retires identity row assembly.
Input copying, unaligned-mask staging, required output storage and lifetime
effects are distinct. These byte differences are not attributed solely to the
two packed input buffers.

All three B peak increases are retained: public Boolean FILL rises from
185,520 to 191,840 bytes (+6,320); public String COLUMN from 4,808,152 to
4,924,136 (+115,984); public String FILL from 4,833,990 to 4,941,518 (+107,528).
There is no uniform peak-memory improvement claim. Original A allocation
results, including its two peak increases, remain in the original artifact
root and are not relabeled as B observations.

Tiny fixtures request 1,000 pieces and actually retain 983 after slicing;
they use one worker. Big fixtures use 64 actual source pieces and four workers.
Flat-normal, published-cache, explicit-owning and prepared-mask controls retain
their individual classifications. The prepared SHIFT-state case is a direct
initialized state supplied through an existing recognizer, with setup outside
the measured call; it is not public window-computation timing.

## Timing and heldout validation

The refined reader (`e005cc8b`) passes all 192 answers in fresh complete M1
BASE and HEAD campaigns, with zero guard overlaps, errors or timeouts. The
unchanged catalog retains all 32 cases and three samples per case and revision:
21 lower, four equal and seven higher medians. Ranges below describe the three
samples; they are not confidence intervals.

| Route at 262,153 rows | BASE median [range], µs | B median [range], µs |
| --- | ---: | ---: |
| Whole source, flat | 539 [372–672] | 70 [54–71] |
| Whole source, 64 chunks | 476 [392–1115] | 71 [56–71] |
| Computed fill, flat source | 148 [142–196] | 168 [137–389] |
| Computed fill, 64 source chunks | 259 [232–260] | 255 [171–456] |
| Explicit owning merge | 807 [788–809] | 805 [787–808] |
| Prepared flat output mask | 71 [66–71] | 70 [66–71] |

The computed flat case retains a 20 µs (+13.5%) median regression. A tiny flat
batch case rises from 5 [5–8] to 6 [5–8] µs (+20%). These are the two defined
ratios above ten percent. A ready-merge control has BASE samples [0, 1, 0] and
B samples [1, 0, 1] µs; its median ratio is undefined because the BASE median
is zero. All samples remain retained. Overlapping ranges do not establish a
noise explanation or absence of regression, and there is no uniform speedup
claim.

The original A reader (`ff76f2fb`) also passed all 192 answers but had material
computed regressions: 138 [131–159] → 424 [403–441] µs for flat source and
254 [229–258] → 487 [425–978] µs for fragmented source. B hoists bitmap pointers
and cursor advancement per physical window, uses direct complete-byte stores,
and retains typed flat flags. It preserves all three sites and computed
`_rows` fallback. All original A sources, binaries and distributions remain
intact; none supplies B's fresh baseline observations.

Both revisions independently match fresh DuckDB answers for all 43 unchanged
quick-tier ClickBench queries: 86 target verdicts and 774 verified target
snapshots, with zero target errors. Aggregate HEAD/BASE elapsed geometric mean
is 1.0070145973 (+0.70%); only the aggregate was inspected, and heldout timings
did not choose the implementation. Existing Polars discrepancies (35 wrong,
eight correct) remain retained without skips.

The accepted BASE exporter is reused at its original admitted path and build
identity; B uses a separately built exporter from `e005cc8b`. Benchmark drivers,
queries and observer policy are unchanged. This frozen plan contains only the
base variant: full external perturbation/development-suite coverage and
helper-specific multi-query trace coverage are not claimed. Functional timing
fields and compiler resource diagnostics support no engine elapsed claim.

## Evidence and remaining scope

Source/SDK/native/build/observer manifests, frozen oracles, complete outputs,
private-process diagnostics and paired allocation statistics are retained under
`build/1003-conditional-predicate-validation-c9911860`. M1 artifacts are under
`/Users/randyzwitch/benchdata/1003-conditional-predicate/627b7e0f`, with local
copyback admissions. Actual HEAD build identities are recorded separately from
the qualifier collector's historical comparison-baseline field.

The full #1003 ledger remains open for calendar/conversion, decimal, text,
nested, categorical and other execution/storage/pipeline consumers. This subset
covers the three conditional predicate preparation sites and the existing
whole-predicate reuse described above.
