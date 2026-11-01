# Boolean and String conditional branch windows

This #1003 subset removes input merges from eligible Boolean and String
conditional branches. Numeric and decimal choice retain their existing
routes. Cold host-readable branch windows arise when choosing labels or
Boolean flags across concatenated Arrow batches or Parquet row groups.
Unequal physical boundaries and whole-Series scalar branches are supported.

The production branch starts from main `c9911860`, independently of #1098.
Actual incremental BASE `96b99066` and HEAD `627b7e0f` include that preceding
String-cast work on both sides. The two implementation files match exactly
between production and incremental changes. Final production `10430982`
adds 28 lines of public Boolean null-otherwise regression assertions to the
initial contract; this supplemental test-only difference is recorded
separately, rather than claiming complete final test-blob identity.

## Selection and ownership

Boolean choice selects the raw payload bit and validity independently.
A selected null can retain raw true; using a logical/predicate helper that
clears invalid payloads would change the legacy result. The new kernel
writes owned packed payload and validity directly, avoiding byte-per-row
Boolean intermediates as well as source merging.

String choice keeps StringBuilder and its existing `_append_row` operation.
Only a valid selected row reads text. A selected null appends an empty span.
The forward cursor owns the Series and current StringColumn, preserving
backing owners for offsets, views and long external text until the copy is
complete. This changes input traversal, not the required owned output.

The output length is the selection length. Only a whole-Series length of
one broadcasts; physical one-row chunks do not. Longer inputs expose their
selected prefix. The legacy path also permits short or empty inputs when
no chosen row accesses their missing tail. These shapes fall back instead
of gaining a new unconditional length error.

Admission checks both logical types, every physical window or published
cache, host backing storage, visible offsets/descriptors, bitmap coverage
and total row coverage before an empty-output shortcut or cursor begins.
Cold encoded String inputs retain the legacy preparation route, which can
read dictionary/code metadata before observation even on unused branches.
Unsupported logical/physical/storage shapes keep their old adapter and
preparation priority.

Published caches are reused separately for each operand without initiating
or waiting for an unpublished cache. Boolean operands that originally have
chunked metadata can still use the new packed-output kernel when both
caches are ready; their allocation changes can therefore include output
intermediate retirement rather than input-copy removal. Both-flat inputs
retain their existing kernel. String warm-cache traversal retains Builder
output and is measured separately.

## Public execution

The null-otherwise execution route previously merged its branch before
calling choice. A matching Boolean/String branch now takes the admitted
window route immediately before that merge. Predicate evaluation and
preparation, active/selection masks, then-branch evaluation and errors keep
their existing order. Numeric null-otherwise validity reuse is unchanged.
Omitted and typed-null otherwise, scalar predicates/branches, outer masks
and chained conditions remain part of validation.

Conditional predicate merging and final batch output assembly are distinct
consumers; this change does not claim to remove them. The complete #1003
inventory remains in scope.

## Correctness and compiler observations

Six baseline-compatible contracts pass on unchanged main. The initial
candidate passes all six, including new cold-cache assertions, with the
pinned compiler at default O3 using one and two compiler threads. A prior
two-thread build crashed in X86 instruction selection for unchanged String
NE code before producing an executable. The exact driver and sources were
retained. An isolated String NE case and unchanged full builds at one and
two threads subsequently passed. The crash was not reproduced and its
cause remains unproven; no source workaround was applied.

Both hosts pass the frozen qualification: 28 complete outputs and 670
independent literal rows, including all 32 Boolean raw/valid/selection
states and safe scalar/short/empty/prefix shapes. Eight initialized private
cases preserve exact same-host signals and complete stdout diagnostics:
Linux SIGILL and M1 SIGTRAP. Raw stderr is retained and hashed separately;
stack addresses, executable names and unwind metadata are not claimed byte
identical. The allowed framing rule was declared before candidate runs.

Native M1 compatibility passes four CPU/accelerator by flat/chunked
contexts with independent literal answers, ownership/null payload checks,
and actual Metal Int64 SUM=130 followed by host conditional selection.
These checks establish host-materialized compatibility, not GPU choice
offload or performance.

All 111 broader CPU checks pass at default O3 with two compiler threads,
including the final public Boolean omitted/typed-null assertions. All 645
source fingerprints match production `10430982`; there are no unexpected
or declared expected failures in these two contract groups.

Both revisions pass all 56 functional cases per Linux/live-Mac host.
The catalog covers both families, bitmap offsets, scalar directions,
nullable and outer-masked public conditionals, safe short/empty/prefix
inputs, owner lifetime, coded/Binary fallback and numeric controls.

## Allocation

All 46 matched Linux heap cases pass complete answer checks with zero
tracking errors. Calls, requested bytes and freed bytes decrease in 27
cases and stay equal in 19; none increases. Peak newly owned bytes decrease
in 21 and stay equal in 25. Live bytes on return decrease in 14 and stay
equal in 32. Lower freed/live bytes alone are not a universal improvement.

The scope begins after source fixture/cache preparation. It measures newly
owned allocations rather than process RSS or all input ownership. Direct
kernels prepare selection outside the scope; whole-expression measurements
include predicate preparation, branch masks, evaluation and output assembly.
Each row below compares identical source configuration across revisions.

| 100K-row case, 64/65 branch chunks unless noted | Requested bytes BASE → HEAD |
| --- | ---: |
| Cold direct Boolean choice | 346,678 → 25,176 |
| Both-published Boolean input, new packed output | 237,764 → 25,176 |
| Cold whole Boolean explicit otherwise | 851,601 → 510,299 |
| Cold whole Boolean null otherwise | 723,985 → 446,643 |
| Cold direct String choice, offsets | 7,774,280 → 3,172,044 |
| Both-published String input | 3,172,044 → 3,172,044 |
| Cold direct String choice, views | 6,435,440 → 3,172,044 |
| Cold direct String choice, mixed layout | 8,892,764 → 3,172,044 |
| Cold whole String explicit otherwise | 11,136,578 → 6,180,813 |
| Cold whole String null otherwise | 6,294,063 → 3,834,962 |

The tiny 1K-row fixtures request 1,000 pieces and expose 983/984 actual
branch chunks. Boolean requested bytes fall 884,711 → 426, and String
953,422 → 26,805. String view merges can share UTF-8 backing buffers, so
all reductions are not attributed to byte copying; merge metadata and
Boolean intermediates are separate costs. Both-flat legacy and existing
numeric controls remain measured and equal. Warm Boolean output retirement
is explicitly distinct from cold input-copy avoidance.

Direct cache observations distinguish input preparation from output:
baseline cold source caches become published and candidate eligible cold
caches stay unpublished. Whole expressions can merge temporary slices while
leaving original source caches unchanged; a zero parent-cache observation
is not proof of zero copying.

## Timing observations and tradeoffs

All 46 frozen timing cells have three complete answer-verified samples per
revision on the live Mac: 276 answers, with zero calibrated activity-guard
overlaps or errors. Medians are lower in 38 and higher in eight; none is
equal. These descriptive three-sample observations do not establish a
universal speedup. Every cell, sample and control remains retained.

| Case | BASE median µs | HEAD median µs |
| --- | ---: | ---: |
| Cold direct Boolean, 100K rows | 1,062 | 269 |
| Cold whole Boolean explicit otherwise, 100K | 1,385 | 503 |
| Cold direct String offsets, 100K | 5,546 | 3,378 |
| Cold whole String explicit otherwise, 100K | 4,077 | 3,729 |
| Both-published Boolean, 100K, new packed output | 417 | 245 |
| Both-published String, 100K | 5,117 | 3,279 |
| Cold Boolean, 1K rows / 983–984 chunks | 580 | 188 |
| Cold String, 1K rows / 983–984 chunks | 716 | 275 |
| Cold whole Boolean null otherwise, 100K | 955 | 433 |
| Cold whole String null otherwise, 100K | 3,143 | 3,018 |

All eight higher-median cases are disclosed below. The largest relative
increase is tiny, already-published Boolean choice: 6 → 10 microseconds
(+66.7%, an absolute four microseconds), despite lower output allocation.
Flat String direct and whole controls are +4.7% and +4.5%. Cold String
whole choice with a scalar left branch is +0.53%. No slower case has fully
separated ranges; that does not establish absence of regression.

| Higher-median case | BASE median [range] µs | HEAD median [range] µs |
| --- | ---: | ---: |
| Flat Boolean scalar-left | 401 [388–405] | 403 [391–404] |
| Flat Boolean scalar-right | 391 [389–404] | 403 [391–405] |
| Flat String direct/vector | 3298 [3298–3449] | 3452 [3436–3560] |
| Flat String whole/vector | 3562 [3538–3830] | 3722 [3673–3837] |
| Cold String whole/scalar-left | 3967 [3951–4232] | 3988 [3958–4186] |
| Tiny both-published Boolean | 6 [6–9] | 10 [5–13] |
| Existing numeric cold control | 306 [299–310] | 312 [304–314] |
| Existing numeric warm control | 302 [301–315] | 303 [300–319] |

No timing-driven source change, threshold, retry or sample exclusion was
introduced. The earlier compiler reproduction builds are retained as build
diagnostics, not selected timing runs. Cold input-copy savings, Boolean
output-intermediate retirement and required String output remain distinct.

## Heldout validation and coverage limits

Both revisions independently match fresh DuckDB answers for all 43
unchanged quick-tier ClickBench queries: 86 target verdicts and 774
snapshots verified, with zero target errors. Controller and collector exit
zero. Aggregate HEAD/BASE elapsed is 1.0092064345, or 0.92% higher. That
observation is retained rather than described as an engine-wide speedup
or proof of no regression. The frozen ClickBench plan has only the base
variant; its worst and only aggregate is the same value. Full external
perturbation/development-suite coverage is not claimed. Polars' 35 wrong
and eight correct outcomes remain retained without query skips.

No individual heldout query timings were inspected to choose or tune this
change. Benchmark, generator and suite-query code is unchanged. Generic
operator traces cannot prove that more than one suite query uses these
new helpers. Benchmark rule 5 permits absent specialized-path coverage when
its data property and real-world examples justify it; those are stated
above. Mechanism cases retain nulls, sliced/differing boundaries,
offset/view/mixed layouts, both/one-sided caches, scalar directions, tiny
fragmentation, explicit/null otherwise and existing numeric controls.

## Evidence and remaining scope

All source/SDK/native/build/observer pins and complete raw outputs are bound
by external admissions under the validation root. The initial compiler
failure, unchanged successful rebuilds, original fixture setup failure and
collector-only provider-label correction are retained. Neither correction
changed the 56/46 case catalog, expected answers or engine implementation.
Original build sidecars remain attached to their actual commands, with an
explicit source-equivalence proof for collector revision 3.

This subset covers Boolean/String branch selection and the earlier
null-otherwise branch merge. Encoded/logical/unsupported adapters,
conditional predicate preparation and other execution/storage/pipeline
consumers remain recorded in the full #1003 ledger. Required String output
ownership and explicit contiguous APIs retain their contracts.
