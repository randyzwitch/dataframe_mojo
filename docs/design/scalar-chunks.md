# Scalar kernel chunk consumption

The numeric-cast change shipped in #1072, metadata consumers in #1082,
numeric unary kernels in #1084, and numeric multi-input kernels in #1087.
The current validity/fill comparison uses merged main `a54c04b4`. The
historical multi-input comparison uses `e4af628a`, including #1085. Earlier metadata
measurements retain baseline `6d3f987aae999e9e10072f926cc6d9817510fbb8`;
historical cast measurements retain baseline
`2d0dce12a41a07c265025a890a10ae121483dc1b`. Earlier gains are not counted
again in the current work.

The initial inventory distinguishes explicit APIs requesting an owned
contiguous column from internal scalar consumers. The former are retained;
the latter are candidates for aligned chunk-window evaluation.

| Consumer | Input-copy status | Constraints to preserve |
| --- | --- | --- |
| `kernels/cast.cast_series` | Unmasked infallible numeric casts already consume chunks; remaining checked/masked/logical paths merge | Global error-row offset, observed-row masks, strict/non-strict behavior, logical dtype |
| `kernels/binary.math_unary` and `unary` | Numeric chunks covered by #1084; validity predicates are the current subset; decimal adapters still merge | NaN/infinity/signed zero, integer errors, null tests and masks |
| `kernels/binary.binary` and `choose` | Ordinary numeric operations consume physical chunks in #1087; Boolean/fill/missing adapters are the current subset; decimal and other adapters still merge | Different chunk boundaries, scalar broadcasting, empty inputs, conditional observation and dtype promotion |
| `kernels/horizontal` | Float folds and numeric extrema consume chunks in #1087; nullable integer sum uses the current fill-null subset | Different boundaries, row null rules, mixed widths and promotions |
| `kernels/temporal.dt_op` and `cast_temporal` | Metadata retags covered by #1082; calendar/conversion paths still merge | Logical temporal units/timezones, calendar behavior, masks and conversion errors |
| `kernels/list.nested_op` | List length and struct selection covered by #1082; other list/string operations still merge | Child offsets, sliced children, empty/null rows, struct field validity |
| `execution` scalar/text paths | Several preparation and result rechunks | Separate actual required contiguous adapters from avoidable input/result copies |

Plain integer and String set membership, shared Boolean equality, retained
adapters and measured tradeoffs are documented in
[scalar-membership-chunks.md](scalar-membership-chunks.md).

## Measurement plan

Compare identical values as contiguous input, nonzero slices and uneven
64/1024-chunk input, with operands having different boundaries. Include
small-input controls, null densities, scalar operands and multiple workers.
Measure direct kernels and whole expression evaluation separately. Direct
`Series.rechunk()` caches the merged array, so separate fresh chunk metadata
from warm operations. Expression evaluation creates batch slices with fresh
chunk metadata: its copies need not populate the original source cache. Fresh metadata shares the same immutable source buffers and is
constructed outside timing. Record retained numeric merge-cache data bytes
after each direct kernel, whole-operation time and process peak RSS; a zero
source-cache count during batched expression evaluation does not prove that
no copying occurred. For conditional kernel measurements, prepare the
selection outside timing and retain the whole-expression measurement as a
separate control. Also distinguish fixture/output memory from kernel-only measurements.

Implement only demonstrated improvements. Keep explicit contiguous fallbacks
where required or faster. Preserve complete output schemas and row order,
dictionary semantics, error-row numbering and inactive conditional rows.
Use independent expectations and the existing contiguous path as a
differential control. Keep generated inputs, binaries and profiles outside
the repository.

The remaining numeric copy inventory independently verifies 36 direct-kernel
cases at current main `6d3f987a`: 1M dense rows, nonzero slices, contiguous or
64/1024 uneven chunks, and cold/warm sources. At either fragmented shape,
sine, integer addition with a scalar, and absolute value retain 8 MB of
merged input values; conditional choice and horizontal sum retain 16 MB.
Contiguous controls retain none. The already-merged numeric cast also
retains none and is a control, not a new improvement. Warm cases retain the
same caches from warm-up; these byte counts do not imply repeated copying.
All complete ordered outputs and exact Arrow schemas pass independent NumPy
checks (sine uses the declared 1e-14 tolerance). All elapsed observations from
this compiler-overlapping inventory are excluded. Retained input-value bytes
are distinct from allocation totals, peak newly owned memory and process RSS.
Evidence: `/tmp/1003-numeric-copy-inventory/receipt.json`. These consumers
remain open beyond the metadata subset below.

## Remaining metadata consumers

The current-main inventory also finds merges before operations that need
only a small part of the input storage:

- String byte lengths need offsets or view lengths; character lengths read
  row bytes but do not need a merged byte buffer.
- List lengths need parent offsets and validity, not child payloads.
- Struct field projection needs the selected field and the parent mask,
  rather than every other field's buffers.
- Time-zone conversion and `with_time_unit` retag stored ticks without
  changing them.

The first remaining-consumer candidate addresses these paths. Other string,
nested, math, binary, horizontal and checked-conversion consumers remain in
scope; this metadata subset does not complete #1003. Existing contiguous
paths and explicit contiguous adapters are retained.

String lengths and struct projection borrow physical storage directly rather
than first constructing temporary `Series` views for every chunk. Byte
lengths read only offsets or view lengths; they do not copy an owning row-byte
adapter. Parent masks and child owners retain their existing behavior.
Struct schemas encode categorical field types without their dictionaries,
so different source chunks can carry different categorical domains. That
case uses the existing batch dictionary union for the selected field only;
unselected fields remain unmerged. Compatible domains retain field chunks.

The baseline probe has independently checked complete values, nulls, row
counts and exact Arrow schemas for 30 small/empty cases and 35 scoped
allocation cases. The latter compare contiguous input with 64 chunks at
100,000 rows, nullable windows starting at row 13, cold metadata and warmed
source caches. Direct kernels and selected whole expressions are separate.
The calibrated allocator observer matches the installed runtime and has zero
tracking errors in those cases. Requested/live/peak newly owned bytes include
output and temporaries; existing fixture buffers and warm caches are outside
the scope. Process RSS includes those excluded buffers and untimed export.
Instrumented elapsed times are excluded from performance evidence.

The initial baseline evidence remains outside the repository at
`/tmp/1003-current-main/baseline-admission.json`.

A subsequent frozen paired comparison uses the same probe source against
current main `6d3f987a` and metadata implementation `f84b5a9d`. All 220
observed calls (55 cases, two implementations, two independent calls) pass
complete ordered-value, null, row-count and exact Arrow-schema checks. Each
call exports its own complete answer. Offset-backed and view-backed strings,
empty and small inputs, direct and whole-expression evaluation, and cold and
warm sources are included. The calibrated observer reports zero tracking
errors throughout.

For nullable 100,000-row windows starting at row 13, in 64 cold chunks:

| Direct operation | Requested bytes, main / candidate | Peak newly owned bytes, main / candidate |
| --- | --- | --- |
| String byte/character length, offsets | 7,028,792 / 812,980 | 6,905,752 / 812,980 |
| String byte/character length, views | 2,488,896 / 812,980 | 2,427,928 / 812,980 |
| List length, 16 child values per row | 14,600,848 / 833,247 | 14,525,992 / 824,576 |
| Struct field, nullable parent | 10,103,264 / 1,673,053 | 9,838,972 / 855,733 |
| Time-zone retag | 877,125 / 17,991 | 827,286 / 9,280 |
| Time-unit retag | 859,446 / 312 | 827,286 / 264 |

All corresponding contiguous direct cases have identical allocation counts
and byte totals. Unchanged string-contains and temporal-year controls also
match. Struct projection increases the allocation count (136 to 408) while
reducing bytes by avoiding unused fields; per-chunk metadata still costs
work. Warm source caches exclude their already-owned merged buffers from
these newly-owned counters, so cold and warm results must remain separate.

Evidence is retained in ignored local build storage:
`/home/randyzwitch/GitHub/dataframe_mojo/build/1003-metadata-validation-f84b5a9d/`.
`paired-alloc/receipt.json` records the complete observations, and
`build-provenance.json` pins compiler, source, binary and native-library
hashes. These allocation observations are from Linux. A zero merge cache on
a whole-expression source does not prove zero temporary copying.

The same frozen probe also completes all 99 elapsed-time cases on the live
M1 CPU: five independently exported calls for each revision per case, 990
calls total. Every ordered output, null, row count and exact Arrow schema is
checked independently, and all retained answer and binary hashes are verified.
The Darwin activity guard uses `ucomm` and is calibrated against a live Mojo
compile/run process. No compiler or known suite worker overlaps were detected
in the complete guarded run. The input cases,
oracle and timing windows match the Linux protocol; only host paths and
unavailable platform instrumentation differ. Allocation counters remain the
separate Linux observations above. M1 process RSS is not collected.

For nullable 1M-row windows in 64 cold chunks, median elapsed times are:

| Operation | Direct main / candidate, ms | Whole expression main / candidate, ms |
| --- | --- | --- |
| String byte length, offsets | 3.095 / 0.813 | 1.738 / 0.817 |
| List length | 13.233 / 2.818 | 3.263 / 1.604 |
| Struct field | 11.202 / 7.289 | 3.574 / 2.284 |
| Time-unit retag | 0.802 / 0.001 | 0.844 / 0.749 |

M1 durations are recorded in microsecond steps. The 0.001 ms retag median
is near the timer's resolution and should not be interpreted as a precise
speed ratio.

Reuse changes the tradeoff. With 1,000 rows in one-row chunks, warm direct
struct projection takes 0.565 ms versus the baseline's cached-input 0.009 ms;
this is the largest measured relative regression. Its cold comparison is
0.566 versus 1.649 ms. Warm byte lengths take 0.025 versus 0.002 ms, while
their cold comparison is 0.025 versus 0.317 ms. Warm list lengths take
0.015 versus 0.007 ms. The baseline already owns its merged input before
these warm calls, while the candidate traverses physical metadata again.
Passing explicitly rechunked input retains the existing contiguous path.

The complete host-specific observations are in
`m1-timing-guarded-receipt.json`, with `m1-timing-build-provenance.json` and
`m1-timing-guarded-admission.json` in the evidence directory. Original M1
answer files remain under
`/Users/randyzwitch/benchdata/1003-metadata/f84b5a9d/timing/`.
The interrupted Linux timing run and the M1 run with the inadequate `comm`
guard are retained as separate diagnostics. Their timings are not combined
with the calibrated run. These are mechanism measurements, not a general
engine speedup claim.

The guarded M1 CPU quick-tier ClickBench run also completes all 43 queries,
three rounds and three repetitions, with the exact same hashed 1M-row input
as Linux. All 86 candidate/baseline cells explicitly pass independent DuckDB
value/schema validation, and all 774 retained target snapshots load with
their recorded hashes. Only the aggregate elapsed comparison was inspected:
candidate/baseline geometric mean 0.998x, effectively unchanged. No per-query
held-out timings were used to choose or tune this implementation.

The Polars reference has 35 wrong-answer verdicts and eight successful cells;
those discrepancies remain visible and do not substantiate a performance
claim. Both measured Mojo revisions independently match DuckDB on all queries.
This is a quick-tier diagnostic, not a full-tier headline report. Admission
and verdicts are retained in `m1-heldout-admission.json` and
`m1-heldout-verdicts.json`; original results and snapshots remain under
`/Users/randyzwitch/benchdata/1003-metadata/f84b5a9d/heldout-guarded/` and the
driver-owned answer directories recorded there. CPU-provider include paths,
native-library hashes, the calibrated host guard and exact source/binary
provenance are recorded in `m1-heldout-manifest.json`.

The frozen metadata implementation passes all 109 selected CPU contracts on
Linux and the live M1: chunk-window, temporal, nested and text/binary groups.
The two pre-existing expected-failure declarations are unchanged. Coverage
includes complete outputs and schemas, UTF-8 offsets/views, uneven nonzero
slices, empty/null windows, parent/child masks, dictionary and field metadata,
shared temporal ticks, and outputs that outlive their source owners. These
checks include distinct categorical dictionaries with complete label domains
and recoded output values. They cover this subset; they do not close the
remaining #1003 inventory.
The native Metal-provider integration also passes on the live M1, asserting
the provider is installed and checking complete string/list lengths, struct
fields and temporal retag outputs through `collect(engine="accel")` with
two-row batches. This is provider integration for CPU metadata operations,
not a claim of new GPU kernels. The M1 engine and test trees match the frozen
implementation. Provenance: `m1-provenance.txt` in the evidence directory.


## Numeric unary consumers after the metadata change

The next comparison starts at merged main `015e2f70`, including #1082,
#1080 and #1081. None of those earlier changes are counted as new gains.
The final numeric implementation is frozen at `226d5ff3`; expanded contracts
are at `236264b1`. Benchmarks, queries, generators and evaluators are unchanged.

Float32/Float64 math and numeric unary operations borrow physical columns,
restart SIMD only inside each valid physical window, and write one owned
output buffer. Sliced validity bits are appended at global output offsets.
Integer unary operations assemble complete validity before applying the
global observation mask, preserving hidden/null overflow values and existing
error messages. Integer sign writes the original integer type; integer
truncation shares its existing owning input. Other integer math uses the
already chunk-aware Float64 cast, retaining the conversion buffer but
avoiding a preceding integer merge. Contiguous dispatch remains unchanged.
Decimal conversions, null/float predicates, binary/conditional/horizontal
consumers and other previously listed adapters remain in #1003.

All 74 selected arithmetic/chunk-window contracts pass on Linux and the
live M1, with two unchanged expected-failure declarations. Every recorded
engine and test source hash matches committed `226d5ff3` on both hosts.
Coverage includes complete eager/lazy math expectations, all integer widths,
global masks hiding overflow poison, mixed bitless/nullable/bitless chunks
across bitmap bytes, Float32/Float64 SIMD/scalar tails and NaN/infinity/signed
zeros, output ownership after sources drop, and large unsigned-to-float math.
The Linux runner records the final revision and retains its dirty-worktree
metadata; the hashes verify the exact committed engine and test sources. M1
uses an isolated source archive, likewise identified by complete source
hashes rather than invented Git metadata.

A fixed paired Linux matrix independently checks all 144 allocation calls
(72 cases, one call per revision), including complete ordered values, nulls,
row counts and exact Arrow schemas. It covers Float32/Float64/Int64 sine,
absolute value and sign, integer truncation, unchanged numeric-cast and
scalar-add controls; contiguous and 64-chunk inputs; nonzero slices; dense
and nullable rows; direct and whole expressions; empty inputs; and warm
one-row chunks. Every retained answer hash is verified. The calibrated
runtime/libc observer reports zero tracking errors. Timing from these
instrumented calls is excluded.

For 100,000 nullable rows starting at row 13, in 64 cold chunks:

| Operation | Requested bytes, main / candidate | Peak newly owned bytes, main / candidate |
| --- | --- | --- |
| Float32 sine | 865,918 / 433,247 | 825,736 / 424,576 |
| Float64 sine | 1,672,114 / 833,247 | 1,625,736 / 824,576 |
| Int64 sine | 2,484,870 / 1,646,003 | 2,438,268 / 1,629,396 |
| Int64 absolute value | 1,672,114 / 833,247 | 1,625,736 / 816,864 |
| Int64 sign | 1,772,114 / 833,247 | 1,725,512 / 824,576 |
| Int64 truncation | 859,358 / 224 | 827,286 / 224 |

Every contiguous case has identical allocation counters between revisions,
as do the unchanged direct cast/add controls. Whole-expression cast/add
controls have identical requested totals but their peak live bytes vary
with worker scheduling. Requested/live/peak newly owned bytes
exclude existing fixtures, warm caches and allocator reservations; they are
not process RSS. Warm baseline source caches are already owned before each
observed call. Source-cache absence alone is not used as a copying metric.
Evidence remains untracked under
`/home/randyzwitch/GitHub/dataframe_mojo/build/1003-numeric-validation-inline/`:
`paired-alloc/receipt.json`, `allocation-admission.json`, and `provenance.json`.


Final native Metal-provider integration also passes on the live M1 with
`collect(engine="accel", batch_size=2)`: complete Float32/Float64/integer
absolute values, signs and truncations, including nulls, schemas and row
order. This validates provider integration for CPU scalar operations; no
new GPU kernel is claimed.

The same frozen probe completes all 432 M1 CPU timing calls (72 cases,
three independently exported calls per revision). Every full ordered answer
and exact Arrow schema passes the independent oracle, and every retained
answer and binary hash verifies. The calibrated Darwin guard uses `ucomm`
and excludes compiler/known-suite-worker overlap. At 100K nullable rows in
64 cold chunks, direct Float64 sine medians are 0.493 / 0.448 ms. Linux
allocation observations remain separate from M1 elapsed observations.

The shared integer writer is kept inline to retain the contiguous loop's
allocation/constant-offset optimizations. A separate 200-call paired repeat
checks the already-contiguous 100K-row Int64 absolute-value case and an
unchanged cast control: median 0.143 / 0.146 ms for absolute value and
0.273 / 0.272 ms for the cast. All full answers and retained hashes pass.
The preliminary out-of-line implementation's reproducible contiguous
slowdown is retained under `1003-numeric-validation/`; it is superseded and
excluded from final implementation claims.

Whole-expression samples show startup variation. In the short three-call
matrix, 1M-row/64-chunk Float64 absolute-value medians are 0.685 / 1.506 ms,
and sign medians are 0.746 / 1.054 ms. The complete samples remain retained.
A supplemental 300-call comparison uses the identical frozen program and
inputs, five alternating process pairs and ten repetitions each, with no
sample filtering. Full independent answers and retained hashes pass. Its
medians are 0.690 / 0.571 ms for absolute value, 0.719 / 0.606 ms for sign,
and 0.721 / 0.725 ms for the unchanged cast control. These observations do
not establish a general frame/engine speedup or a blanket non-regression.

Warm fragmented input still has a tradeoff: for 1K rows in one-row chunks
(requested 1024, canonicalized to 1000), direct Float32 absolute value takes
16 us versus the cached baseline's 2 us. The baseline already owns its
merged input before the warm call. Passing explicitly rechunked input keeps
the contiguous path. Durations are recorded in 1 us steps; five empty/tiny
samples are zero, so near-resolution results do not support precise ratios.

M1 evidence remains at
`/Users/randyzwitch/benchdata/1003-numeric-inline/`: `paired-time/receipt.json`,
`timing-admission.json`, `contiguous-repeat/receipt.json`, and
`whole-repeat/admission.json`. Local copies of admissions and complete
case medians are in the final Linux evidence directory above.

The final guarded quick-tier held-out ClickBench comparison completes all
43 queries, all 86 candidate/baseline cells and all 774 retained target
snapshots. Both revisions independently match complete DuckDB values and
schemas. All snapshot content/schema hashes verify using the unchanged
answer loader. Only the aggregate held-out elapsed comparison is inspected:
candidate/baseline geometric mean 1.000x, effectively unchanged. This is not
a full-tier headline result. Polars has 35 wrong-answer verdicts and eight
successful cells; those discrepancies remain visible in the retained report.
No benchmark, query, generator or evaluator is changed or tuned here.
The same hashed 1M-row input, four threads, three rounds/repetitions, all
variants, tracing and fresh independent references are used for both
revisions. The baseline suite binary is reused from its verified fresh
`015e2f70` build; the final candidate suite is freshly built against
`226d5ff3`. Admission/provenance/verdicts are under `m1-heldout/` in the local
evidence directory and `heldout/` in the M1 directory. This completes this
numeric unary subset; the remaining #1003 consumers stay open.

## Numeric multi-input candidate

The prepared follow-up uses merged main `e4af628a` and frozen implementation
`78ba0eb8`. It reads ordinary numeric binary arithmetic/comparisons and
conditional choices through physical input cursors. Differing partition
boundaries are intersected without merging either operand; whole one-row
operands broadcast, while physical one-row chunks remain ordinary rows.
Float horizontal sum/mean and numeric extrema traverse each input's windows
into one output, preserving input order. Contiguous paths remain intact.

Validity and observation masks use global row positions. Checked narrow
integer SIMD retains widened overflow detection and only examines observed,
valid lanes for errors. Comparison and extrema bit stores align to global
output bytes. Choice propagates the selected branch's validity. Logical and
decimal fallbacks retain their existing dispatch. Nullable integer horizontal
sum still goes through the fill-null adapter and retains its input merge;
dense integer sums benefit from the ordinary numeric binary route.

All 92 selected horizontal, arithmetic and chunk-window contracts pass on
Linux and the live M1, with two unchanged expected-failure declarations.
Engine and test source hashes match the frozen revision on both hosts.
Tests include all numeric widths, every bitmap offset, different partitions,
scalar broadcasts, masked overflow, zero divisors, dense SIMD, mixed
bitless/nullable chunks, float NaNs/infinities/signed zero and source-owner
lifetimes. The baseline measurement probe separately passes 600 complete
independent value/null/schema cases before the paired comparison.

The fixed Linux allocation matrix passes all 738 observed calls across
369 cases. Each call exports a complete ordered result, independently
checked against NumPy and exact Arrow schemas; all retained answer hashes
verify. The calibrated runtime/libc observer reports zero tracking errors.
Contiguous and unchanged direct controls have identical counters across 138
cases. Instrumented elapsed measurements are excluded.

For 100K nullable rows starting at row 13 in 64 cold chunks, with two
operands using different boundaries:

| Direct operation | Requested bytes, main / candidate | Peak newly owned bytes, main / candidate |
| --- | --- | --- |
| Int64 addition | 2,531,581 / 866,014 | 2,438,492 / 816,864 |
| Float64 comparison | 1,756,501 / 78,434 | 1,650,912 / 40,960 |
| Int8 conditional choice | 410,063 / 127,316 | 338,492 / 114,336 |
| Float32 horizontal sum | 1,707,233 / 800,928 | 1,640,711 / 800,448 |
| Float32 horizontal mean | 1,719,733 / 813,428 | 1,640,711 / 812,948 |
| Int64 horizontal minimum | 2,546,365 / 827,764 | 2,453,211 / 813,204 |

Requested/live/peak newly owned counters include output and temporaries but
exclude existing fixtures, warm merged caches and allocator reservations.
They are separate from RSS and source-cache bytes. Fresh chunk metadata is
prepared outside each observation. Generated evidence stays untracked at
`/home/randyzwitch/GitHub/dataframe_mojo/build/1003-multi-input-validation-78ba0eb8/`:
`paired-alloc/receipt.json`, `allocation-admission.json`,
`allocation-summary.json`, `provenance.json` and CPU admissions.

Native integration passes on both baseline and candidate on the live M1
through `collect(engine="accel", batch_size=3)`, checking complete literal
values, nulls, dtypes and batch order. This exercises provider integration
for CPU scalar operations; it does not claim new GPU execution. All 754
engine and test files in the candidate archive match the frozen committed
contents. The fixed M1 CPU timing matrix completes all 2,214 observed calls (369
cases, three calls per revision), with complete independent ordered answers,
exact schemas and retained answer/binary hashes verified. The calibrated
Darwin activity guard detects no compiler or known suite-worker overlap.
No zero-duration samples occur; the timer records microsecond steps.
Linux allocation counters and M1 elapsed measurements remain separate.

Representative median durations on the M1 are:

| Operation | Direct 100K nullable rows / 64 cold chunks, main / candidate, ms | Whole expression 1M nullable rows / 64 cold chunks, main / candidate, ms |
| --- | --- | --- |
| Int64 addition | 0.342 / 0.294 | 1.359 / 1.259 |
| Float64 comparison | 0.163 / 0.099 | 0.861 / 0.722 |
| Int64 conditional choice | 0.433 / 0.286 | 1.689 / 1.211 |
| Float32 horizontal sum | 0.268 / 0.153 | 1.099 / 0.690 |
| Float64 horizontal mean | 0.386 / 0.261 | 1.445 / 1.011 |
| Int64 horizontal minimum | 0.441 / 0.279 | 1.586 / 1.058 |

Warm fragmented input retains a clear tradeoff: with 1K rows in one-row
chunks (requested 1024, canonicalized to 1000), Int32 multiplication takes
161 us versus the baseline's cached-input 5 us; Int64 choice takes 143 us
versus 6 us. The baseline's merged input already exists before the observed
call. Explicitly rechunked inputs retain the contiguous dispatch. In the
short three-call contiguous 100K-row controls, Int64 addition takes
0.203 / 0.203 ms, Float64 comparison 0.024 / 0.024 ms and Int64 minimum
0.051 / 0.058 ms. These samples do not establish blanket non-regression
or general engine speedup. No cases or slower samples are dropped.
Complete observations remain at
`/Users/randyzwitch/benchdata/1003-multi/78ba0eb8/paired-time/receipt.json`,
with `timing-admission.json` and `timing-summary.json`; copies of admissions
and medians are in the local evidence directory.

The final guarded quick-tier held-out ClickBench comparison completes all
43 queries, with all 86 candidate/baseline cells independently matching
complete DuckDB values and schemas. All 774 retained target snapshots load
and verify their content/schema hashes. Only the aggregate elapsed
comparison is inspected: candidate/baseline geometric mean 1.004x,
effectively unchanged. This is a quick-tier validation, not a full-tier
performance headline. The complete Polars report retains 35 wrong-answer
verdicts and eight successful cells.

Both CPU suite binaries are freshly built against `e4af628a` and `78ba0eb8`.
The unchanged suite, evaluator and hashed 1M-row input use all queries and
variants, four threads, three rounds/repetitions, tracing and fresh
independent references. No benchmark, query, generator or evaluator changes
are included. The full verdicts, raw results, manifest and admission are
retained under `heldout/` in the M1 evidence directory and `m1-heldout/` in
the local directory. Validation is complete for this subset; remaining adapters listed above
are outside it.

## Validity, Boolean and fill consumers after #1087

This subset uses baseline `a54c04b4` and candidate engine/test revision
`c30afb58`. It reads owned parent validity windows for null tests and
`KEEP_NULLS`, packed Boolean windows for NOT/Kleene logic/Boolean fill and
missing comparison, and numeric windows for float predicates, fill and
missing comparison. Physical chunks may have different boundaries, empty
pieces and nonzero bitmap offsets. Only an entire one-row operand broadcasts.
Parent null tests do not inspect or merge string payloads or nested children.

Full packed-bit windows use the native SIMD byte width. They stay within a
physical span and write only at global output-byte boundaries; short spans
and boundaries use masked partial-byte writes. This follows development
measurements of the initial byte loop, not held-out query timings. The two
large contracts scale with SIMD width to exercise full vectors, unaligned
source offsets, different boundaries and scalar truth tables on each host.

Numeric paths apply only to plain matching numeric dtypes. Existing
logical/decimal conversion paths and public schema validation remain in
place. These operators historically ignore their observation-mask argument;
the new paths preserve that behavior. Numeric fill and keep operations retain
selected physical payloads under null rows, while packed Boolean outputs
clear null-result value bits. Missing comparison has no null results, treats
two nulls or two valid NaNs as equal, and keeps full integer precision.
Length compatibility is checked before empty results are returned.

The frozen external probe contains 49 operation/dtype variants. Baseline and
candidate pass all 672 supported functional cells with complete ordered
Parquet outputs independently checked for values, nulls and exact Arrow
schema. Fourteen nested-left whole-expression `KEEP_NULLS` cells retain the
baseline binder rejection. The expanded fixed allocation matrix passes 874
observations across 437 supported pairs, retaining eight unsupported worker
receipts and zero tracking errors. Each call has its own complete result.
Instrumented elapsed observations are excluded. Fixture storage and previously
owned warm caches are outside the allocation scope; the source-cache presence
bitmask is diagnostic metadata, not a byte count or proof of copying.

For cold direct operations on 100K nullable rows sliced at offset 13 across
64 chunks, the final paired allocation matrix records:

| Operation | Baseline requested bytes | Candidate requested bytes | Baseline peak newly owned bytes | Candidate peak newly owned bytes |
| --- | ---: | ---: | ---: | ---: |
| Int64 null test | 884,534 | 25,400 | 838,156 | 25,400 |
| String null test | 15,488,823 | 25,400 | 15,424,187 | 25,400 |
| List parent null test | 27,113,268 | 25,400 | 26,925,736 | 25,400 |
| Struct parent null test | 16,635,529 | 25,400 | 16,286,235 | 25,400 |
| Boolean AND | 159,314 | 25,400 | 100,352 | 25,400 |
| Int64 fill-null | 2,631,581 | 812,980 | 2,538,268 | 812,980 |
| Float64 missing equality | 1,794,177 | 12,900 | 1,663,188 | 12,900 |
| Nullable Int64 horizontal sum | 4,386,701 | 2,468,100 | 4,064,228 | 2,438,716 |

These are scoped allocation observations, not a general query speedup claim.
All 99 focused contracts pass on Linux and M1 CPU, including the 15 new
cases. Both runs retain the same two expected-failure declarations with no
skips, and all 632 source fingerprints match the frozen candidate. Baseline
and candidate pass the actual installed native Metal-provider integration
probe: 21 literal expressions with complete values/nulls/dtypes and batch order
at batch size 3. This checks provider integration for CPU scalar operations.

The fixed M1 timing matrix admits all 2,622 complete independently checked
results across 437 supported pairs, retaining eight unsupported worker
receipts. The calibrated actual-process guard detects no compiler or known
worker overlap. Instrumented Linux allocation times are excluded. Representative
median durations are below; the complete matrix retains every case and sample.

| Operation | Direct 100K nullable rows / 64 cold chunks, baseline / candidate, ms | Whole expression 1M nullable rows / 64 cold chunks, baseline / candidate, ms |
| --- | --- | --- |
| Int64 null test | 0.069 / 0.023 | 0.899 / 0.166 |
| String null test | 0.594 / 0.021 | 3.121 / 0.140 |
| List parent null test | 2.852 / 0.022 | 12.389 / 0.522 |
| Boolean AND | 0.092 / 0.035 | 0.834 / 0.542 |
| Int64 fill-null | 0.412 / 0.167 | 1.466 / 0.834 |
| Float64 fill-NaN | 0.609 / 0.177 | 2.017 / 1.052 |
| Int64 missing equality | 0.202 / 0.152 | 0.978 / 0.580 |
| Nullable Int64 horizontal sum | 0.937 / 0.517 | 2.911 / 2.054 |
| Float32 is-finite | 0.306 / 0.089 | 1.046 / 1.388 |

Warm fragmented input retains an explicit tradeoff. With 1K rows in one-row
chunks (1024 requested, 1000 retained after slicing), the baseline already
owns its merged input before the observed call: Float32-left `KEEP_NULLS`
takes 2 us versus 57 us, and a String null test 2 us versus 48 us. The
candidate walks physical descriptors without creating that merged cache.
Explicitly rechunked input retains the contiguous dispatch.

An initial contiguous 100K-row Int64 null-test observation was 14 / 35 us.
A fixed six-cell supplement used three alternating process rounds with ten
calls per revision, checking all 360 full answers and preserving the original
matrix unchanged. Its per-round median ranges were 13.5–14 / 13–13 us for
that null test, 208–208 / 207.5–208.5 us for the addition control, and
37–38 / 37–37.5 us for the cast control. The original slower whole-expression
Float32 is-finite cell above also did not reproduce in that supplement:
round medians were 0.993–1.018 / 0.548–0.557 ms. All original observations,
all supplemental samples, their ranges and source/binary identities remain
retained. These short timing samples do not establish blanket non-regression.

A separate direct-kernel supplement closes numeric-output `KEEP_NULLS`
coverage: eight left/right dtype combinations across seven fixed shapes,
112 independently checked allocation results and 336 M1 timing results.
It includes numeric, String, List and Struct parents, different operand
boundaries, scalar RHS, dense/null inputs, empty input and warm one-row chunks.
The original fixture/fresh helpers are byte-identical; its NumPy/Arrow oracle
checks complete numeric values, nulls, schema, NaNs and signed zero. All
allocation tracking-error counts are zero, with no detected timing overlap.
For cold nullable 100K-row/64-chunk Int64 inputs/output, requested bytes are
2,731,581 / 812,980 and medians 0.577 / 0.130 ms. String-left/Int64-output
records 17,335,870 / 812,980 bytes and 1.241 / 0.132 ms; List-left/Int64-output
records 28,860,315 / 812,980 bytes and 4.566 / 0.127 ms. Warm one-row chunks
retain the same cached-baseline tradeoff: String-left/Float64-output takes
7 / 65 us. This supplement measures the direct numeric-output kernel; it does
not qualify rejected nested whole expressions. Every original rejected cell
remains in the main receipts.

The unchanged complete quick-tier held-out ClickBench run passes all 43
queries: all 86 baseline/candidate cells independently match complete DuckDB
values and schemas, and all 774 target snapshots load and verify their hashes.
Only aggregate timing is inspected: candidate/baseline elapsed geometric mean
1.0003x, effectively unchanged. This is validation at the quick tier, not a
full-tier performance headline. The full report retains Polars' 35 wrong-answer
and eight successful cells.

The qualified baseline suite binary from the initial build-only phase is
reused after verifying its original build command, complete source hashes,
compiler/SDK, lockfile, native library, input and benchmark identities.
The candidate suite binary is freshly built against `c30afb58`. The complete
43-query workload, all variants, four threads, three rounds/repetitions,
tracing, fresh references and independent evaluator remain unchanged.
No held-out query timings selected the vectorization change.

Final-candidate evidence is outside the tracked tree under
`build/1003-validity-validation-a54c04b4/vector-iteration/` on Linux and
`/Users/randyzwitch/benchdata/1003-validity/a54c04b4/vector-iteration/` on M1.
The package includes `frozen-sources.json`, `allocation-admission.json`,
`focused-contract-admission.json`, `native-m1-admission.json`,
`m1-timing-admission.json`, `contiguous-controls-receipt.json` and
`m1-heldout-admission.json` and `keep-numeric/admission.json`, plus frozen
producers/oracles/controllers.
The M1 raw snapshots and full receipts remain at their original paths;
`heldout/baseline-reuse.json` records the baseline qualification. Earlier
`e8e08e81`/`916a0e4f` observations are preserved in the parent directory.
The remaining #1003 consumers stay open.

## Initial numeric cast path

Unmasked numeric conversions that cannot fail read each input chunk into one
owned output allocation and append its sliced validity bits. Null payloads
are zeroed as in the contiguous path. Input merge caches remain untouched.
The contiguous path, checked casts, masks, categorical dictionaries, temporal
units and decimal rules keep their existing converters. Other consumers
remain under measurement; this initial change does not complete #1003.

[Metadata-only casts](scalar-cast-chunks.md) also retain source arrays for
accepted identities and String-to-Binary retags, including whole expressions.

[Checked and masked integer casts](scalar-integer-checked-cast-chunks.md)
also consume cold host chunks for plain integer sources and numeric targets.
Their exact bounds, global error rows, mask-length rules and Float32 rounding
modes are preserved. Published caches and contiguous inputs retain the old
converters. Float-source, Bool, parsing and logical conversions remain in the
full #1003 inventory.

## Initial cast measurements

The frozen paired comparison at implementation revision `b7c8d971` completed
576 case/layer pairs (1,152 workers), with complete ordered NumPy/schema
checks and no compiler overlap. The one-worker runs are busy-host diagnostics;
four-worker runs are unflagged. Results are retained outside the repository
at `/tmp/1003-comparison-b7c8d971/receipt.json`.

For the unsliced 1M-row Int64-to-Float32 direct cast at four workers, cold
64/1024-chunk medians changed from 1.313/1.605 ms to 0.369/0.503 ms.
The baseline retained an 8 MB merged input; the candidate retained none.
Whole-expression cold medians changed from 0.529/1.089 ms to 0.453/0.833 ms.
Already-contiguous inputs take the existing path. These are mechanism
measurements, not an external-suite performance claim; unchanged controls
also show noise, especially for tiny operations.

A retained tradeoff is the warm direct 1K-row/1024-chunk case: the baseline
reuses its merged input at 2.26 us, while the candidate walks descriptors
at 8.13 us. Its cold comparison is 454.74 us versus 8.43 us. No blanket
non-regression claim or threshold chosen to hide this case is made. Caller
reuse and whether a contiguous input already exists need further study.
Process peak RSS includes fixtures, outputs and untimed exports; retained
merge-cache bytes do not measure every temporary allocation. Scoped owned-allocation diagnostics are recorded below. The remaining
text/nested/temporal consumers and external-suite validation remain pending.
The initial cast change passes all 75 focused CPU contracts on the live M1
and complete native Metal-provider integration checks at `7b720192`.

## Scoped owned-allocation diagnostics

An unchanged diagnostic program now compares the baseline and implementation
using 128 paired cases (256 workers, three observed calls per worker),
checking complete ordered NumPy values, nulls and Arrow schemas. It covers
empty, 1K and 1M inputs; contiguous, 64/1024 chunks; nonzero slices; cold and
warm inputs; direct kernels and whole expressions; null-free and nullable
values; and unchanged unary controls. Every call has zero observer tracking
errors. Evidence: `/tmp/1003-allocation-accounting/receipt.json`.

The observer delegates to the installed Mojo runtime's AlignedAlloc/Free
and standard libc allocation entry points, counting requested bytes and
tracking live and peak newly owned bytes around the call. Exact-size controls
exercise the actual runtime allocator, libc, aligned allocation, reallocation,
fixture exclusion and concurrent threads. A libc-only prototype missed Mojo's
private allocator; its zero observations were rejected. Fixture construction,
warm-up, existing warm caches and output export are outside the observed call.
All instrumented runtimes are excluded from performance claims.

For nullable, unsliced 1M-row inputs in 1024 chunks, median bytes per call are:

| Layer / cache | Requested, baseline / candidate | Peak newly owned, baseline / candidate | Still owned on return, baseline / candidate |
| --- | --- | --- | --- |
| direct / cold | 12,726,616 / 4,500,038 | 12,250,672 / 4,374,784 | 12,250,672 / 4,250,304 |
| whole / cold | 19,001,284 / 9,932,916 | 4,490,700 / 4,302,216 | 4,132,888 / 4,132,888 |
| direct / warm | 4,125,448 / 4,500,038 | 4,125,448 / 4,374,784 | 4,125,448 / 4,250,304 |
| whole / warm | 19,000,164 / 9,931,796 | 4,478,236 / 4,298,532 | 4,132,888 / 4,132,888 |

The warm direct baseline already owns its 8 MB input merge before the call;
that retained cache is additional to the newly owned bytes shown here. The
candidate avoids that cache but grows a validity buffer while walking chunks,
which explains its higher warm allocation. Contiguous direct controls have
identical allocation counts and byte totals. Unchanged unary controls retain
their existing path.

These are requested logical buffer/metadata bytes, including temporary and
output allocations, rather than allocator arena/reservation or process RSS.
Nested allocator bookkeeping is excluded to avoid double counting. Direct
mmap and hidden allocator-internal calls are outside this diagnostic; the
observed runtime/library hashes are retained. This does not finish the wider
consumer inventory or external-suite/held-out validation.

Literal string predicate chunk processing, retained cache/regex adapters and
measured allocation costs are documented in [scalar-literal-predicates.md](scalar-literal-predicates.md).

The remaining two-input math adapter is documented in
[scalar-arctan2-chunks.md](scalar-arctan2-chunks.md), including identity-cast
handling, retained strict conversions and cold/published-cache controls.
