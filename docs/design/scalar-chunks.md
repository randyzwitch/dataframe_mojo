# Scalar kernel chunk consumption

The initial numeric-cast change shipped in #1072. The remaining consumers
are still being measured for #1003 against current main
`6d3f987aae999e9e10072f926cc6d9817510fbb8`, which already includes that cast
optimization. Historical cast measurements below retain their original
baseline `2d0dce12a41a07c265025a890a10ae121483dc1b`; those gains are not new
improvements in the remaining-consumer work.

The initial inventory distinguishes explicit APIs requesting an owned
contiguous column from internal scalar consumers. The former are retained;
the latter are candidates for aligned chunk-window evaluation.

| Consumer | Input copy at the remaining-consumer baseline | Constraints to preserve |
| --- | --- | --- |
| `kernels/cast.cast_series` | Unmasked infallible numeric casts already consume chunks; remaining checked/masked/logical paths merge | Global error-row offset, observed-row masks, strict/non-strict behavior, logical dtype |
| `kernels/binary.math_unary` and `unary` | Rechunk before scalar dispatch | NaN/infinity/signed zero, integer errors, null tests and masks |
| `kernels/binary.binary` and `choose` | Rechunk each operand | Different chunk boundaries, scalar broadcasting, empty inputs, conditional observation and dtype promotion |
| `kernels/horizontal` | Rechunks each input | Different boundaries, row null rules, mixed widths and promotions |
| `kernels/temporal.dt_op` and `cast_temporal` | Rechunk before scalar dispatch | Logical temporal units/timezones, calendar behavior, masks and conversion errors |
| `kernels/list.nested_op` | Rechunks list/string input | Child offsets, sliced children, empty/null rows, struct field validity |
| `execution` scalar/text paths | Several preparation and result rechunks | Separate actual required contiguous adapters from avoidable input/result copies |

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

## Initial numeric cast path

Unmasked numeric conversions that cannot fail read each input chunk into one
owned output allocation and append its sliced validity bits. Null payloads
are zeroed as in the contiguous path. Input merge caches remain untouched.
The contiguous path, checked casts, masks, categorical dictionaries, temporal
units and decimal rules keep their existing converters. Other consumers
remain under measurement; this initial change does not complete #1003.

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
