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

The evidence remains outside the repository at
`/tmp/1003-current-main/baseline-admission.json`. Timing and offsets/view
layout comparisons are still in progress. No new elapsed-time or general
engine speedup is claimed from these allocation observations. A zero merge
cache on a whole-expression source does not prove zero temporary copying.

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
