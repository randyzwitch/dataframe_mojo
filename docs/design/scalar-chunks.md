# Scalar kernel chunk consumption

Inventory and measurements for #1003 are in progress. Baseline revision:
`2d0dce12a41a07c265025a890a10ae121483dc1b`.

The initial inventory distinguishes explicit APIs requesting an owned
contiguous column from internal scalar consumers. The former are retained;
the latter are candidates for aligned chunk-window evaluation.

| Consumer | Current input copy | Constraints to preserve |
| --- | --- | --- |
| `kernels/cast.cast_series` | Rechunks any chunked input | Global error-row offset, observed-row masks, strict/non-strict behavior, logical dtype |
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
merge-cache bytes do not measure every temporary allocation. Full owned
allocation accounting and the remaining text/nested/temporal consumers are
still pending, as are external-suite and live M1 validation of this change.
