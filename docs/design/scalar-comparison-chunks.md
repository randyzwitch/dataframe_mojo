# Ordinary comparisons over owned chunks

Ordinary Boolean NE/LT/LE/GT/GE and String EQ/NE/LT/LE/GT/GE previously
materialized chunked operands before reaching their contiguous kernels.
These paths now read owned physical windows. Boolean EQ belongs to the
preceding membership change and remains a control in this investigation.

Boolean comparisons combine payload and validity bitmap windows, including
unaligned offsets and operands with different physical boundaries. String
comparisons retain owned offset/StringView descriptors until all worker jobs
join. Jobs write disjoint global output bitmap bytes. Valid scalar String
operands use the existing literal comparison semantics and direction reversal;
literal setup occurs once per invocation. Existing StringView head checks for
literal equality and inequality remain in use.

Each operand independently reuses an already published contiguous cache. A
comparison neither initiates nor waits for a cache merely to traverse chunks.
Contiguous operands keep their existing dispatch. Logical and physical storage
guards check every cold physical chunk, or the actual published cache, and
preserve the fallback for private constructors whose dtype tag does not
describe their storage. Binary, decimal and other logical payload paths
are outside this change.

## Semantics

The output is Boolean with conjunction validity; invalid result value bits
remain zero. Whole-operand scalar broadcasting, empty inputs, length errors,
column names and ownership are preserved. Direct ordinary Boolean/String
comparisons retain their legacy behavior of ignoring observation masks,
including short masks. Private numeric storage retains its numeric mask
dispatch. Public binding and dictionary execution are unchanged; actual coded
String storage can use the String adapter after those decisions.

Valid String scalar shortcuts precede the general length check, matching the
contiguous implementation. Other unequal lengths must be equal or have a
scalar operand; an empty operand does not suppress an incompatible-length
error. Comparison uses exact UTF-8 bytes without normalization.

## Reproducible validation

The independent production branch starts at main `62706a5f`. Incremental
baseline `e71609de` includes the earlier arctan2 and membership changes on
both measurement sides. Production code `40218ad3` and incremental candidate
`c8b2a6e4` have equivalent comparison changes. A thirteen-check source proof
records the necessary integration into the existing membership dispatch;
it does not claim that their complete kernel files are identical.

The frozen catalog has 852 functional cases: 780 complete answers and 72
exact incompatible-length controls. It covers every operator, scalar direction,
null/empty inputs, masks, unequal boundaries, sliced bitmap offsets, dense
inputs, offset/view/mixed strings, actual coded storage, public categorical
controls and published caches. Pure expected functions independently specify
values, validity, operand lengths and raw payload scope. Whole dictionary
results constrain invalid payloads only where the existing API specifies them.

The separate performance catalog has 218 cells, including contiguous inputs,
fragmented 100K/1M inputs, 1K singleton chunks, both scalar directions, null
scalars and independently prepublished caches. Every measured repetition
constructs a fresh fixture and prepares only its designated active operands
outside the observed call. This prevents a warm-up from populating a side
that is meant to stay cold.

Linux allocation observation and M1 elapsed measurements are separate runs.
The source/compiler/runtime/native-library and producer hashes are pinned;
receipts retain every answer, raw output, sample and expected rejection.
The M1 activity guard excludes known overlapping work and retains its stated
scope. It is not proof of complete host isolation.

The completed baseline demonstrates the mechanism: nullable 100K-row operands
with 64 and 65 chunks request 346,902 bytes for Boolean NE/LT and populate both
input caches. Corresponding fragmented String NE/LT cells request 5,979,604
and 4,319,075 bytes. These are baseline costs, not a speedup claim.

The candidate passes all 852 functional cases on Linux and M1 and all 218
Linux allocation cells. Requested and peak newly owned bytes decrease in 126
pairs and are equal in 92; neither increases. The representative nullable
100K-row Boolean NE/LT vector calls request 25,400 bytes. The corresponding
String NE/LT calls request 91,824 bytes. These cold calls leave both input
caches unpopulated. All eight focused production contracts pass on each host. Four isolated
Linux subprocess controls also preserve the exact legacy Variant aborts for
private heterogeneous physical chunks, including null-scalar shortcuts and
length-error precedence.

The final adjacent M1 timing pair retains all 654 verified answers per
revision. Representative nullable 100K-row NE medians, with 64 and 65 input
chunks, are:

| Input | Direct baseline → candidate | Whole-expression baseline → candidate |
| --- | ---: | ---: |
| Boolean | 528 → 34 µs | 747 → 162 µs |
| Mixed offset/view String | 2,783 → 1,231 µs | 2,933 → 1,466 µs |

The new Boolean setup has an explicit no-inline boundary. The preceding full
grid exposed 34–39% overhead for four flat whole-expression String ordering
cells. M1 disassembly showed extra unconditional variant packing in their
shared binary wrapper. In the inspected LT width8 specialization, outlining
restores the entry to the baseline's 459 instructions, versus 768 in that
candidate; the existing contiguous String loop stays unchanged. The final
full pair measures these cells as follows:

| Flat 100K-row whole-expression comparison | Baseline | Candidate |
| --- | ---: | ---: |
| LT | 1,673 µs | 1,685 µs |
| LE | 1,685 µs | 1,676 µs |
| GT | 1,733 µs | 1,711 µs |
| GE | 1,676 µs | 1,671 µs |

All four final ranges overlap. The immutable baseline stayed stable between
the two grids, so this result does not come from its timing drifting to the
slower candidate. The static instruction difference alone does not establish
the cost of those instructions.

There are still tradeoffs: 41 of 218 candidate medians are slower. Seven have
separated three-sample ranges. The largest is flat Boolean GT through a whole
expression: 668 µs (663–674) → 745 µs (694–790), an 11.5% increase. The other
six separated increases range from 0.3% to 7.4%, including existing controls.
Every cell and sample remains in the report; three samples are not a
significance test, and no universal speedup is claimed.

Native CPU/installed Metal composition passes all four cases per revision,
including an actual device query and downstream Float32 projections. This
checks integration and complete answers, without a GPU-launch or GPU timing
claim.

Both revisions independently match fresh DuckDB results for all 43 queries
in the unchanged quick-tier external suite. All 86 target verdicts, 258 target
worker records and 774 target snapshots are verified, together with all 43
trace sets. The candidate/baseline geometric mean elapsed ratio is 0.99344,
effectively unchanged; individual query timings were not inspected. Existing
Polars discrepancies (35 wrong, eight correct) remain retained, without skips
or scope changes. This check does not establish new query-route coverage.

The frozen protocol is identified by SHA256
`8837b29c528647d7277360a928fb93dab78c3e12d434241b40d08f072f8cfc62`.
Evidence is retained under
`build/1003-comparison-validation-e71609de/guard-refinement/dispatch-outline`
on Linux and
`/Users/randyzwitch/benchdata/1003-comparison/e71609de/guard-refinement/dispatch-outline`
on M1. The complete profile analysis, baseline drift check, source-equivalence
proof and native/external admissions retain their actual source and binary
identities. Earlier candidates, failures and the complete preceding grid
remain in the parent directories.

## Remaining scope

This subset does not complete #1003. The broader inventory still includes
general membership fallbacks, logical/temporal/private payload adapters,
checked and masked casts, decimal operations, nested/calendar consumers,
categorical decoding/recoding and execution/storage preparation. Each requires
its own semantic and allocation evidence; a census alone does not establish
that a materialization is necessary or beneficial.
