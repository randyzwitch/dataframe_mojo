# Literal string predicates over source chunks

This subset of #1003 covers `starts_with`, `ends_with` and
`contains(literal=True)`. Production changes start from main `a54c04b4`.
Validation applies pending #1089 and #1090 to both sides: baseline
`fb3d1087`, candidate engine/tests `12914b49`. Production engine/tests
are frozen at `bebab599`; the implementation and test patches are
byte-identical between the independent and combined checkouts.

## Mechanism and retained adapters

The chunked literal route prepares one `_Bytes` descriptor per physical
chunk, sharing its existing offsets/bytes or StringView buffers. Jobs retain
those owners and a prepared literal until they join. Each job owns a globally
byte-aligned output range. A retained descriptor cursor and cached chunk end
serve bounded spans within each byte, avoiding per-row list lookups.
Chunk boundaries may fall anywhere within a byte;
local row indices include the physical slice offset exactly once. Invalid
rows are skipped before reading payload bytes and leave zero result bits.
Validity is appended in source order into a single output bitmap.

Contiguous inputs keep the existing predicate implementation. If the source
already has a fully published immutable merge cache, the adapter reuses it.
Empty or building caches do not initiate a merge or wait for another caller.
The source arrays remain immutable throughout evaluation.

Regex contains, LIKE, membership, transforms and other text routes retain
their existing adapters. Optional compiled regexes remain ignored by literal
contains. Public binding, Binary rejection, zero-row validation and strict or
relaxed invalid-regex behavior are unchanged. These retained text routes
still require their own measured decisions; this does not complete #1003.

## Frozen mechanism catalog

An external Mojo producer and independent Python/PyArrow oracle verify full
ordered Bool outputs, nulls, field names and schemas. Small semantic cases
cover all three predicates, empty/ASCII/UTF-8/NUL/metacharacter/long/missing
literals, offset/StringView/mixed/coded inputs, dense/mixed/all-null validity,
bitmap offsets 0–7, clipped uneven chunks and canonical empty chunks.
There are 576 functional cases per revision on each Linux and M1 host.
Direct null payload bits are checked separately after the observed call.

The 150 performance cells cover four representative operation/pattern pairs,
three physical layouts, 0/1K/100K/1M rows, contiguous/64/1024 requested chunks,
cold and warm caches, and direct/whole evaluation. Six coded-string controls
are separate: public dictionary evaluation can avoid the row kernel.
Actual physical counts and layout counts are retained after canonicalization.
An empty or single-chunk mixed fixture does not establish mixed-layout coverage.

Cold calls rebuild all source owners outside measurement. Warm calls populate
the source merge cache before measurement. Before-call, after-call and
post-export cache states are recorded separately. Whole expression preparation
can create local caches, so an absent original cache alone does not prove
absence of copying. The calibrated allocator observes six counters only around
evaluation; construction, warm-up and complete Parquet exports are excluded.
Allocation instrumentation is never used for elapsed-time claims.

## Allocation results

All 150 pairs verify complete outputs with zero tracking errors. Requested
bytes, allocation calls and peak newly owned bytes decrease in 87 cells and
are equal in 63; none increase. Existing contiguous and ready-cache direct
controls retain their allocation behavior.

For nullable 100K-row inputs in 64 chunks, sliced from offset 13, starts-with:

| Storage / layer | Requested bytes, baseline → candidate | Peak newly owned bytes, baseline → candidate |
| --- | --- | --- |
| Offsets / direct | 2,512,261 → 34,318 | 2,447,315 → 33,894 |
| StringView / direct | 1,667,226 → 34,318 | 1,638,812 → 33,894 |
| Mixed / direct | 2,886,432 → 34,318 | 2,450,246 → 33,894 |
| Offsets / whole | 2,844,907 → 168,793 | 251,773 → 41,488 |
| StringView / whole | 1,806,589 → 168,793 | 167,142 → 41,488 |
| Mixed / whole | 3,028,246 → 168,793 | 240,655 → 41,488 |

These are requested buffer/metadata allocations and peak newly owned memory,
not process RSS or allocator reservation. Existing input and warm-cache memory
is outside the observed allocation interval. The result still owns fresh Bool
values and validity; this change removes input materialization.

## Elapsed time and contiguous overhead

The complete 150-cell timing grid passes 900 full-answer checks. A second
measurement runs the same 150 cells in three rounds with adjacent,
alternating baseline/candidate processes: 900 fresh workers and complete
answers. No cells or samples are discarded. Both use the unchanged calibrated
activity guard before and during each worker; no recognized overlap or monitor
errors are reported. Allocation instrumentation is disabled. M1 clock samples
have a 1 microsecond quantum; small cached-call differences are not speed claims.

Adjacent starts-with medians, in microseconds:

| Shape / storage / layer | Baseline | Candidate |
| --- | ---: | ---: |
| Nullable 100K / 64 offset chunks / direct | 615 | 338 |
| Nullable 100K / 64 offset chunks / whole | 608 | 445 |
| Nullable 100K / 64 StringView chunks / direct | 833 | 361 |
| Nullable 100K / 64 StringView chunks / whole | 857 | 471 |
| Nullable 100K / 64 mixed chunks / direct | 1,192 | 350 |
| Nullable 100K / 64 mixed chunks / whole | 1,178 | 460 |
| Dense 100K / contiguous offsets / direct | 234 | 263 |
| Dense 100K / contiguous offsets / whole | 270 | 301 |
| Dense 100K / contiguous StringView / direct | 281 | 299 |
| Dense 100K / contiguous StringView / whole | 326 | 335 |

Contiguous inputs retain the existing algorithm and identical allocation
counts, but these timing controls show a measurable overhead, approximately
12% for offset starts-with. The source-level fast path alone does not establish
zero elapsed overhead; its cause is not isolated by these measurements.
The fragmented gains and lower owned memory come with this tradeoff. All other
operation/pattern cells and ranges remain in the receipt.

For cached 1K-row inputs with 1024 requested chunks (1000 after slicing),
direct starts-with medians are 6 microseconds on both sides for all three
layouts, with overlapping ranges. Whole-expression medians are 941→654
microseconds for offsets, 956→750 for views, and 1,217→699 for mixed storage.
Coded-string whole evaluation is 384→384 microseconds with overlapping ranges;
it can use dictionary evaluation and is not counted as a row-kernel gain.

## Correctness and ownership validation

All 89 selected CPU contracts pass on both Linux and the live M1, with no
failures, expected failures, capability skips or process cases. The runner's
634 source fingerprints on each host match the final committed candidate;
281 engine fingerprints also match the held-out source export. The five new
contracts cover all packed offsets, mixed/sliced storage, globally disjoint
four-worker ranges, direct null payload bits, source lifetime, ready caches,
compiled-regex dispatch, empty/all-null inputs and public Binary boundaries.

Each revision also passes all 576 external functional cases on each host.
Every one of the 150 allocation pairs and 150 timing pairs verifies complete
results independently; all raw cells and sample ranges are retained.

## Native composition and held-out workload

The unchanged nine-row composition probe checks CPU and installed native
Metal-provider collection on contiguous and uneven sliced offset/StringView
inputs, with batch size 3. All four cases per revision verify complete literal
values, nulls, dtypes, names and order. It composes the predicates with pending
validity/Boolean behavior, guarded identity casts and String-to-Binary output.
The admitted fb3 baseline binary is reused by hash; the candidate is built and
run freshly at 12914b49. The M1 reports one available Metal device. This is
public scalar integration evidence, not a GPU performance claim.

The unchanged complete quick-tier 43-query ClickBench comparison uses four
threads, three rounds and three repetitions with fresh DuckDB references.
All 86 baseline/candidate answer cells are independently correct; all 774
retained target snapshots are hash-verified. Aggregate candidate/baseline
elapsed geometric mean is **1.0001×**, effectively unchanged. No individual
query timing is inspected or used to choose an implementation. The qualified
fb3 baseline suite binary is reused; candidate compilation and paired workload
measurements are fresh. Non-target Polars discrepancies remain visible
(35 wrong answers, 8 correct); DuckDB and both target revisions have 43 correct.

## Reproduction evidence

Linux artifacts are retained under
`build/1003-literal-predicate-validation-fb3d1087/spans`; M1 artifacts under
`/Users/randyzwitch/benchdata/1003-literal-predicates/fb3d1087/spans`.
The parent directories retain the initial variant and admitted baseline.
The unchanged baseline binaries, source/runtime pins and complete answers are
reused explicitly, while candidate outputs and measurements are fresh.
Source freezes, build/runtime sidecars, full Parquet answers, worker logs,
cache states and admissions bind results to the actual engine revisions,
compiler, native library, lockfile and observer. The frozen catalog's
`actual_baseline_revision` denotes the comparison baseline on both sides;
candidate build sidecars identify its compiled engine explicitly.
