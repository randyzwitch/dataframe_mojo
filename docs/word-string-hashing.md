# Shared string hashing in eight-byte blocks

Strings longer than eight bytes previously used a serial FNV-1a recurrence
for every byte. The shared partition hash now mixes eight bytes per iteration,
adapting [DuckDB 1.5.5 HashBytes](https://github.com/duckdb/duckdb/blob/d8cdaa33fda8df955cc76ef58a280f68f4cd43fa/src/common/types/hash.cpp).
The existing short-key encoding and column finalizer are retained. There is no
query-plan, cardinality, or benchmark-name dispatch. Join and grouping callers
use the same primitive. This does not change the planner or remove existing
query specializations; those require separate corrective work.

All wide loads are inside the supplied span, including an overlapping final
load for partial blocks. Equality still compares original keys after hashing.
The block layout follows the project's existing little-endian short-key layout
on the tested x86-64 and Apple Silicon targets. DuckDB's license and attribution
are recorded in THIRD_PARTY_NOTICES.md.

## Paired measurements

Baseline engine: merged commit `39c3f55`. Candidate engine differs only in the
shared string hash. Four workers, no concurrent compilation. Count queries use
three alternating-order process rounds, five checked repetitions per round;
numbers are median round medians in milliseconds. Input loading is excluded.
Raw files include inputs, binaries, source fingerprints, and individual samples.

| Host | Rows | Layout | Baseline ms | Candidate ms | Speedup |
|---|---:|---|---:|---:|---:|
| Linux Threadripper 3970X | 1M | base | 76.544 | 68.030 | 1.13x |
| Linux Threadripper 3970X | 1M | shuffled | 80.633 | 70.462 | 1.14x |
| Linux Threadripper 3970X | 10M | base | 430.766 | 366.464 | 1.18x |
| Linux Threadripper 3970X | 10M | shuffled | 457.530 | 391.875 | 1.17x |
| Apple M1 | 1M | base | 25.313 | 19.601 | 1.29x |
| Apple M1 | 1M | shuffled | 26.930 | 21.684 | 1.24x |
| Apple M1 | 10M | base | 225.360 | 181.503 | 1.24x |
| Apple M1 | 10M | shuffled | 224.677 | 180.239 | 1.25x |

These are the existing duplicate-string **count queries**, which use the existing
join-count specialization in both revisions. They are not materialized-join
throughput and do not establish parity with DuckDB or Polars.

Raw results: [Linux](benchmarks/word-hashing-linux.json),
[Mac](benchmarks/word-hashing-mac.json).

## Full-result joins and controls

The initial 1M-row matrix covers full-result string and compound joins and
integer semi/anti controls. Its base/shuffled decimal strings are shorter than
nine bytes, so those cases do not exercise the changed code. The wide string
case does: 132.813 to 115.092 ms (1.15x). Unchanged controls also vary, including
wide anti join 60.108 to 65.006 ms; do not attribute all timing differences to
this hash change. Full raw matrix, including regressions:
[matrix](benchmarks/word-hashing-matrix-linux.json).

Additional long-string matrix cases construct shared-prefix keys before timing
and materialize all join payload columns. Both baseline and candidate are built
with the identical updated benchmark source. Five alternating-order rounds,
one warmed timed query per round; medians in milliseconds. The case names
indicate approximate key lengths in the base layout (a fixed prefix plus a
variable-width decimal ID), not fixed-width types.

| Case | Layout | Baseline ms | Candidate ms | Speedup |
|---|---|---:|---:|---:|
| inner_string_16 | base | 124.379 | 118.839 | 1.05x |
| inner_string_16 | shuffled | 125.390 | 118.924 | 1.05x |
| inner_string_16 | wide | 151.059 | 139.585 | 1.08x |
| inner_string_64 | base | 234.008 | 199.136 | 1.18x |
| inner_string_64 | shuffled | 233.862 | 199.574 | 1.17x |
| inner_string_64 | wide | 244.178 | 210.023 | 1.16x |

[All long-key samples](benchmarks/word-hashing-long-matrix-linux.json).
These full-result results support a shared hashing improvement, not a claim
that the whole engine is competitive with DuckDB. Remaining work includes
join memory traffic, general aggregation, and planner column pruning.

The matrix checks timed output heights and warmed output checksums against the
other revision. It does not compare every timed cell. The separate correctness
tests provide exact assertions for join ordering, nulls, duplicate matches,
compound keys, storage views, slices, and prepared groups.

## Validation

Five targeted modules pass on both Linux and Apple M1 (41 tests each):
`test_partitioned_grouping`, `test_join_hash`, `test_join_count`,
`test_typed_join_probes`, and `test_prepared_join_groups`.
Hash tests cover lengths across block boundaries, all eight pointer alignments,
spans ending at the allocation boundary, embedded NULs, Unicode, and
contiguous/view/sliced equivalence. Formatting and diff checks pass.
