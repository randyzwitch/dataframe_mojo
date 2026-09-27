# Inner-join count fusion (removed)

> **Removed in #304.** This shortcut fired only for an in-memory inner join
> followed directly by `len()` or `count()` of a key, which is the upstream
> `duplicate_strings` benchmark query. It answered by summing key
> multiplicities, so the join was never executed and the numbers below
> measure skipped work. General projection pushdown replaces it: a lazy join
> now receives only its keys and the columns the plan above reads, on every
> scan kind, so this query joins key columns alone. The record below is kept
> for history.

A SELECT of simple column LEN expressions or COUNT of inner-join keys can
sum match multiplicities without constructing joined row pairs or payload
columns. This implementation applies to direct in-memory scan inputs and
non-nested keys. Other shapes, grouped aggregates, nullable payload COUNT,
file scans and non-inner joins retain ordinary execution.

The metadata execution path validates schema, key types, suffix collisions,
referenced columns and output names before executing the shortcut. Hash
equality and null handling reuse the prepared index. The smaller input is
prepared, each occupied slot gets one multiplicity, and parallel probes sum
those counts with checked Int64 arithmetic. Validated Int64 progressions
retain direct lookup, including incomplete duplicate runs and signed extremes.
No joined output is allocated. Existing source frames remain alive.

## Validation

Hash and streaming modules pass on both hosts (26 tests); the final dedicated
count module adds five passing tests per host. The tests independently count
tens of millions of duplicate matches without materializing them, and cover
empty/all-null inputs, compound keys, NaNs, signed zero, booleans, partial
progression runs, signed extremes, overflow, metadata errors, nullable payload
counts and non-inner fallbacks. The benchmark checks every complete answer.

## Paired full-query measurements

Baseline: #299 (`39e462d`); #300 changes integer range filtering and does not
affect these string queries. Each host has 120 timed results at 1M/10M right
rows, base/shuffled layouts, three alternating process rounds and five
repetitions after a warmup per process. Both use four workers, unpinned CPUs,
Mojo 1.2.0.dev2026092105 (e9569894), and matching inputs/source hashes.
Loading is outside timing. Times are medians of per-round medians.

| Host | Right rows | Layout | Materialized ms | Count ms | Speedup |
|---|---:|---|---:|---:|---:|
| Linux | 1,000,000 | base | 146.118 | 94.368 | 1.55x |
| Linux | 1,000,000 | shuffled | 157.566 | 124.158 | 1.27x |
| Linux | 10,000,000 | base | 1146.647 | 628.520 | 1.82x |
| Linux | 10,000,000 | shuffled | 1134.394 | 904.896 | 1.25x |
| M1 | 1,000,000 | base | 68.072 | 35.080 | 1.94x |
| M1 | 1,000,000 | shuffled | 88.931 | 38.860 | 2.29x |
| M1 | 10,000,000 | base | 833.296 | 318.365 | 2.62x |
| M1 | 10,000,000 | shuffled | 908.702 | 326.738 | 2.78x |

The baseline already streams output batches; “materialized” here means
constructing each batch’s matches and columns, not retaining the full joined
result. Whole-process peak RSS (including loading/warmup/runtime) is preserved
in the raw data and should not be described as query-only allocation.

[Linux raw results](benchmarks/join-count-linux.json) and
[M1 raw results](benchmarks/join-count-mac.json) include all samples and
binary/source/input hashes. Reproduce with `scripts/bench_upstream_revision.py`
using the #299 baseline runner, this candidate, the PR297 CSV directory,
`--cases duplicate_strings` and an output JSON path.

This removes output materialization; it does not remove all hashing, key
access or equality work. Those remaining probe costs are separate follow-ups.
