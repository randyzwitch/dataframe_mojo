# Inner-join count fusion (removed)

> **Retired benchmark.** The scripts this record uses to reproduce its measurements were retired in favor of the external suites ([benchmarks.md](benchmarks.md#retired-benchmarks)); check out revision `b9db9c6` to rerun them.

> **Removed in #304.** This shortcut fired only for an in-memory inner join
> followed directly by `len()` or `count()` of a key, which is the upstream
> `duplicate_strings` benchmark query. It answered by summing key
> multiplicities, so the join was never executed and the numbers below
> measure skipped work. General projection pushdown replaces it: a lazy join
> now receives only its keys and the columns the plan above reads, on every
> scan kind, so this query joins key columns alone. The record below is kept
> for history.

## After removal

Measured 2026-09-27 on the Threadripper 3970X at 32 workers with Mojo 1.2.0.dev2026092105 (e9569894), using `-O3 -g1` builds of each revision. `scripts/bench_join_revision.py` alternates the two binaries every round (five rounds; three for the control cases). Each process warms once and times one query, and the driver checks that both revisions return the same height and checksum. Values are medians in milliseconds; speedup above 1 means the change is faster. Baseline is the #308 branch. The upstream queries use
`scripts/bench_upstream_revision.py` with three rounds of five repetitions per
process. Loading is untimed.

| Rows | Query | Layout | before ms | after ms | Speedup |
|---:|---|---|---:|---:|---:|
| 1,000,000 | duplicate_strings | base | 55.5 | 98.8 | 0.56x |
| 1,000,000 | duplicate_strings | shuffled | 60.9 | 107.3 | 0.57x |
| 10,000,000 | duplicate_strings | base | 173.1 | 753.0 | 0.23x |
| 10,000,000 | duplicate_strings | shuffled | 177.2 | 755.3 | 0.23x |
| 1,000,000 | highcardinality | base | 6.0 | 5.9 | 1.02x |
| 1,000,000 | highcardinality | shuffled | 3.5 | 3.5 | 1.02x |
| 10,000,000 | highcardinality | base | 44.7 | 44.2 | 1.01x |
| 10,000,000 | highcardinality | shuffled | 10.7 | 10.6 | 1.01x |

`duplicate_strings` now executes its join, producing 4 matches per right row,
so it is 1.8x slower at 1M right rows and 4.3x slower at 10M. The earlier
numbers measured skipped work. `highcardinality` is unchanged. The
materializing narrow join reads only its keys and two payload columns:

| Rows | Case | Layout | before ms | after ms | Speedup |
|---:|---|---|---:|---:|---:|
| 1,000,000 | lazy_narrow | base | 12.40 | 11.44 | 1.08x |
| 1,000,000 | lazy_narrow | shuffled | 25.37 | 24.45 | 1.04x |
| 1,000,000 | lazy_narrow | wide | 25.79 | 24.75 | 1.04x |
| 10,000,000 | lazy_narrow | base | 388.48 | 380.18 | 1.02x |
| 10,000,000 | lazy_narrow | shuffled | 438.11 | 451.11 | 0.97x |
| 10,000,000 | lazy_narrow | wide | 434.75 | 457.08 | 0.95x |

Those are within noise, because `lazy_narrow`'s inputs have few unused
columns. Wider inputs gain more.

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
