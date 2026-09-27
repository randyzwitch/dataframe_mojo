# Join performance follow-ups to PR #297

Baseline: `fbef94a` (merged PR #297), with production engine measurements
recorded in [the upstream baseline](upstream-join-baseline.md).

The implementation work covers these five opportunities, in dependency order:

1. **Prepared streaming joins.** Separate immutable build-side state from
   per-batch probing and share it across streamed inner/left/semi/anti jobs.
   Preserve key equality, null handling, duplicate order, output schema,
   early termination, and cleanup. Retain supported nested-key behavior.
2. **Cardinality-based build selection.** Build on the smaller input when
   profitable, independently of logical output order. Restore documented
   left-major/right-input duplicate order when the physical roles reverse.
3. **Safe derived key filters.** Use build-side key bounds to reject impossible
   probes. Nulls, signed extremes, and outer/anti unmatched rows must retain
   their semantics. Filters must be derived from actual data, not workload
   names or assumed distributions.
4. **Join-count fusion.** Count matches without constructing joined frames
   when the expression semantics permit it; handle duplicates, nulls, empty
   inputs, and supported count/length expressions explicitly. Other plans
   retain ordinary execution.
5. **Hash-probe efficiency.** Use the measured semi-join probe hotspot to
   reduce per-row dispatch and memory access. Validate collisions and all
   affected dtypes; do not remove bounds checks without proving safety.

For each opportunity, completion requires semantic regressions, paired
before/after measurements on identical inputs/compiler, and evidence that
its intended work was removed. Check ordered, shuffled, and wide keys;
small/large inputs; strings, compound keys, nulls, duplicates, and skew where
applicable. Preserve raw measurements and report regressions as well as wins.
Run profiles separately from timing and do not overlap compilation with
comparison runs. Compare both Linux and Apple Silicon before finalizing
performance claims.

## Prepared streaming joins

The first implementation shares immutable right-side hash buckets across
streamed inner/left/semi/anti jobs. Each batch computes only its probe hashes
and matches. Ascending Int64 progressions retain a compact prepared
base/stride/repeat representation, validated once rather than rescanned per
batch. Exact identity matches share left columns. Nested keys retain the
existing eager fallback; cross joins retain their existing execution path.
The eager public API and direct-hash algorithm selection are unchanged.
Heavily duplicated hash buckets retain contiguous duplicate-row groups and
a table sized to distinct keys. This preserves the locality of the previous
dictionary/CSR streaming path while allowing preparation to be reused.
Unique and lightly duplicated buckets retain their linked representation.

Seven affected Linux test modules passed (61 tests). After the final
profile-guided probe changes, the hash and streaming modules passed again
on Linux and M1 (25 tests on each). New assertions cover independent reuse,
compound keys, duplicate order, nulls, empty build inputs, chained joins,
early limits, signed extremes, irregular progressions, and incomplete final
runs. Existing tests cover collision equality, strings, NaNs, signed zero,
large probes, and output ordering. A further regression verifies that null
string, numeric, or Boolean key components never occupy hash slots or
duplicate chains; the hash/streaming modules pass with this correction on
both hosts (26 tests). Heavy duplicate groups have an additional numeric,
string, and compound-key regression checking exact row order, nulls, repeated
probes, identity fallback, and semi/anti membership; the three modules pass
on Linux and M1 (27 tests).

The following times are medians of five samples from separate processes;
each process warms once before one timed query. Revision order alternates
between rounds. Both sides use `-O3 -g1`, Mojo
`1.2.0.dev2026092105 (e9569894)`, four workers, and identical 1M-row matrix
inputs. Source, lockfile, and input hashes match between machines. These
measurements concern the materializing `lazy_narrow` join, not the upstream
aggregate-over-join queries. Each timed iteration checks height; warmed
results agree in height/checksum across revisions. The semantic tests make
full-result assertions. Affinity is unpinned, and the
compiler guard does not exclude every possible background task.

| Host | Layout | Baseline ms | Prepared ms | Speedup |
|---|---|---:|---:|---:|
| Linux Threadripper 3970X | ordered | 25.751 | 22.390 | 1.15x |
| Linux Threadripper 3970X | shuffled | 104.173 | 62.463 | 1.67x |
| Linux Threadripper 3970X | wide | 215.107 | 61.091 | 3.52x |
| Mac M1 | ordered | 13.632 | 8.840 | 1.54x |
| Mac M1 | shuffled | 94.793 | 22.002 | 4.31x |
| Mac M1 | wide | 169.319 | 22.230 | 7.62x |

[Linux samples](benchmarks/prepared-joins-linux.json),
[Mac samples](benchmarks/prepared-joins-mac.json), and
[Linux profile](benchmarks/prepared-joins-linux-profile.json) retain the
supporting evidence. Baseline engine code is the merged PR #297 baseline.
Candidate source digests identify this implementation before its commit.

Before the duplicate-group correction, a separate 199 Hz user-cycle/DWARF recording of 40 wide-key repetitions
attributes 55.95% self samples to hash probing and 4.11% to hash building.
The repeated dictionary encoding/growth routines that dominated the earlier
profile are absent from the leading entries. This is a whole-process profile
including loading and warmup, not a query-only counter. Raw captures remain
in `build/profiles`. The profile motivates the later probe-efficiency work;
its percentages are not speedup estimates.

Reproduce with baseline and candidate matrix runners built from their
respective revisions, then run after all compilation has stopped:

```bash
python3 scripts/bench_join_revision.py \
  --baseline /path/to/baseline/bench_join_matrix \
  --candidate build/bench_join_matrix_prepared \
  --data /path/to/bench_polars --output build/prepared-comparison.json
```

Opportunities 2–5 remain separate changes; these measurements do not claim
the high-cardinality or duplicate-string count gaps are closed.

## Full-query regression check

After adding compact duplicate groups, paired full-query runs compare the
merged PR #297 engine with the final prepared implementation. Each host has
300 timed samples: three alternating process rounds, one warmup and five
checked repetitions, at 1M and 10M right rows with four workers. Every timed
result passes the upstream runner’s complete-answer check. The table gives
medians of per-round medians for the 10M cases; raw files also include 1M.
Compiler, affinity, loading, and RSS caveats are the same as the upstream
baseline. Input and candidate-source hashes match across hosts.

| Host | Query | Layout | Baseline ms | Prepared ms | Speedup |
|---|---|---|---:|---:|---:|
| Linux | highcardinality | base | 41.555 | 37.757 | 1.10x |
| Linux | highcardinality | shuffled | 736.270 | 186.874 | 3.94x |
| Linux | highcardinality | wide | 740.627 | 185.339 | 4.00x |
| Linux | duplicate_strings | base | 1749.861 | 1133.218 | 1.54x |
| Linux | duplicate_strings | shuffled | 1830.178 | 1123.313 | 1.63x |
| M1 | highcardinality | base | 33.573 | 21.269 | 1.58x |
| M1 | highcardinality | shuffled | 403.373 | 133.077 | 3.03x |
| M1 | highcardinality | wide | 404.680 | 133.608 | 3.03x |
| M1 | duplicate_strings | base | 866.592 | 677.592 | 1.28x |
| M1 | duplicate_strings | shuffled | 986.790 | 737.417 | 1.34x |

The first prepared representation regressed shuffled duplicate strings by
following long, scattered duplicate chains. A separate profile attributed
56.93% self samples to probing, 12.17% to generic row equality and 10.32% to
string access. Compact groups remove that regression while preserving row
order. These profile percentages include loading and warmup; they are not
query-only counters or speedup estimates.

[Linux full-query samples](benchmarks/prepared-upstream-linux.json) and
[M1 full-query samples](benchmarks/prepared-upstream-mac.json) preserve all
measurements, binary/source hashes, input hashes and whole-process peak RSS.
Reproduce using `scripts/bench_upstream_revision.py --baseline <runner>`
`--candidate <runner> --data <PR297 CSV directory> --output <result.json>`.

## Completed implementation records

The five opportunities are now implemented in separate PRs. Probe efficiency
has separate ownership and typed-dispatch changes. The
[integration record](performance-followups-integration.md) maps each change
to its review/dependency and records combined Linux/M1 semantic validation.
Each PR retains its own paired measurements rather than attributing the
whole sequence's gains to one change.
