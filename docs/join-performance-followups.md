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

Seven affected Linux test modules passed (61 tests). After the final
profile-guided probe changes, the hash and streaming modules passed again
on Linux and M1 (25 tests on each). New assertions cover independent reuse,
compound keys, duplicate order, nulls, empty build inputs, chained joins,
early limits, signed extremes, irregular progressions, and incomplete final
runs. Existing tests cover collision equality, strings, NaNs, signed zero,
large probes, and output ordering. A further regression verifies that null
string, numeric, or Boolean key components never occupy hash slots or
duplicate chains; the hash/streaming modules pass with this correction on
both hosts (26 tests).

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
| Linux Threadripper 3970X | ordered | 28.382 | 25.369 | 1.12x |
| Linux Threadripper 3970X | shuffled | 100.084 | 47.311 | 2.12x |
| Linux Threadripper 3970X | wide | 212.061 | 47.341 | 4.48x |
| Mac M1 | ordered | 13.690 | 8.804 | 1.55x |
| Mac M1 | shuffled | 94.808 | 19.487 | 4.87x |
| Mac M1 | wide | 170.932 | 19.519 | 8.76x |

[Linux samples](benchmarks/prepared-joins-linux.json),
[Mac samples](benchmarks/prepared-joins-mac.json), and
[Linux profile](benchmarks/prepared-joins-linux-profile.json) retain the
supporting evidence. Baseline engine code is the merged PR #297 baseline.
Candidate source digests identify this implementation before its commit.

A separate 199 Hz user-cycle/DWARF recording of 40 wide-key repetitions
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

Opportunities 2–5 remain to be implemented and measured. Broader upstream
query measurements will follow those changes; the table above does not
claim the high-cardinality or duplicate-string count gaps are closed.
