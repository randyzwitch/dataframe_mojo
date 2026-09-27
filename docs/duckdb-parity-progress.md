# Ten-PR performance effort

The user will evaluate progress after ten distinct optimization PRs following
merged PR #303. Count PRs actually opened, not experiments, benchmark-only
changes, or previous merged optimizations. The objective is general execution
performance approaching DuckDB on equivalent workloads and hardware.

## Acceptance criteria

- Start with the relevant DuckDB or Polars implementation and explain the
  transferred algorithm, architectural limitation, or measured bottleneck.
- Improve a shared execution mechanism. Query shapes copied from a benchmark
  are not an acceptable dispatch criterion.
- Check semantics, including nulls, duplicates, ordering, and supported dtypes.
- Record baseline/candidate sources, input identity, hardware, worker count,
  loading boundaries, raw samples, and regressions.
- Distinguish primitive measurements, full-result operators, and complete
  queries. Count fusion is not evidence of full-result join throughput.
- Validate beyond the workload that motivated the change. Claims must stay
  within the evidence. A loss or a small gain is not parity.
- Open one PR per distinct change, with dependencies explicit. Do not merge
  automatically or use a ten-PR quota to justify low-impact specialization.

## Published optimizations

1. [PR #305](https://github.com/randyzwitch/dataframe_mojo/pull/305): shared
   eight-byte block hashing for long strings. Open, awaiting CI and review.
   Linux full-result joins improve 1.05–1.18x on the measured longer-key layouts;
   this is not a whole-engine parity claim.

## First optimization

Shared long-string hashing using DuckDB's eight-byte block mixer, with paired
Linux and Apple M1 measurements and materialized-join checks. See
[word-string-hashing.md](word-string-hashing.md).

## Corrective architectural work still required

The prior benchmark-shaped paths remain outstanding, not implicitly endorsed
by their presence in the baseline:

- Replace the narrow sum-plus-count group execution with a shared aggregation
  mechanism supporting arbitrary accumulator lists and numeric dtypes.
- Implement general lazy projection pushdown across joins and intervening
  operators; report count-only and materialized queries separately.
- Remove the GCD-stride and equal-repeat-run join specializations while
  retaining dense direct addressing and ordinary sequential-key handling.
- Evaluate the general hash-join build/probe design using profiles and upstream
  implementations, including allocation, payload gathering, and memory traffic.

The remaining PR slots are deliberately unassigned until profiles and validated
changes justify them. Unpublished count-only caching experiments are set aside.
