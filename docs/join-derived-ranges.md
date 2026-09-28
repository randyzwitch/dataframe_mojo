# Safe join-derived range filters

> **Retired benchmark.** The scripts this record uses to reproduce its measurements were retired in favor of the external suites ([benchmarks.md](benchmarks.md#retired-benchmarks)); check out revision `b9db9c6` to rerun them.

This change follows build-side selection in #299. For small-left inner/left
joins selected by its cardinality gate, exact bounds from valid physical
Int64 left keys can reject right rows before hashing or payload gathering.
For compound keys, any supported component supplies a necessary condition;
the ordinary join still checks every key component.

Bounds use comparisons without subtraction, including signed extremes and
temporal storage. Null right keys cannot match. Left rows are never removed,
so unmatched left-join rows retain their ordinary null-filled outputs. Other
join modes and prepared streaming joins retain their existing paths.

A 256-position sample attempts filtering only when at most one quarter of
the sampled right rows survive. This is a cost guard, never a correctness
assumption: a parallel full scan checks all rows, preserves their order and
declines gathering if more than one quarter actually survive. A bounded
worker pool handles arbitrary chunk layouts without rechunking key buffers.
This selectivity threshold is conservative, not claimed universally optimal.
Broad ranges retain ordinary hashing, even if many individual keys miss.

## Validation

Seventeen tests in the range, build-side and general join modules passed on
Linux and M1. New tests cover nulls, empty inputs, signed extremes, temporal
storage, chunk/slice offsets, compound-key false positives and exact inner/left
output ordering. An adversarial sample that rejects every sampled row while
most actual rows match proves that sampling cannot silently drop matches.

## Paired full-query measurements

The baseline is #299 (`39e462d`). Both hosts use the same input/source hashes,
Mojo 1.2.0.dev2026092105 (e9569894), four workers and unpinned affinity. Each
host has 180 checked samples: 1M/10M right rows, three layouts, three alternating
process rounds, five repetitions and one warmup per process. Loading is
outside query timing; raw peak RSS covers the whole process. Times below
are medians of per-round medians. No compilation overlapped timing.

| Host | Right rows | Layout | Baseline ms | Filter ms | Speedup |
|---|---:|---|---:|---:|---:|
| Linux | 1,000,000 | base | 4.376 | 4.451 | 0.98x |
| Linux | 1,000,000 | shuffled | 14.253 | 3.077 | 4.63x |
| Linux | 1,000,000 | wide | 12.410 | 13.665 | 0.91x |
| Linux | 10,000,000 | base | 37.081 | 36.943 | 1.00x |
| Linux | 10,000,000 | shuffled | 101.286 | 27.554 | 3.68x |
| Linux | 10,000,000 | wide | 101.697 | 100.872 | 1.01x |
| M1 | 1,000,000 | base | 2.475 | 2.475 | 1.00x |
| M1 | 1,000,000 | shuffled | 6.306 | 1.555 | 4.06x |
| M1 | 1,000,000 | wide | 6.286 | 6.342 | 0.99x |
| M1 | 10,000,000 | base | 19.619 | 19.607 | 1.00x |
| M1 | 10,000,000 | shuffled | 59.331 | 9.963 | 5.96x |
| M1 | 10,000,000 | wide | 58.603 | 58.836 | 1.00x |

Shuffled contiguous-domain keys benefit from rejecting impossible probes.
Ordered inputs retain compact progression lookup. Wide-key bounds span
most of the right domain, so the sample declines filtering; this change
does not claim to solve that case. The Linux 1M wide case was 9% slower in this run; no benefit is claimed
for unselective ranges. Smaller variations on unchanged paths should not
be interpreted as algorithmic gains.

[Linux raw results](benchmarks/join-ranges-linux.json) and
[M1 raw results](benchmarks/join-ranges-mac.json) retain provenance and all
samples. Reproduce using `scripts/bench_upstream_revision.py` with the #299
runner as `--baseline`, this runner as `--candidate`, the PR297 CSV directory
as `--data`, `--cases highcardinality`, and an `--output` JSON path.
