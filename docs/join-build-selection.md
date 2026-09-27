# Cardinality-based join build selection

This change follows prepared streaming joins in #298. Inner and left joins
may build on the logical left input when the right side has at least 524,288
rows and is at least 32 times larger. This uses actual cardinalities, not
key values or workload names. Existing progression lookup remains preferred
when available. Other sizes retain the existing dispatch.

Reversing the physical roles produces right-major matches. A stable counting
scatter restores left-major output and right-input duplicate order, including
unmatched left rows. Public schemas, suffixes and coalescing are unchanged.

A lazy join uses this route only when its left height is known cheaply and
fits one execution batch. Unknown or larger left inputs retain prepared
streaming, so this optimization does not turn multi-batch joins into unbounded
materializing joins. Direct in-memory right scans retain prepared progression lookup when a full
validation proves that representation applies. Explain marks the conditional
boundary without scanning the data. Exact heights can
propagate through row-preserving projections and nonnegative slices; filters
remain unknown.

## Cutoff evidence

The forced-orientation benchmark measures row matching and order restoration,
excluding input construction and payload gathering. Every timed run checks
its complete ordered row pairs against the ordinary right-build result.
There are 360 samples per host: five alternating repetitions, left sizes
4,096 and 131,072, unique and fourfold-duplicate keys, ratios from 1 to 1,024,
and a cap of 8,388,608 right rows. Each orientation is warmed first.

Small tables can regress despite large ratios, especially on M1. The
conservative two-part gate avoids those cases. Within the selected region,
median right-build/left-build ratios were 1.53–2.36x on Linux and 1.30–2.28x
on M1. These are calibration measurements on the source hashes recorded in
the artifacts, before #298's final duplicate-group correction. Unique and
fourfold-duplicate keys do not activate that heavy-duplicate representation.
Final full-query checks against the updated #298 parent are recorded below.
The gate is a measured conservative heuristic, not a guarantee for every
machine, dtype, distribution or payload.

[Linux samples](benchmarks/smaller-build-ratios-linux.csv),
[Linux metadata](benchmarks/smaller-build-ratios-linux.json),
[M1 samples](benchmarks/smaller-build-ratios-mac.csv), and
[M1 metadata](benchmarks/smaller-build-ratios-mac.json) retain the raw evidence.
Both hosts use four workers and Mojo 1.2.0.dev2026092105 (e9569894), without
CPU pinning. The Linux host is a Threadripper 3970X; the Mac is an Apple M1.

Reproduce by building `benchmarks/bench_join_build_ratio.mojo` with `-O3 -g1`
and running it with `DATAFRAME_THREADS=4` after compilation has stopped.

## Semantic validation

Regression tests assert exact inner/left output with duplicate and null keys,
unmatched rows, and lazy batch sizes 1, 3, and 65,536. Compound Float64/string
keys check NaNs and signed zero. Additional tests cover unknown/empty sizes,
Int32 index capacity and cutoff guards. Eight affected modules passed on
Linux before the final parent update; build-side and streaming regressions
are repeated on both hosts against the updated parent. A regression also
checks ordered progression retention and rejection of irregular keys.

## Final full-query comparisons

The paired baseline is the corrected prepared implementation in #298
(`74122d8`). Each host has 180 timed results: high-cardinality queries at
1M and 10M right rows, ordered/shuffled/wide layouts, three alternating
process rounds and five checked repetitions after one warmup per process.
Every timed result passes the complete-answer check. Input and candidate
source hashes match across hosts. These runs exclude compilation, use four
workers, and measure the whole aggregate-over-join query with data loading
outside timing. Reported times are medians of per-round medians.

| Host | Right rows | Layout | Prepared ms | Smaller-build ms | Speedup |
|---|---:|---|---:|---:|---:|
| Linux | 1,000,000 | base | 4.584 | 4.398 | 1.04x |
| Linux | 1,000,000 | shuffled | 17.018 | 12.836 | 1.33x |
| Linux | 1,000,000 | wide | 17.065 | 12.795 | 1.33x |
| Linux | 10,000,000 | base | 38.307 | 37.644 | 1.02x |
| Linux | 10,000,000 | shuffled | 183.649 | 100.978 | 1.82x |
| Linux | 10,000,000 | wide | 186.743 | 103.369 | 1.81x |
| M1 | 1,000,000 | base | 2.679 | 2.527 | 1.06x |
| M1 | 1,000,000 | shuffled | 11.095 | 6.330 | 1.75x |
| M1 | 1,000,000 | wide | 10.849 | 6.313 | 1.72x |
| M1 | 10,000,000 | base | 21.387 | 19.756 | 1.08x |
| M1 | 10,000,000 | shuffled | 134.975 | 58.772 | 2.30x |
| M1 | 10,000,000 | wide | 134.495 | 58.855 | 2.29x |

An earlier candidate materialized ordered joins and regressed those cases.
The final planner retains validated prepared progression state for direct
in-memory right scans; the table measures that corrected implementation.
Small timing differences for ordered inputs are not evidence of a new
ordered-key algorithm. Duplicate-string count queries have multi-batch left
inputs and retain the prepared path; this PR does not claim to improve them.

[Linux full-query results](benchmarks/smaller-build-linux.json) and
[M1 full-query results](benchmarks/smaller-build-mac.json) include binary hashes,
source hashes, input hashes, raw samples and whole-process peak RSS. RSS
includes loading and warmup, not just query execution. Both hosts passed the
final build-side and streaming modules (12 tests each).

Reproduce with `scripts/bench_upstream_revision.py --baseline <PR298 runner>`
`--candidate <runner> --data <PR297 CSV directory> --cases highcardinality`
`--output <result.json>`. Safe derived filters, count fusion and probe work
remain separate follow-ups.
