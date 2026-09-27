# Low-cardinality sort calibration (#271)

Measured on 2026-09-26 with Mojo 1.2.0.dev2026092105 (e9569894), Linux
x86-64, Threadripper 3970X (32 physical cores, 64 logical CPUs), 32 workers.
Each paired process warms once and records five repetitions. Two rounds
reverse algorithm order; tables report the mean of per-round minima in ms.
No affinity pinning; the governor is schedutil. Input construction and checks
of permutation, lexicographic order and stable ties are outside timing.

The benchmark creates rank words directly, separating the algorithm decision
from dtype encoding. It covers 2/3/4 rank words, 12/47/63-bit remaining ranks,
row counts around 200k, first-key domains 8 through 1024, and hot buckets of
20/25/30/50/80%. Forced copies remove the production bucket range/balance
rejection or disable buckets entirely. Production files are never patched.

These shapes model sorting within a category (region, status, device class)
by one or more numeric/temporal attributes. Uniform categories are the good
case; dominant categories and wide ranks are deliberate counterexamples.
This is kernel calibration, not a general comparison against another engine.

## Evidence

For 16 balanced first-key values:

| Rows | Remaining ranks | Merge | Bucket |
|---:|---|---:|---:|
| 150k | 3 × 47-bit | 7.02 | 4.73 |
| 200k | 3 × 47-bit | 8.70 | 6.26 |
| 250k | 3 × 47-bit | 8.80 | 7.78 |
| 500k | 3 × 47-bit | 16.37 | 17.83 |
| 1M | 3 × 47-bit | 29.06 | 29.54 |
| 1M | 3 × 12-bit | 36.14 | 22.26 |

Rank width matters because the radix path skips unchanged bytes. The old
200k cutoff was conservative for narrow ranks; removing it unconditionally
would select losing wide-rank cases.

There is no cliff at 64/65 first-key values. With 100k rows and two words,
64/65/128/256 values take 2.93/2.98/2.99/2.99 ms in buckets versus about
5.1–5.4 ms in the merge path. At 65,536 rows and four words, 256 values
still win (3.35 vs 3.79 ms), 512 are marginal (3.91 vs 3.78), and 1024
lose (4.68 vs 3.71). Large bucket counts have allocation and job overhead,
so the range guard is a resource/amortization policy, not a correctness limit.

The former 25% balance allowance also ignores remaining-word work. At 100k
rows with 16 first-key values and a 25% hot bucket, two words win (3.07 vs
5.07 ms) but four words lose (6.27 vs 5.47 ms). At 1M rows, even a 20%
hot bucket with four words loses (49.27 vs 34.10 ms). The load-balance guard
must account for work in the largest serial bucket.

## Reproduce

```bash
pixi run python3 scripts/bench_sort_cutoffs.py --build
pixi run python3 scripts/bench_sort_cutoffs.py --run rows
pixi run python3 scripts/bench_sort_cutoffs.py --run cardinality
pixi run python3 scripts/bench_sort_cutoffs.py --run skew
pixi run python3 scripts/bench_sort_cutoffs.py --run edge
pixi run python3 scripts/bench_sort_cutoffs.py --run wide
pixi run python3 scripts/bench_sort_cutoffs.py --build --variants baseline,candidate
pixi run python3 scripts/bench_sort_cutoffs.py --run validation --variants baseline,candidate
```

The pinned baseline is `b2013a55ae9c19ce447e2a37f692e7a394eeaa9a`.
The 12/47-bit fixtures use bits above the generator's low 16 bits; the
63-bit fixture uses all remaining bits after the low bit. First-key generation
is identical for all widths. Raw CSVs are in `build/sort-cutoffs`.

## Selected policy and validation

The domain histogram is capped at 256 values (2 KiB), with at least 128 rows
per occupied bucket to amortize bucket/job setup. For more than three rank
words, cap the remaining-rank input at 5 MiB: 200k four-word rows qualify,
250k do not. This deliberately leaves some narrow-rank wins on the merge
path; inspecting effective rank widths would require another scan.

A bucket may hold at most one quarter of the input. Tighten that allowance
when its remaining-rank work exceeds 16 average worker shares:
`largest <= rows / max(4, ceil(workers * (words - 1) / 16))`.
This accounts for the parallel merge alternative instead of applying the
32-worker balance rule to an eight-worker machine. The sort computes its
worker count once and reuses it for dispatch, balance and pool sizing.

An initial stricter balance rule regressed useful bucket sorts on the M1.
Final validation uses an Apple M1 Mac mini (4 performance + 4 efficiency
cores, 16 GiB RAM, 12 MiB performance-cluster L2 and 4 MiB efficiency-cluster
L2), Darwin 25.5, the same Mojo nightly, and eight workers.
The three-word/eight-value case remains on the bucket path and is flat
(21.79 → 21.73 ms at 1M rows). The 8,192-row/64-value case is also retained
(0.36 → 0.39 ms). Selected final baseline/candidate results:

| Rows | Words | First-key values | Baseline | Candidate |
|---:|---:|---:|---:|---:|
| 65,536 | 4 | 256 | 3.17 | 3.73 |
| 100k | 2 | 65 | 4.76 | 1.69 |
| 100k | 4 | 256 | 4.71 | 4.22 |
| 200k | 4 | 8 | 6.64 | 6.63 |
| 1M | 2 | 65 | 49.54 | 13.35 |
| 1M | 2 | 256 | 50.51 | 13.63 |

Small cases vary: the 65,536-row/four-word/256-value case won in earlier
paired runs (about 2.9 vs 3.2 ms) but loses in the final run above. The
100k/four-word/25%-hot case is 3.54 → 3.90 ms. These are not uniformly
faster dispatch rules. The large new-domain gains are much more substantial;
all final samples, including losses, are retained in the
[companion CSV](sort-cutoffs.csv). Linux rows in that CSV are forced-kernel
calibration; `mac-validation-8` is the final production-policy comparison.
Worker-pool calibration and its separate spin change belong to #273.

Final tests cover stable row order across 64/65/256/257 values, 2/3/4 words,
negative ranks, ties and skew, as well as the existing serial/parallel sort
agreement tests. Every benchmark repetition also validates its complete
stable permutation outside timing.

To export the committed data after reproducing the Linux suites and copying
the Mac validation CSV to `build/sort-cutoffs/mac-validation-8.csv`:

```bash
pixi run python3 scripts/bench_sort_cutoffs.py --report docs/sort-cutoffs.csv
```
