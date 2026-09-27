# Join and gather cutoffs (#268, #272)

## Measurement setup

Measured 2026-09-26 against `b2013a55ae9c19ce447e2a37f692e7a394eeaa9a`,
using Mojo `1.2.0.dev2026092105` (`e9569894`) on Linux x86-64. The machine is
an AMD Threadripper 3970X: 32 physical cores, 64 logical CPUs, 512 KiB L2
per core and eight **16 MiB** L3 cache domains (128 MiB aggregate, not one
128 MiB shared cache). The governor was `schedutil`; CPU affinity was not
pinned. Desktop background services remained, but no other benchmark or
compiler ran during a timing sweep.

Each input/algorithm pair ran in two rounds, reversing algorithm order in
round two. Each process warmed once, then recorded five timed repetitions.
Tables below report the mean of the two per-round minima, in milliseconds.
Inputs and validation are outside timing. These are calibration measurements,
not universal crossover guarantees or performance claims against another
engine. The [companion CSV](join-cutoffs.csv) retains every round and sample. The focused final recheck uses four
rounds of seven repetitions, with the same alternating order and summary rule.

`benchmarks/bench_join_cutoffs.mojo` exercises CSR construction, progression
mapping, bounded row chains, membership, dense IDs followed by CSR, chunked
gathering, and public joins. Join keys are shuffled, with irregular gaps at
100%, 50%, 25%, and 10% density so the sparse cases cannot silently compress
into a small GCD domain. Separate cases cover regular strides, ordered IDs,
and chunked probes. Pair/order checks and the existing join tests guard
semantics. A sequence of independently shuffled dimension-table IDs is a
real input for the bounded path; calendar/sequence lookups serve the ordered
progression path. Neither shape stands in for general join performance.

## Reproduce

The driver extracts the pinned baseline into disposable directories under
`build/cutoffs`, then builds forced serial, parallel, and hash alternatives.
It widens allocation guards only in those copies to measure domains that
production correctly declines. Production source is never patched by the
driver. `--revision` is explicit and the driver refuses revisions whose old
allocation guards no longer match.

```bash
pixi run python3 scripts/bench_join_cutoffs.py --build
pixi run python3 scripts/bench_join_cutoffs.py --run basic --threads 32
pixi run python3 scripts/bench_join_cutoffs.py --run ranges --threads 32
pixi run python3 scripts/bench_join_cutoffs.py --run strided --threads 32
pixi run python3 scripts/bench_join_cutoffs.py --run ids --threads 32
pixi run python3 scripts/bench_join_cutoffs.py --run fine --threads 32
pixi run python3 scripts/bench_join_cutoffs.py --run shape --threads 32
pixi run python3 scripts/bench_join_cutoffs.py --run tails --threads 32
pixi run python3 scripts/bench_join_cutoffs.py --run compact --threads 32
pixi run python3 scripts/bench_join_cutoffs.py --run portability --threads 8
pixi run python3 scripts/bench_join_cutoffs.py --build --variants baseline,candidate
pixi run python3 scripts/bench_join_cutoffs.py --run validation --threads 32
pixi run python3 scripts/bench_join_cutoffs.py --run recheck --threads 32 --rounds 4 --reps 7
pixi run python3 scripts/bench_join_cutoffs.py --report docs/join-cutoffs.csv
```

Use optimized `mojo build` executables, not JIT runs. CSVs under
`build/cutoffs` keep every repetition; each filename includes its worker
limit. `portability` rechecks CSR, progression, 2/8-column gathers, and
100%/25%-dense row chains at 500k, 1M, 2M, and 4M rows. It changes worker
count on this machine, **not cache architecture**; #273 still covers other
machines and the remaining pool/worker constants.

## Allocation policy

All three bounded paths use `_range_join_table_capacity`. It caps the domain
table at four times the relevant Int64 input-key storage (32 bytes per input
row) and at 128 MiB, dividing by actual per-slot storage before allocation.
Arithmetic saturates before multiplying, including for `Int.MAX` rows.

| Path | Input rows charged | Bytes per domain slot | Additional guard |
|---|---|---:|---|
| Right-row chains | right/build rows | 8 (head) | Above 8 MiB, at most two slots per build row |
| Membership | right/build rows | 1 (presence flag) | At most 12,000,000 bytes |
| Dense IDs / CSR | both inputs | 16 (starts and cursor) | Reserve the two extra end entries |

The relative guard counts input rows, including duplicates and nulls; it does
not claim to estimate the number of distinct keys.

These are **domain-table budgets**, not total operation-memory bounds. Input
buffers, next-row links, row IDs, worker scratch, gather output, and join
fanout consume additional memory. The row-chain budget drops from up to
512 MB of heads to 128 MiB. The CSR absolute limit remains about eight million
slots, now explained by its two per-domain arrays rather than an unrelated
slot constant. Sorted consecutive membership needs only bounds and allocates
no presence table, so the table cap does not reject that case.

Measured row-chain totals, including scan, build, and probe:

| Build/probe rows | Density | Serial bounded | Parallel bounded | Hash |
|---:|---:|---:|---:|---:|
| 100k | 25% | 2.31 | 2.28 | 7.19 |
| 500k | 25% | 32.79 | 19.30 | 15.89 |
| 1M | 100% | 25.11 | 23.26 | 26.23 |
| 1M | 25% | 82.70 | 29.90 | 26.39 |
| 4M | 100% | 268.39 | 84.16 | 93.83 |
| 4M | 25% | 350.03 | 112.92 | 94.14 |
| 10M | 100% | 792.50 | 217.40 | 228.55 |
| 10M | 50% | 862.35 | 235.55 | 238.15 |

The 128 MiB ceiling is a resource policy between the winning 80 MB head table
and a 160 MB table that merely ties hashing; it is not a claim that 128 MiB
is a measured sharp crossover. The density guard avoids the larger sparse
tables that lose even with parallel construction. Small tables retain the
existing allowance of up to four slots per build row; the 10%-dense forced
measurements do not authorize unbounded sparse allocations.

Membership has a different measured limit because its table is constructed
serially. At 2M build/probe rows, 10/12/14/16 MB presence tables take about
32/38/46/55 ms; hashing is about 39 ms. At 10M rows, the 10 MB table wins
(136 vs 207 ms), but 20 MB loses (322 vs 208 ms). This is why applying the
same **slot count** or the full 128 MiB ceiling to membership is inappropriate.
The byte-relative guard also permits small sparse membership sets that the
old four-slots-per-row condition unnecessarily sent to hashing.

## Execution cutoffs

### Parallel index construction

Partition a dense index build when heads occupy at least 8,000,000 bytes and
keys + next-row links + heads occupy at least 20 MiB. The existing cap of 16
builders stays unchanged. At 500k keys / 8 MB heads, serial wins (14.7 vs
15.6 ms). At 750k / 12 MB, parallel wins (20.1 vs 29.4 ms); at 1M / 8 MB it
also wins (23.3 vs 25.1 ms). Both old two-million-row conditions disappear.
Eight-worker checks retain the larger-case gains. This is a cache/working-set
criterion; very small head tables stay serial even with many duplicate rows.

### Strided bounded indexes (removed)

The GCD scan and strided scatter were removed in #307. Shuffled keys that
are all multiples of a common stride do not occur in real tables; the only
workload that used this mode was the join matrix's `jk * 17` case. The mode
won only through 250k build keys and lost from 1M. Such keys now use the
dense index when their span is compact and the hash index otherwise. The
measurements below are kept for the record.

At 250k build keys, direct addressing won for probe/build ratios 0.25,
0.5, 1 and 2; at 500k the results were mixed; at 1M hashing won at every
ratio. With 2M keys on each side, direct took about 120 ms vs hash's 46 ms.

### Stable CSR scatter

Use parallel scatter only when output rows + cursors exceed 16 MiB and the
cursor has at least 4,096 entries (32 KiB). Small-key sweeps at 6M rows give:

| Groups | Serial | Parallel |
|---:|---:|---:|
| 64 | 63.3 | 82.8 |
| 512 | 34.7 | 66.9 |
| 4,098 | 63.8 | 62.1 |
| 5,859 | 126.6 | 71.1 |

High-cardinality shuffled IDs favor parallelism sooner: 750k distinct rows
lose (15 vs 18 ms), 1M win (33 vs 21–26 ms), and 2M win decisively (135–137
vs 36–38 ms). The conservative 16 MiB guard deliberately leaves some marginal
wins serial: an initial 14 MiB guard helped the uniform shuffled case but
regressed the 1M dictionary-encoded full join. Cardinality and allocation
layout make these transitions non-monotonic; do not infer a precise universal
boundary from a single fixture.

Ordered IDs are a counterexample even at large sizes: 1M/2M/4M distinct rows
take 6/13/25 ms serial vs 19/40/69 ms parallel. After the footprint guards,
check at most 64 sampled adjacent pairs for an inversion. No inversion keeps
serial scatter. This is only an algorithm-cost heuristic: both algorithms
preserve exactly the same row order; a missed inversion cannot change
results. Random IDs usually find an inversion in the first few pairs.

### Progression mapping

For single-row-per-key progressions, parallelize only contiguous probes with
at least 24 MiB of key data. At 32 workers, 1.5–2M probes lose and 3–6M win;
eight workers still lose at 2M and win at 4M, so scaling the threshold as rows
per worker selected a losing path. Chunked probes stay serial: the required
rechunk loses at 2M, 4M, 6M, 8M and 10M. The existing repeated-key progression
path is unchanged; its broader scope belongs to #269.

### Sorted chunk gathers

Require at least 16,384 selected rows per dispatched column/partition job,
more than one effective worker, and the existing chunk-count/partition
conditions. Two-column gathers win from roughly 100–125k selected rows;
eight-column gathers are mixed or flat around 500–750k and win at 1M.
The rule accounts for job count instead of treating every frame width as the
same two-million-row workload. The existing four-part limit stays unchanged.

The fused-expression (200k) and aligned-filter (50k) thresholds had already
been measured in #282 and are not changed here. Their evidence remains next
to their constants.

## Public-join validation

The final paired baseline/candidate results are recorded in the companion
CSV's `validation` rows. They time the public join, including output gathering,
on identical generated inputs; they are separate from the forced-kernel
calibration above. Ordered full joins and 1.25M-row cases straddle the new
CSR decision, and 10M-row cases exercise the large-table policy.

Selected public results (milliseconds; density refers to the key domain):

| Rows per input | Shape / operation | Baseline | Candidate |
|---:|---|---:|---:|
| 1M | 25% dense inner | 88.56 | 28.09 |
| 1M | 25% dense full | 160.43 | 140.15 |
| 1M | 10% dense semi | 33.46 | 25.89 |
| 2M | 25% dense inner | 68.12 | 51.74 |
| 4M | 25% dense inner | 126.70 | 103.58 |
| 4M | 25% dense semi | 134.51 | 113.88 |
| 2M | ordered full | 130.83 | 114.00 |
| 10M | dense inner | 249.63 | 247.51 |
| 10M | 50% dense semi | 364.09 | 273.25 |

The sweep is not uniformly faster: 4M dense full takes 317.88 → 337.49 ms,
and 4M 10%-dense full takes 778.83 → 806.17 ms. These cases retain the same
algorithm selections; timings alone do not establish a cause. Two apparent
9–10% losses on unchanged paths received a focused four-round, seven-repeat
recheck: 2M dense inner measured 49.19 → 48.15 ms and 2M 25%-dense semi
40.43 → 39.81 ms. An unchanged 1M 10%-dense inner control measured
28.47 → 28.01 ms. The initial losses did not persist. Both the original
sweep and recheck samples are retained; the table above uses the original
final sweep, without substituting favorable recheck results.

The CSV also retains `validation-first` from the preliminary 14 MiB CSR
policy; those rows are diagnostic history, not the final candidate.
Correctness validation passed all 65 test modules (including Parquet) and
150 generated Polars-oracle cases. Tests explicitly cover byte-budget
saturation, membership fallback past its cap, and contiguous/chunked
progression order and null handling. Formatting, version/dtype checks and
API documentation generation also passed.
