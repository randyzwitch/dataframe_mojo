Measurements in this note were taken at commit 51c40c90 (2026-10-20); they are a record of that decision, not current numbers.

# Measuring Apple Metal development mechanisms

`benchmarks/bench_metal.mojo` and `benchmarks/tools/bench_metal.py` compare the existing
DataFrame CPU executor with the Mojo Metal backend on the same Mac.
These are development mechanisms, not an external suite or evidence of general
performance. `chain32` deliberately measures fusion across lazy projection
stages; its ratio includes the current CPU executor's per-stage overhead.

Run from an Apple Silicon checkout:

```sh
pixi run python3 benchmarks/tools/bench_metal.py
```

The driver builds one optimized Mojo binary. For a quick correctness check:

```sh
pixi run python3 benchmarks/tools/bench_metal.py --rows 4097 --reps 2 --rounds 1
```

Optional `--baseline-binary /path/to/older/bench-metal` compares separately
built Mojo executables. Each baseline includes its own CPU/provider revision;
source/compiler/binary fingerprints identify the comparison. `--binary PATH
--skip-build` reuses an already built runner. The driver records compiler
version and flags, source/driver/binary hashes, engine commit and status,
selected thread count, Mac hardware/OS/SDK details, actual Mojo profiles, load
before each process, raw samples, and failed or unsupported outcomes.

Each workload, variant, executable build, and round runs in a fresh process.
First-use CPU and Metal collections are recorded separately. Two additional
warmups precede alternating CPU/Metal samples. The initial engine order and
executable order rotate between rounds; case order reverses. First use
includes device/context/shader setup, but the machine and filesystem are not
cold. The input is constructed once outside timing; every collection allocates
and materializes a complete ordinary CPU result. Validation and result release
are outside the measured interval. Every measured result is checked exactly
against the current DataFrame CPU result. These checks supplement the precision and overflow contracts; they are not an independent SQL oracle.

Cases cover a simple projection, a 15-repetition single expression, a 32-stage projection
chain, filters retaining about half or one percent of rows, count/length after a
filter, exact Int32 sum, filtered Int32 sum, and a filter whose predicate column
is omitted from the output. Additional cases exercise stable sorting, low/high
cardinality grouping, a self-left-join and a bounded fanout self-left-join.
join_fanout uses ceil(N/4) balanced key classes, so every key appears at most
four times and the result contains at most 4N rows for every positive row count.
The wide_float32 projection returns eight columns, at most 8N Float32 values
(256 MiB at the default largest input). float64_arithmetic evaluates
x * 2 + 0.25 over exact binary fractions. decimal_sum uses Decimal(15,2)
inputs from -0.08 to 0.08, multiplies by 1.25, and sums the result.
Base, sorted, and nullable/permuted distributions all run by default. Float32
inputs are exact multiples of 1/1024. Int32 sum inputs stay small enough that
the final Int32 result fits. The raw record states the deterministic input
recipe. The report includes every variant and each case's lowest CPU/Metal
ratio. Ratios below one favor CPU; failures are never removed from the selected
case set.

Results default to `build/metal/results.json`, with the Markdown report beside
it and complete process stdout in a sibling directory. Re-render without
rerunning measurements:

```sh
python3 benchmarks/tools/bench_metal.py --report-from build/metal/results.json
```

Provider profile timings come from an additional warm collection after the timed
samples. Its GPU interval is not whole-query wall time. Shared-storage staging,
allocation, launch overhead, synchronization, result copying, and CPU executor
fusion choices can each explain a measured difference. Power and scheduling
noise affect short queries on the M1. These cases do not enable automatic GPU
selection; placement requires separate validated calibration.

External-suite reporting continues to follow [the benchmark rules](benchmarks.md).
Floating reductions remain outside this development matrix. The cases have
no independent oracle or holdout claim, and do not represent external-suite
coverage.

The driver records best-effort read-only pmset -g batt and pmset -g therm
snapshots in run metadata. Missing commands, nonzero statuses, and timeouts remain
explicitly unavailable; these snapshots provide operating context rather than a
controlled power measurement.

The retained initial smoke record includes three unsupported 16-repetition
single-expression cases: their 65 nodes exceeded the then-current shared 64-node
limit. The supported performance case uses 15 repetitions (61 nodes). All
other initial smoke cases passed. This is a planner bound, not an arithmetic
precision failure.

## M1 development run, 2026-10-10

The [complete report](https://github.com/randyzwitch/dataframe_mojo/blob/892b3626964719109af740703ee719b119bdc879/benchmarks/results/apple-metal-m1-20261010/results.md)
and [raw record](https://github.com/randyzwitch/dataframe_mojo/blob/892b3626964719109af740703ee719b119bdc879/benchmarks/results/apple-metal-m1-20261010/results.json)
retain 252 successful processes: seven workloads, three row counts, three data
variants, two native builds, and two rounds, with seven warm samples per engine
per process. The earlier [initial smoke record](https://github.com/randyzwitch/dataframe_mojo/blob/892b3626964719109af740703ee719b119bdc879/benchmarks/results/apple-metal-m1-20261010/initial-smoke.json)
retains the three unsupported 65-node expressions described above. The corrected
21-case smoke matrix and all full-run cases passed exact CPU-reference checks.

This Mac has an Apple M1, 16 GiB RAM, and eight physical CPU cores (four
performance and four efficiency cores). CPU comparisons use
`DATAFRAME_THREADS=8`; this is a fixed thread budget, not a best-of-thread-count
sweep. The first native runtime is commit `848ec39`; fusion is `1a4dfa9`.
Both use the same CPU engine and runner. Source, binary, native-build, compiler,
and SDK fingerprints are in the raw record. The benchmark working tree was
modified during measurement; the recorded source and driver hashes identify
exactly what ran.

At 8,388,608 rows, these are complete warm-collection medians for the base
variant, pooled across the two rounds. The last column gives each case's lowest
CPU/Metal ratio across all three variants.

| Development mechanism | CPU ms | Metal ms | Base CPU/Metal | Worst variant CPU/Metal |
|---|---:|---:|---:|---:|
| Simple affine projection | 17.181 | 13.227 | 1.30 | 1.30 |
| Single expression, 15 affine repetitions | 180.265 | 20.205 | 8.92 | 8.92 |
| 32 lazy projection stages | 532.382 | 25.675 | 20.74 | 19.70 |
| Half-selective filter then projection | 5.336 | 29.359 | 0.18 | 0.18 |
| Sparse filter then projection | 2.467 | 24.197 | 0.10 | 0.08 |
| Filter then count/length | 1.984 | 28.497 | 0.07 | 0.07 |
| Exact Int32 sum | 3.380 | 9.223 | 0.37 | 0.18 |

The projection chain improved from 118.172 ms in the first native runtime to
25.675 ms with fusion, about 4.6 times faster. Observed launches fell from 34 to
3, and shared-buffer payload fell from 1,387,266,741 bytes to 87,032,501 bytes.
It still performs one synchronization and returns a complete CPU result.

These measurements include the advantage of specialized GPU expression fusion
over the current CPU executor; they are not isolated arithmetic-throughput
comparisons. Filtering and the tested reductions favor CPU, sometimes strongly.
The full report includes smaller inputs, where startup and launch costs matter
more. First-use numbers remain separate from warm medians. There is no general
GPU-win claim and no automatic-selection threshold derived from this matrix.

## Filtered reductions and gather liveness, M1, 2026-10-10

The [filtering report](https://github.com/randyzwitch/dataframe_mojo/blob/892b3626964719109af740703ee719b119bdc879/benchmarks/results/apple-metal-filter-m1-20261010/results.md)
and [raw record](https://github.com/randyzwitch/dataframe_mojo/blob/892b3626964719109af740703ee719b119bdc879/benchmarks/results/apple-metal-filter-m1-20261010/results.json)
retain 72 successful processes: three workloads, two row counts, three data
variants, two native builds, and two rounds with seven warm samples per engine.
All measured outputs passed exact CPU-reference checks. The same optimized
runner and CPU engine compare the resident runtime before and after predicate
reductions and gather liveness; both native builds already include buffer reuse.
The [source manifest](https://github.com/randyzwitch/dataframe_mojo/blob/892b3626964719109af740703ee719b119bdc879/benchmarks/results/apple-metal-filter-m1-20261010/source-manifest.json)
identifies the modified engine and both native sources. Raw process stdout is
retained beside the report. First-use measurements remain separate in the JSON.

At 8,388,608 rows in the base variant, complete warm Metal count/length
collections fell from 22.919 ms to 10.957 ms, about 2.09 times faster. Launches
fell from eight to five, and requested shared buffers fell from 357.16 MiB to
165.03 MiB. Filtered integer sum fell from 16.706 ms to 10.923 ms, about
1.53 times faster, with shared buffers falling from 275.14 MiB to 123.02 MiB.
Omitting the predicate-only column from gathers removed one launch; its base
median fell from 30.546 ms to 28.948 ms.

CPU remained faster for every selected workload/row-count/variant median.
These changes reduce GPU overhead but do not justify automatic GPU selection
for the measured filtering cases. They are development measurements, not an
external-suite score or held-out calibration evidence.
