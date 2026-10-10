# Measuring Apple Metal development mechanisms

`benchmarks/bench_metal.mojo` and `scripts/bench_metal.py` compare the existing
DataFrame CPU executor with the optional native Metal runtime on the same Mac.
These are development mechanisms, not an external suite or evidence of general
performance. `chain32` deliberately measures fusion across lazy projection
stages; its ratio includes the current CPU executor's per-stage overhead.

Build the native bridge, then run from an Apple Silicon checkout:

```sh
pixi run build-dfmetal
pixi run python3 scripts/bench_metal.py
```

The driver builds one optimized Mojo binary. For a quick correctness check:

```sh
pixi run python3 scripts/bench_metal.py --rows 4097 --reps 2 --rounds 1
```

Optional `--baseline-library /path/to/older/libdfmetal.dylib` compares two native
builds with that same binary and CPU executor. Runtime build fingerprints and
library SHA-256 hashes identify both. This isolates native runtime changes; it
is not a comparison of different CPU engine revisions. `--binary PATH
--skip-build` reuses an already built runner. The driver records compiler
version and flags, source/driver/binary hashes, engine commit and status,
selected thread count, Mac hardware/OS/SDK details, native profiles, load before
each process, raw samples, and failed or unsupported outcomes.

Each workload, variant, runtime build, and round runs in a fresh process.
First-use CPU and Metal collections are recorded separately. Two additional
warmups precede alternating CPU/Metal samples. The initial engine order and
runtime-build order rotate between rounds; case order reverses. First use
includes device/context/shader setup, but the machine and filesystem are not
cold. The input is constructed once outside timing; every collection allocates
and materializes a complete ordinary CPU result. Validation and result release
are outside the measured interval. Every measured result is checked exactly
against the current DataFrame CPU result. These checks supplement the native
ABI precision and overflow suite; they are not an independent SQL oracle.

Cases cover a simple projection, a 15-repetition single expression, a 32-stage projection
chain, filters retaining about half or one percent of rows, count/length after a
filter, and exact Int32 sum. Base, sorted, and nullable/permuted distributions
all run by default. Float32 inputs are exact multiples of 1/1024. Int32 sum
inputs stay small enough that the final Int32 result fits. The raw record states
the deterministic input recipe. The report includes every variant and each
case's lowest CPU/Metal ratio. Ratios below one favor CPU; failures are never
removed from the selected case set.

Results default to `build/metal/results.json`, with the Markdown report beside
it and complete process stdout in a sibling directory. Re-render without
rerunning measurements:

```sh
python3 scripts/bench_metal.py --report-from build/metal/results.json
```

Native profile timings come from an additional warm collection after the timed
samples. Its GPU interval is not whole-query wall time. Shared-storage staging,
allocation, launch overhead, synchronization, result copying, and CPU executor
fusion choices can each explain a measured difference. Power and scheduling
noise affect short queries on the M1. No threshold for automatic GPU selection
is derived from these cases; automatic execution remains on CPU.

External-suite reporting continues to follow [the benchmark rules](benchmarks.md).
Joins, group-by, Float64 and floating reductions are outside this development
matrix. No external-suite coverage or holdout claim is made here.

The retained initial smoke record includes three unsupported 16-repetition
single-expression cases: their 65 nodes exceed the unchanged shared 64-node
limit. The supported performance case uses 15 repetitions (61 nodes). All
other initial smoke cases passed. This is a planner bound, not an arithmetic
precision failure.

## M1 development run, 2026-10-10

The [complete report](../benchmarks/results/apple-metal-m1-20261010/results.md)
and [raw record](../benchmarks/results/apple-metal-m1-20261010/results.json)
retain 252 successful processes: seven workloads, three row counts, three data
variants, two native builds, and two rounds, with seven warm samples per engine
per process. The earlier [initial smoke record](../benchmarks/results/apple-metal-m1-20261010/initial-smoke.json)
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
