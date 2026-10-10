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

Cases cover a simple projection, a long single expression, a 32-stage projection
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
