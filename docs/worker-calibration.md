# Worker-count and machine calibration (#273)

## Machines and method

Measured 2026-09-26 with Mojo 1.2.0.dev2026092105 (e9569894):

- Linux x86-64, AMD Threadripper 3970X, 32 physical/64 logical CPUs,
  512 KiB L2/core and eight 16 MiB L3 domains, schedutil, unpinned affinity.
- Darwin 25.5 arm64, Apple M1 Mac mini, four performance and four efficiency
  cores, 16 GiB RAM, 12 MiB performance-cluster and 4 MiB efficiency-cluster
  L2. Eight is the physical count; sixteen configured workers oversubscribe
  this machine and intentionally test that case.

The unchanged overall and random/skewed join benchmarks run at 4, 8 and 16
configured workers on **both** machines, twice with worker order reversed.
Overall workloads use 1k/100k/1M rows and five repetitions; join shapes use
200k×2k, 2k×200k and 100k×100k with three repetitions. The baseline overall
harness prints a stale `mojo=1.1.0` literal; the actual compiler is the 1.2
nightly above. It is not a measurement of Mojo 1.1.

Separate experiments vary spin counts, minimum rows/worker, index-builder
caps, gather parts and physical chunk counts. These use isolated source
copies from `a2b045c` (PR #288). Knobs are read once per worker or stage,
outside hot loops; comparisons use the same experimental binary. Inputs and
correctness checks are untimed. Each microbenchmark warms once and records
five repetitions in each of two rounds, reversing setting order. Tables below
report the mean of per-round minima in milliseconds. All samples and macro
benchmark aggregates are in [worker-calibration.csv](worker-calibration.csv).
`source` identifies the suite, configured worker count, setting and/or round;
settings are caps, not promises that every worker is used.

Linux shares the host with other development work. The driver waits for
Mojo processes to exit and discards a case if it detects a new one after the
run (`BENCH_WAIT_FOR_COMPILERS=1`). This excludes detected compiler overlap,
not all possible background activity. Equivalent capped settings sometimes
vary, so small differences are not treated as precise crossovers. Mac timing
ran separately from its compiler jobs. Neither machine had pinned affinity.

## Decisions

### Pool spinning follows usable cores

Keep the existing two-million-spin ceiling when the pool fits in usable
cores. Park immediately when it exceeds them. Linux uses physical cores;
macOS uses `hw.perflevel0.physicalcpu` when available, with a physical-core
fallback. The decision is made once before creating pool threads, so workers
read immutable policy state. A partial thread-creation failure remains safe:
parking is conservative even if fewer workers actually start.

The Threadripper still benefits from spinning: at 16 workers, a 1M-row
four-word merge sort takes 48.02 ms with no spin, 47.69 at 200k spins,
39.40 at 1M and 38.30 at 2M. M1 behaves differently: four workers can
benefit, but spinning competes with work at eight and especially sixteen.
The exact production-policy comparison on M1 is:

| Configured workers | Rows | Rank words | Baseline | Core-aware policy |
|---:|---:|---:|---:|---:|
| 4 | 100k | 2 | 3.63 | 3.67 |
| 4 | 1M | 2 | 50.62 | 50.55 |
| 8 | 100k | 2 | 3.86 | 2.79 |
| 8 | 1M | 2 | 47.56 | 41.60 |
| 16 | 100k | 2 | 26.55 | 3.15 |
| 16 | 1M | 2 | 67.37 | 46.91 |

### Bound builders and gather partitions by cores

The bounded-index cap remains 16, additionally limited by physical cores and
`worker_count(rows)`. On Threadripper at 16 configured workers, a 4M-key
build improves from 99.27/89.98/86.01 ms with caps 4/8/16. On M1 at sixteen
configured workers, a 1M-key build takes 12.56 ms at cap 8 and 15.09 at cap
16 (15 effective workers because of the row floor). Oversubscribing a
memory-bound build does not earn its extra workers here.

Sorted-gather partitions remain capped at four, but available partition jobs
now use the smaller of configured workers and physical/performance cores,
divided by column count. On M1 with eight configured workers, two columns,
16 chunks and 1M selected rows, two parts take 2.07 ms versus four parts'
2.46 ms. With eight columns, one job per column is already enough: at sixteen
configured workers, splitting is slightly slower than the unsplit path.
This changes a resource budget rather than hard-coding an M1 row threshold.

Keep the 16-chunk guard and give its three call sites one named constant.
Four chunks win in several M1 cases but lose on Threadripper: at four workers,
250k selected rows and two columns, generic gather takes 2.37 ms versus
2.88 ms for sorted gather; with sixteen chunks, the partitioned sorted path
takes 1.25 ms versus generic's 2.03 ms. Width, selection and locality matter;
there is no cross-machine evidence for removing the guard globally.

### Keep the shared rows-per-worker floor

The 64k minimum remains shared across operations. On M1, lowering it to 16k
improves 100k-row arithmetic/compare/filter (roughly 0.74/0.55/1.21 →
0.35/0.24/0.48 ms), but cheap counts worsen by 5–7×. At sixteen configured
workers, 100k-row high/skewed grouping also worsens by about 4×. A global
reduction would trade substantial regressions for expression wins; tuning
those operations independently needs separate evidence.

The fixed ceilings and floors now have machine/worker evidence next to them.
This is calibration on two machines, not a universal model of heterogeneous
CPUs or an instruction to use sixteen workers on an eight-core Mac.

## Reproduce

```bash
pixi run python3 scripts/bench_workers.py --build
BENCH_WAIT_FOR_COMPILERS=1 pixi run python3 scripts/bench_workers.py --scaling
for threads in 4 8 16; do
  for group in spin builders gather minrows; do
    BENCH_WAIT_FOR_COMPILERS=1 pixi run python3 scripts/bench_workers.py --run "$group" --threads "$threads"
  done
done
pixi run python3 scripts/bench_workers.py --sort-policy
```

Run the same commands on the second machine. To reproduce the combined
artifact, put its CSVs in `build/workers/mac`, then run:

```bash
pixi run python3 scripts/bench_workers.py --report docs/worker-calibration.csv
```

Pool tests cover multi-round reuse, ordered errors, dynamic/produced jobs,
teardown, and an oversubscribed parked pool that continues after a worker
error. Linux also validates chunked consumers and integer-range joins; the
Mac pool tests exercise the actual sysctl-based policy.
