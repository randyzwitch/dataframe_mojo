# MAX CPU worker evaluation

The existing worker implementation is retained. This experiment evaluated
Mojo `1.3.0.dev2026111005` / MAX `26.8.0.dev2026111005` on an AMD Ryzen
Threadripper 3970X running Linux and an Apple M1 running macOS 26.5.2.
The baseline is core revision `3e008c08da7a4163f4249502c1b9cc7483f398d3`.

`max.algorithm.parallelize` imports and runs on both machines, but flushes
subnormal operands/results inside callbacks. Doubling the smallest positive
Float64 subnormal returned zero in all 17 jobs at thread budgets 2, 4 and 8.
The current runtime and `DeviceContext.cpu().enqueue_cpu_range` both returned
the correct bit pattern, 2, in every job. This rules out a direct replacement
with `parallelize` for general dataframe arithmetic on this pin.

The lower-level public CPU context was also prototyped for completed `Pool`
rounds. It retained the global helper reservation, static and dynamic job
assignment, per-job error capture and draining, produced-work workers, and the
query crew. It removed 66 net lines from `parallel.mojo`. The existing worker,
lifecycle and query-scheduler contracts passed, including nested work, exact
cleanup, first-error propagation, retry, and shared thread budgets. The
prototype is preserved as `pool-prototype.patch`; it is not production code.

Exploratory timings did not establish a general benefit. The context adapter
improved some fresh-pool cases, but warm rounds regressed substantially on the
M1. For 32 jobs of 4096 arithmetic iterations at four workers, static rounds
were roughly 57 microseconds with the current pool versus 260–280 microseconds
with the adapter; dynamic rounds were roughly 57 versus 107–127 microseconds.
Across two runs of the stable-sort workload, the median million-row,
eight-worker sort was 32.03 ms versus 39.55 ms on the M1 and 32.88 ms versus
33.95 ms on Linux. These are diagnostic measurements, not controlled
performance claims: other compilation was active on the hosts. Smaller sorts
sometimes favored the adapter.

After stopping compiler activity on the M1, two further runs in alternating
order (`mac-r3` and `mac-r4`) gave million-row medians of 26.24 ms for the
current pool and 31.76 ms for the adapter at eight workers, a 21.0% increase.
At two workers the medians were 51.15 versus 50.97 ms; at four they were 29.30
versus 29.68 ms. This confirms the eight-worker regression for this workload
without the earlier compiler contention. It does not establish a universal
ranking of the dispatchers.

The query dependency scheduler cannot be discarded. Its workers can wait for
other tasks, and the crew's helpable/unhelpable distinction and schedule stamps
prevent recursive wait cycles. A generic range dispatcher does not provide
that contract. The prototype deliberately left that scheduler in place.
Future work should first establish a public MAX dispatch contract that preserves
floating-point behavior, safely supports blocking/nested work, and improves
warm scheduling costs. No private MAX runtime API is used here.

The regression test in
`tests/execution_modes/test_worker_float_environment.mojo` now checks positive
and negative Float32/Float64 subnormals through serial, eager, crew, static-pool
and dynamic-pool paths, with repeated rounds at thread budgets 1, 2 and 4.
It passes on both machines. The final `execution.lifecycle` contract run with
the retained runtime passes all 31 tests (six storage and 25 execution),
including the new regression.

The adjacent `../max_metal_reductions.mojo` audit separately exercises the
unchanged Metal exact Float32 state and MAX block reductions. It passed on the
M1 with this pin at the default optimization level, covering cancellation that
retains subnormal contributions, null counts, nonfinite flags, and Int64 sums
above 2^53. It avoids query planning; a combined four-query-contract compile
was stopped after high memory use, and its optimization-level-0 retry was
killed with exit 137. No full Metal query-suite result is claimed here.

## Reproduce

Use the repository's pinned Pixi environment, from the repository root:

```sh
pixi run mojo run -I . benchmarks/experiments/max_cpu_workers/dispatch.mojo
pixi run mojo build -I . benchmarks/experiments/max_cpu_workers/pool.mojo -o /tmp/pool-base
pixi run mojo build -I . benchmarks/experiments/max_cpu_workers/sort.mojo -o /tmp/sort-base
```

`dispatch.mojo` compares mode 0 (current `run_jobs`), mode 1 (`parallelize`),
and mode 2 (public CPU context). It checks nested completion, helper budgets,
error propagation, destruction and subnormals before timing. Mode 1's observed
subnormal failures are printed rather than asserted as correct behavior.

To reproduce the pool prototype, use a separate checkout at the baseline
revision, apply `pool-prototype.patch`, then build `pool.mojo` and `sort.mojo`
against that checkout into different binaries. Run baseline and prototype
binaries in alternating order on an otherwise idle machine. Do not apply the
prototype to a production checkout. All benchmark results are checked outside
the timed sections; sorting checks the complete permutation and stable ties.

Raw samples are retained in the [measurement commit](https://github.com/randyzwitch/dataframe_mojo/tree/5fd07456b6cc677504254fcee250aafa77345cb9/benchmarks/experiments/max_cpu_workers),
following the repository rule that result dumps stay out of the current tree.
Pool columns are
`workers claim loops jobs fresh_pool repetition nanoseconds_per_round`.
Sort columns are `rows workers repetition nanoseconds`. Sort samples exclude
the first warmup. `base` denotes the unchanged implementation; `next` denotes
the preserved pool patch. `r2`/`r3`/`r4` are repeat process runs.
