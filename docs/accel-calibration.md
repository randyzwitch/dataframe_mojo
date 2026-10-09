# NVIDIA executor calibration (development evidence)

This benchmark measures the public executor added in #544–#546. It introduces
no kernels, execution shortcuts, or placement policy. `engine="auto"` still uses
CPU. These generated queries are development mechanism probes, not external
suite results or claims about general dataframe performance.

## Reproduce

Use the repository's pinned GPU and oracle environments. Build the optional
package from the engine revision being measured, then compile the consumer
outside the checkout so source imports cannot hide the registered provider:

```sh
pixi run -e gpu python3 scripts/build_accel_package.py --output dist/nvidia
pixi run -e gpu python3 scripts/bench_accel_calibration.py \
  --build --package-dir dist/nvidia --binary build/accel-calibration
pixi run -e oracle python3 scripts/bench_accel_calibration.py \
  --package-dir dist/nvidia --binary build/accel-calibration \
  --output build/accel-calibration.json --threads 1 --repetitions 7
python3 -m unittest discover -s scripts -p test_bench_accel_calibration.py
```

`--quick` reduces the grid to 64 cases while retaining every dimension; the
full grid has 432 cases. A changed benchmark source, binary, package manifest,
or package artifact invalidates the recorded build identity. Rebuild before
measuring. Compilation is outside timing. Run on an otherwise idle machine;
keep CPU thread count, power settings, and GPU settings fixed. The runner
records the initial process inventory, CPU governors, GPU/driver information,
SDK version, source and artifact checksums, git state, and thread settings.
Results on a shared desktop remain exploratory evidence.

## Coverage and correctness

The full grid crosses eight row counts (1,000 through 10,000,000), Float32 and
Float64, nominal null levels 0/10/50, predicate selectivities 1/50/99%, and
1/2/4 aggregate projections. Values are a permuted deterministic sequence of
signed dyadic fractions. Validity uses a separate modulus (101), so the nominal
null levels are thresholds out of 101, not exact percentages. Nullable columns
start at row offset 19, exercising an unaligned bitmap. Selection percentages
describe the value-domain predicate; actual selected counts depend on nulls
and finite sample size.

All projections share one input. Even-numbered projections multiply by
`1.25 + projection_index / 8` in the input dtype and sum with Float64
accumulation, then return the input dtype. Odd-numbered projections count
selected valid rows. Inputs and query plans are built outside timing. A scalar
oracle checks every materialized result after its timed interval; DuckDB
independently checks all case aggregates with explicit casts matching those
semantics. Checks use relative and absolute tolerances of `1e-10`; generated
dyadic values keep reduction-order variation small. Integer counts and the
Float32 sums in this grid cannot differ by a representable step within that
tolerance. This supplements the executor's separate NaN/Inf, rounding, and
lifetime tests.

Wrong answers, unsupported queries, process failures, and incomplete timing
records remain in JSON/CSV and make the runner exit unsuccessfully. Results
are checkpointed after every case; `planned_cases` distinguishes a partial run
from completed coverage. JSON includes every sample and process output; CSV
includes medians, minimum, maximum, and sample standard deviation. Both should
accompany any derived report.

## Timing boundaries

| Field | Measured boundary |
|---|---|
| `context_init` | First `NvidiaRuntime` construction in each new process, before GPU metadata lookup. Driver/system caches can already be warm. |
| `first_query` | First collect through that runtime, before query warmup; can include module/kernel loading. |
| `cpu` | Warm public CPU collect through full result materialization. |
| `gpu_fresh_handle` | Warm registered-provider collect, including a new runtime handle, binding/preflight, allocations, uploads, kernels, waits, download and CPU result construction. |
| `gpu_reused_handle` | Same public collect with an explicit reused runtime; allocations and uploads still happen for every query. |
| `kernel_interval` | Separate profile pass using CUDA events, summing per-projection intervals around two kernels. Includes launch gaps; excludes transfers, binding, allocation, result construction and context initialization. |

The three collect modes rotate order each repetition. The event-profile pass
runs separately because its extra waits would change end-to-end latency.
Profile counters describe that pass, including its extra synchronization per
projection. The default SDK allocator can cache reservations, so “fresh handle”
does not mean cold CUDA initialization or uncached device allocation. Kernel
time alone is never used as an end-to-end placement estimate.

## Conservative empirical cost evidence

For each exact dtype/null/selectivity/aggregate bucket and runtime mode, the
report compares the **maximum observed GPU collect time multiplied by 1.25**
against the **minimum observed CPU collect time**. It reports the first measured
row count passing that margin at that size and every larger measured size in
the same bucket. Any failed case in the larger tail blocks a recommendation.
No matching point means “not established,” not “GPU can never win.”

This is an empirical envelope over sampled costs, not a fitted universal row
threshold, a confidence interval, or permission to extrapolate between sizes.
Context startup and first query are reported separately and must be included
for a cold workload; the envelope only covers warm collections. Before any
future automatic policy, repeat across CPU thread budgets, devices, competing
loads, query shapes, and external development suites. Preserve the worst
perturbation beside aggregate speedup summaries. Held-out suite timing must
not be used to tune this model.

For a future repeated-workload model, an observed session cost can be written
as `I + F + (N - 1) * G`, where `I` and `F` are upper cost estimates for
initialization and the first query, `G` is the warm GPU envelope, and `N` is
the expected number of queries. Compare that with `N * C` for the CPU envelope.
This run records only one cold sample per case, so it does not establish
reliable cold-cost bounds. Workload reuse and uncertainty still need to be
estimated; the benchmark does not make that decision or assume that a caller
can amortize startup.

## Measured evidence: 2026-10-09

Engine revision: `ad437ac84b94649cfbf8e5946d308b5c5c17c002` (repaired #545,
including the diagnostics from #546 and CPU pipeline from #543).
Benchmark revision: `4b802536a03fe64e1be7ba1027eb63246e1e7098`.
Hardware: AMD Ryzen Threadripper 3970X (32 cores / 64 logical CPUs), NVIDIA
GeForce RTX 5070 Ti (16 GB), NVIDIA driver 615.71.09, Mojo
`1.2.0.dev2026092105 (e9569894)`, and DuckDB 1.5.5. CPU governors were
`performance`; the GPU used its default clocks and allocator.

These runs used a shared desktop, including browser/display GPU use;
concurrent CPU compilation was also observed during this work. Process
inventories are retained. These are exploratory mechanism measurements,
not isolated-machine performance or deployment thresholds. CPU thread
budgets must be interpreted separately.

| CPU threads | Per-case statistics | Raw samples, outputs, provenance and model evidence |
|---:|---|---|
| 1 | [CSV](accel-calibration-results/full-t1.csv) | [Compressed JSON](accel-calibration-results/full-t1.json.gz) |
| 8 | [CSV](accel-calibration-results/full-t8.csv) | [Compressed JSON](accel-calibration-results/full-t8.json.gz) |

Both 432-case sweeps passed every scalar and DuckDB check, with seven
repetitions per case. Compressed JSON is readable with `gzip -dc FILE.json.gz`;
it includes the complete planned grid, every result, both oracle values,
and source/package checksums.

Initialization and first query are separate from the warm timing tables:

| CPU threads | Context min / median / max (ms) | First query min / median / max (ms) |
|---:|---:|---:|
| 1 | 184.142 / 188.843 / 267.847 | 0.490 / 0.761 / 9.546 |
| 8 | 184.066 / 188.656 / 332.574 | 0.492 / 0.757 / 11.797 |

The following tables calculate CPU median / GPU median for each case, then
report the geometric mean and worst ratio across all 54 dtype/null/selectivity/
aggregate combinations at each size. Values above one favor GPU. The worst
perturbation stays visible beside each average.

With **1 CPU thread**:

| Rows | Fresh handle, geomean | Fresh handle, worst | Reused handle, geomean | Reused handle, worst |
|---:|---:|---:|---:|---:|
| 1,000 | 0.64× | 0.50× | 0.90× | 0.65× |
| 10,000 | 0.91× | 0.52× | 1.27× | 0.67× |
| 100,000 | 2.20× | 0.57× | 2.68× | 0.67× |
| 250,000 | 3.23× | 0.68× | 3.74× | 0.77× |
| 500,000 | 4.69× | 0.91× | 5.21× | 0.95× |
| 1,000,000 | 5.97× | 1.06× | 6.60× | 1.13× |
| 5,000,000 | 6.84× | 1.36× | 7.24× | 1.40× |
| 10,000,000 | 7.09× | 1.33× | 7.28× | 1.29× |

With **8 CPU threads**:

| Rows | Fresh handle, geomean | Fresh handle, worst | Reused handle, geomean | Reused handle, worst |
|---:|---:|---:|---:|---:|
| 1,000 | 3.12× | 2.27× | 4.80× | 3.10× |
| 10,000 | 3.27× | 2.24× | 4.92× | 3.06× |
| 100,000 | 3.61× | 1.75× | 4.60× | 2.11× |
| 250,000 | 3.39× | 1.68× | 3.98× | 1.89× |
| 500,000 | 3.11× | 1.29× | 3.63× | 1.40× |
| 1,000,000 | 2.62× | 0.93× | 2.96× | 1.04× |
| 5,000,000 | 1.56× | 0.47× | 1.60× | 0.47× |
| 10,000,000 | 1.37× | 0.41× | 1.40× | 0.42× |

The eight-thread results are not monotonic in row count. CPU collection starts
and joins its configured helper crew for each call (see
[`Crew.start`](../dataframe/parallel.mojo) and `LazyFrame.collect`), so small
queries include that setup cost. Large inputs can amortize it and benefit from
parallel execution, while GPU collection still uploads the input each time.
A future cost model therefore needs both fixed costs and throughput terms,
conditioned on the CPU thread budget; these observations do not support a
single “GPU above N rows” rule.

The empirical envelope first passes the 25% margin at the following measured
sizes. Counts describe the 54 query buckets for each thread budget/runtime
mode. “Not established” means no measured size passed at that and every larger
size; it does not rule out GPU gains in other conditions.

| CPU threads | Runtime handle | First passing measured size (bucket count) |
|---:|---|---|
| 1 | Fresh | 10,000 (9); 100,000 (24); 250,000 (3); 500,000 (8); 1,000,000 (4); 5,000,000 (5); not established (1) |
| 1 | Reused | 10,000 (16); 100,000 (20); 250,000 (3); 500,000 (6); 1,000,000 (4); 5,000,000 (3); not established (2) |
| 8 | Fresh | 1,000 (22); 10,000 (2); 100,000 (1); not established (29) |
| 8 | Reused | 1,000 (24); 10,000 (1); not established (29) |

The Float64 / nominal 50% null / 50% selection / four-aggregate case illustrates
the separation between CUDA events and full collection. All values are medians.

| CPU threads | Rows | CPU collect (ms) | Fresh GPU collect (ms) | Reused GPU collect (ms) | GPU event intervals (ms) |
|---:|---:|---:|---:|---:|---:|
| 1 | 1,000 | 0.161 | 0.247 | 0.187 | 0.053 |
| 1 | 1,000,000 | 3.868 | 0.750 | 0.711 | 0.142 |
| 1 | 10,000,000 | 40.107 | 7.340 | 7.248 | 0.950 |
| 8 | 1,000 | 0.404 | 0.167 | 0.123 | 0.038 |
| 8 | 1,000,000 | 1.366 | 0.809 | 0.730 | 0.144 |
| 8 | 10,000,000 | 7.601 | 8.172 | 8.104 | 0.901 |

At ten million rows this query uploads 81,250,001 bytes and downloads 64,
with 16,384 bytes of workspace, eight launches and thirteen profile wait
boundaries. Ordinary collection has nine wait boundaries; event profiling
adds one per projection. The kernel intervals exclude most host and transfer
costs, so they cannot stand in for collection latency.

These sampled cost envelopes do not change `engine="auto"`, which remains CPU.
Cold startup, CPU thread budget, query shape, memory limits and transfers all
remain necessary inputs to any future placement model.
