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
semantics. Dyadic data avoids an arbitrary float tolerance hiding reduction
order errors. This supplements, rather than replaces, the executor's separate
NaN/Inf, rounding, and lifetime tests.

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
