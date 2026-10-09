# NVIDIA GPU feasibility

This experiment supports [GPU execution epic #528](https://github.com/randyzwitch/dataframe_mojo/issues/528).
It exercises existing nullable column buffers on an NVIDIA GPU without adding
a backend, changing the public API, or changing CPU package dependencies.

The workload is `filter(x > 0)`, multiply the surviving values by `1.25`, and
return sum and count. The GPU implementation is handwritten for that query:
it fuses the predicate and multiplication into a partial reduction, then
reduces those partials in a second kernel. It does not yet implement stable
filter compaction, arbitrary expression lowering, or automatic placement.

## Local results

These measurements retain their original baseline below. The draft PR is
based on `main` at `9d9c8c0`, which has newer CPU execution changes. Rerun the
measurements after the pending CPU optimization stack (#518, #519, #523)
merges before treating them as a comparison with current main. All 118
correctness cases and a 1M-row smoke sweep also passed on `9d9c8c0`; that
smoke run is validation, not a replacement for the measurements below.

Measured on 2026-10-08 at repository baseline `eca182c` plus this experiment,
with the pinned toolchain described below, an NVIDIA GeForce RTX 5070 Ti (16,303 MiB,
compute capability 12.0, driver 610.57.04), and an AMD Threadripper 3970X
(32 physical cores). CPU worker selection was left at its default, with a
32-worker budget at 10M rows. Linux governor: `schedutil`; no affinity or GPU
clock pinning. No other benchmark or compilation was run concurrently with
the timing sweep.

All **118 correctness cases passed**, including CPU API comparisons and
bit-preserving GPU round trips. The existing `examples/expressions.mojo`
also passed in the default environment without the MAX accelerator library;
existing lockfile environment definitions are unchanged.

The final measured build took **205.95 seconds**. In the default sweep,
context discovery/setup took **361.81 ms**. First GPU query calls took
**0.367 ms for Float32** and **0.137 ms for Float64**, after ahead-of-time
compilation and excluding context setup. These are warm-driver observations,
not promises about an empty driver cache or an independent cold Float64 run.

Selected **mean wall times in milliseconds**, seven repetitions:

| Dtype | Rows | Variant | CPU lazy | GPU allocation + transfers + query | GPU resident + result download |
| --- | ---: | --- | ---: | ---: | ---: |
| Float32 | 1,000 | 10% nulls | 0.069 | 0.030 | 0.014 |
| Float32 | 1,000,000 | 10% nulls | 2.369 | 0.367 | 0.039 |
| Float64 | 1,000,000 | 10% nulls | 2.685 | 0.672 | 0.038 |
| Float32 | 10,000,000 | 10% nulls | 11.272 | 3.769 | 0.124 |
| Float64 | 10,000,000 | 10% nulls | 13.533 | 7.407 | 0.138 |
| Float64 | 10,000,000 | 33% nulls, low selectivity | 6.341 | 7.190 | 0.138 |

The 1M nullable Float32 case was approximately **6.45× faster** than the
existing CPU API even including allocation and transfers. Conversely, the
10M low-selectivity Float64 case was **slower** when the input had to be
uploaded. Retaining that same input on the GPU changed the comparison
substantially. A policy based only on row count would miss this distinction.

At 1,000 nullable Float32 rows, the diagnostic fused scalar CPU loop took
**0.003 ms**, faster than either the general CPU API or the GPU path. The
warm GPU win over the API at small sizes is not evidence that GPU hardware
is preferable to an equally fused CPU implementation. Context setup also
outweighs the savings for a small number of these queries.

Raw results are in `build/gpu-feasibility.csv`. These measurements justify
continuing the backend work, but do not set an automatic-selection threshold.

A follow-up with fifteen repetitions refined the low-selectivity Float64
case. At 7.5M rows it was effectively tied: CPU mean **5.500 ms**, GPU with
allocation/transfers mean **5.446 ms**. At 10M rows the repeat was noisy:
CPU best/mean **6.227/13.807 ms**, GPU best/mean **7.185/7.613 ms**. Thus the
repeat reversed the mean comparison even though the CPU's best time remained
lower. Do not treat the first run's crossover as stable. Raw follow-ups are
`build/gpu-feasibility-7500000.csv` and
`build/gpu-feasibility-10000000.csv`; the single-size command below reproduces
those fifteen-repetition sweeps. No hardware tuning or threshold was selected
from these results.

## Reproduce

```bash
pixi install -e gpu
pixi run -e gpu test-gpu-feasibility
/usr/bin/time -p pixi run -e gpu build-gpu-feasibility
pixi run -e gpu ./build/bench_gpu_feasibility > build/gpu-feasibility.csv

# A single size, both dtypes, all data variants, fifteen repetitions:
pixi run -e gpu ./build/bench_gpu_feasibility 1000000 15
```

The `gpu` feature supports Linux x86-64 for this experiment and pins
`max-core=26.7.0.dev2026092105` alongside
`mojo=1.2.0.dev2026092105`. `max-core` provides the accelerator library without
requiring the Python MAX API. Existing CPU environments and package host/run
dependencies are unchanged. GPU hardware/driver access is needed at build
and run time; sandboxed execution may need accelerator access.

Source: [bench_gpu_feasibility.mojo](../benchmarks/bench_gpu_feasibility.mojo).
The default sweep covers 1k, 10k, 100k, 250k, 500k, 1M, 5M, and 10M rows,
Float32 and Float64, with seven repetitions per case. Every invocation checks
correctness first; `--check` stops before the timing sweep.

## Semantics and memory

- Upload only the selected value window and the necessary validity bytes.
  Preserve the remaining bit offset instead of repacking the bitmap. Nullable
  benchmark slices begin at row 19, exercising both byte and bit offsets.
- Preserve null payload bits during round trips. Kernels test validity before
  reading/evaluating a payload. Null slots in generated inputs contain NaNs.
- Compare in the input dtype, multiply in that dtype, accumulate in Float64,
  and cast the sum back to the input dtype. This matches the current CPU sum
  implementation for both Float32 and Float64. Empty selections return zero
  for the default `sum(min_count=0)` and for count.
- Count is represented internally as Float64 in this bounded experiment; all
  measured counts are exactly representable. This is not a proposed general
  count-kernel representation. CPU result extraction also verifies the public
  sum and count dtypes; normalized pairs are only the benchmark's comparison
  format.
- Partial sums remain on the GPU. Only the final two scalars are downloaded.
  This fusion is valid for this aggregate query but does not implement a
  materialized filtered dataframe or establish ordering behavior.
- `GpuInput` retains the host column and owns the device buffers. All successful
  execution paths synchronize before reading results or releasing source
  storage. Production error-path lifetime handling and cross-stream ownership
  remain separate work.
- Inputs use existing pageable CPU allocations. There is no explicit pinned
  staging buffer, transfer/compute overlap, device allocation pool implemented
  by this experiment, or GPU-memory spilling.

Correctness covers empty and all-null inputs, absent and present validity
bitmaps, sliced offsets, sizes around byte/warp/block boundaries, NaNs,
infinities, signed zero, non-binary fractions, wide magnitudes, and product
overflow. Results are compared with the existing lazy CPU API and a fused
scalar CPU reference; payload round trips compare bits, including null slots.
Counts must match exactly. Finite sums use relative tolerance `1e-10` with an
absolute floor of `1e-10`; matching infinities and NaNs are handled explicitly.

## Timing methodology

CSV metadata records the device, context initialization, the first GPU query
for each dtype, and the CPU worker budget. Build time is measured separately;
first-query time includes runtime loading/warmup and is not presented as
compiler time. Float64's first query shares a context already warmed by the
Float32 checks, so these first-query figures are not independent cold starts.

Input generation, CPU lazy-plan construction, and answer verification are
outside the timed regions. `collect()` performs its normal planning and
execution. Each case warms both paths before interleaving CPU and GPU wall
measurements. The modes are:

| CSV mode | Included work |
| --- | --- |
| `cpu_lazy` | Existing lazy API `collect()`, with default optimization and streaming |
| `cpu_fused_scalar` | Fused scalar reference; diagnostic only, not an optimized parallel CPU implementation |
| `gpu_alloc_upload_query_download` | Fresh buffer handles/allocations, source upload, both kernels, result download, synchronization, and scope cleanup |
| `gpu_reuse_upload_query_download` | Same computation and transfers, with allocations reused |
| `gpu_resident_query_download` | Inputs already resident; both kernels, result download and host synchronization |
| `gpu_device` | Device event timing of both kernels and the stream gap between launches; no data transfers or host result access |

The fresh-allocation mode may benefit from driver/runtime caches after warmup;
it is not a cold device allocation benchmark. Device timings are taken in a
separate loop with result checks after each sample. They exclude most host
overhead and are not end-to-end latency. CSV reports best and mean times in
nanoseconds. Use means and losing cases when assessing placement; best values
alone are insufficient.

| Variant | Input distribution |
| --- | --- |
| `0` | Seeded shuffled values centered on zero, no bitmap, approximately 50% pass |
| `1` | Same value distribution, 10% nulls, slice offset 19, approximately 45% pass |
| `2` | Values shifted toward negative, approximately 33% nulls, slice offset 19, approximately 1.3% pass |

This compares a specialized fused GPU query with an existing general CPU
executor. Differences include algorithmic fusion and materialization costs,
not just hardware. The GPU path returns two host scalars and does not include
future planner or DataFrame result-wrapping costs. Repetitions reuse the same
input data and may benefit from CPU/GPU caches. It is a mechanism benchmark,
not a library speedup claim,
an automatic-placement threshold, or external-suite evidence.

## Toolchain findings

The pinned nightly successfully supports NVIDIA context creation and kernel
launch on the local RTX 5070 Ti. Kernel scalar arguments require fixed-width
types: ordinary Mojo `Int` and `Bool` are not accepted by the device-passing
path used here. The benchmark uses Int64 launch arguments.

The supplied `block.sum` Float64 reduction fails compilation through its warp
shuffle implementation (`unhandled shuffle dtype`). The experiment therefore
uses a simple shared-memory reduction tree with block barriers. It preserves
Float64 accumulation without changing the compiler or reducing precision.
This implementation is a correctness-first baseline, not a tuned reduction.

Float32 input does not make this complete query Apple-compatible: its sum
still accumulates in Float64. A future Apple backend needs a supported
equivalent or a planned CPU reduction. Capability checks must inspect
intermediate/accumulator types and final result conversion, not only input
column types.

## Implications for the backend design

The next implementation should introduce internal buffer ownership and
accessibility/readiness tracking, plus capability checks for whole expressions.
Keep those compatible with Apple's shared-memory model. Reuse the device
context across queries and account for first-use initialization. Explicitly account for
transfers, allocation reuse, launch overhead, reduction precision, and output
size before choosing GPU execution. Do not install a row-count threshold from
this single query.

General expression lowering, stable compaction, integer overflow semantics,
memory-pressure behavior, fault cleanup, Apple validation, and automatic
selection remain follow-up milestones. The public API is unchanged by this
experiment.
