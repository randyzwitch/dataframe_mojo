# Apple Silicon Metal backend

Metal runs inside the CPU executor. Apple silicon shares memory between the
CPU and the GPU, so the executor keeps the query and hands Metal the
operators it supports; everything else runs on the CPU in the same query.

```mojo
from dataframe_metal.metal import MetalRuntime

var plan = frame.lazy().filter(col("x") > 0).select(col("x").sum())
var forced = plan.collect(accelerator=MetalRuntime())  # every supported node
var placed = plan.collect()  # engine="auto": by measured cost, see below
print(plan.explain(accelerator=MetalRuntime()))  # HANDOFF metal: ...
```

**Supported so far:** `select(col(c).sum())` over filters that compare one
column `p` with untyped numeric literals (`>`, `>=`, `<`, `<=`, `==`, joined
by `&`). The value column `c` and predicate column `p` may differ; each can
independently be Int32, Int64 or Float32. All nine type pairs are supported,
including columns with different chunk boundaries. Results equal the CPU's: integer
sums are exact and raise the CPU's overflow error; Float32 sums use Kahan
compensation per GPU thread and can differ from the CPU in the last bits.
Native column chunks and their validity bitmaps are read in place, without
coalescing or copying their values. Each chunk retains its own value and
bit offsets. The reductions queue into separate partial-result ranges and
share one final synchronization and readback; chunk count does not multiply
the full reduction grid.

One projection can also combine `sum`, `count` and `mean` over up to four
distinct Int32, Int64 or Float32 columns, with the same optional range filter.
Aliases of the same input share their sum and non-null count. A single GPU
scan evaluates the predicate once and accumulates every requested field;
different input chunk boundaries are intersected into native views. Count
returns Int64, mean returns Float64, and sums preserve their input dtype and
`min_count`. Empty means are null; empty counts are zero. Integer means do
not overflow merely because a same-dtype sum would overflow.

Float32 means accumulate exact integer exponent bins on the GPU and finish
in Float64 on the host, without GPU Float64 arithmetic. If the finite input's
significant-bit span and count cannot prove the CPU's Float64 additions
exact, the CPU adjudicates the result in its own accumulation order. This
preserves the CPU's behavior for extreme dynamic ranges and cancellation.
Compensated Float32 sums with non-finite partials also use CPU adjudication,
including finite inputs whose GPU partials overflow Float32.
The common path uses three bins for values whose significant bits fit between
2**-53 and 2**43, such as ordinary prices and sensor readings. A device flag
detects any contribution outside that window and retries with all nine bins;
large finite values and subnormals retain the same supported semantics. The
kernel also specializes for the number of distinct aggregate fields.
For projections containing a Float32 mean, a schema-only remap packs Float32
fields into a prefix while preserving output order and aliases. Exact-bin
registers and block reductions are compiled only for that prefix; integer
fields keep their native integer reductions. These projections use the
`exact_float:typed_v1` calibration revision so earlier measurements are not
reused for the specialized kernels.

A single SUM may reduce a binary `+`, `-` or `*` expression, for example
`(col("price") * col("quantity")).sum()`. Both columns must have the same
native dtype (Int32, Int64 or Float32). An operand may instead be a matching
typed literal or an exactly representable untyped numeric literal, on either
side. Arithmetic, the optional common filter and the reduction run in one GPU
scan without an intermediate result column. Operand nulls discard that row;
integer intermediate overflow is detected even if a final sum would cancel.
Overflow and non-finite Float32 partials use CPU adjudication. These shapes
have separate `arithmetic_sum:` calibration keys; nested expressions,
division and mixed-dtype arithmetic remain on CPU.
Arithmetic expressions can also appear inside SUM, COUNT and MEAN in the
same projection as column aggregates, for example:

```mojo
frame.lazy().filter(col("p") > 0).select_exprs([
    (col("price") * col("quantity")).sum().alias("sales"),
    col("discount").sum().alias("discount"),
    col("quantity").count().alias("quantity_count"),
])
```

Up to four distinct aggregate values share one scan and range predicate.
Repeated SUM/COUNT/MEAN of an identical expression share its computed value
and reduction state; expressions using different operators or literals retain
separate states. Four binary expressions may read eight distinct value
columns plus an independent predicate column. Operands retain the same dtype,
literal and null rules as the single arithmetic SUM. COUNT checks integer
intermediate overflow even when no sum is requested. Float32 means use the
same exact-bin finish and CPU-order proof described above. These projections
use `composed_reductions:v1:` calibration keys. Subnormal arithmetic operands
and possible gradual underflow use CPU adjudication because Metal arithmetic
can flush these values to zero before reduction. The existing single arithmetic
SUM and pure column aggregate paths keep their specialized kernels.

New fused projections have separate `fused_reductions:` calibration keys;
without a measurement they stay on CPU in automatic mode. Float64 input
columns and other operators continue to run on CPU.

Float32 range predicates use ordered integer keys when a bound is near
zero, preserving subnormal comparisons, signed-zero equality and NaN
exclusion even when Metal's floating comparisons flush denormals. Normal
bounds retain native floating comparisons.

The explicit-copy reference path packs unaligned validity windows in bytes
and SIMD batches. It does not copy an entire unaligned bitmap one bit at a
time. These operations live in `dataframe_metal`; CPU storage helpers are
unchanged.

For example, a range on an Int32 quantity column can filter a Float32 revenue
sum. Predicate nulls discard rows; value nulls are skipped by the sum. The
columns' chunk boundaries are intersected into native views, without making
a filtered column or coalescing either input. A query with predicates on
multiple distinct columns remains on CPU. Cross-column sums use distinct
`cross_filter_sum:<predicate dtype>:<value dtype>` calibration keys; the
single-column estimates are not reused for these larger memory reads.

**Placement.** `engine="accel"` (or passing a `MetalRuntime`) hands Metal
every node it supports. `engine="auto"` places each such node by measured
cost when `DATAFRAME_HANDOFF_CALIBRATION` names a calibration file, and
runs on the CPU otherwise:

Build the packages first so the measurements carry actual CPU and Metal
source/compiler identities. A source checkout deliberately has no valid
calibration identity. Capture on the target Mac with the application's CPU
worker budget, and keep the packaged artifacts for the application:

```sh
python3 scripts/build_package.py --output dist
mojo build -O3 -I dist benchmarks/bench_metal_calibration.mojo -o build/metal-calibration
DATAFRAME_THREADS=8 build/metal-calibration 5 > build/metal-capture.jsonl
python3 scripts/metal_calibration.py build/metal-capture.jsonl --output-dir build/calibration
export DATAFRAME_HANDOFF_CALIBRATION=/absolute/path/printed/by/the/fitter.json
```

The built-in capture uses 50M, 75M and 100M rows, four representative operator
shapes, contiguous/eight-chunk inputs and independent nullable masks. Optional
row counts follow the repetition count. It checks answers against a periodic
scalar oracle, includes mixed Float32-mean bundles, alternates CPU/Metal timing order and records unsupported or
CPU-adjudicated cases explicitly. Every Metal timing requires direct GPU
execution. These are development calibration workloads, not external-suite
performance results.

For application workloads, call `dataframe_metal.calibration_capture.capture`
with a supported `LazyFrame`, independently computed expected `DataFrame`
and at least three repetitions; save its returned JSON as one line. Capture
at least three distinct sizes per query/layout key, and pass the JSONL file
to the same fitter. Check `capture_records` and `skipped` in the resulting
profile. Query constants, output work, chunk counts and bitmap presence are
part of the key; calibrating one layout does not measure another.

The fitter takes a lower CPU and upper Metal envelope over the observed
samples, adds a safety margin, and limits placement to each key's measured
row range. Its content-addressed output filename is immutable. Models are
empirical: changed selectivity or value distributions can change runtime
even for the same schema. Recalibrate representative data after changes.

Profiles must match both source/compiler build IDs, the device/host power
and thermal state, and the effective CPU worker budget. A CPU rewrite, new
Metal kernel, compiler/SDK update or thread change requires new measurements.
Legacy unversioned profiles, missing keys, out-of-range inputs and identity
mismatches stay on CPU. Replacing a file at an already cached path also stays
on CPU if its models differ from the cached ones. Use a new profile filename.
`dataframe_metal.calibration.calibration_status()` explains identity rejection
without constructing a GPU context.

`explain(engine="auto")` prints a `PLACE` line for measured supported nodes
with the CPU and Metal cost models. The displayed linear crossover is an
estimate; the provider also enforces the measured row range. Profiles record
each decision with both estimates.

### Historical measurements on an M1 Mac mini

These earlier, unversioned measurements are historical context and cannot
be loaded as current automatic-placement profiles.

`bench_handoff`, best of 5 warm runs from 10 thousand to 200 million rows,
against the CPU executor on 8 threads, with every Metal answer checked
against the CPU's. Each engine's times are fitted to a fixed cost plus a cost
per row; the table gives where the fits cross and what they give at 50
million rows (single runs at that size vary by a few milliseconds on this
machine). Metal's fixed cost is about 0.8 ms per hand-off; the column is read
in place.

| Operator | Metal estimated faster from | At 50 million rows (CPU / Metal) |
| --- | --- | --- |
| Int32 filtered sum | about 2.8 million rows | 23.2 ms / 9.4 ms |
| Float32 filtered sum | about 7.9 million rows | 14.0 ms / 9.7 ms |
| Int64 filtered sum | about 28 million rows | 17.1 ms / 16.5 ms |
| Float32 sum | never: the CPU runs near memory bandwidth | 4.5 ms / 9.4 ms |
| Int32 sum | never: the CPU runs near memory bandwidth | 4.6 ms / 9.8 ms |
| Int64 sum | never: 64-bit integer work is slow on the GPU | 8.1 ms / 17.9 ms |

The crossovers come from the fitted calibration; `engine="auto"` with it
keeps the plain sums on the CPU at every size. Until #1023, the CPU summed
Float32 one row at a time, and Metal looked faster from about 1.9 million
rows; that margin was the CPU's slow loop, not the GPU.

An earlier Metal core ran whole plans and was 30 to 250 times slower than
the CPU per row, at every data size measured. It stored every value as a 21-byte generic cell, ran
expressions through a per-row interpreter, emulated floating point in
software, and ran several kernels serially to reproduce the CPU's summation
order. It was removed rather than repaired. Its source remains in git history
at commit `d51540d4` (`dataframe_metal/`) for reference.

## What the package holds

- `dataframe_metal/metal.mojo`: `MetalRuntime`, `default_runtime()`,
  `metal_installed()` and the calibration entry points. `MetalRuntime`
  validates its arguments and owns a lazy device service. Constructing it
  does not create a GPU context.
- `dataframe_metal/provider.mojo`: the provider functions the core calls
  (docs/design/accelerator-contract.md), including the three for hand-off.
- `dataframe_metal/handoff.mojo`: which regions Metal accepts, their
  calibration keys and estimates, and their execution;
  `dataframe_metal/accelerator/gpu/filter_sum.mojo`: the kernels. The core never imports
  `dataframe_metal`; `scripts/check_accelerator_boundary.sh` enforces that,
  and also keeps tests outside a short Metal list from importing it.
- `dataframe_metal/accelerator/gpu/`: device detection (`target`,
  `identity`), the device service and per-query scope with byte budgets
  (`query_scope`, `scratch_pool`), and typed columns uploaded, read back or
  wrapped in place (`typed_storage`).
- `dataframe_metal/accelerator/faults.mojo`: the data-fault marker. When a
  kernel reports a data fault (overflow, a failed strict cast, an
  out-of-range temporal value), Metal reruns the query on the CPU and returns
  the CPU's outcome, since the CPU decides evaluation order. Profiles of such
  a query carry `fault_adjudication="cpu"`.

Building is unchanged. Metal is the default provider (`-I .`); on Linux
`installed()` is False. `-I providers/cpu -I .` compiles no GPU code.
`DATAFRAME_ACCEL_DEVICE` and `DATAFRAME_ACCEL_MEMORY_LIMIT` still configure
the default runtime.

## Design

On Apple silicon the CPU and the GPU share memory, so Metal is treated as
part of the Apple CPU engine rather than as a separate engine. The CPU
executor runs every query and hands individual operators to Metal when two
things hold: Metal supports the operator on the column types natively, and
measured costs on the device say it will be faster. Everything else stays on
the CPU, in the same query.

- **No emulated types.** Apple GPUs have no 64-bit floating point, and Metal
  has no 64-bit atomic add on any Apple GPU. Work on Float64 columns, or work
  that would need those atomics, runs on the CPU, which does it better than
  an emulation would.
- **Kernels.** Typed column buffers, fused and compiled kernels, and
  parallel reductions with no emulation of the CPU's evaluation order. Each
  operator first checks MAX's GPU kernel libraries (warp and block
  reductions, 32-bit atomics and `elementwise` run on Apple GPUs) before
  writing its own.
- **Visibility.** `explain` and `profile` show which operators ran on Metal.

The provider contract stays vendor-neutral: columns in and out, and the core
never imports a backend. A discrete GPU (NVIDIA, AMD) is a separate provider
that keeps data on the device and runs whole queries or large regions. The
same cost-based placement decides both, with the cost of moving data
included.

Coverage, measured against the CPU with `bench_handoff` and `report-apple`, is listed at the top of this page as
each operator lands.

## Validation

Hosted macOS runners cannot run Metal (their virtual GPU gives Metal buffers
no GPU address), so Metal contracts run on a local Apple silicon Mac:

```sh
scripts/test_metal_local.sh
```

`pixi run -e oracle report-apple` times CPU against forced Metal on the
benchmark suites. While no query shape is supported, every case reports
`unsupported`.
