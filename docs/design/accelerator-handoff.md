# Handing work to an accelerator

How the CPU executor gives parts of a query to an accelerator, what the
first version does, and where hand-off should attach next.

## Two memory models, one contract

An accelerator either shares memory with the CPU (Apple silicon; AMD
APUs) or has its own (NVIDIA and AMD discrete GPUs). The difference
decides how work should be split:

- **Shared memory.** Moving a column to the device costs almost nothing:
  the device reads the CPU's buffer in place (`wrap_host_memory`). The CPU
  executor stays in charge of the query and hands individual operators
  over where the device supports the types natively and is measured to be
  faster. Types the device would have to emulate (Float64 on Apple GPUs,
  which have no 64-bit floating point) stay on the CPU.
- **Discrete memory.** Every crossing of the bus costs more than most
  operators save, so a provider keeps data on the device and runs whole
  queries or large regions of them, falling back at region boundaries.

The provider contract serves both. A discrete provider implements the
whole-plan functions (`execute`, `describe`, ...). A shared-memory
provider implements the hand-off functions:

- `handoff_supports(region)`: decided from the region's plan and its input
  types, before any input runs.
- `handoff_execute(region)`: the region's columns from the input columns.
- `handoff_estimate(region, rows)`: the operator's calibration key and the
  provider's estimated time for `rows` input rows, data movement included.

The core never imports a backend; `scripts/check_accelerator_boundary.sh`
enforces that for the library and for every test outside the backend's own.

## What the first version does

The hook is at plan nodes (`LazyFrame._handoff_region`). When the executor
reaches a projection or an aggregation, it offers the node with the
filters directly below it as a plan over the input's schema. If the
provider accepts it, the CPU computes the input, and then:

- with `engine="accel"` the provider always runs the node;
- with `engine="auto"` the executor compares the provider's estimate with
  the CPU's measured cost for the same operator (`dataframe/placement.mojo`:
  `fixed_ns + row_ns * rows`, fitted on this machine by
  `benchmarks/tools/fit_handoff.py` from a `bench_handoff` sweep) and runs
  the node where it is estimated faster. No row count is chosen by hand,
  so faster hardware moves the decision without code changes. Without a
  calibration nothing is probed.

`explain` prints a `PLACE` line per supported node with both cost models
and the row count from which the accelerator wins; profiles record each
decision with both estimates. Metal accepts one operator so far: a
filtered sum of an Int32, Int64 or Float32 column.

## Where hand-off should attach next: pipelines

The node-level hook works, but it sits beside the executor that matters
most. The streaming executor runs thread-owned pipelines
(`pipeline.mojo`): a source, row-local steps and a sink, over morsels,
with filters narrowing a selection instead of copying. A node handed off
today receives its input fully materialized, and a node `engine="auto"`
places on the CPU runs eagerly on those columns instead of in its
pipeline. Both cost time the pipeline would not spend.

The next form of the contract offers a pipeline, not a node:

1. **Unit.** A pipeline's source columns, its steps (filters, projections)
   and its sink (an aggregation, a hash-join build or probe, a top-k).
   This is DuckDB's unit of physical execution, and the steps are exactly
   what a GPU fuses into one kernel.
2. **Granularity.** On shared memory the whole source can go in one launch
   (morsels exist to spread work over CPU threads; a GPU's fixed launch
   cost of about 2 ms on an M1 makes CPU-sized morsels a poor unit).
   A provider could also take every Nth morsel, with the CPU running the
   rest, when both should work at once.
3. **Sink states.** The provider returns partial sink states in the CPU's
   own layout (aggregate partials, join build partitions), so the CPU's
   existing combine and finish steps produce the result. This keeps
   semantics, ordering and error decisions in one place.
4. **Costs** are per pipeline: source rows, the steps' work, and the sink
   kind, with data movement included for discrete devices. A discrete
   provider uses the same costs to group pipelines into device-resident
   regions.

Three pieces of CPU work bear directly on this, which is why the structure
settles first:

- what counts as one step or one fused projection (#1000);
- how pipelines meet at breakers, where serial tails now sit (#999);
- whether columns reach an operator as one chunk (#1003). The in-place read
  needs a single contiguous buffer, and a chunked column is copied.

## Measured limits on Apple GPUs (M1)

Measured with `bench_handoff` against the CPU's 8 threads:

- fixed cost of about 2 ms per hand-off, mostly the kernel launch and
  readback, with no compiled-kernel reuse yet;
- reading a column in place costs about 0.02 ms where copying it cost
  20-40 ms per 50 million rows;
- no 64-bit atomic add on any Apple GPU, so 64-bit sums use per-block
  partials; MAX's warp and block reductions and 32-bit atomics work, while
  its full-column reduction and its sorts are not safe on Apple GPUs.
