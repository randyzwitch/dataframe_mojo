# Lazy queries

`frame.lazy()`, `scan_csv(path[, schema])` or `scan_parquet(path)` starts a `LazyFrame`. Its methods
(`filter`, `select`, `select_exprs`, `with_columns`, `group_by(...).agg(...)`,
`join`, `sort`, `slice`, `head`, `limit`, `unique`, `drop`) only record plan
nodes; nothing reads data until `collect()`. `fetch(n)` collects the first `n`
rows. `explain()` prints the optimized plan with execution annotations, root first; `explain(optimize=False)`
and `collect(optimize=False)` show and run the plan as written.

`collect`, `profile`, and `fetch` accept `engine="cpu"` (the default),
`engine="auto"`, or `engine="accel"`. Auto conservatively selects CPU.
CPU and auto need no accelerator dependency or device discovery.

For NVIDIA execution without vendor imports, build the optional distribution:

```sh
pixi run -e gpu package-nvidia
# Run from outside this source checkout, using the generated packages:
pixi run --manifest-path /path/to/repo/pixi.toml -e gpu \
  mojo run -I /path/to/repo/dist/nvidia /path/to/app.mojo
```

The application's imports remain `from dataframe import ...`; collection is
`query.collect(engine="accel")`. Both generated `.mojoc` packages must be on
the import path. The GPU environment supplies MAX; CPU installations do not
need it. The package builder registers the provider in a temporary copy and
never changes the checked-out CPU sources. Do not put the source checkout
before the optional distribution on the compiler's import path.

`DATAFRAME_ACCEL_DEVICE` selects a nonnegative CUDA-visible ordinal (default
0). Unsupported plans are rejected before context creation; absent devices
and invalid ordinals produce clear errors. Each default collection owns its
context reference until completion; SDK-managed allocation caches may outlive
that reference. `cpu` and `auto` do not initialize a GPU, even when
the optional distribution is installed. Auto remains CPU in this milestone.

An explicit runtime still overrides provider/device discovery and permits
context reuse, including when compiling directly from the source checkout:

```mojo
from dataframe_accel.nvidia import NvidiaRuntime

var runtime = NvidiaRuntime(device_id=0)
var query = frame.lazy().filter(col("x") > 0).select_exprs([
    (col("x") * 1.25).sum().alias("total"),
    col("x").count().alias("count"),
])
print(query.explain(engine="accel", accelerator=runtime))
var result = query.collect(engine="accel", accelerator=runtime)
```

The bounded GPU region accepts an in-memory, nonchunked frame whose columns
are Bool and one common numeric dtype, Float32 or Float64. It supports:

- `select`, `with_columns`, `drop`, repeated stable `filter`, and final
  `head`/`fetch`;
- row-local `+`, `-`, `*`, negation, all six comparisons, Boolean AND/OR/XOR/NOT,
  `is_null`, `is_not_null`, and `fill_null`;
- column/column and column/literal expressions, nullable typed literals, aliases,
  scalar broadcasting alongside row expressions, and nullable slices;
- terminal float `sum(min_count=...)` and float/Boolean `count` projections.

The shared expression binder resolves literals and validates types. Logical
Boolean outputs remain bit-packed Bool columns. No implicit mixed-dtype
arithmetic is added. Row arithmetic rounds in the input dtype; sums accumulate
in Float64 and return the input dtype, and counts use Int64. Parallel reduction
can change floating-point summation order. Boolean expressions use the CPU
executor's Kleene null semantics; null filter predicates discard their rows.

The resident path has explicit bounds: 64 source/intermediate value slots,
64 plan steps, and 64 nodes per expression. Slots are immutable within a
collection and include temporary predicates. Unsupported plans raise before
device submission; `explain` gives the capability reason. File scans, chunked
input, integer execution, mixed Float32/Float64 sources, division, scalar-only
select, intermediate slices, joins, grouping, and other reductions remain
unsupported. `engine="accel"` without an installed provider or explicit runtime
explains or raises the missing-provider error. Explicit acceleration never
falls back. `engine="auto"` still runs on CPU even when a runtime is passed.

A runtime selects one CUDA-visible device and can be reused across queries.
Each collection uploads its source once. Row expressions execute as bounded
programs, fusing nodes within each expression; columns and filter row counts
remain on device between steps. Stable compaction uses contiguous block scans,
a bounded block-offset pass, and gather into a separate matrix. One thread
owns each packed Boolean/validity output byte. Final collection downloads the
result into ordinary CPU-accessible columns; there is no resident-query cache.
The existing specialized scalar-filter/float-reduction path remains available
for its supported shapes. All GPU work completes before returning.
`profile(accelerator=runtime)` reports one observed fused region with executor
`nvidia`, source rows, result rows, and executor wall time including binding,
memory preflight, allocations, and transfers. It appends GPU diagnostic columns
to the existing CPU report columns: `device_id`, `device_name`, `upload_bytes`,
`download_bytes`, `workspace_bytes`, `device_output_bytes`,
`peak_requested_device_bytes`, `memory_budget_bytes`, `free_device_bytes`,
`free_device_after_execution_bytes`, `kernel_launches`, `synchronizations`,
`kernel_ms`, `initialization_ms`, and `boundaries`.

Kernel time uses CUDA events around the resident pipeline and final packing
or reductions (or each specialized two-kernel reduction). It includes the
inter-kernel gaps. Only `profile` enables those timers and their waits; ordinary
`collect` does not. `kernel_ms` is null in an untimed provider execution report.
Synchronization counts describe explicit library wait boundaries (including
profile timer waits), not every internal driver operation. Default-provider
context setup is reported separately as `initialization_ms`; an explicit
runtime was initialized by its caller and reports zero for that column. CPU
profiling retains its existing columns and behavior.

Before uploading, the GPU path checks requested input, workspace, and output
bytes against currently free device memory. `NvidiaRuntime(memory_limit_bytes=N)`
adds a query payload cap; `DATAFRAME_ACCEL_MEMORY_LIMIT=N` applies the same cap
to the registered provider. The setting is nonnegative bytes. Explicit runtime
configuration overrides provider environment settings. Unsupported plans and
insufficient estimated memory are rejected before upload. Auto still selects
CPU; allocation failures, kernel faults, and driver errors propagate without
CPU retry.

The estimate includes bitmap slice offsets and all simultaneously requested
query buffers. It excludes SDK allocator reservations and bookkeeping, so it
is not a hard process VRAM limit or an allocation guarantee. The pinned SDK
can reserve substantially more than a tiny query's payload (a 256 MiB arena
was observed locally). Free-memory snapshots include other GPU users and SDK
reservations; their difference is not an attributable peak-memory measurement.
`explain(engine="accel")` may create a temporary context to inspect device
memory, while CPU/auto explain never does. It lists device, budget, allocations,
and the upload → resident execution → download → host-result boundaries.
For the resident path, `workspace_bytes` includes both matrices, source bitmap
staging, descriptors, filter ranks, reduction partials and reusable packed
outputs; `device_output_bytes` is zero because output slots already belong to
the matrices. `upload_bytes` reports actual transfer traffic separately.
`download_bytes` in the profile reflects the actual filtered result size;
`explain` states that this size is data-dependent.

`DATAFRAME_EXECUTION_REPORT` stderr tracing currently applies only to CPU
execution; use `profile` or Nsight for this GPU path.

GPU lowering is independent of CPU scheduling and optimization; `optimize`
and `streaming` affect only CPU execution. `batch_size` is validated for all
engines but does not partition this GPU region. Default CPU `explain()` is
unchanged. Automatic placement, additional operations, and other accelerator
vendors remain work under [#528](https://github.com/randyzwitch/dataframe_mojo/issues/528).

`join(other, on, how)` joins on keys named alike on both sides;
`join(other, left_on=[...], right_on=[...], how, suffix, coalesce)` pairs
differently named keys (`o_custkey` with `c_custkey`), with the eager join's
output rules. `explain()` shows the pairs as `JOIN inner on o_custkey =
c_custkey`.

`collect(streaming=True, batch_size=65536)` is the default. Contiguous
row-wise operations run as ordered batches on a scoped worker pool. At most
one wave of batches (one per active worker) is in flight; workers return
results in source order, regardless of completion order. `streaming=False`
retains the materializing executor for comparison. Eager operations keep
their existing behavior.

The collected plan has the semantics documented in [semantics](semantics.md)
and [expressions](expressions.md). `collect_schema()` returns `name: dtype`
pairs by running with zero-row scans. CSV inference samples the usual rows,
but a schema probe does not decode data chunks. Explicit schemas therefore
validate field/plan structure without evaluating data values.

## Streaming and state

- CSV scans use the existing quote-aware scanner and typed decoder, with at
  most `batch_size` records in a job. Consumed mapped pages are unmapped after
  the wave has copied them, keeping resident input memory bounded on Linux
  and macOS. A single unusually large CSV record
  still requires memory for that record. Empty/unmappable sources retain the
  existing reader fallback; bounded input memory is guaranteed for regular
  mapped files, not arbitrary pipes.
- Parquet scans own an Arrow stream and decode one row group at a time, then
  slice it into batches. Temporary input memory also depends on the file's
  row-group size. EOF, early stop and errors release the stream exactly once.
- Filters, row-wise selects, with-columns, drop, explode and unnest operate
  within each batch. A select consisting only of scalar literals stays with
  the materializing evaluator so it produces one row for the entire input.
- Sum/count/mean/min/max/first/last, moments, boolean reductions and distinct
  counts use existing reducer states. Compound expressions over those
  reductions are evaluated after merging. Integer sums keep their exact
  128-bit state until final overflow checking. Floating-point reassociation
  follows the existing contract. Group representatives and state merges stay
  in first-occurrence order. Memory grows with groups and distinct-value
  state, rather than total input rows.
- Inner/left/semi/anti/cross joins retain the build side once and stream probe
  batches. Right/full joins, sort (except a sort cut short by a slice, which
  keeps each batch's first rows), unique, windows, median/quantile, implode
  and unsupported aggregate compositions retain materialization boundaries.
  Pipelines resume after these boundaries. Collect itself retains its final
  output, so collecting all rows is not a constant-memory operation.
- Positive slices preserve global row offsets and may stop the input after a
  wave satisfies the limit. Negative slices require the full result. Errors
  encountered in a wave propagate without exposing partial results; an early
  stop does not validate unread input. Batch size and scheduling can change
  how far beyond a requested head the reader evaluates.

`explain()` annotates streaming operators, aggregate state and materialization
boundaries; joins distinguish their probe and build behavior.
`explain(streaming=False)` retains the unannotated logical plan. A marker
reports the operator's execution capability: an upstream materialization
boundary still has to complete before downstream batches can run.

See [the streaming measurements](lazy-streaming.md) for peak RSS, timings,
raw samples and reproduction commands.

## Optimizations

- **Predicate pushdown.** A filter moves below a `with_columns` or `select` when
  it reads none of the columns that node produces (for `select`, only plain
  column selections), and into the side of a join that owns every column it
  reads: either side of an inner join, the left side of left, semi, and anti
  joins. A coalesced right key is not an output column, so no filter moves
  to the right side on its account. Row-local filters also move below a stable sort, since filtering the
  sorted rows preserves their relative order. Whole-column filters stay above
  the sort. A row-local filter directly above a CSV scan without a row limit
  runs on each decoded range before the ranges are assembled. Filters never
  move past a slice, unique, group_by, or right/full join, since that would
  change which rows those operators see.
- **Projection pushdown.** Scans read only the columns some operator above them
  uses; CSV scans pass them as `columns=`, so other fields are never decoded.
  A join passes each input only its own keys (the left keys to the left
  input, the right keys to the right) and the columns read above it from that
  side. Selectors, and joins with `coalesce=False`, keep every column. A plain column
  projection directly after a sort keeps the sorted row permutation and gathers
  only the selected output columns; sort-only key columns are not materialized
  in the result.
- **Slice pushdown.** `head(n)` moves below row-local `select`/`with_columns`
  (no reductions, windows, or `over`), and a leading slice directly over a CSV
  scan becomes the reader's `n_rows`, so the rest of the file is not read.
- **Top-k.** A slice directly above a sort (`sort(...).head(k)` or
  `slice(offset, k)`, SQL's `ORDER BY ... LIMIT`) needs only the sort's
  first `offset + k` rows, so the sort selects them instead of sorting every
  row, and `explain()` shows it as `TOP_K`. The result is exactly the full
  sort's rows, including stable ties and null and NaN placement. When the
  plan streams, each batch keeps its own first rows, so memory stays at a
  batch plus those rows. At a tenth of the input or more, the full sort is
  used and cut short.
- **Row-group pruning.** A row-local filter directly above a Parquet scan
  first reads the file's footer statistics (`parquet_row_group_statistics`)
  and decodes only the row groups whose minimum and maximum admit a match.
  Bounds are used for `col <op> literal` with `<`, `<=`, `>`, `>=`, `==` on
  numbers and strings, joined with `&` and `|`; any other predicate, a
  column without statistics, or a text bound the writer did not mark exact
  keeps every group. The filter still runs on the rows that were read, so
  pruning only changes how much is decoded, never the result.

Tests check that optimized and unoptimized plans produce identical results for
every rewrite. Tests also compare batch sizes, worker counts, joins and aggregate
state against the materializing executor, including failures and early cleanup.

The [accelerator execution contract](accelerator-contract.md) specifies the
shared capability, placement, ownership and readiness requirements for
discrete and unified memory, including the gates for future automatic placement.
