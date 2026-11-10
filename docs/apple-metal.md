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

**Supported:** fused global reductions and bounded integer-key grouped
reductions. Numeric value programs read Int32, Int64 and Float32 inputs;
Boolean, ordered and distinct reductions also read packed Bool columns.
Accepted expressions and predicates run within the reduction scan.

| Area | Accepted regions |
| --- | --- |
| Numeric reductions | SUM, COUNT, MEAN, MIN, MAX, VAR and STD |
| Values | Native numeric columns and casts, bounded arithmetic trees, unary numeric operations, zero-decimal rounding, clipping, supported division and remainder |
| Null-aware expressions | `fill_null`, Float32 `fill_nan`, `eq_missing` and `ne_missing` |
| Boolean and count reductions | ANY/ALL, global LEN, NULL_COUNT and Bool COUNT |
| Exact distinct counts | Native N_UNIQUE globally or within bounded integer-key groups |
| Ordered reductions | Native FIRST/LAST and ARG_MIN/ARG_MAX, globally or within bounded integer-key groups |
| Conditional values | Numeric `when/then/otherwise` chains feeding reductions |
| Filters | Literal ranges, same-dtype column comparisons, null tests, AND/OR/NOT |
| Grouping | Up to four native Int32/Int64 keys, at most 65,536 groups including null tuples |
| Input layout | Native nullable columns, slices and chunked inputs |
| CPU execution | Other input types, joins, sorting, windows, row deduplication and unsupported expressions |

Supported regions still require matching measurements before automatic
placement selects Metal. The existing filtered-SUM fast path compares a
numeric predicate column with numeric literals. The value and predicate
columns may differ; all nine native dtype pairs are supported, including
columns with different chunk boundaries. Results equal the CPU's: integer
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

Typed numeric and null-aware expression chains use scalar accumulator schedules
inside the existing reduction kernels. The planner records each operation's
arity; chains of up to three operations use fixed scalar steps, and longer
chains use a scalar loop. Integer chains feed statically assigned reduction fields directly; Float32
chains keep the shared field evaluator alongside their exact aggregate state.
This avoids indexed interpreter stacks while retaining
native rounding, null validity and eager checked intermediates. Nested right
subtrees and conditional expressions retain the general interpreter. These
kernels use the `typed_reductions:v2:scalar_chain` calibration revision.

A single SUM may reduce a binary `+`, `-` or `*` expression, for example
`(col("price") * col("quantity")).sum()`. Both columns must have the same
native dtype (Int32, Int64 or Float32). An operand may instead be a matching
typed literal or an exactly representable untyped numeric literal, on either
side. Arithmetic, the optional common filter and the reduction run in one GPU
scan without an intermediate result column. Operand nulls discard that row;
integer intermediate overflow is detected even if a final sum would cancel.
Overflow and non-finite Float32 partials use CPU adjudication. These shapes
have separate `arithmetic_sum:` calibration keys. Deeper arithmetic uses the
bounded tree path below. Explicit casts can reconcile native operand types;
integer true division and Float64 results remain on CPU. The general value
program also supports native numeric operations described below.
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
use `composed_reductions:v2:typed_fields:` calibration keys for the general
composition kernel; the compact all-Float32 kernel retains its v1 keys.
A schema-only remap groups Float32, Int32 and Int64 fields while preserving
output aliases and order. Native dtype branches and reduction states are
compiled for those groups, including Int64 accumulation for narrow integers
and split upper/lower words for Int64. COUNT-only fields omit unused sum
collectives while retaining checked integer expression evaluation.
Subnormal arithmetic operands
and possible gradual underflow use CPU adjudication because Metal arithmetic
can flush these values to zero before reduction. The existing single arithmetic
SUM and pure column aggregate paths keep their specialized kernels.

A projection with one arithmetic value and otherwise Float32 column
aggregates uses a compact typed kernel. It carries no integer reduction
state; a mean-bearing prefix uses exact bins, while independent trailing
sums use Float32 compensation. Output aliases retain their requested order.
Compact projections without a mean specialize by predicate dtype: Int32
comparisons stay native, and unused predicate branches and state are compiled
out. Exact-mean projections retain their existing traversal and predicate
path.
Supported composition regions stay fused before CPU common-expression
materialization. The pipeline adapter can also re-inline eligible shared
arithmetic values across row-local definitions; it declines unused checked
work and expressions evaluated before filters. These are schema-only
transformations in the Metal backend; the CPU optimizer is unchanged.

New fused projections have separate `fused_reductions:` calibration keys;
without a measurement they stay on CPU in automatic mode. Float64 input
columns continue to run on CPU.

A projection can also combine native `min` and `max` over up to four
distinct Int32, Int64 or Float32 columns, with the same optional range
predicate. Aliases and MIN/MAX on the same column share a scan and reduction
state. Empty or all-null inputs return typed nulls. Float32 extrema compare
ordered integer keys, preserving subnormals and infinities without floating
arithmetic on the GPU. NaNs are skipped when a numeric value exists;
all-NaN fields and extrema tied between opposite zero signs use CPU
adjudication to preserve the CPU's payload and tie behavior.
These native-column regions use `native_extrema:v1:` calibration keys and remain
on CPU in automatic mode until measured for the current build and layout.

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
a filtered column or coalescing either input. Predicates on multiple columns
use the Boolean program described below. Cross-column sums use distinct
`cross_filter_sum:<predicate dtype>:<value dtype>` calibration keys; the
single-column estimates are not reused for these larger memory reads.

Adjacent native slices that share both value and validity owners can share a
zero-copy mapping while retaining their original logical ranges. Composition
batches those ranges into one submission when every input has one shared
native span, preserving each segment's grid stride and partial-output order.
Separate allocations, gaps, reordered windows and different bitmaps keep their
separate ranges. Physical input layout still distinguishes calibration keys;
the changed Metal build identity invalidates earlier submission profiles.
Batched and ordinary compositions use separate compile-time kernel variants.
The ordinary variant removes the batch-table lookup and row-base pointer
arithmetic; both variants preserve the original per-segment reduction grids.

**Placement.** `engine="accel"` (or passing a `MetalRuntime`) hands Metal
every node it supports. `engine="auto"` places supported pipelines by measured
cost when `DATAFRAME_HANDOFF_CALIBRATION` names a calibration file, and
runs on the CPU otherwise:

Build the packages first so the measurements carry actual CPU and Metal
source/compiler identities. Build both the capture program and application
with `-O3` and the same compiler settings. A source checkout has no valid
calibration identity. Capture on the target Mac with the application's CPU
worker budget, and keep the packaged artifacts for the application:

```sh
python3 scripts/build_package.py --output dist
mojo build -O3 -I dist benchmarks/bench_metal_calibration.mojo -o build/metal-calibration
DATAFRAME_THREADS=8 build/metal-calibration 5 > build/metal-capture.jsonl
python3 scripts/metal_calibration.py build/metal-capture.jsonl --output-dir build/calibration
export DATAFRAME_HANDOFF_CALIBRATION=/absolute/path/printed/by/the/fitter.json
```

The built-in capture uses 50M, 75M and 100M rows, eleven representative operator
shapes, contiguous/eight-chunk inputs and independent nullable masks. Optional
row counts follow the repetition count. It checks answers against a periodic
scalar oracle, includes shared Float32 arithmetic SUM/COUNT/MEAN and mixed
Float32-mean bundles, mixed MIN/MAX/SUM/COUNT/MEAN projections, arithmetic
extrema and multi-column AND/OR filters. It alternates
CPU/Metal timing order and records unsupported or
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

Automatic collection uses the existing pipeline planner, which preserves
physical input chunks. Profiles store conservative CPU pipeline models under
`operators`, and layout-specific Metal models under `regions`. When several
measured queries share a CPU pipeline key, its CPU model is a lower envelope
across them, so pooling does not inflate CPU cost to favor Metal.
Unmeasured region layouts remain on CPU even when their pipeline key matches.

`explain(engine="auto")` shows actual placement in its `PIPELINE` lines.
The older `PLACE` lines describe a separate schema-only handoff mechanism;
versioned profiles do not supply costs to it. The current CPU `profile()`
entry point uses that older mechanism, so it does not audit pipeline
placement. To audit automatic collection, set
`DATAFRAME_METAL_PLACEMENT_TRACE` to a writable JSONL path. The Metal provider
appends a record after a pipeline finishes on Metal, or `cpu-adjudication`
when input values require CPU evaluation. With tracing unset it writes
nothing. Capture timings always use direct GPU completion checks instead.

Pipeline data faults retain CPU semantics: overflow, non-finite compensated
partials and gradual-underflow cases are adjudicated on CPU. Hardware and
allocation failures propagate as errors.

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
- **Visibility.** `explain(engine="auto")` shows pipeline placement; the
  opt-in JSONL trace records automatic GPU completion. Profiles with an
  explicit `MetalRuntime` show forced handoffs.

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
benchmark suites. Cases outside the accepted regions report `unsupported`.

### Property-matched placement costs

The fitter retains each operator/expression and physical layout's CPU and
Metal measurements as a matched pair. Layout keys distinguish bitmap
presence, chunk counts, approximate chunk proportions and whether adjacent
chunks share one native span. They contain no process-specific addresses.
Profiles apply only within their measured row range and matching build,
hardware and thread identities. Unknown inputs and unmeasured shapes stay
on CPU.

Automatic placement checks the CPU-optimized pipeline. If optimization adds
an intermediate that Metal cannot accept, that pipeline stays on CPU even
when the original expression can run as a forced Metal region. Forced GPU
execution and automatic placement are validated separately.

Each side uses an affine fit to per-scale sample medians. Negative-coefficient
boundary cases are refitted instead of clamping the intercept of an overly
steep line. The Metal decision estimate includes one switching margin: at
least 5%, increased by within-scale paired timing variation and by any
overstatement of GPU speedup at a measured scale. Typical Metal costs and the applied margin remain in the
profile for inspection. Thus small advantages prefer CPU, while one timing
outlier does not veto a clear repeated win.

The generic CPU pipeline model remains a fallback for other providers. Metal
supplies the validated CPU cost for the exact measured region through the
provider-neutral `pipeline_costs` hook, at both explain/planning and execution
time. CPU execution kernels and operators are unchanged. Legacy explicit
backends inherit their existing estimate and generic CPU costs.

### Mixed aggregate projections

MIN/MAX can share one scan with SUM, COUNT and MEAN, including supported
binary arithmetic expressions. Up to four distinct
aggregate values share the range predicate, input mappings and segment grids;
repeated outputs and aliases reuse their value's state. MIN/MAX accept native
Int32, Int64 and Float32 columns and one ADD, SUB or MUL with the same operand
and literal rules as arithmetic SUM. Extrema-only arithmetic also shares one
scan, without computing an unused sum. Checked integer intermediate overflow
and Float32 arithmetic requiring gradual underflow use CPU adjudication.

The mixed scan uses the existing typed integer accumulation and Float32
sum/mean policies. Extrema preserve raw Float32 bits, ignore NaNs when numeric
values exist, and adjudicate all-NaN or ambiguous signed-zero results on CPU.
COUNT includes non-null NaNs. Empty extrema/means remain typed nulls and SUM
retains its minimum-count and overflow behavior. Adjacent shared-owner chunks
batch into one launch; independent allocations retain their segment geometry.

When all outputs share one native Float32 value, a dedicated mixed kernel
reads that value directly, without binary-operand selection or integer
accumulator state. SUM/MEAN and extrema reuse its raw bits. It retains the
same predicate types, exact-bin retries, CPU-order proof and IEEE adjudication
as the general mixed scan, including copied and chunked inputs.

Mixed programs use `mixed_reductions:v1:` calibration keys. Existing pure
aggregate paths retain their keys and entry signatures. A changed package
identity requires fresh calibration before automatic placement uses the mixed
program; unmeasured shapes stay on CPU.

### Numeric predicates

Aggregate projections can combine predicates with `&`, `|` and `~`:

- Column-to-literal ranges use `>`, `>=`, `<`, `<=`, `==` and `!=` with
  exactly representable literals.
- Column-to-column comparisons support those six operators when both inputs
  have the same native Int32, Int64 or Float32 dtype.
- `is_null()` and `is_not_null()` read native numeric validity bitmaps.

Up to eight distinct leaves and nine total value/predicate input columns
share the aggregate scan. Separate filter calls compose as AND. NOT preserves
SQL unknown values; it is not implemented by complementing a leaf's is-true
bit after a null comparison. Float32 comparisons preserve NaNs, signed zero,
infinities and subnormals through ordered raw-bit comparisons.

Existing positive range programs retain `predicate_reductions:v1:` keys.
Null tests, negation, inequality and column comparisons use v2 keys describing
the complete predicate program. Existing single-range fast paths keep their
specialized kernels. No intermediate filtered table is materialized.

### Bounded arithmetic trees

SUM, COUNT, MEAN, MIN and MAX can reduce deeper same-dtype expressions, such
as `(col("price") * col("quantity") * (1 - col("discount"))).sum()`.
Programs contain at most 15 value nodes, with at most 8 stack entries,8 distinct
value inputs and 4 distinct aggregate values. Matching typed literals and
exactly representable untyped literals are accepted. Literal-only arithmetic
subtrees are declined when their intermediate binding could change dtype.

The GPU evaluates checked arithmetic and null validity at each node, then
reuses the root across requested aggregates. Valid integer intermediates
still report overflow when a later operand makes the root null. Filtered-out
rows skip value evaluation. Float32 arithmetic retains rounding at each node
and the existing gradual-underflow adjudication policy. Multi-stage row-local
shared definitions can be inlined when all definitions feed outputs and
filters precede them; grouping keys cannot be replaced by this transformation.
These programs use `expression_reductions:v1:` keys. Existing native and
one-binary kernels retain their specialized routes.

### Boolean and count reductions

Native Bool columns and supported numeric predicates can feed `any` and `all`.
Boolean values stay in the CPU column's packed bitmap; value-bit and validity-bit
offsets are independent. Shared slices map in place, and the explicit-copy path
concatenates packed bitmaps without expanding values to bytes.
Both the default ignore-null behavior and `ignore_nulls=False` Kleene behavior
use shared true-count, valid-count and row-count state in the GPU scan. Empty
ANY is false and empty ALL is true. With Kleene semantics, an unknown remains
null unless a true ANY or false ALL decides the result.

Unfiltered global projections of plain-column logical reductions use packed-word
counts. The GPU counts 32 value/validity bits at a time, shares true/valid counts
across repeated outputs and Boolean negations, and reads no payload for numeric
counts. All-valid counts use the known row count without scanning the input.
Filtered, grouped, derived-value and mixed numeric-aggregate projections retain
the general fused scan, including checked-expression errors.

Global `len`, `null_count` on native values, and Bool `count` use the same
state. They can share a projection with supported numeric aggregates and
bounded integer-key grouping. Existing grouped LEN retains its original route.
The existing limits of four distinct aggregate values and nine native input
bindings apply after lowering; shared Boolean state consumes one value and
row counts consume one more. The host only finishes result rows, with no input
payload conversion or Boolean mask materialization. New logical programs use
`logical_reductions:v1:` keys and require their own measured placement costs.

### Null-aware values and total comparisons

Native Int32, Int64 and Float32 aggregate expressions support `fill_null`.
Float32 expressions also support `fill_nan`, including a null replacement.
The replacement retains normal binding rules. `fill_nan` leaves input nulls
unchanged; a valid NaN becomes the replacement, whose validity determines the
result. COUNT-only projections still inspect NaN payloads when this can change
validity. Both operands evaluate eagerly, including checked integer arithmetic;
CASE continues to evaluate only its selected branch.

`eq_missing` and `ne_missing` support same-dtype native columns and exactly
representable literals, including typed nulls and Float32 NaNs. These predicates
always produce a valid Boolean: two nulls compare equal, one null compares
unequal, all NaN payloads compare equal, and signed zeros compare equal.
Their complements therefore include null rows when appropriate. They compose
with existing filter Boolean programs and CASE conditions.

### Conditional aggregate values

SUM, COUNT, MEAN, MIN and MAX can share a conditional value, for example
`when(col("quantity") > 0).then(col("price") * 2).otherwise(0)`.
Conditions use the same native numeric comparison, null-test and AND/OR/NOT
rules as filters; predicate and value columns may have different native dtypes.
All branches of one value retain the same Int32, Int64 or Float32 dtype.
Exactly representable literals, nested/chained CASE, constant Boolean
conditions and omitted ELSE (null) are supported. Supported arithmetic can
appear inside a branch or around a conditional value. Global filters run first.

Only the selected branch evaluates arithmetic. A null condition takes ELSE,
and overflow in an untaken branch does not fault. A selected checked fault
uses the existing CPU adjudication policy, including integer intermediates
in COUNT-only projections. Float32 reductions retain their existing exact-mean,
rounding, extrema and IEEE policies. Identical conditional values share their
computed value and reduction state across aliases and aggregate outputs.

Programs have at most 64 instructions, 8 conditions and 8 value-stack entries.
Up to 4 distinct values share 8 value inputs plus a ninth predicate binding.
The same values can feed bounded grouped aggregation. No conditional result
column is materialized. Shared row definitions are inlined only when doing so
preserves eager checked work: an earlier WITH_COLUMNS arithmetic expression
cannot be moved into a CASE branch that would suppress its errors.
Conditional programs use `conditional_reductions:v1:` calibration keys;
unmeasured shapes retain CPU placement.

### Numeric membership predicates

Numeric `is_in` and its negation can filter existing global/grouped reductions
and provide CASE conditions. The input is a native Int32, Int64 or Float32
column; candidates must bind to that dtype without losing precision. Up to
4,096 distinct candidate keys are supported. Duplicate candidates share one
key. Small sets use a linear lookup; larger sets use binary search over
sorted keys, without expanding the GPU predicate into one comparison per key.

Membership follows the engine's public API: null inputs remain null under
both membership and negation, so neither selects a null row in a filter.
Null candidates are ignored. Float32 signed zeros compare equal; NaN inputs
never match, including NaN candidates, and therefore pass negated membership
when non-null. Infinities and subnormals retain their native comparison
semantics through ordered integer keys. An empty set matches no non-null
value; its negation selects every non-null value.

Canonical candidate contents are included in predicate and conditional-value
identities. Membership uses `predicate_reductions:v3:` predicate keys and the
existing build-matched placement policy; new shapes require measurements
before automatic execution selects Metal.

### Bounded grouped reductions

Up to four native Int32/Int64 keys can group the same numeric aggregate
values, including CASE, membership conditions and bounded arithmetic trees.
Arbitrary signed values and independent nulls are supported in each key.
A tuple's null pattern participates in equality; null payload bytes do not.
The limit is 65,536 actual groups. Group existence and row counts include
rows whose aggregate values are null. LEN outputs use the existing row
counter, without consuming an aggregate value or an input binding. Four
values/eight value inputs and a ninth independent predicate can therefore
share a scan with four keys and row-count aliases. Empty inputs produce typed
empty columns; all-null groups preserve reduction and minimum-count rules.

GPU hashing uses immutable source-row tickets and 32-bit atomics. Ordinary
single-key, low-cardinality workloads retain per-workgroup partial states.
A bounded key sample chooses workgroup or global state for the general tuple
scan; higher cardinalities share one state per group rather than multiplying
scratch by the workgroup count. A sample cannot prove the cardinality limit:
if an initial table fills, the query retries on GPU with the full supported
table. Capacity and checked numeric faults have separate flags, so a retry
cannot hide an arithmetic error. Excessive cardinality beyond the supported
limit and checked data faults retain CPU adjudication.

Exact wide integer sums, Float32 exponent bins and winning row tickets for
Int64 extrema retain the existing numeric policies without 64-bit atomics.
Float32 sums/means require the CPU-order proof; non-finite sums and ambiguous
extrema use CPU adjudication. Device measurements reject adjudication as a
Metal timing. Scratch buffers share one bounded allocation; native key,
value and bitmap buffers remain separate mapped views. Independent chunk
allocations use an explicit upload when random ticket lookup requires one
logical span. Shared key/value/predicate mappings reuse their reservation.

The host merges GPU-discovered partials and orders groups by their first
qualifying row. Large results use radix ordering and one flat native column
per output, without per-group column allocations or retained input key
buffers. Grouped costs use `grouped_reductions:v3:local_updates:` identities, including
key types/layouts, programs, output work and bounded cardinality/skew/null
samples. Sampling describes cost only; GPU hashing computes every group.
Unmeasured shapes retain CPU placement.


Repeated-key grouped scans can batch row-count/first-occurrence updates and
reuse hash slots after comparing each actual key tuple. Int64 extrema cache
the evaluated value of an immutable winning-row ticket separately for every
field and extremum, avoiding repeated evaluation while that ticket remains
current. Numeric sums and value counts retain the existing exact GPU state.
Changing keys flushes row counts; changed tickets refresh their value cache.
Checked expressions still evaluate every selected row. Moment reductions retain
their existing scan. A bounded stride-reuse sample chooses this path; sampling
describes cost only and never discovers the complete groups.

### Native casts inside aggregate values

Explicit casts between Int32, Int64 and Float32 can share the existing
aggregate scan, including bounded arithmetic trees and integer-key grouping.
For example, `(col("quantity").cast(DataType.FLOAT32) * col("price")).sum()`
reads the original integer and Float32 buffers without a cast result column.
Arithmetic operands must have matching types after casting, as the CPU API
requires. Float64 input columns and Float64 cast targets remain on CPU.

Integer-to-Float32 conversion rounds to nearest, ties to even using integer
bit operations. Float32-to-integer conversion truncates toward zero and
checks the target range, infinities and NaNs. Strict invalid conversions use
CPU adjudication to preserve its error; non-strict conversions produce null.
Input nulls remain null. Checked integer operations retain their own native
width before a later widening cast, including overflow before a later null.

Count-only values omit Float32 loads and arithmetic when no Float32-to-integer
cast needs those values for validity or checked errors. Integer arithmetic and
range checks still execute. Required Float32 arithmetic retains intermediate
rounding and gradual-underflow adjudication. Row-local cast definitions can
be inlined under the existing definition-order and checked-work rules.

Typed programs have separate `typed_reductions:v2:scalar_chain:` identities.
Existing homogeneous arithmetic and proven simple CASE kernels keep their
specialized routes; nested typed trees use the general value path. Automatic placement requires
matching measurements for the current build, types and physical layout.

### Native numeric value programs

Fused aggregate values can use unary negation, absolute value, sign, floor,
ceiling, truncation and zero-decimal half-away rounding, plus lower/upper
clipping. These compose with the existing native casts, arithmetic and CASE
branches, and retain the same global/grouped reduction and predicate support.
Float32 rounding with a nonzero decimal count remains on CPU because its
decimal scaling uses Float64 arithmetic. Integer rounding retains the CPU's
identity behavior.

Int32 and Int64 floor division and remainder preserve the divisor's sign in
the remainder. A zero divisor makes that row null. The minimum signed value
divided by -1 retains the CPU's overflow outcome; modulo -1 is zero. Unary
negation and absolute value also preserve minimum-integer overflow checks,
including inside COUNT-only values. Untaken CASE branches remain lazy.

Float32 true/floor division uses integer significands and round-to-nearest-even
to reproduce native Float32 division, including subnormal operands/results,
without device Float64. Float32 remainder follows the CPU's rounded quotient,
fused recomposition and sign adjustment, rather than substituting a different
mathematical remainder. Non-finite operands, zero Float32 divisors and
underflow-sensitive remainder windows use explicit CPU adjudication. Bit-based
unary and clipping operations retain signed zeros, subnormals and NaN rules.

Programs retain the existing bounded typed-value format. Operation tags,
parameters and native per-node types distinguish calibration keys; unmeasured
shapes stay on CPU, and adjudicated executions cannot train GPU timings.
Grouped physical keys also distinguish the requested `maintain_order` policy:
CPU grouping may reorder its output when first-occurrence order is not
requested, and its cost must not be reused for an ordered workload.

### Variance and standard deviation

`var(ddof)` and `std(ddof)` can share native, expression, predicate and grouped
scans with the existing aggregates. Outputs are Float64; inputs remain native
Int32, Int64 or Float32. The GPU accumulates exact first and second moments,
and the host subtracts them exactly before Float64 finishing. Ordinary values
use narrow bins, with wider retries for extreme Float32 inputs.

Public results preserve CPU statistical behavior within the floating-point
tolerance `abs(metal-cpu) <= 1e-10*(1+abs(cpu))`. Int64 values first round to
Float64; Int32 and Float32 values convert exactly. Sensitive centering at
large offsets uses explicit CPU adjudication, preserving the behavior shared
with DuckDB and Polars rather than returning a materially different exact
answer. The internal mathematical finisher remains independently tested.
Large populations whose square sums exceed the Float64 exact-accumulation
span also use CPU adjudication, including some full-precision Float32 data
with zero mean. Quantized values can retain direct GPU completion.
Empty inputs and insufficient counts return typed nulls. Negative
`ddof` stays on CPU; non-finite values and checked expression faults retain
CPU adjudication. Float64 input columns remain on CPU.

The [numerical contract and error-bound derivation](design/metal-moments.md)
separate direct GPU accuracy from CPU adjudication. Moment queries use their
own build- and layout-matched calibration keys; unmeasured shapes remain on CPU.

### Native ordered reductions

`first`, `last`, `arg_min` and `arg_max` accept native Int32, Int64,
Float32 and Bool columns, globally or with the existing integer tuple keys
and fused filters. FIRST/LAST preserve the source dtype and payload;
`ignore_nulls=True` skips null values. ARG_MIN/MAX ignore null values,
return nullable UInt32 positions and select the first tie. Numbers precede
NaNs for extrema; an all-NaN input selects the first NaN. Signed zeros tie.

Positions refer to the selected rows within each group, after filtering.
An explicit `drop_nulls().arg_min()` or `drop_nulls().arg_max()` also removes
null positions. Empty global FIRST/LAST and ARG_MIN/MAX results are null.
FIRST/LAST return the chosen value without numeric conversion, preserving
Float32 NaN payloads, signed zeros and subnormals, and full-width Int64 values.

One scan shares up to four distinct `(column, drop_nulls)` states across the
ordered outputs. Inputs can have nullable, sliced or chunked native storage;
Bool values remain bit-packed. Grouping uses up to four Int32/Int64 keys and
at most 65,536 groups. Expressions other than a native column with optional
DROP_NULLS, and projections mixing these with other aggregate families, use
CPU execution. Automatic placement requires matching measurements for
`ordered_reductions:v1:` keys.

### Exact distinct counts

`n_unique()` accepts native Int32, Int64, Float32 and Bool columns, globally
or with the existing integer tuple keys and fused filters. Results are
non-null Int64 counts. The default counts null as one distinct value;
`ignore_nulls=True` and `drop_nulls().n_unique()` exclude it. All Float32 NaNs
share one class and both signed zeros share one class. Int64 keys retain
all their bits. An empty global input returns zero.

Up to four source columns share one scan; different null modes on the same
column share the exact hash state. The GPU stores original-row tickets and
compares native values and tuple keys on collisions. It does not convert
input payloads, construct a per-row group-ID array or estimate the result.
The host finishes only result rows.

GPU resources are bounded to 65,536 groups and 1,048,576 distinct
`(group, value)` pairs per source column. A table can grow once on GPU;
excess cardinality or the bounded probe limit triggers the existing checked
CPU fallback. Four maximum-capacity states use less than 40 MB of scratch.
Unsupported types, derived values and projections mixed with other aggregate
families use CPU execution.

Placement keys include the null modes and sampled cardinality. A mostly
unique sample cannot establish that the bounded table will fit, so automatic
placement retains CPU for `distinct_cardinality=uncertain` shapes. Other
shapes require matching measured costs for `distinct_reductions:v1:` keys.
