# Apple Silicon Metal backend

`MetalRuntime` runs Mojo kernels in `dataframe/accelerator/gpu/` through the
Mojo GPU APIs supplied by MAX. Allocation, transfers, launches, synchronization,
query orchestration and intermediate ownership are implemented in Mojo inside
the dataframe library. GPU execution requires no repository dylib, C ABI bridge,
Metal source compiler or separate GPU library. Parquet file decoding uses the
existing CPU decoder in `native/dfparquet`.

The pinned `max-core` SDK is a normal build and package dependency, including
CPU environments. Importing dataframe, constructing a runtime, explaining a
query or explicitly collecting on the CPU does not construct a GPU context.
Mojo itself may load the system Metal framework independently of dataframe.

## Using the provider

```mojo
from dataframe.metal import MetalRuntime

var runtime = MetalRuntime(memory_limit_bytes=256 * 1024 * 1024)
var result = query.collect(engine="accel", accelerator=runtime)
var measured = query.profile(engine="accel", accelerator=runtime)
```

The default Apple accelerator uses the same implementation. Runtime copies
retain configuration; query execution owns its context and allocations.
`DATAFRAME_ACCEL_DEVICE` and `DATAFRAME_ACCEL_MEMORY_LIMIT` configure default
collections. `DATAFRAME_METAL_LIBRARY` is unnecessary and does not select the
implementation. `MetalRuntime.device_count()` explicitly discovers devices;
`metal_installed()` reports compilation-target support without initializing one.
The Metal provider selects the Metal device API explicitly.

Forced accelerator queries reject unsupported work rather than retrying on the
CPU. The complete reachable plan is bound before device initialization. Queries
with `engine="cpu"` ignore accelerator policy and device configuration.

## Operations and precision

The existing lazy expression and plan APIs are preserved. Supported work
includes the following operations and their valid compositions:

| Area | Mojo execution |
| --- | --- |
| Storage | Boolean, all eight integer widths, Float32/64, Decimal32/64/128, Date/Datetime/Duration/Time, nullable/sliced/chunked inputs, empty rows and columns |
| Sources | CSV and Parquet scans with decoder projection and reader head pushdown; decoded columns feed GPU operators |
| Expressions | Checked arithmetic, integer floor division/modulo/power, comparisons, Kleene Boolean operations, conditionals, null/NaN handling, clipping, membership, numeric and decimal casts |
| Floating point | Arithmetic, square root, rounding, exponentials, logarithms and power with CPU-compatible special values and subnormal handling |
| Reductions | Sum, mean, counts, min/max, first/last, arg min/max, Boolean any/all, variance/std, skewness/kurtosis, covariance/correlation, median and five quantile interpolation modes |
| Distributions | Scalar mode and value counts, multiple distributions with equal cardinality, sorted typed winners, nulls and ordinary scalar siblings |
| Rows and keys | Stable filters, slices, mixed-key sorting, grouped aggregation, distinct rows/counts, unions and rank methods |
| Windows | Partition aggregates, cumulative operations, shift/fill, fixed-size rolling operations, interpolation and coordinate interpolation |
| Duration windows | Numeric/Date/Datetime indices, supported closed bounds, fixed/calendar intervals, UTC and fixed-offset zones |
| Temporal expressions | UTC, fixed-offset and IANA local fields; fixed/calendar offsets and truncation, duration totals, timezone metadata conversion, bounded UTC replacement; direct temporal arithmetic and physical casts |
| Joins | Inner, left, right, full, semi, anti and cross; typed composite keys, null-equal left joins and key coalescing; backward/forward/nearest as-of joins with typed by-keys and tolerance |
| Composition | Selectors, scalar projections/broadcasts, aggregate expressions, computed join branches, post-join consumers and chained joins |
| Text | Ordered dictionary transport, supported string operations/concatenation, categorical reconciliation and restoration |
| Struct fields | Direct source Struct projection over fixed-width primitive children, parent/child validity and field metadata; terminal `as_struct` over primitive fields |

Float64 values retain their exact storage words. Arithmetic uses paired-word
integer IEEE emulation because Apple GPUs lack native binary64 arithmetic.
Float32 operations preserve separate operator rounding and special-value
semantics. This code implements dataframe precision requirements on top of
Mojo's ordinary integer and floating operations.

Int64/UInt64 sums and means use exact paired-word accumulation before checking
the typed result. Decimal arithmetic likewise uses paired UInt64 words rather
than device Int128 instructions. Decimal sums/means preserve precision checks,
ordered worker-prefix behavior, cancellation and scaled conversion semantics.
Moving, sorting, ranking, joining and arg min/max on decimals retain both words.

Sorts preserve source order on ties, configured per-key null placement, and
original value encodings. Nulls, NaNs and signed zeros retain the CPU grouping
and join equivalence rules. Rank direction follows the CPU rank contract,
including its NaN ordering. Scalar floating sums use a 256-lane reduction tree
and can differ in rounding from eager or streaming CPU sums. Statistical states
preserve contiguous worker ranges and ordered merges. Partition reductions
reconstruct CPU dense/hash group order and whole-group worker runs, including
skewed partitions and original categorical-code versus decoded-string keys.

Calendar helpers reuse `dataframe/calendar.mojo` on the GPU, including
proleptic Gregorian behavior across the supported storage domain. Duration
totals truncate coarser units toward zero and retain fixed-width multiplication
for finer units. Timezone conversion is a physical identity with validated
destination metadata, including IANA names. Fixed-offset local fields use the
CPU's wrapped local ticks. IANA local fields serialize the CPU timezone's
transition table and POSIX footer once per distinct zone, then select offsets
and extract fields on the GPU. Transition endpoints, historical offsets and
footer rules follow the CPU lookup; table owners and uploads count toward the
query budget. Replacement is limited to naive/UTC values and preserves the CPU's checked
reconstruction behavior.

Temporal arithmetic and physical casts currently require a direct in-memory
scan and terminal row projection without a final head. Operand and predicate
checks reject compositions that could introduce a different fault category or
change error order. Duration multiplication and subtraction retain valid
Int64 minimum results; temporal casts honor conditional masks and strict/lax
conversion semantics. This bounded subset does not enable arbitrary temporal
consumers across resident regions.

Shared Mojo normalization expands selectors and decomposes supported aggregate,
partition, distinct and union operations before GPU dispatch. String operations
build bounded host dictionary lookup tables once per distinct value, then apply
those tables through GPU row/join operations. Profiles report dictionary staging
and restoration separately; they do not imply that string transformation code
executes on the GPU.

## Residency, errors and memory

Computed query regions retain immutable Mojo-owned device tables. A consumer
copies selected columns into its own matrix on the device, so shared branches
cannot overwrite one another. Owners release after their last consumer. Only
the final result exports ordinary CPU-owned columns. Original immutable CPU
scans stage when first consumed. Structural empty/scalar results can be retained
without initializing a device, including positive row counts with zero columns.

Each logical matrix cell uses 21 requested bytes: two UInt64 words, one validity
byte and one UInt32 deferred-error word. A query budget includes all retained
intermediates, active region matrices/descriptors/scratch, control observations,
and final host export/materialization. Bounds use checked host arithmetic and
are checked before allocation/submission. They measure requested payload, not
allocator or driver overhead, and cannot guarantee physical allocation succeeds.

File scans compact decoded source buffers before GPU execution. The budget
includes their retained values, validity, text offsets, categorical domains and
field metadata; decoder temporary memory is outside that requested-payload
budget. Profiles report source staging bytes and decode time. Readable-schema
value-decode failures remain attached to their source and are observed in plan
order, allowing an earlier GPU computation to raise first.
If a later file has no readable schema, earlier plan prefixes observe their
errors before the file error is raised. These observations download deferred
error words without materializing unused values. Metadata planning failures
retain the default CPU planning order.

Bare column aliases and supported row movement retain Arrow field metadata;
computed expressions follow the CPU metadata rules. Terminal struct packing
wraps exported primitive fields without copying their payloads. Parents remain
valid independently of nullable children. Direct source Struct field projection
combines parent and child validity on the GPU, so hidden child values under a
null parent remain unobserved. This currently requires a direct scan and
terminal primitive projection; nested or variable-byte children, bare Struct
outputs and identity casts of fields remain explicit restrictions. Profiles
include conservative host staging for chunk compaction and parent bitmaps.

Errors travel with values through conditionals and row movement. Selected
terminal values and observing consumers check them; unselected conditional
branches remain unobserved. Intermediate region boundaries observe public
errors without downloading values. Some checked expressions and reductions
still require a terminal boundary when metadata cannot prove safe evaluation
under downstream pruning. Rejection is an explicit capability restriction.

Supported bounds are 128 local expression nodes, 128 device steps and 256
logical slots/source/output columns per row region. Joins support 256 combined
input columns and 256 output columns. Row indices are bounded by Int32.MAX;
allocation/high-plane offsets use wider host/device scalars. Bounded join heads
count/gather only the observed output prefix instead of materializing full
Cartesian fanout.

Profiles retain `executor="metal"` and identify `runtime_implementation="mojo"`.
They report actual kernel launches, synchronizations, upload/download bytes,
host result bytes, resident input/output bytes and requested-memory peaks.
DAG reports identify intermediate resident boundaries, final CPU export and
whole-query traffic/peak/wall time. Whole-query wall time includes planning,
staging and materialization; no driver-only interval is presented as that time.

## Automatic placement and calibration

`estimate_auto`, `calibration_key` and `select_auto` use metadata and
context-free Apple host facts. Estimation reads no source values, performs no
GPU work and shares execution's requested-memory calculation. Hardware
identity includes host/chip/OS and power/thermal state. Package build identity
binds the actual Mojo/compiler/SDK/source configuration.

Mojo calibration has a separate provider-build/device/family identity. Native
Metal curves cannot match it. Operation, reduction, storage and normalization
signatures isolate differently priced work. Automatic placement uses the shared
conservative cost policy and requires matching validated end-to-end evidence,
independent held-out queries, unexpired records and memory headroom. Otherwise
it selects CPU. Dictionary staging and multi-region DAGs currently lack their
own analytical features and stay on CPU under automatic placement; forced
execution supports them.

The existing `bench_accelerator_cost` runner/fitter collects complete CPU/GPU
materialization times and validates parity outside timing. Raw samples alone
are not deployable calibration. See [benchmark rules](benchmarks.md) and
[development measurements](apple-metal-benchmarks.md).

## Remaining CPU coverage

Migrating the existing Metal implementation does not establish complete CPU
query parity. Grouped mode/value counts, implode,
IANA-zone duration windows, row-valued
text parsing/formatting, remaining timezone operations, nested list/struct
operations, broader source Struct consumers and explode/unnest remain outside
this accelerator subset. CUT/QCUT and opaque Binary/Geometry transport are also
pending. Some checked-expression/window compositions also reject.
These are implementation gaps, not demonstrated limitations of Mojo.

Whole-DAG metadata validation, including invalid as-of metadata or tolerance,
and speculative dictionary transformations can still precede an earlier
computation's error. File-prefix planning follows the default optimized CPU
path; the backend interface does not yet convey `optimize=False` to that
planning pass. Distribution expressions require equal output cardinalities;
ordinary scalars broadcast, while differing distribution lengths raise.

## Validation

Run the integrated query contracts and independent IEEE oracle on an Apple
Silicon Mac with the pinned environment:

```sh
pixi run --locked -e gpu test-mojo-gpu
pixi run --locked -e gpu test-mojo-gpu-floating
pixi run --locked -e gpu test-mojo-gpu-decimal
pixi run --locked -e gpu test-mojo-gpu-integer
```

The independent integer oracle uses Python arbitrary-precision arithmetic:

```sh
mkdir -p build/contract-tests
python3 tests/tooling/mojo_gpu_integer_probe.py build/contract-tests/integers.bin
pixi run --locked -e gpu mojo run -I . tests/tooling/mojo_gpu_integer_probe.mojo build/contract-tests/integers.bin
```

Installed-package validation can include actual GPU routing and execution:

```sh
pixi run package-smoke --gpu
```

The maintained backend contracts retain independent typed/bitwise expectations,
CPU comparisons, checked-error recovery and exact/one-byte-under budget checks.
The `execution.apple_parity` and `public.apple_parity` runner groups cover the
additional source, Struct, mode, temporal, decimal-extrema and partition-schedule
work. Temporal coverage includes independent arithmetic/cast boundary anchors,
IANA transition/footer fields and conditional/error recovery contracts.
Hosted macOS runners may expose no Metal device; CI reports that absence
explicitly and still runs metadata contracts. Execution claims require a real
local GPU run.
