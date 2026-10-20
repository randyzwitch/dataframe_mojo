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
retain configuration and share a lazy device service. Direct row and scalar
collections exclusively lease its one cached context; overlapping collections
use private contexts. Other resident DAG/operator routes still own fresh
contexts. Query allocations, budgets and fault state remain independent. Public
collection, profiling and explanation preserve the requested optimization,
streaming and batch settings through provider dispatch and recursive planning.
With `optimize=False`, file staging preserves the original source prefix rather
than applying optimizer projection or head pushdown. A requested streaming batch
does not replace the CPU's eager expression batch size.
Checked scalar SUM consumers apply the CPU's canonical predicate optimization
before issuing observation boundaries. With `optimize=True`, a filter folded to
constant false eliminates its input; with `optimize=False`, that input is evaluated.
Surviving checked producers are observed before downstream consumers.
`DATAFRAME_ACCEL_DEVICE` and `DATAFRAME_ACCEL_MEMORY_LIMIT` configure default
collections. `DATAFRAME_METAL_LIBRARY` is unnecessary and does not select the
implementation. `MetalRuntime.device_count()` explicitly discovers devices;
`metal_installed()` reports compilation-target support without initializing one.
The Metal provider selects the Metal device API explicitly.

Forced accelerator queries reject unsupported work rather than retrying on the
CPU. The complete reachable plan is bound before device initialization. Queries
with `engine="cpu"` ignore accelerator policy and device configuration.

### Data faults

A data fault is a failure caused by input values: integer or Decimal
overflow, a failed strict cast, a temporal value out of range or a
nonexistent local time. Metal computes every checked value, including values
the CPU executor never evaluates because a head, filter or HAVING step prunes
their rows first, and it does not reproduce the CPU's batch and expression
order. So when any kernel writes a fault, Metal reruns the query on the CPU
with the same settings and returns the CPU's outcome: its error, or its
result when the faulting values are never evaluated there. A fault-free run
returns Metal's own result. Profiles of a decided query carry
`fault_adjudication="cpu"`. Device, budget and planning failures still raise
as they are; only errors marked as data faults
(`dataframe/accelerator/faults.mojo`) are decided this way.

Each region flags a fault in any step's output slot as soon as the step
writes it, and checks grouped outputs and reduction states, so no pruned
fault goes unseen. Checked work therefore needs no fault-order proof: checked
integer and Decimal arithmetic, strict casts, checked rank, window-axis and
temporal operands, and checked sums run before heads, filters, joins and other
regions. Conditional Decimal branches remain restricted: the CPU evaluates
both branches where Metal masks the unselected one.

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
| Temporal expressions | UTC, fixed-offset and IANA local fields; fixed/calendar offsets and truncation, duration totals, timezone metadata conversion; direct IANA replacement, offsets and truncation; temporal arithmetic and physical casts |
| Joins | Inner, left, right, full, semi, anti and cross; typed composite keys, null-equal left joins and key coalescing; backward/forward/nearest as-of joins with typed by-keys and tolerance |
| Composition | Selectors, scalar projections/broadcasts, aggregate expressions, computed join branches, post-join consumers and chained joins |
| Text | Ordered dictionary transport, supported string operations/concatenation, categorical reconciliation and restoration |
| Struct fields | Direct source Struct projection over fixed-width primitive children, parent/child validity and field metadata; terminal `as_struct` over primitive fields |
| Lists | Direct source `len`/constant-index `get`, terminal primitive `implode`, direct keyed grouped primitive `implode`, source `explode` with resident primitive consumers |

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

An eager scalar SUM with one checked Decimal arithmetic occurrence can remain
resident through later filters, joins and heads. Its replay observes each
8192-row operand batch before that batch's sum prefix, checks worker failures
before merging partials, and keeps the original nulls and result dtype. This
certificate does not cover streaming morsels, grouped/OVER feeds, multiple
fallible operand occurrences or fallible sibling expressions.

Grouped SUM/MEAN and count siblings additionally accept multiple checked
Decimal ADD/SUB/MUL occurrences when every operand/intermediate uses at most
64-bit physical storage and scale 18. All possible arithmetic failures then
have the same precision category, and every UInt32-bounded SUM prefix/merge
fits precision 38. The whole original AGG is observed before any downstream
head or join, including streaming collections; this proof does not assume a
worker or morsel assignment. Nullable operands and supported Boolean masks
retain their original evaluation. Wide Decimal intermediates, other fallible
operations/states, categorical keys and OVER feeds require separate proofs.

Scalar Float64 SUM also accepts these precision-only Decimal operands,
including same-type WHEN branches and infallible Decimal-to-Float64 casts.
Bounded Decimal SUM siblings retain their precision-38 state proof. The whole
original scalar projection observes operand faults before any later head,
filter or join, including when `min_count` would make the SUM null. Both
certificates require an infallible original input prefix and accept canonical
sign-extended Int64 literals with a Decimal128 dtype; arbitrary Decimal128
source words and implicit WHEN common-type casts retain separate guards.

The same grouped proof accepts Float64 SUM/MEAN states after infallible
Float64 ADD/SUB/MUL/DIV, including a checked Decimal operand converted to
Float64. IEEE infinities and NaNs affect values rather than introducing another
exception category; the existing Float64 reassociation contract applies.
Original checked Decimal operands still finish their observations before
`min_count`, heads or later consumers. Float32 operations and foreign checked
casts remain outside this proof.

Grouped WHEN/FILL_NULL operands may also select a canonical signed64 Decimal
literal with a different precision or storage width when its scale already
matches and its exact raw value fits the common result precision. These
implicit casts cannot raise, including on an unchosen branch. No row values
are inspected to prove this; differing scales, wide raw sources and unproved
column casts retain their guards.
Typed Decimal128 constants also qualify when their exact high word proves an
Int64 sign extension; this does not admit arbitrary Decimal128 source values.
The original input prefix, including both JOIN arms, must be infallible. A GPU
input-zero check independently rejects foreign flags and never substitutes for
that original streaming-order proof.

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
query budget. IANA timezone replacement, calendar offsets and truncation support
a direct in-memory scan followed by a single SELECT. They preserve datetime
units, ambiguity/gap policies, checked reconstruction, conditional masks and
scalar activation within each original CPU batch.

Eligible original strict-cast row projections can feed resident filters,
sorting, head, grouping and joins. Their typed producer certificate retains the
complete original phase and immutable source owners, so a later consumer cannot
hide an earlier cast failure. Conditional branches retain their original masks.
Scalar casts and casts nested inside aggregate/window operands retain their
existing restrictions; structural rewrites must preserve the certified prefix.

Eligible original IANA row producers can also feed resident sorting, head and
grouped consumers. Typed certificates retain the original expression, source
owners, input geometry and phase order; each checked producer is observed before
its downstream consumer can hide a fault. This includes the order of checked
integer prefixes and temporal errors. Regression coverage checks scalar mask
activation across requested streaming batch settings and inactive branches.
This route requires an exact certified in-memory prefix; arbitrary temporal
rewrites and resident compositions remain restricted.

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

Categorical transport preserves original codes and dictionary owners through
row, filter, sort, group, window, join and as-of operations. Duplicate category
labels remain distinct codes. Temporary join dictionaries do not replace the
output's original dictionary. Coded String grouping follows decoded byte
equality and publishes ordinary String values; numeric UInt32 columns retain
numeric semantics.

Direct keyed aggregation can produce primitive Lists alongside independent
primitive reductions, followed by direct outputs, List `len`/constant-index
`get`, or slices. The route preserves original key domains, child validity,
input order and supported checked-expression fault order. General composed
List consumers and uncertified parallel reduction associations remain outside
this direct route.

Certified primitive grouped List results can also remain resident for downstream
explode, row, join and grouping operations, including slices and primitive
reduction siblings. The producer retains device owners until its consumers
finish and checks original input phases before publishing results. Encoded
grouping keys still require a separate typed-domain owner; arbitrary checked
resident inputs are not admitted by this route.

Grouped primitive List producers can retain String, Binary, Geometry and
categorical key domains through EXPLODE and supported resident row, join and
group consumers. Query-scoped owners certify the original plan, source view
and dictionary or byte domain, and retain staging charges through the last
consumer. Categorical keys retain their original dictionary and codes; coded
String grouped keys publish ordinary String values. This route keeps the
existing primitive List and certified-producer limits; it does not add general
nested List values or arbitrary composed producers.

Partitioned `cut(...).over(...)` and `qcut(...).over(...)` keep numeric operands
resident, preserve original key domains and eager operand/control error order,
and support immutable custom or default labels. Certified nested numeric OVER
scopes preserve layout-cache order. `include_breaks=True` requires a terminal
projection; intermediate Struct consumers and other nested stateful expressions
remain unsupported.

## Residency, errors and memory

Computed query regions retain immutable Mojo-owned device tables. A consumer
copies selected columns into its own matrix on the device, so shared branches
cannot overwrite one another. Owners release after their last consumer. Only
the final result exports ordinary CPU-owned columns. Original immutable CPU
scans stage when first consumed. Structural empty/scalar results can be retained
without initializing a device, including positive row counts with zero columns.

Each legacy logical matrix cell uses 21 requested bytes: two UInt64 words,
one validity byte and one UInt32 deferred-error word. A query budget includes all retained
intermediates, active region matrices/descriptors/scratch, control observations,
and final host export/materialization. Bounds use checked host arithmetic and
are checked before allocation/submission. They measure requested payload, not
allocator or driver overhead, and cannot guarantee physical allocation succeeds.

### Initial native storage and fused scalar paths

Certified single-input, single-output Float32 row projections use native-width
values, a packed bitmap only when nulls are present, and no deferred-error plane
for the supported infallible expressions. Values transfer in bulk; sliced
bitmaps retain their bit offset, and chunked inputs repack only validity.
Supported stable filters compact native output. Final CPU columns adopt the
readback lists. Multi-output, grouped and other unconverted regions still use
the legacy representation. This Float32 kernel still interprets a bounded
expression program; it is not the interpreter-free fused reduction path.

The first precompiled reduction family accepts one Int32 input, an optional
`x > Int32(threshold)` filter, optional checked `(x + addend) * factor`, an
explicit Int64 cast, and terminal SUM. It preserves the filter boundary and
checks each Int32 arithmetic operation before accumulating. At most 4096
device partial records and a 24-byte final control record replace per-row
predicate, expression and filtered-table intermediates. With at most
`Int32.MAX` input rows, summing surviving Int32 values in Int64 cannot overflow.
The final valid count implements `min_count`; observed arithmetic faults retain
the existing CPU adjudication contract.

A second precompiled family accepts one Int64 or UInt64 input and one or more
terminal SUM outputs. Each output may use the source directly, checked addition
or multiplication by a same-type literal, or checked addition followed by
multiplication. This family does not accept filters. It retains native input
storage and accumulates exact two-word integer totals in at most 4096 device
partial records, shared sequentially across outputs. Final narrowing happens on
the device after all partials merge; only one 24-byte record per output returns
to the host. `min_count` may suppress final narrowing for a null result, but
never suppresses a checked row arithmetic fault. The Int32 row-count bound
ensures the exact total fits the two-word state. Other expression shapes keep
their existing execution paths.

The runtime service retains one idle context and up to 8 MiB / 32 entries of
exact-capacity Int64 and UInt32 reduction scratch. These idle limits are per
shared runtime service, separate from each query's requested-payload limit.
Every checked-out allocation is charged in full to the current query, including
pool hits. A smaller budget cannot become admissible just because buffers are
warm. Scratch returns only after its final GPU use completes; the exclusive
context lease stays held until the last query owner is released. An overlapping
query uses a private context and cannot consume the cached context's scratch.
Failure discards the cached context and its idle scratch before another query
can lease them. Runtime copies share the service; dropping its final owner
releases cached resources. Native column handles and arbitrary buffer views are
not pooled because their aliases can outlive an individual operator.

For the fused Int32 scalar reduction only, the service also retains up to nine
public compiled function handles: eight nullable/filter/arithmetic partial
specializations and one shared finish kernel. The cache is tied to the same
exclusive context lease; an overlapping query launches through its private
context and does not populate or consume these slots. Poison clears handles
before releasing the cached context. The SDK keeps executable storage opaque,
so this fixed handle count is reported as metadata and its unknown executable
bytes are outside the requested-payload budget. Profiles expose current
pipeline entries, cumulative hits and misses, poison clears, and the entry cap.

Direct execution currently performs one conservative final drain before marking
its context reusable. Profiles include this wait and expose cumulative service
context generations, scratch hits/misses, current idle scratch bytes/entries,
and fused pipeline-cache counters.
Context creation remains lazy: metadata-only work, empty native input and budget
rejection before submission do not initialize the device. Pipeline-handle reuse
outside the fused scalar path remains separate work.

Plain grouped SUM, COUNT, MIN and MAX over integer columns can split each
group across GPU blocks and merge exact partial states. The schedule uses
row and group counts, with at most 4096 partial states across a batch;
high-cardinality or small groups retain the existing reducer. One 256 KiB
scratch buffer is reused across eligible outputs. Metadata resource estimates
include that conservative cap whenever a plan could select the parallel path,
while profiles report actual allocation and partial geometry through the
`grouped_int_*` fields. Integer SUM keeps two-word totals until final narrowing,
and every partial retains source error bits. Floating, decimal, temporal and
window reductions retain their existing schedules. Group membership still
uses the existing sorted generic representation.

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
outputs remain explicit restrictions. Identity casts of primitive fields retain
their Arrow field metadata and parent validity. Profiles include conservative
host staging for chunk compaction and parent bitmaps.

Binary and Geometry columns use reversible, ordered byte dictionaries for
supported row movement, comparison, joins, grouping and windows. Transport
preserves raw bytes, geometry CRS/SRID and Arrow field metadata; it does not
interpret WKB or decode arbitrary bytes as UTF-8. Byte casts, text operations,
spatial computation and nested byte consumers remain restricted. Profiles
include byte staging and restoration in whole-query traffic and memory peaks.

Numeric CUT supports literal edges, custom or default labels, nullable operands
and terminal breakpoint Struct outputs. Operand errors precede edge validation.
Scalar and reduction siblings retain CPU evaluation order, including empty
inputs; a private one-row interpreter path selects the first scalar expression
fault. Eligible numeric, Boolean and temporal projections directly from an
original in-memory scan select faults in CPU projection, 8192-row batch and
AST occurrence order. Normalized sources,
state producers and streaming regions need separate ordering proofs.
QCUT supports scalar quantile controls, custom or default labels, nullable
operands and terminal breakpoint Structs. GPU-produced edge controls reach
the shared label formatter; input values and row codes remain on the device.
Immutable per-column dictionaries retain distinct labels through projections,
filters, joins and supported windows. Consumers remap tagged text slots on
the GPU into a shared dictionary while retaining numeric UInt32 and opaque
code identities. Default labels add eight control bytes per quantile edge
without an extra wait. Partitioned CUT/QCUT, intermediate breakpoint Structs,
NaN probability controls and string lookup operations over dynamic label
dictionaries remain pending.

Errors travel with values through conditionals and row movement. Selected
terminal values and observing consumers check them; unselected conditional
branches remain unobserved. Intermediate region boundaries observe public
errors without downloading values, and any fault a region writes is decided
on the CPU (see Data faults).

Direct source List `len` and constant-index `get` support primitive row outputs.
The backend stages each used child once, consumes offsets and parent validity
on the GPU, and retains primitive child metadata for direct `get` aliases.
Null parents and out-of-bounds indices mask child reads. Terminal `implode`
also supports primitive row operands, preserving source order, nulls and child
metadata. Certified private producers select operand faults in CPU AST and
8192-row batch order. Prepared OVER states remain eager, and original projection
order determines which state or operand fails first. Certification requires a
proven original source; synthetic or unobserved foreign inputs cannot opt in.
GPU producers finish before trailing slices, including `head(0)`, observe the
one-row List result. Intermediate List values, nested children and general
checked reduction-state input compositions remain restricted.

Source `explode` supports fixed-width primitive List children and ordinary
primitive fields from an immutable scan. It stages descriptors and children
once, preserves hidden physical child backing, and expands logical counts on
the GPU. Null and empty parents each produce one null placeholder. Selected
List columns must have equal logical counts, and complete count validation
precedes downstream pruning such as `head(0)`. The expanded primitive table
feeds row, sort, join and aggregate consumers without intermediate value export.
Child Arrow metadata, Decimal128 high words and IEEE bit patterns survive.
Chunk compaction uses private buffers and leaves caller caches unchanged;
profiles and budgets include staging, controls and final typed materialization.
Every retained nested field must be expanded. Computed/grouped List producers,
nested children and unselected nested output fields need further integration.

Scalar and grouped quantiles preserve CPU signed-zero representatives, including
quickselect fallback permutations, and use the same separate rounding steps
for linear interpolation. A shared device index plane adds four bytes per input
row; correction launches retain the existing synchronization and export boundaries.

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

Profiled generic row regions additionally report host wall-clock phases:

| Field | Measured interval |
| --- | --- |
| `preflight_ms` | Standalone row-plan binding and resource preparation before execution |
| `host_prepare_ms` | Region checks, descriptors and host buffer preparation before source staging |
| `staging_copy_ms` | CPU source conversion into the upload representation |
| `initialization_ms` | Acquisition of the query's context, including construction when no cached context is available |
| `submit_wait_ms` | Device allocation, enqueueing and completion waits, excluding context construction |
| `synchronization_ms` | Time inside region synchronization calls; overlaps execution phases |
| `readback_ms` | Final host matrix export, including its allocation and waits |
| `result_copy_ms` | Typed CPU column construction from the completed host matrix |

These clocks are enabled by profiling, not ordinary collection. They are not
GPU event timings: submission can include lazy pipeline compilation, and
readback can include device gathering. Do not add the synchronization interval
to the other phases. Small accounting/reporting and ownership-cleanup costs
remain outside these intervals; caller wall time is the complete measurement.
Resident DAGs retain each measured region phase and use nulls for phases a
specialized operator does not measure. Only the exporting region reports
readback/result construction; `query_preflight_ms` is the whole DAG's metadata
preflight, repeated on each report row, and must not be summed across rows.
Allocation counts and isolated GPU/pipeline times are not supplied by these
host intervals.

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
query parity. Grouped mode/value counts and nested implode,
IANA-zone duration windows, row-valued text parsing/formatting, remaining timezone
operations, nested List/Struct operations, broader source Struct consumers,
computed/grouped List expansion and unnest remain outside this accelerator
subset. Partitioned CUT/QCUT, broader dynamic text consumers and Binary/Geometry
operations remain pending. Some checked-expression/window compositions also
reject.
These are implementation gaps, not demonstrated limitations of Mojo.

Whole-DAG metadata validation, including invalid as-of metadata or tolerance,
and speculative dictionary transformations can still precede an earlier
computation's error. Execution settings preserve requested options, while
full CPU streaming schedule and checked state-feed equivalence still need
separate proofs. Distribution expressions require equal output cardinalities;
ordinary scalars broadcast, while differing distribution lengths raise.

## Engine performance overhaul

The CPU and Metal engines should select the same families of algorithms,
with implementations appropriate to their hardware. The CPU already has
bounded-domain aggregation, partitioned hash aggregation, bounded top-k,
hash joins and fused expression batches. General Metal grouping currently
sorts row indices, ranking materializes complete orderings, and row
expressions interpret descriptors into a slot matrix. Narrow dictionary SUM
paths avoid some of this work; they do not replace general operator dispatch.

The implementation sequence is by operator, with small dependent PRs:

| Order | Operator work | CPU implementation to reuse or match | Completion gate |
| --- | --- | --- | --- |
| 1 | General dense/hash grouped reductions; sorting only where reducer semantics require it | `GroupBy.agg`, `perfect_aggregate`, `aggregate_table` | Multiple keys and reducers share dispatch; runtime scales with rows and group count rather than full key sorting |
| 2 | Bounded global and per-group top-k; fuse ordinal-rank predicates into selection | `top_k`, `aggregate_top_rows` | Work and storage depend on requested k; no complete ranking or sorting of rejected rows |
| 3 | Equality hash joins with bounded-domain membership paths and late payload gathering | `join_hash`, bounded integer joins in `frame` | Build/probe paths cover supported join styles, duplicates, null policy and composite keys without sorting both inputs |
| 4 | Typed fused expression regions and filter-to-reducer feeds | `execution`, `fusion`, reducer input/update/finish contracts | Simple arithmetic/filter/aggregate pipelines avoid per-node global intermediate columns |
| 5 | One query execution scope with reusable resident buffers and device-side partial-state merging | `pipeline` ownership model; GPU `residency` | Batches and dependent operators retain device state instead of re-entering runtime and concatenating host partial frames |

These overlap: grouping needs compact reducer states and device-side merging,
and joins and selection need the same key equality and ordering contracts.
Factor those contracts into shared descriptors rather than adding admission
rules based on a benchmark name or one particular query shape. Existing CPU
`PerfectLayout` and reducer descriptions are starting points; GPU state
updates and parallel ownership remain backend implementations.

For grouping, first separate algorithm admission, key identity, state
accumulation and result publication. Choose dense slots when the complete
key domain has a bounded layout, hash partitions for general equality keys,
and retain the sorted implementation for states that require its ordering
until an equivalent feed schedule exists. Admission must include key and
reducer types, nulls, domain/cardinality bounds and memory budget. Sampling
may choose an algorithm but cannot prove that a domain or capacity fits:
capacity exhaustion needs a bounded retry or an explicit existing fallback.
First/last, argument positions, stable representatives, checked feed errors
and floating update order must survive changes in partitioning.

Every operator PR has two short gates before broader work:

1. A focused correctness matrix covering supported physical and logical
   types, composite keys, nulls, empty inputs, sliced/chunked storage,
   stable ties, checked errors, budget rejection and recovery. Compare
   public CPU outcomes and independent typed or bitwise expectations where
   available. Approximate numerical checks do not establish exact parity.
2. A small repeatable performance experiment covering small and large
   inputs, low and high cardinality, skew and relevant output sizes. Record
   algorithm choice, launches, synchronization, transfers and peak memory
   alongside warm timings. Separate primitive timings from complete query
   timings and keep compilation, setup and output verification explicit.

Broad query reports remain a separate milestone after these operator gates;
they are not the development feedback loop.

## Validation

Metal is validated on an Apple silicon Mac before a pull request, not in CI.
GitHub's hosted macOS runners are virtual machines whose GPU gives Metal
buffers no GPU address, which MAX's device context needs, so the Mojo GPU
provider cannot run there (288 of 354 contracts failed, 278 of them on
"Failed to get GPU address from Metal buffer"). One script runs everything
the retired `native-metal` job did, and fails if no device is found:

```sh
scripts/test_metal_local.sh
```

It builds the provider for a CPU-only target, probes the device
(`tests/tooling/metal_available.mojo` prints why it decides), builds the
Parquet reader, then runs every contract group that has a module marked
`# requires: metal` or `# requires: mojo_gpu`, with both capabilities. It
then runs the integrated query contracts and independent oracles, and reports
every failed step at the end. Those last steps can also run alone, with the
pinned environment:

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
