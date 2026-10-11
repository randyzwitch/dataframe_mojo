# Apple Silicon Metal backend

The optional native Metal runtime executes the shared bound row plan on Apple
Silicon. The normal dataframe package needs no MAX package or GPU SDK. Building
the runtime needs macOS 15 or later and Xcode's Metal compiler tools:

```sh
pixi run build-dfmetal
```

From that checkout, use the existing engine interface:

```mojo
var result = frame.lazy().select((col("x") * 2 + 1).alias("y")).collect(
    engine="accel"
)
```

For an installed Mojo package, put `libdfmetal.dylib` in the environment's `lib`
directory, or set `DATAFRAME_METAL_LIBRARY` to its full path. An explicit path
is authoritative: an absent or invalid configured library does not silently
select a different build. Importing dataframe, explaining a query, or collecting
with `engine="cpu"` does not load this optional runtime or initialize a device
through the provider. The upstream Mojo runtime may itself load the system Metal
framework; that also happens with the unchanged CPU-only package.

Explicit runtime configuration uses the same backend interface:

```mojo
from dataframe.metal import MetalRuntime

var runtime = MetalRuntime(memory_limit_bytes=256 * 1024 * 1024)
var result = query.collect(accelerator=runtime)
var measured = query.profile(accelerator=runtime)
```

Construction validates configuration without loading the library. A query is
bound and checked before device initialization. Default collections also accept
`DATAFRAME_ACCEL_DEVICE` and `DATAFRAME_ACCEL_MEMORY_LIMIT`. Native device zero
retains a serialized queue and a bounded cache of 64 specialized shader libraries
for the process. Runtime copies retain their configuration. Nonzero devices use
a collection-scoped context. Literal values, row counts, and output names are
runtime data; changing those does not require recompiling an identical shader.

## Native precision and supported operations

Sources must be contiguous in-memory Float32, any signed or unsigned integer
width (8, 16, 32, or 64 bits), packed Bool, temporal, or Decimal32/64 columns. Mixed numeric dtypes
retain their native types through expressions, filtering, and materialization.
Sliced numeric and bitmap windows are supported. Expressions support literals, add/subtract/multiply/negate,
comparisons, Kleene Boolean logic, null tests, fill-null, and explicit casts
among these numeric and Boolean types. Out-of-range casts and NaN-to-Bool
conversions raise in strict mode and become null in non-strict mode. Nulls stay
null. Strict casts currently require terminal projections without head.
Integer expressions additionally support floor division, remainder, powers,
absolute value, clipping, and integral rounding operations. Division and
remainder by zero return null; signed floor division and remainder follow
floor semantics. Float32 supports absolute value, floor, ceiling, rounding to
a whole number, clipping, NaN filling, and NaN/finite/infinite predicates.
Float32 decimal rounding, floating floor division/remainder/power, and
transcendental functions still require CPU execution.

Conditional `when/then/otherwise` expressions propagate an observation mask
through each branch. Inactive branches do not raise integer overflow, strict
cast, or precision errors. A null predicate selects the otherwise branch.

Projection,
with-columns, drop, stable filtering, and final head remain resident until the
result is copied to ordinary CPU columns. Stable sorting supports multiple keys
with separate direction and null placement for each key. NaNs follow numeric
values in either direction; equal keys retain their input order. Sorting can
appear between projections and filters without materializing CPU columns.
The GPU sorts row indices with a parallel merge sort and gathers the live
columns. Preflight includes both index buffers and both value matrices.
Checked integer arithmetic currently
requires terminal projections without head, matching the shared planner's
observable-overflow boundary.


Inner, left, right, full, semi, anti, and cross joins run matching and output
gathering on the GPU. Keys can use the supported native and fixed logical types;
paired logical dtypes must match. Null keys do not match. Float32 NaNs form one
matching class, and signed zeros compare equal. Duplicate matches retain build
input order: inner/left/full probe left rows in order, right joins probe right
rows in order, and full joins append unmatched right rows in original order.
Key coalescing and suffix collisions follow the CPU API, including separate
key columns for full joins with `coalesce=False`.

The complete join DAG is bound before device initialization. Supported row
regions can feed a join and follow its result, including further joins and
grouped or scalar reductions. Intermediate results remain in native Metal
buffers. Consumers read their words and validity on the GPU without constructing
CPU columns or staging intermediate payloads. Original CPU scans stage at their
first consumer; only the final result becomes ordinary CPU columns. Immutable
result handles retain their query and context and are released after their last
consumer. Regions still synchronize for errors and cardinality, so a resident
DAG is not a single command-buffer submission.
Checked integer arithmetic, strict casts, and checked Int32/UInt32 sums in
join inputs require CPU execution because those inputs are not terminal
observable boundaries. Pure expressions and bounded Int8/Int16/UInt8/UInt16
sums can feed a join. Each join supports at most 64 combined input columns,
64 output columns, and Int32.MAX rows per input and output.

A final head immediately above a join or above pure projections of its rows
bounds the join output allocation, even when its full match count is larger
than Int32.MAX. Match counts and prefix offsets
use native 64-bit integers. Head is not pushed through filters, sorts, or
aggregations. Strings, categorical dictionaries, and nested join columns are
not yet supported by this native path.

Temporal and Decimal32/64 columns retain their complete logical metadata
through projection, stable filtering, null tests, fill-null, counts, and
extrema. Comparisons use the native physical integer representation; decimal
comparisons currently require matching scales and storage widths. Logical
arithmetic and casts are checked separately and currently require CPU
execution. Decimal sums retain the CPU contract of Decimal128 accumulation
and require CPU execution; Decimal128 storage is not yet supported by this
row backend.

Grouped aggregation supports multiple keys with minimum, maximum, count,
length, first, last, null count, arg-min/arg-max, Boolean any/all, and the
supported integer sums. Null keys form a group; Float32 NaNs
form one equality class, and signed zeros compare equal. `maintain_order=True`
returns groups in first-seen order, while rows within each group retain their
input order. The default group order is key order. Projections, filtering,
sorting, further grouping, and final reductions can follow a grouped result
without returning intermediate data to the CPU. Checked Int32/UInt32 grouped
sums currently require terminal aggregation without head.

The GPU sorts row indices, marks group boundaries, and builds group ranges.
Each group reduction uses a 256-thread workgroup; GPU-produced dispatch
arguments schedule only the observed groups. Preflight includes these arguments
and all temporary slots. Grouping requires at least one key, matching the CPU
API.

Minimum and maximum preserve the input dtype, ignore nulls, and return null
for empty or all-null inputs. NaNs order above numeric values, so maximum
returns NaN if any valid value is NaN; minimum returns NaN only when every
valid value is NaN. Equal values retain the earliest selected row, including
signed zeros. Float32 extrema compare native encodings and preserve subnormal
values without arithmetic emulation.

First and last preserve the input dtype and select a row even when it is null;
empty inputs produce null. Null count returns Int64. Boolean any/all support
both ignoring nulls and Kleene null propagation; empty any is false and empty
all is true. Arg-min and arg-max return UInt32 positions, skip nulls, prefer
numeric values over NaNs, and retain the first tie. If every valid value is
NaN, the first valid NaN supplies the position. Positions are within the group
for grouped aggregation and within the selected input for scalar reductions.
These reductions also run on the GPU after filtering or sorting.

Count and length return Int64. Int8, Int16, UInt8, and UInt16 sums return Int64.
Int32 and UInt32 sums retain their input dtype with a final range check. These
sums use an exact Int64 accumulator. The initial input has at most Int32.MAX
rows, so this accumulator cannot overflow even when partial sums cancel. Sum's
`min_count` is retained. Float32 row arithmetic preserves separate operator
rounding for supported operations and disables shader contraction and unsafe math
optimizations. Metal can flush Float32 subnormal operands and results even in
safe math mode ([Metal specification, sections 8.1 and 8.5](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)). Native kernels detect subnormal arithmetic and
potential underflow, then reject the result with a precision error after draining
the command buffer. Copying subnormal values and flipping their sign preserve
their bits. Float32 comparisons, predicates, clipping, and integral rounding
use native encodings and support subnormals. This guard is conservative: a tiny product that would correctly round
to zero can also require CPU execution. There is no arithmetic emulation or
automatic retry after submission.

Float64 inputs, floating sum/mean, and Int64/UInt64 or duration sums require CPU execution. Their
precision contracts need hardware types Apple GPUs do not supply. The backend
does not emulate Float64 or wide integer accumulation. Forced accelerator
execution rejects these statically unsupported plans before submission.
Integer-to-Float32 casts use native conversion. The CPU cast contract rounds
through Float64 first; a small region around Float32 halfway points above
2**53 can therefore double-round differently. Kernels detect that region and
require CPU execution without emulating Float64. Value-dependent precision
failures are reported during execution. Automatic selection remains
on CPU until matched end-to-end cost evidence is available.

## Memory, synchronization, and profiles

Row requests use ABI version 2. Joins add a separate version 1 two-phase
interface without changing row request layouts; rebuild the optional library
when upgrading. Homogeneous plans retain their original native-width matrices.
Mixed plans and extrema reductions store each value as an exact 64-bit word,
with generated native operations for each expression dtype. Numeric inputs are staged at their original
width and packed on the GPU. Output words are copied into the matching CPU
column width after synchronization; CPU code does not evaluate expressions.

The native bridge uses documented Metal shared-storage buffers with the default
CPU cache mode. A row region stages input bytes once, queues all dependent
kernels, waits once, and copies its result into CPU-owned columns. A join stages
its inputs once and uses two waits: matching determines cardinality, then a
second submission gathers and packs the output. Shared memory still
requires synchronization; CPU code never reads an in-flight GPU result.

Each context keeps up to 64 idle shared buffers, bounded by the smaller of
256 MiB and one sixteenth of the device's recommended working set. Repeated
collections reuse exact-size allocations; recycled storage is cleared before
use. Shape changes release unmatched idle storage before allocating. Live join
sessions retain exclusive ownership of their buffers, and storage enters the
idle cache only after submitted GPU work has drained. Idle bytes are reclaimable
for headroom checks; explicit payload estimates still count every live request
allocation. Native callers can inspect or release idle storage with
`dfm_workspace_info` and `dfm_workspace_trim` without invalidating live sessions.

Consecutive projection steps, including the next filter predicate, execute in
one specialized kernel. Intermediate expression values stay in thread-local
variables. Shared matrices hold input columns, final outputs, filter predicates,
and values needed after a filter boundary. Backward use analysis omits columns
from stable gathers when they are needed only before that boundary. Operator
rounding, validity, and precision/overflow guards remain
in the fused expression sequence.

Filtered scalar count, length, and supported integer sums consume predicate
states directly when no sort or group operation intervenes. These queries do
not allocate the second value matrix, prefix ranks or block offsets, and do not
compact selected rows. Later expression segments run only on lanes retained by
all preceding predicates. Length counts selected rows even when their values
are null; count and sum retain their own validity and `min_count` contracts.
Profiles identify this path as `native_metal_predicate_reductions`. Row outputs
still use stable compaction, and other scalar reductions retain their existing
execution path.

Preflight uses this same physical slot layout and counts each requested shared
buffer once, both matrices when row compaction is needed,
prefix ranks and block offsets, bitmap windows, typed literals, output bitmaps,
reduction partials/results, GPU descriptors, metadata, and error storage. It also
counts the worst-case CPU result allocation. Checked arithmetic rejects size
overflow. The explicit payload budget is limited by current recommended Metal
working-set headroom. Planning descriptors, compiler/cache allocations, allocator
rounding, retained source columns, and unrelated process memory are outside this
payload estimate; it is not an operating-system memory limit.


Join preflight estimates the known matching buffers without assuming a
Cartesian-sized result. The GPU sorts build indices and finds each probe's
matching range. Full joins scan range-coverage differences and compact unmatched
build rows in original order. After the first wait, the runtime checks the
complete matching-plus-output peak against the payload budget and maximum
buffer length before the caller allocates CPU result columns. It checks current
headroom again before output submission. Budget failure can therefore occur
after matching, while caller outputs remain untouched. Resident DAG budgets
include every live intermediate buffer, the current region's allocations, and
the final CPU result reservation. Shared owners are counted once; branches
retain their allocation until their last consumer. Region profiles still show
their own requested allocations, while `query_peak_requested_bytes` reports the
complete requested peak. Retained original CPU inputs remain outside this
payload estimate.

Profiles report observed launch and wait counts; upload/download traffic;
requested shared/result bytes; pipeline cache hits and compilation time;
staging-copy, submission-and-wait, and result-copy time; and the complete Metal
command-buffer GPU interval when available. For joins, `gpu_ms` sums the two
command-buffer intervals. `gpu_ms` is null if Metal supplies no valid interval. It is not a sum of isolated kernel durations. Whole-query
`wall_ms` includes binding, initialization, compilation, allocation, staging,
execution, and result construction for a resident row query. Join DAG profiles
have one row per executed region, with per-region `wall_ms`; `query_wall_ms`
reports the complete DAG duration including preflight and materialization. `runtime_build_id` fingerprints the native
source, ABI, compiler, SDK, and build flags.

`resident_input_bytes` and `resident_output_bytes` identify retained native
payloads at each region boundary. Intermediate regions report zero CPU result
bytes and download only synchronization metadata (plus scalar checks for checked
sums); their consumers report no upload traffic for those resident columns.
Scalar reduction results can also feed a join or another row region without
copying their payloads through CPU columns. Resident DAGs require the additive
resident ABI version 1; older row-only and two-phase join interfaces retain their
existing layouts.

Submission errors propagate. The bridge drains committed work before releasing
buffers or returning an error; execution never retries the query on CPU after a
GPU fault. Output owners and source owners remain alive across that boundary.

## Validation

```sh
python3 native/dfmetal/test_native.py
python3 native/dfmetal/test_join.py
python3 native/dfmetal/test_resident.py
pixi run mojo run -I . tests/test_accelerator_planning.mojo
pixi run mojo run -I . tests/test_metal.mojo
```

The native ABI suite requires the built library and executes on Apple Silicon.
The Mojo suite always checks configuration and capability rejection; its GPU
cases run when the optional library is present. Cases include non-byte-aligned
slices, nulls, Boolean logic, stable filtering, empty outputs, exact Int64
literals, checked integer overflow, cancelled Int32 sums, memory rejection,
Float32 operator rounding, exceptional floating values, subnormal precision
rejection, cache reuse, and long fused projection chains across multiple filter
boundaries with live branches and pruned temporary gathers.

The join ABI fixtures cover seven join styles and ten native types, composite
nullable keys, non-byte-aligned Boolean windows, empty sides, all-null keys,
large scans, bounded large fanout, exact key payloads, budget rejection before
both submissions, output-metadata rejection, and session ownership. Public
parity cases also cover nested joins, row regions on both sides, post-join
grouping and sorting, scalar reduction inputs, logical metadata, and profiling
of the two synchronization points. Hosted macOS VMs still run pure ABI
validation when no GPU is exposed.

The resident suite checks row/join/row pipelines, multiple consumers, scalar
handoffs, all native dtypes, empty and nullable inputs, exact UInt64 and Float32
payloads, whole-query budget rejection, invalid output metadata, failure
recovery, and tables that outlive the caller's query and context handles.

Native grouping and joins now have bounded hash paths. Hash grouping preserves
first-occurrence group order and verifies complete composite keys after hashing;
hash joins preserve left and right duplicate order and every join type's null
contract. Float32 hashes canonicalize signed zero and NaN equivalence while
retaining the first original key payload in the output. Hash tables are limited
to inputs between 4,096 and 8,388,608 rows. Sorted fallback remains available;
explicit join budgets use that fallback.

Integer sorting can use stable four-bit radix passes for 65,536 through
8,388,608 input rows when the total key width requires at most 18 passes.
Null placement, signed integer order, descending order, and stable multikey ties
retain the public sort contract. Float32 keys and sorts following grouping use
the merge path. If radix/group hash scratch would exceed an explicit row budget,
the estimator and execution both select the bounded fallback.

The additive plan-feature functions expose the selected native mechanism in
profiles. Native ABI, join, and resident suites passed on Apple Silicon, and all
19 public Mojo Metal tests passed with the rebuilt library loaded. New fixtures
exercise the hash threshold on both sides, composite nullable keys, signed-zero
and NaN equivalence, radix ordering, bounded fallback, and error recovery. These
correctness results do not establish performance crossovers for the dispatch
bounds.
