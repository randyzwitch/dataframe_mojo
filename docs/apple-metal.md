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

## Precision and supported operations

Metal accepts up to 128 logical value slots, source/output columns, device
steps, and nodes per row expression. Joins accept up to 128 combined input
columns and 128 output columns. These bounds are independent: computed values
and grouped-reduction temporaries also consume slots. Other accelerator
backends retain their existing 64-entry defaults. The extended bounds require
rebuilding the optional native Metal runtime; ABI layouts and opcode meanings
are unchanged. Memory budgets and semantic preflight checks still apply.

Direct column projections and aliases reuse an existing resident value slot.
They preserve separate output names and column order without a copying device
step. Filters, sorts, and slices gather each shared value and validity bitmap
once. Computed expressions still allocate slots, and aliases still count
toward the independent 128-output-column limit. Automatic placement uses a
separate `metal-column-slots-v1` cost family for plans that reuse columns, so
older calibration cannot select them for GPU execution. A complete supported
cost estimate is still required; wider explicit plans can remain on CPU
automatically when the shared analytical estimator rejects their shape.

Sources must be contiguous in-memory Float32, any signed or unsigned integer
width (8, 16, 32, or 64 bits), packed Bool, temporal, or Decimal32/64/128 columns.
Float64 columns use IEEE words throughout resident execution. Add, subtract,
multiply, divide, floor division, remainder, square root, comparisons, numeric/Boolean casts,
absolute value, negation,
floor/ceiling, decimal rounding, classification, clipping, and NaN filling
use integer-only binary64 primitives on the GPU. Basic arithmetic rounds to
nearest with ties to even, preserves gradual underflow, and does not contract
separate multiply/add operations. Float64 floor division matches the CPU
contract of flooring the rounded quotient. Remainder (`%`) matches the CPU SIMD
sequence: truncate the rounded quotient, use the CPU’s fused multiply/subtract, and adjust
for differing operand signs. It preserves each binary64 intermediate, including
the CPU's positive-zero results and NaNs for infinite divisors. This differs from
native `fmod` for extreme quotients. Square root is correctly rounded using an
integer root and exact remainder; negative operands produce NaNs, signed zeros
retain their sign, and subnormal inputs do not flush to zero. Integer square
roots first convert to Float64, matching the CPU result type. Float32 square
root computes in emulated binary64 and rounds back to Float32, including for
subnormal inputs. Integer true division produces Float64.
Copying values preserves all bits, including NaN payloads and signed zero;
arithmetic quiets NaNs, whose resulting sign/payload is not a portable CPU
contract. Float64 keys, stable sorting, grouping, joins, distinct rows, extrema,
and arg-extrema use bitwise IEEE ordering and equality. Mixed numeric dtypes
retain their native types through expressions, filtering, and materialization.
Sliced numeric and bitmap windows are supported. Expressions support literals, add/subtract/multiply/negate,
comparisons, Kleene Boolean logic, null tests, fill-null, and explicit casts
among these numeric and Boolean types. Out-of-range casts and NaN-to-Bool
conversions raise in strict mode and become null in non-strict mode. Nulls stay
null. Checked casts to integer/Boolean targets require terminal
projections without head unless the input's static range fits the target.
Numeric casts to Float32/Float64 cannot raise range
errors and may occur in child regions or before head.
Integer expressions additionally support floor division, remainder, powers,
absolute value, clipping, and integral rounding operations. Division and
remainder by zero return null; signed floor division and remainder follow
floor semantics. Float32 supports absolute value, floor, ceiling, rounding, clipping, NaN
filling, and NaN/finite/infinite predicates. Float32 and Float64 rounding at
any decimal count uses binary64 emulation and follows the CPU half-away formula,
including its large-value early return and every intermediate rounding step.
The scalar power-of-ten scale is bound once using the CPU scalar formula; all
row computation stays on the GPU. Float32 floor division also uses binary64
emulation: it rounds the quotient to Float32 before flooring, preserving the
CPU result at integer boundaries and for subnormal inputs. Float32 remainder
also preserves each binary32 rounding step, including a directly rounded fused
multiply/subtract. Floating power and transcendental functions still require
CPU execution.

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
Average, min, max, dense, and ordinal rank also run on the GPU, globally or
within one or more `over()` partition keys. Ranks retain original row order;
ordinal ties follow input order. Null values remain null, null partition keys
form a group, signed zeros tie, and NaNs form one tie class (last ascending,
first descending), matching the CPU rank contract. Numeric, Boolean, String,
categorical, temporal, and Decimal32/64/128 inputs use the same stable sort
workspace. Dense ranks scan tie boundaries without downloading row data.
Filters, projections, grouped results, joins, slices, and final head can surround
rank operations. Integer arithmetic and casts can feed rank when static ranges
prove every checked input node safe. Unproved checked inputs and rank inside
grouped aggregation remain unsupported. Rank plans carry a separate
`metal-rank-v1` calibration family;
automatic placement stays on CPU without matching validated evidence.

Supported scalar reductions also run within `over()` partitions: Metal groups a
shared resident prefix and broadcasts each result back in original row order.
Null partition keys form one group; multiple keys, dictionary codes, fixed logical
types, and two-word decimal keys retain their normal equality. Composed aggregate
expressions and windows in select, with-columns, or filter remain resident,
including after joins and before later consumers. Empty inputs remain empty.
Floating sums and means preserve source order within the CPU window worker
partitions and merge those partials in worker order using binary64 emulation.
These aggregate-window producers are full observation boundaries, matching the
CPU's eager preparation of window states even inside a conditional or before
head. Checked sums and checked expressions inside these producers retain their
normal overflow checks. Ordinary checked join inputs keep their existing
restrictions. Nested aggregate windows materialize their inner results before
feeding them to the outer partition aggregate, retaining the original row order
and the same observation boundaries at each level. Row-valued partition
expressions, selectors, other nested window operations, and unsupported
underlying reductions still require CPU execution.
Automatic placement keeps these plans on CPU until their complete DAG has a
supported estimate and matching calibration.

Integer arithmetic can precede filters, joins, and head when expression types
and literals prove that every checked operation is safe for all input values.
The proof propagates conservative ranges through literals, casts, clipping,
conditionals, and arithmetic without reading source data. Each checked child
needs its own proof. Unproved arithmetic still requires terminal projections
without head, matching the shared planner's observable-overflow boundary.


Inner, left, right, full, semi, anti, and cross joins run matching and output
gathering on the GPU. Keys can use the supported native and fixed logical types;
paired logical dtypes must match. Null keys do not match. Float32 NaNs form one
matching class, and signed zeros compare equal. Duplicate matches retain build
input order: inner/left/full probe left rows in order, right joins probe right
rows in order, and full joins append unmatched right rows in original order.
Float64 NaNs and signed zeros use the same key equivalence.
Key coalescing and suffix collisions follow the CPU API. With
`coalesce=False`, inner, left, right, and full joins retain both input key
columns; unmatched rows preserve nulls on the missing side.

The complete join DAG is bound before device initialization. Supported row
regions can feed a join and follow its result, including further joins and
grouped or scalar reductions. Intermediate results remain in native Metal
buffers. Consumers read their words and validity on the GPU without constructing
CPU columns or staging intermediate payloads. Original CPU scans stage at their
first consumer; only the final result becomes ordinary CPU columns. Immutable
result handles retain their query and context and are released after their last
consumer. Regions still synchronize for errors and cardinality, so a resident
DAG is not a single command-buffer submission.
Unproved integer arithmetic, unproved casts to integer/Boolean targets, and checked Int32/UInt32/Int64/UInt64 sums in
join inputs require CPU execution because those inputs are not terminal
observable boundaries. Pure expressions and bounded Int8/Int16/UInt8/UInt16
sums can feed a join. Each join supports at most 128 combined input columns,
128 output columns, and Int32.MAX rows per input and output.

A final head immediately above a join or above pure projections of its rows
bounds the join output allocation, even when its full match count is larger
than Int32.MAX. Match counts and prefix offsets
use native 64-bit integers. Head is not pushed through filters, sorts, or
aggregations. String and categorical columns use query-wide dictionary codes
for native matching and gathering, as described below. Nested values still
require CPU execution.

Temporal and decimal columns retain their complete logical metadata through
projection, stable filtering, null tests, fill-null, counts, and extrema.
Decimal128 values use two 64-bit words throughout resident storage, sorting,
ranking, grouping, and joins; final export reconstructs the complete 16-byte
payload. Decimal comparisons require matching scales. They also require matching
storage widths unless one operand is Decimal128, in which case Decimal32/64
operands are sign-extended exactly. Logical arithmetic and row-valued casts
remain unsupported.

Decimal32/64 sums accumulate exactly in two words and return Decimal128 with
precision 38 and the original scale, including through joins and scalar
broadcasts. The Int32.MAX row bound keeps every possible partial below 2**94,
so no value-dependent range proof or CPU row evaluation is needed. Empty,
all-null, and `min_count` results match the CPU. Reductions of Decimal128 inputs
support counts, minimum, maximum, first, and last. Sums of Decimal128 inputs and decimal means still require CPU execution.

Native Int64 kernels extract year, month, day, hour, minute, second,
nanosecond, ISO weekday and ordinal day from dates, naive datetimes and times
where the CPU API permits each field. Datetime date/time extraction preserves
the output logical dtype. All ms/us/ns datetime units, negative instants and
nulls retain CPU semantics. Fixed-interval truncation supports storage-aligned
positive intervals, including Monday-aligned whole weeks. Month/year truncation,
zone-aware temporal operations, offsets and logical arithmetic/casts remain
unsupported. Source-independent literal `strptime` expressions and casts to
numeric, Decimal32/64/128, or temporal types are folded before dictionary encoding.
The CPU scalar cast runs once per literal, with exact integer payloads and
logical metadata retained; input rows stay on the GPU. Invalid strict literals
remain unsupported. Decimal128 literals retain both words as runtime payloads.
Row-valued parsing and binary String concatenation reject before encoding, rather than binding text operations to
integer dictionary codes.

Grouped aggregation supports multiple keys with sum, mean, minimum, maximum, count,
length, first, last, null count, arg-min/arg-max, and Boolean any/all. Null keys form a group; Float32 NaNs
form one equality class, and signed zeros compare equal. `maintain_order=True`
returns groups in first-seen order, while rows within each group retain their
input order. The default group order is key order. Projections, filtering,
sorting, further grouping, and final reductions can follow a grouped result
without returning intermediate data to the CPU. Checked Int32/UInt32 grouped
sums, and sums requiring the Int64/UInt64 range proof, currently require
terminal aggregation without head.

The GPU sorts row indices, marks group boundaries, and builds group ranges.
Each group reduction uses a 256-thread workgroup; GPU-produced dispatch
arguments schedule only the observed groups. Preflight includes these arguments
and all temporary slots. Grouping requires at least one key, matching the CPU
API.

Expressions combining supported aggregate results are decomposed into native
aggregate leaves and a resident projection. This includes grouped and global
counts, extrema, supported sums, null counts and conditional expressions.
Global reductions can feed later row regions without CPU materialization.
They can also broadcast into row-valued projections, filters, and with-columns
expressions, such as `col("x") > col("x").mean()`. A shared resident prefix
feeds the scalar reduction and row branch, and a GPU cross join broadcasts the
scalar result. Empty inputs stay empty in row-valued consumers; all-null means
broadcast null. Temporary columns do not appear in the output.

Conditional scalar leaves remain restricted when eager evaluation could expose
a data-dependent precision rejection: Float32 arithmetic inside such a leaf,
and Int64/UInt64 means needing a range proof, reject before device execution.
Nonthrowing Float64 leaves and direct Float32 column reductions can broadcast
through conditional expressions. File-backed inputs still reject before source
reads. Automatic placement keeps these multi-region plans on CPU until it can
estimate the complete DAG.
Checked sums in child regions and checked arithmetic followed by head retain
their existing rejection boundaries. Selectors, windows and unsupported
aggregate leaves remain unsupported.

`n_unique()` aggregates in a select or grouped aggregation run as resident
deduplication followed by row counting. Null contributes one distinct category,
matching CPU semantics. Multiple distinct counts and other supported aggregate
siblings share a resident input and combine their results on the GPU. Grouped
branches match null keys and preserve first-occurrence order when requested;
global branches combine their single rows. Numeric, String/categorical, and
logical keys retain the existing key semantics. Composed aggregate expressions,
filters, joins, and empty inputs are supported. Checked sums, strict casts, and
checked arithmetic retain their normal child-region restrictions. Only existing
window observation boundaries permit checked window producers; splitting
ordinary aggregates does not create such a boundary.

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
`min_count` is retained. Int64, UInt64, and duration sums retain their result
dtype and require a GPU range proof. For N input rows (or rows in a group),
every valid positive value must fit the native positive maximum divided by N;
every negative magnitude must fit 2**63 divided by N. Null rows do not trigger
the check. This proves that every parallel subset and merge fits, without a
wider accumulator. Scalar filtered reductions conservatively use the original
row count when they skip compaction. An unproved input raises an unsupported
accumulator error before caller outputs change, even when cancellation could
make the final result fit. A later query can reuse the runtime after rejection.
Float32 row arithmetic preserves separate operator
rounding for supported operations and disables shader contraction and unsafe math
optimizations. Metal can flush Float32 subnormal operands and results even in
safe math mode ([Metal specification, sections 8.1 and 8.5](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)). Native kernels detect subnormal arithmetic and
potential underflow, then reject the result with a precision error after draining
the command buffer. Copying subnormal values and flipping their sign preserve
their bits. Float32 comparisons, predicates, clipping, floor, and ceiling use native
encodings and support subnormals. Float32 rounding widens the encoding to
emulated Float64, then narrows once, also preserving subnormal behavior. This guard is conservative: a tiny product that would correctly round
to zero can also require CPU execution. These Float32 row operations retain
native arithmetic and its conservative precision guard. There is no automatic CPU retry after submission.

Float32 and Float64 sum/mean accumulate with emulated Float64 arithmetic;
Float32 sum narrows only once at completion, while means return Float64.
As in the CPU mergeable floating state, reduction reassociation is allowed and
bitwise reproducibility across execution schedules is not promised. Integer
means retain an exact integer total before Float64 conversion and division.
Their Int64/UInt64 inputs require the same native range proof as wide sums.
Sums of Decimal128 inputs and decimal means still require CPU execution.
Int64/UInt64 sums retain native accumulation with a range proof. Forced accelerator execution rejects statically
unsupported operations before submission; Int64/UInt64 sum and integer mean
range proofs can reject during GPU execution.
Int64/UInt64-to-Float32 casts emulate the CPU contract of rounding through
Float64 before narrowing, including halfway boundaries above 2**53.
Value-dependent precision failures are reported during execution. Automatic selection uses the shared
analytical cost policy when a matching validated calibration is configured;
otherwise it selects CPU before submission.

## Distinct rows, union, and slicing

`unique` supports subset keys, all four keep modes (`first`, `last`, `any`,
`none`), null keys, and ordered output. It lowers to resident grouping and,
for `keep="none"`, a GPU count filter. Float32/Float64 NaNs and signed zeros follow the
same key equality as grouping.

Lazy concat/union-all preserves left-then-right order, duplicate rows, nulls,
and complete logical dtypes. Opposite Boolean tags form a full GPU join with
no matches, and projections gather the original columns from each branch.
Nested concats remain resident. This lowering currently supports at most 31
source columns, because the join's combined input includes both tagged branches.

Offset slices accept clipped positive or negative offsets and a nonnegative
length, or the remaining rows. Their offset and length are runtime parameters;
intermediate slices compact all live columns on the GPU. Slices can occur after
filters, sorting, grouping, distinct rows, or joins and before reductions.

## Memory, synchronization, and profiles

Row requests use ABI version 2. Joins add a separate version 1 two-phase
interface without changing row request layouts; rebuild the optional library
when upgrading. Homogeneous plans retain their original native-width matrices.
Mixed plans and extrema reductions store each value as an exact 64-bit word
or two words for Decimal128, with generated operations for each expression dtype. Numeric inputs are staged at their original
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

Binary64 emulation and native Float32 kernels have different costs. Plans
containing Float64 values, floating sum/mean, rounding, square root, floor division or remainder, or
Int64/UInt64-to-Float32 conversions carry the separate `metal-emulated-float64-v1` calibration family.
Native-only calibration cannot select those plans. Without matching, validated
end-to-end evidence, automatic placement stays on CPU. Future calibration may
select emulation only when its conservative GPU estimate beats the CPU bound;
forced accelerator execution remains available independently of that decision.

Floating power, transcendentals, and variance/std are not yet
supported by this subset.

## Validation

```sh
python3 native/dfmetal/test_native.py
python3 native/dfmetal/test_join.py
python3 native/dfmetal/test_resident.py
python3 native/dfmetal/test_float64.py
pixi run mojo run -I . tests/test_accelerator_planning.mojo
pixi run mojo run -I . tests/test_metal.mojo
pixi run mojo run -I . tests/test_metal_completeness.mojo
pixi run mojo run -I . tests/test_metal_sets.mojo
pixi run mojo run -I . tests/test_metal_slices.mojo
pixi run mojo run -I . tests/test_metal_concat.mojo
pixi run mojo run -I . tests/test_metal_float64_storage.mojo
pixi run mojo run -I . tests/test_metal_float64_emulation.mojo
pixi run mojo run -I . tests/test_metal_bounded_sums.mojo
pixi run mojo run -I . tests/test_metal_decimal128.mojo
pixi run mojo run -I . tests/test_metal_constants.mojo
pixi run mojo run -I . tests/test_metal_aggregate_compositions.mojo
pixi run mojo run -I . tests/test_metal_distinct_counts.mojo
pixi run mojo run -I . tests/test_metal_temporal_fields.mojo
```

The unchanged external-suite query expressions can be checked with Metal
forced at every materialization boundary:

```sh
pixi run -e oracle python3 scripts/bench_metal_suites.py --all-suites \
  --scale smoke --reps 2 --threads 8 \
  --library build/dfmetal/libdfmetal.dylib \
  --output build/metal-suites/results.json
```

This is correctness and coverage evidence, not a performance report. Generated
CPU and Metal workers, source hashes, per-case logs, original input dtypes, and
every unsupported or failed case are retained. Comparison checks the full schema
and exact row multiset, including duplicates; original-order equality is also
recorded. No tolerance, Float32 recasting, automatic placement, or CPU retry is
used. A materialization-only adapter wraps H2O's eager operations as equivalent
lazy queries in its Metal worker; the CPU worker retains the original eager
expressions. Unsupported operations remain coverage gaps.

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
contract. Float32/Float64 hashes canonicalize signed zero and NaN equivalence while
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
bounds. Matched M1 development measurements are now available in
[`apple-metal-algorithms-m1-20261011`](../benchmarks/results/apple-metal-algorithms-m1-20261011/README.md):
hash joins improve all selected cases, while grouping depends on cardinality
and input order. Radix sorting is slower than the preserved merge path in all
six selected cases. These bounds therefore remain structural eligibility bounds,
not validated performance crossovers. Explicit Metal selection can regress for
these mechanisms; the analytical automatic policy excludes them.

String and categorical frame inputs use a query-wide ordered UInt32 dictionary.
All inputs and string literals share the same mapping, so equality and ordering
remain correct across independently encoded categorical inputs. Filtering,
sorting, grouping, joins, and projection run on native codes; the final result
restores String values or each output's original categorical dictionary. UTF-8
values, empty strings, nulls, and unused categorical members retain their public
semantics. Casts involving String/categorical values still require CPU
execution; unrelated supported numeric casts remain available.

Unary String predicates and transformations run through dictionary lookup
joins. Contains/LIKE, starts/ends-with, character/byte lengths, case conversion,
strip, replace, slicing, reverse, padding, zero-fill, and membership share the
CPU String kernel once per distinct dictionary value. Native left joins apply
the resulting table to resident row codes, preserve nulls and row order, and
feed later GPU operators. Newly generated text joins the dictionary before
row encoding. Integer membership supports every signed and unsigned width
without converting exact large integers through floating literals.

Encoding and decoding are host staging work. Profiles include
`dictionary_staging_bytes`, `dictionary_encode_ms`, and `dictionary_decode_ms`.
`dictionary_operation_tables` counts the generated unary lookup tables.
`query_wall_ms` includes both boundaries. An explicit payload budget reserves
source code storage, generated lookup tables, and dictionary bytes before
native execution, and checks a conservative restoration bound before allocating decoded outputs. Cross-input
categorical parity passed for all seven join types, including unmatched full
join keys and nulls. The four public dictionary tests also cover string filter,
sort, grouped sums, output metadata, profiling, and budget rejection.

Metal implements the shared backend's `estimate_auto`, `calibration_key`, and
`select_auto` methods for the common numeric row subset. Estimates use bound
expression counts and native launch, transfer, and allocation bounds without
reading source values. The additive native metadata probe does not create a
context, command queue, buffers, or pipelines. It reports device identity,
memory headroom, and whether the exact row pipeline is already cached.

Calibration keys include native and CPU build identities, device and host
identity, worker count, and execution options. An identified package is required;
development source, missing or stale evidence, unsupported plans, and rejected
memory estimates select CPU. Grouping, sorting, joins, and dictionary staging
currently require explicit accelerator selection because their complete costs
are outside the calibrated numeric row model. Errors after device submission
propagate to the caller.

The provider-neutral `scripts/bench_accelerator_cost.py` runner records cold and
warm whole-query timings, parity, preflight overhead, package hashes, generated
training queries, and separate held-out query identities. Failed observations
remain in the evidence. Coefficients and residual envelopes are frozen before
held-out validation; rejected models cannot enable automatic GPU execution.
Its regret report replays the policy against measured explicit-engine samples;
it does not measure automatic-execution latency directly.


The packaged M1 cost collection is preserved in
[`apple-metal-cost-m1-20261011`](../benchmarks/results/apple-metal-cost-m1-20261011/README.md).
Of 284 training and 128 held-out cases, 117 and 104 respectively succeeded;
unsupported cases remain in the raw evidence. The frozen Float32 model passed
its accelerator-safety gate; Int32 lacked validation coverage. All 208 replayed
held-out cold/warm decisions selected CPU, including 46 slower-engine choices.
The model therefore misses useful longer-expression placements. No calibration
from this collection is installed or enabled, and no external-suite performance
claim follows from it.

A clean identified package also passed 20 provider-neutral consumer modes,
including a synthetic matching calibration that executes a real Metal query,
stale and missing evidence selecting CPU, and submitted strict-cast errors
propagating followed by successful recovery. The synthetic calibration is
isolated test data and is never installed as measured evidence.

The binary64 primitive test compiles the same integer-only source as host C++
and as Metal. It checks millions of randomized and boundary cases against
native CPU arithmetic, including subnormal rounding, infinities, signed zeros,
NaNs, floating conversions, and signed/unsigned integer cast boundaries.
Finite row arithmetic and conversions are checked bit for bit; NaN payload
variation is separated from numeric equality. Public query tests cover resident
composition, nullable inputs, keys, reductions, and the separate cost family.
