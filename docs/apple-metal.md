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

Metal accepts up to 256 logical value slots and source/output columns,
128 device steps, and 128 nodes per row expression. Joins accept up to 256
combined input columns and 256 output columns. These bounds are independent:
computed values and grouped-reduction temporaries also consume slots. Other accelerator
backends retain their existing 64-entry defaults. The extended bounds require
rebuilding the optional native Metal runtime; ABI layouts and opcode meanings
are unchanged. Memory budgets and semantic preflight checks still apply.

Selectors (`all`, named column lists, exclusions, dtype matching, and positional
`nth`) expand against the input schema before dictionary, aggregate, or window
normalization. Operations and name prefixes/suffixes apply independently to each
matched column, in the CPU selector's order. Select, with-columns, grouped and
global aggregates, supported windows, and filters with exactly one match retain
their normal precision and observation checks. Expansion reads no source rows.
An empty selector in with-columns is a no-op. A select whose selectors match no
columns returns one row with zero columns, matching the CPU scalar selection
shape. Explicit `select([])` and dropping every column preserve the input row
count, including after filters, groups, joins, and limits. Native result and
dictionary materialization retain that row count without allocating column
payloads. Zero-column plans use `metal-empty-outputs-v1` so automatic placement
requires matching calibration for this result shape. Zero-column sources and
row consumers also execute natively, retaining row
counts without a synthetic host column. Their calibration families include
`empty-inputs-v1` and `metal-empty-inputs-v1`.

Scalar-only selections such as `select(lit(7))` produce one row even from an
empty or zero-column source. Native counts establish the scalar shape, then
resident expressions evaluate and project the literals. Typed nulls, text,
fixed logical values, later filters, reductions, joins, and limits retain CPU
behavior. Private counter aliases avoid user names and preserve the input
producer checks. These plans use the resident scalar DAG; its automatic
placement estimates remain unavailable.

Skewness and kurtosis execute global, grouped, and partitioned central-moment
reductions in emulated Float64, including bias correction and Fisher/Pearson
conventions. Empty and insufficient samples retain CPU nulls; constant and
nonfinite samples retain CPU NaNs. Worker states merge in CPU partition order.
`higher-moments-v1` and `metal-higher-moments-v1` require separate calibration.
As with variance, numerical comparisons allow Float64 reduction rounding;
decimal inputs use an explicit Float64 cast.

Direct column projections and aliases reuse an existing resident value slot.
They preserve separate output names and column order without a copying device
step. Filters, sorts, and slices gather each shared value and validity bitmap
once. Computed expressions still allocate slots, and aliases still count
toward the independent 256-output-column limit. Automatic placement uses a
separate `metal-column-slots-v1` cost family for plans that reuse columns, so
older calibration cannot select them for GPU execution. A complete supported
cost estimate is still required; wider explicit plans can remain on CPU
automatically when the shared analytical estimator rejects their shape.

Sources may be contiguous or chunked in-memory Float32, any signed or unsigned integer
width (8, 16, 32, or 64 bits), packed Bool, temporal, or Decimal32/64/128 columns.
Chunked sources upload their original fragments directly into resident storage;
columns may have different fragment boundaries and independent Boolean/validity
bitmap offsets. The wrapper owns child descriptors through the native call and
retains source arrays without a host rechunk. Native estimates include each
fragment's payload, bitmap window, alignment, and unpack launch, including empty
fragments. Resident joins upload chunked identity inputs before reusing them.
Calibration families include `chunked-inputs-v1:<fragment-count>` and
`metal-chunked-inputs-v1:<fragment-count>` so evidence from flat inputs or different
fragment counts cannot select these plans automatically. Floating reductions
retain the CPU contract allowing reassociation; cancellation-heavy sums can differ
between CPU eager, CPU streaming, and GPU routes.

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

Cumulative `cum_sum`, `cum_min`, `cum_max`, and `cum_count` run globally or
within one or more `over()` partition keys. Each partition follows original row
order, or reverse row order when requested; results retain their original row
positions. Null inputs remain null and do not change sum/min/max state.
Cumulative count returns a valid count at every row, including null input rows.
Integer sums check every prefix, decimal sums preserve their declared precision,
and Float32 sums accumulate in emulated Float64 before rounding each output.
Numeric, Boolean, dictionary, fixed logical, and Decimal128 values retain their
normal min/max ordering, including NaNs and signed-zero ties. Cumulative plans
use the `metal-cumulative-v1` calibration family.

`shift`, `forward_fill`, and `backward_fill` also run globally or within
one or more `over()` keys, following original row order. Shift preserves the
input type and inserts nulls beyond a partition boundary, including for extreme
signed offsets. Fill copies the nearest original non-null value in the requested
direction; its limit counts consecutive null rows, with a limit of -1
meaning unlimited and zero preserving the input. These operations copy stored
words, retaining floating-point NaN payloads, signed zeros, dictionary identity,
logical metadata, and both Decimal128 words. Their parameters remain runtime
values and their cost family is `metal-shift-fill-v1`.

`interpolate("linear")` and `interpolate("nearest")` fill interior null gaps
in original row order, globally or within one or more `over()` keys. Leading,
trailing, and all-null gaps remain null. Linear interpolation converts numeric
and decimal inputs to Float64 and uses the neighboring original values;
nearest preserves the input dtype and payload words, with ties choosing the
following neighbor. NaNs and infinities remain valid anchors. A partition owner
walks each gap once, without host downloads or per-gap scratch storage. These
plans use separate `interpolation-v1` and `metal-interpolation-v1` cost families;
linear interpolation also uses emulated binary64 arithmetic. Interpolation by
another expression remains unsupported.

Fixed-size `rolling_sum`, `rolling_mean`, `rolling_min`, and `rolling_max`
run globally or within `over()` partitions. Each trailing window includes the
current row and skips null inputs; `min_samples` defaults to the window size.
An empty or all-null window remains null even with a zero minimum. Integer sums
accumulate in 128 bits and check valid final outputs against the CPU result type;
UInt64 sum/mean inputs retain the CPU Int64 widening check. Floating sums and
means recompute each window in original row order with binary64 emulation,
rounding Float32 sum outputs afterward. Decimal means convert each input to
Float64 before accumulation. Extrema preserve original payload words, including
NaN ties, dictionary values, and Decimal128. Decimal rolling sum requires an
explicit Float64 cast, matching the CPU's current unsupported operation.
Rolling plans use `metal-rolling-v1`. Work grows with the number of rows times
the trailing window length; these kernels prioritize matching CPU evaluation.

Fixed-size `rolling_var` and `rolling_std` use sliding Welford updates with
removal and CPU-contracted binary64 fused multiply-adds, translating inputs by
the partition's first finite value to preserve
small variances around large offsets. Each valid nonfinite value contributes
to the sample count and makes that window NaN; the finite state recovers after
it leaves the window. A result is null unless the non-null count meets
`min_samples` and exceeds `ddof`. Primitive inputs, including the full UInt64
range, and Decimal32/64/128 inputs convert privately to Float64 first, matching
the CPU window path. These operations support nested `over()` scopes and later
resident consumers. Each partition walks its source rows once, with integer-only
binary64 arithmetic, and uses the separate `metal-rolling-moments-v1` cost
family. Duration-based rolling windows remain unsupported.

Variance (`var`) and standard deviation (`std`) run as scalar, grouped,
or `over()` reductions for primitive numeric inputs. They convert each value
to Float64, skip nulls, and use integer-only Welford updates and Chan merges.
The result is null when the valid count does not exceed `ddof`; nonfinite inputs
follow the CPU state updates. UInt64 conversion retains its unsigned value.
Decimal moments require an explicit Float64 cast, avoiding the CPU's existing
decimal moment storage defect. These plans carry `metal-moments-v1`.
Metal uses deterministic contiguous partition merges. CPU results can differ
slightly between streaming, hash, and materializing routes, and Metal can
likewise differ by floating-point rounding.
Partitioned `over()` states follow the materializing CPU worker boundaries.

Paired covariance (`cov`) and correlation (`corr`, Pearson or Spearman) also
run globally, by group, and within `over()` partitions. Only rows where both
operands are valid enter the state. Covariance is null with no pairs and zero
when the pair count does not exceed `ddof`; correlation is valid NaN below two
pairs or for a constant operand. Inputs convert independently to Float64,
including full UInt64 values. Spearman ranks only paired valid rows after that
conversion, uses average ranks for ties, and returns NaN for a paired NaN.
The ranks, paired states, and broadcasts remain on the GPU. Decimal inputs
require explicit Float64 casts. Checked expressions feeding staged ranks need
the existing static no-error proofs. Pair plans carry
`metal-paired-moments-v1`; their contiguous merges can differ from CPU routes
by floating-point rounding, as for variance.

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
restrictions. Nested aggregate, rank, cumulative, shift, fill, and rolling windows materialize their
inner scopes before feeding outer scopes, retaining original row order and the
same observation boundaries at each level. Cumulative prefixes remain checked
inside unused conditional branches and before head. Unproved checked inputs to
rank/cumulative/shift/fill/rolling kernels, other window kernels, and unsupported
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
sums can feed a join. Each join supports at most 256 combined input columns,
256 output columns, and Int32.MAX rows per input and output.

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
payload. Decimal comparisons support different scales and storage widths. Both
operands are sign-extended and rescaled to the common scale using exact integer
arithmetic, including the CPU's checked Int128 rescaling overflow. Comparisons
whose rescaling cannot be proved safe require a terminal projection without a
head or conditional branches. This preserves errors on rows the CPU observes.
The CPU has an unchecked one-row fast path when rescaling Decimal128 against
Decimal64; this combination requires a static proof on Metal. Safe comparisons
can feed filters, joins, aggregates, and windows. Estimates
include `metal-decimal-comparisons-v1`; prior comparison calibrations do not
apply.

Decimal addition and subtraction produce Decimal128 at precision 38 and the
larger input scale. Rescaling checks Int128 overflow, followed by the CPU's
wrapped Int128 add/subtract and precision check. Negation and absolute value
preserve the input decimal metadata and storage width and check signed minimum
values. CPU unary kernels also retain narrow decimal storage and handle Int128
explicitly. Nulls and conditional masks preserve which rows can raise errors.
Operations without a static range proof require terminal projections without a
head. Safe arithmetic can feed resident filters, joins, aggregates, and windows.
Estimates include `metal-decimal-arithmetic-v1`.

Decimal multiplication and division also return precision 38 at the larger
input scale. Products check the CPU's Int128 multiplication bounds before
half-even scale reduction. Quotients preserve the CPU's sequence of checked
precision, wrapping intermediate arithmetic, signed division, floor modulo,
and final half-even rounding; division by zero produces nulls. Both operations
retain the checked arithmetic placement rules and conditional row masks.
Multiplication with different scales requires a static rescaling proof because
the CPU's dense and nullable paths have different rescaling checks. Safe
operations can feed resident consumers. Estimates include
`metal-decimal-multiplicative-v1`; previous calibrations do not apply.

Fixed `dt.offset_by` intervals execute on Date, naive Datetime (all storage
units), and Time columns. Intervals must align to the storage unit. Addition
checks Int64 overflow before Time wraps within a day; nulls and conditional
branches retain CPU observation behavior. The CPU offset kernel now honors
conditional masks so inactive overflowing branches are skipped. Nonzero offsets require terminal
projections without a head, following the checked arithmetic boundary rules.
Calendar month/year `dt.truncate` intervals also run on Metal for Date and naive
Datetime values. Buckets align to proleptic year zero, including negative years,
leap years, century boundaries, and multi-month intervals. Date results preserve
the CPU's Int64 wrapping storage behavior at the full-domain endpoints; Datetime
conversion checks signed storage overflow. Conditional masks suppress errors in
unobserved branches on both CPU and Metal. Calendar Datetime truncation requires
terminal projections without head or later filters. Calendar Date buckets may
feed resident consumers. Estimates use `calendar-truncation-v1` and
`metal-calendar-truncation-v1` families.

Calendar `dt.offset_by` intervals also run on Metal for Date and naive Datetime.
Signed month/year/quarter shifts clamp the day to the destination month's length
and retain the time and fractional ticks. Mixed intervals apply the calendar
shift first, followed by checked fixed ticks. The native expression preserves
the CPU's signed month-index and Date storage wrapping, plus each checked step
of Datetime reconstruction. Conditional masks skip unobserved overflow. Calendar
Datetime offsets and nonzero fixed additions require terminal projections without
head or later filters; pure calendar Date offsets can feed resident consumers.
Estimates use `calendar-offsets-v1` and `metal-calendar-offsets-v1` families.

Time zone offsets remain unsupported. Estimates use
`fixed-offsets-v1` and `metal-fixed-offsets-v1` calibration families.

Median and quantile reductions return Float64 for numeric inputs, including
full-width UInt64 and Decimal32/64/128 with their numeric scales. CPU sample
conversion also preserves these values instead of the internal UInt64 ordering
keys or raw decimal payloads. Global,
grouped, and partitioned reductions support `linear`, `lower`, `higher`,
`nearest`, and `midpoint` interpolation. Nulls are excluded; NaNs count as valid
samples and sort above every number. Empty or all-null samples produce null.
The original Float64 probability is retained without narrowing, and linear and
midpoint arithmetic preserve the CPU operation sequence, including infinities.

Selection scans one byte of the sortable binary64 encoding at a time. Eight
passes find each required order statistic without allocating a sorted copy;
one GPU lane performs each reduction. Values and selected ranks remain on the
GPU. Estimates use separate `quantiles-v1` and `metal-quantiles-v1` calibration
families. Rebuild the optional runtime for the appended quantile opcodes.

Decimal32/64 sums accumulate exactly in two words and return Decimal128 with
precision 38 and the original scale, including through joins and scalar
broadcasts. The Int32.MAX row bound keeps every possible partial below 2**94,
so no value-dependent range proof or CPU row evaluation is needed. Empty,
all-null, and `min_count` results match the CPU. Reductions of Decimal128 inputs
support counts, minimum, maximum, first, last, sum, and mean. Sums preserve the
input precision and scale; means return Float64 using the same exact
scaled-integer quotient and remainder sequence as the CPU. Global, grouped,
null, empty, `min_count`, resident, and composed results retain CPU behavior.
Every input prefix and ordered worker-state merge is checked against the
declared precision, so cancellation after an invalid intermediate value still
raises. Rebuild the optional native runtime for these opcodes. Estimates use
`decimal128-reductions-v1` on CPU and `metal-decimal128-reductions-v1` on Metal,
so prior calibrations do not apply.

Decimal32/64/128 casts to Float64 convert the signed integer payload to
binary64 first, then divide by the separately converted integer `10**scale`,
matching the CPU two-rounding sequence. Both conversions and row division run
on the GPU using integer-only primitives; scale is bound once as a runtime
literal. Nulls remain null, and strict/non-strict modes produce the same result
because every valid decimal fits Float64. These casts can appear in filters,
join branches, aggregate consumers, windows, or before a head. Decimal32/64
SUM results can consequently be cast from their Decimal128 result to Float64.
Estimates include `metal-decimal-float-casts-v1`; prior calibrations do not apply.

Decimal32/64/128 casts also support changing scale or storage width, converting
to every supported signed or unsigned integer width, Bool, or Float32, and
converting integer or Boolean inputs to decimals. Reducing scale and converting
to integers rounds half to even, including negative ties. Float32 conversion
follows the CPU sequence: convert the integer to Float64, divide by the
Float64 scale divisor, then narrow to Float32. Rescaling and range checks use
integer arithmetic. Strict casts raise for observed
invalid values; lax casts produce nulls. Conditional branches, nulls, filters,
resident joins, reductions, windows, and final heads retain CPU error observation.
Strict casts without a static range proof retain the requirement for a terminal
projection without a head; lax casts can appear throughout resident regions.
Rebuild the native library for these cast descriptors. Estimates include
`metal-decimal-casts-v1`, so previous cast calibrations do not apply.

Decimal32/64 means return Float64. The GPU divides the exact two-word total
by the valid count and then by the decimal scale, retaining both integer
remainders before conversion. The separate Float64 conversions, divisions and
additions match the CPU rounding sequence. Global, grouped and resident scalar
means preserve empty and all-null results. Rebuild the optional native runtime
for this opcode; its cost family is `metal-decimal-means-v1`.

Int64 and UInt64 sums and means accumulate exactly in two words on the GPU.
The Int32.MAX row bound keeps every partial below 2**95, including unsigned
inputs. Large intermediate totals may cancel before the final sum is checked
against its result dtype. Mean rounds the exact total to Float64 and then divides
by the valid count, matching the CPU integer accumulator. Empty, all-null and
`min_count` results retain CPU semantics. Rebuild the optional native runtime
for these reduction opcodes; legacy native sum and mean opcodes retain their
existing range-proof behavior. Cost estimates identify the new accumulator
family separately from older calibrations.

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
Row-valued parsing still rejects before encoding. String concatenation uses bounded
lookup tables keyed by both operand codes. The host evaluates each distinct
dictionary pair once, and resident left joins map row pairs to result codes.
Null operands remain null; a literal-only operand has one dictionary value.
A lookup is rejected before allocation when it exceeds 1,048,576 pairs,
512 MiB of conservative staging, or the configured payload budget.

Grouped aggregation supports multiple keys with sum, mean, minimum, maximum, count,
length, first, last, null count, arg-min/arg-max, and Boolean any/all. Null keys form a group; Float32 NaNs
form one equality class, and signed zeros compare equal. `maintain_order=True`
returns groups in first-seen order, while rows within each group retain their
input order. The default group order is key order. Projections, filtering,
sorting, further grouping, and final reductions can follow a grouped result
without returning intermediate data to the CPU. Checked Int32/UInt32/Int64/UInt64
grouped sums require terminal aggregation without head.

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

Conditional Int64/UInt64 means use exact wide accumulation and can broadcast
through either branch, including when that branch is unselected. Empty and
all-null means remain null. Conditional scalar Float32 arithmetic uses exact
binary32 operations and can broadcast without exposing precision errors from
an unselected branch.
Nonthrowing Float64 leaves and direct Float32 column reductions can also
broadcast through conditional expressions. File-backed inputs still reject before source
reads. Automatic placement keeps these multi-region plans on CPU until it can
estimate the complete DAG.
Checked sums in child regions and checked arithmetic followed by head retain
their existing rejection boundaries. Unsupported windows and aggregate leaves retain their rejection boundaries.

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
dtype. They accumulate exactly in two words, so intermediate totals may exceed
64 bits and cancel before the final checked conversion. Null rows contribute
nothing; an unmet `min_count` suppresses the result and its overflow check.
A final total outside the result dtype raises an integer overflow error before
caller outputs change. A later query can reuse the runtime after rejection.
Float32 row arithmetic preserves separate operator rounding and disables shader
contraction and unsafe math optimizations. Metal can flush Float32 subnormal
operands and results even in safe math mode. Addition, subtraction and
multiplication retain native arithmetic for normal operands and results, then
use integer-only binary32 recovery for subnormal operands or underflow. True
division uses integer-only arithmetic and rounds once to Float32. Valid inputs
including signed zeros, subnormals, infinities and NaNs no longer raise a
precision error. Copying subnormal values and flipping their sign preserve
their bits. Float32 comparisons, predicates, clipping, floor, and ceiling use native
encodings and support subnormals. Float32 rounding widens the encoding to
emulated Float64, then narrows once, also preserving subnormal behavior.
Rebuild the optional native runtime for complete Float32 arithmetic. Estimates
use `metal-float32-arithmetic-v1`, invalidating earlier arithmetic calibration.
There is no automatic CPU retry after submission.

Float32 and Float64 sum/mean accumulate with emulated Float64 arithmetic;
Float32 sum narrows only once at completion, while means return Float64.
As in the CPU mergeable floating state, reduction reassociation is allowed and
bitwise reproducibility across execution schedules is not promised. Integer
means retain an exact integer total before Float64 conversion and division.
Int64/UInt64 inputs use the exact two-word accumulator without a value-dependent
range proof. Decimal128 sums and means use their precision-checked two-word
accumulator. Forced accelerator execution rejects statically unsupported
operations before submission; checked final sums can raise overflow during
execution.
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
containing Float64 values, floating sum/mean, rounding, square root, exponential/logarithm, floor division or remainder, or
Int64/UInt64-to-Float32 conversions carry the separate `metal-emulated-float64-v1` calibration family.
Native-only calibration cannot select those plans. Without matching, validated
end-to-end evidence, automatic placement stays on CPU. Future calibration may
select emulation only when its conservative GPU estimate beats the CPU bound;
forced accelerator execution remains available independently of that decision.

Natural `exp` and `log` execute on Metal for Float32, Float64, and primitive
integer inputs. Integer inputs return Float64; Float32 inputs retain Float32.
The shared integer-only binary64 helpers reduce `exp` around multiples of log(2)
and use a Taylor series, and normalize `log` around one with an atanh series.
They preserve subnormal inputs and outputs, quiet NaNs, and match zero/infinity
and domain classifications. These are numerical approximations: validation
allows 16 ULP against independent host libm rather than promising identical
vendor-libm bits. The GPU differential fixture covers all normal exponents,
boundaries around overflow/underflow and one, exceptional values, and random
encodings. Conditional expressions, resident consumers, and chunked inputs use
the same kernels. Estimates include separate exponential/logarithm node counts
in `transcendentals-v1:e<count>:l<count>` and corresponding `metal-` families.
Decimal inputs require an explicit Float64 cast, as with square root. CPU
expression kernels use system libm for logarithms to preserve positive infinity
and the existing numeric tolerance (the toolchain approximation misses it for
`log(10)`). Exponential NaNs and boundary lanes also use libm to avoid premature
overflow and retain gradual underflow; ordinary finite exponentials keep their
vector kernel.

Floating power is not yet supported by this subset.

## CPU parity work remaining

The Metal engine does not yet cover the complete CPU lazy-query surface.
The following gaps are visible in the public expression and plan APIs and the
current accelerator lowering; this is an implementation inventory, not a claim
that unlisted query combinations are already equivalent.

| Area | Remaining CPU operations or behavior |
| --- | --- |
| Numeric expressions | Floating power. |
| Reductions | Mode, implode, and value counts. |
| Ordered windows | Duration-based rolling sum/mean/min/max/var/std, `interpolate_by`, cut, and qcut. |
| Temporal expressions | Row-valued parsing/formatting and timezone operations. |
| Nested data | String split, list expressions, struct packing/fields, explode, and unnest. |
| Plans and sources | As-of joins and file-backed scans. |
| Composition | Remaining checked-expression/aggregation observation restrictions, including ordered windows inside grouped aggregation. |

Duration totals execute from ms/us/ns storage into days, hours, minutes,
seconds, milliseconds, microseconds, or nanoseconds. Coarser units truncate
toward zero without negating Int64.MIN; finer units retain the CPU kernel's
Int64 storage multiplication. Nulls and native row/group/window/join consumers
retain those semantics. `duration-totals-v1` and `metal-duration-totals-v1`
require separate calibration.

Date year, month, day, weekday, and ordinal fields preserve proleptic Gregorian
meaning across the complete Int64 day domain. CPU and Metal reduce days into
400-year cycle quotients and remainders before applying epoch alignment,
avoiding signed overflow. Boundary and deterministic full-width values are
checked against an independent Int128 calendar model.

Native validation and the backend contract suite cover each implemented feature.
They must be extended as these gaps close; passing the current suite alone does
not establish CPU parity. Fixed-size rolling variance/std now have an independent
host/GPU state-word differential test in addition to CPU query comparisons.

## Validation

```sh
python3 native/dfmetal/test_native.py
python3 native/dfmetal/test_join.py
python3 native/dfmetal/test_resident.py
python3 native/dfmetal/test_float64.py
python3 native/dfmetal/test_moments.py
python3 native/dfmetal/test_higher_moments.py
python3 native/dfmetal/test_interpolation.py build/dfmetal/libdfmetal.dylib
python3 native/dfmetal/test_rolling_moments.py
python3 native/dfmetal/test_decimal.py
python3 native/dfmetal/test_decimal_casts.py
python3 native/dfmetal/test_decimal_comparisons.py
python3 native/dfmetal/test_decimal_arithmetic.py
python3 native/dfmetal/test_decimal_multiplicative.py
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
pixi run mojo run -I . tests/test_metal_decimal_means.mojo
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
`dictionary_operation_tables` counts generated unary and concatenation lookup tables.
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
