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

Temporal and Decimal32/64 columns retain their complete logical metadata
through projection, stable filtering, null tests, fill-null, counts, and
extrema. Comparisons use the native physical integer representation; decimal
comparisons currently require matching scales and storage widths. Logical
arithmetic and casts are checked separately and currently require CPU
execution. Decimal sums retain the CPU contract of Decimal128 accumulation
and require CPU execution; Decimal128 storage is not yet supported by this
row backend.

Grouped aggregation supports multiple keys with minimum, maximum, count,
length, and the supported integer sums. Null keys form a group; Float32 NaNs
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

The wrapper and native library use ABI version 2; rebuild the optional library
when upgrading. Homogeneous plans retain their original native-width matrices.
Mixed plans and extrema reductions store each value as an exact 64-bit word,
with generated native operations for each expression dtype. Numeric inputs are staged at their original
width and packed on the GPU. Output words are copied into the matching CPU
column width after synchronization; CPU code does not evaluate expressions.

The native bridge uses documented Metal shared-storage buffers with the default
CPU cache mode. It stages input bytes once, queues all dependent kernels, waits
once, and copies the final result into CPU-owned columns. Shared memory still
requires synchronization; CPU code never reads an in-flight GPU result.

Consecutive projection steps, including the next filter predicate, execute in
one specialized kernel. Intermediate expression values stay in thread-local
variables. Shared matrices hold input columns, final outputs, filter predicates,
and values needed after a filter boundary. Dead temporary slots are omitted from
stable gathers. Operator rounding, validity, and precision/overflow guards remain
in the fused expression sequence.

Preflight uses this same physical slot layout and counts each requested shared
buffer once, both matrices when filtering,
prefix ranks and block offsets, bitmap windows, typed literals, output bitmaps,
reduction partials/results, GPU descriptors, metadata, and error storage. It also
counts the worst-case CPU result allocation. Checked arithmetic rejects size
overflow. The explicit payload budget is limited by current recommended Metal
working-set headroom. Planning descriptors, compiler/cache allocations, allocator
rounding, retained source columns, and unrelated process memory are outside this
payload estimate; it is not an operating-system memory limit.

Profiles report observed launch and wait counts; upload/download traffic;
requested shared/result bytes; pipeline cache hits and compilation time;
staging-copy, submission-and-wait, and result-copy time; and the complete Metal
command-buffer GPU interval when available. `gpu_ms` is null if Metal supplies no
valid interval. It is not a sum of isolated kernel durations. Whole-query
`wall_ms` includes binding, initialization, compilation, allocation, staging,
execution, and result construction. `runtime_build_id` fingerprints the native
source, ABI, compiler, SDK, and build flags.

Submission errors propagate. The bridge drains committed work before releasing
buffers or returning an error; execution never retries the query on CPU after a
GPU fault. Output owners and source owners remain alive across that boundary.

## Validation

```sh
python3 native/dfmetal/test_native.py
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
