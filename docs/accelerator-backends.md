# Accelerator backends

Optional backends use the same `DataFrame`, expressions, binder, and lazy
collection API as CPU execution. An explicit runtime implements
`dataframe.lazy.AcceleratorBackend`; an optional distribution can register its
default through `dataframe._accel_provider`. Ordinary CPU imports and builds
must not require MAX, a driver, or a physical GPU.

`dataframe.accelerator` supplies CPU-only row-plan lowering, semantic capability
checks, and overflow-checked requested-memory estimates. It does not discover
devices, allocate accelerator storage, scan source values, or submit work.
Backends own their runtime, physical storage, kernels, synchronization, and
performance policy. Applications continue to use `collect`, `profile`, and
`explain` with `engine="cpu"`, `"auto"`, or `"accel"`.

## Bound row plans

`lower_rows(query, capabilities)` binds a complete bounded in-memory region
before device submission. The supported common subset includes select,
with-columns, drop, stable filter, a final head, and final sum/count/mean/len
reductions where the backend can implement their semantics. Existing limits
remain explicit: at most 64 source/intermediate slots, 64 nodes per row
expression, and 64 plan steps; one common numeric dtype plus Boolean columns;
no chunked input. Unsupported plans raise with a backend and rejection category.

The result describes source columns, row programs, materialization steps,
gathers, output schema, and reductions. Source owners are retained. The shared
binder controls intermediate dtypes, scalar broadcasting, null behavior, and
literal coercion. Integer literals retain their raw bits, including values
above 2**53. The host-side literal table uses Float64 as a bit container;
backends must encode literals into supported device storage before submission.
It is not permission to run Float64 instructions on a backend lacking them.

`RowCapabilities` distinguishes source Float64, checked Int64 arithmetic, and
exact wide integer accumulation. Float32 sum and mean still require the
library's Float64 accumulator contract. A backend without Float64 rejects
those reductions; it must not silently accumulate in Float32. Int32 sum can
use an exact Int64 accumulator for at most Int32.MAX input rows. Int64 sum
requires exact wide partials so intermediate overflow followed by cancellation
does not reject a representable final result.

Checked integer arithmetic remains at an observable terminal projection,
without a final head. This preserves the current CPU optimizer's observable
overflow behavior. Forced acceleration rejects unsupported regions before
submission. Automatic selection may choose CPU, but execution errors after
submission must propagate rather than retrying on CPU.

## Memory and execution ownership

`row_memory` estimates requested payload for the common resident matrix layout,
including validity, filter compaction, descriptors, reductions, and errors.
Launch dimensions and wide-accumulator storage are backend parameters. It is
one physical layout estimate, not a requirement to use that layout. A backend
with fused expressions or different allocation lifetimes must account for its
own peak simultaneously live buffers. Uploaded bytes are traffic, not an
additional resident allocation. Payload estimates are distinct from allocator
reservations and other processes' memory use.

Backend runtime objects retain sources and queued work until completion.
Aliased immutable inputs must not be overwritten. Host-readable results are
published only after producer completion and required visibility operations.
An Apple unified-memory system does not make arbitrary host pointers legal
kernel arguments: shared storage must be created or registered through the
runtime. Backend memory budgets must reflect actual shared or discrete storage
and include any staging allocations.

## Apple implementation scope

The Apple provider lives in this repository and uses native Metal operations.
It will accelerate supported Float32, integer, and Boolean regions, with no
Float64 emulation or implicit downcasts. Float64 work and reductions requiring
unsupported precision remain CPU candidates. GPU selection needs measured
complete-query benefit including initialization, allocation, transfers,
launches, synchronization, and host result construction. Resident-kernel
speedups alone do not justify automatic selection.

The implementation is delivered separately from this shared-planning change.
CUDA runtime and kernel code remain in the NVIDIA extension. Existing providers
can adopt the common lowering and estimates without changing the runtime trait
or application syntax.
