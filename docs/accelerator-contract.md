# Accelerator execution contract

This is the shared execution and memory design for [#528](https://github.com/randyzwitch/dataframe_mojo/issues/528),
Phase 0. It specifies the target contract for NVIDIA discrete memory and an
Apple shared-memory backend. A requirement below is not a claim that both
backends implement it. The implementation/status table at the end separates
what exists from the remaining work.

## Public API and semantic boundary

The existing `DataFrame`, `Series`, `Column`, expressions and lazy plans remain
the application model. Collection returns the existing CPU-accessible
`DataFrame`; dtype, field names, row count, null behavior and ordering are
independent of placement. Eager execution can stay on CPU. No application
needs a vendor import, explicit upload, device handle, cast or GPU-only result.
An optional distribution may register a provider; CPU installations must not
load GPU libraries merely to import or execute dataframe code.

The target default for ordinary lazy collection is the automatic policy below.
Current code still uses CPU by default and `auto` still selects CPU. Enabling
automatic placement is Phase 2, not part of declaring this contract.

| Control | Required behavior |
|---|---|
| `engine="cpu"` | CPU execution, without device discovery or validation of GPU-only settings. |
| `engine="auto"` | Choose only a fully supported, memory-feasible region with credible end-to-end cost evidence; otherwise CPU. |
| `engine="accel"` | Require accelerator execution for the complete requested region; explain unsupported capability or preflight rejection and raise before submission. |
| Explicit diagnostic runtime/device | Override registered provider defaults; preserve the same logical API and semantics. |

A diagnostic override must not silently modify the automatic calibration or
become required application syntax. Invalid explicit options raise. `explain`
and profiling expose the decision and observed execution separately from the
ordinary result. Schema inspection must not execute a query to learn its types.

Binding happens before backend selection. The shared binder defines literal
coercion, intermediate and output dtypes, accumulator types, null propagation,
scalar broadcasting, aliases and invalid-expression errors. Providers consume
that bound meaning rather than inventing a parallel expression language.

Floating row arithmetic rounds in its bound dtype. Reductions use the CPU
contract's accumulator type and final conversion; allowed reassociation does
not permit Float64-to-Float32 substitution. Integer row operations retain
checked overflow semantics. Integer reductions require exact wide partials
and a final range check; partial Int64 overflow followed by cancellation must
not reject a mathematically representable final result. Counts retain Int64.

A filter accepts only a valid true predicate. Stable filtering/gather preserves
input order, duplicate rows and the association of values with their validity
bits. Null payloads must not be evaluated by dangerous arithmetic. Boolean
operations use the existing three-valued/Kleene semantics, including valid
false AND null and valid true OR null. An implementation may use a different
internal physical Boolean representation, but must restore the normal packed
representation and logical Bool dtype at the public boundary.

## Capability model

Capability is a property of a bound region and its backend, not just an input
dtype or device. The provider returns either a supported physical description
or a structured rejection with a stable category and the offending plan node.
The first implementation can use explicit checks; a registration table is not
required. The logical requirements are:

| Capability field | What it must establish |
|---|---|
| Backend/toolchain/device | The installed provider, compiled kernel ABI and physical device support the operation. |
| Input layout | Dtypes, chunking, strides, slices, bitmap offsets and accessibility are accepted. |
| Expression | Every operator, literal and intermediate dtype is supported, including predicate subexpressions. |
| Reduction | Accumulator precision, min_count, empty/all-null behavior and final overflow/conversion match CPU. |
| Shape/order | Projection widths, scalar broadcasting, stable compaction and requested head/slice behavior are preserved. |
| Ownership/dependencies | Sources can remain alive and accessible through all submitted work; pending producers can be awaited. |
| Memory description | Inputs, outputs, masks, indices and peak simultaneously live temporary allocations can be bounded without executing the region. |

Unsupported chunking, multi-column expressions or operation depth are explicit
capability rejections until implemented; the provider must not silently run
the unsupported portion on CPU in forced mode. A future mixed planner can
split only at visible region boundaries, with each transfer charged once.

## Conservative automatic-selection policy

Selection is a pre-submission decision with these ordered stages:

1. Honor explicit CPU execution. For automatic execution, absence of a
   registered/available backend is an ordinary CPU decision.
2. Bind and validate the logical query independently of device support. An
   invalid expression is still an error, not an excuse to retry on another
   backend.
3. Form maximal contiguous supported regions. Initially require one complete
   supported pipeline; mixed execution remains disabled until its boundary
   costs are measured.
4. Compute liveness-aware memory bounds and compare them with the provider
   payload cap and current device headroom. Unknown/unbounded memory, arithmetic
   overflow in an estimate, or insufficient headroom selects CPU in automatic
   mode and rejects forced acceleration.
5. Look up cost evidence for this backend, device/toolchain, CPU thread budget,
   input accessibility, dtype, operation family, expression/aggregate count,
   null/selectivity range, input/output size and workspace. Missing, stale or
   nonmatching evidence selects CPU. Unknown selectivity must use a conservative
   range; it must not trigger an untimed scan whose cost is omitted.
6. Compare complete-region costs, including initialization when not yet paid,
   transfers, allocation, launches, synchronization and result construction.
   Choose GPU only if its upper cost estimate has a configured safety margin
   below the lower CPU estimate. Record the estimates and reason. Otherwise CPU.

There is no universal row-count cutoff and no monotonicity assumption. The
[calibration work in #547](https://github.com/randyzwitch/dataframe_mojo/pull/547)
shows why CPU thread setup and parallel throughput both matter. Its development
rule, `1.25 * max(observed GPU collect) < min(observed CPU collect)`, is an
empirical envelope, not a statistical confidence bound or a production policy.
The 25% factor is a recorded experiment setting, not a hard-coded planner
constant. New devices, thread budgets and query families require matching
validation before automatic enablement. Shared-desktop measurements alone do
not justify activating a calibration.

For repeated workloads, charge startup once only if the runtime actually
retains usable initialized state. A possible session comparison is
`I + F + (N - 1) * G` against `N * C`, using conservative initialization,
first-query and warm-query costs. Do not assume future reuse or resident input
when it is unknown. Device-resident inputs may avoid a copy only if their
ownership, context, contents and readiness are valid for this execution.

Publish calibration identity, coverage and uncertainty alongside any enabled
policy. Tests must force both winning and losing decisions using deterministic
cost fixtures; timing noise must not decide unit-test outcomes. Real hardware
validation must then confirm that those fixtures correspond to supported work
and do not hide accelerator failures behind CPU selection.

## Allocation, accessibility and ownership

Each internal allocation has an identity and generation, byte extent,
alignment, logical dtype/layout, owning backend/context, allocation class,
host/device access permissions, and a completion dependency. Each view retains
the allocation owner and carries value offset/stride, row count, validity
allocation and bit offset. A host address is not a device capability.

| Allocation class | Host access | Device access | Required transition |
|---|---|---|---|
| Ordinary host memory | Yes | Only if explicitly supported/registered by the runtime | Otherwise allocate device storage and copy the selected windows. |
| Discrete device memory | No | Owning context/device | Explicit download into retained host storage before host use. |
| Runtime-created shared memory | According to runtime contract | According to runtime contract | Producer completion plus required visibility/coherency operations before the other processor reads. |
| Registered/pinned host memory | According to registration contract | Only where that registration permits it | Retain registration and original owner until all users complete. |

The allocation classes are capabilities, not hardware-name shortcuts. An
Apple unified physical memory system does not authorize exposing an arbitrary
`ArcPointer[List[T]]` to a kernel. The Apple provider must obtain a runtime-
validated shared allocation or register a supported allocation, and record the
actual accessibility and any required staging copy. Host and device virtual
addresses need not be equal. Unsupported registration falls back to an explicit
copy or a CPU plan before submission.

Current immutable host columns can be shared without copying while a queued
upload reads them, provided the source owner is retained. A device expression
writes a fresh output or exclusively owned scratch; it cannot overwrite an
input alias. A view cannot outlive the allocation owner it retains. Reusable
scratch is a lease: reuse is permitted only after the prior completion token
is satisfied, with no outstanding views that could observe the next contents.
Cache keys must include allocation identity/generation, layout, dtype, context
and slice/validity offsets; a recycled raw address is not a cache identity.

For shared allocations, expose a host-readable result only after acquiring
producer completion and any backend-required visibility operation. Host writes
must finish and become visible before GPU submission. Public immutability
eliminates conflicting writes through the dataframe API; it does not eliminate
these producer/consumer transitions or external-buffer ownership obligations.

## Readiness and queued work

Accessibility answers *who can address the allocation*. Readiness answers
*which producer has completed and whose writes are visible*. Keep them separate.
A confirmed-ready flag is not a live device query and cannot substitute for
a dependency when new work is submitted.

The internal transition model is:

- **Owned / host-ready:** a retained host producer has completed; upload or
  shared-memory publication may be queued.
- **Device-pending:** queued uploads/kernels retain every input, output,
  descriptor, context, registration and host pointer they may access.
- **Device-ready:** the producer token completed successfully; a dependent
  device consumer may read it. Same-stream ordering can satisfy the dependency
  without a host wait.
- **Host-pending:** download or shared-memory acquisition has been queued;
  the host destination and its registration stay alive.
- **Host-ready:** transfer/acquisition completed successfully and public CPU
  access is permitted.
- **Failed:** no result is published. Cleanup retains resources until it can
  establish that pending users have stopped.

Cross-stream or cross-context consumption requires an explicit supported wait
or transfer. Equality of physical device IDs is insufficient. Initial NVIDIA
submission remains serial through one context/ordered stream; later concurrent
submission must preserve these dependencies rather than borrowing an unsafely
shared scratch pool.

Destruction and exception unwinding drain pending users before releasing host
pointers or device scratch. If synchronization fails and safety cannot be
established, do not free potentially in-use memory: use a documented fatal
cleanup path or retain/quarantine resources until context teardown. The current
NVIDIA implementation aborts on an unresolvable destructor drain. A successful
error return may not conceal a use-after-free or background GPU work.

## Memory budgeting and failure policy

Estimate peak *simultaneously live* requested payload, not the sum of all
allocations ever made. Include source values/validity, expression descriptors,
resident intermediate values/validity, selection masks, prefix counts/indices,
compaction destinations, reduction partials and final outputs. Stable gather
usually needs distinct source/destination storage. Use checked size arithmetic
and a worst-case output cardinality when the exact selection is not known.

Discrete host/device copies consume separate allocations. A shared allocation
is charged once to the shared physical budget; retained staging copies are
additional. Shared-memory capacity is constrained by system headroom and
concurrent CPU/GPU use, not an assumed independent VRAM pool. Record requested
payload separately from SDK reservations, fragmentation and process-wide peaks.
A free-memory snapshot and a payload cap are preflight checks, not hard physical
memory limits. Backend-specific reserve/headroom policies need measurement.

Before submission, unsupported capability, unavailable devices and inadequate
estimated memory are planned CPU decisions under automatic mode. An actual
allocation failure after choosing GPU is an execution error in the initial
policy: drain/clean up and propagate it. Do not automatically retry on CPU.
Kernel errors, semantic overflow and driver faults likewise propagate. Any
future allocation-only retry must be separately specified with a proven
no-side-effect boundary and visible diagnostics; catching every exception and
rerunning on CPU is forbidden.

## Required diagnostics and acceptance evidence

Explain must identify requested and chosen backend, supported/rejected region,
rejection category, device/calibration identity when applicable, input and
output accessibility, estimated transfer/workspace/peak payload, cost bounds,
and every materialization/ownership transition. CPU-only explanation must not
initialize a GPU. Actual profile data adds observed rows, transfers, launches,
waits and wall/event timing. Unknown values remain unknown, not fabricated zero.
Compilation, first initialization, first query and warm collection are separate
measurement boundaries. Event intervals do not replace end-to-end timing.

| Requirement | Current evidence / next gate |
|---|---|
| Shared API controls and binding | #541 and #544; registration and memory diagnostics in #545 (including #546). |
| NVIDIA ownership/readiness and exact window transfers | #542 runtime tests, including null/slice/empty cases and exceptional unwinding. |
| Float filter/arithmetic/reduction semantics | #544 CPU/GPU parity, sanitizer and profiler evidence. |
| End-to-end calibration methodology | #529 mechanism experiment; #547 public-executor sweeps and retained raw outcomes. |
| Automatic policy implementation | Phase 2: deterministic decision tests, matched calibration validation and explicit reasons; currently CPU-only. |
| Stable row results and Boolean expressions | Phase 1: exact order/schema/validity checks, byte-boundary and all-null tests, race checking of packed writes. |
| Integer execution | Phase 1: checked row arithmetic, exact wide partial sums, cancellation and final overflow tests. |
| Shared/unified allocation implementation | Phase 3: actual Apple toolchain/device validation, alias/lifetime tests, producer/consumer visibility and measured copy counts. |
| Fault/memory-pressure validation | Device-loss/driver/OOM injection remains an explicit gate; ordinary exception unwinding is not evidence for those faults. |

The shared-memory design can be reviewed without Apple hardware. Its runtime
implementation, performance, Float64 limitations and synchronization guarantees
must be validated on supported Apple hardware before that backend is enabled.
