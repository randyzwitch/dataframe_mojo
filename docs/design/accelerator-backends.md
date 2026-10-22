# Accelerator backends

The boundary between the core and a backend, and how a backend attaches, is
specified in [accelerator-contract.md](accelerator-contract.md).

Optional backends use the same `DataFrame`, expressions, binder, and lazy
collection API as CPU execution. An explicit runtime implements
`dataframe.lazy.AcceleratorBackend`; an optional distribution can register its
default through `dataframe._accel_provider`, which re-exports the
`dataframe_provider` first on the import path. Metal is the default provider
of the open-source package (the root `dataframe_provider`); another backend,
such as NVIDIA in the extensions repository, supplies its own
`dataframe_provider` ahead of it, and `providers/cpu` is the CPU-only choice.
Ordinary CPU imports and runs must not require a driver or a physical GPU.

`dataframe_metal.accelerator` supplies CPU-only row-plan lowering, semantic capability
checks, and overflow-checked requested-memory estimates. It does not discover
devices, allocate accelerator storage, scan source values, or submit work.
Backends own their runtime, physical storage, kernels, synchronization, and
capability and cost facts for the shared core policy. Applications continue to use `collect`, `profile`, and
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
without a final head, unless the backend sets `adjudicated_faults`. This
preserves the current CPU optimizer's observable overflow behavior. Forced
acceleration rejects unsupported regions before submission. Automatic
selection may choose CPU, but execution errors after submission must
propagate rather than retrying on CPU. The one exception is a data fault
(`dataframe_metal/accelerator/faults.mojo`): a backend with `adjudicated_faults`
computes every checked value and lets the CPU decide a query whose kernels
wrote a fault, since the CPU's evaluation order defines which error, if any,
the query raises.

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

The Apple provider lives in this repository, in the `dataframe_metal` package, and runs Mojo kernels through MAX's Metal device API.
It accelerates supported Float32, integer, Boolean, and integer-emulated
Float64 regions, with no implicit downcasts. Emulated binary64 work uses a
separate calibration family; uncalibrated plans and unsupported operations
remain CPU candidates. See [Apple Metal](../apple-metal.md) for the current
operation and precision coverage. GPU selection needs measured
complete-query benefit including initialization, allocation, transfers,
launches, synchronization, and host result construction. Resident-kernel
speedups alone do not justify automatic selection.

The implementation is delivered separately from this shared-planning change.
CUDA runtime and kernel code remain in the NVIDIA extension. Existing providers
can adopt the common lowering and estimates without changing the runtime trait
or application syntax.

## Shared analytical automatic placement (prepared; validation pending)

`dataframe_metal.accelerator.cost.select_auto` owns the CPU/accelerator comparison.
Providers implement `estimate_auto(plan) -> PlanEstimate`: two feature vectors,
peak requested bytes, dtype family, and candidate placement. Estimation binds
metadata without reading source values or submitting device work. Runtime
backends can expose the same method; the backend exports it for calibration. Existing third-party providers inherit a rejecting
estimate and keep their CPU default.

The ordered `COST_FEATURES` schema covers fixed overhead, rows, source bytes,
projection/filter/reduction work, integer and cast work, launches, explicit
synchronizations, upload/download bytes, allocation, cold first use, and CPU
materialization stages. CPU work comes from the shared `cpu_features` binder.
Providers supply hardware facts and coefficients, never independent CPU
thresholds. Memory is checked before comparing finite cost bounds. A validated
CPU lower estimate must exceed the accelerator upper estimate by at least
1.5×. The decision reason reports both estimates and the largest deciding term;
existing explain/profile selection reporting carries it to the user.

Calibration records use `dataframe-analytical-cost-v1`, keyed by provider,
device, runtime build, CPU build, host/affinity, workers, optimize/streaming/
batch settings, and dtype family. They contain nonnegative coefficients,
feature coverage bounds, residual envelopes, expiry and independent query
validation counts and a zero failed-execution count. A held-out failure after
estimate/identity emission invalidates that calibration identity; it cannot be
hidden by the remaining successful cases. Unidentified builds, absent/stale/ambiguous records,
extrapolation and unsupported candidates select CPU before submission. Device
failures continue to propagate. CPU tails currently require additional cost
terms and conservatively stay on CPU; region dynamic programming and learned
corrections in #583 remain separate phases.

CPU profiles now include `cpu_build_id`. Source execution uses a development
sentinel and cannot produce deployable calibration. `scripts/build_package.py`
stages a fingerprint of core Mojo sources, lockfile, compiler and compile
settings; provider builders use `scripts/build_identity.py` for the same CPU
identity before substituting their registration slot.

The provider-neutral `benchmarks/bench_accelerator_cost.mojo` runner requires
`calibration_runtime`, `calibration_key` and `estimate_auto` hooks from the
backend (`dataframe_metal.metal` for Metal). They are not part of the core's provider
contract. `benchmarks/tools/bench_accelerator_cost.py` writes a generated-plan
manifest by default; `--collect --binary /path/to/runner` explicitly measures.
Compile the runner against identified packages outside the source import path.
The matrix varies 1–61 expression nodes, pipeline stages, dtype, filter
position/selectivity, null density and 4K–32M rows. Failed/unsupported cases
remain in JSONL. Warm samples alternate engines; cold observations include
runtime creation. Measured automatic preflight overhead is included in GPU
cost, and returned result parity is checked outside timings.

Fit on generated queries; freeze coefficients and residual envelopes before
validating named chain/count and other mechanisms. Canonical operation hashes
prevent the same plan with another display name or row count from leaking into
both sets. Reports replay choices against measured explicit-engine samples and preflight
overhead; automatic-execution latency is not directly measured. Invalid
calibrations report the CPU fallback while retaining hypothetical model choices
for diagnosis. Reports include coverage, wrong choices, engine regret and total
regret including planning overhead. Named mechanisms were previously used in
development and are held out only from this fit. ClickBench policy in
`benchmarks/tools/bench_policy.py` remains validation-only. No external-suite coverage or
measured speedup is claimed by these source changes.

Compilation, CPU/GPU tests, package checks, calibration and held-out runs are
pending. A generated table remains unusable unless independent parity and
residual-envelope validation succeed; no fitted coefficients ship by default.
