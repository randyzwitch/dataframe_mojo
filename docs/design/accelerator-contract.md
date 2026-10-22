# Accelerator contract

This note says what an accelerator backend must provide to the core, what the
core promises in return, and how a backend is attached. It is written against
the tree at `12980823`. It supersedes the draft in the private extensions
repository (`dataframe_mojo_extensions/docs/accelerator-contract.md`) for
everything about the core boundary. That draft still holds the NVIDIA-side
memory, ownership and calibration design.

[accelerator-backends.md](accelerator-backends.md) describes the planner-side
machinery (row-plan lowering, memory estimates, calibration). This note is the
boundary: what crosses it, and what must not.

The boundary is checked in CI by `scripts/check_accelerator_boundary.sh`.

## The provider interface

`dataframe/_accel_provider.mojo` is the only module the core calls to reach a
backend. Only `dataframe/lazy.mojo` and `dataframe/execution_backend.mojo`
import it. A provider is a module with these functions. `LazyFrame` and
`DataFrame` are the core types; `PlanEstimate` and `CalibrationKey` are in
`dataframe/accelerator/cost.mojo`.

| Function | Signature | What the core expects |
| --- | --- | --- |
| `installed` | `() -> Bool` | Whether the backend could run here. Must not open a driver, create a context or allocate. `engine="accel"` uses it to decide availability, and importing or CPU-collecting must never initialize a device. |
| `execute` | `(plan: LazyFrame) raises -> Tuple[DataFrame, DataFrame]` | Run the whole plan. Returns the result and a report frame. Raises for unsupported work before submitting anything. Called for `engine="accel"`. |
| `describe` | `(plan: LazyFrame) -> String` | Text for `explain`, as `ENGINE accel: ...` lines. Does not raise: errors become text. Must not execute the plan or initialize a device. |
| `execute_profiled` | `(plan: LazyFrame) raises -> Tuple[DataFrame, DataFrame]` | Same as `execute`, with measured data in the report. May return `execute`'s report unchanged. Called by `profile`. |
| `calibration_runtime` | `() raises -> Runtime` | A runtime object for the calibration benchmark driver (`benchmarks/bench_accelerator_cost.mojo`). Not used by `collect`. The return type is backend-specific. |
| `calibration_key` | `(optimize: Bool, streaming: Bool, batch_size: Int) raises -> CalibrationKey` | Identity of the provider, device, build and CPU thread budget under which a calibration is valid. No submission. |
| `estimate_auto` | `(plan: LazyFrame) raises -> PlanEstimate` | Plan-time cost and peak-byte facts for the calibration runner. No device submission and no reads of source values. Raise if the plan has no estimate. |
| `select_auto` | `(plan, optimize, streaming, batch_size) -> Tuple[String, String]` | Used only for `engine="auto"`. Returns `(engine, reason)` with engine `"cpu"`, `"accel"` or `"mixed"`; anything else is an error. Does not raise: a rejection becomes `("cpu", reason)`. Must not execute the plan. |
| `execute_auto` | `(plan, optimize, streaming, batch_size, profiling: Bool) raises -> Tuple[DataFrame, DataFrame]` | Run a region `select_auto` chose. Errors propagate; the core does not retry on CPU (one exception, below). |

The core calls `installed`, `execute`, `describe`, `execute_profiled`,
`select_auto` and `execute_auto` during collection. `calibration_runtime`,
`calibration_key` and `estimate_auto` are called only by benchmark and
calibration code, but the provider slot must still define them so those
programs compile.

The same shapes exist as the `dataframe.lazy.AcceleratorBackend` trait, for an
explicit runtime object passed by the application. Its methods other than
`execute` and `describe` have defaults (CPU in auto mode, no estimate), so a
minimal backend implements only those two.

### Engine settings the provider honors

`collect`, `profile` and `explain` take `engine`, `optimize`, `streaming` and
`batch_size`.

- `engine`: `"cpu"` never reaches the provider. `"auto"` calls `select_auto`,
  and calls `execute_auto` only if it returned `"accel"` or `"mixed"`.
  `"accel"` requires `installed()`, then calls `execute` (or
  `execute_profiled`). An unknown name raises in the core.
- `optimize`, `streaming`, `batch_size`: passed to `select_auto`,
  `execute_auto` and `calibration_key`. They are the requested CPU-side
  settings, carried in `ExecutionSettings`
  (`dataframe/execution_settings.mojo`). A backend must not treat `batch_size`
  as an expression batch size. They are part of the calibration identity.
- `DATAFRAME_ACCEL_DEVICE` and `DATAFRAME_ACCEL_MEMORY_LIMIT`: read by the
  default provider, not by the core. A bad value raises; they never change
  results. A replacement provider should keep the names so applications can
  move between backends.

## What the core guarantees, wherever the work runs

- **Same results.** Column names, dtypes, row count, null behavior and result
  ordering do not depend on placement. A backend returns a normal CPU
  `DataFrame`, with normal packed validity. Result types are the core's
  (decimal precision and scale follow DuckDB); a backend may not change them.
- **Binding first.** The core binds the query (literal coercion, intermediate
  and output dtypes, accumulator types, null propagation) before choosing a
  backend. A backend consumes the bound meaning and does not reinterpret an
  expression. An invalid expression is an error, not a reason to try another
  engine.
- **Float and integer rules.** Reductions use the CPU accumulator type and
  final conversion; a backend without Float64 rejects Float64-accumulated
  reductions instead of accumulating in Float32. Integer row arithmetic is
  checked. Integer sums keep exact wide partials, so overflow followed by
  cancellation does not reject a representable result. The detail is in
  accelerator-backends.md.
- **Ordering.** Filters are stable. Sort ties are unspecified unless
  `maintain_order=True`, as in Polars, so a backend may break ties differently
  from the CPU but must still return a valid order.
- **Errors propagate.** A rejected forced-accelerator query raises before
  submission. Automatic selection may choose CPU; once work is submitted,
  allocation failures, kernel errors and driver faults propagate. The core does
  not catch them and rerun on CPU.
- **Data faults are adjudicated by the CPU.** This is the one rerun. A data
  fault is a value-dependent error a kernel can only flag (checked overflow,
  for example; `dataframe/accelerator/faults.mojo`). A backend that computes
  every checked value and reports a fault lets the CPU rerun the query, because
  the CPU's evaluation order decides which error, if any, the query raises.
  Budget, planning and device failures are not data faults.
- **CPU never depends on a backend at run time.** `engine="cpu"` and the
  default `auto` without an installed backend never initialize a device or load
  a GPU library.

## What a backend may not do

- Reach into core internals beyond the provider functions above and the public
  types they carry (`LazyFrame`, `DataFrame`, `Series`, `Column`, expression
  and plan-node structures). Today the Metal backend does import many core
  modules; those imports are the surface the migration must name and narrow.
- Be imported by core code other than through `_accel_provider`. The check
  fails if any core file outside the provider and Metal files imports
  `dataframe.accelerator` or `dataframe.metal`. (One known exception exists;
  see the status section.)
- Import from `tests/`. Backend code is part of the package; tests are not.
- Change result types, add a GPU-only result type, or require an application
  to call a vendor API, upload data or hold a device handle.
- Run an unsupported part of a forced-accelerator plan on CPU without saying
  so, or retry on CPU after submitting work (except a data fault).
- Open a device in `installed`, `describe`, `select_auto`, `estimate_auto` or
  `calibration_key`.
- Free host memory or device scratch while queued work may still use it.

## Registration model

**Today: the provider file is replaced at build.** `_accel_provider.mojo`
ships with the Metal import hard-coded. The NVIDIA package builder
(`dataframe_mojo_extensions/scripts/build_accel_package.py`) copies the core
into a temporary tree, overwrites `dataframe/_accel_provider.mojo` with a file
that re-exports from `dataframe_accel.provider`, and precompiles both packages.
Consequences:

- Every build on every platform compiles the Metal backend (155 files, about
  60,000 lines under `dataframe/accelerator/` plus `dataframe/metal.mojo`),
  because the provider imports it. A local measurement put each test program at
  6 to 13 minutes of compile for about 90 seconds of run time. A Linux CPU
  installation pays this for code it cannot run.
- Two backends cannot be present at once. Choosing one means editing a core
  file.
- The replacement must track the provider interface by hand. It is a second
  copy of the contract with no check against the first.

**Target: a backend package the application imports.** A backend is a separate
package that registers itself with a small registry in core, and the core
never names a backend. Core's `_accel_provider` holds the registry and
dispatches through it; `installed()` is false when nothing registered. An
application that wants a GPU writes one import (for example
`import dataframe_metal`), and a CPU-only application compiles no GPU code. The
provider interface above becomes the registry's trait, so a backend that does
not match fails to compile in its own package, not in core.

## Status

| Backend | Where it lives | How it attaches | What exists | What is missing |
| --- | --- | --- | --- | --- |
| Metal | In this repository: `dataframe/accelerator/`, `dataframe/metal.mojo` | Hard-imported by `_accel_provider.mojo` | Runs on Apple silicon through Mojo/MAX kernels. Implements all nine provider functions. Tested on a local Mac (hosted CI cannot run it). | Compiled on every platform. No registry. Core (`lazy.mojo`) imports five modules of the backend package for plan-node and estimate types. |
| NVIDIA | Private repository `dataframe_mojo_extensions`, package `dataframe_accel` | Build step replaces `_accel_provider.mojo` | Float64 and Int32/Int64 row paths with parity tests on Linux x86-64. Builds against the core revision pinned in `core.json`. | Its replacement file re-exports four names (`installed`, `execute`, `execute_profiled`, `describe`), but `lazy.mojo` on main also imports `select_auto` and `execute_auto`, so it cannot build against current main without being updated. It has no calibration functions. Pinned to an older core revision. |
| Other | Nothing | n/a | n/a | A third backend would have to replace the provider file too, so it could not coexist with either. |

Known boundary violation. `dataframe/lazy.mojo` imports `PlanEstimate`,
`WindowKey`, `KeyQuery`, `ByteDomain` and `TextField` from
`dataframe.accelerator`. They are plan-node field types and an estimate type
that the planner and the `AcceleratorBackend` trait use, and they live in the
backend package. The check lists these five imports as a known exception, in
the script, so the boundary is enforced for everything else today. Step 2
below moves these types to the core, and the exception list is then emptied.
Do not add to it.

## Migration plan

Each step is its own pull request.

1. **This note and the boundary check.** No code moves. The check passes on
   main with the one recorded exception.
2. **Registry in `_accel_provider`, and shared types in core.** Replace the
   hard-coded dispatch with a registry (a trait for the nine functions plus
   `register`). The Metal import becomes one line in one file. Move
   `PlanEstimate`, `CalibrationKey`, `WindowKey`, `KeyQuery`, `ByteDomain` and
   `TextField` into core modules (they are value types the planner owns), and
   empty the exception list in the check. The NVIDIA draft is updated to this
   trait at the same time.
3. **Move the Metal backend into its own package.** `dataframe/accelerator/`
   and `dataframe/metal.mojo` go to a `dataframe_metal` package that registers
   itself on import. I recommend a top-level `dataframe_metal/` over
   `dataframe/backends/metal/`: packaging the `dataframe` package (`mojo
   package`, the `.mojoc` the build scripts produce) would still carry a nested
   directory, while a top-level package is built, versioned and shipped
   separately and left out of a CPU installation entirely. (Not measured here;
   step 5 confirms the compile drop.) It also matches how the
   extensions repository already lays out `dataframe_accel`, so both backends
   have the same shape. The backend's remaining core imports are then the list
   to shrink, and the boundary check grows to cover the new package.
4. **Build the NVIDIA extension against the same registry.** Drop the file
   replacement from `build_accel_package.py`; the extension imports the
   registry and registers itself. Repin `core.json`.
5. **Measure the compile-time drop on Linux.** Time a test program and a CI run
   before and after step 3, with the same method as the 6 to 13 minute
   measurement, and record it in the PR.

Coordination cost. Several Metal pull requests are open at any time, and step
3 renames every file under `dataframe/accelerator/`, so it conflicts with all
of them. It should land when the Metal queue is quiet, announced ahead of time,
with open Metal branches merged first or rebased by their owners right after.
Steps 1, 2, 4 and 5 touch few files and do not have this problem.
