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
backend. Only `dataframe/lazy_api.mojo` and `dataframe/execution_backend.mojo`
import it. A provider is a module with these twelve functions: four for whole
plans, three for hand-off and five for devices. Their signatures mention only
core types: `LazyFrame`, `DataFrame`, `String`, `Bool`, `Int`, `Float64` and
the device types in `dataframe/devices.mojo`.

| Function | Signature | What the core expects |
| --- | --- | --- |
| `installed` | `() -> Bool` | Whether the backend could run here. Must not open a driver, create a context or allocate. `engine="accel"` uses it to decide availability, and importing or CPU-collecting must never initialize a device. |
| `execute` | `(plan: LazyFrame) raises -> Tuple[DataFrame, DataFrame]` | Run the whole plan. Returns the result and a report frame. Raises for unsupported work before submitting anything. Called for `engine="accel"`. |
| `describe` | `(plan: LazyFrame) -> String` | Text for `explain`, as `ENGINE accel: ...` lines. Does not raise: errors become text. Must not execute the plan or initialize a device. |
| `execute_profiled` | `(plan: LazyFrame) raises -> Tuple[DataFrame, DataFrame]` | Same as `execute`, with measured data in the report. May return `execute`'s report unchanged. Called by `profile`. |
| `handoff_supports` | `(region: LazyFrame) -> Bool` | Whether the provider runs `region`: a node the CPU executor offers with the filters directly below it, as a plan over the input's schema. Decided before any input runs; does not raise. A provider for a discrete device returns False. |
| `handoff_execute` | `(region: LazyFrame) raises -> DataFrame` | Run an accepted region on its input columns and return the node's columns. |
| `handoff_estimate` | `(region: LazyFrame, rows: Int) -> Tuple[String, Float64]` | The region's calibration key and the provider's estimated nanoseconds for `rows` input rows, data movement included; -1 when it has no measured cost. Used for `engine="auto"` placement ([Apple Metal](../apple-metal.md)). |

The device functions describe a backend's devices and run pipelines on them.
They are written against memory models (discrete or unified) and capabilities,
not vendors, so one placement serves every backend. `engine="auto"` places
each pipeline with them (`dataframe/planning/device_placement.mojo`).

| Function | Signature | What the core expects |
| --- | --- | --- |
| `devices` | `() -> List[DeviceInfo]` | Each device the backend can run on: backend name, ordinal, readable name, memory model (`MEMORY_DISCRETE` or `MEMORY_UNIFIED`) and the bytes the planner may plan to hold there (-1 for no limit). May read driver metadata; must not create an execution context. Empty when nothing can run. |
| `transfer_model` | `(device: Int) -> TransferModel` | Measured cost of moving bytes to and from the device and of creating a context. Unified memory reports zero transfer cost. Negative fields mean not measured. Does not raise. |
| `pipeline_supports` | `(pipeline: LazyFrame, device: Int) -> Bool` | Whether the device runs `pipeline`: a plan whose leaves are the pipeline's inputs, its root the pipeline's sink. Decided from the plan and input types; does not raise or open a device. |
| `pipeline_estimate` | `(pipeline, device, features: PipelineFeatures) -> Float64` | Estimated nanoseconds to run the pipeline on the device, excluding moving its inputs and outputs: the core adds those from `transfer_model` and where each input already lives. -1 when there is no measurement. Does not raise. |
| `run_pipeline` | `(pipeline: LazyFrame, device: Int) raises -> DataFrame` | Run an accepted pipeline. Inputs may be host or device buffers (`dataframe/storage/residence.mojo`); outputs may stay on the device as device buffers. |

The core calls the whole-plan functions for `engine="accel"`, the hand-off
functions from the CPU executor when a provider runs a plan through it, and
the device functions for `engine="auto"`. No provider chooses a whole plan's
engine: the whole-plan automatic policy (`select_auto`, `execute_auto`) was
removed when placement per pipeline replaced it.

The same shapes exist as the `dataframe.lazy.AcceleratorBackend` trait, for an
explicit runtime object passed by the application. Its methods other than
`execute` and `describe` have defaults (no devices, nothing measured), so a
minimal backend implements only those two.

### Engine settings the provider honors

`collect`, `profile` and `explain` take `engine`, `optimize`, `streaming` and
`batch_size`.

- `engine`: `"cpu"` never reaches the provider. `"auto"` asks for `devices`
  and, where a pipeline is measured faster on one, calls `run_pipeline`; with
  a hand-off calibration it offers nodes to `handoff_supports`. `"accel"`
  requires `installed()`, then calls `execute` (or `execute_profiled`). An
  unknown name raises in the core.
- `optimize`, `streaming`, `batch_size`: the requested CPU-side settings,
  carried to `execute` in `ExecutionSettings`
  (`dataframe/execution_settings.mojo`). A backend must not treat `batch_size`
  as an expression batch size.
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
  for example; `dataframe_metal/accelerator/faults.mojo`). A backend that computes
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
- Be imported by core code. The check fails if any file under `dataframe/`
  imports `dataframe_metal`, or imports `dataframe_provider` from anywhere
  but `_accel_provider.mojo`.
- Import from `tests/`. Backend code is part of the package; tests are not.
- Change result types, add a GPU-only result type, or require an application
  to call a vendor API, upload data or hold a device handle.
- Run an unsupported part of a forced-accelerator plan on CPU without saying
  so, or retry on CPU after submitting work (except a data fault).
- Open a device in `installed`, `describe` or `devices`.
- Free host memory or device scratch while queued work may still use it.

## Registration model

Mojo has no conditional imports and no import-time side effects, so a backend
cannot register itself by being imported. The backend is chosen at build time,
by import path.

- `dataframe/_accel_provider.mojo` only re-exports the twelve provider
  functions from `dataframe_provider`.
- `dataframe_provider/` in the repository root is the default provider, Metal:
  it re-exports the twelve functions from `dataframe_metal.provider`. Where Metal
  cannot run (Linux), `installed()` is false and collection stays on the CPU.
- `providers/cpu/dataframe_provider/` is the CPU-only provider: `installed()`
  is false, `execute` raises "Accelerator provider is not installed",
  and it reports no devices. `-I providers/cpu -I .` puts it first and
  compiles no GPU code.
- Another backend ships its own `dataframe_provider/` package that re-exports
  the twelve functions from its implementation package, and puts it first on the
  import path. Two backends in one build are not supported; the first
  `dataframe_provider` on the path wins.
- Packaging: `scripts/build_package.py` picks the artifact by platform. macOS
  arm64 holds `dataframe`, `dataframe_provider` (Metal) and `dataframe_metal`;
  Linux holds `dataframe` and the CPU-only `dataframe_provider`.

The core, the backend and the provider reference each other across package
boundaries (`dataframe.lazy` imports `_accel_provider`, which imports
`dataframe_provider`, which imports `dataframe_metal`, which imports
`dataframe.lazy`). Mojo resolves this at compile time, as it did inside the one
package before.

The NVIDIA package ships its own `dataframe_provider/` package and puts it
ahead of the core's on the import path.

The checks are in `scripts/check_accelerator_boundary.sh`: core may import
`dataframe_provider` only from `_accel_provider.mojo` and never imports
`dataframe_metal`; `dataframe_metal` may import `dataframe` but not `tests`;
the root `dataframe_provider` (Metal) may import `dataframe_metal`;
`providers/cpu/dataframe_provider` imports nothing from a backend; and tests
import `dataframe_metal` only in the Metal test files the script lists, so
CPU tests never depend on a backend.

## Status

| Backend | Where it lives | How it attaches | What exists | What is missing |
| --- | --- | --- | --- | --- |
| Metal | Package `dataframe_metal` in this repository; its provider is the root `dataframe_provider` | The default provider of the open-source package: a plain `-I .` build | Runs inside the CPU executor through hand-off ([Apple Metal](../apple-metal.md)); accepts filtered sums of Int32, Int64 and Float32 columns. Implements all twelve provider functions. Compiled in every default build; Linux CI contract areas opt out with `-I providers/cpu`, and the `linux-default-build` job compiles it for a CPU target. Tested on a local Mac (hosted CI cannot run it). | Operators beyond the filtered sum; pipeline-level hand-off. A trait-checked registry: the functions are checked where the core imports them, not in the backend package. |
| NVIDIA | Private repository `dataframe_mojo_extensions`, package `dataframe_accel` | Its own `dataframe_provider/` package, first on the import path | Whole-plan execution of row pipelines over every numeric type and Bool, with casts and scalar reductions, on Linux x86-64. Builds against the core revision pinned in `core.json`. | The device functions. Pinned to an older core revision, so it needs them before it moves to this one. |
