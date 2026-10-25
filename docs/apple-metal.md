# Apple Silicon Metal backend

**Status: the Metal execution core is being rebuilt and runs no queries yet.**
`engine="accel"` (or passing a `MetalRuntime` as `accelerator=`) raises an
error that starts with `Metal unsupported [rebuild]:` for every query.
Nothing falls back to the CPU silently. Collect with `engine="cpu"` (the
default) in the meantime. `engine="auto"` always chooses the CPU. Once the
first operators land, Metal runs inside CPU queries (see below) instead of
as a separate engine.

The previous core was 30 to 250 times slower than the CPU per row, at every
data size measured. It stored every value as a 21-byte generic cell, ran
expressions through a per-row interpreter, emulated floating point in
software, and ran several kernels serially to reproduce the CPU's summation
order. It was removed rather than repaired. Its source remains in git history
at commit `d51540d4` (`dataframe_metal/`) for reference.

## What is still here

- `dataframe_metal/metal.mojo`: `MetalRuntime`, `default_runtime()`,
  `metal_installed()` and the calibration entry points. `MetalRuntime`
  validates its arguments and owns a lazy device service. Constructing it
  does not create a GPU context.
- `dataframe_metal/provider.mojo`: the six provider functions the core calls
  (docs/design/accelerator-contract.md). The core never imports
  `dataframe_metal`; `scripts/check_accelerator_boundary.sh` enforces that,
  and also keeps tests outside a short Metal list from importing it.
- `dataframe_metal/accelerator/gpu/`: device detection (`target`,
  `identity`), the device service and per-query scope with byte budgets
  (`query_scope`, `scratch_pool`), and typed column upload and readback
  (`typed_storage`). The new core builds on these.
- `dataframe_metal/accelerator/faults.mojo`: the data-fault marker. When a
  kernel reports a data fault (overflow, a failed strict cast, an
  out-of-range temporal value), Metal reruns the query on the CPU and returns
  the CPU's outcome, since the CPU decides evaluation order. Profiles of such
  a query carry `fault_adjudication="cpu"`.

Building is unchanged. Metal is the default provider (`-I .`); on Linux
`installed()` is False. `-I providers/cpu -I .` compiles no GPU code.
`DATAFRAME_ACCEL_DEVICE` and `DATAFRAME_ACCEL_MEMORY_LIMIT` still configure
the default runtime.

## The new design

On Apple silicon the CPU and the GPU share memory, so Metal is treated as
part of the Apple CPU engine rather than as a separate engine. The CPU
executor runs every query and hands individual operators to Metal when two
things hold: Metal supports the operator on the column types natively, and
measured costs on the device say it will be faster. Everything else stays on
the CPU, in the same query.

- **No emulated types.** Apple GPUs have no 64-bit floating point, and Metal
  has no 64-bit atomic add on any Apple GPU. Work on Float64 columns, or work
  that would need those atomics, runs on the CPU, which does it better than
  an emulation would.
- **Kernels.** Typed column buffers, fused and compiled kernels, and
  parallel reductions with no emulation of the CPU's evaluation order. Each
  operator first checks MAX's GPU kernel libraries (warp and block
  reductions, 32-bit atomics and `elementwise` run on Apple GPUs) before
  writing its own.
- **Visibility.** `explain` and `profile` show which operators ran on Metal.

The provider contract stays vendor-neutral: columns in and out, and the core
never imports a backend. A discrete GPU (NVIDIA, AMD) is a separate provider
that keeps data on the device and runs whole queries or large regions. The
same cost-based placement decides both, with the cost of moving data
included.

Coverage, measured against the CPU with `report-apple`, is listed here as
each operator lands.

## Validation

Hosted macOS runners cannot run Metal (their virtual GPU gives Metal buffers
no GPU address), so Metal contracts run on a local Apple silicon Mac:

```sh
scripts/test_metal_local.sh
```

`pixi run -e oracle report-apple` times CPU against forced Metal on the
benchmark suites. While no query shape is supported, every case reports
`unsupported`.
