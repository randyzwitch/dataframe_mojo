# Testing

The suite starts from public dataframe contracts and independent expected values.
[The coverage guide](design/test-coverage-plan.md) describes the contracts, independent
models and adversarial dimensions to maintain. Physical layout, planner settings
and execution modes vary independently of the logical input.
The full suite requires `libdfparquet`; build it with
`pixi run -e native build-dfparquet` before the first full run.

```sh
pixi run test
pixi run test --areas public storage
pixi run test --contracts table.join types.cast
```

The six areas separate public behavior, storage and ownership, kernel equivalence,
planner correctness, execution modes and interchange. Shared Mojo models,
generators and typed comparisons live in `tests/support/`; orchestration and its
own failure checks live in `tests/tooling/`. Test modules exercise production
imports. Independent reference semantics have no dataframe dependency.

Shared comparators check names, logical types, shapes, ordered cells and validity.
Known implementation defects run with exact diagnostic matching; an unrelated
failure or unexpected pass fails the suite. Fatal defects run in isolated
processes with exact platform-specific exit-status matching. Native hardware
cases report explicit capability skips and require actual device execution when enabled. Named module
groups, failure signatures and capability gates are marker comments in the test
modules they describe (`# contract-group:`, `# expected-failure:`,
`# requires:`), read by
[`tests/tooling/suite_policy.mojo`](../tests/tooling/suite_policy.mojo), so a
change adding tests edits only its own files.

Run metadata and compile/execute logs are written under `build/contract-tests/`.
They record the compiler, source fingerprints, revision, selected cases, settings
and capability omissions.
By default, each area runs in its own executable and later areas still run after
a build or process failure. `--packaging combined` explicitly selects a shared
executable when that packaging is useful for investigation.
Test functions are discovered automatically. `--contracts` selects named module
groups; it does not require assigning every test to a coverage registry.

Builds use the default (Metal) provider, as the library does. `--provider cpu`
(`pixi run test-cpu`) adds `-I providers/cpu` ahead of `-I .` to every build, so
no GPU code is compiled; the Linux CI areas use it. `--capabilities metal` (or
`mojo_gpu`) with `--provider cpu` is an error.

Native Metal contracts run on a local Apple silicon Mac through
`scripts/test_metal_local.sh`; hosted macOS runners expose no usable Metal
device, so CI reports those contracts as capability skips. Extreme
Date calendar fields are checked against an independent Int128 Gregorian model,
including the complete signed day-storage endpoints.
Capability skips do not establish GPU parity. Multiworker positional global
FIRST/LAST and ARGMIN/ARGMAX also remain a known coverage limit: schedule-dependent results were
observed, so their deterministic assertions currently use one worker. No flaky
expected-failure test substitutes for a deterministic regression.

## Interchange and native interfaces

Build the native Parquet library before running interchange contracts:

```sh
pixi run -e native build-dfparquet
pixi run check-parquet-abi
pixi run test --areas interchange
pixi run -e oracle test-interchange-oracles
```

The Mojo ABI check compares actual exports with the maintained C symbol allowlist.
Parquet contracts exercise real explicit, prefix and fallback library discovery,
projection and row-group requests, malformed input and observed release callbacks.
Arrow ownership tests retain aliases after foreign buffers are released and pages
unmapped. CSV fixtures use an independent field parser and literal typed models.
Spatial tests construct and consume independent WKB and format metadata.

External PyArrow adapters validate actual public writer artifacts; GeoPandas and
DuckDB Spatial adapters distinguish unavailable engines from semantic failures.
Python is used for these external engines and unit tests of Python tooling.
Dataframe tests, semantic references and suite orchestration use Mojo.

`pixi run package-smoke` builds an identified package and compiles consumers outside
the checkout. It checks independent example output, typed cells and build identity.
Native calibration and execution checks additionally require a provisioned device;
`tests/tooling/package_contracts.mojo PACKAGE_DIRECTORY --native` enables them.

## Benchmarks

Headline performance comes from the external suites run by
`benchmarks/tools/bench_suites.py`; [benchmarks.md](design/benchmarks.md) states the rules
and describes the suites, their data variants and the answer checks. The
in-repo `benchmarks/bench_*.mojo` programs below measure one mechanism at a
time and are development tools, not evidence of general performance.

When a mechanism benchmark's data qualifies for a fast path, it needs a
partner case that misses the path on the same data, and any row-count cutoff
needs sizes between the benchmark points: a cutoff chosen by comparing 1M
with 10M rows can sit anywhere between them.

The join/gather cutoffs and bounded-table budgets have a reproducible
[measurement report](design/join-cutoffs.md), including density, cardinality,
ordered-input and chunked-input counterexamples. Its sweeps are retired;
check out revision `8a50a95` to rerun them before changing those guards, and
record the machine and the losing side of a crossover.

## Broad join comparison

The join matrix (`bench_join_matrix`) and its Polars and DuckDB drivers are
retired. The H2O join suite, PDS-H and TPC-DS now compare joins with both
engines through `benchmarks/tools/bench_suites.py`; check out revision `8a50a95` to
rerun the old matrix.

[worker-calibration.md](design/worker-calibration.md) records 4/8/16-worker sweeps
on Threadripper and Apple M1, plus isolated worker-policy experiments.

The [Parquet streaming report](design/parquet-streaming.md) records paired read-time
and peak-RSS measurements, raw samples, and reproduction commands.

The [lazy streaming report](design/lazy-streaming.md) compares the materializing
executor with ordered batch pipelines at 1M and 10M rows, including peak RSS
on a CSV larger than 1 GB.

Sort bucket calibration and reproducible sweeps are recorded in
[sort-cutoffs.md](design/sort-cutoffs.md), including fast-path misses and Mac validation.

## CI checks and feedback

Required PR CI preserves unit, interoperability, and query-answer checks.
Example compilation runs separately on main, tags, manual dispatch, and PRs
changing examples or the Pixi toolchain/configuration. Library-only PRs rely
on the main-branch example check; release publishing also builds the examples.
Do not require the path-filtered Examples workflow in branch protection.
Superseded runs of the same PR are cancelled; main and tag runs are retained.

`scripts/ci_time.py --label NAME -- COMMAND ...` records wall time, child CPU,
peak child RSS and exit status in `build/ci-timings/`. CI publishes these files
and a job summary even after failure. Nested rows overlap; peak child RSS is
not aggregate concurrent memory. GitHub's job timestamps remain the source
for setup/cache time and queue delay. Compare elapsed time and runner minutes
separately, and compare native-cache hits with hits (misses with misses).

Each contract area is one compiled program, and compiling dominates the suite
(6 to 13 minutes per program on a 64-core machine against 90 seconds of test
execution in total), so CI runs the areas as parallel jobs:
`pixi run test --areas AREA` once per area on Linux, after the native Parquet
library is built once and its exported ABI checked. The Linux public area uses
four module shards to bound individual compiler resource use:
`pixi run test-cpu --areas public --shard-count 4 --shard-index 0`, repeated
with indices `1`, `2` and `3`. Sorted modules cycle between shards; all
tests in a module stay together. Their union covers ordinary tests, isolated
expected failures and capability skips exactly once. Run metadata records
the shard index and count. The unsharded command remains the default.
An empty selection is an error, so use fewer shards when narrowing contracts.
CI also passes `--compiler scripts/ci_mojo_public.sh` for these public shards
to cap compiler workers at two. Optimization remains at the compiler's default
`-O3`; the cap limits compilation concurrency, not runtime test threads.
The `interchange` job then validates the actual writer artifacts with
`pixi run -e oracle test-interchange-oracles`. On pull requests macOS runs the
`public` and `interchange` areas; pushes to main, the nightly schedule and
manual runs cover every area there. Runner failure handling, formatting,
version consistency, public docstrings and Python production-tool checks run
separately. Compilation cost does not constrain semantic coverage.

Query-answer CI uses `bench_suites.py --correctness-only`: each selected query
runs once per engine and variant, with no timed repetitions, trace reruns,
reference-cache reuse, or performance reports. The usual smoke-scale suites
and all their variants remain selected. Wrong answers, missing outcomes,
failed workers, timeouts, and missing successful reference answers fail the
check. Explicitly unsupported queries retain their visible unsupported status.
JSON outcomes are saved as `build/suites/answers.json` by default, separately
from performance results. Normal benchmark commands keep their existing behavior.

### Independent required jobs

Fast checks, one Linux job per contract area, one macOS job per selected
area, and two query-answer jobs run independently. One Linux job
restores/builds `libdfparquet` and publishes it for the Linux consumers,
avoiding multiple cold native builds. Each macOS job restores/builds its own
library and does not wait for Linux.

`test (ubuntu-latest)` aggregates every Linux job and `test (macos-latest)`
every macOS area job; both fail for failed, cancelled, or skipped
dependencies, so those two names are the ones to require. Timing artifacts
have unique names for every job, including the area.

Hosted runners cannot execute Metal contracts (their virtual GPU gives Metal
buffers no device address), so no CI job runs them. The `linux-default-build` job
compiles the backend for a CPU-only target and runs its dispatch checks, since
the Linux contract areas build with `--provider cpu` and never compile it. A change under
`dataframe_metal/accelerator/` runs `scripts/test_metal_local.sh` on an Apple
silicon Mac before its pull request is opened.

Release validation builds the native Parquet library, runs the maintained suite
and independent consumers, then checks
documentation, examples and the packaged consumer before publishing.
