# Testing

The suite starts from public dataframe contracts and independent expected values.
[The maintained coverage plan](test-coverage-plan.md) records each contract's
oracle, required dimensions and named witnesses. Physical layout, planner settings
and execution modes vary independently of the logical input.
The full suite requires `libdfparquet`; build it with
`pixi run -e native build-dfparquet` before the first full run.

```sh
pixi run test
pixi run test --areas public storage
pixi run test --contracts table.join types.cast
pixi run coverage-check
```

The six areas separate public behavior, storage and ownership, kernel equivalence,
planner correctness, execution modes and interchange. Shared Mojo models,
generators and typed comparisons live in `tests/support/`; orchestration and its
own failure checks live in `tests/tooling/`. Test modules exercise production
imports. Independent reference semantics have no dataframe dependency.

The runner compares names, logical types, shapes, ordered cells and null validity.
Known implementation defects run with exact diagnostic matching; an unrelated
failure or unexpected pass fails the suite. Fatal defects run in isolated
processes with exact exit-status matching. Native hardware cases report explicit
capability skips and require actual device execution when enabled. See
`tests/plan.json` for the current precise gaps.

Run metadata and compile/execute logs are written under `build/contract-tests/`.
They record the compiler, source fingerprints, revision, selected cases, settings
and capability omissions.
By default, each area runs in its own executable and later areas still run after
a build or process failure. `--packaging combined` explicitly selects a shared
executable when that packaging is useful for investigation.
The inventory validator rejects unmapped entry points,
stale coverage evidence and retirement records without live replacement witnesses.
The preservation ledger records unique regressions; it does not define the suite.
Retired entry points, source fingerprints, assignments and review evidence are
retained in `tests/preservation_archive.json`. Full source remains in Git history.
Review decisions live in `tests/preservation_reviews/`, with named maintained
witnesses in `tests/preservation.json`.

Native Metal execution remains unvalidated on the Linux redesign host. Capability
skips do not establish GPU parity. Multiworker positional global FIRST/LAST and
ARGMIN/ARGMAX also remain a known coverage limit: schedule-dependent results were
observed, so their deterministic assertions currently use one worker. The catalog
records this limitation without installing a flaky expected-failure test.

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
Python is used for these external engines, AST inventory, and unit tests of Python
tooling. Dataframe tests, semantic references and suite orchestration use Mojo.

`pixi run package-smoke` builds an identified package and compiles consumers outside
the checkout. It checks independent example output, typed cells and build identity.
Native calibration and execution checks additionally require a provisioned device;
`tests/tooling/package_contracts.mojo PACKAGE_DIRECTORY --native` enables them.

## Benchmarks

Headline performance comes from the external suites run by
`scripts/bench_suites.py`; [benchmarks.md](benchmarks.md) states the rules
and describes the suites, their data variants and the answer checks. The
in-repo `benchmarks/bench_*.mojo` programs below measure one mechanism at a
time and are development tools, not evidence of general performance.

When a mechanism benchmark's data qualifies for a fast path, it needs a
partner case that misses the path on the same data, and any row-count cutoff
needs sizes between the benchmark points: a cutoff chosen by comparing 1M
with 10M rows can sit anywhere between them.

The join/gather cutoffs and bounded-table budgets have a reproducible
[measurement report](join-cutoffs.md), including density, cardinality,
ordered-input and chunked-input counterexamples. Run its sweeps before
changing those guards; record the machine and the losing side of a crossover.

## Broad join comparison

`pixi run -e oracle bench-joins-polars --sizes 1000000,10000000 --threads 32`
compares thirteen join shapes with the installed Polars version. It includes
dense and sparse Int64 keys, string and composite keys, unmatched
left/right/full joins, semi/anti joins, duplicate matches, and a lazy join
followed by a narrow projection. Both engines use the same generated CSV input;
frame construction is outside join timing, and row counts and numeric totals
must agree.

Every case runs on three key layouts, reported in the `keys` column:

- `base`: the generated files. Every Int64 right key derives from a sorted
  `range(n)`, which the ordered-key path (`_dense_right_int64_rows`) and the
  direct-address paths (`_bounded_int64_join_rows` and the membership checks)
  recognise, so these are fast-path numbers. Real data with this shape is a
  lookup table keyed by sequential IDs stored in order.
- `shuffled`: the same right rows in random order. The join and its result
  are identical, but no ordered-key path applies.
- `wide`: every key mapped one-to-one onto a 2^40 range, so no
  direct-address path applies either.

Treat `shuffled` and `wide` as the general join performance and `base` as
the ordered-key case. A change that improves `base` alone has tuned a fast
path, not the join. `--variants base` runs only the ordered layout.

`pixi run -e oracle bench-joins-duckdb --sizes 1000000,10000000 --reps 5 --threads 32`
adds DuckDB to the same matrix. DuckDB times `CREATE TEMP TABLE AS SELECT` for
each join, so the full result is materialized in DuckDB's native format without
Arrow export. Input tables and derived keys are prepared outside timing. The
DuckDB result's columns are checked against the expected projection, and all
three engines must agree on row counts and numeric totals before results print.
The DuckDB time includes creating and filling a temporary table, while Mojo and
Polars produce their native dataframe results.

For CPU profiling, build `benchmarks/bench_join_matrix.mojo` and pass a case
name as its fourth argument to run only that case.

[worker-calibration.md](worker-calibration.md) records 4/8/16-worker sweeps
on Threadripper and Apple M1, plus isolated worker-policy experiments.

The [Parquet streaming report](parquet-streaming.md) records paired read-time
and peak-RSS measurements, raw samples, and reproduction commands.

The [lazy streaming report](lazy-streaming.md) compares the materializing
executor with ordered batch pipelines at 1M and 10M rows, including peak RSS
on a CSV larger than 1 GB.

Sort bucket calibration and reproducible sweeps are recorded in
[sort-cutoffs.md](sort-cutoffs.md), including fast-path misses and Mac validation.

The [upstream join baseline](upstream-join-baseline.md) records an earlier
adaptation of two DuckDB join queries; its scripts are retired (see
benchmarks.md) and the external suites supersede it.

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

Linux and macOS run `pixi run test` after building the native Parquet library
and checking its exported ABI. Independent consumers then validate the actual
writer artifacts with `pixi run -e oracle test-interchange-oracles`. Coverage
inventory, runner failure handling, formatting, version consistency, public
docstrings and Python production-tool checks run separately. Compilation cost
does not constrain semantic coverage.

Query-answer CI uses `bench_suites.py --correctness-only`: each selected query
runs once per engine and variant, with no timed repetitions, trace reruns,
reference-cache reuse, or performance reports. The usual smoke-scale suites
and all their variants remain selected. Wrong answers, missing outcomes,
failed workers, timeouts, and missing successful reference answers fail the
check. Explicitly unsupported queries retain their visible unsupported status.
JSON outcomes are saved as `build/suites/answers.json` by default, separately
from performance results. Normal benchmark commands keep their existing behavior.

### Independent required jobs

Fast checks, Linux contracts, macOS contracts, and two query-answer
jobs run independently. One Linux job restores/builds `libdfparquet` and
publishes it for the Linux consumers, avoiding multiple cold native builds.
macOS restores/builds its own library and does not wait for Linux.

The existing `test (ubuntu-latest)` check aggregates every Linux job and
fails for failed, cancelled, or skipped dependencies. `test (macos-latest)`
continues to select the maintained macOS suite and independent interchange
consumers. These workflow definitions are not evidence that a new macOS run has
passed. Existing branch-protection names remain valid. Timing artifacts have
unique names for every job.

The separate Native Metal workflow builds the runtime, checks its ABI and probes
the device. It enables `--capabilities metal` only when that probe reports an
actual device. An absent device leaves native witnesses explicitly skipped;
broken runtime loading and enabled execution failures fail the job.

Release validation builds the native Parquet library, validates the coverage
inventory, runs the maintained suite and independent consumers, then checks
documentation, examples and the packaged consumer before publishing.
