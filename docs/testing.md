# Testing

`pixi run test` runs every `tests/test_*.mojo` module, several at once:
`TEST_JOBS` sets the concurrency (default: CPU count; `TEST_JOBS=1` is serial),
and `bash scripts/run_tests.sh tests/test_x.mojo ...` runs chosen modules.
Output is printed per module in a stable order, followed by a summary naming
every failed module. Modules must not share temporary file paths.

Compiling, not running, is nearly all of a test run: each program compiles
the library code it uses again, 30 to 180 seconds per module, while every
test together runs in under a minute. So a full run builds about
`TEST_GROUPS` programs (default 10), each running several modules' tests
(`scripts/test_drivers.py`), instead of one program per module. A module
joins a group when its `main` only runs its tests; one whose `main` does more
(the Parquet modules) stays a program of its own. Modules in a group share a
process, so a test that changes process state, such as `DATAFRAME_THREADS`,
affects the modules after it: the driver resets `DATAFRAME_THREADS` before
each module, and anything else a test changes it must undo. Test functions
are imported by name, so every test is a top-level `def test_...() raises`.
Tests that declare a C function with `external_call`, such as `setenv`,
must pass it the same argument types everywhere, or a group fails to link;
pass addresses as `Int`. Beyond hand-written
contract tests, three generative layers look for interaction bugs.

## Differential testing against Polars

`scripts/oracle.py` generates random inputs (all four dtypes, 15% nulls,
`-0.0`, large floats, empty strings, quotes, separators, and multi-byte text)
and a random operation from a registry: filters, arithmetic, grouped
aggregates, multi-column sorts, joins, `unique`, `cum_sum`, and casts. Polars
computes the expected result with its ordering options pinned
(`maintain_order=True`, `maintain_order="left_right"` for joins) so both sides
are deterministic; the Mojo runner (`tests/oracle/runner.mojo`) executes the
same operation. Both results are read back through Polars' CSV reader with one
schema, and floats compare with a relative tolerance of 1e-9.

    pixi run -e oracle oracle --cases 200 --seed 1
    pixi run -e oracle oracle --mutation-check

Polars lives only in the `oracle` pixi environment; it is never a runtime
dependency. A failure prints its seed, the operation, and inputs minimized by
removing rows while the mismatch persists, so it reproduces with
`--seed N --cases 1`. `--mutation-check` makes the runner drop a result row and
requires every case to be caught, proving the comparison is not vacuous. CI runs
150 cases with a seed that changes per run, plus the mutation check.

To cover a new operation, add it to `gen_op`, `expected`, and the runner's
`run` function, choosing Polars options that pin the same semantics.

## CSV tokenizer fuzzing

`tests/test_csv_fuzz.mojo` feeds random byte strings made of CSV-significant
bytes (separators, quotes, CR, LF, digits, signs, multi-byte and invalid
UTF-8) to `read_csv` under strict, text-only, lossy, and permissive settings.
Every input must parse or raise a clean `Error`, and the outcome, including
the error message, must be identical for buffer sizes 1, 2, 3, 7, and the
whole file. `tests/test_csv_options.mojo` additionally splits fixed fixtures at
every byte position.

## Mergeable reduction states

Reduction states (`IntSumState`, `FloatSumState`, `LogicState`, `VarState`) are
merged across every split point and association order in
`tests/test_expressions.mojo`, `tests/test_expr_logic.mojo`, and
`tests/test_expr_reductions.mojo`: integer results must be identical and float
results equal within tolerance. These are the properties parallel execution
relies on.

## Parquet tests in CI

`tests/test_parquet.mojo` needs `libdfparquet` and skips itself, with a
notice, when the library is missing, so `pixi run test` passes on a machine
without a C++ toolchain. CI on Linux and Apple Silicon builds the library
first (`pixi run -e native build-dfparquet`, cached by OS, architecture and
the contents of `native/dfparquet/`). Then
`pixi run bash scripts/check_parquet_library.sh` checks that the stripped
library exports exactly the declared C entry points and runs the compiled
Parquet tests twice: once loading from `build/dfparquet`, and once from an
isolated `$CONDA_PREFIX/lib` with no build-directory fallback. An explicit
`DATAFRAME_PARQUET_LIBRARY` override is removed for both runs. Either run
failing or reporting `skipped` fails CI, so type coercions, row-group pruning,
chunk preservation, and both library search paths are exercised on every push.
`tests/test_parquet_stream.mojo` uses a fake producer to count releases on
EOF, schema/import errors and failure after a consumed batch. It requires no
native library. `scripts/check_parquet_stream.py` uses PyArrow and the real
shim to check projection, reordered/duplicate groups, empty selections,
early close and a corrupted later row group.

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

## CI cost and feedback

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

Before adding another compiled program, check whether an existing driver can
run its cases. Review timing changes across several runs when adding checks.
The October 8 baseline was 72–91 minutes on Linux and 42–70 on macOS, excluding
queue time. Example compilation alone cost 8–10 and 7–8.5 minutes respectively.
Initial targets are under 60 minutes per required job, then under 45 after
measured driver tuning, without dropping test modules, cases, or query variants.
These are review targets, not timeouts that terminate correctness coverage.
