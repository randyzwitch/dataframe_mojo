# Testing

`pixi run test` runs every `tests/test_*.mojo` module, several at once:
`TEST_JOBS` sets the concurrency (default: CPU count; `TEST_JOBS=1` is serial),
and `bash scripts/run_tests.sh tests/test_x.mojo ...` runs chosen modules.
Output is printed per module in a stable order, followed by a summary naming
every failed module. Modules must not share temporary file paths. Beyond hand-written
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
without a C++ toolchain. CI on Linux builds the library first
(`pixi run -e native build-dfparquet`, cached by the contents of
`native/dfparquet/`) and then fails the run if the module reports `skipped`,
so the Parquet reader, its type coercions and row-group pruning are tested
on every push. macOS CI still skips them (#279).

## Benchmarks and the fast paths they hit

`pixi run -e oracle bench-polars` (see `scripts/bench_polars.py`) is the
yardstick. Its generated data qualifies for several fast paths, and a case
that always qualifies only measures that path, so each such case has a
partner that misses it on the same data:

| case | fast path it exercises | partner that misses it |
|---|---|---|
| `grouped_low`, `grouped_skew` | direct lookup for Int64 keys spanning fewer than 4,096 values | `grouped_outlier`: `key_low` with a few keys at 10^12 |
| `grouped_high`, `grouped_str` | general hash grouping | — |
| `sort_multi` | bucket sort on a first key with at most 64 evenly spread values | `sort_high`: first key with rows/10 distinct values |
| `join_inner` | ordered right keys (see below) | the join comparison's `shuffled` and `wide` layouts |

The default sizes are 100k, 1M and 2.5M rows. The middle-of-the-range size is
there because a row-count cutoff chosen by comparing 1M with 10M can sit
anywhere between them; if a workload's time per row jumps between two sizes,
sweep the sizes in between before drawing conclusions.

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
