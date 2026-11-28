# Differential fuzzer

Checks the engine against Polars on random inputs and random operation
chains, outside the fixed benchmark queries and contract tests. It does not
run in CI.

## How it works

- `fuzz.py` makes random cases: one to three tables, mostly small (0 to
  6,000 rows) and sometimes up to 240,000 rows so several threads and
  morsels run, with nulls, integers of
  every width, floats with NaN and infinities, strings with empty and
  non-ASCII values, booleans, dates, datetimes, decimals, categoricals,
  lists and structs. Each table is written as Parquet, Arrow IPC or CSV, so
  the file scans run too.
- A plan is a random chain of filter, select, with_columns (arithmetic,
  string, temporal, list and struct expressions, `when`, `over` windows),
  group_by + agg, joins (inner, left, right, full, semi, anti, cross; one to
  three keys, including string and categorical keys), sort, unique,
  head/tail/slice, explode, unnest, concat and a few others.
- `fuzz_driver.mojo` is one Mojo program that reads a case and runs the
  plan four ways: `eager` (DataFrame calls), `lazy` (scans + collect),
  `lazy_noopt` (collect with optimize=False) and `lazy_mem` (read eagerly,
  then `.lazy()`). It runs under `DATAFRAME_THREADS=1` and `8`. One compile
  serves every case.
- Polars runs the same plan as the reference, collected with
  `engine="in-memory"`. Floats compare within
  rtol 1e-6 / atol 1e-8. Row order is compared only where the plan fixes it
  (a sort on every column, `maintain_order=True`, head after an ordered
  step); a sort with ties checks its key columns in order and the rest as a
  multiset.
- A failing case is shrunk (plan steps, list entries, sub-expressions,
  rows, columns and file formats are dropped while it still fails the same
  way) and saved to `build/fuzz/failures/seed<N>/` with the tables as
  Parquet, `case.json` and a readable `report.txt`. The case as found is
  saved beside it in `unshrunk/`.
- "Fails the same way" means the same set of failures: kind, mode and
  thread count, the first line of an error, and for a mismatch what
  differs (a row value, the height, a dtype, a sort key) and how the
  failing column is made (the top of its expression, or its aggregate's
  kind, with `ddof` for `std`/`var`). A smaller case that fails some other
  way, or in other modes or thread counts, is not taken.

The generator follows our typing rules (no implicit numeric promotion), so
an error from our side is a finding. Where we differ from Polars on purpose
the oracle is adjusted, or the error is allowed:

- Float comparisons follow IEEE (NaN compares false, `NaN != NaN`); Polars
  orders NaN as the largest value. The oracle rewrites comparisons.
- `explode` gives one null row for an empty list; the oracle passes
  `empty_as_null=True`.
- `round` is half away from zero; the oracle asks Polars for that mode.
- Integer overflow raises here and wraps in Polars; such errors are allowed.
- A join's `validate` check always runs here; Polars' lazy engine skips it
  when an input turns out empty. A validation error is allowed when Polars
  returns no rows. Polars also checks after pushing filters into the join's
  inputs, as the optimized mode here does; the eager and unoptimized modes
  check the inputs as written, so their validation error is allowed when
  the optimized mode matches.
- Expressions on categoricals return String, and `len`/`n_unique` return
  Int64 where Polars returns UInt32: values are compared, and integer
  widths and categorical-versus-string are not. The oracle casts Polars'
  `len`, `count`, `null_count` and `n_unique` to Int64, so the generator
  types the expressions it builds on top of them as we do (a typed null
  of UInt32 next to a count would otherwise be an error here).
- String functions on a categorical read its labels; the oracle casts to
  String first.
- Mean, median, std and var of Float32 are Float64 here; the generator does
  not build them on Float32 columns, so later steps type alike.
- Polars' streaming engine gives a grouped `var`/`std` with `ddof=0` of an
  all-null group as 0.0 (its eager and in-memory engines and DuckDB give
  null, as we do), so the oracle runs on the in-memory engine.
- `arg_min`/`arg_max` return the first row holding the extreme; Polars may
  pick any tied row. The oracle asks for the first:
  `(c == c.min()).arg_true().first()`.

Known oracle noise (not engine bugs): a comparison of an aggregated float
with a literal can flip between thread counts when the value sits on the
literal.

Each driver run is capped at 8 GB (`FUZZ_MEMORY_MAX`, through
`systemd-run --user`), and the generator never builds a join step whose
result, counted from key frequencies, passes 600,000 rows.

## Running

Build the driver once (and `libdfparquet`, for Parquet inputs:
`pixi run -e native build-dfparquet`), then run cases:

```bash
pixi run -e oracle fuzz-build
pixi run -e oracle fuzz --cases 500 --seed 1
```

Options: `--threads 1 8` (thread counts to run), `--seeds N ...` (rerun
case seeds from a log), `--jobs N` (cases at once; default 1), `--no-shrink`,
`--shrink-budget N`, `--max-per-bucket N` (how many failures of one kind to
shrink and save), `--out DIR` (default `build/fuzz`). A summary of failure
kinds goes to `build/fuzz/summary.json`.

Replay a saved case (after a fix, it should print PASS):

```bash
pixi run -e oracle python3 tests/fuzz/fuzz.py replay build/fuzz/failures/seed123
```

Set `FUZZ_DRIVER` to use a driver built elsewhere. Set `FUZZ_FOCUS` to a
comma-separated list of plan step kinds (for example
`interpolate,time_window`) to make those steps 20 times as likely. Join
options have their own focus names: `join_validate` (a `validate` check on
most inner and left joins); use it with `join`.
