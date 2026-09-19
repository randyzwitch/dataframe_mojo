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
