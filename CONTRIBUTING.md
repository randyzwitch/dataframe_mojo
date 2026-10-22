# Contributing

Thanks for helping. This page says how to build and test, which checks CI
runs, and what a pull request should contain. [ROADMAP.md](ROADMAP.md) says
where the project is going.

## Build and test

You need [Pixi](https://pixi.sh). It installs Mojo 1.2 and the other tools.

```bash
pixi install
pixi run -e native build-dfparquet   # once: the Parquet library the tests load
pixi run test                        # the contract suite
pixi run test --areas public storage
pixi run test --contracts table.join types.cast
pixi run example
pixi run build
```

A full `pixi run test` is slow: each test program takes about three minutes
and 4 GB to compile. While you work, run only the areas or contract groups
your change touches. CI runs everything. Contract groups come from
`# contract-group:` markers at the top of test files; a new test file needs
one. [docs/testing.md](docs/testing.md) and [tests/README.md](tests/README.md)
cover the layout, expected failures and capability skips.

Metal changes need `scripts/test_metal_local.sh` run on an Apple silicon Mac
before you open the pull request. Hosted CI cannot run Metal contracts.

## What CI checks

Run these locally before pushing.

| Rule | Check |
| --- | --- |
| Format: mblack, 80 columns | `pixi run format` fixes it; `pixi run format-check` is the CI check |
| `docs/api.md` matches the public docstrings | `pixi run docs` regenerates it; CI runs `pixi run docs` then `git diff --exit-code docs/api.md` (`scripts/api_docs.py`) |
| Engine and benchmark changes stay in separate PRs | `scripts/check_benchmark_separation.sh`. A PR that really needs both says why in a `Benchmark-Change:` commit trailer |
| No evidence files: nothing tracked under `benchmarks/results/`, and no `.csv`, `.json`, `.log`, `.html` or `.js` under `docs/` | `scripts/check_no_evidence_files.sh` |
| Every program under `tests/` is run or imported by something | `scripts/check_test_reachability.py` |
| Library code uses `DataType` constants, not dtype name strings | `scripts/check_dtype_literals.sh` |
| The version in `pixi.toml` and the install snippets agree | `scripts/check_version.py` (`pixi run check-version`) |
| The test runner's own checks | `mojo run -I . tests/tooling/test_contract_runner.mojo` |

## Pull requests

- One change per pull request. Smaller is easier to review and to revert.
- The description says what changed and why. Link the issue; write
  `Closes #N` if it closes one.
- Performance changes give measurements on the external suites (H2O, PDS-H,
  TPC-DS), not only a micro-benchmark. Report the worst data variant and any
  query that got slower. Port the design DuckDB or Polars uses; do not add a
  path tuned to one benchmark query. ClickBench is held out and is not used to
  choose what to optimize. The pull request template has the checklist, and
  [docs/design/benchmarks.md](docs/design/benchmarks.md) has the rules.
- Features follow Polars semantics, and DuckDB's result types. They work
  eagerly and lazily, and have tests for both.
- List the tests you ran. Say so if you could not run something.
- Write plainly. Short sentences, no jargon.
- Merge `main` into your branch before you push, since main moves quickly.

## Where things go

- Design notes and measurement records go in [docs/design/](docs/design/README.md).
  They explain why the engine looks the way it does, for people changing it.
- Pages for people using the library go in `docs/`.
- Benchmark reports are not committed. Link a commit or attach the report to
  the pull request.

## Reporting problems

Use the issue templates: bug, feature (Polars and DuckDB names of the
feature), or performance (timings for this library, Polars and DuckDB).
Security problems go through [SECURITY.md](SECURITY.md).
