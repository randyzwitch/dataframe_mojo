# Dataframe contract suite

This suite starts from observable dataframe behavior and independent reference
semantics. [plan.json](plan.json) is the maintained specification: each contract
records its oracle, required input dimensions, implemented witnesses and open
obligations. The catalog includes planned work; it does not claim full coverage.

| Directory | Responsibility |
| --- | --- |
| `public_contracts/` | API values, schema, ordering, nulls and errors |
| `storage_ownership/` | Views, aliases, chunks and resource lifetime |
| `kernel_equivalence/` | Specialized implementations versus scalar semantics |
| `planner_correctness/` | Rewrites with execution settings fixed |
| `execution_modes/` | Batches, workers and backends with optimization fixed |
| `interchange/` | Format compatibility, metadata and foreign ownership |
| `support/` | Logical generators, production adapters and typed comparisons |
| `tooling/` | Catalog validation and runner failure handling |

Tests import and exercise the production dataframe library. Only the independent
reference model avoids those imports. Logical inputs are generated first, then
materialized into different physical layouts without changing expected results.

`runner.mojo` discovers modules by contract area and checks their test entry points
against the catalog. `tooling/coverage_inventory.py` validates catalog evidence
and regression preservation. Test-specific orchestration lives in this tree.

Run `pixi run test`, or select areas with `pixi run test --areas public storage`.
Select a semantic obligation with `pixi run test --contracts types.cast`.
Default `--packaging areas` isolates each area in its own executable and continues
after build or process failures. Use `--packaging combined` explicitly to run
selected areas in one executable. Packaging does not change contract selection.
Verified implementation gaps are explicit in `plan.json` under `known_failures`.
They run with strict expected-failure checks: an unrelated error or an unexpected
pass fails the suite. An `XFAIL` is coverage evidence, not a working API claim.
Verified fatal defects run in isolated processes and must match both the exact
exit code and diagnostic recorded under `known_process_failures`.
Hardware prerequisites are declared under `required_capabilities`. The runner
records omitted cases as `SKIP(capability)` in output and run metadata. Enable
actual Metal checks with `--capabilities metal` after probing the native device;
an enabled test fails if its prerequisites are absent. The Metal CI job probes
hardware explicitly and runs the appropriate selection.

Semantic tests, generators, comparisons and runner checks use Mojo. Python is
limited to source inventory and its validator, independent external format
adapters, and tests that directly exercise the repository's Python benchmark,
calibration and diagram tools. Ordinary dataframe contract execution needs no
Python runtime.
Run `pixi run coverage-check` after changes to the catalog or test sources.
After the full suite, `pixi run -e oracle test-interchange-oracles` checks actual
writer artifacts with independent external consumers; CI runs this separately.
See [the design and workflow](../docs/test-coverage-plan.md) before extending it.

The retired suite's source review and unique regressions are recorded in
[preservation.json](preservation.json) and [inventory.json](inventory.json), with
named maintained witnesses. [preservation_archive.json](preservation_archive.json)
retains retired entry points, source fingerprints, assignments and review evidence;
full source remains in Git history. [preservation_reviews/](preservation_reviews/)
retains the review decisions.
These records do not define this suite's structure or completion criteria.
