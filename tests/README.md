# Dataframe contract suite

Tests start from observable dataframe behavior and independent reference
semantics. The [coverage guide](../docs/test-coverage-plan.md) describes the
contracts, adversarial dimensions and evidence to maintain.

| Directory | Responsibility |
| --- | --- |
| `public_contracts/` | API values, schema, ordering, nulls and errors |
| `storage_ownership/` | Views, aliases, chunks and resource lifetime |
| `kernel_equivalence/` | Specialized implementations versus scalar semantics |
| `planner_correctness/` | Rewrites with execution settings fixed |
| `execution_modes/` | Batches, workers and backends with optimization fixed |
| `interchange/` | Format compatibility, metadata and foreign ownership |
| `support/` | Logical generators, production adapters and typed comparisons |
| `tooling/` | Suite orchestration, runtime policy and runner checks |

Tests import and exercise the production dataframe library. Independent models
compute expected values; adapters materialize the same logical inputs into
different physical layouts.

Build the native Parquet dependency once with
`pixi run -e native build-dfparquet`, then run `pixi run test`.
Select areas with `pixi run test --areas public storage`, or related modules with
`pixi run test --contracts types.cast`.
The runner discovers test functions automatically; named groups, strict known
failures and capability gates live in
[tooling/suite_policy.mojo](tooling/suite_policy.mojo).

Default `--packaging areas` builds one ordinary executable per contract area and
continues after area build or process failures. `--packaging combined` explicitly
shares one ordinary executable and can require substantially more compiler memory.
Both modes compile one fatal dispatcher and launch a fresh process for each
selected full case ID. A full default run therefore builds seven executables
instead of sixteen, while retaining contract areas, canonical IDs and fixtures.
Run metadata and logs are stored under `build/contract-tests/`. See the
[packaging evidence](tooling/PERFORMANCE.md) for measurements and their limits.

An unrelated error or unexpected pass fails a strict expected-failure check.
Fatal defects run in isolated processes and must match both the recorded
platform-specific exit status and diagnostic. An XFAIL establishes the first reached defect; later
combinations in that test remain unvalidated.

Hardware omissions are recorded as `SKIP(capability)`. Enable actual Metal checks
with `--capabilities metal` after probing the native device. An enabled test fails
if its prerequisites are absent. macOS CI runs the native contracts when a device
is exposed; extreme Date epoch-shift overflow is a strict expected failure.
Multiworker global positional reductions also remain a known
limit: deterministic FIRST/LAST/ARGMIN/ARGMAX assertions currently use one worker
because schedule-dependent results were observed.

Semantic tests, generators, comparisons and runner checks use Mojo. Python is
limited to independent external format adapters and tests of Python benchmark,
calibration and diagram tools. After the full suite,
`pixi run -e oracle test-interchange-oracles` checks actual writer artifacts with
independent consumers; CI runs this separately.
