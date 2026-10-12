# Test coverage guide

Tests start from observable dataframe behavior and independent expected results.
This guide records the obligations to maintain; executable tests provide the
evidence. Test counts and the retired suite's layout do not define completeness.

## Six areas

| Directory | Responsibility | Evidence |
| --- | --- | --- |
| `public_contracts/` | API values, schema, order, nulls and errors | Literal examples and scalar models |
| `storage_ownership/` | Views, aliases, chunks and resource lifetime | Typed values, identity and release counters |
| `kernel_equivalence/` | Specialized implementations | Independent models plus actual dispatch hit/miss checks |
| `planner_correctness/` | Rewrites and dependencies | Optimization on/off with execution fixed |
| `execution_modes/` | Batches, workers, scheduling and backends | Execution modes varied with optimization fixed |
| `interchange/` | Formats, metadata and foreign ownership | Independent producers/consumers and lifecycle checks |

Tests exercise the production dataframe library. Shared logical models in
`tests/support/` avoid production implementations of the operation being tested;
production adapters materialize those inputs into physical layouts.

## Contracts to maintain

| Family | Required behavior |
| --- | --- |
| Tables and expressions | Names, types, shape, projection dependencies, checked arithmetic, null propagation and errors |
| Reductions and windows | Empty/all-null identities, exact wide totals, ordering, duplicate membership, precision and nonfinite recovery |
| Sorting, grouping and joins | Stable order, multiplicity, composite keys, null/NaN classes, unmatched rows, tolerances and schema |
| Casts and logical types | Narrowing boundaries, decimal scale/precision, temporal units/zones, dictionary identity and nested validity |
| Text, binary and spatial | Unicode/byte semantics, grammar, malformed inputs, coordinates, dimensions and metadata |
| Storage and scheduling | Alias isolation, offset/chunk boundaries, owner retention, exactly-once release, late errors and retry |
| Planning and backends | Dependency barriers, eligibility, actual route selection, budgets, unsupported requests and mode parity |
| Interchange | Independent format agreement, sliced/unaligned buffers, metadata, malformed input and ownership |

For each change, state the result, ordering, type, null and error rules before
choosing fixtures. Resolve ambiguous semantics from the public contract.
Keep unique regression examples even when broader generators are reorganized.

## Dimensions and models

Vary logical data independently from physical layout and execution settings:

- Logical: empty/singleton and boundary lengths; independent cell validity;
  duplicates, skew, signed bounds, special floats, Unicode and nested null parents.
- Physical: contiguous columns, offset views, unequal chunks and foreign buffers.
- Execution: optimizer on/off, eager/lazy, streaming/materializing, batch sizes,
  worker counts and available backends.

Use literal anchors, then systematic interactions. Seeded random exploration can
add interactions; it does not replace boundary examples. Parallel or specialized
route claims need inputs that actually dispatch and observable route evidence.

Shared models include `reference_model.mojo`, `reference_statistics.mojo`,
`window_reference.mojo`, `temporal_reference.mojo`, `asof_reference.mojo` and
typed domain helpers. Scalar scans, independently enumerated memberships and
Int128 arithmetic supply expected results. `coverage_generators.mojo` materializes
the same logical rows into different layouts. An expected result must not call
the production operation it is checking.

Compare complete schema, row count, ordered typed cells and validity.
`coverage_compare.mojo` supplies shared comparisons; typed fixtures cover domains
beyond the core four-type reference table. Ignore hidden null payloads.
Floating comparisons declare NaN, infinity, signed-zero and rounding policies.
Approximate results need a stated error budget and sensitivity examples.
Rolling moments use an independent two-pass calculation per window; nonfinite
cases also check recovery after eviction. A roundtrip alone cannot establish
interoperability, and equivalent rows alone cannot prove release or pruning.

## Running and runtime policy

Build `libdfparquet` before the full suite:

```sh
pixi run -e native build-dfparquet
pixi run test
pixi run test --areas public storage
pixi run test --contracts types.cast table.join
pixi run -e oracle test-interchange-oracles
```

The runner discovers test functions automatically. Named `--contracts` groups
select modules through [suite_policy.mojo](../tests/tooling/suite_policy.mojo);
they are a convenience for running related tests, not a coverage ledger.
Default `--packaging areas` isolates areas and continues after failures.
`--packaging combined` explicitly shares an executable. Logs and run metadata
live under `build/contract-tests/`.

The typed runtime policy also records strict known failures and capability gates.
An unrelated diagnostic or unexpected pass fails the suite. Fatal defects run in
isolated processes and must match the recorded exit status and diagnostic.
An XFAIL proves only the first reached failure; later combinations are unvalidated.
Metal cases require `--capabilities metal` after a real device probe. Skips remain
explicit and do not establish GPU parity.

## Current validation limits

Actual native Metal execution remains pending on suitable macOS hardware.
Multiworker streamed global FIRST/LAST/ARGMIN/ARGMAX have produced
schedule-dependent results; deterministic positional checks currently use one
worker. No flaky expected-failure test substitutes for a deterministic regression.
Lifecycle tests cover producer cancellation/unwind and consumer errors; the public
API has no external query cancellation-token contract.

Mojo owns semantic tests, generators, comparisons and orchestration. Python is
limited to independent external format engines and tests of Python production
tools. CI separately consumes actual writer artifacts. Optional GeoPandas and
DuckDB Spatial adapters report unavailable capabilities without treating them as
passing interoperability evidence or downloading extensions.
