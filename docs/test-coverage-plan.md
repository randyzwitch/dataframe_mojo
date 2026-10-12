# Contract-based test suite

The new suite starts from specified behavior, independent reference semantics and
adversarial input dimensions. Its completion criterion is satisfying those
obligations. Test counts, implementation file boundaries and parity with an older
suite do not define the design.

[`tests/plan.json`](../tests/plan.json) is the primary catalog.
Each contract names its invariant, semantic source, oracle, required dimensions,
maintained witnesses, status and remaining obligations. Partially validated
contracts remain visible, including native Metal execution awaiting suitable
hardware. A capability skip is not passing execution evidence.

## Reference semantics first

`tests/support/reference_model.mojo` has no dataframe imports. It represents an
ordered schema and typed rows with separate validity for every cell. Its initial
schema vocabulary is Int64, Float64, String and Bool. Straightforward scalar
implementations define projection, take, clipped slicing, filtering, stable integer
sorting, integer arithmetic and wide sum/count. Independent scalar key
comparisons define stable lexicographic ordering, first-occurrence groups and
nested-loop join pairs. `reference_statistics.mojo` uses sorted values and
two-pass formulas for quantiles and variance. Literal, hand-computed examples
anchor the reference itself, including ties, nulls, negative slicing and invalid
arguments. Extended types use separate typed fixtures and scalar models, including
`selection_types_reference.mojo`, `reduction_types_reference.mojo`,
`typed_storage_reference.mojo` and `set_reshape_reference.mojo`. They preserve
logical type, payload and parent/child validity without coercing every value into
the four-type model. Required dimensions and remaining limits stay in the catalog.

`window_reference.mojo` defines scalar scans, independently enumerated rolling
memberships, nearest-neighbor fills, interpolation and pairwise ranks. It never
calls a dataframe sort or window kernel. Numeric width limits have separate exact
literal cases so the bounded Float64 model cannot hide wide-integer errors.
`text_binary_reference.mojo` models SQL LIKE with a full prefix relation over
Unicode code points. `temporal_reference.mojo` counts Gregorian civil days and
enumerates interval membership, including duplicate timestamps. Calendar and
membership literals anchor that model; timezone transition fixtures use explicit
expected instants and local fields.

`generated_model` creates logical inputs independently of storage. The adapter in
`coverage_generators.mojo` materializes those same inputs into production columns,
offset windows and unequal per-column chunks. `frame_from_model` varies the
physical offset while holding every logical row fixed. Expected results come from reference
rows, not from reading production inputs back or evaluating another production
operation. Changes to logical data and physical representation are separate axes.

## Six kinds of evidence

| Area | Question | Reference evidence |
| --- | --- | --- |
| Public contracts | What result or error does the API promise? | Explicit typed examples and independent scalar/relational models |
| Storage and ownership | What survives slicing, copying, mutation or release? | Reference values plus alias identity, lifetime and release counters |
| Kernel equivalence | Does each specialization implement the same semantics? | Scalar oracle over eligibility hits, misses, boundaries and tails |
| Planner correctness | Does each rewrite preserve meaning? | Independent expected results; optimization on/off with execution fixed |
| Execution modes | Can batching, workers or backends change meaning? | Independent expected results; modes varied with optimization fixed |
| Interchange | Can independent producers and consumers exchange data correctly? | Typed fixtures, independent format implementations, metadata/lifetime checks |

The catalog covers table/schema operations, expressions, reductions, grouping,
sorting, joins, sets/reshaping, casts, inspection, views, alias isolation, chunks,
resource release, specialized kernels, mergeable states, rewrites, route eligibility,
execution lifecycle/backends, and CSV/Arrow/Parquet/geospatial boundaries. These
contracts were specified from API behavior, not inferred from existing tests.
The public expression surface also requires explicit obligations for ordered
windows, text/binary, temporal and nested operations. Those families are tracked
individually rather than being implied by arithmetic or schema coverage.

## Generators, comparison and sensitivity

Start with explicit semantic boundary examples. Add systematic interactions across
logical data, physical representation and execution settings. Then add seeded,
shrinkable random cases; random volume does not substitute for known boundaries.

The implemented matrix includes empty/singleton inputs; lengths around bitmap and
vector boundaries; all-valid, independently mixed and all-null cells; duplicates,
negative numbers, UTF-8, empty text, quotes, embedded newlines and external-length
strings. Physical layouts include contiguous data, offset views and different
chunk boundaries per column. Execution checks vary batch sizes 1/7/64/257 and
workers 1/2/4. Parallel route claims require dispatch-sized inputs and actual
eligibility evidence; small-input worker checks alone do not prove dispatch.

`coverage_compare.assert_model` compares production results directly to reference
rows, without constructing expected dataframes. `assert_frame` supports additional
metamorphic and mutation checks. Both compare schema, column order, row count,
validity and every typed cell in order, using explicit typed comparison rather
than the production dataframe/value equality functions. Metamorphic comparisons
also inspect narrow numeric, temporal, decimal, binary and nested payloads;
the core reference-table comparator models four types, while extended fixtures
assert their own independent typed expected cells. Null payloads are ignored.
Floats compare exactly,
NaN equals NaN, infinity signs must match and signed-zero bits are preserved. The
initial generated floats are binary-exact quarters. Operations requiring approximate
comparison must declare an error budget and explicit nonfinite policy. An unordered
contract needs complete canonical row ordering that preserves multiplicity.

Rolling moments use a two-pass reference recomputed for each window. Their finite
comparison budget is 1e-12 absolute plus 1e-12 relative in variance units.
Standard deviations are compared after squaring, with a separate nonnegative
check: square root amplifies tiny variance errors near zero. Sensitivity cases
verify that this policy rejects material errors, negative results and spurious
NaNs. Nonfinite fixtures also check recovery after a bad value leaves a window.

Mutation checks challenge the comparator with unchanged-total row swaps, changed
column order/names/types, dropped rows, changed validity, signed zero and
finite/infinite mismatches. The catalog also calls for mutation and lifecycle
checks specific to each contract; comparator sensitivity alone is insufficient.

Planner tests compare optimized and unoptimized execution against the scalar
reference while fixing the executor. Execution tests compare batching and worker
modes directly against reference rows. Scalar reduction tests use Int128 totals.
Ownership and format roundtrips add evidence; roundtrip alone is insufficient for
interoperability, and independent format producers/consumers remain required.

## Development workflow

For each contract:

1. Specify observable behavior, errors, ordering and type rules independently of
   any implementation. Resolve ambiguous behavior before encoding expectations.
2. Choose an independent oracle and anchor it with literal examples.
3. Define logical, physical and execution dimensions, including the failure modes
   that ordinary generated values would miss.
4. Implement examples and systematic/reference comparisons, then prove their
   sensitivity with targeted mutations or lifecycle counters.
5. Record completed dimensions and open obligations in the catalog. Add seeded
   random exploration with shrinking where it adds useful interaction coverage.
6. Keep executable packaging separate from logical contract boundaries.

## Layout and running

The maintained `tests/` contains:

```text
tests/
  runner.mojo
  run_tests.sh
  plan.json
  preservation.json
  preservation_archive.json
  preservation_reviews/
  inventory.json
  public_contracts/
  storage_ownership/
  kernel_equivalence/
  planner_correctness/
  execution_modes/
  interchange/
  support/
  tooling/coverage_inventory.py
  tooling/contract_runner.mojo
  tooling/test_contract_runner.mojo
  tooling/test_coverage_plan.py
```

```sh
pixi run test
pixi run test --areas public storage
pixi run test --contracts types.cast table.sets_reshape
pixi run coverage-check
```

The runner defaults to `--packaging areas`: each selected area has its own
executable, TestSuite and temporary directory. It continues after failed builds
or processes so one area's failure does not hide another area's evidence.
Thread settings reset before each area. `--packaging combined` explicitly builds
the selected areas into a shared executable; ordinary test errors are collected,
but a build or process failure affects all areas in that program. Verified fatal
defects remain isolated in either mode, with both exact exit status and diagnostic
required. Run provenance is stored under `build/contract-tests/` by default.
Compilation cost does not constrain the contract design or its required coverage.

Hardware-gated witnesses declare `required_capabilities` in the catalog. Default
runs record them separately as `SKIP(capability)` without importing or counting
them as passes. `--capabilities metal` enables actual native tests, which fail if
the runtime or device is unavailable. Metal CI probes the built runtime before
selection; a broken installation fails the probe, while genuine device absence
records skipped native witnesses. Portable provider protocol tests establish
dispatch behavior and do not substitute for actual GPU parity evidence.

Verified implementation violations remain executable, with the specified expected
behavior unchanged. `plan.json` records each known failure by exact witness,
reason and diagnostic signature. The runner prints `XFAIL` only for that signature;
an unrelated error fails, and a passing test fails with `XPASS` until the annotation
is removed. These entries describe implementation gaps, not missing test coverage.
Production fixes remain separate from this redesign.

The runner and its failure-handling checks use Mojo. Python is retained for the
inventory tool's AST inspection of Python sources, its validator checks, and
independent external format adapters, and unit checks for Python benchmark,
calibration and diagram tools. These exceptions do not supply reference
semantics for dataframe operations. A strict failing case proves the first reached
failure; later combinations in its body remain unexecuted until the defect is
repaired. They must not be reported as passing evidence.

Lifecycle tests distinguish producer cancellation/unwind and consumer exception
cleanup. The public API has no external cancellation-token contract; those tests
do not imply one.

## Regression preservation and retirement

The retired suite has a separate preservation ledger. Its organization and case
count do not define the specification or completion checklist.
[`preservation_archive.json`](../tests/preservation_archive.json) retains retired
entry points, source fingerprints, assignments and review evidence. Full source
remains in Git history. [`preservation_reviews/`](../tests/preservation_reviews/)
records source-body review decisions and many-to-many replacement rationale.
[`preservation.json`](../tests/preservation.json) records source/case assignments,
protected regressions and named maintained witnesses.
[`inventory.json`](../tests/inventory.json) records source entry points, lines and
fingerprints. It includes test/check functions, unittest methods, main/script
entry points, fixtures, native tests,
oracles and shell validators. Entry-point counts include orchestrators and helpers;
they are not independent test counts. Loop cases remain represented by their
containing functions, with source fingerprints detecting changes.

`tests/tooling/coverage_inventory.py` fails on unmapped sources/cases, stale assignments, changed
source fingerprints, missing protected witnesses or missing new contract witnesses.
A completed contract must map every required dimension to named witnesses;
an untested dimension cannot be hidden by changing its status.
After reviewing a change, update the appropriate specification or ledger and run
`--write`. Regeneration does not authorize deletion. Broad source triage identifies
coverage areas; it does not establish replacement equivalence.

Retirement follows review of actual source bodies for unique failure modes,
with named witnesses in the independently designed suite. Protected examples include
copy-on-write alias isolation, unaligned validity offsets, stored null payloads,
signed-zero extrema, partial-overflow cancellation and strings surviving unmapped
source pages. These regressions remain obligations regardless of the old layout.

`pixi run test` runs the maintained suite. Linux and macOS CI select that suite,
then independently consume its interchange artifacts. Retirement preserves source
and review evidence; it does not turn unavailable OS or device validation into a
pass. Native Metal validation remains pending on suitable hardware. Multiworker
positional global FIRST/LAST and ARGMIN/ARGMAX have an explicit schedule-dependent
coverage limit; deterministic assertions use one worker rather than a flaky
expected-failure test. Production fixes remain separate work.
