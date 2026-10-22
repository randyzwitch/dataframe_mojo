# Benchmarks

Performance numbers in this project must describe how the engine behaves on
work other people chose, not on inputs we shaped. Two audits (#274, and the
follow-ups #304 and #306–#308) found fast paths whose only trigger was one of
our own benchmark queries or key layouts. The fix is structural: measure on
external suites, keep the measuring code apart from the code being measured,
and make benchmark-shaped paths visible. Keep development and holdout roles
useful going forward, while acknowledging prior exposure.

## Rules

1. **Headline numbers come from external suites.** Their queries and data
   generators are defined upstream, so a path that only helps a shape we
   invented shows no gain there. The in-repo micro-benchmarks
   (`benchmarks/bench_*.mojo`) remain development tools for one mechanism at
   a time; they are not evidence of general performance.
2. **Benchmarks and engine changes go in separate PRs.** A PR that changes
   `dataframe/` must not also add or change a suite,
   its data generator, queries or drivers (anything under `benchmarks/`). This
   stops a fast path and the benchmark that justifies it from arriving together.
   `scripts/check_benchmark_separation.sh` enforces this in CI; a PR that
   genuinely needs both (for example, this one, which adds tracing hooks and
   the suites together) says why in a `Benchmark-Change:` commit trailer.
3. **Development and held-out suites.** Tune on H2O, PDS-H, TPC-DS and
   mechanism benchmarks. Keep ClickBench as held-out validation of
   completed changes. Do not use its per-query timings to choose
   optimization targets, tune thresholds, or add query-specific exceptions.
   Some earlier optimization work used ClickBench results; record that
   caveat and enforce the boundary going forward. Correctness bugs can be
   investigated and fixed generally; retain the failing result and the
   rerun rather than hiding failures.
4. **Every measurement includes perturbed data.** The data variants below
   always run, and the report puts each query's worst variant beside its base
   result. A fast path that fires only on sorted keys, or only without nulls,
   shows up as a gap between the two.
5. **Coverage, not just time.** `--trace` records which specialized join,
   group-by, filter and sort paths each query took (`dataframe/trace.mojo`).
   A path that only one suite query exercises, or that none do, needs a data
   property with real-world examples to justify it.
6. **Equal terms.** Each engine uses its idiomatic API, the same thread
   count, the same machine and the same Parquet files loaded into memory;
   loading is untimed; every timed run materializes the full result; answers
   are checked against DuckDB; ratios use geometric means; wrong answers and
   unsupported queries are counted, never dropped.

Rules 3 to 6 apply to both tiers. The quick tier makes them cheap enough to
follow on every change; the full tier is what a report cites.

## Suites

| Suite | Role | Source | Queries | Variants |
|---|---|---|---:|---|
| `h2o_groupby` | development | [db-benchmark](https://github.com/duckdblabs/db-benchmark) group-by | 10 | cardinality parameter k=100, 10 and 2; 5% nulls; sorted |
| `h2o_join` | development | db-benchmark join | 5 | none; 5% nulls |
| `pdsh` | development (held out until 2026-10-05) | [PDS-H](https://github.com/pola-rs/polars-benchmark), TPC-H derived | 22 | money columns as DOUBLE (`base`) or DECIMAL(15,2) (`decimal`) |
| `tpcds` | development (held out until 2026-10-15) | TPC-DS, from DuckDB's `tpcds` extension (`dsdgen`, `tpcds_queries()`) | 99, of which 89 are translated | money columns as DOUBLE (`base`) or as declared decimals (`decimal`) |
| `clickbench` | held out | [ClickBench](https://github.com/ClickHouse/ClickBench) `hits` | 43 | — |

The H2O data follows db-benchmark's R generators with seeded Polars sampling,
so distributions match but values do not. TPC-H tables come from DuckDB's
`dbgen` in two variants: `base` stores the DECIMAL(15,2) money columns as
DOUBLE, as the suite did before this library had decimals (#229), and
`decimal` keeps them as DECIMAL(15,2), TPC-H's own type. The `base` numbers
stay comparable with earlier reports; `decimal` measures decimal arithmetic,
aggregation and joins. ClickBench uses the first N one-million-row
partitions of `hits`, converted once from raw byte arrays and integer times to
strings, timestamps and dates, as ClickBench's own Polars and DuckDB scripts
do at load time. All engines read the same files.

Queries live in `benchmarks/suites/`:

- `engines.py`: DuckDB runs the upstream SQL (db-benchmark's queries,
  `tpch_queries()`, `tpcds_queries()`, ClickBench's `queries.sql`); Polars
  runs idiomatic lazy translations.
- `h2o.mojo`: this library's eager API; `pdsh.mojo`: lazy query plans,
  including joins with differently named keys. In the `decimal` variant, `pdsh.mojo`
  writes money literals as decimals and converts to Float64 where a query
  divides or compares with an average, as DuckDB's DOUBLE division does;
  this library does not mix decimal and float operands implicitly.
- `tpcds.mojo`: lazy query plans over the 24 TPC-DS tables, with the same
  decimal handling as `pdsh.mojo`. See "TPC-DS coverage" below.
- `clickbench.mojo`: the lazy API, whose projection pushdown reads only the
  columns each query uses from the 105-column table.

A query an engine cannot express is reported as unsupported with its reason;
failed and unsupported queries remain visible in the report.

## Running

### External suites

There are two tiers. Data is generated or downloaded once into a directory
beside the main checkout, `<checkout>_benchdata` (override with
`DATAFRAME_BENCH_DATA`), and shared by every git worktree.

```bash
pixi run -e native build-dfparquet                                   # once

# After a code change: a few minutes once built.
pixi run -e oracle python3 benchmarks/tools/bench_suites.py --baseline main

# For a report: hours; run it occasionally, not per change.
pixi run -e oracle python3 benchmarks/tools/bench_suites.py --tier full --all-suites
```

**Quick** (the default) runs the development suites (H2O at 1M rows, PDS-H
and TPC-DS at scale factor 0.1) with every data variant, three rounds of three timed runs, and times only this library.
`--baseline REF` builds the suite runners against the library at `REF`
(checked out once as a git worktree under `build/suites/baseline/`, cached by
commit) and alternates the two builds. The report opens with **Changes vs
baseline**: only the cells where every round of one build beat every round
of the other by more than 3%. Polars and DuckDB outcomes come from a
reference cache keyed by data file, engine version, thread count and
repetitions and reference query-source hash, because their code does not
change when this library does.
Narrow a run with `--suites h2o_join` or `--queries q1,q2`.

**Full** runs both development and held-out suites at 10M rows (PDS-H and
TPC-DS at scale factor 1, 10M ClickBench rows), three rounds, measures every
engine afresh and records fast-path coverage. `--all-suites` also selects
all five with the quick tier; `--suites` can explicitly narrow either tier.
The `--heldout` option adds the held-out suite, ClickBench, to a run.

Both tiers give each (suite, variant, engine, round) its own process, which
loads the tables once, untimed, then warms up and times each query. The
driver waits while a compiler runs, records the load average before every
worker and flags a busy host in the report. It writes raw samples with
provenance to `build/suites/results.json` and the report beside it, as
Markdown (`results.md`) and as a self-contained HTML page (`results.html`,
from `benchmarks/tools/bench_html.py`) that opens in any browser; `--report-from`
re-renders both from a saved file. Provenance includes engine and benchmark
working-tree status, local query/generator/runner SHA-256 hashes, the CPU
frequency governor (a parked thread wakes on a core `schedutil` has clocked
down, which moves the short queries; #527) and an exposure record for each
selected suite. These identify the local workload;
historical upstream commits were not recorded and remain explicitly unknown.
DuckDB's recorded version identifies its `tpch_queries`, `dbgen`,
`tpcds_queries` and `dsdgen` implementations.

Re-rendering older raw JSON preserves its samples and displays a legacy
metadata note; it does not invent missing historical provenance.
The coverage table separates distinct suite/query pairs from query/variant
cases: five variants of one query are still one query when evaluating how
widely a specialized path is exercised. Instrumentation includes all literal
`trace_path` names, including nested paths and lazy/rank paths. Paths without
instrumentation are outside this report's coverage.

The [completed M1 forced-Metal development baseline](../benchmarks/results/metal-development-baseline/50137ecc-m1-20261020/)
retains 1,812 worker records across 302 cases, exact correctness receipts, raw
logs and a hash verifier. Only 210 cases qualify for CPU/Metal timing ratios;
unsupported and unverified cases remain visible. This is development-scale
evidence for #869, not a full-scale headline or a GPU-only execution certificate.

`benchmarks/tools/bench_metal_suites.py` separately checks exact CPU/Apple
query results with every query `collect` forced onto Metal. Summarize one or
more immutable receipt files with
`python3 benchmarks/tools/metal_query_report.py RECEIPT.json --output REPORT.json`.
The tool checks consistent engine, SDK, workload and runner identities,
rejects duplicate cases, and derives catalog and translation counts from the
retained query sources. Partial grids remain partial. JSON and text reports
separate populated and empty matches, preserve unsupported and differing
answers, and exclude source-audited hybrid cases from conservative counts.
An unequal CPU repeat remains diagnostic evidence, not an exemption from the
exact comparison. The case audit registry records q24's eager CPU mean;
forcing its `collect` calls does not make that aggregate a GPU operation.
Coverage `collection_ns` values are diagnostic and are not performance results.

`benchmarks/tools/bench_metal.py` is a development harness for individual
mechanisms. It includes cumulative and fixed-width rolling sums, a skewed
group-by (three quarters of rows share one key), and the existing mechanism
cases. Each worker has a finite timeout (`--timeout-seconds`, default 300);
timeout output is retained in its run log and raw JSON as a failed case. The
raw metadata records the compiler version and executable hash, lockfile hash,
checkout status, and a content hash of the full `dataframe/` tree, including
untracked source files, alongside the benchmark and binary hashes. Each binary
has a sidecar build receipt tying its hash to its own source tree, compiler,
lockfile, harness source and flags. Legacy binaries without receipts stay
source-unknown; historical results were not retroactively attributed. The raw
run also records the exact command line and effective timeout. The Markdown
report lists cases where Metal is slower than CPU as well as the worst variant
for each mechanism. Profile phase fields are retained when supplied by the
runtime. Host phase values may overlap and are reported separately; the report
does not derive GPU-only or pipeline compilation time when the runtime does not
provide it.

Four optional SUM cases cover native Int64 and UInt64 inputs: `int64_sum`,
`uint64_sum`, `int64_affine_sum`, and `int64_wide_sum16`. The signed inputs are
`i % 17 - 8`, unsigned inputs are `i % 17`; sorted variants distribute those
same ranges monotonically, and null variants use
`UInt32` wrapping of `1664525*i + 1013904223` before taking `% 17` (then
subtract 8 for signed values), with every twentieth row null. The affine case
sums `(x + 3) * 2`, bounded to magnitude 22 per row. The wide case returns
16 separately named sums over one Int64 source column. Each output is checked
against the CPU executor's result; that check validates execution but is not an
independent oracle. These names are selectable through `--workloads` and are
excluded from the default workload list. The new `--warmups` setting defaults
to the existing two extra collections per engine. For paired runs, the driver
requires verified receipts with matching benchmark source, driver, lockfile,
compile flags, compiler identity and executable hash, build platform, SDK, and
Xcode version. Engine commit and engine-source hashes may differ. This guard
checks those recorded identity fields; host hardware and runtime conditions
remain separate recorded context.

The paired native Int64/UInt64 run is archived at
[`benchmarks/results/metal-native64/a52334e7-52efeb3a/`](../benchmarks/results/metal-native64/a52334e7-52efeb3a/).
It compares baseline engine `a52334e758b7138c547883eb776d744442a1a0db` with
candidate `52efeb3ab784c8949366870b7cd6b007b3bf77f9`, using benchmark and driver
file contents matching commit
`3447f6026a4a9d3745d56040ae4b93bc157de477`. The run contains
144 validated baseline/candidate run keys across two row counts, four SUM
workloads and three input variants. At 1,048,576 rows, baseline-to-candidate
Metal median ratios range from 3.07× to 8.08× across the four workloads and
their variants. Candidate CPU warm medians are lower than candidate Metal in
all 24 workload/size/variant cells, so these are implementation comparisons,
not evidence of a general GPU advantage. The package retains raw samples,
source and binary receipts, and qualification logs. The archived prior
full-gate log records four failures: selector expansion, categorical grouped
row count and qcut whole-DAG synchronization parity were independently
reproduced on baseline; an ordered profile/resource-estimate failure was
followed by focused resource-bound fixes whose tests passed. The full gate was
not rerun as all-green. See `summary.md` for the paired table and `validation/`
for qualification details.

The grouped integer candidate comparison is archived at
[`benchmarks/results/metal-group-integer/a52334e7-0a9297c3/`](../benchmarks/results/metal-group-integer/a52334e7-0a9297c3/).
It retains 234 validated workers across frozen-baseline and #865-intermediate
comparisons. The candidate improves all warm cases over the frozen baseline,
but is about 7% slower than #865 on two large non-null uniform cases while
improving large skewed cases by 4.75–4.89×. Reported peak allocation increases,
and CPU remains faster throughout. These results require a scheduling revision;
they do not establish universal improvement or grouping replacement completion.
The archive preserves both comparisons, first-use regressions, raw logs,
shared harness bytes, build receipts and the focused correctness gate.

The trace run also sets `DATAFRAME_EXECUTION_REPORT=1`, so every lazy query
prints its execution report (`LazyFrame.profile`): one `dataframe-operator:`
line per executed plan node with observed rows in and out, and for a join
the rows it indexed, the index kind and the build side. The runner keeps
them under `operators` in `results.json`, and both reports end with the
joins that indexed more rows than probed them, the largest first. That
table is the evidence for a join cost model: it says, from real counts,
where the executor hashed the larger side.

Two limits apply when reading a comparison. Timings on a shared machine move
with its load: prefer a quiet host, and rerun before acting on a small
change. And two builds can differ by up to about 10% on one query from code
layout alone: an A/B of builds that differed only by the inert tracing hooks
showed one query 10% faster in every round. Prefer the geometric means over
single cells, and confirm a single-query change on the full tier.

`--scale` overrides the tier's size: `smoke` (seconds; CI checks answers at
this size), `dev` (1M rows, scale factor 0.1), `default` (10M) or `large` (100M).

## Keeping holdouts useful

The practical boundary is tuning versus validation. Use development workloads
and general data properties to design the engine change, then check the
completed change on the holdouts and publish the whole selected result set,
including regressions, failures and unsupported queries. Do not reshape a
query, hide an unfavorable case, or add a query-specific branch to improve
a reported score. A path exercised by only one query needs a general reason
and coverage beyond that query.

Prior exposure is a limitation of the historical results. ClickBench stays
held out with that caveat recorded. Source hashes, benchmark/engine
separation and fast-path coverage make changes reviewable; they do not
certify anybody's tuning process.

### PDS-H became a development suite on 2026-10-05

Held-out suites only help if the development suites cover the same kinds
of work. They did not: H2O has ten group-by queries and five single joins,
and nothing with a multi-join plan, while PDS-H, the one suite behind
Polars, is made of them. Under the rule above its results could not be used
to choose work, so effort went to group-by, where the library already led.

PDS-H is therefore a development suite from that date, and the quick tier
runs it. TPC-DS takes its place as the held-out suite for join plans.
Reports made before the change keep the note they were made under
(`bench_policy.EARLIER_NOTES`); their PDS-H numbers were held-out numbers.

### TPC-DS became a development suite on 2026-10-15

By then H2O and PDS-H were within 1.5x of both Polars and DuckDB on every
query except PDS-H q2, while 10 of the 12 queries still outside 1.5x were
TPC-DS queries. With TPC-DS held out, the development suites gave no
evidence to choose the next piece of work from, and the results that did
show the gaps could not be used.

TPC-DS is therefore a development suite from that date, and the quick tier
runs it at scale factor 0.1. Its per-query timings may now be used to
choose work. ClickBench remains held out, and stays the only held-out suite
until another suite is added. Reports made before the change keep the note
they were made under; their TPC-DS numbers were held-out numbers.

### TPC-DS coverage

DuckDB runs all 99 queries. Polars and this library run the queries that
have been translated, and report the others as `unsupported: not
translated`, so every report counts them. Geometric means use only queries
every engine answered.

Translations are added in batches, each chosen by a rule applied to the
SQL text before any query in it is timed:

1. The 23 single-block queries: one SELECT, with no window function,
   rollup or set operation (q3, q7, q13, q15, q17, q19, q25, q26, q29, q37,
   q40, q42, q43, q48, q50, q52, q55, q72, q82, q84, q85, q91, q96).
2. The 13 with exactly one more SELECT, a derived table or a subquery,
   and still no window function, rollup, set operation, EXISTS or WITH
   (q21, q32, q34, q41, q45, q46, q62, q68, q73, q79, q92, q93, q99).
3. The 7 with several derived tables or subqueries under the same
   exclusions (q6, q9, q28, q61, q65, q88, q90).
4. The 5 that add only EXISTS or NOT EXISTS (q10, q16, q35, q69, q94).
5. The 7 whose WITH clause defines one table, under the same exclusions
   (q1, q24, q30, q39, q59, q81, q95).
6. The 23 with set operations or several WITH tables, and no window
   function or ROLLUP (q2, q4, q8, q11, q23, q27, q31, q33, q38, q54, q56,
   q58, q60, q64, q66, q71, q74, q75, q76, q78, q83, q87, q97; #575).
7. The 11 with window functions and no ROLLUP (q12, q20, q44, q47, q49,
   q51, q53, q57, q63, q89, q98; #575).

That is 89 of 99. The remaining 10 (q5, q14, q18, q22, q36, q67, q70, q77,
q80, q86) use ROLLUP or GROUPING and stay `unsupported: not translated`
until the library has ROLLUP, CUBE and GROUPING SETS (#802; #564 tracks
their translation).

Add further batches by a rule of the same kind, never by which queries run
well. A translation uses the API as a user would; where the
library cannot express or run a query, the cell stays failed or unsupported
with its reason until the library changes. The first run found two such
failures on the `decimal` variant, since fixed: q43 crashed (#464) and q40
needed decimal `fill_null` (#465).

## Query diagrams

Open [the query explorer](query-diagrams.html) in a browser to choose a
suite and query. It includes all 179 query IDs, including untranslated
TPC-DS queries, with Mermaid diagrams, query source, zoom controls, and
Mermaid/SVG downloads. The renderer is bundled in `docs/vendor`, so the explorer works offline.
Keep the HTML file and its `vendor` folder together when copying it.

For lazy suites, compare `explain(optimize=False)` with `explain()`.
These are logical plans with streaming capability annotations, not runtime
traces: `collect()` additionally orders eligible joins. The H2O diagrams
describe the eager operator sequences in their source. PDS-H q11 and q22
collect scalar subqueries before constructing their final plans; the plan
view shows the final plan, and the overview identifies that prerequisite.

Regenerate from the current checkout and existing smoke/base benchmark
data (requires the oracle environment and built Parquet library):

```bash
pixi run -e oracle python benchmarks/tools/query_diagrams.py
```

The command reads existing data under the usual benchmark data root;
`--data-root PATH` overrides it. It does not generate data or time queries.
It records the source fingerprint and saves the captured catalog in
`build/query-plans/plans.json`. To rerender that snapshot without compiling
Mojo or loading benchmark data:

```bash
python3 benchmarks/tools/query_diagrams.py --from-json build/query-plans/plans.json
```

## Answer checks

Every engine prints the same order-insensitive summary of its result: the row
count and one number per column (numeric sums, string byte lengths, dates in
days, datetimes in seconds). Values must agree within a relative 1e-6. Where a
query's result is not unique, only the columns that determine it are
compared: rows tied at a `LIMIT` boundary may legitimately differ between
engines, and between runs of one engine, so ClickBench's top-N queries compare
their `ORDER BY` column; q17, a `LIMIT` without `ORDER BY`, compares the row
count. `CHECKS` in `benchmarks/tools/bench_suites.py` lists them.

## Mechanism benchmarks

The programs in `benchmarks/` measure one mechanism at a time. Use them to
develop and calibrate a change, then confirm it on the suites above. Their
Python drivers, the suite driver `bench_suites.py`, its report builders and
the query diagram generator live in `benchmarks/tools/`.

| Benchmark | What it measures |
|---|---|
| `bench_csv`, `bench_numeric_parse`, `bench_cast_parse` | CSV ingestion and text-to-number parsing; the suites load data untimed, so these are the only parsing coverage |
| `bench_parquet_stream`, `bench_arrow_import` | Parquet reads and Arrow import, with their drivers in `benchmarks/tools/` |

## Retired benchmarks

These were removed because the external suites cover the same ground with
upstream queries and checked answers, or because the code they measured is
gone. Check out the revision listed for each to rerun it, for example to
reproduce a historical record in `docs/`.

| Benchmark | Revision | Why it was retired |
|---|---|---|
| `bench_vs_polars.mojo`, `scripts/bench_polars.py` (`bench-polars`) | `b9db9c6` | The old head-to-head; its grouped query was the shape the removed sum-plus-count group-by path matched (#306). The H2O suites replace it. Its data generator moved to `scripts/bench_join_data.py`, retired later (below) |
| `bench_upstream_joins.mojo`, `scripts/bench_upstream_joins.py`, `scripts/bench_upstream_revision.py` | `b9db9c6` | Hand ports of two DuckDB join micro-benchmarks, one of them the target of the removed count fusion (#304). The H2O join suite and PDS-H replace them. Its compiler guard moved to `benchmarks/tools/bench_host.py` |
| `bench_progression.mojo`, `scripts/bench_progression.py` | `b9db9c6` | Written with the equal-run progression path it measured (#308); only its rejection cases remained meaningful, and the join matrix's `shuffled` layout covered them until it was retired too |
| `bench_aggregate_fusion.mojo` | `b9db9c6` | Measured the fused Float64 group-by path removed in #319 |
| `bench_agg_shapes.mojo`, `scripts/bench_agg_shapes_polars.py` | `b9db9c6` | One-off comparison for #319; the H2O group-by suite and its variants cover it |
| `bench_join_matrix.mojo`, `scripts/bench_join_revision.py`, `scripts/bench_join_polars.py`, `scripts/bench_join_duckdb.py`, `scripts/bench_join_data.py` (`bench-joins-polars`, `bench-joins-duckdb`) | `8a50a95` | Thirteen hand-made join shapes against Polars and DuckDB. The H2O join suite, PDS-H and TPC-DS compare joins with both engines on upstream queries |
| `bench_join_cutoffs.mojo`, `bench_sort_cutoffs.mojo`, `bench_worker_sort.mojo`, `bench_join_build_ratio.mojo`, `scripts/bench_join_cutoffs.py`, `scripts/bench_sort_cutoffs.py`, `scripts/bench_workers.py` | `8a50a95` | One-time sweeps behind the cutoffs in join-cutoffs.md, sort-cutoffs.md and worker-calibration.md, which keep the results; the build-ratio results are at revision `0c1115a` |
| `bench_join.mojo`, `bench_group_by.mojo`, `bench_sort.mojo`, `bench_late_sort.mojo`, `bench_concat.mojo` (`bench-join`, `bench-group-by`, `bench-sort`, `bench-concat`) | `8a50a95` | Scaling of single operators over sizes, skew and key counts; the suites cover joins, group-by and sorts on real queries |
| `bench_lazy_pipeline.mojo`, `bench_lazy_streaming.mojo`, `bench_lazy_measurement.mojo`, `scripts/bench_lazy_streaming.py` | `8a50a95` | Lazy CSV pipelines, measured-input reuse and the streaming executor; the suites measure full queries end to end |
| `bench_suite.mojo` (`bench`, `bench-smoke`) | `8a50a95` | Quick mixed CPU workloads, replaced by the suites |
| `bench_fallbacks.mojo`, `bench_fallbacks_polars.py` | `8a50a95` | Development checks for mixed aggregation and general row sorting, with a Polars counterpart; the suites cover both on upstream queries |


Forced-Metal mismatch diagnostics retain the exact schema and duplicate-row
multiset comparator. Failed comparisons additionally record schema/cardinality,
whole-row and per-column multisets, and a second CPU query's comparison. Numeric
row comparisons require source-pinned output keys and runtime checks that both
outputs have unique, identical key sets; a zero- or one-row result can align by
position. Each aligned column reports mismatch and null counts; floating columns
also report finite ULP, absolute, and relative maxima separately from nonfinite
differences. These distances never change pass/fail. NaNs and signed zero retain
`DataFrame.equals` semantics. Declared sort keys can expose output ties but do
not prove that a tie caused a truncated-result difference. Missing certificates
and secondary diagnostic failures remain explicit. Case audits are embedded in
receipts, with their metadata hash, so a later checkout cannot silently replace
the measured alignment or hybrid classification. The CPU-only literal entry
[`mojo_cpu_metal_suite_diagnostics_probe`](https://github.com/randyzwitch/dataframe_mojo/blob/892b3626964719109af740703ee719b119bdc879/tests/tooling/mojo_cpu_metal_suite_diagnostics_probe.mojo) validated diagnostic
behavior independently of GPU execution (include `-I benchmarks`). Collection
durations remain diagnostic and are not performance measurements.

For ClickBench q11, q17, q18, q30–q32 and q38–q40, an independent ordering
audit can distinguish a valid unordered selection from a wrong result:

```sh
pixi run -e oracle python benchmarks/tools/metal_ordering_oracle.py \
  --coverage /path/to/coverage/coverage.json \
  --answer-root /path/to/coverage/answers \
  --input-root /path/to/benchdata \
  --output /path/to/ordering-audit.json
```

The audit requires exported answers and the original input files. It verifies
their receipt hashes and schemas, then uses DuckDB to calculate each complete
grouped relation. Every selected row must belong to that relation with the
correct multiplicity and aggregate values. Sorted queries must also return the
requested descending rank sequence, including the slice offset. This permits
different rows at tied boundaries only when both selections satisfy the query;
q17 permits any valid subset of up to ten rows because its group order is unspecified.
The reference definitions accept only the reviewed suite-source hash and fail
closed on unhandled null or nonfinite values. Numerical differences are not
tolerated by this audit.

The resulting `selection_valid` evidence is separate from the retained
`strict_status`. It does not rewrite coverage receipts, qualify timing pairs,
or certify GPU-only execution. Other queries require their own independent
references; a matching rank sequence alone never establishes correctness.

Float64 aggregate and arithmetic results use a separate numerical comparison
policy with relative tolerance `1e-9` and absolute tolerance `1e-12`:

```text
abs(actual - expected) <= max(1e-12, 1e-9 * max(abs(actual), abs(expected)))
```

The absolute threshold applies in the output column's units near zero; the
relative threshold scales with the result's magnitude. These are comparison
tolerances, not rounding of the computed results. Source-pinned entries in
`benchmarks/tools/metal_numerical_policy.json` identify eligible result fields.
Group keys, copied source attributes, integers, decimals, schemas and row
multiplicities remain exact. Nulls match only nulls, NaNs match only NaNs, and
infinities require the same sign. Signed zeros follow `DataFrame.equals`.

```sh
pixi run -e oracle python benchmarks/tools/metal_numerical_audit.py \
  --coverage /path/to/coverage/coverage.json \
  --answer-root /path/to/coverage/answers \
  --output /path/to/numerical-audit.json
```

This compares complete retained CPU and Metal row multisets. Exact fields
partition the rows; a one-to-one matching of the remaining fields preserves
duplicate counts without relying on a greedy pairing of approximately equal
values. If the comparison exceeds its candidate-pair limit, it reports
`unverified`. The original strict status remains available alongside the
numerical result. Agreement with CPU under this policy is not an independent
query oracle or a certificate of output ordering, and does not itself change
performance admission.
