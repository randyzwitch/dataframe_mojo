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
   `dataframe/` must not also add or change a suite, its data generator or
   its queries (`benchmarks/`, `scripts/bench_*`). This stops a fast path and
   the benchmark that justifies it from arriving together.
   `scripts/check_benchmark_separation.sh` enforces this in CI; a PR that
   genuinely needs both (for example, this one, which adds tracing hooks and
   the suites together) says why in a `Benchmark-Change:` commit trailer.
3. **Development and held-out suites.** Tune on H2O, PDS-H and mechanism
   benchmarks. Keep TPC-DS and ClickBench as held-out validation of
   completed changes. Do not use their per-query timings to choose
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
| `tpcds` | held out | TPC-DS, from DuckDB's `tpcds` extension (`dsdgen`, `tpcds_queries()`) | 99, of which 43 are translated | money columns as DOUBLE (`base`) or as declared decimals (`decimal`) |
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

There are two tiers. Data is generated or downloaded once into a directory
beside the main checkout, `<checkout>_benchdata` (override with
`DATAFRAME_BENCH_DATA`), and shared by every git worktree.

```bash
pixi run -e native build-dfparquet                                   # once

# After a code change: about two minutes once built.
pixi run -e oracle python3 scripts/bench_suites.py --baseline main

# For a report: hours; run it occasionally, not per change.
pixi run -e oracle python3 scripts/bench_suites.py --tier full --all-suites
```

**Quick** (the default) runs the development suites (H2O at 1M rows, PDS-H at
scale factor 0.1) with every data variant, three rounds of three timed runs, and times only this library.
`--baseline REF` builds the suite runners against the library at `REF`
(checked out once as a git worktree under `build/suites/baseline/`, cached by
commit) and alternates the two builds. The report opens with **Changes vs
baseline**: only the cells where every round of one build beat every round
of the other by more than 3%. Polars and DuckDB outcomes come from a
reference cache keyed by data file, engine version, thread count and
repetitions and reference query-source hash, because their code does not
change when this library does.
Narrow a run with `--suites h2o_join` or `--queries q1,q2`.

**Full** runs both development and held-out suites at 10M rows (TPC-H and
TPC-DS scale factor 1, 10M ClickBench rows), three rounds, measures every
engine afresh and records fast-path coverage. `--all-suites` also selects
all five with the quick tier; `--suites` can explicitly narrow either tier.
The `--heldout` option adds TPC-DS and ClickBench to a run.

Both tiers give each (suite, variant, engine, round) its own process, which
loads the tables once, untimed, then warms up and times each query. The
driver waits while a compiler runs, records the load average before every
worker and flags a busy host in the report. It writes raw samples with
provenance to `build/suites/results.json` and the report beside it, as
Markdown (`results.md`) and as a self-contained HTML page (`results.html`,
from `scripts/bench_html.py`) that opens in any browser; `--report-from`
re-renders both from a saved file. Provenance includes engine and benchmark
working-tree status, local query/generator/runner SHA-256 hashes and an
exposure record for each selected suite. These identify the local workload;
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

Two limits apply when reading a comparison. Timings on a shared machine move
with its load: prefer a quiet host, and rerun before acting on a small
change. And two builds can differ by up to about 10% on one query from code
layout alone: an A/B of builds that differed only by the inert tracing hooks
showed one query 10% faster in every round. Prefer the geometric means over
single cells, and confirm a single-query change on the full tier.

`--scale` overrides the tier's size: `smoke` (seconds; CI checks answers at
this size), `dev` (1M rows), `default` (10M) or `large` (100M).

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
pixi run -e oracle python scripts/query_diagrams.py
```

The command reads existing data under the usual benchmark data root;
`--data-root PATH` overrides it. It does not generate data or time queries.
It records the source fingerprint and saves the captured catalog in
`build/query-plans/plans.json`. To rerender that snapshot without compiling
Mojo or loading benchmark data:

```bash
python3 scripts/query_diagrams.py --from-json build/query-plans/plans.json
```

## Answer checks

Every engine prints the same order-insensitive summary of its result: the row
count and one number per column (numeric sums, string byte lengths, dates in
days, datetimes in seconds). Values must agree within a relative 1e-6. Where a
query's result is not unique, only the columns that determine it are
compared: rows tied at a `LIMIT` boundary may legitimately differ between
engines, and between runs of one engine, so ClickBench's top-N queries compare
their `ORDER BY` column; q17, a `LIMIT` without `ORDER BY`, compares the row
count. `CHECKS` in `scripts/bench_suites.py` lists them.

## Mechanism benchmarks

The programs in `benchmarks/` measure one mechanism at a time. Use them to
develop and calibrate a change, then confirm it on the suites above.

| Benchmark | What it measures |
|---|---|
| `bench_csv`, `bench_numeric_parse`, `bench_cast_parse` | CSV ingestion and text-to-number parsing; the suites load data untimed, so these are the only parsing coverage |
| `bench_parquet_stream`, `bench_arrow_import` | Parquet reads and Arrow import, with their drivers in `scripts/` |
| `bench_join_matrix` | Thirteen join shapes on `base`, `shuffled` and `wide` key layouts; `scripts/bench_join_revision.py` pairs two revisions, and `bench_join_polars.py` / `bench_join_duckdb.py` add the other engines. Inputs come from `scripts/bench_join_data.py`. Treat `shuffled` and `wide` as the general case |
| `bench_join_cutoffs`, `bench_sort_cutoffs`, `bench_worker_sort`, `bench_join_build_ratio` | Sweeps behind named cutoff constants; see join-cutoffs.md, sort-cutoffs.md and worker-calibration.md |
| `bench_join`, `bench_group_by`, `bench_sort`, `bench_late_sort`, `bench_concat` | Scaling of single operators over sizes, skew and key counts |
| `bench_lazy_pipeline`, `bench_lazy_streaming` | Lazy CSV pipelines and the streaming executor |
| `bench_suite` | Quick CPU workloads; `pixi run bench-smoke` runs it in CI |

## Retired benchmarks

These were removed because the external suites cover the same ground with
upstream queries and checked answers, or because the code they measured is
gone. Check out revision `b9db9c6` to rerun them, for example to reproduce a
historical record in `docs/`.

| Benchmark | Why it was retired |
|---|---|
| `bench_vs_polars.mojo`, `scripts/bench_polars.py` (`bench-polars`) | The old head-to-head; its grouped query was the shape the removed sum-plus-count group-by path matched (#306). The H2O suites replace it. Its data generator moved to `scripts/bench_join_data.py` |
| `bench_upstream_joins.mojo`, `scripts/bench_upstream_joins.py`, `scripts/bench_upstream_revision.py` | Hand ports of two DuckDB join micro-benchmarks, one of them the target of the removed count fusion (#304). The H2O join suite and PDS-H replace them. Its compiler guard moved to `scripts/bench_host.py` |
| `bench_progression.mojo`, `scripts/bench_progression.py` | Written with the equal-run progression path it measured (#308); only its rejection cases remained meaningful, and the join matrix's `shuffled` layout covers them |
| `bench_aggregate_fusion.mojo` | Measured the fused Float64 group-by path removed in #319 |
| `bench_agg_shapes.mojo`, `scripts/bench_agg_shapes_polars.py` | One-off comparison for #319; the H2O group-by suite and its variants cover it |
