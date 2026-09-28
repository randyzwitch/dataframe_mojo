# Benchmarks

Performance numbers in this project must describe how the engine behaves on
work other people chose, not on inputs we shaped. Two audits (#274, and the
follow-ups #304 and #306–#308) found fast paths whose only trigger was one of
our own benchmark queries or key layouts. The fix is structural: measure on
external suites, keep the measuring code apart from the code being measured,
and make benchmark-shaped paths visible.

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
3. **Development and held-out suites.** Tune against the development suites.
   Report from the held-out suites and do not read their per-query results to
   decide what to optimize. A change that helps the development suites but
   not the held-out ones probably does not generalize.
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
| `h2o_groupby` | development | [db-benchmark](https://github.com/duckdblabs/db-benchmark) group-by | 10 | 100, 10 and 2 groups per key; 5% nulls; sorted |
| `h2o_join` | development | db-benchmark join | 5 | none; 5% nulls |
| `pdsh` | held-out | [PDS-H](https://github.com/pola-rs/polars-benchmark), TPC-H derived | 22 | — |
| `clickbench` | held-out | [ClickBench](https://github.com/ClickHouse/ClickBench) `hits` | 43 | — |

The H2O data follows db-benchmark's R generators with seeded Polars sampling,
so distributions match but values do not. TPC-H tables come from DuckDB's
`dbgen`, with DECIMAL columns stored as DOUBLE because this library has no
decimal type yet (#229). ClickBench uses the first N one-million-row
partitions of `hits`, converted once from raw byte arrays and integer times to
strings, timestamps and dates, as ClickBench's own Polars and DuckDB scripts
do at load time. All engines read the same files.

Queries live in `benchmarks/suites/`:

- `engines.py`: DuckDB runs the upstream SQL (db-benchmark's queries,
  `tpch_queries()`, ClickBench's `queries.sql`); Polars runs idiomatic lazy
  translations.
- `h2o.mojo`, `pdsh.mojo`: this library's eager API, filtering each input
  before joining, because the lazy join cannot yet join on differently named
  keys, which every TPC-H join needs.
- `clickbench.mojo`: the lazy API, whose projection pushdown reads only the
  columns each query uses from the 105-column table.

A query this library cannot express is reported as unsupported with the
missing feature: H2O q9 (`corr`, #226) and ClickBench q28 (regular
expressions, #219).

## Running

There are two tiers. Data is generated or downloaded once into a directory
beside the main checkout, `<checkout>_benchdata` (override with
`DATAFRAME_BENCH_DATA`), and shared by every git worktree.

```bash
pixi run -e native build-dfparquet                                   # once

# After a code change: about two minutes once built.
pixi run -e oracle python3 scripts/bench_suites.py --baseline main

# For a report: hours; run it occasionally, not per change.
pixi run -e oracle python3 scripts/bench_suites.py --tier full --heldout
```

**Quick** (the default) runs the development suites at 1M rows with every
data variant, three rounds of three timed runs, and times only this library.
`--baseline REF` builds the suite runners against the library at `REF`
(checked out once as a git worktree under `build/suites/baseline/`, cached by
commit) and alternates the two builds. The report opens with **Changes vs
baseline**: only the cells where every round of one build beat every round
of the other by more than 3%. Polars and DuckDB outcomes come from a
reference cache keyed by data file, engine version, thread count and
repetitions, because their code does not change when this library does.
Narrow a run with `--suites h2o_join` or `--queries q1,q2`.

**Full** runs at 10M rows (TPC-H scale factor 1, 10M ClickBench rows with
`--heldout`), three rounds, measures every engine afresh and records
fast-path coverage.

Both tiers give each (suite, variant, engine, round) its own process, which
loads the tables once, untimed, then warms up and times each query. The
driver waits while a compiler runs, records the load average before every
worker and flags a busy host in the report. It writes raw samples with
provenance to `build/suites/results.json` and the report beside it;
`--report-from` re-renders a saved file.

Two limits apply when reading a comparison. Timings on a shared machine move
with its load: prefer a quiet host, and rerun before acting on a small
change. And two builds can differ by up to about 10% on one query from code
layout alone: an A/B of builds that differed only by the inert tracing hooks
showed one query 10% faster in every round. Prefer the geometric means over
single cells, and confirm a single-query change on the full tier.

`--scale` overrides the tier's size: `smoke` (seconds; CI checks answers at
this size), `dev` (1M rows), `default` (10M) or `large` (100M).

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

