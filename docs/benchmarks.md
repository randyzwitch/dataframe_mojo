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
