# Roadmap

dataframe_mojo is a native dataframe library for Mojo. It has three goals.
Each has one number we track, so progress is a measurement and not an opinion.

## Goal 1: on the CPU, match DuckDB and Polars

The library should do what both engines do, and do it as fast.

**Performance metric.** Per-query parity: every query in the external suites
(H2O, PDS-H, TPC-DS; ClickBench is held out) runs within 1.5x of both Polars
and DuckDB. Gaps of about a millisecond are exempt. The judge is each query,
never the suite geomean. See [benchmarks](docs/design/benchmarks.md).

- Latest: checkpoint 20, 2026-10-19, main `158285f7`. 1 of 169 queries was
  outside (q32), since fixed by #886.
  230 of 282 development-suite measurements were at or faster than both engines.
- Source: the benchmark report kept outside the tree (reports are not committed;
  see [benchmarks](docs/design/benchmarks.md), rule 7).
- Open work: small TPC-DS queries against Polars, where fixed cost dominates.

**Feature metric.** The open issues labelled `api-parity`. Each is a user-facing
feature Polars or DuckDB has and we do not. Open on 2026-10-22:

- #228 GeoJSON, JSON and NDJSON read and write
- #792 Per-group expression operations: filter, sort_by, head/tail, unique, top_k_by, max_by/min_by
- #793 over(order_by=...), per-partition row numbers, shift fill_value, ntile/percent_rank/cume_dist
- #794 Temporal literals and missing dt functions
- #795 Arrow IPC (Feather) read, scan and write
- #796 Multiple files, globs, hive partitions, sink_parquet/sink_csv, compressed CSV
- #797 Wider list namespace
- #798 json_decode, json_path_match, Unicode normalize
- #799 to_dummies, transpose and upsample
- #800 Rolling median/quantile, centered and weighted windows, ewm_mean/ewm_std
- #801 join_where, join validate and expression join keys
- #802 ROLLUP, CUBE and GROUPING SETS
- #803 eq_missing, expression-level is_duplicated and friends, approx_n_unique
- #804 DataFrame conveniences: frame reductions, fill_null strategies, to_dict, row iterators
- #805 User-defined functions
- #806 A SQL front end that compiles to LazyFrame
- #807 Array, Enum and Int128 dtypes, cat/bin namespaces
- #878 Regex follow-ups

## Goal 2: hand work to the Apple GPU when it pays

On Apple silicon, the engine should send a query to Metal when a cost model says
that is faster, and keep it on the CPU otherwise. This project exists partly to
show that automatic hand-off is workable.

**Metric.** Auto-placement ratio: on the external suites on Apple silicon,
`engine="auto"` time divided by the better of CPU and forced Metal, per query.
A ratio near 1.0 means the cost model picks well.

- Latest: not measured. Automatic placement is deferred (#583) behind the Metal
  backend rebuild (#868).
- Baseline today: forced Metal is 32.6x slower than CPU on the development
  suites (PR #895). The rebuild has to close that gap before placement has
  anything to choose.
- Metal changes are tested on a local Apple silicon Mac (see
  [Apple Metal](docs/apple-metal.md)).

## Goal 3: other accelerators plug in through one contract

Mojo is meant to target many accelerators. Adding one should not mean touching
the planner or the executor.

**Metric.** Backends that run through the shared contract. Today there are two
in use: Metal lives in this repository; NVIDIA lives in the private extensions
repository and replaces the provider file at build time.

- Status: the shared contract is being written
  (`docs/design/accelerator-contract.md`, in progress). Until it lands, the
  boundary is described in [accelerator backends](docs/design/accelerator-backends.md).
- Done when: both backends use only the contract, and a third can be added
  without edits outside its own directory.

## How we work

A cleanup in seven steps, so the repository looks like what it is:

| Step | State |
| --- | --- |
| 1. Benchmark evidence out of the tree | Done, #918 |
| 2. Runnable tests only, docs split into user docs and design notes | Done, #920, #921 |
| 3. CI run per area | Open, #923 |
| 4. Coverage checks | In progress |
| 5. Scheduled reports | In progress |
| 6. Accelerator boundary | In progress |
| 7. Roadmap, contributing guide, changelog, issue templates (this change) | Done in this PR |

Rules that shape every change:

- Performance work ports the design DuckDB and Polars use, and is measured on the
  external suites. No fast path tuned to one benchmark query.
- Features follow Polars semantics, and DuckDB's result types.
- Features come with eager and lazy support and tests of both.
- One change per pull request. See [CONTRIBUTING.md](CONTRIBUTING.md).
