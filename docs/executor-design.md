# Executor redesign: thread-owned pipelines over morsels

Status: design, 2026-10-09. Tracked by #538 (part of #491).

## Why

With the benchmark machine on the `performance` governor (#527), 67 of
134 queries are outside the goal of 1.5x of both Polars and DuckDB. The
profiles of the queries that remain outside say the same thing whatever
the query: on 8 threads, 3 to 4 are busy on average. ClickBench q7 spends
55% of its samples in idle crew threads, PDS-H q2 and q9 56%. The kernels
inside operators are within reach of the other engines'; the time is in
the seams between operators and in the main thread's work between
rounds. Kernel fixes move a few queries by 5 to 30% per PR; they do not
change the shape that puts the threads to sleep.

Three structural differences from DuckDB and Polars, each visible in a
query that is outside the goal:

1. **Every operator materializes its whole output.** PDS-H q2 joins 1,987
   suppliers with partsupp: the join builds 159K-row pair lists, restores
   left-major order, gathers every selected column into a 159K-row frame,
   then a filter keeps 460 rows and a top-100 keeps 100. DuckDB runs
   probe, filter and top-k as one pipeline over 2,048-row chunks: the
   probe side is a selection over the chunk, the build side is gathered
   per chunk, and the next operator consumes the chunk while it is in
   cache. Nothing of size 159K ever exists.
2. **Parallelism is fork-join per operator.** Each stage splits its input
   across workers and joins them; the main thread concatenates,
   rechunks, samples and decides between stages, and intermediate results
   of 100K to 300K rows run on two or three workers under the 64K-row
   floor. DuckDB's `PipelineExecutor` and Polars' `polars-stream` give
   each thread the whole pipeline: a thread takes a morsel from the
   source, runs every operator on it, and pushes into its own sink state;
   the states are combined once at the end.
3. **Fixed cost per batch and per stage.** The streaming executor's round
   structure (a job list per round, states held and judged on the main
   thread, replayed batches) and the eager exits (a grouped aggregation
   after a filter leaves the stream; a join whose left may be small leaves
   the stream) are why ClickBench q1, q7 and q19 take 3 to 7 ms where the
   other engines take 1 to 3.

## What the other engines do, from their source

DuckDB (`src/parallel/pipeline_executor.cpp`,
`src/include/duckdb/execution/physical_operator.hpp`,
`src/execution/join_hashtable.cpp`, `src/common/types/data_chunk.cpp`):

- A query is a set of pipelines; each pipeline is a source, a list of
  operators and a sink. Pipelines are ordered by dependencies: the hash
  join's build pipeline completes before its probe pipeline starts.
- Each thread runs its own `PipelineExecutor` for a pipeline: it fetches
  a chunk from the source (`GetData` with a local source state), pushes it
  through the operators (`Execute(input, output)`, each with its own
  intermediate chunk; an operator may answer `HAVE_MORE_OUTPUT` and be
  called again for the same input, which is how a join emits several
  output chunks for one input chunk), and pushes the result into the sink
  with a thread-local sink state (`Sink`). When the source is exhausted
  the thread's local state is merged into the global one (`Combine`), and
  `Finalize` runs once.
- A chunk holds at most 2,048 rows (`STANDARD_VECTOR_SIZE`). A filter
  computes a `SelectionVector` and the output chunk is `Slice`d: its
  vectors become dictionary vectors over the input, no values copied. The
  hash join probe does the same for the probe side
  (`ScanStructure::NextInnerJoin`) and gathers only the build side's
  payload columns, per output chunk, from the hash table's row layout
  (`GatherResult`).
- Sinks keep per-thread state: the hash aggregate has thread-local
  partitioned tables combined by partition at the end; the hash join
  build appends to a thread-local collection, then the table is built
  over all of them in parallel; top-n keeps a thread-local heap.

Polars (`crates/polars-stream/src/nodes/*.rs`, `morsel.rs`,
`nodes/joins/equi_join.rs`):

- Nodes are connected by pipes carrying morsels of about 100,000 rows
  (`DEFAULT_IDEAL_MORSEL_SIZE`), one task per pipeline lane per node. A
  node like `FilterNode` evaluates the predicate on the morsel and filters
  the frame sequentially; parallelism comes from the lanes, not from
  splitting the morsel.
- `EquiJoinNode` builds per thread: each builder partitions its morsels
  by key hash and keeps per-partition row indices; the probe tables are
  then assembled per partition from every builder's rows. The probe task
  hashes a morsel's keys, probes each partition's table, and
  `gather_extend`s the matched payload from both sides into frame
  builders that emit output morsels of at most the morsel size. Order is
  preserved only when `preserve_order_probe` asks for it; otherwise the
  output morsels take a fresh sequence number.

The two agree on the shape and differ on the morsel size. DuckDB's 2,048
suits its vector kernels with no per-call cost; our kernels are column
operations with a few microseconds of per-call cost (a `Series`, an
`ArcPointer`, a job), so Polars' size is the one to copy: 64K to 128K
rows a morsel, and no splitting inside a morsel.

## The design

### Morsel

A morsel is what travels through a pipeline on one thread:

```
struct Morsel:
    var frame: DataFrame            # columns, possibly shared with the source
    var rows: Optional[List[Int]]   # a selection over `frame`, or every row
    var sequence: Int               # position in the source, for ordered sinks
```

A filter narrows `rows` and does not copy a column. An operator that
reads a column through a selection gathers it then, and the gathered
column replaces the shared one in the morsel so the next operator reads
it directly: materialization happens at most once per column and only
for the columns an operator reads. The hash join probe produces a morsel
whose probe-side columns are the input's with a new selection and whose
build-side columns are gathered from the build payload for the matched
rows, as DuckDB's `NextInnerJoin` does. A projection that drops a column
drops it before any gather.

The current `_StreamJob` already carries a frame per batch and applies
the stream's operations to it; the selection is what is new, together
with the rule that nothing gathers before an operator reads.

### Pipelines

A plan becomes pipelines as in DuckDB: walking down from the collect, a
pipeline breaker (a grouped or ungrouped aggregation, a sort or top-k, a
join's build side, a unique) ends a pipeline and becomes its sink; the
build side of a join is a pipeline of its own that must finish first. The
`_stream_execute` walk that already finds operations and joins below a
terminal is the starting point; the eager exits in it go away, because
the sinks below make the streamed version at least as fast.

Sources: an in-memory frame (morsels are slices), Parquet row groups,
CSV batches. Each source hands out morsels through an atomic cursor, so
a thread takes the next morsel when it finishes its last one; no round
structure and no job list per round.

### Thread-owned execution

`run_pipeline(pipeline, workers)`: every worker thread (the crew of #514,
the caller included) loops: take a morsel from the source, run every
operator on it, push into the thread's local sink state. When the source
is exhausted, local states are combined (in parallel where the sink
allows it, as the partitioned aggregate does) and the sink finalizes
once. The main thread does no per-morsel work.

### Sinks, each with thread-local state

- **Materialize** (the collect itself): each thread appends its morsels'
  gathered frames to a local list; combine concatenates in sequence
  order (or any order when the plan above is order-free).
- **Hash aggregate**: the plan picks one of two modes for every thread
  from 4,096 sampled key rows. Few groups (a quarter or fewer of the
  sample distinct) and fixed-width keys: each thread keeps a table of
  group ids keyed by the key's equality words and their hash, split
  into 16 hash parts once it passes 4,096 groups, updating reducer
  states in place from the morsel's rows (a reduction of a computed
  input evaluates it once per morsel); combine merges the threads'
  parts of one hash on one thread, parts in parallel, or the unsplit
  tables as one (DuckDB's thread-local radix-partitioned aggregate
  table, which also partitions only as it grows).
  Many groups: each thread keeps its morsels' selections over the source
  frame (or their output frames when a step made columns), and the
  finish gathers the kept rows once and runs the eager partitioned
  group-by over them: one scatter of every row into buckets, each bucket
  encoded on its own thread, which inserting rows one at a time into
  tables that outgrow the caches does not match (ClickBench q32, 10M
  groups: 850 ms inserted against 300 scattered); Polars' streaming
  group-by and DuckDB's sink partition and finalize the same way once
  their tables grow. There is no per-batch `_StreamReduction` and no
  state merge on the main thread; the collect-or-merge judgement (#530)
  is made once, before any row is read. Open: strings. The tables
  compare keys by their words, exact for fixed-width keys only, and the
  late gather of string rows costs more than the eager route's per-chunk
  filter (q12: 85 ms against 48), so a plan that groups or reduces
  strings under a filter or projection still takes the eager route; the
  fix is a row-layout key store with the strings' bytes, as DuckDB's
  aggregate hash table keeps them.
- **Hash join**: the build side is a sub-plan collected before the
  pipeline runs (its own pipelines), rechunked once, and indexed with
  `prepare_hash_index` (parallel over buckets); the index is shared by
  every worker. The probe is a step: a worker probes the index with its
  morsel's rows on its own thread and gathers the matched rows of both
  sides in one piece, which is the next step's morsel (DuckDB's
  `PhysicalHashJoin::ExecuteInternal` on one chunk, Polars' probe task).
  Inner, left, semi and anti joins run this way under every sink except
  a grouped sink that must keep first-occurrence order, since a morsel's
  output rows carry no unique position in the join's left-major order.
  Still to do from the design: a build sink that partitions each
  worker's morsels by hash and builds per partition (Polars'
  `BuildState`), and a probe-side selection instead of a gather for
  joins that keep most rows.
- **Top-k**: each thread keeps its morsels' output rows as candidates and
  folds them to the k best once they pass a few thousand rows; after a
  fold it publishes its k-th first-key value into a bound shared by
  every thread through two atomics, and before each morsel adopts the
  tightest, narrowing the morsel to the rows that can still enter the
  top k when at most one in eight can (DuckDB's top-n heap boundary
  shared across threads). Candidates carry their place in the source as
  a hidden last sort key, so ties come out as a stable sort of the whole
  input would place them; the finish sorts all threads' candidates once.
- **Sort**: still the materialize sink followed by the eager sort; sorted
  runs per thread merged once remain to do.
- **Unique**: a hash aggregate with no reductions.

### Order

Joins keep the documented left-major order only where the plan above
can see it (`_order_free_above` decides today); otherwise output morsels
take probe order, as Polars does without `preserve_order_probe`. Ordered
sinks use the morsel sequence. A sort or top-k above a join sees ties in
probe order, which is the same contract Polars has had since 1.0.

### What it replaces

- The round loop in `_stream_execute` (held states, replay, the judge).
- The eager exits for grouped aggregations after a filter and for joins
  with a small left.
- Per-batch `_StreamReduction` states and their merge.
- The whole-output join (`_join_impl` through `_smaller_build_join_rows`)
  as the lazy path's join; the eager `DataFrame.join` stays for eager use.

The eager `DataFrame` API is unchanged; `collect()` runs pipelines.

## Stages and gates

Every stage is a PR against main, measured under the `performance`
governor with the quick tier, the full test suite, and best-of-7 direct
runs of the queries named; the full tier at the checkpoints of #491.

1. **Pipeline runner and morsels.** `Morsel`, the source cursor,
   `run_pipeline` on the crew, the materialize sink, and filter, select,
   with_columns, drop as morsel operators with selections. The plans this
   serves (row-local steps under a collect) run through it; everything
   else as today. Gate: ClickBench q19 and q1 within 1.5x; no quick-tier
   regression over 3%.
2. **Hash aggregate sink.** Thread-local partitioned tables, parallel
   combine, grouped and ungrouped. Removes the per-batch states and the
   eager exit after a filter. Gate: ClickBench q7, TPC-DS q39 and q65,
   H2O group-by unchanged or faster.
3. **Hash join as build sink and probe operator.** Build pipeline per
   join, probe-side selection, build payload gathered per morsel, order
   only where seen. Gate: PDS-H q2, q5, q9, q10, q21 and TPC-DS q24, q84
   at the checkpoint; no PDS-H query slower than 3%.
4. **Top-k and sort sinks; unique.** Gate: ClickBench q26, q37, q40, q41.
5. **Remove the old paths** that no plan reaches, and the trace paths
   with them; the coverage table in the report shows what remains.

Each stage's numbers go into `docs/benchmarks.md` and the epic.
