# Lazy queries

`frame.lazy()`, `scan_csv(path[, schema])` or `scan_parquet(path)` starts a `LazyFrame`. Its methods
(`filter`, `select`, `select_exprs`, `with_columns`, `group_by(...).agg(...)`,
`join`, `sort`, `slice`, `head`, `limit`, `unique`, `drop`) only record plan
nodes; nothing reads data until `collect()`. `fetch(n)` collects the first `n`
rows. `explain()` prints the optimized plan with execution annotations, root first; `explain(optimize=False)`
and `collect(optimize=False)` show and run the plan as written.

`join(other, on, how)` joins on keys named alike on both sides;
`join(other, left_on=[...], right_on=[...], how, suffix, coalesce)` pairs
differently named keys (`o_custkey` with `c_custkey`), with the eager join's
output rules. `explain()` shows the pairs as `JOIN inner on o_custkey =
c_custkey`.

`collect(streaming=True, batch_size=65536)` is the default. Contiguous
row-wise operations run as ordered batches on a scoped worker pool. At most
one wave of batches (one per active worker) is in flight; workers return
results in source order, regardless of completion order. `streaming=False`
retains the materializing executor for comparison. Eager operations keep
their existing behavior.

The collected plan has the semantics documented in [semantics](semantics.md)
and [expressions](expressions.md). `collect_schema()` returns `name: dtype`
pairs by running with zero-row scans. CSV inference samples the usual rows,
but a schema probe does not decode data chunks. Explicit schemas therefore
validate field/plan structure without evaluating data values.

## Streaming and state

- CSV scans use the existing quote-aware scanner and typed decoder, with at
  most `batch_size` records in a job. Consumed mapped pages are unmapped after
  the wave has copied them, keeping resident input memory bounded on Linux
  and macOS. A single unusually large CSV record
  still requires memory for that record. Empty/unmappable sources retain the
  existing reader fallback; bounded input memory is guaranteed for regular
  mapped files, not arbitrary pipes.
- Parquet scans own an Arrow stream and decode one row group at a time, then
  slice it into batches. Temporary input memory also depends on the file's
  row-group size. EOF, early stop and errors release the stream exactly once.
- Filters, row-wise selects, with-columns, drop, explode and unnest operate
  within each batch. A select consisting only of scalar literals stays with
  the materializing evaluator so it produces one row for the entire input.
- Sum/count/mean/min/max/first/last, moments, boolean reductions and distinct
  counts use existing reducer states. Compound expressions over those
  reductions are evaluated after merging. Integer sums keep their exact
  128-bit state until final overflow checking. Floating-point reassociation
  follows the existing contract. Group representatives and state merges stay
  in first-occurrence order. Memory grows with groups and distinct-value
  state, rather than total input rows.
- Inner/left/semi/anti/cross joins retain the build side once and stream probe
  batches. Right/full joins, sort, unique, windows, median/quantile, implode
  and unsupported aggregate compositions retain materialization boundaries.
  Pipelines resume after these boundaries. Collect itself retains its final
  output, so collecting all rows is not a constant-memory operation.
- Positive slices preserve global row offsets and may stop the input after a
  wave satisfies the limit. Negative slices require the full result. Errors
  encountered in a wave propagate without exposing partial results; an early
  stop does not validate unread input. Batch size and scheduling can change
  how far beyond a requested head the reader evaluates.

`explain()` annotates streaming operators, aggregate state and materialization
boundaries; joins distinguish their probe and build behavior.
`explain(streaming=False)` retains the unannotated logical plan. A marker
reports the operator's execution capability: an upstream materialization
boundary still has to complete before downstream batches can run.

See [the streaming measurements](lazy-streaming.md) for peak RSS, timings,
raw samples and reproduction commands.

## Optimizations

- **Predicate pushdown.** A filter moves below a `with_columns` or `select` when
  it reads none of the columns that node produces (for `select`, only plain
  column selections), and into the side of a join that owns every column it
  reads: either side of an inner join, the left side of left, semi, and anti
  joins. A coalesced right key is not an output column, so no filter moves
  to the right side on its account. Row-local filters also move below a stable sort, since filtering the
  sorted rows preserves their relative order. Whole-column filters stay above
  the sort. A row-local filter directly above a CSV scan without a row limit
  runs on each decoded range before the ranges are assembled. Filters never
  move past a slice, unique, group_by, or right/full join, since that would
  change which rows those operators see.
- **Projection pushdown.** Scans read only the columns some operator above them
  uses; CSV scans pass them as `columns=`, so other fields are never decoded.
  A join passes each input only its own keys (the left keys to the left
  input, the right keys to the right) and the columns read above it from that
  side. Selectors, and joins with `coalesce=False`, keep every column. A plain column
  projection directly after a sort keeps the sorted row permutation and gathers
  only the selected output columns; sort-only key columns are not materialized
  in the result.
- **Slice pushdown.** `head(n)` moves below row-local `select`/`with_columns`
  (no reductions, windows, or `over`), and a leading slice directly over a CSV
  scan becomes the reader's `n_rows`, so the rest of the file is not read.
- **Row-group pruning.** A row-local filter directly above a Parquet scan
  first reads the file's footer statistics (`parquet_row_group_statistics`)
  and decodes only the row groups whose minimum and maximum admit a match.
  Bounds are used for `col <op> literal` with `<`, `<=`, `>`, `>=`, `==` on
  numbers and strings, joined with `&` and `|`; any other predicate, a
  column without statistics, or a text bound the writer did not mark exact
  keeps every group. The filter still runs on the rows that were read, so
  pruning only changes how much is decoded, never the result.

Tests check that optimized and unoptimized plans produce identical results for
every rewrite. Tests also compare batch sizes, worker counts, joins and aggregate
state against the materializing executor, including failures and early cleanup.
