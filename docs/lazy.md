# Lazy queries

`frame.lazy()`, `scan_csv(path[, schema], ...)` (with `read_csv`'s options; see
[CSV](csv.md)), `scan_ndjson(path[, schema], ...)` (see [JSON](json.md)),
or `scan_parquet(path)` starts a `LazyFrame`. Its methods
(`filter`, `select`, `select_exprs`, `with_columns`, `group_by(...).agg(...)`,
`join`, `sort`, `slice`, `head`, `limit`, `tail`, `reverse`, `shift`,
`with_row_index`, `unique`, `drop`, `rename`, `cast`) only record plan nodes; nothing reads data until `collect()`. `fetch(n)` collects the first `n`
rows. `explain()` prints the optimized plan with execution annotations, root first; `explain(optimize=False)`
and `collect(optimize=False)` show and run the plan as written.

`collect`, `profile`, and `fetch` default to `engine="auto"`; `engine="cpu"`
forces CPU execution and `engine="accel"` requires an accelerator provider.
In a CPU-only build (the default), auto uses CPU without accelerator dependencies or
device discovery. Eager operations retain their existing behavior.

The core exposes an `AcceleratorBackend` extension interface. A separately
built provider owns automatic capability, memory and cost checks through
`select_auto(plan, optimize, streaming, batch_size)`, returning a placement (`cpu`, `accel`, or `mixed`) and a
reason before execution. Providers without a policy inherit CPU selection.
The core delegates selected accelerator/mixed plans to `execute_auto`, which
receives the CPU optimization, streaming and batch-size options for any CPU
portion. Provider execution errors propagate; collection never retries them
on CPU. `explain(engine="auto")` shows the placement reason and automatic
profiles add a `selection_reason` column.

`engine="accel"` raises a missing-provider error unless a provider is registered
or explicitly passed with `accelerator=...`. Explicit acceleration never
falls back to CPU. An explicitly passed provider owns automatic selection
when `engine="auto"` is supplied; omitting `engine` on these explicit-provider
overloads retains forced acceleration for compatibility.

`rename(mapping, strict=True)` and `cast(dtypes, strict=True)` give the eager
methods' results. As Polars builds them, `rename` is a `select` of every input
column, aliased where renamed (the input's names are resolved when it is
called, as `collect_schema` resolves them), and `cast` is a `with_columns` of
`col(name).cast(dtype, strict)` per entry, so both let filters and column
selections through (see Optimizations); a filter moved below a `rename`
reads the input's names. `strict=False` makes `rename` ignore
names the input lacks; a strict cast that fails raises when the plan runs.

`group_by_rollup`, `group_by_cube` and `group_by_grouping_sets` expand to one
`group_by` per grouping set, concatenated; see [Expressions](expressions.md).

`group_by` takes column names or expressions, as `DataFrame.group_by` does.
An expression key is evaluated once and named by its output name, and the
aggregations still read the input's columns: computed keys become a
`with_columns` below the group_by (DuckDB moves group expressions into a
projection under its aggregate the same way), under a placeholder name that
is renamed afterwards when an aggregation reads a column of the key's name.
Besides `agg`, a `LazyGroupBy` has `len(name="len")` (rows per group as
Int64) and `sum`, `mean`, `min`, `max`, `median`, `first`, `last`,
`n_unique` and `quantile(q, interpolation="nearest")`, which reduce every
column that is not a key, as Polars' do; the eager `GroupBy` has the same
methods.

`join(other, on, how)` joins on keys named alike on both sides;
`join(other, left_on=[...], right_on=[...], how, suffix, coalesce)` pairs
differently named keys (`o_custkey` with `c_custkey`), with the eager join's
output rules. `explain()` shows the pairs as `JOIN inner on o_custkey =
c_custkey`.

`gather`, `top_k`, `bottom_k` and `clear` record plan nodes too.
`describe(percentiles, interpolation)` returns `DataFrame.describe`'s summary,
computed as Polars computes it: one `select` of the summary reductions over
the plan, collected, so the rows are not materialized whole. `clear(n)` is
`head(0)` for `n = 0` and otherwise, as in Polars, a new plan over `n` rows of
nulls with the schema. `top_k(k, by, reverse=False)` and `bottom_k` are a
sort with nulls last followed by `head(k)`, which runs as one top-k selection
(see Optimizations). `gather(indices, null_on_oob=False)` takes a list of
positions or a plan with one integer column (a null position gives a row of
nulls); it is its own operator, like Polars' `Gather`, and filters and slices
above it stay above it.

`concat([lf1, lf2, ...], how)` combines plans with the eager `concat`'s
results and errors. `vertical` stacks them (`lf1.concat(lf2)` is the
two-plan form). `diagonal`, as Polars plans it, stacks the inputs each
selected to the union of the columns, a typed null filling the columns an
input lacks; the inputs' names and types are resolved when `concat` is
called, and a filter or column selection above reaches every input.
`horizontal` places the inputs side by side (an `HCONCAT` operator); a column
selection above reaches each input, and the heights must match when it runs.

`collect(streaming=True, batch_size=65536)` is the default. Contiguous
row-wise operations run as ordered batches on a scoped worker pool. At most
one wave of batches (one per active worker) is in flight; workers return
results in source order, regardless of completion order. `streaming=False`
retains the materializing executor for comparison. Eager operations keep
their existing behavior.

`pivot(on, on_columns, index=[...], values=..., aggregate_function="")` is
`DataFrame.pivot` with the new columns named up front, as Polars 1.44's
`LazyFrame.pivot(on, on_columns, ...)` takes them, so the plan's schema is
known before it runs. The output is the index columns, then one column per
entry of `on_columns` in that order (a list of names, or a Series of `on`
values named by their text, `null` for a null), each holding the cells whose
`on` value has that name. A name no row has gives a column of nulls of the
aggregate's type; rows whose `on` value is not listed still give their index
key a row. The pivot runs on its whole input; a filter on index columns alone
moves below it, and the scan below reads only the index, `on` and values
columns.

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
  batches. Right/full joins, sort (except a sort cut short by a slice, which
  keeps each batch's first rows), unique, `tail`, `reverse`, `shift`,
  `with_row_index`, windows, median/quantile, implode and unsupported
  aggregate compositions retain materialization boundaries. The four
  row-order operators take their whole input in order and run the eager
  DataFrame method on it.
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

See [the streaming measurements](design/lazy-streaming.md) for peak RSS, timings,
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
  move past a slice, unique, group_by, right/full join, `tail`, `reverse`,
  `shift`, `with_row_index`, `gather` or a horizontal concat, since that would change which rows those
  operators see (or, for `with_row_index`, the numbers they give). Slices
  do not move below them either.
- **Projection pushdown.** Scans read only the columns some operator above them
  uses; CSV scans pass them as `columns=`, so other fields are never decoded.
  A join passes each input only its own keys (the left keys to the left
  input, the right keys to the right) and the columns read above it from that
  side. A `select` or `with_columns` drops the expressions whose output
  nothing above reads (a `select` only when every expression is row-local and
  one it keeps reads a column), so a column selection passes through a
  `rename` or `cast` and an unread cast is never computed. Selectors, and joins with `coalesce=False`, keep every column. A plain column
  projection directly after a sort keeps the sorted row permutation and gathers
  only the selected output columns; sort-only key columns are not materialized
  in the result.
- **Slice pushdown.** `head(n)` moves below row-local `select`/`with_columns`
  (no reductions, windows, or `over`), and a leading slice directly over a CSV
  scan becomes the reader's `n_rows`, so the rest of the file is not read.
- **Top-k.** A slice directly above a sort (`sort(...).head(k)` or
  `slice(offset, k)`, SQL's `ORDER BY ... LIMIT`) needs only the sort's
  first `offset + k` rows, so the sort selects them instead of sorting every
  row, and `explain()` shows it as `TOP_K`. The result is exactly the full
  sort's rows, including null and NaN placement (and tie order under
  `maintain_order=True`). When the
  plan streams, each batch keeps its own first rows, so memory stays at a
  batch plus those rows. At a tenth of the input or more, the full sort is
  used and cut short.
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
