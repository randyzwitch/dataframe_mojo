# Lazy queries

`frame.lazy()`, `scan_csv(path[, schema])` or `scan_parquet(path)` starts a `LazyFrame`. Its methods
(`filter`, `select`, `select_exprs`, `with_columns`, `group_by(...).agg(...)`,
`join`, `sort`, `slice`, `head`, `limit`, `unique`, `drop`) only record plan
nodes; nothing reads data until `collect()`. `fetch(n)` collects the first `n`
rows. `explain()` prints the optimized plan, root first; `explain(optimize=False)`
and `collect(optimize=False)` show and run the plan as written.

Plan nodes execute through the matching eager operator except that a row-local
filter directly above an unrestricted CSV scan runs inside each decode worker.
The collected plan has the semantics documented in [semantics](semantics.md) and
[expressions](expressions.md). `collect_schema()` returns `name: dtype` pairs by
running the plan with every scan returning zero rows; binding therefore
validates expressions exactly as `collect` would, without reading rows (a CSV
scan without a schema still samples the file to infer types).

## Optimizations

- **Predicate pushdown.** A filter moves below a `with_columns` or `select` when
  it reads none of the columns that node produces (for `select`, only plain
  column selections), and into the side of a join that owns every column it
  reads: either side of an inner join, the left side of left, semi, and anti
  joins. Row-local filters also move below a stable sort, since filtering the
  sorted rows preserves their relative order. Whole-column filters stay above
  the sort. A row-local filter directly above a CSV scan without a row limit
  runs on each decoded range before the ranges are assembled. Filters never
  move past a slice, unique, group_by, or right/full join, since that would
  change which rows those operators see.
- **Projection pushdown.** Scans read only the columns some operator above them
  uses; CSV scans pass them as `columns=`, so other fields are never decoded.
  Selectors and joins conservatively keep every column. A plain column
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
every rewrite. Other plan nodes still materialize their output through the eager API.
