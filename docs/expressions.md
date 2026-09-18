# Expression semantics and CPU execution

The API draws on Narwhals' separation of expressions, eager dataframes, and
series, and on its explicit broadcasting rules. This is a native implementation,
not a Narwhals adapter or a claim of complete behavioral compatibility.

References:

- https://narwhals-dev.github.io/narwhals/how_it_works/
- https://narwhals-dev.github.io/narwhals/concepts/order_dependence/
- https://narwhals-dev.github.io/narwhals/api-reference/expr/#narwhals.expr.Expr.sum

## Binding and shape

`col(name)` refers to a column; `lit(value)` holds an Int64, Float64, Bool, or
String scalar. `Expr` owns a flat topological node list. Building an expression
never reads data. `alias` changes the output name, not column resolution.
Without an alias, binary operations retain the left expression's output name;
literals are named `literal`, and reductions retain their input expression name.

The binder resolves names to column indices, validates operand dtypes, records
scalar/row-valued/aggregate shape, and tracks aggregate dependencies. Every
expression in an operation is bound before any expression kernel executes.
Duplicate output names, unknown columns, invalid types, negative `min_count`,
and nested aggregates raise even on empty dataframes.

Initial arithmetic is `+`, `-`, and `*` for matching Int64 or Float64 operands.
Greater-than requires matching numeric operands. `.eq()` supports all four
matching dtypes. There is no implicit type promotion or index alignment.
Nulls propagate through binary operations; NaN is distinct from null.
Int64 elementwise arithmetic checks overflow per valid element.

`select(expr)` returns one row for a scalar/aggregate expression or input height
for a row-valued expression. `select_exprs` broadcasts scalar/aggregate results
when any expression is row-valued. An empty expression list preserves height and
returns zero columns. On zero-row inputs, mixing row-valued and scalar expressions
produces zero rows; a scalar-only selection still produces one row.

`with_columns` always retains input height and broadcasts scalar results. All
siblings see the original input; use a second call to reference a newly created
column. `filter` requires Boolean output, broadcasts scalar predicates, and drops
null predicates. `batch_size` must be positive and defaults to 1024.

## Reduction rules that allow parallelism

`sum(min_count=0)` ignores nulls and returns zero when empty/all-null.
`sum(min_count=1)` returns null when no values are valid. Larger thresholds are
supported. A result below its threshold is null, with no final integer narrowing.
`count()` returns the number of non-null rows as Int64, including valid NaNs.
Aggregate inputs must be row-valued and contain no aggregates. Arithmetic on
aggregate outputs is supported, as is broadcasting aggregates against ordinary
columns outside grouping.

Integer sums use signed 128-bit totals plus valid counts. Every possible Int64
column with length representable by Int fits this accumulator. Partial states
for disjoint partitions can therefore merge in any order, and overflow is checked
only when producing a valid final Int64 result. For example, `[MAX, 1, -1]`
succeeds. Intermediate elementwise operations still check their own overflow.

Floating sums use Float64 totals plus counts. Their contract permits reassociation,
including across batches and partitions; results are compared with appropriate
tolerances, not a promise of bitwise reproducibility. SIMD and parallel reduction
scheduling can evolve without changing the public expression API. NaNs and
infinities retain IEEE behavior, subject to reassociation.

The original column sum functions and `group_by_sum` preserve their earlier
contracts: null for empty/all-null input and input-order prefix overflow checks.
They are reference/legacy APIs, not implementations of expression reductions.

## Grouping and ordering

`group_by` currently accepts one String key. Null keys form one group.
`agg` accepts aggregate-shaped expressions, including arithmetic on reductions;
bare columns, literals alone, and mixed row/aggregate outputs are rejected.
Output names must be unique and must not collide with the key.
An empty `agg([])` returns the distinct grouping keys. Empty input yields zero
groups with the expected output schema.

Group output order is unspecified by default. `maintain_order=True` requests
first-occurrence ordering. The current serial implementation produces first-seen
order in either case; only the explicit option guarantees it. This leaves future
parallel hash grouping free to choose its output layout. Maintaining order later
requires merging each group's earliest source position and ordering groups by it.

The eager GroupBy object currently owns a copy of its input snapshot. It does not
copy a dataframe per group. Immutable shared buffers or borrowed views can remove
that snapshot copy in a future storage revision.

## Execution architecture and present limits

- IR and schema binding are independent of CPU kernels and thread scheduling.
- Kernels dispatch operation and dtype outside row loops.
- Float64 arithmetic/comparisons use explicit four-lane SIMD, with masked null
  payloads and tail handling. Tests compare widths 1, 4, and 8 across batch sizes.
- Node intermediates are bounded by batch size. Full results, group mappings,
  and per-group aggregate states are materialized. Memory is approximately
  O(nodes * batch_size + aggregates * groups + input + output).
- A reduction scans its expression input in batches. Global reduction results
  can be broadcast in a subsequent row pass. Grouped reductions use group IDs
  and accumulator arrays rather than per-group frames.
- Reduction states have explicit `add`/`merge` operations. Workers will need
  private states or disjoint output ranges; shared accumulator mutation is not
  safe. Bitmap writes also require byte-aligned partitions or private masks
  followed by a merge; disjoint rows can still share a validity byte. No worker
  threads are launched by this version.

The current evaluator still allocates/copies intermediate batch columns, keeps
node outputs until the batch ends, and scans separately for each aggregate.
SIMD loads/stores are assembled by lane rather than a zero-copy buffer view.
There is no kernel fusion, shared-expression elimination, lazy relational planner,
parallel scheduler, or measured end-to-end speedup claim yet. Those are execution
improvements to pursue with benchmarks, not reasons to change expression semantics.
