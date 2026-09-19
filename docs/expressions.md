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

Operands must have matching dtypes; there is no implicit promotion or index
alignment. Nulls propagate through every operator below; NaN is distinct from
null. Int64 arithmetic checks overflow per valid element and raises.

| Operation | Operands | Result | Notes |
|---|---|---|---|
| `+`, `-`, `*` | Int64, Float64 | same | checked for Int64 |
| `/` | Int64, Float64 | Float64 | the one operator where Int64 input yields Float64; IEEE division by zero |
| `//` | Int64, Float64 | same | floor division; Int64 by zero is null; `Int64.MIN // -1` raises |
| `%` | Int64, Float64 | same | remainder takes the divisor's sign; Int64 modulo zero is null; Float64 `x % 0`, `inf % y`, and NaN operands give NaN |
| `**`, `.pow()` | Int64, Float64 | same | Int64 is checked and raises on a negative exponent |
| `<`, `<=`, `>`, `>=` | all four | Bool | IEEE for floats (NaN compares false); strings byte-lexicographic, as in sort; `false < true` |
| `.eq()`, `.ne()` | all four | Bool | IEEE: NaN `.ne()` NaN is true |
| unary `-`, `.abs()` | Int64, Float64 | same | `Int64.MIN` raises |
| `.sqrt()`, `.exp()`, `.log()` | Int64, Float64 | Float64 | natural log; IEEE results for negative inputs and zero |
| `.floor()`, `.ceil()`, `.round(decimals=0)` | Int64, Float64 | same | identity for Int64; `round` is half away from zero and accepts negative decimals |
| `.clip(lower, upper)`, `.clip_min`, `.clip_max` | Int64, Float64 | same | NaN inputs and NaN bounds leave the value unchanged |

Float64 `+ - * /` and all Float64 comparisons use explicit SIMD kernels; the
other Float64 operations run per lane. `sqrt`, `exp`, `log`, and `pow` use
Mojo's `std.math`, whose results can differ from the correctly rounded value by
about 1e-12 relative; tests compare them with a tolerance. Binder errors name
the operator and dtypes, for example `/ requires matching dtypes, found int64
and float64; use typed literals`.

### Boolean logic and nulls

`&`, `|`, `^`, and `~` (also `and_`, `or_`, `xor`, `not_`) require Bool
operands and use three-valued Kleene logic: `false & null` is false,
`true | null` is true, and otherwise a null operand yields null. `^` and `~`
propagate nulls. Mojo cannot overload `and`/`or`/`not`, so use the operators.
`filter` keeps only true rows, so Kleene nulls are dropped.

`null(dtype)` is a typed null literal. `is_null()`/`is_not_null()` accept any
dtype and never return null. `is_nan`, `is_not_nan`, `is_finite`, and
`is_infinite` require Float64 and propagate nulls.

`fill_null(value)` replaces nulls with a matching-dtype value or column;
`fill_nan(value)` replaces valid NaNs in Float64 input. `coalesce([a, b, ...])`
takes the first non-null value per row. `is_in([values])` is true when the input
equals any candidate; a null candidate never matches, a null input stays null,
and an empty list yields false. `is_between(lower, upper, closed="both")`
accepts `both`, `left`, `right`, or `none` and combines the two comparisons with
Kleene AND, so a null bound yields null unless the other side is false.

### Strings

`col("s").str()` returns a namespace of String operations; Mojo has no
attribute-style namespaces, so later namespaces (`dt()` and others) follow the
same method convention. Inputs must be String and nulls propagate.

| Method | Result | Notes |
|---|---|---|
| `len_chars()`, `len_bytes()` | Int64 | Unicode code points / UTF-8 bytes |
| `to_uppercase()`, `to_lowercase()` | String | Mojo's Unicode case mapping, no locale rules |
| `strip_chars(chars="")`, `strip_chars_start`, `strip_chars_end` | String | empty `chars` means ASCII whitespace; otherwise any listed code point |
| `starts_with(p)`, `ends_with(p)`, `contains(p)` | Bool | literal text; no regular expressions |
| `replace(p, v)`, `replace_all(p, v)` | String | literal; an empty pattern inserts once for `replace` and is a no-op for `replace_all` |
| `slice(offset, length=-1)`, `head(n)`, `tail(n)` | String | by code point; a negative offset counts from the end; out-of-range parts clip |
| `reverse()` | String | by code point |
| `pad_start(w, c=" ")`, `pad_end(w, c)`, `zfill(w)` | String | pad to `w` code points with one fill character; `zfill` keeps a leading sign first |

`concat_str([exprs], separator="")` joins String expressions row-wise and is
null when any input is null. Character operations work on code points, not
grapheme clusters: `"e\u0301"` has length 2 and reverses its combining mark.

### Conditionals

`when(p).then(a).when(q).then(b).otherwise(c)` picks, per row, the value of the
first branch whose predicate is true. Predicates must be Bool; a null predicate
is not true and falls through to the next branch. Without `otherwise`, unmatched
rows are null. A chain without `otherwise` converts implicitly to `Expr`, and
`.alias()` works at either stage. Every branch must have the same dtype: there
is no promotion, so use typed literals or `null(dtype)`. The result is named
after the first `then` value. Scalars and aggregates broadcast as elsewhere; an
all-scalar chain is scalar, and conditions on aggregates work inside `agg`.

Branches are evaluated under a row mask. A kernel that can raise (checked Int64
arithmetic, negation, `abs`, `//`, `pow`) only evaluates rows whose result can
be selected, so `when(x <= 0).then(lit(MAX) + x).otherwise(x)` does not raise
for positive `x`, while an overflow in a selected row still raises. Masks narrow
through nested conditionals. A scalar branch is evaluated only if some row
selects it. Aggregates inside a branch (for example `col("x").sum()`) are still
computed over the whole input before the conditional runs, so their own errors
are not masked.

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
`null_count()` returns the number of nulls. `any()` and `all()` require Bool
input. With the default `ignore_nulls=True`, nulls are skipped: an empty or
all-null input gives `any` false and `all` true. With `ignore_nulls=False` they
follow Kleene logic: `any` is null when nothing is true and some value is null,
and `all` is null when nothing is false and some value is null. Their states
(`LogicState`) merge in any order.
Aggregate inputs must be row-valued and contain no aggregates. Arithmetic on
aggregate outputs is supported, as is broadcasting aggregates against ordinary
columns outside grouping.

| Reduction | Input | Result | Empty or all-null | Mergeable state |
|---|---|---|---|---|
| `sum(min_count=0)` | Int64, Float64 | same | 0, or null below `min_count` | yes |
| `count()`, `null_count()`, `len()` | any | Int64 | 0 | yes |
| `min()`, `max()` | any | same | null | yes |
| `mean()` | Int64, Float64 | Float64 | null | yes (sum + count) |
| `first()`, `last()` | any | same | null | needs partition order |
| `n_unique()` | any | Int64 | 0, or 1 for all-null | yes (distinct set) |
| `var(ddof=1)`, `std(ddof=1)` | Int64, Float64 | Float64 | null when fewer than `ddof + 1` values | yes (Welford/Chan) |
| `median()`, `quantile(q, interpolation)` | Int64, Float64 | Float64 | null | no: keeps every valid value |
| `any()`, `all()` | Bool | Bool | false / true | yes |

`min` and `max` order strings byte-lexicographically and `false < true`. Float
NaN sorts above every number, as in `sort`: `max` is NaN if any valid value is
NaN, and `min` is NaN only when every valid value is. Ties keep the first value.
`mean` of Int64 input divides the exact 128-bit total, so it cannot overflow.
`first` and `last` return the first or last row's value even when it is null.
`n_unique` counts null as one value, treats every NaN as one value, and treats
`-0.0` as equal to `0.0`. `var` and `std` convert Int64 to Float64; infinities
produce NaN. `median` is `quantile(0.5, "linear")`. For `quantile`, `q` must be
in [0, 1] and the position is `q * (n - 1)` over sorted valid values (NaN last);
`linear` interpolates, `lower`/`higher` take the neighbouring value, `midpoint`
averages them, and `nearest` rounds the position half up. Int64 values above
2^53 lose precision in `var`, `std`, `median`, and `quantile`. Every reduction
works globally and inside `group_by(...).agg(...)`, and arithmetic on
reduction outputs (for example `col("x").max() - col("x").min()`) is allowed.

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

`group_by` accepts one column name, a list of names, or a list of key
expressions, and keys may have any dtype. Expression keys are evaluated once and
named by their output names; they may be row-valued or scalar (scalars form one
group) but not aggregates. Aggregations always see the original columns, even
when a key expression reuses a column's name. Output columns are the keys in the
order given, then the aggregates.

Keys compare exactly per column through a shared row-key layer
(`dataframe/hashing.mojo`): each column is mapped to dense codes and the codes
are combined column by column, so there is no string concatenation and no
reliance on hash uniqueness. A null is a key value of its own column: rows group
together only when their nulls fall in the same key columns. Float64 keys treat
every NaN as one value and `-0.0` as equal to `0.0`. The same layer, with nulls
never matching, is used for joins. `GroupBy.len(name="len")` counts rows per
group. `pixi run bench-group-by` reports cost by key count and cardinality.

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
