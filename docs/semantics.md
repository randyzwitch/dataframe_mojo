# Behavior contract

## Shape and types

A dataframe has ordered, uniquely named, equal-length columns. Column names are
case-sensitive. Empty names are allowed. Its row count is retained by projection,
even when no columns are selected. Explicit constructor height must match the
columns. An unspecified height on a zero-column frame is zero.

Supported dataframe logical types include signed and unsigned integer widths, Float32 and Float64, Bool, String, temporal and nested types, and `DataType.decimal(precision, scale)` for decimal128 fixed-point values. `Series.dtype()`, `DataFrame.dtypes()`, `Field.dtype`,
`AnyValue.dtype()`, and `CsvField.dtype` return `DataType` values, which compare
with `==` and print their canonical names (`int64`, `float64`, `bool`,
`string`); `DataType.parse(name)` is the inverse, and `is_numeric`,
`is_integer`, `is_float`, `is_signed`, and `bit_width` describe a type. APIs
that name a dtype (`cast("int64")`, `by_dtype([...])`, `null("int64")`,
`schema_overrides`) accept either form.
There is no implicit numeric promotion. Typed extraction raises on a mismatch.
`Column[T]` is more general, but `Series` accepts only the four listed types.

## Nulls and access

`DataType.NULL` is the logical type of an entirely null inferred JSON field.
It carries no valid values, including inside List and Struct schemas; Arrow
interchange uses the Arrow Null format. An explicit schema gives an all-null
field another logical dtype when that type is known in advance.

Validity is separate from payloads and stored as an LSB-first bitmap. A false
validity bit denotes null, regardless of its payload. Constructors require a
payload slot for every row, including null rows. Null payloads are never evaluated
by numeric kernels. NaN is a valid floating-point value, not a null.

`value(i)` raises on null and on out-of-bounds access. Negative indexing is not
supported. `take` preserves the supplied order, including duplicates; it raises
on invalid indices. `take_or_null` additionally accepts -1 for an absent row and
is used by left joins. Results are immutable: slices, projections, and extracted
columns may share buffers with their source, which is unobservable because no
public operation mutates a column in place.

## Slicing, inspection, and rows

`slice(offset, length=-1)` returns rows `[offset, offset + length)` clipped to
the frame. A negative offset counts from the end and clips at zero; an offset
past the end returns zero rows. `length=-1` means all remaining rows, and any
other negative length raises. `head(n)` and `tail(n)` default to five rows; a
negative `n` drops `-n` rows from the other end. `limit` is
`head`. `reverse` reverses row order and `clear(n=0)` keeps the schema with
`n` rows of nulls (zero rows by default). `gather(indices, null_on_oob=False)`
returns the rows at `indices` in that order: a negative
index counts from the end, and an index out of range raises, or with
`null_on_oob=True` gives a row of nulls.
`shift(n=1, fill_value=None)` moves every column's values `n` rows later
(earlier for a negative `n`): height and schema
stay, and the `min(|n|, height)` vacated rows are null, or `fill_value` cast to
each column's type. `fill_value` must be a scalar (a literal or a reduction
such as `col("a").max()`); the input's own nulls are not filled. The lazy
`tail`, `reverse`, `shift` and `with_row_index` give the same results.
Zero-column frames keep a correctly clipped height through all of these.

`drop` and `rename` validate every name before building output: unknown names
raise, `drop` rejects a name listed twice, and `rename` raises if the result
would contain duplicate names. Renames apply simultaneously, so swapping two
names works; `rename(mapping, strict=False)` ignores names the frame lacks. `with_row_index(name="index", offset=0)` prepends an Int64 column
and raises if the name already exists.

`shape()` is `(height, width)`. `null_count()` returns a one-row frame of Int64
counts per column.

`describe(percentiles=[0.25, 0.5, 0.75], interpolation="nearest")` returns a `statistic` column with rows `count`, `null_count`, `mean`,
`std`, `min`, one row per percentile in ascending order (labelled `25%` and so
on; an empty list gives none), and `max`, then one column per input column.
Numeric and Bool columns give Float64; Bool `mean` is the share of true
values and has no `std` or percentiles. `min` and `max` skip NaN unless every
valid value is NaN, as the `min` and `max` expressions do, while `mean` and
`std` propagate it. String columns give
String counts, `min` and `max`. Temporal columns give String counts, `mean`
(a datetime for a Date column), `min`, percentiles and `max`, formatted as a
cast to String formats them, retaining the dtype's time unit and precision.
Nested columns give only the counts.

`sample(n=None, fraction=None, with_replacement=False, shuffle=False,
seed=None)` on DataFrame and Series draws `n` rows, or `floor(fraction *
height)`, or one row when neither is given; passing both raises. Without
replacement no row repeats and a sample larger than the frame raises. Rows
keep their original order unless `shuffle=True`, which also randomizes the
order. The generator is
SplitMix64 with unbiased bounded draws, so a seed gives the same rows on every
platform and release. Without a
seed it is seeded from the clock. `row(i)` returns a `List[AnyValue]` in schema order;
`AnyValue` carries a dtype tag and validity, and its typed accessors raise on a
dtype mismatch or null. `item()` requires exactly one cell; `item(row, column)`
reads one cell. `rows()` materializes every row and is meant for small frames.

`equals(other, null_equal=True)` compares names, dtypes, column order, height,
and cells. Floats compare structurally: NaN equals NaN and `-0.0` equals `0.0`.
Null payloads never participate. With `null_equal=False`, any null in either
frame makes them unequal. `DataFrame` deliberately has no `==` operator.

## Display

`print(frame)` and `String(frame)` render a box table with the shape, column
names, short dtypes (`i64`, `f64`, `bool`, `str`), and at most 10 rows and 12
columns. `to_string(max_rows=10, max_columns=12, max_string_length=32)` sets the
limits; a negative limit means unlimited. Elided rows and columns show `…`, with
the extra row or column at the front when the limit is odd. Only displayed cells
are formatted, so rendering a large frame costs O(displayed cells).

Nulls render as `null`. Strings render unquoted unless they are empty, equal to
`null`, begin or end with a space or tab, or contain control characters; those
are double-quoted with `\\`, `"`, `\n`, `\r`, and `\t` escaped. Strings longer
than `max_string_length` code points are cut and end in `…`. Floats use Mojo's
shortest round-trippable form (`1.0`, `1e+300`, `nan`, `inf`, `-0.0`).

Column widths count Unicode code points. Combining characters and East Asian
wide characters therefore misalign in a terminal; display width is not computed.

`Series` renders its shape, name, dtype, and values one per line with the same
`max_rows` rule. `glimpse(max_width=100)` prints one line per column with as
many leading values as fit in `max_width` code points, then `…`. Typed `Column`
values are displayed by wrapping them in a `Series`.

## Duplicates and nulls

`unique(subset=[], keep="any", maintain_order=False)` drops duplicate rows,
comparing the subset columns (every column by default) with the grouping key
rules: nulls equal nulls, every NaN is one value, and `-0.0` equals `0.0`.
`keep="any"` and `"first"` keep each key's first row, `"last"` its last row,
and `"none"` drops every row whose key repeats. Output order is unspecified
unless `maintain_order=True`, which keeps surviving rows in input order (the
current implementation always does). `n_unique(subset)` counts distinct keys;
`is_duplicated(subset)` and `is_unique(subset)` return Bool series. A subset
naming a column twice raises, and a zero-column frame cannot be deduplicated.

`drop_nulls(subset=[])` keeps rows with no null in the subset (default every
column). `fill_null(value, subset=[])` replaces nulls with a scalar expression:
without a subset only columns of the value's dtype change; every listed subset
column must match it. NaN is a value and is never filled.

## Reshaping

`unpivot(on=[], index=[], variable_name="variable", value_name="value")` turns
columns into rows: one output row per input row and `on` column, ordered by
`on` column and then input row. `on` defaults to every non-index column, and all
`on` columns must share one dtype because there is no promotion. The variable
column holds the source column names as strings.

`pivot(on, index=[...], values=..., aggregate_function="", sort_columns=False)`
turns rows into columns: one row per distinct index key (first-occurrence
order) and one column per distinct `on` value (first-occurrence order, or sort
order with `sort_columns=True`). Without `aggregate_function`, a cell with
several rows raises; otherwise use `first`, `last`, `sum`, `mean`, `min`,
`max`, `count`, `len`, or `median`, computed with the ordinary grouped
reductions. Missing cells are null. New column names are the `on` values' text
(`null` for a null key) and must not collide with index names. An empty index
list yields a single row. `pivot` followed by `unpivot` on the same names
restores the original rows up to order. The lazy `pivot` takes the new
columns' names up front (`on_columns`) and gives the same cells. A missing
cell is null for every aggregation function; within existing cells, `count`
skips null values and `len` counts rows.

## Concatenation

`concat(frames, how="vertical")` raises on an empty list because no schema is
known. Every input is validated before any column is built, and errors name the
frame position and column.

- `vertical` requires identical names, order, and dtypes. Rows appear frame by
  frame in input order. Zero-column frames sum their heights. `vstack` is the
  two-frame form.
- `diagonal` unions columns by name in first-seen order. A name shared by
  several frames must have one dtype; there is no promotion. Frames lacking a
  column contribute nulls.
- `horizontal` requires equal heights and globally unique names. `hstack`
  accepts a dataframe or a list of series; differing heights raise rather
  than inserting padding rows.

`concat` also takes a list of LazyFrames and gives the same results and errors
as one plan (see [lazy queries](lazy.md)).

`Series.append` returns a new series and requires matching dtypes; the result
keeps the left name. `Series.full_null(name, dtype, length)` builds an all-null
column. Validity bitmaps are appended bytewise, with a shifted merge when the
destination length is not a multiple of eight, so concatenation is linear in the
output size.

## Series operations

`Series` has operators (`+ - * / // % **`, unary `-`, `< <= > >=`, `.eq()`,
`.ne()`, `& | ^ ~`), elementwise helpers (`is_null`, `is_not_null`,
`fill_null`, `abs`, `round`), reductions returning an `AnyValue` (`sum`, `mean`,
`min`, `max`, `median`, `quantile`, `std`, `var`, `first`, `last`, `any`,
`all`) or an `Int` (`count`, `n_unique`), `sort`, `unique`, `value_counts`,
`head`, `tail`, `to_values`, `apply(expr)`, and `series[i]` (negative indices
count from the end). `frame["name"]` returns a column as a Series.

Every one of these evaluates the matching expression over a one-column frame,
so it shares the expression kernels and contracts exactly: typed literals, no
promotion, the same null, NaN, and overflow rules. A binary operation between
two series pairs rows by position, requires equal lengths, and keeps the left
name; the other operand may also be a scalar expression such as `lit(...)` or
a bare number.
The cost is a small constant for building the frame and binding the
expression, plus the usual copies.

## Filtering, arithmetic, and grouping

Filtering, arithmetic, reductions, and grouping go through expressions; see
[expressions](expressions.md) for their full contract. `filter` keeps only
rows whose predicate is true (false and null are dropped) and preserves input
order. Sums skip nulls, return zero for empty or all-null input unless
`min_count` is set, accumulate Int64 exactly in 128 bits, and raise only when
the final result overflows. Grouped output order is unspecified unless
`maintain_order=True`. Besides `agg`, a `GroupBy` has `len(name="len")` and
`sum`, `mean`, `min`, `max`, `median`, `first`, `last`, `n_unique` and
`quantile(q, interpolation="nearest")`, each the reduction of every column
that is not a key; a key computed
from an expression keeps its output name, so `group_by([col("a") % 2]).sum()`
does not sum `a`.
A categorical or dictionary-coded key groups by its label: codes that name the
same label are one group, whichever route runs (eager, lazy with or without
`maintain_order`, Metal), so a category never
repeats a string. Arrow and Parquet dictionaries may repeat a label and are
kept as given; only grouping merges the codes, and the group's key shows the
first code of its label. `over()` partitions and row numbering for `unique`
still compare codes.
Categorical expressions operate on decoded String values, so expression
reductions such as first/last and min/max return String. A bare column selection
and direct table row operations retain categorical metadata.

## Sorting

`sort(by)` accepts one name or a list of names and orders rows
lexicographically by those columns, ascending by default. `descending` and
`nulls_last` are either one Bool for every key or, passed together as
keywords, one list entry per key: `sort(["a", "b"], descending=[False, True],
nulls_last=[True, False])`. An empty key list or a length mismatch raises.

Rows with equal keys come out in no particular order unless the sort is
given `maintain_order=True`, which keeps their input order in either
direction. Leaving ties free lets the planner reorder joins below a sort.
Null placement is independent of direction and defaults to last. NaN is the
largest value: it follows the numbers ascending and precedes them
descending (every NaN is one value); nulls remain first or
last according to the option. `-0.0` and `0.0` are equal keys. Strings use
Mojo's byte-lexicographic comparison, without locale collation. Boolean false
sorts before true in ascending order.

Each key column is converted once into dense integer ranks that already encode
direction, NaN, and null placement; a bottom-up mergesort then compares rank
tuples. `arg_sort` returns the row order. `top_k(k, by)` returns exactly the
first `k` rows of `sort(by, descending=True)` with nulls last, and `bottom_k`
the first `k` rows of `sort(by)`; `reverse=True` flips the direction (nulls
stay last). Both select with a bounded heap in O(n log k)
instead of sorting every row.

## Joins

`join(right, on, how="inner", suffix="_right", coalesce=True)` takes one key
name or a list; `join(right, left_on=[...], right_on=[...], ...)` pairs keys
with different names. Keys may be any dtype, but each pair must have the same
dtype, key lists must be nonempty and equally long, and a left key may appear
once. Null keys never match, including another null. Empty strings are ordinary
keys. Float64 keys match structurally: every NaN matches every NaN, and `-0.0`
matches `0.0`. Duplicate keys emit every matching pair.

| `how` | Rows |
|---|---|
| `inner` | matching pairs |
| `left` | every left row; unmatched right side null |
| `right` | every right row; unmatched left side null |
| `full` | left join plus unmatched right rows |
| `semi` | left rows with at least one match, once each |
| `anti` | left rows with no match (including null keys) |
| `cross` | every pair, via `join(right, how="cross")` |

A join's output row order is unspecified, eager or lazy: the engine may build on either side
and stream the other. Sort the result when an order matters. `join_asof` is the
exception: it keeps the left input's row order, and sorts its inputs itself
when they are not already in key order.

Output columns are the left columns, then the right non-key columns in right
schema order. Right names that collide with left names gain `suffix`; any
remaining collision raises before any work is done. Key columns keep their left
names and positions. A coalesced `right` join is the mirror image: the left's non-key columns, then every right column in right schema
order, its keys included under their right names (and their values, including
the sign of zero); right names that collide with the left's non-key names gain
`suffix`. For a `full` join the keys' values come from whichever side is
present, so a right-only row still shows its key; matched rows use the left
keys. With
`how="full", coalesce=False`, left keys are null for right-only rows and the
right key columns are kept as ordinary right columns. `semi` and `anti` return
only left columns. Empty inputs keep the joined schema; `cross` checks that the
output row count does not overflow.

Keys may also be expressions: `join(right, left_on=[col("email").str().to_lowercase()],
right_on=[col("email").str().to_lowercase()])`. Each key is computed once per
side, as a hidden column the output drops, so the output holds the inputs'
columns and not the computed keys. When any key is more than a plain column,
both sides' columns are all kept (the right ones suffixed where names collide),
as with `coalesce=False`; keys that are all plain columns join as key names do.

`validate="1:m" | "m:1" | "1:1"` checks that join keys are unique where the
join expects them to be: the left input's for `1:m`, the right input's for
`m:1`, both for `1:1`. The default `m:m` checks nothing. A repeated key raises
`join keys did not fulfill m:1 validation: the right input has key (k=1)
more than once`, naming one repeated key. Rows with a null key never match,
so they never count as repeats. It applies to inner, left, right and full
joins; semi and anti joins reject it. The side the join builds its hash table
on is checked by that table as its rows are inserted; the other side is
checked with a table of its keys, so checking a left input costs one more
hash build (and runs that join on its whole inputs rather than streaming the
left side). A validated join keeps its inputs: the planner does not reorder
it with other joins.

`join_where(right, [predicates], suffix="_right")` is an inner join on any
predicates over both inputs: every pair of a left and a right row for which
all predicates hold, as a filtered cross join would give. Predicates name a
right column that the left side also has by its suffixed name, as the output
does. The output is every left column, then every right column, in an
unspecified row order. Planning picks the join from the predicates:
equalities of a left and a right column of one type run as a hash join (the
rest filter its output); otherwise up to two inequalities (`<`, `<=`, `>`,
`>=`) of integer, float or temporal operands (one type on both sides,
integers of any width but UInt64) run as an inequality join that finds matches from
sorted keys. Two comparisons bounding one key of a side (`ts >= start` and
`ts < end`) make a range join: that side's keys are sorted and every row of
the other side finds its run of matches by binary search. Two comparisons
with different keys on both sides make a two-key join: both sides are sorted
by the first key and cut into blocks, and each pair of blocks whose ranges
can meet is scanned in the order of the second key. Operands may be
expressions of one side (computed once, as hidden columns). Other predicates
filter the joined rows; with no usable comparison the cross join is filtered.
Null and NaN keys match nothing, as in the comparisons themselves.

 by the shared row-key layer
(`dataframe/kernels/hashing.mojo`) and right rows are bucketed by dense key id, so the
probe does no dictionary lookup. Unsupported join modes or key dtype mismatches
raise explicitly.

## Ownership and errors

There are no public in-place mutations, shared mutable views, or implicit index
alignment. `with_column` replaces a matching name in position or appends a new
name, and requires matching height. An initially empty zero-row frame cannot
acquire nonzero height via `with_column`; construct with columns or explicit height.

Invalid shape, names, indices, masks, dtype requests, join modes, and integer
sum overflow raise Mojo `Error`. Underscored storage fields are implementation
details; mutating them directly is outside this contract.

## Memory limit

The engine counts the large memory it holds (column buffers, hash table and
aggregate blocks, and freed blocks it keeps for reuse) against one
process-wide limit, and also compares the process's resident memory with it
(read at most once a millisecond), which covers what the count misses. Work
that would go over it raises an `Error` whose message starts with
`out of memory`, instead of the operating system killing the process. What
the failed operation held is freed, so the program can catch the error and
go on.

- Default: 80% of the memory available to the process, which is physical
  memory or, when lower, the memory limit of the control group (container or
  systemd scope) the process runs in.
- `DATAFRAME_MEMORY_LIMIT` sets it at startup: a size such as `8GB` or
  `512MiB`, a share such as `50%`, or `none`.
- `set_memory_limit(8 << 30)` or `set_memory_limit("8GB")` sets it at any
  time; `memory_limit()` and `memory_usage()` read it and the bytes held.
- Tables the program holds count toward the limit, so a limit below them
  makes the next operation fail.
- Nothing spills to disk: an operation that needs more than the limit
  fails.
- The limit is checked between pieces of work (each morsel, scheduled task
  and round of worker jobs) and before allocations of known size, so memory
  can grow past it by what one piece of work allocates. The default's
  headroom covers that; a limit set close to a container's own limit can
  still let the container kill the process.

## Threads

Reductions, row-wise expressions, and filters run as one task per worker
thread, each task at least one vector of 2,048 rows, so an input of two
vectors or more runs on worker threads (POSIX threads; `DATAFRAME_THREADS`
caps the count, default the physical core count, and `1` disables). Results do not depend on the thread count except for the
last bits of floating-point sums, means, and variances, which reassociate.
Integer sums are exact, and first/last, min/max ties, and n_unique match a
single-threaded pass. A worker's error (such as an overflow) is raised on the
calling thread.

