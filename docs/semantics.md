# Behavior contract

## Shape and types

A dataframe has ordered, uniquely named, equal-length columns. Column names are
case-sensitive. Empty names are allowed. Its row count is retained by projection,
even when no columns are selected. Explicit constructor height must match the
columns. An unspecified height on a zero-column frame is zero.

Supported dataframe logical types are `int64`, `float64`, `bool`, and `string`.
There is no implicit numeric promotion. Typed extraction raises on a mismatch.
`Column[T]` is more general, but `Series` accepts only the four listed types.

## Nulls and access

Validity is separate from payloads and stored as an LSB-first bitmap. A false
validity bit denotes null, regardless of its payload. Constructors require a
payload slot for every row, including null rows. Null payloads are never evaluated
by numeric kernels. NaN is a valid floating-point value, not a null.

`value(i)` raises on null and on out-of-bounds access. Negative indexing is not
supported. `take` preserves the supplied order, including duplicates; it raises
on invalid indices. `take_or_null` additionally accepts -1 for an absent row and
is used by left joins. Public reads and transformations return owned copies.

## Slicing, inspection, and rows

`slice(offset, length=-1)` returns rows `[offset, offset + length)` clipped to
the frame. A negative offset counts from the end and clips at zero; an offset
past the end returns zero rows. `length=-1` means all remaining rows, and any
other negative length raises. `head(n)` and `tail(n)` default to five rows; a
negative `n` drops `-n` rows from the other end, as in Polars. `limit` is
`head`. `reverse` reverses row order and `clear` keeps the schema with zero rows.
Zero-column frames keep a correctly clipped height through all of these.

`drop` and `rename` validate every name before building output: unknown names
raise, `drop` rejects a name listed twice, and `rename` raises if the result
would contain duplicate names. Renames apply simultaneously, so swapping two
names works. `with_row_index(name="index", offset=0)` prepends an Int64 column
and raises if the name already exists.

`shape()` is `(height, width)`. `null_count()` returns a one-row frame of Int64
counts per column. `row(i)` returns a `List[AnyValue]` in schema order;
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
restores the original rows up to order.

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
  accepts a dataframe or a list of series.

`Series.append` returns a new series and requires matching dtypes; the result
keeps the left name. `Series.full_null(name, dtype, length)` builds an all-null
column. Validity bitmaps are appended bytewise, with a shifted merge when the
destination length is not a multiple of eight, so concatenation is linear in the
output size (`pixi run bench-concat`).

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
name; the other operand may also be a scalar expression such as `lit(...)`.
The cost is a small constant for building the frame and binding the
expression, plus the usual copies.

## Filtering, arithmetic, and grouping

Filtering, arithmetic, reductions, and grouping go through expressions; see
[expressions](expressions.md) for their full contract. `filter` keeps only
rows whose predicate is true (false and null are dropped) and preserves input
order. Sums skip nulls, return zero for empty or all-null input unless
`min_count` is set, accumulate Int64 exactly in 128 bits, and raise only when
the final result overflows. Grouped output order is unspecified unless
`maintain_order=True`.

## Sorting

`sort(by)` accepts one name or a list of names and orders rows
lexicographically by those columns, ascending by default. `descending` and
`nulls_last` are either one Bool for every key or, passed together as
keywords, one list entry per key: `sort(["a", "b"], descending=[False, True],
nulls_last=[True, False])`. An empty key list or a length mismatch raises.

Sorting is stable: rows with equal keys keep input order, in either direction.
Null placement is independent of direction and defaults to last. NaNs follow
all non-null non-NaN numeric values in either direction; nulls remain first or
last according to the option. `-0.0` and `0.0` are equal keys. Strings use
Mojo's byte-lexicographic comparison, without locale collation. Boolean false
sorts before true in ascending order.

Each key column is converted once into dense integer ranks that already encode
direction, NaN, and null placement; a bottom-up mergesort then compares rank
tuples. `arg_sort` returns the row order. `top_k(k, by)` returns exactly the
first `k` rows of `sort(by, descending=True)` with nulls last, and `bottom_k`
the first `k` rows of `sort(by)`; both select with a bounded heap in
O(n log k) instead of sorting every row. `pixi run bench-sort` compares the
rank-based sort with the previous per-comparison comparator (kept as
`Series._argsort_reference` for tests).

## Joins

`join(right, on, how="inner", suffix="_right", coalesce=True)` takes one key
name or a list; `join(right, left_on=[...], right_on=[...], ...)` pairs keys
with different names. Keys may be any dtype, but each pair must have the same
dtype, key lists must be nonempty and equally long, and a left key may appear
once. Null keys never match, including another null. Empty strings are ordinary
keys. Float64 keys match structurally: every NaN matches every NaN, and `-0.0`
matches `0.0`. Duplicate keys emit every matching pair.

| `how` | Rows | Row order |
|---|---|---|
| `inner` | matching pairs | left order; each left row's matches in right order |
| `left` | every left row; unmatched right side null | as inner |
| `right` | every right row; unmatched left side null | right order; each right row's matches in left order |
| `full` | left join plus unmatched right rows | left join order, then unmatched right rows in right order |
| `semi` | left rows with at least one match, once each | left order |
| `anti` | left rows with no match (including null keys) | left order |
| `cross` | every pair, via `join(right, how="cross")` | left-major |

Output columns are the left columns, then the right non-key columns in right
schema order. Right names that collide with left names gain `suffix`; any
remaining collision raises before any work is done. Key columns keep their left
names and positions. For `right` and `full` joins their values come from
whichever side is present, so a right-only row still shows its key. With
`how="full", coalesce=False`, left keys are null for right-only rows and the
right key columns are kept as ordinary right columns. `semi` and `anti` return
only left columns. Empty inputs keep the joined schema; `cross` checks that the
output row count does not overflow.

Both sides' keys are encoded together by the shared row-key layer
(`dataframe/hashing.mojo`) and right rows are bucketed by dense key id, so the
probe does no dictionary lookup. `pixi run bench-join` compares side sizes and
key skew. Unsupported join modes or key dtype mismatches raise explicitly.

## Ownership and errors

There are no public in-place mutations, shared mutable views, or implicit index
alignment. `with_column` replaces a matching name in position or appends a new
name, and requires matching height. An initially empty zero-row frame cannot
acquire nonzero height via `with_column`; construct with columns or explicit height.

Invalid shape, names, indices, masks, dtype requests, join modes, and integer
sum overflow raise Mojo `Error`. Allocation failure behavior follows Mojo's
standard containers. Underscored storage fields are implementation details;
mutating them directly is outside this contract.
