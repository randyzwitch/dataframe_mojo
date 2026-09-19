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

## Filtering and arithmetic

Filter masks must match the frame's height. Only valid true entries retain rows;
false and null entries are dropped. Retained rows preserve input order.
Comparisons and multiplication propagate nulls. NaN comparisons follow IEEE
behavior (NaN is never greater than a threshold).

Sums skip nulls. Empty/all-null inputs return an empty `Optional`, or a null
aggregate cell for a group. Zero is a valid sum. Int64 sums preserve their dtype
and raise before each overflowing addition. Consequently an intermediate overflow
raises even if later cancellation would make the final mathematical result fit.
Float64 sums accumulate left to right and retain IEEE NaN/infinity behavior;
there is no compensated summation or cross-platform bitwise guarantee.

## Grouping

`group_by_sum` supports one String key and one Int64/Float64 value. Groups appear
in order of first occurrence. Null keys form one group, distinct from every
string, including the empty string. Groups with no non-null values have null
sums. Empty input returns zero rows with both output columns and their dtypes.
The aggregate output name must differ from the key name.

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

Only inner and left joins on one shared String column are supported. Null keys
never match, including another null. Empty strings are ordinary matching keys.
Duplicate keys emit all matching pairs. Output traverses left rows in order;
for each left row it traverses right matches in right input order.

The shared key appears once, using the left column. Other left columns come first,
followed by right non-key columns in schema order. Right column names that overlap
left names gain the suffix (default `_right`). Any remaining output collision
raises. Unmatched left rows have null right-side values. Empty inputs retain the
joined output schema. Unsupported join modes or key types raise explicitly.

## Ownership and errors

There are no public in-place mutations, shared mutable views, or implicit index
alignment. `with_column` replaces a matching name in position or appends a new
name, and requires matching height. An initially empty zero-row frame cannot
acquire nonzero height via `with_column`; construct with columns or explicit height.

Invalid shape, names, indices, masks, dtype requests, join modes, and integer
sum overflow raise Mojo `Error`. Allocation failure behavior follows Mojo's
standard containers. Underscored storage fields are implementation details;
mutating them directly is outside this contract.

## Expression API

The reduction and grouping rules above describe the original column-kernel and
`group_by_sum` APIs. The new expression API has a separate, deliberately more
parallel-friendly contract: zero-for-empty sums with `min_count`, wide integer
accumulation with final overflow checking, floating-point reassociation, and
explicit `maintain_order` for grouped output. See [expressions](expressions.md).
