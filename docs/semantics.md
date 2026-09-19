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

Sorting is stable, on one column, ascending by default. Equal keys preserve input
order in either direction. Null placement is independent of direction and defaults
to last. NaNs follow all non-null non-NaN numeric values in either direction;
nulls remain first or last according to the explicit option. Strings use Mojo's
ordinary lexicographic comparison, without locale collation. Boolean false sorts
before true in ascending order. The implementation is bottom-up mergesort.

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
