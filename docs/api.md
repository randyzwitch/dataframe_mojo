# API reference

Generated from `mojo doc` by `scripts/api_docs.py`; covers every name
exported by `dataframe/__init__.mojo`. Contracts live in the guides:
[semantics](semantics.md), [expressions](expressions.md), [csv](csv.md).

## `all`

Every column, in schema order.

```mojo
def all() -> Expr
```

## `AnyValue`

One nullable cell of a supported dtype.

Only the payload field matching `dtype` is meaningful. Equality is
structural: nulls equal nulls of the same dtype and NaN equals NaN.

- `def __init__[D: DType](out self, value: Scalar[D])`
  A numeric value. Integers are held in an Int64 slot (UInt64 by bit pattern) and floats in a Float64 slot; both are exact.
- `def __init__(out self, value: Bool)`
- `def __init__(out self, value: String)`
- `def __init__(out self, dtype: DataType, valid: Bool, integer: Int64, floating: Float64, boolean: Bool, var string: String)`
- `def __eq__(self, other: Self) -> Bool`
- `def null(dtype: DataType) -> Self`
- `def null(dtype: String) -> Self`
- `def temporal(dtype: DataType, value: Int64) -> Self`
  A date, datetime, duration, or time from its stored Int64.
- `def to_physical(self) -> Int64`
  The stored Int64 of an Int64 or temporal value.
- `def dtype(self) -> DataType`
- `def is_null(self) -> Bool`
- `def numeric[D: DType](self) -> Scalar[D]`
  The value as Scalar[D]; the dtype must match exactly.
- `def int64(self) -> Int64`
- `def float64(self) -> Float64`
- `def int8(self) -> Int8`
- `def int16(self) -> Int16`
- `def int32(self) -> Int32`
- `def uint8(self) -> UInt8`
- `def uint16(self) -> UInt16`
- `def uint32(self) -> UInt32`
- `def uint64(self) -> UInt64`
- `def float32(self) -> Float32`
- `def bool(self) -> Bool`
- `def string(self) -> String`
- `def write_to(self, mut writer: T)`

## `ArrowArray`

`struct ArrowArray` from the C Data Interface (80 bytes).

- `def __init__(out self)`
  A released (empty) struct, ready to be filled by a producer.

## `ArrowSchema`

`struct ArrowSchema` from the C Data Interface (72 bytes).

- `def __init__(out self)`
  A released (empty) struct, ready to be filled by a producer.

## `BoolColumn`

A window onto shared, bit-packed values and validity.

- `def __init__(out self, values: List[Bool])`
- `def __init__(out self, values: List[Bool], valid: List[Bool])`
- `def __init__(out self, column: Column[Bool])`
  Pack a byte-per-value Boolean column.
- `def __init__(out self, *, var values: List[UInt8], var bits: List[UInt8], length: Int)`
  Adopt finished value and validity bitmaps.
- `def __len__(self) -> Int`
- `def is_null(self, index: Int) -> Bool`
- `def value(self, index: Int) -> Bool`
- `def null_count(self) -> Int`
- `def true_count(self) -> Int`
  Rows whose value bit is set (including null rows).
- `def take(self, indices: List[Int]) -> Self`
- `def take_or_null(self, indices: List[Int], fill: Bool) -> Self`
  Gather rows, treating only -1 as a missing row (for outer joins).
- `def slice(self, offset: Int, length: Int) -> Self`
  A zero-copy window sharing this column's buffers.

## `by_dtype`

Columns whose dtype is listed, in schema order.

```mojo
def by_dtype(dtypes: List[String]) -> Expr
```

## `coalesce`

First non-null value per row, left to right; dtypes must match.

```mojo
def coalesce(exprs: List[Expr]) -> Expr
```

## `col`

Select several columns; operations apply to each one.

```mojo
def col(name: String) -> Expr
```

```mojo
def col(names: List[String]) -> Expr
```

## `Column`

A window onto a shared payload buffer and validity bitmap.

Underscored storage is internal. Public operations return new columns;
no mutable buffers or implicit negative indexing are exposed.

- `def __init__(out self, var values: List[T])`
- `def __init__(out self, var values: List[T], valid: List[Bool])`
- `def __len__(self) -> Int`
- `def is_null(self, index: Int) -> Bool`
- `def value(self, index: Int) -> T`
- `def null_count(self) -> Int`
- `def take(self, indices: List[Int]) -> Self`
- `def take_or_null(self, indices: List[Int], fill: T) -> Self`
  Gather rows, treating only -1 as a missing row (for outer joins).
- `def slice(self, offset: Int, length: Int) -> Self`
  A zero-copy window sharing this column's buffers.

## `concat`

Combine frames: 'vertical', 'diagonal', or 'horizontal'.

```mojo
def concat(frames: List[DataFrame], how: String = "vertical") -> DataFrame
```

## `concat_str`

Join String expressions row-wise; any null input makes the row null.

```mojo
def concat_str(exprs: List[Expr], separator: String = "") -> Expr
```

## `CsvField`

One CSV output field. Names and dtypes are always explicit.

Temporal fields take an optional strftime-style format; an empty format
means ISO 8601 (see dataframe/temporal.mojo).

- `def __init__(out self, name: String, dtype: DataType, nullable: Bool = True, format: String = "")`
- `def int64(name: String, nullable: Bool = True) -> Self`
- `def float64(name: String, nullable: Bool = True) -> Self`
- `def bool(name: String, nullable: Bool = True) -> Self`
- `def string(name: String, nullable: Bool = True) -> Self`
- `def date(name: String, format: String = "", nullable: Bool = True) -> Self`
- `def datetime(name: String, unit: String = "us", format: String = "", nullable: Bool = True) -> Self`
- `def time(name: String, format: String = "", nullable: Bool = True) -> Self`

## `CsvOptions`

Validated reader options; see read_csv for their meaning.

- `def validate(self)`

## `CsvSchema`

An ordered, non-empty set of uniquely named CSV fields.

- `def __init__(out self, var fields: List[CsvField])`
- `def __len__(self) -> Int`
- `def of(frame: DataFrame) -> Self`
  A nullable schema matching a frame's names and dtypes.
- `def field(self, index: Int) -> CsvField`

## `DataFrame`

Own equal-length, uniquely named columns; transformations copy storage.

- `def __init__(out self, var columns: List[Series], *, height: Int = Int(-1))`
- `def __getitem__(self, name: String) -> Series`
- `def height(self) -> Int`
- `def write_to(self, mut writer: T)`
- `def to_string(self, *, max_rows: Int = Int(10), max_columns: Int = Int(12), max_string_length: Int = Int(32)) -> String`
  Render a bounded table. Negative limits mean unlimited.
- `def glimpse(self, *, max_width: Int = Int(100), max_string_length: Int = Int(32)) -> String`
  Transposed summary: one line per column with leading values.
- `def width(self) -> Int`
- `def schema(self) -> List[Field]`
- `def shape(self) -> Tuple[Int, Int]`
- `def __len__(self) -> Int`
- `def is_empty(self) -> Bool`
- `def columns(self) -> List[String]`
- `def dtypes(self) -> List[DataType]`
- `def column(self, name: String) -> Series`
  Return an owned copy. Column lookup is linear in the schema width.
- `def lazy(self) -> LazyFrame`
  Start a lazy query over this frame; see LazyFrame.
- `def get_column(self, name: String) -> Series`
- `def null_count(self) -> Self`
  One row holding each column's null count as Int64.
- `def row(self, index: Int) -> List[AnyValue]`
  Return one row as tagged values, in schema order.
- `def rows(self) -> List[List[AnyValue]]`
  Materialize every row; intended for small frames and tests.
- `def item(self) -> AnyValue`
  Return the only cell of a 1x1 dataframe.
- `def item(self, row: Int, column: String) -> AnyValue`
- `def equals(self, other: Self, *, null_equal: Bool = True) -> Bool`
  Same names, dtypes, order, height, and cells (NaN equals NaN).
- `def slice(self, offset: Int, length: Int = Int(-1)) -> Self`
  Rows [offset, offset + length), clipped to the frame.
- `def head(self, n: Int = Int(5)) -> Self`
  First n rows; a negative n drops the last -n rows.
- `def tail(self, n: Int = Int(5)) -> Self`
  Last n rows; a negative n drops the first -n rows.
- `def limit(self, n: Int = Int(5)) -> Self`
- `def reverse(self) -> Self`
- `def vstack(self, other: Self) -> Self`
  Append other's rows; schemas must match exactly.
- `def hstack(self, other: Self) -> Self`
  Append other's columns; heights must match, names stay unique.
- `def hstack(self, columns: List[Series]) -> Self`
- `def clear(self) -> Self`
  Zero rows with the same schema.
- `def drop(self, name: String) -> Self`
- `def drop(self, names: List[String]) -> Self`
  Remove columns; every name must exist and appear once.
- `def rename(self, mapping: Dict[String, String]) -> Self`
  Rename columns by old name; all names are validated first.
- `def with_row_index(self, name: String = "index", offset: Int64 = Int64(0)) -> Self`
  Prepend an Int64 row index starting at offset.
- `def select(self, names: List[String]) -> Self`
- `def select(self, expression: Expr, *, batch_size: Int = Int(1024)) -> Self`
- `def take(self, indices: List[Int]) -> Self`
- `def filter(self, mask: Column[Bool]) -> Self`
  Filter with a byte-per-value Boolean column (packed first).
- `def filter(self, mask: BoolColumn) -> Self`
  Keep true rows, dropping false and null mask entries, in input order.
- `def filter(self, predicate: Expr, *, batch_size: Int = Int(1024)) -> Self`
- `def with_column(self, var column: Series) -> Self`
  Replace by name or append; the input dataframe is unchanged.
- `def sort(self, by: String, descending: Bool = False, nulls_last: Bool = True) -> Self`
  Stable single-column sort. NaNs follow non-null numbers.
- `def sort(self, by: List[String], descending: Bool = False, nulls_last: Bool = True) -> Self`
  Stable lexicographic sort by several columns, one direction.
- `def sort(self, by: List[String], *, descending: List[Bool], nulls_last: List[Bool]) -> Self`
  Per-column direction and null placement; list lengths match by.
- `def arg_sort(self, by: List[String], descending: Bool = False, nulls_last: Bool = True) -> List[Int]`
- `def arg_sort(self, by: List[String], *, descending: List[Bool], nulls_last: List[Bool]) -> List[Int]`
  Row order of a stable sort; equal keys keep input order.
- `def top_k(self, k: Int, by: List[String]) -> Self`
  The k rows that sort(by, descending=True) would put first.
- `def top_k(self, k: Int, by: String) -> Self`
- `def bottom_k(self, k: Int, by: List[String]) -> Self`
  The k rows that sort(by) would put first; nulls rank last.
- `def bottom_k(self, k: Int, by: String) -> Self`
- `def join(self, right: Self, on: String, how: String = "inner", suffix: String = "_right", coalesce: Bool = True) -> Self`
- `def join(self, right: Self, on: List[String], how: String = "inner", suffix: String = "_right", coalesce: Bool = True) -> Self`
- `def join(self, right: Self, *, left_on: List[String], right_on: List[String], how: String = "inner", suffix: String = "_right", coalesce: Bool = True) -> Self`
  Hash join on key columns of any dtype; null keys never match.
- `def join(self, right: Self, *, how: String, suffix: String = "_right") -> Self`
  Cross join: every left row paired with every right row, left-major.
- `def select_exprs(self, expressions: List[Expr], *, batch_size: Int = Int(1024)) -> Self`
  Evaluate against the original frame. Scalar-only output has one row.
- `def with_columns(self, expression: Expr, *, batch_size: Int = Int(1024)) -> Self`
- `def with_columns(self, expressions: List[Expr], *, batch_size: Int = Int(1024)) -> Self`
  All siblings see the original schema and data; aliases are outputs.
- `def unpivot(self, on: List[String] = List(), index: List[String] = List(), *, variable_name: String = "variable", value_name: String = "value") -> Self`
  Wide to long: one row per (input row, `on` column).
- `def pivot(self, on: String, *, index: List[String], values: String, aggregate_function: String = "", sort_columns: Bool = False, batch_size: Int = Int(1024)) -> Self`
  Long to wide: one row per distinct index key, one column per distinct `on` value (first-occurrence order unless sort_columns).
- `def cast(self, dtypes: Dict[String, String], *, strict: Bool = True) -> Self`
  Cast named columns in place of the originals; order is kept.
- `def unique(self, subset: List[String] = List(), *, keep: String = "any", maintain_order: Bool = False) -> Self`
  Drop duplicate rows compared on subset (default: every column).
- `def n_unique(self, subset: List[String] = List()) -> Int`
  Number of distinct rows on subset (default: every column).
- `def is_duplicated(self, subset: List[String] = List()) -> Series`
  True for every row whose key occurs more than once.
- `def is_unique(self, subset: List[String] = List()) -> Series`
  True for every row whose key occurs exactly once.
- `def drop_nulls(self, subset: List[String] = List()) -> Self`
  Keep rows with no null in subset (default: every column).
- `def fill_null(self, value: String, subset: List[String] = List()) -> Self`
  Fill nulls in string columns with a string.
- `def fill_null(self, value: Expr, subset: List[String] = List()) -> Self`
  Fill nulls with a scalar value.
- `def group_by(self, key: String, *, maintain_order: Bool = False) -> GroupBy`
- `def group_by(self, keys: List[String], *, maintain_order: Bool = False) -> GroupBy`
  Group by one or more columns of any dtype.
- `def group_by(self, keys: List[Expr], *, maintain_order: Bool = False, batch_size: Int = Int(1024)) -> GroupBy`
  Group by computed keys; each key is evaluated once and named by its output name. Aggregations still see the original columns.
- `def group_indices(self, key: String) -> GroupIndices`
- `def group_indices(self, keys: List[String]) -> GroupIndices`
  Which rows belong to which group, without aggregating them.

## `DataType`

A logical column type.

Numeric: INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64,
FLOAT32, FLOAT64. Also BOOL, STRING, DATE (days since 1970-01-01), TIME
(nanoseconds since midnight), and datetime(unit) / duration(unit) with
unit "ns", "us", or "ms". Temporal types are stored as Int64.

- `def __eq__(self, other: Self) -> Bool`
- `def __ne__(self, other: Self) -> Bool`
- `def is_untyped(self) -> Bool`
  Whether this is an untyped literal awaiting a dtype (binder only).
- `def default(self) -> Self`
  The dtype an untyped literal takes on its own: Int64 or Float64.
- `def of(dtype: DType) -> Self`
  The DataType stored as Scalar[dtype] (numeric types only).
- `def storage(self) -> Optional[DType]`
  The numeric storage DType (int64 for temporal types); None for bool and string.
- `def datetime(unit: String = "us") -> Self`
  A time-zone-naive instant counted in unit since the epoch.
- `def duration(unit: String = "us") -> Self`
  A signed length of time counted in unit.
- `def parse(name: String) -> Self`
  The type with this canonical name; raises for unknown names.
- `def is_known(name: String) -> Bool`
- `def name(self) -> String`
  The canonical name, as accepted by parse.
- `def short_name(self) -> String`
  The compact name used in table headers (i64, f64, bool, str).
- `def unit(self) -> String`
  The time unit of a datetime or duration ("" otherwise).
- `def per_second(self) -> Int64`
  Ticks per second: 1e9 for ns, 1e6 for us, 1e3 for ms, and 1e9 for TIME; 0 for other types.
- `def is_temporal(self) -> Bool`
- `def is_date(self) -> Bool`
- `def is_datetime(self) -> Bool`
- `def is_duration(self) -> Bool`
- `def is_time(self) -> Bool`
- `def physical(self) -> Self`
  The storage type: INT64 for temporal types, otherwise self.
- `def is_numeric(self) -> Bool`
- `def is_integer(self) -> Bool`
- `def is_float(self) -> Bool`
- `def is_unsigned(self) -> Bool`
- `def is_signed(self) -> Bool`
  Whether values can be negative (numeric types only).
- `def bit_width(self) -> Int`
  Bits per value for fixed-width types; 0 for variable-width.
- `def sum_type(self) -> Self`
  The result type of sum and cumulative sums (as in Polars): 8- and 16-bit integers widen to INT64; other types keep their type.
- `def write_to(self, mut writer: T)`

## `date_range`

Dates from start through end (inclusive, ISO 8601 text) every interval, such as "1d", "1w", or "1mo".

```mojo
def date_range(start: String, end: String, interval: String = "1d", name: String = "date") -> Series
```

## `datetime_range`

Datetimes from start through end (inclusive) every interval.

```mojo
def datetime_range(start: String, end: String, interval: String, unit: String = "us", name: String = "datetime") -> Series
```

## `DtNamespace`

Temporal operations on date, datetime, time, and duration expressions.

Fields use the proleptic Gregorian calendar with no time zones. Nulls
propagate.

- `def year(self) -> Expr`
- `def month(self) -> Expr`
- `def day(self) -> Expr`
- `def hour(self) -> Expr`
- `def minute(self) -> Expr`
- `def second(self) -> Expr`
- `def nanosecond(self) -> Expr`
  Nanoseconds within the second.
- `def weekday(self) -> Expr`
  ISO weekday: Monday is 1, Sunday is 7.
- `def ordinal_day(self) -> Expr`
  Day of the year, starting at 1.
- `def date(self) -> Expr`
  The calendar date of a datetime.
- `def time(self) -> Expr`
  The time of day of a datetime.
- `def truncate(self, every: String) -> Expr`
  Round down to a multiple of every, such as "1d", "15m", "1w" (Monday-aligned), "1mo", "3mo", or "1y" (calendar-aligned).
- `def offset_by(self, by: String) -> Expr`
  Shift by an interval such as "2d", "-3h", "1mo" or "1y2mo"; month shifts clamp to the last day of the target month.
- `def total(self, unit: String) -> Expr`
  A duration as a whole number of days, hours, minutes, seconds, milliseconds, microseconds, or nanoseconds (truncated), as Int64.
- `def total_days(self) -> Expr`
- `def total_hours(self) -> Expr`
- `def total_minutes(self) -> Expr`
- `def total_seconds(self) -> Expr`
- `def total_milliseconds(self) -> Expr`
- `def strftime(self, format: String) -> Expr`
  Format as text; see dataframe/temporal.mojo for directives.

## `exclude`

Every column except the listed ones, in schema order.

```mojo
def exclude(names: List[String]) -> Expr
```

## `export_arrow`

Export a frame as an Arrow struct array (format `+s`), the shape pyarrow and Polars import as a RecordBatch.

```mojo
def export_arrow(frame: DataFrame, mut array: ArrowArray, mut schema: ArrowSchema)
```

```mojo
def export_arrow(frame: DataFrame, array_address: Int, schema_address: Int)
```

## `export_arrow_series`

Fill consumer-allocated ArrowArray and ArrowSchema structs.

```mojo
def export_arrow_series(series: Series, mut array: ArrowArray, mut schema: ArrowSchema)
```

```mojo
def export_arrow_series(series: Series, array_address: Int, schema_address: Int)
```

## `Expr`

A flat, topologically ordered tree; composition never evaluates data.

- `def __init__(out self, then: Then)`
  A when/then chain without otherwise yields null for unmatched rows.
- `def __init__(out self, value: Int)`
- `def __init__(out self, value: Float64)`
- `def __init__(out self, value: Bool)`
- `def __neg__(self) -> Self`
- `def __invert__(self) -> Self`
- `def __lt__(self, other: Self) -> Self`
- `def __lt__(self, other: String) -> Self`
- `def __le__(self, other: Self) -> Self`
- `def __le__(self, other: String) -> Self`
- `def __eq__(self, other: Self) -> Self`
  Elementwise equality (an expression, not a Bool); same as eq.
- `def __eq__(self, other: String) -> Self`
- `def __ne__(self, other: Self) -> Self`
  Elementwise inequality (an expression, not a Bool); same as ne.
- `def __ne__(self, other: String) -> Self`
- `def __gt__(self, other: Self) -> Self`
- `def __gt__(self, other: String) -> Self`
- `def __ge__(self, other: Self) -> Self`
- `def __ge__(self, other: String) -> Self`
- `def __add__(self, other: Self) -> Self`
- `def __sub__(self, other: Self) -> Self`
- `def __mul__(self, other: Self) -> Self`
- `def __truediv__(self, other: Self) -> Self`
  True division; always Float64, including for Int64 operands.
- `def __floordiv__(self, other: Self) -> Self`
  Floor division; Int64 division by zero yields null.
- `def __mod__(self, other: Self) -> Self`
  Remainder with the divisor's sign; Int64 modulo zero yields null.
- `def __pow__(self, other: Self) -> Self`
- `def __and__(self, other: Self) -> Self`
  Kleene AND: false wins over null; otherwise null propagates.
- `def __or__(self, other: Self) -> Self`
  Kleene OR: true wins over null; otherwise null propagates.
- `def __xor__(self, other: Self) -> Self`
- `def __radd__(self, other: Self) -> Self`
- `def __rsub__(self, other: Self) -> Self`
- `def __rmul__(self, other: Self) -> Self`
- `def __rtruediv__(self, other: Self) -> Self`
- `def __rfloordiv__(self, other: Self) -> Self`
- `def __rmod__(self, other: Self) -> Self`
- `def __rpow__(self, other: Self) -> Self`
- `def __rand__(self, other: Self) -> Self`
- `def __ror__(self, other: Self) -> Self`
- `def __rxor__(self, other: Self) -> Self`
- `def alias(self, name: String) -> Self`
- `def name_prefix(self, prefix: String) -> Self`
  Prefix the output name; for selectors, every expanded name.
- `def name_suffix(self, suffix: String) -> Self`
  Suffix the output name; for selectors, every expanded name.
- `def pow(self, exponent: Self) -> Self`
- `def eq(self, other: String) -> Self`
- `def eq(self, other: Self) -> Self`
- `def ne(self, other: String) -> Self`
- `def ne(self, other: Self) -> Self`
- `def fill_null(self, value: String) -> Self`
- `def fill_null(self, value: Self) -> Self`
  Replace nulls with value; the dtypes must match.
- `def is_in(self, values: List[String]) -> Self`
- `def is_in(self, values: List[Self]) -> Self`
  True when equal to any value; null input stays null.
- `def abs(self) -> Self`
- `def sqrt(self) -> Self`
- `def exp(self) -> Self`
- `def log(self) -> Self`
  Natural logarithm.
- `def floor(self) -> Self`
- `def ceil(self) -> Self`
- `def round(self, decimals: Int = Int(0)) -> Self`
  Round half away from zero to `decimals` places.
- `def clip(self, lower: Self, upper: Self) -> Self`
  Bound values to [lower, upper]; nulls and NaN pass through.
- `def clip_min(self, lower: Self) -> Self`
- `def clip_max(self, upper: Self) -> Self`
- `def and_(self, other: Self) -> Self`
- `def or_(self, other: Self) -> Self`
- `def xor(self, other: Self) -> Self`
- `def not_(self) -> Self`
- `def is_null(self) -> Self`
- `def is_not_null(self) -> Self`
- `def is_nan(self) -> Self`
- `def is_not_nan(self) -> Self`
- `def is_finite(self) -> Self`
- `def is_infinite(self) -> Self`
- `def fill_nan(self, value: Self) -> Self`
  Replace valid NaNs in a Float64 expression with value.
- `def is_between(self, lower: Self, upper: Self, closed: String = "both") -> Self`
  True when lower <= x <= upper; `closed` is both, left, right, or none.
- `def any(self, ignore_nulls: Bool = True) -> Self`
  Any true value. With ignore_nulls=False, Kleene: null if no true and some null.
- `def all(self, ignore_nulls: Bool = True) -> Self`
  All values true; empty is true. With ignore_nulls=False, Kleene: null if no false and some null.
- `def null_count(self) -> Self`
  Number of null values, as Int64.
- `def cast(self, dtype: DataType, strict: Bool = True) -> Self`
  Convert to dtype; see the String overload.
- `def cast(self, dtype: String, strict: Bool = True) -> Self`
  Convert to any dtype by name (numeric, bool, string, temporal).
- `def cum_sum(self, reverse: Bool = False) -> Self`
  Running sum of non-null values; null rows stay null. Int64 is checked for overflow.
- `def cum_min(self, reverse: Bool = False) -> Self`
- `def cum_max(self, reverse: Bool = False) -> Self`
- `def cum_count(self, reverse: Bool = False) -> Self`
  Running count of non-null values, as Int64 (never null).
- `def shift(self, n: Int = Int(1)) -> Self`
  Move values n rows later (earlier when negative); vacated rows are null.
- `def diff(self, n: Int = Int(1)) -> Self`
  Difference from the value n rows earlier.
- `def pct_change(self, n: Int = Int(1)) -> Self`
  Relative change from the value n rows earlier, as Float64.
- `def rank(self, method: String = "average", descending: Bool = False) -> Self`
  Rank non-null values: average, min, max, dense, or ordinal.
- `def rolling_sum(self, window_size: Int, min_samples: Int = Int(-1)) -> Self`
  Sum over the current row and the window_size - 1 rows before it.
- `def rolling_mean(self, window_size: Int, min_samples: Int = Int(-1)) -> Self`
- `def rolling_min(self, window_size: Int, min_samples: Int = Int(-1)) -> Self`
- `def rolling_max(self, window_size: Int, min_samples: Int = Int(-1)) -> Self`
- `def forward_fill(self, limit: Int = Int(-1)) -> Self`
  Fill nulls with the last valid value, at most limit rows ahead (-1 means unlimited).
- `def backward_fill(self, limit: Int = Int(-1)) -> Self`
- `def over(self, partition_by: String) -> Self`
- `def over(self, partition_by: List[String]) -> Self`
  Evaluate within partitions of the key columns, keeping row order.
- `def str(self) -> StrNamespace`
  String operations: col("name").str().to_uppercase().
- `def dt(self) -> DtNamespace`
  Temporal operations: col("when").dt().year().
- `def min(self) -> Self`
  Smallest non-null value. NaN sorts above every number, so it is the minimum only when every valid value is NaN.
- `def max(self) -> Self`
  Largest non-null value; any valid NaN makes the maximum NaN.
- `def mean(self) -> Self`
  Arithmetic mean of non-null values as Float64; null when empty.
- `def first(self) -> Self`
  The first row's value, which may be null. Order-dependent.
- `def last(self) -> Self`
  The last row's value, which may be null. Order-dependent.
- `def n_unique(self) -> Self`
  Distinct values, counting null once. NaNs are one value and -0.0 equals 0.0.
- `def std(self, ddof: Int = Int(1)) -> Self`
  Standard deviation; null when fewer than ddof + 1 values.
- `def var(self, ddof: Int = Int(1)) -> Self`
  Variance; null when fewer than ddof + 1 values.
- `def median(self) -> Self`
- `def quantile(self, quantile: Float64, interpolation: String = "linear") -> Self`
  Interpolation: nearest, lower, higher, midpoint, or linear.
- `def len(self) -> Self`
  Number of rows including nulls, as Int64.
- `def sum(self, min_count: Int = Int(0)) -> Self`
  Skip nulls; zero when empty unless fewer than min_count are valid.
- `def count(self) -> Self`
  Number of non-null values, as Int64.

## `Field`

One schema entry: a column name and its dtype.


## `first`

The first column in the schema.

```mojo
def first() -> Expr
```

## `GroupBy`

An eager grouping request. No per-group dataframe materialization.

It owns a snapshot of the input and the evaluated key columns.

- `def agg(self, expression: Expr, *, batch_size: Int = Int(1024)) -> DataFrame`
- `def agg(self, expressions: List[Expr], *, batch_size: Int = Int(1024)) -> DataFrame`
- `def len(self, name: String = "len") -> DataFrame`
  Row count per group, including rows with null values.

## `GroupIndices`

Which rows belong to which group, plus a representative row per group.

`ids()[i]` is row i's group. `representative(g)` is the first row of
group g, which is where to read that group's key values from.

- `def __init__(out self, var ids: List[Int], var representatives: List[Int])`
- `def count(self) -> Int`
  The number of groups.
- `def height(self) -> Int`
  The number of rows these groups cover.
- `def ids(self) -> List[Int]`
  Each row's group id, in row order.
- `def group_of(self, row: Int) -> Int`
  Row's group id.
- `def representative(self, group: Int) -> Int`
  The first row of this group: read the group's key values there.
- `def representatives(self) -> List[Int]`
  The first row of every group, in group order.
- `def rows(self, group: Int) -> List[Int]`
  One group's rows, in row order.
- `def all_rows(self) -> List[List[Int]]`
  Every group's rows, in group order then row order, in one pass.
- `def sizes(self) -> List[Int]`
  Each group's row count, in group order.

## `import_arrow`

Copy an exported Arrow struct array (a record batch) into a frame, then release it. The input structs are consumed even when import fails.

```mojo
def import_arrow(mut array: ArrowArray, mut schema: ArrowSchema) -> DataFrame
```

```mojo
def import_arrow(array_address: Int, schema_address: Int) -> DataFrame
```

## `import_arrow_series`

Copy an exported Arrow array into a Series, then release it.

```mojo
def import_arrow_series(mut array: ArrowArray, mut schema: ArrowSchema) -> Series
```

```mojo
def import_arrow_series(array_address: Int, schema_address: Int) -> Series
```

## `last`

The last column in the schema.

```mojo
def last() -> Expr
```

## `LazyFrame`

A deferred query; build it with DataFrame.lazy() or scan_csv().

- `def __init__(out self, frame: DataFrame)`
- `def __init__(out self, var nodes: List[PlanNode], var frames: List[DataFrame], var schemas: List[Optional[CsvSchema]])`
- `def filter(self, predicate: Expr) -> Self`
- `def select(self, expr: Expr) -> Self`
- `def select(self, names: List[String]) -> Self`
- `def select_exprs(self, exprs: List[Expr]) -> Self`
- `def with_columns(self, expr: Expr) -> Self`
- `def with_columns(self, exprs: List[Expr]) -> Self`
- `def group_by(self, keys: List[String], *, maintain_order: Bool = False) -> LazyGroupBy`
- `def group_by(self, key: String, *, maintain_order: Bool = False) -> LazyGroupBy`
- `def sort(self, by: List[String], descending: Bool = False, nulls_last: Bool = True) -> Self`
- `def sort(self, by: String, descending: Bool = False) -> Self`
- `def slice(self, offset: Int, length: Int = Int(-1)) -> Self`
- `def head(self, n: Int = Int(5)) -> Self`
- `def limit(self, n: Int = Int(5)) -> Self`
- `def unique(self, subset: List[String] = List(), *, keep: String = "any", maintain_order: Bool = False) -> Self`
- `def drop(self, names: List[String]) -> Self`
- `def join(self, other: Self, on: List[String], how: String = "inner", suffix: String = "_right") -> Self`
  Join with another lazy plan; see DataFrame.join.
- `def join(self, other: Self, on: String, how: String = "inner", suffix: String = "_right") -> Self`
- `def collect(self, *, optimize: Bool = True) -> DataFrame`
  Optimize (unless disabled) and execute the plan.
- `def fetch(self, n: Int = Int(5)) -> DataFrame`
  Collect only the first n rows of the result.
- `def collect_schema(self) -> List[String]`
  Output names and dtypes as "name: dtype", computed without reading rows: every scan yields zero rows, then the plan runs as usual, so binding validates each expression exactly as collect would.
- `def explain(self, *, optimize: Bool = True) -> String`
  The (optimized) plan, one operator per line, root first.

## `LazyGroupBy`

A pending lazy grouping; finish it with agg.

- `def agg(self, exprs: List[Expr]) -> LazyFrame`
- `def agg(self, expr: Expr) -> LazyFrame`

## `lit`

A typed scalar literal; there is no implicit numeric promotion.

```mojo
def lit(value: Int64) -> Expr
```

```mojo
def lit(value: Float64) -> Expr
```

```mojo
def lit[D: DType](value: Scalar[D]) -> Expr
```

```mojo
def lit(value: Int8) -> Expr
```

```mojo
def lit(value: Int16) -> Expr
```

```mojo
def lit(value: Int32) -> Expr
```

```mojo
def lit(value: UInt8) -> Expr
```

```mojo
def lit(value: UInt16) -> Expr
```

```mojo
def lit(value: UInt32) -> Expr
```

```mojo
def lit(value: UInt64) -> Expr
```

```mojo
def lit(value: Float32) -> Expr
```

```mojo
def lit(value: Bool) -> Expr
```

```mojo
def lit(value: String) -> Expr
```

## `nth`

The column at a position; negative positions count from the end.

```mojo
def nth(index: Int) -> Expr
```

## `null`

A typed null literal of any dtype name (see DataType.parse).

```mojo
def null(dtype: String) -> Expr
```

## `read_csv`

Read a strict UTF-8 CSV file into typed, nullable columns.

```mojo
def read_csv(path: String, schema: CsvSchema, *, has_header: Bool = True, separator: String = ",", quote_char: String = "\22", comment_prefix: String = "", skip_rows: Int = Int(0), n_rows: Int = Int(-1), columns: List[String] = List(), null_values: List[String] = List(), ignore_errors: Bool = False, truncate_ragged_lines: Bool = False, encoding: String = "utf8", buffer_size: Int = Int(65536)) -> DataFrame
```

```mojo
def read_csv(path: String, *, infer_schema_length: Int = Int(10000), schema_overrides: Dict[String, String] = Dict(), has_header: Bool = True, separator: String = ",", quote_char: String = "\22", comment_prefix: String = "", skip_rows: Int = Int(0), n_rows: Int = Int(-1), columns: List[String] = List(), null_values: List[String] = List(), ignore_errors: Bool = False, truncate_ragged_lines: Bool = False, encoding: String = "utf8", buffer_size: Int = Int(65536)) -> DataFrame
```

## `scan_csv`

Lazily read a CSV file with an inferred schema. Nothing is read until collect; projection and head() are pushed into the reader.

```mojo
def scan_csv(path: String) -> LazyFrame
```

```mojo
def scan_csv(path: String, schema: CsvSchema) -> LazyFrame
```

## `Series`

A named column of one supported dtype, plus expression-backed methods.

- `def __init__[D: DType](out self, var name: String, var column: Column[Scalar[D]])`
- `def __init__(out self, var name: String, var column: BoolColumn)`
- `def __init__(out self, var name: String, column: Column[Bool])`
  Pack a byte-per-value Boolean column into bits.
- `def __init__(out self, var name: String, var column: StringColumn)`
- `def __init__(out self, var name: String, column: Column[String])`
  Convert list-backed strings to the contiguous UTF-8 layout.
- `def __getitem__(self, index: Int) -> AnyValue`
  One cell; raises when out of bounds. Negative indices count from the end.
- `def __neg__(self) -> Self`
- `def __invert__(self) -> Self`
- `def __lt__(self, other: Self) -> Self`
- `def __lt__(self, other: Expr) -> Self`
- `def __le__(self, other: Self) -> Self`
- `def __le__(self, other: Expr) -> Self`
- `def __gt__(self, other: Self) -> Self`
- `def __gt__(self, other: Expr) -> Self`
- `def __ge__(self, other: Self) -> Self`
- `def __ge__(self, other: Expr) -> Self`
- `def __add__(self, other: Self) -> Self`
- `def __add__(self, other: Expr) -> Self`
- `def __sub__(self, other: Self) -> Self`
- `def __sub__(self, other: Expr) -> Self`
- `def __mul__(self, other: Self) -> Self`
- `def __mul__(self, other: Expr) -> Self`
- `def __truediv__(self, other: Self) -> Self`
- `def __truediv__(self, other: Expr) -> Self`
- `def __floordiv__(self, other: Self) -> Self`
- `def __floordiv__(self, other: Expr) -> Self`
- `def __mod__(self, other: Self) -> Self`
- `def __mod__(self, other: Expr) -> Self`
- `def __pow__(self, other: Self) -> Self`
- `def __pow__(self, other: Expr) -> Self`
- `def __and__(self, other: Self) -> Self`
- `def __or__(self, other: Self) -> Self`
- `def __xor__(self, other: Self) -> Self`
- `def with_dtype(self, dtype: DataType) -> Self`
  The same values tagged with another logical type that shares their storage (temporal types and INT64).
- `def name(self) -> String`
- `def write_to(self, mut writer: T)`
- `def to_string(self, *, max_rows: Int = Int(10), max_string_length: Int = Int(32)) -> String`
  Render at most max_rows values; negative means unlimited.
- `def cast(self, dtype: DataType, strict: Bool = True) -> Self`
  Convert to another dtype; see Expr.cast.
- `def cast(self, dtype: String, strict: Bool = True) -> Self`
- `def renamed(self, var name: String) -> Self`
- `def dtype(self) -> DataType`
- `def __len__(self) -> Int`
- `def null_count(self) -> Int`
- `def get(self, index: Int) -> AnyValue`
  Return one cell as a tagged value; raises when out of bounds.
- `def equals(self, other: Self, *, null_equal: Bool = True, check_names: Bool = False) -> Bool`
  Structural equality: NaN equals NaN and -0.0 equals 0.0.
- `def eq(self, other: Self) -> Self`
- `def eq(self, other: Expr) -> Self`
- `def ne(self, other: Self) -> Self`
- `def ne(self, other: Expr) -> Self`
- `def apply(self, expr: Expr) -> Self`
  Evaluate any expression written against this series' name.
- `def is_null(self) -> Self`
- `def is_not_null(self) -> Self`
- `def fill_null(self, value: Expr) -> Self`
- `def abs(self) -> Self`
- `def round(self, decimals: Int = Int(0)) -> Self`
- `def sum(self) -> AnyValue`
- `def mean(self) -> AnyValue`
- `def min(self) -> AnyValue`
- `def max(self) -> AnyValue`
- `def median(self) -> AnyValue`
- `def quantile(self, quantile: Float64, interpolation: String = "linear") -> AnyValue`
- `def std(self, ddof: Int = Int(1)) -> AnyValue`
- `def var(self, ddof: Int = Int(1)) -> AnyValue`
- `def count(self) -> Int`
- `def n_unique(self) -> Int`
- `def first(self) -> AnyValue`
- `def last(self) -> AnyValue`
- `def any(self, ignore_nulls: Bool = True) -> AnyValue`
- `def all(self, ignore_nulls: Bool = True) -> AnyValue`
- `def head(self, n: Int = Int(5)) -> Self`
- `def tail(self, n: Int = Int(5)) -> Self`
- `def sort(self, descending: Bool = False, nulls_last: Bool = True) -> Self`
- `def unique(self, maintain_order: Bool = False) -> Self`
  Distinct values, null counted once.
- `def value_counts(self, sort: Bool = True, name: String = "count") -> DataFrame`
  Distinct values and their row counts (nulls included), most frequent first when sort=True; ties keep first-occurrence order.
- `def to_values(self) -> List[AnyValue]`
- `def int64(self) -> Column[Int64]`
  Return an owned typed copy, raising on a dtype mismatch.
- `def float64(self) -> Column[Float64]`
  Return an owned typed copy, raising on a dtype mismatch.
- `def numeric[D: DType](self) -> Column[Scalar[D]]`
  The (shared, immutable) column as Scalar[D], raising on a dtype mismatch. Temporal columns read as their Int64 storage.
- `def int8(self) -> Column[Int8]`
- `def int16(self) -> Column[Int16]`
- `def int32(self) -> Column[Int32]`
- `def uint8(self) -> Column[UInt8]`
- `def uint16(self) -> Column[UInt16]`
- `def uint32(self) -> Column[UInt32]`
- `def uint64(self) -> Column[UInt64]`
- `def float32(self) -> Column[Float32]`
- `def bool(self) -> BoolColumn`
  Return an owned typed copy, raising on a dtype mismatch.
- `def string(self) -> StringColumn`
  Return the (shared, immutable) column, raising on a dtype mismatch.
- `def take(self, indices: List[Int]) -> Self`
- `def take_or_null(self, indices: List[Int]) -> Self`
- `def argsort(self, descending: Bool = False, nulls_last: Bool = True) -> List[Int]`
  Stable sort order: ranks are resolved once, then merged by Int.
- `def slice(self, offset: Int, length: Int) -> Self`
- `def full_null(var name: String, dtype: String, length: Int) -> Self`
- `def full_null(var name: String, dtype: DataType, length: Int) -> Self`
  A column of `length` nulls with the requested dtype.
- `def append(self, other: Self) -> Self`
  Return a new series with other's rows after this one's.
- `def reverse(self) -> Self`

## `StringBuilder`

Appends rows into fresh UTF-8, offset, and validity buffers.

Kernels build string results here instead of collecting `String`s.

- `def __init__(out self, rows: Int = Int(0), bytes: Int = Int(0))`
  Reserve for about `rows` rows and `bytes` bytes of text.
- `def __len__(self) -> Int`
- `def append(mut self, text: StringSpan)`
- `def append(mut self, text: String)`
- `def append_null(mut self)`
- `def finish(deinit self) -> StringColumn`

## `StringColumn`

A window onto shared UTF-8 bytes, Int64 offsets, and validity.

- `def __init__(out self, values: List[String])`
- `def __init__(out self, values: List[String], valid: List[Bool])`
- `def __init__(out self, column: Column[String])`
  Convert a list-backed string column into the UTF-8 layout.
- `def __init__(out self, *, var bytes: List[UInt8], var offsets: List[Int64], var bits: List[UInt8], length: Int)`
  Adopt finished buffers; offsets must have length + 1 entries.
- `def __len__(self) -> Int`
- `def is_null(self, index: Int) -> Bool`
- `def value(self, index: Int) -> String`
- `def null_count(self) -> Int`
- `def take(self, indices: List[Int]) -> Self`
- `def take_or_null(self, indices: List[Int], fill: String) -> Self`
  Gather rows, treating only -1 as a missing row (for outer joins).
- `def slice(self, offset: Int, length: Int) -> Self`
  A zero-copy window sharing this column's buffers.

## `StrNamespace`

String expressions. Character operations work on Unicode code points; there is no grapheme clustering or locale-specific case mapping. Nulls propagate. Patterns are literal text; regular expressions are not supported.

- `def len_chars(self) -> Expr`
  Number of Unicode code points, as Int64.
- `def len_bytes(self) -> Expr`
  Number of UTF-8 bytes, as Int64.
- `def to_uppercase(self) -> Expr`
- `def to_lowercase(self) -> Expr`
- `def strip_chars(self, characters: String = "") -> Expr`
  Strip characters from both ends; empty means ASCII whitespace.
- `def strip_chars_start(self, characters: String = "") -> Expr`
- `def strip_chars_end(self, characters: String = "") -> Expr`
- `def starts_with(self, prefix: String) -> Expr`
- `def ends_with(self, suffix: String) -> Expr`
- `def contains(self, literal: String) -> Expr`
  Literal substring test; regular expressions are not supported.
- `def replace(self, pattern: String, value: String) -> Expr`
  Replace the first occurrence of a literal pattern.
- `def replace_all(self, pattern: String, value: String) -> Expr`
- `def slice(self, offset: Int, length: Int = Int(-1)) -> Expr`
  Code points [offset, offset + length); a negative offset counts from the end and length=-1 takes the rest. Out-of-range parts clip.
- `def head(self, n: Int) -> Expr`
- `def tail(self, n: Int) -> Expr`
- `def reverse(self) -> Expr`
  Reverse code point order.
- `def pad_start(self, width: Int, fill_char: String = " ") -> Expr`
  Left-pad to width code points; longer strings are unchanged.
- `def pad_end(self, width: Int, fill_char: String = " ") -> Expr`
- `def strptime(self, dtype: String, format: String = "", strict: Bool = True) -> Expr`
  Parse text as "date", "datetime[unit]", or "time" using a strftime-style format (ISO 8601 when empty); unparseable text raises when strict, or is null otherwise.
- `def to_date(self, format: String = "") -> Expr`
- `def to_datetime(self, format: String = "", unit: String = "us") -> Expr`
- `def zfill(self, width: Int) -> Expr`
  Left-pad with zeros, after a leading + or - sign.

## `Then`

A when/then chain; add branches with `when` or finish with `otherwise`.

It converts implicitly to an Expr whose unmatched rows are null.

- `def when(self, condition: Expr) -> When`
- `def otherwise(self, value: String) -> Expr`
- `def otherwise(self, value: Expr) -> Expr`
- `def end(self) -> Expr`
- `def alias(self, name: String) -> Expr`

## `to_csv_string`

Render the whole frame as CSV text; use write_csv for large frames.

```mojo
def to_csv_string(frame: DataFrame, *, has_header: Bool = True, separator: String = ",", quote_style: String = "necessary", null_value: String = "", line_terminator: String = "\n") -> String
```

## `When`

A pending condition; call `then` to supply its value.

- `def then(self, value: String) -> Then`
- `def then(self, value: Expr) -> Then`

## `when`

Start a conditional: when(p).then(a).when(q).then(b).otherwise(c).

```mojo
def when(condition: Expr) -> When
```

## `write_csv`

Stream a frame to a UTF-8 CSV file that read_csv reads back exactly.

```mojo
def write_csv(frame: DataFrame, path: String, *, has_header: Bool = True, separator: String = ",", quote_style: String = "necessary", null_value: String = "", line_terminator: String = "\n", buffer_size: Int = Int(65536))
```
