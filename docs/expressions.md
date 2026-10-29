# Expression semantics and CPU execution

The API draws on Narwhals' separation of expressions, eager dataframes, and
series, and on its explicit broadcasting rules. This is a native implementation,
not a Narwhals adapter or a claim of complete behavioral compatibility.

References:

- https://narwhals-dev.github.io/narwhals/how_it_works/
- https://narwhals-dev.github.io/narwhals/concepts/order_dependence/
- https://narwhals-dev.github.io/narwhals/api-reference/expr/#narwhals.expr.Expr.sum

## Binding and shape

`col(name)` refers to a column; `lit(value)` holds a scalar of any numeric
type (`lit(Int32(1))`, `lit(Float32(0.5))`, ...), Bool, or String.

Bare values work wherever an expression is expected: `col("x") > 0`,
`col("x") * 2.5`, `1 + col("x")`, `col("k") == "a"`, `col("k").is_in(["a",
"b"])`, `col("x").fill_null(0)`, `when(c).then(1).otherwise(0)`. Bare numbers
are *untyped*: an integer adopts the numeric dtype of the operand it meets
(range-checked, so `col("u8") + 300` is a bind error), and becomes a float
next to a float column; a float adopts Float32 or Float64 and is an error next
to an integer column. Adoption follows arithmetic between untyped values
(`col("x") * (2 + 3)`), and an untyped expression on its own is Int64 or
Float64. `lit(...)` fixes a type and never adopts. Two Mojo limits: a string
or number on the *left* of `==` does not compile (write `col("k") == "a"`),
and strings convert only in the comparison, `is_in`, `fill_null`, and
`then`/`otherwise` methods (elsewhere use `lit("a")`). `Expr` owns a flat topological node list. Building an expression
never reads data. `alias` changes the output name, not column resolution.
Without an alias, binary operations retain the left expression's output name;
literals are named `literal`, and reductions retain their input expression name.

The binder resolves names to column indices, validates operand dtypes, records
scalar/row-valued/aggregate shape, and tracks aggregate dependencies. Every
expression in an operation is bound before any expression kernel executes.
Duplicate output names, unknown columns, invalid types, negative `min_count`,
and nested aggregates raise even on empty dataframes.

A bare `col("category")` retains its categorical dtype and dictionary. An
expression that operates on categorical values reads their decoded String
values. Its result follows the String operation's type, including String
extrema and first/last, String mode values, and String fields in value counts.
The reduction table's "same" and "input type" refer to this decoded input.
Direct table operations such as filtering, sorting, taking rows, and joining
retain categorical metadata under their own contracts.

Operands must have matching dtypes; there is no implicit promotion or index
alignment (`cast` mixed widths explicitly). Nulls propagate through every
operator below; NaN is distinct from null. Numeric types are Int8, Int16,
Int32, Int64, UInt8, UInt16, UInt32, UInt64 ("integers") and Float32, Float64
("floats"). Integer arithmetic checks overflow per valid element at the
operand's width and raises (`int8 addition overflow`); widths below 64 bits
and UInt64 compute in 128 bits and range-check the result.

| Operation | Operands | Result | Notes |
|---|---|---|---|
| `+`, `-`, `*` | numeric | same | checked for integers; unsigned `0 - 1` raises |
| `/` | numeric | Float64 (Float32 for Float32) | integers yield Float64; IEEE division by zero |
| `//` | numeric | same | floor division; integer by zero is null; signed `MIN // -1` raises |
| `%` | numeric | same | remainder takes the divisor's sign; integer modulo zero is null; float `x % 0`, `inf % y`, and NaN operands give NaN |
| `**`, `.pow()` | numeric | same | integers are checked and raise on a negative exponent |
| `<`, `<=`, `>`, `>=` | any | Bool | IEEE for floats (NaN compares false); strings byte-lexicographic, as in sort; `false < true` |
| `==`, `!=`, `.eq()`, `.ne()` | any | Bool | IEEE: NaN `!=` NaN is true |
| unary `-`, `.abs()` | numeric | same | signed `MIN` raises; unsigned negation raises unless zero |
| `.sqrt()`, `.exp()`, `.log()` | numeric | Float64 (Float32 for Float32) | natural log; IEEE results for negative inputs and zero |
| `.log(base)`, `.log10()`, `.log1p()`, `.cbrt()` | numeric | Float64 (Float32 for Float32) | `log(base)` is `ln(x) / ln(base)`, with `log10` and base 2 exact at powers of the base; `log10(0)` is -inf, `log10(-1)` NaN; `log1p(-1)` is -inf |
| `.sin()`, `.cos()`, `.tan()`, `.cot()`, `.arcsin()`, `.arccos()`, `.arctan()` | numeric | Float64 (Float32 for Float32) | radians; out-of-domain input (`arcsin(2)`) is NaN; `cot(0)` is inf |
| `.sinh()`, `.cosh()`, `.tanh()`, `.arcsinh()`, `.arccosh()`, `.arctanh()` | numeric | Float64 (Float32 for Float32) | `arccosh` is NaN below 1; `arctanh(1)` is inf |
| `.degrees()`, `.radians()` | numeric | Float64 (Float32 for Float32) | one multiply by `180 / pi` or `pi / 180`, as Rust does |
| `arctan2(y, x)` | two numeric | Float64 (Float32 when `y` is Float32) | `x` is cast to `y`'s float type; null when either side is null |
| `.sign()` | numeric | same | -1, 0 or 1 (unsigned: 0 or 1); a float NaN stays NaN and a signed zero keeps its sign; a decimal's sign is at its scale |
| `.truncate()` | numeric | same | toward zero; identity for integers |
| `.floor()`, `.ceil()`, `.round(decimals=0)` | numeric | same | identity for integers; `round` is half away from zero and accepts negative decimals |
| `.clip(lower, upper)`, `.clip_min`, `.clip_max` | numeric | same | NaN inputs and NaN bounds leave the value unchanged |

For a finite float numerator and an infinite divisor, `%` follows Python's
remainder rule. A nonzero numerator with the divisor's sign is unchanged;
a nonzero numerator with the opposite sign yields the signed infinite divisor.
A zero numerator yields zero with the divisor's sign, including signed zero.

Float `+ - * /` use explicit SIMD kernels (Float64 and Float32 alike); the
other float operations run per lane. Comparisons on every numeric type, and
`&`, `|`, `^` and `~` on Booleans, write packed value and validity bitmaps
directly, eight rows per step. Fusion covers Float64 arithmetic, and
comparisons over it; a comparison of two plain columns or literals uses the
bitmap kernel instead. The logarithms beyond `log()`, `log1p`, `cbrt`, the
trigonometric and hyperbolic functions and `arctan2` call the platform's IEEE
`libm` lane by lane, as Polars does through Rust; `sign`, `truncate`,
`degrees` and `radians` are SIMD. Integer and decimal inputs compute in
Float64. `sqrt`, `exp`, `log`, and `pow` use
Mojo's `std.math`, whose results can differ from the correctly rounded value by
about 1e-12 relative; tests compare them with a tolerance. Binder errors name
the operator and dtypes, for example `/ requires matching dtypes, found int64
and float64; use typed literals`.

Decimal arithmetic follows current Polars result types: `+`, `-`, `*`, and `/`
return precision 38 with the larger operand scale. Multiplication and division
round the digits past that scale half to even, as Polars does (0.625 becomes
0.62, 0.375 becomes 0.38, and -0.625 becomes -0.62). Casts that drop decimal
digits, to a smaller scale or to an integer, round the same way (1.5 and 2.5
both become 2). Decimal division by zero produces null. Decimal `sum`, `min`,
and `max` keep the input's decimal type (`sum` is exact and raises when it
exceeds the precision); `mean` returns Float64, as Polars does, computed from
the exact sum so that only the final conversion rounds.

### Selectors

`all()`, `col([names])`, `exclude([names])`, `by_dtype([dtypes])`, `nth(i)`
(negative counts from the end), `first()`, and `last()` select columns. Any
operation on a selector applies to each matched column: before binding, the
expression expands into one expression per match, in schema order for `all`,
`exclude`, and `by_dtype`, and in the listed order for `col([...])`. The
exception is a selector among the inputs of a horizontal reduction
(`sum_horizontal([by_dtype(["int64"])])`): it expands into more inputs of
that one expression, as in Polars. Expansion
happens for `select`, `select_exprs`, `with_columns`, `agg`, and `group_by`
key expressions; `filter` requires exactly one match. An empty expansion adds
no columns.

Each expanded output is named after its column. `name_prefix(p)` and
`name_suffix(s)` transform those names (or, on ordinary expressions, the one
output name). An `alias` gives every expansion the same name, so it only works
when one column matches; duplicate output names raise. An expression may hold
at most one selector. Unknown names, unknown dtypes, and out-of-range `nth`
positions raise at bind time, even on empty frames. Sibling expressions in one
`with_columns` still see the original input, not each other's outputs.

### Order-dependent and window expressions

These depend on row order, so they are computed over the whole input column
(and per partition) before batches are evaluated. Their input must be
row-valued. See the Narwhals order-dependence notes cited above.

| Method | Result | Nulls |
|---|---|---|
| `cum_sum(reverse=False)` | numeric, same dtype; Int64 checked; Bool gives UInt32, a running count of true values | null rows stay null and are skipped |
| `cum_min`, `cum_max` | same dtype; NaN is skipped, so a NaN row repeats the running value and is NaN only before the first number, as in Polars | as `cum_sum` |
| `cum_prod(reverse=False)` | as `product()`: Int64 for Bool and integers up to 32 bits, UInt64/Int64/floats keep theirs, decimals Float64; integers checked | as `cum_sum` |
| `cum_count(reverse=False)` | Int64, never null | counts non-null values |
| `shift(n=1)` | same dtype; negative `n` shifts earlier | vacated rows are null |
| `diff(n=1)`, `pct_change(n=1)` | `x - x.shift(n)`; `pct_change` is Float64 | null when either side is null |
| `rank(method="average", descending=False)` | `average` Float64; `min`, `max`, `dense`, `ordinal` Int64 | nulls get null; ranks start at 1; `ordinal` breaks ties by row order; NaN is the largest value, so it ranks after numbers ascending and before them descending |
| `rolling_sum/mean/min/max(window_size, min_samples=window_size)` | sum/min/max same dtype, mean Float64 | the window is the current row and the `window_size - 1` before it; nulls are skipped; fewer than `min_samples` valid values give null; a window holding NaN gives NaN for `rolling_min` and `rolling_max`, as in Polars |
| `forward_fill(limit=-1)`, `backward_fill(limit=-1)` | same dtype | fill from the nearest valid value at most `limit` rows away |
| `interpolate(method="linear"|"nearest")` | Float64 for linear, input dtype for nearest | fills only interior nulls; nearest ties choose the following value |
| `interpolate_by(by)` | Float64 | linear interpolation at the numeric `by` coordinates |
| `cut(breaks, labels=None, left_closed=False, include_breaks=False)` | String, or a breakpoint/category struct | null and NaN stay null; outer bins are open-ended |
| `qcut(quantiles, ..., allow_duplicates=False)` | as `cut` | uses linear quantiles of valid values as edges |

`expr.over(partition_by)` evaluates `expr` separately in each partition of the
key columns (one name or a list; null is its own key value) and keeps row
order: aggregates broadcast back to their partition's rows
(`col("x").sum().over("k")`), and window operations restart in each partition
(`col("x").cum_sum().over("k")`). Inside `group_by(...).agg(...)`, window
operations likewise run per group, so `col("x").cum_sum().max()` is each
group's running peak. Rolling min/max are O(n x window); other kernels are
linear apart from `rank`, which sorts each partition.

Expression row selection is available before a reduction or as a standalone
column: `col("amount").filter(col("refunded")).sum()`, `head(n)`, `tail(n)`,
`slice(offset, length=-1)`, `gather(indices)`, and scalar `get(index)`.
Null filter predicates are false, while selected null values remain in the
result and count toward `len()`. Negative offsets and indices count from
the end; out-of-bounds gather/get indices raise. Head/tail counts and
explicit slice lengths are nonnegative. Inside `group_by().agg()`, these
operations act separately within each group; unreduced selections produce
one list per group. Outside aggregation, selections that change height
cannot share `select` output with columns of a different height.

### Temporal types

`DataType.DATE` (days since 1970-01-01), `DataType.datetime(unit, time_zone)`
(ticks since the epoch), `DataType.duration(unit)`, and `DataType.TIME`
(nanoseconds since midnight) are stored as Int64, with unit `"ns"`, `"us"`
(default), or `"ms"`. The calendar is proleptic Gregorian. Sorting, grouping,
joins, `unique`, windows, `min`/`max`, and comparisons between equal types
work on the stored values.

| Operation | Result |
|---|---|
| datetime[u] - datetime[u] | duration[u] |
| datetime[u] ± duration[u], duration[u] + datetime[u] | datetime[u] |
| date - date | duration[ms] |
| date ± duration[u] | datetime[u] |
| time - time | duration[ns] |
| duration ± duration (same unit), `-`, `abs`, `sum`, `cum_sum` | duration |
| duration * Int64, Int64 * duration, duration // Int64 | duration (Int64 `//` 0 is null) |

Units must match; cast first otherwise. Arithmetic is overflow-checked.

`col(...).dt()` provides `year`, `month`, `day`, `hour`, `minute`, `second`,
`nanosecond`, `weekday` (ISO, Monday = 1), `ordinal_day`, `date()` and
`time()` of a datetime, `truncate(every)`, `offset_by(interval)`, and
`strftime(format)`; durations have `total(unit)` and `total_days()` through
`total_milliseconds()` (truncated toward zero). Intervals combine amounts and
units: `ns`, `us`, `ms`, `s`, `m`, `h`, `d`, `w`, `mo`, `y` (for example
`"1h30m"` or `"-2mo"`). `truncate` accepts one positive interval; fixed units
align to the epoch, whole weeks to Monday, and months/years to the calendar.
Month offsets clamp to the target month's last day. Date values reject
sub-day intervals. `str().strptime(dtype, format, strict)`, `to_date`, and
`to_datetime` parse text.

Formats use `%Y %m %d %H %M %S %f %j %%`; `%f` reads up to nine fractional
digits and writes the unit's precision (omitted when zero). Without a format,
dates are `YYYY-MM-DD`, datetimes accept `YYYY-MM-DD[T| ]HH:MM[:SS[.f]]`, and
times `HH:MM[:SS[.f]]`. Invalid dates such as `2023-02-29` fail. Durations
display compactly (`1d 2h 3.5s`). `date_range(start, end, interval="1d")` and
`datetime_range(start, end, interval, unit)` build inclusive sequences; month
steps are measured from the start, so `2024-01-31` stepping `1mo` gives
`2024-02-29`, `2024-03-31`, ...

Calendar reporting adds `quarter()` (1–4), ISO `week()` and `iso_year()`,
`century()`, `millennium()`, `millisecond()` and `microsecond()` (within the
second). `days_in_month()` returns the local month's length and
`is_leap_year()` returns a Boolean. `month_start()` and `month_end()` change
only the day, preserving a datetime's local time of day. `round(every)` shifts by half the interval and then truncates, as in
[Polars](https://github.com/pola-rs/polars/blob/main/crates/polars-time/src/windows/window.rs).
Fixed-unit ties round forward; month/year intervals use Polars' 28-day
estimate for each month. Zoned inputs apply this rule to local time.

`epoch(time_unit="us")` returns Int64 elapsed units since 1970-01-01 in
`"d"`, `"s"`, `"ms"`, `"us"` or `"ns"`; `timestamp(time_unit="us")` accepts
`"ms"`, `"us"` or `"ns"`. These read UTC instants, including on zoned
columns, and floor negative values. `with_time_unit(unit)` retags datetime
or duration storage without changing its ticks; `cast` rescales ticks.

The expression constructors `date(year, month, day)`,
`datetime(year, month, day, hour=0, minute=0, second=0, microsecond=0,
time_unit="us", time_zone="")`, and
`time(hour=0, minute=0, second=0, microsecond=0)` accept scalar numbers or
expressions for each field. `duration(weeks=0, days=0, hours=0, minutes=0,
seconds=0, milliseconds=0, microseconds=0, nanoseconds=0, time_unit="us")`
constructs a fixed elapsed duration. Components finer than the selected unit truncate toward zero before
summing; use `time_unit="ns"` to retain nanoseconds. Null
components produce null results; invalid dates, times and overflow raise.

```mojo
from dataframe import col, date, datetime, time, duration, lit, AnyValue, DataType

var report = df.with_columns([
    col("order_date").dt().quarter().alias("q"),
    col("order_date").dt().month_end().alias("period_end"),
    col("ts").dt().epoch("ms").alias("epoch_ms"),
]).filter(col("order_date") >= date(2024, 1, 1))

# Typed temporal literals can also use stored ticks or a nullable AnyValue.
var cutoff = lit(Int64(19723), DataType.DATE)
var same_cutoff = lit(AnyValue.temporal(DataType.DATE, 19723))
```

`dt().combine(time_expr, time_unit="us")` combines the local date of a
Date/Datetime with a Time expression. Datetime input retains its time zone;
Date input produces a naive datetime. Ambiguous or nonexistent combined
local times raise.
`dt().replace(year=..., month=..., day=..., hour=..., minute=..., second=...,
microsecond=..., ambiguous="raise")` changes only supplied local fields.
Fields accept scalars or expressions; invalid dates and nonexistent local
times raise. Ambiguous times accept `"raise"`, `"earliest"`, `"latest"` or
`"null"`.

`dt().is_business_day()` and `dt().add_business_days(n, roll="raise")` use
Monday through Friday by default. Both accept `week_mask` (seven Booleans,
Monday first) and `holidays` (ISO date strings, duplicates ignored).
`add_business_days` preserves local time and accepts an expression for `n`;
`roll="forward"` or `"backward"` first moves a nonbusiness start date to a
business date. `business_day_count(start, end, week_mask=..., holidays=...)`
counts business dates from start toward end, excluding end in either
direction. Backward counts are negative. At least one weekday must be enabled.

Additional format directives are `%b`/`%B` (English abbreviated/full month),
`%a`/`%A` (English abbreviated/full weekday), `%y` (two-digit year, parsed
in 1970–2069), `%I`/`%p` (12-hour clock and AM/PM), `%e` (space-padded day),
`%U`/`%W` (Sunday/Monday week numbers, including week 00), and `%u` (ISO
weekday). Week-number parsing requires a year and weekday; inconsistent
weekdays and dates raise. Names parse without case sensitivity.

#### Time zones

A datetime without a time zone is naive: a wall-clock reading. With one
(`DataType.datetime("us", "America/New_York")`, shown as
`datetime[us, America/New_York]`), it holds UTC instants, exactly as an Arrow
timestamp with its zone set, and the zone decides how values are shown,
parsed and split into fields. Zones are IANA names, `"UTC"`, or Arrow fixed
offsets such as `"+05:30"`. Dtypes in different zones are different types:
comparing, subtracting, joining or concatenating them raises, as in Polars.

- `dt().replace_time_zone(zone, ambiguous="raise", non_existent="raise")`
  keeps each wall-clock time and reads it in `zone` (`""` makes it naive).
  When clocks fall back a local time happens twice: `ambiguous` is
  `"raise"`, `"earliest"`, `"latest"` or `"null"`. When they spring forward
  it never happens: `non_existent` is `"raise"` or `"null"`.
- `dt().convert_time_zone(zone)` keeps the instant and changes the zone
  (a naive input is read as UTC).
- Fields, `date()`, `time()`, `strftime`, `truncate`, `round`, month boundaries,
  business dates, and calendar offsets
  (`d`, `w`, `mo`, `y`) use local time; fixed offsets (`h`, `m`, `s`, ...)
  and duration arithmetic move the UTC instant. A shifted or truncated
  local time that falls in a DST gap or overlap is resolved as Polars does
  (RFC 5545): the result keeps the original's daylight-saving state.
- Casting to string, date or time uses local time; casting between
  datetimes keeps the instant, so a zone-aware value cast to a naive
  datetime is its UTC time.
- `strptime` with `%z` in the format gives UTC-aware values; with a
  zone-aware target (`to_datetime(time_zone=...)`), text without an offset
  is local time in that zone. `%z` formats as `+HHMM`, `%:z` as `+HH:MM`,
  and `%Z` as the zone's abbreviation (`EDT`). `datetime_range(...,
  time_zone=...)` steps calendar units in local time.
- `read_csv` with `try_parse_dates=True` reads a column whose values all
  carry an offset as `datetime[us, UTC]` (a column mixing offsets with
  naive text stays a string); `CsvField.datetime(name, time_zone=...)`
  reads offset-free text as local time. `write_csv` writes the local time
  and its offset, so the instant reads back exactly (in UTC).

Zone rules come from the system time-zone database (compiled TZif files):
`$TZDIR`, then `$CONDA_PREFIX/share/zoneinfo` (the conda `tzdata` package),
then `/usr/share/zoneinfo`. Reading the system copy keeps the package small
and follows tzdata updates, so results for recently changed zones follow the
machine's tzdata version; a container without tzdata needs the conda
package or `TZDIR`. Instants after a file's last transition use its POSIX
rule. Leap-second (`right/`) files are not supported.

### Casts

`cast(dtype, strict=True)` converts between any two dtypes (a `DataType` or
its name). Integer targets are range-checked exactly (text is parsed at the
target width, never through a float); float to integer truncates toward zero
and rejects NaN and infinities. `Series.cast` and
`DataFrame.cast({"name": "dtype"})` are shorthands. Nulls stay null. A value
that cannot convert raises `cast from <a> to <b> failed at row <n> for value
'<v>': <reason>` when strict, or becomes null when `strict=False`. Inside a
`when` branch, only rows that branch can select are checked.

| From \ To | int64 | float64 | bool | string |
|---|---|---|---|---|
| int64 | identity | exact below 2^53, rounded above | `x != 0` | decimal digits |
| float64 | truncate toward zero; NaN, ±inf, and values outside [-2^63, 2^63) fail | identity | `x != 0`; NaN fails | shortest round-trip form |
| bool | 0 / 1 | 0.0 / 1.0 | identity | `true` / `false` |
| string | `read_csv` Int64 rules | `read_csv` Float64 rules | exactly `true` or `false` | identity |

Temporal casts: to and from Int64 (the stored value), to and from String
(ISO 8601), date ↔ datetime (midnight / floor to the day), datetime → time,
and between datetime or duration units (finer is exact and checked, coarser
floors).

String parsing shares `dataframe/parse.mojo` with `read_csv`, so a string casts
exactly when the same text would load from CSV: no surrounding whitespace, an
optional sign, and only explicit infinity spellings. Casting numbers and
Booleans to string and back reproduces them exactly, including `-0.0` and NaN.

Binary casts follow Polars: anything that casts to string casts to binary as
its UTF-8 bytes, and binary casts only to string, where each value must be
valid UTF-8 (strict raises; otherwise the row is null, where Polars raises).

### Binary

`DataType.BINARY` holds arbitrary bytes per row, the Arrow `large_binary`
type. It shares the string layout (one byte buffer, Int64 offsets, validity)
but its bytes are never read as text, so they need not be UTF-8. Build one
with `Series.binary(name, values, valid)` from `List[List[UInt8]]`, by
casting, or from Arrow or Parquet; `get(i).bytes()` reads a value back.

Operations that do not interpret the bytes work as they do for strings,
comparing bytes: filtering, `take`, `slice`, `concat`, sorting, `unique`,
grouping and join keys, `min`/`max`/`first`/`last`/`mode`/`n_unique`/
`count`/`value_counts`, `fill_null`, `when`, windows, and equality and
ordering comparisons (a string literal stands for its UTF-8 bytes:
`col("b") == "abc"`). Binary and string are different dtypes, so joining
or stacking one with the other raises, as in Polars; `.str()` operations
need a cast to string. Values display as `b"..."` with non-printable bytes
escaped (`b"\xff\x00z"`). `write_csv` rejects binary columns, as Polars
does, since CSV has no agreed encoding for raw bytes.

### Boolean logic and nulls

`&`, `|`, `^`, and `~` (also `and_`, `or_`, `xor`, `not_`) require Bool
operands and use three-valued Kleene logic: `false & null` is false,
`true | null` is true, and otherwise a null operand yields null. `^` and `~`
propagate nulls. Mojo cannot overload `and`/`or`/`not`, so use the operators.
`filter` keeps only true rows, so Kleene nulls are dropped.

`null(dtype)` is a typed null literal; `dtype` is a `DataType` or its name.
`is_null()`/`is_not_null()` accept any dtype and never return null. `is_nan`,
`is_not_nan`, `is_finite`, and `is_infinite` require Float64 and propagate
nulls.

`fill_null(value)` replaces nulls with a matching-dtype value or column;
`fill_nan(value)` replaces valid NaNs in Float64 input. `coalesce([a, b, ...])`
takes the first non-null value per row. `is_in([values])` is true when the input
equals any candidate; a null candidate never matches, a null input stays null,
and an empty list yields false. `is_between(lower, upper, closed="both")`
accepts `both`, `left`, `right`, or `none` and combines the two comparisons with
Kleene AND, so a null bound yields null unless the other side is false.

### Recoding values

`replace` and `replace_strict` map values to other values in one expression,
following Polars:

```mojo
df.with_columns(col("fruit").replace(["aple", "banan"], ["apple", "banana"]))
df.with_columns(
    col("state").replace_strict(
        ["CA", "NY", "TX"], ["California", "New York", "Texas"], default="Other"
    )
)
df.with_columns(col("code").replace_strict([1, 2, 3], ["low", "mid", "high"]))
```

- `old` and `new` are lists of literals: numbers, strings, `lit(...)` and
  `null(dtype)`. `new` has one value per entry of `old`, or a single value that
  every entry maps to. A scalar `old` and `new`, and a `Dict`
  (`Dict[String, String]`, `Dict[Int, Int]`, and for `replace_strict` also
  `Dict[String, Int]` and `Dict[Int, String]`), are accepted too. `old` must
  not repeat a value.
- `replace(old, new)` keeps every unmatched value and the input dtype. `old` and
  `new` are cast strictly to that dtype, so a value that does not fit raises;
  strings cast too, so `col("d").replace(["2024-01-01"], ["2000-01-01"])`
  works on a Date column.
- `replace_strict(old, new, default=None, return_dtype=None)` maps every value.
  A non-null value with no entry raises unless `default` is given. The default
  is any expression (a literal, a column, an aggregate), and with one, both
  unmatched values and null input take it. The result dtype is `return_dtype`,
  or else the type of `new` and `default` together: a string with anything is
  a string, integers widen, and a float with an integer is Float64.
- A null in `old` matches null input, so `replace([null("int64")], [lit(0)])`
  is a typed `fill_null`. Otherwise null input stays null. Values match as
  join keys do: NaN matches NaN and -0.0 matches 0.0.
- A categorical stays categorical under `replace` when the replacement is the
  whole expression. With a small dictionary next to the rows, the expression
  runs once per dictionary value and the rows take their value's result by
  code, so no string is compared per row. Inside a larger expression a
  categorical is read as its strings, as elsewhere.

The old values are cast and put in a lookup table once per expression; every
row probes it, in parallel, and the result is written in a second parallel
pass (Polars does this as a left join against the old values). Batches that a
streamed reduction feeds itself build the table per batch. Accelerators do not run these expressions and fall back to
the CPU. Polars also accepts expressions for `old` and `new`; here they must
be literals.

### Strings

`col("s").str()` returns a namespace of String operations; Mojo has no
attribute-style namespaces, so later namespaces (`dt()` and others) follow the
same method convention. Inputs must be String and nulls propagate.

| Method | Result | Notes |
|---|---|---|
| `len_chars()`, `len_bytes()` | Int64 | Unicode code points / UTF-8 bytes |
| `to_uppercase()`, `to_lowercase()` | String | Mojo's Unicode case mapping, no locale rules |
| `strip_chars(chars="")`, `strip_chars_start`, `strip_chars_end` | String | empty `chars` means ASCII whitespace; otherwise any listed code point |
| `starts_with(p)`, `ends_with(p)` | Bool | literal text |
| `contains(p, literal=False, strict=True)` | Bool | regex match anywhere; an invalid pattern raises, or gives null with `strict=False` |
| `replace(p, v, literal=False, n=1)`, `replace_all(p, v, literal=False)` | String | regex; `v` may use `$1`, `${name}`; see below for when the pattern counts as literal; an empty pattern matches at every code point boundary |
| `extract(p, group_index=1)` | String | capture group of the first match; null without a match or when the group took no part |
| `extract_all(p)` | List[String] | every non-overlapping match |
| `extract_groups(p)` | Struct | one String field per capture of the first match, named by capture name or numeric index; no match gives null fields; a null input gives a null struct; no captures gives a valid empty struct |
| `escape_regex()` | String | escape regex metacharacters in each string; nulls propagate |
| `slice(offset, length=-1)`, `head(n)`, `tail(n)` | String | by code point; a negative offset counts from the end; out-of-range parts clip |
| `reverse()` | String | by code point |
| `pad_start(w, c=" ")`, `pad_end(w, c)`, `zfill(w)` | String | pad to `w` code points with one fill character; `zfill` keeps a leading sign first |
| `to_titlecase()` | String | lowercase, then uppercase each character after a non-alphabetic one (Unicode `Alphabetic`): `"a_b 1st"` gives `"A_B 1St"`, `"ßa"` gives `"SSa"` |
| `strip_prefix(p)`, `strip_suffix(s)` | String | remove the literal once if present (not a character set) |
| `find(p, literal=False)` | UInt32 | **byte** offset of the first regex match, null if none; an empty pattern is at 0 |
| `count_matches(p, literal=False)` | UInt32 | non-overlapping regex matches; an empty pattern counts code points + 1 |
| `contains_any(patterns, ascii_case_insensitive=False)` | Bool | any literal pattern occurs |
| `replace_many(patterns, replace_with, ascii_case_insensitive=False, leftmost=False)` | String | one pass over all literal patterns; `replace_with` is one string or one per pattern |
| `encode("hex" \| "base64")` | String | UTF-8 bytes as lowercase hex or padded standard base64 |
| `decode("hex" \| "base64", strict=True)` | Binary | invalid text raises, or is null with `strict=False`; base64 must be canonical and padded |
| `normalize(form="NFC")` | String | Unicode 17.0 NFC, NFD, NFKC or NFKD; nulls propagate |
| `json_decode(dtype)` | `dtype` | decode into an explicit scalar, list or struct schema; invalid JSON and incompatible value kinds raise |
| `json_path_match(path)` | String | first match, with strings unquoted and containers compact; malformed JSON, missing matches and JSON null return null; invalid paths raise |
| `to_integer(base=10, strict=True)` | Int64 | optional sign then digits of base 2-36, as Rust's `from_str_radix`; failures raise, or are null with `strict=False` |
| `split_exact(by, n, inclusive=False)` | Struct | `n + 1` String fields `field_0`..`field_n`; extra parts are dropped, missing parts null |
| `splitn(by, n)` | Struct | `n` String fields; the last keeps the rest of the string |
| `join(delimiter="", ignore_nulls=True)` | String | an aggregation: every value (per group in `agg`) joined; nulls skipped, or with `ignore_nulls=False` a null makes the result null; no values give `""` |

`concat_str([exprs], separator="", ignore_nulls=False)` joins String
expressions row-wise and is null when any input is null. With
`ignore_nulls=True` null inputs are left out with their separator and a row
of only nulls is `""` (DuckDB's `concat_ws`). Character operations work on
code points, not grapheme clusters: `"e\u0301"` has length 2 and reverses its
combining mark.

These follow Polars 1.44. `normalize` uses Unicode 17.0 and matches the pinned
Polars 2.0 normalization and JSON oracles. Where DuckDB differs:
`strpos`/`instr` count code points from 1 and give 0 when absent (`find`
counts bytes from 0 and gives null), `split_part` gives `""` for a missing
part (null here), and `to_hex` writes uppercase digits. `splitn(by, 0)`
raises; Polars returns a struct with no fields. The `Alphabetic` table behind
`to_titlecase` is Unicode 15.0. Row kernels read offsets and string-view
storage alike; stripped affixes and split fields are views into the input's
buffers rather than copies, and a categorical input runs each function once
per dictionary value when it has few values. The accelerator backends run none of these and leave
them to the CPU.

Normalization first checks the requested form without constructing output
strings. A fully normalized chunk retains its input buffers, including Arrow
views. Changed chunks use a shared output builder and reuse decomposition
scratch across rows. Canonical ordering uses bounded insertion for short
combining runs and stable counting sort for long runs. Tables and all 20,034
conformance vectors come from the official Unicode 17.0 data under the
[Unicode License v3](../scripts/vendor/unicode-LICENSE). Regenerate them with
`pixi run python3 scripts/normalization_tables.py --ucd DIRECTORY`, where
`DIRECTORY` contains the versioned `UnicodeData.txt`,
`DerivedNormalizationProps.txt`, and `NormalizationTest.txt` from
<https://www.unicode.org/Public/17.0.0/ucd/>.

`json_decode` requires its output dtype before execution, for example
`col("json").str().json_decode(DataType.struct(["id", "tags"],
[DataType.INT64, DataType.list(DataType.STRING)]))`. It supports numeric,
Boolean, String, categorical, decimal, date, time, duration, naive datetime
and UTC datetime output, including nested lists and structs. Missing struct
fields become null and extra fields are validated then ignored. Duplicate
fields keep their first value. A non-null scalar decoded as a list becomes
a one-element list; JSON null and input null remain null. Numeric overflow
becomes null; incompatible value kinds raise with the input row. Datetime
strings without a zone keep their wall time for a naive output; UTC output
requires an explicit offset. Other output time zones and Binary/geometry
schemas raise during binding. Unpaired Unicode surrogates are rejected,
where Polars 2.0's decoder can produce NUL instead.

`json_path_match` accepts a constant path rooted at `$`: dot or quoted
fields, array indices (including negative ones), wildcards, recursive descent,
name/index unions, slices with positive steps, and comparison filters with
`&&`, `||`, `@` and `$` references. It follows the first-match ordering and
selection behavior of Polars 2.0's JSONPath evaluator. Duplicate object keys
keep their last value. Strings are returned unquoted; numbers, Booleans and
containers use compact JSON spelling. A first match of JSON null ends the
search and returns null. Unsupported syntax, including regular-expression
filters, raises before rows are evaluated. Zero or negative slice steps
raise. A reusable shared parser validates the whole input row, even when
only one field is requested. Each kernel call compiles its schema or path
once, and output builders write directly into typed buffers. Nested result
batches are concatenated once per child level, avoiding repeated copies of
earlier batches. Categorical children share one dictionary mapping for the
whole concatenation.
Wildcard paths inside predicates are also supported; Polars 2.0 can panic
for this combination, so those contracts use an independent Python JSON
selection reference. JSONPath uses Polars' serde-style floating-point
conversion and spelling, separately from typed decoding's correctly rounded
conversion.

#### Regular expressions

`contains`, `count_matches`, `extract`, `extract_all`, `extract_groups`, `replace`, `replace_all`
and `split(literal=False)` take patterns in the syntax of Rust's `regex` crate,
which Polars uses, and return what Polars returns: leftmost-first matches,
non-overlapping and searched left to right, with Rust's rule that an empty match
right after the previous match is skipped. The engine (`dataframe/regex`) is in
the RE2 / Rust family: a Thompson NFA, a lazily built DFA for match ends and a
reverse DFA for match starts, a Pike VM for capture groups, and literal
prefilters. Nothing backtracks, so every search is linear in the string. One
compiled pattern is shared by all workers; each worker has its own DFA cache.

| Syntax | Meaning |
|---|---|
| `a`, `\.`, `\n \t \r \f \v \a`, `\x7F`, `\x{1F600}`, `é`, `\U0001F600` | literals; any ASCII punctuation can be escaped |
| `.` | any code point except `\n` (any at all with `(?s)`) |
| `[abc]`, `[a-z]`, `[^a-z]`, `[a[0-9]]`, `[a-z&&[^aeiou]]`, `[a-z--b]`, `[a-c~~b-d]` | classes: ranges, negation, nesting, intersection, difference, symmetric difference |
| `[[:alpha:]]`, `[[:^digit:]]` | ASCII classes: alnum alpha ascii blank cntrl digit graph lower print punct space upper word xdigit |
| `\d \w \s`, `\D \W \S` | Unicode digits (Nd), word characters and white space, and their negations |
| `^ $`, `\A \z` | start and end of text (start and end of line with `(?m)`) |
| `\b \B`, `\< \>` | Unicode word boundary and its negation; word start / end |
| `\p{Greek}`, `\pL`, `\P{Greek}`, `\p{gc=Lu}`, `\p{scx=Hira}` | Unicode 16 properties: categories, scripts, script extensions, binary properties, age and break properties; negation with `\P` or `!=`; names ignore ASCII case, spaces, underscores and hyphens |
| `x\|y`, `(x)`, `(?:x)`, `(?P<name>x)`, `(?<name>x)` | alternation; capturing, non-capturing and named groups |
| `x* x+ x? x{n} x{n,} x{n,m}` and `x*? x+? x?? x{n,m}?` | greedy and lazy repetition |
| `(?i) (?m) (?s) (?U) (?x)`, `(?i:x)`, `(?-i)` | case-insensitive (Unicode simple case folding), multi-line, dot matches newline, swap greed, verbose |

Matching works on code points: `.` and classes consume a whole character, and
matches start and end on character boundaries. Look-around and backreferences
raise an error, as they do in Rust. `(?-u)` switches Perl classes, word
boundaries and case folding to ASCII semantics; scoped `(?-u:...)` can mix
with Unicode parts. As in Polars' UTF-8 string regex, a pattern that could
match invalid UTF-8 (such as `(?-u).`, `(?-u)\W`, or `(?-u)[^a]`) raises.
Literal Unicode characters remain valid, as do Unicode escapes like
`(?-u)\u00E9` and `(?-u)\x{E9}`; raw byte escapes above ASCII such as
`(?-u)\xE9` are rejected. CRLF mode `(?R)` is not supported.

Replacement values refer to groups as `$1`, `${1}`, `$name` or `${name}`;
`$name` takes the longest run of letters, digits and `_`, so write `${1}x` for
group 1 followed by `x`; `$$` is a dollar sign. As in Polars, a pattern without
ASCII punctuation is replaced literally and its value is not expanded, and
`literal=True` treats any pattern as text. A pattern that matches one literal
string is searched as a substring.

Regex selection uses the same engine: `col("^cost_.*$")` expands to matching
column names in schema order, as Polars does. Anchored patterns also work in
`col(["id", "^cost_.*$"])`; a list containing a regex selects the union in
schema order, while a list of exact names keeps its requested order. Operations, name prefixes/suffixes, reductions,
`with_columns` and horizontal inputs use the existing selector expansion.
An ordinary `col("name")` still requires that exact column name.
`columns("cost")` searches anywhere in each name, like DuckDB's
`COLUMNS('cost')`; use anchors for a whole-name match. This library has no
SQL text parser: `columns` is the Mojo expression equivalent. Invalid
selector patterns raise even on an empty frame.

### Horizontal reductions

`sum_horizontal`, `min_horizontal`, `max_horizontal`, `mean_horizontal`,
`any_horizontal` and `all_horizontal` take a list of expressions and reduce
across them within each row, as Polars' functions of the same names do
(DuckDB's `greatest` and `least` are `max_horizontal` and `min_horizontal`).
The result is named after the first input. Selectors among the inputs expand
into more inputs: `sum_horizontal([all()])` adds every column of a row.
Inputs may be columns, literals or other expressions, including aggregates
inside `group_by().agg`; one-row inputs broadcast.

```mojo
frame.with_columns(
    [
        sum_horizontal([col("q1"), col("q2"), col("q3")]).alias("total"),
        max_horizontal([col("shipped"), col("returned")]).alias("latest"),
        any_horizontal([by_dtype(["bool"])]).alias("flagged"),
    ]
)
```

Unlike the binary operators, these functions cast their inputs to a common
type first: Polars' supertype for integers, floats and Bool (Int8 with
UInt8 is Int16, Int32 with Float32 is Float64, Int64 with UInt64 is
Float64), and DuckDB's for decimals (an integer meeting a decimal becomes a
decimal holding its digits: Int64 with Decimal(10, 2) is Decimal(21, 2);
a decimal meeting a float is Float64). An untyped number adopts the other
inputs' type. Other combinations, such as a String with an integer, raise.

| Function | Result | Nulls |
|---|---|---|
| `sum_horizontal(exprs, ignore_nulls=True)` | common type; UInt32 for all-Bool inputs | with `ignore_nulls` a null counts as zero, so a row of nulls sums to 0; without it any null makes the row null |
| `mean_horizontal(exprs, ignore_nulls=True)` | Float32 when the common type is Float32, else Float64 | with `ignore_nulls` the mean of the row's non-null values, null when there are none; without it any null makes the row null |
| `min_horizontal(exprs)`, `max_horizontal(exprs)` | common type (numeric, Bool, String, temporal) | nulls are skipped; a row of nulls is null |
| `any_horizontal(exprs)`, `all_horizontal(exprs)` | Bool | Kleene OR and AND, as `\|` and `&` are; numeric inputs read nonzero as true |

Integer sums keep `+`'s overflow check and raise where Polars wraps. A
decimal sum has the type DuckDB gives the same chain of `+` (one more
integer digit per addition, up to 38). NaN propagates through sums and
means; `min_horizontal` and `max_horizontal` skip it, as Polars does, so
the result is NaN only when a row has nothing else. DuckDB's `greatest`
treats NaN as the largest value instead. A selector that matches no columns
leaves `any_horizontal` false and `all_horizontal` true and is an error for
the others. As filters, `any_horizontal` and `all_horizontal` keep the rows
that are true.

Each function is one expression node. Its inputs are evaluated once, cast
to the common type, and reduced pairwise as Polars does: Float sums and
means, and numeric `min`/`max`, accumulate into one output buffer a SIMD
vector at a time; integer and decimal sums fill nulls with zero and add
with the `+` kernel; `any`/`all` run the Kleene bitmap kernels. No
intermediate column is kept per pair of inputs for float sums, means or
`min`/`max`. Accelerators run none of them yet: a plan that uses one runs
on the CPU.

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
| `sum(min_count=0)` | numeric or Bool | same; 8/16-bit integers give Int64; Bool gives UInt32, the number of true values | 0, or null below `min_count` | yes (128-bit exact for integers) |
| `count()`, `null_count()`, `len()` | any | Int64 | 0 | yes |
| `min()`, `max()` | any | same | null | yes |
| `mean()` | numeric or Bool | Float64; for Bool the share of true values | null | yes (sum + count) |
| `product()` | numeric or Bool | Int64 for Bool and integers up to 32 bits; UInt64, Int64 and floats keep theirs; decimals Float64 | 1 | yes (128-bit exact for integers; raises on leaving the result type) |
| `first()`, `last()` | any | same | null | needs partition order |
| `n_unique(ignore_nulls=False)` | any | Int64 | 0, or 1 for all-null (0 when ignoring nulls) | yes (distinct set) |
| `var(ddof=1)`, `std(ddof=1)` | numeric | Float64 | null when fewer than `ddof + 1` values | yes (Welford/Chan) |
| `median()`, `quantile(q, interpolation)` | numeric | Float64 | null | no: keeps every valid value |
| `any()`, `all()` | Bool | Bool | false / true | yes |
| `arg_min()`, `arg_max()` | any | UInt32 | null | needs partition order |
| `mode()` | any | list of the input type | empty | yes (value counts) |
| `value_counts(sort=False, name="count", normalize=False)` | any | list of struct {input, UInt32 or Float64} | empty | yes (value counts) |
| `skew(bias=True)`, `kurtosis(fisher=True, bias=True)` | numeric | Float64 | null | yes (moments to the fourth) |
| `corr(a, b, method="pearson")`, `cov(a, b, ddof=1)` | two numeric | Float64 | NaN / null | Pearson and `cov` yes; Spearman keeps pairs |

`min` and `max` order strings byte-lexicographically and `false < true`. Float
`min` and `max` skip NaN, as in Polars: the result is NaN only when every valid
value is NaN (#816). DuckDB instead treats NaN as the largest value, so its
`max` is NaN if any value is. `sort`, `rank` and `top_k` still put NaN above
every number. Ties keep the first value.
`mean` of Int64 input divides the exact 128-bit total, so it cannot overflow.
A Bool sums as zeros and ones, as in Polars: `sum` and `cum_sum` count the true
values as UInt32 (Polars' index type; `count`, `len` and `null_count` stay
Int64 here), and `mean` is the share of true values among the valid ones.
Ungrouped sums count set bits 64 rows at a time without unpacking them.
`product` follows Polars' result types, but where Polars wraps an integer
product silently, this one raises, as `cum_sum` does. A decimal's product is
Float64, as DuckDB's `product` is (Polars keeps the scale of a single factor).
`first` and `last` return the first or last row's value even when it is null.
`n_unique` counts null as one value, treats every NaN as one value, and treats
`-0.0` as equal to `0.0`. Use `n_unique(ignore_nulls=True)` for SQL
`COUNT(DISTINCT x)`, which excludes nulls and returns 0 for empty or all-null
inputs. This is equivalent to Polars' `drop_nulls().n_unique()`; Polars does
not expose an `ignore_nulls` parameter. Both modes use the same distinct
counting paths, including grouped, streaming and categorical inputs.
`var` and `std` convert Int64 to Float64; infinities
produce NaN. `median` is `quantile(0.5, "linear")`. For `quantile`, `q` must be
in [0, 1] and the position is `q * (n - 1)` over sorted valid values (NaN last);
`linear` interpolates, `lower`/`higher` take the neighbouring value, `midpoint`
averages them, and `nearest` rounds the position half up. Int64 values above
2^53 lose precision in `var`, `std`, `median`, and `quantile`. Every reduction
works globally and inside `group_by(...).agg(...)`, and arithmetic on
reduction outputs (for example `col("x").max() - col("x").min()`) is allowed.

`arg_min` and `arg_max` return the index of the first minimum or maximum,
counting every row including nulls; inside `group_by(...).agg(...)` and
`over(...)` the index is within the group. Nulls are skipped, and NaN counts
only when every valid value is NaN (then the first NaN's index), as in
Polars. Polars' global `arg_max` on strings returns the last of tied indices;
this library returns the first in every context.

`mode` returns every most frequent value, ascending, with NaN above numbers
and null last; null counts as a value and `-0.0` equals `0.0`. In `select` it
yields one row per mode and cannot be combined with row-valued expressions;
in `group_by(...).agg(...)` it yields a list per group; it is not available
in `over(...)`.

`value_counts` returns each distinct value with its count as a struct whose
fields are the input's name and `name` (`"count"`, or `"proportion"` with
`normalize=True`, which gives Float64 shares of all rows, nulls included).
Counts are UInt32. Values are ascending, with NaN above numbers and null
last; `sort=True` orders by count, highest first, with ties in value order.
Polars leaves both orders unspecified. Null counts as a value and `-0.0`
equals `0.0`. Like `mode`, it yields one row per value in `select`, a list
per group in `group_by(...).agg(...)`, and is not available in `over(...)`.
`name` must differ from the input's name.

`skew` and `kurtosis` follow scipy and Polars: `bias=False` applies the
sample-size correction and gives null below three (skew) or four (kurtosis)
values; constant input gives NaN; `fisher=True` reports excess kurtosis.

`corr(a, b)` and `cov(a, b)` are functions of two columns, named after `a`.
Rows where either side is null are dropped pairwise. `corr` is Pearson's r,
or Pearson on average ranks with `method="spearman"`, and is NaN below two
pairs or for a constant side. `cov` is null with no pairs and 0.0 when the
pairs do not exceed `ddof`, as Polars returns. A NaN in any pair makes both
NaN. Polars' grouped `corr`/`cov` do not always drop null pairs; the
differential oracle compares complete pairs.

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

## Rollups, cubes and grouping sets

`group_by_rollup(keys)`, `group_by_cube(keys)` and
`group_by_grouping_sets(sets)` on `DataFrame` and `LazyFrame` are DuckDB's
`GROUP BY ROLLUP`, `CUBE` and `GROUPING SETS`. Each returns an object whose
`agg(exprs)` aggregates as `group_by(...).agg(...)` does, once per grouping
set, and returns the rows of every set one after another (a `DataFrame` from
the eager form, a `LazyFrame` from the lazy form).

```mojo
var report = sales.lazy().group_by_rollup(["region", "product"]).agg([
    col("amount").sum().alias("total"),
    grouping_id(["region", "product"]).alias("level"),
]).collect()
```

- ROLLUP `(a, b)` is the sets `(a, b)`, `(a)`, `()`. CUBE `(a, b)` is all four
  subsets. GROUPING SETS lists the sets itself; the empty set is the grand
  total. A set listed twice gives its rows twice, a key listed twice in one
  set counts once, and `rollup(a, a)` is `(a, a)`, `(a)`, `()` (all as in
  DuckDB).
- The output columns are the keys (every key any set names, in order of
  first appearance), then the aggregations. In a set's rows the keys the set
  does not group by are typed nulls.
- `grouping(name)` is 1 in the rows where `name` is aggregated away and 0
  where the row groups by it; `grouping_id([a, b, ...])` is the bit mask of
  those, the first name the highest bit (3 for `(a, b)` in the grand total).
  Both are Int64 and usable in `agg` only. They tell a subtotal's null from a
  null key value: the rows `(x, null)` with `grouping_id` 0 and 1 are
  different groups.
- The empty set yields one row even over no rows (a sum there is 0, as `sum`
  of no rows is everywhere in this library; DuckDB's is null); every other
  set yields no rows over no rows.
- Row order is unspecified unless the result is sorted.
- Known limits: `grouping` and `grouping_id` must be a whole aggregation
  expression (`grouping(a) + 1` raises), and when one is present every other
  aggregation must produce a single column named by its output name (a
  selector such as `all().sum()` next to `grouping` is not supported).

The plan is `group_by(set).agg(...)` per set, each with its missing keys
added as null columns, joined by `concat`. Every set reads the input, so the
optimizer's pushdowns, common-subplan sharing and the accelerator see the
same operators they see for hand-written queries. Aggregates for coarser sets
are not yet merged from finer ones.

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
group.

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

Float64 elementwise subtrees are fused: the binder marks Float64 columns and
literals combined with `+ - * /`, with at most one comparison at the root, and
the evaluator runs each such subtree as one small register program over SIMD
vectors loaded directly from the source column buffers. No inner node
materializes a column and source values are not copied into batch slices.
Validity is the AND of the leaf columns' validity, which is exactly how nulls
propagate through those operations, and results are bit-identical to the
unfused kernels (`bind(expr, columns, fuse=False)`), which tests compare across
special values, nulls, SIMD widths, and batch sizes. Integer arithmetic is not
fused, because its checked overflow and branch masks need the per-row kernels;
reductions, windows, strings, and conditionals are fusion boundaries.

For one Int64 grouping key and three or more simple Float64
`sum`, `count`, or `mean` expressions, the group executor uses one
range-local pass for all aggregates, then merges worker states. Contiguous
columns and columns with matching chunk boundaries use this path without
rechunking. A sample limits
the path to bounded cardinality; if a worker discovers more groups than its
state budget permits, execution falls back to the standard grouped reducer.
Other grouped aggregate shapes still scan separately for each aggregate.
Outside fused subtrees the evaluator still allocates/copies intermediate batch
columns and keeps node outputs until the batch ends.
SIMD loads/stores are assembled by lane rather than a zero-copy buffer view.
There is no shared-expression elimination, parallel scheduler, or measured end-to-end speedup claim yet. Those are execution
improvements to pursue with benchmarks, not reasons to change expression semantics.


## List and struct expressions

`col(x).str().split(by, inclusive=False, literal=True, strict=True)` turns a string column into
`list[string]`: an empty string gives one empty part, a separator at either
end gives an empty part there, and a null string gives a null list. With
`inclusive=True` each part but the last keeps the separator. With
`literal=False` the separator is a regular expression, split as Rust's
`Regex::split` does; an empty pattern splits into code points.
`str().extract_all(p)` also gives `list[string]`.

The `.list()` namespace works on any list column. `len()` counts elements
(Int64); `get(i)` returns element `i`, counting from the end when negative and
null when out of range; `first()` and `last()` are `get(0)` and `get(-1)`;
`contains(value)` tests a string, integer or float against the list's element
type; `join(separator)` joins a string list, skipping null elements; `sum`,
`min`, `max` and `mean` reduce numeric lists (`sum` widens 8- and 16-bit
integers to Int64, `mean` is Float64, and an empty or all-null list gives
null except for `sum`, which gives 0). A null list gives a null result.

`col(s).field(name)` reads one field of a struct column; rows where the
struct itself is null are null. `DataFrame.pack_struct(name, columns)` builds
a struct column from existing columns and `unnest(name)` puts its fields back
as top-level columns, in the struct's position.

`as_struct([exprs], name="")` packs expressions into one struct column; each
field takes its expression's output name, scalars broadcast, and the result
is named after the first field unless `name` is given. `col(x).implode()` is
a reduction whose result is a list: inside `agg` it gathers each group's
values in input order (nulls included), and on its own it gathers the whole
column into one row. `group_by(...).agg(col(x).implode())` followed by
`explode(x)` gives the rows back.

Struct columns work as `group_by`, `unique`/`n_unique`/`is_duplicated` and
inner/left/semi/anti join keys. They are compared field by field together
with the struct's own validity, so a null struct forms a group of its own,
distinct from a struct whose fields are all null. Right and full joins on
struct keys, struct keys with nested fields, and list keys raise.

`DataFrame.explode(columns)` and `LazyFrame.explode(columns)` produce one row
per list element and repeat the other columns; an empty or null list gives
one row holding null, and several columns explode together only when every
row has the same element count in each.

Nested columns cannot yet be sort keys, be cast, be reduced (other than
`implode`), be used in `when`/`then`, or be written to CSV; each raises an
error that says so.

See [expressions within groups](group-expressions.md) for filtered and ordered
aggregates, row selection, distinct/top-k operations and `GroupBy.head/tail`.
