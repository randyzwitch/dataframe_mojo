# Window functions

`Expr.over` evaluates an expression independently within each partition.
By default, it maps its result to the original rows. A scalar aggregate is
broadcast to every row of its partition; a row-sized result follows the partition's rows.

```mojo
from dataframe import col

col("sales").sum().over("region")
col("sales").mean().over([String("region"), "year"])
col("sales").cum_sum().over("region", order_by="date")
col("sales").cum_sum().over(order_by=[String("date"), "id"])
```

With the default mapping, an ordered window sorts each partition, evaluates
the expression, and restores the original input row order. Equal ordering
keys keep input order; add an explicit tie-breaker such as `id` for reproducible results across differently
ordered inputs. `descending=True` reverses ordering and `nulls_last=True`
places null ordering keys last. Both options apply to every ordering key.
With only `order_by`, the entire frame is one partition. With only partition
keys, order-dependent expressions use input order. Calling `over()` without
partition or ordering keys raises.

Ordering does not add SQL's default peer-based `RANGE` frame. A reduction such
as `sum().over(...)` reads the entire partition, even when `order_by` is
present. `cum_sum().over(..., order_by=...)` corresponds to SQL
`ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`: tied ordering values are
processed as separate rows. `rolling_sum(k)` uses the current row and the
preceding `k - 1` rows; see [time windows](time-windows.md) for rolling moments,
duration windows and time grouping. Arbitrary SQL frame bounds and exclusion
clauses are not accepted by `over`.

## Mapping partition results

Every `over` overload accepts `mapping_strategy`, defaulting to
`"group_to_rows"`:

| Strategy | Result |
| --- | --- |
| `"group_to_rows"` | Broadcast scalar results; scatter row-sized vectors to the original rows. Other vector lengths raise. |
| `"join"` | Collect a vector result as a list and broadcast that list to each original partition row. Scalar and aggregate results remain scalar. |
| `"explode"` | Emit each partition's result directly, in partition order. The output height can change. |

```mojo
col("sales").head(2).over("region", order_by="date", mapping_strategy="join")
col("sales").head(2).over("region", order_by="date", mapping_strategy="explode")
```

Partitions appear in the order their keys first occur in the input. With
`explode`, each partition's rows follow `order_by` when supplied; results are
not scattered back to input positions. Select expressions with compatible
output heights: mixing an exploded result with an original source column of
a different height raises.

## Row counts and row numbers

The free `len()` expression counts every row, including null rows, without
requiring a source column. Alias it on import to distinguish it from the
built-in `len` function. `Expr.len()` remains available when an expression is
already at hand; `Expr.count()` counts only non-null values.

```mojo
from dataframe import int_range, len as row_len, lit

row_len().over("region")
int_range(row_len()).over("region", order_by="date")  # zero-based row number
(int_range(row_len()).over("region", order_by="date") + lit(Int64(1)))
```

The final expression gives SQL's one-based `ROW_NUMBER`, including rows where
all source values are null. `int_range(end, step=1)` starts at zero;
`int_range(start, end, step=1)` accepts an explicit start. Both bounds are
expressions that must each evaluate to one non-null integer scalar or
reduction. The end is exclusive, negative steps are supported, and a zero
step raises. Results are Int64.
Inside `over`, bounds such as `row_len()` are evaluated per partition.

## Lag, lead and partition values

```mojo
col("sales").shift(1).over("region", order_by="date")   # LAG
col("sales").shift(-1).over("region", order_by="date")  # LEAD
col("sales").shift(1, fill_value=0).over("region", order_by="date")
col("sales").shift(1, ignore_nulls=True).over("region", order_by="date")
col("sales").shift(-1, ignore_nulls=True).over("region", order_by="date")
col("sales").first().over("region", order_by="date")
col("sales").last().over("region", order_by="date")
```

A shift never crosses partition boundaries. Its default result for a vacated
position is null. `fill_value` replaces only vacated positions, so a shifted
null stays null. `first()` and `last()` include nulls and read the whole
partition; in SQL terms their frame is
`ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING`.

`shift` accepts `ignore_nulls=True` with either sign and with every
`fill_value` overload. Positive offsets select previous non-null source
values; negative offsets select following non-null source values. An offset
of zero preserves the current value, including null, and `fill_value` does
not replace that null.

For the first or last non-null value, pass `ignore_nulls=True` to the
reduction, or remove nulls before reducing. For whole-partition `NTH_VALUE`,
collect the ordered values and select the zero-based list index. An index outside the partition produces null.

```mojo
col("sales").first(ignore_nulls=True).over("region", order_by="date")
col("sales").last(ignore_nulls=True).over("region", order_by="date")
# Equivalent first-value recipe using the existing null filter:
col("sales").drop_nulls().first().over("region", order_by="date")
# SQL NTH_VALUE(sales, 3), with a whole-partition frame:
col("sales").implode().list().get(2).over("region", order_by="date")
# NTH_VALUE(sales, 3) IGNORE NULLS, with that same frame:
col("sales").drop_nulls().implode().list().get(2).over("region", order_by="date")
```

SQL `NTH_VALUE` uses a positive, one-based argument `n`; pass `n - 1` to
`list().get`. The first/last non-null recipes return null for all-null
partitions.

## Ranking recipes

`rank("min")` gives SQL `RANK` for non-null values, `rank("dense")` gives
`DENSE_RANK` for non-null values, and
`rank("ordinal")` assigns distinct positions, breaking ties by input order.
`rank("max")` gives the last position of each peer group and
`rank("average")` gives its mean position. Ranks start at one. Average ranks
are Float64; the other methods return Int64. Native rank skips null values
and leaves their ranks null. The following SQL-compatible recipes count null
rows and treat them as a peer group.

As an alternative to `int_range(row_len())`, count a non-null Boolean
indicator to obtain one-based row numbers that include null values:

```mojo
col("sales").is_null().cum_count().over("region", order_by="date")
```

For `PERCENT_RANK` and `CUME_DIST`, use the first and last peer positions,
respectively. This example uses partition column `g` and ranked column `x`.
`descending` controls value ranking independently of null placement.

```mojo
from dataframe import Expr, col, lit, when


def sql_rank(method: String, descending: Bool, nulls_first: Bool) -> Expr:
    # Call with "min" for RANK or "max" for the last peer position.
    var ranked = col("x").rank(method, descending=descending).over("g")
    var missing = col("x").is_null()
    var nulls = missing.cast("int64").sum().over("g")
    if nulls_first:
        var null_rank = nulls.copy() if method == "max" else lit(Int64(1))
        return when(missing).then(null_rank).otherwise(ranked + nulls)
    var null_rank = col("x").len().over("g") if method == "max" else col("x").count().over("g") + lit(Int64(1))
    return when(missing).then(null_rank).otherwise(ranked)


def percent_rank(descending: Bool, nulls_first: Bool) -> Expr:
    var n = col("x").len().over("g")
    var one = lit(Int64(1))
    var denominator = when(n > one).then(n - one).otherwise(one)
    return (sql_rank("min", descending, nulls_first) - one).cast("float64") / denominator.cast("float64")


def cume_dist(descending: Bool, nulls_first: Bool) -> Expr:
    return sql_rank("max", descending, nulls_first).cast("float64") / col("x").len().over("g").cast("float64")
```

`len()` includes null rows and `count()` counts non-null values. Cast the
Boolean null indicator to Int64 before `sum()` to keep rank arithmetic in
Int64. Cast both division operands to Float64 for fractional results.
`PERCENT_RANK` is zero for a singleton partition; `CUME_DIST` is one. In an
all-null partition every row is a peer: the results are zero and one,
respectively, with either null placement and either sort direction.

SQL `NTILE(buckets)` divides rows as evenly as possible, giving the first
remainder buckets one extra row. Require a positive bucket count. With more
buckets than rows, each row receives a distinct bucket starting at one.

```mojo
def ntile(buckets: Int) raises -> Expr:
    if buckets < 1:
        raise Error("bucket count must be positive")
    var one = lit(Int64(1))
    var zero = lit(Int64(0))
    var n = col("x").len().over("g")
    var row = col("x").is_null().cum_count().over("g", order_by="t") - one
    var q = n // lit(Int64(buckets))
    var extra = n % lit(Int64(buckets))
    var boundary = (q + one) * extra
    # Guard the unused branch when buckets > n, where q is zero.
    var safe_q = when(q > zero).then(q).otherwise(one)
    return when(row < boundary).then(row // (q + one) + one).otherwise((row - boundary) // safe_q + extra + one)
```

Here `t` defines row order, including a tie-breaker when necessary. Six rows
and four buckets produce bucket sizes `2, 2, 1, 1`; a formula that simply
multiplies the row fraction by the bucket count gives different boundaries.
All these expressions restore results to the original row positions.

Run `pixi run test --contracts expr.ordered_over`. The permanent ranking
recipe contracts compare literal expected values independently checked against
SQL, covering tied values, shuffled partitions, ascending and descending
ranks, both null placements, singleton and all-null groups, bucket counts
larger than partitions, empty typed frames, eager execution, and lazy execution
with optimization and streaming independently enabled or disabled.
