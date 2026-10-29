# Expressions within groups

Expression row operations work on a whole column in `select` and separately
within each group in `group_by().agg()`. A following reduction returns one
value per group; an unreduced sequence is collected into a list per group.

```mojo
var result = events.group_by("customer_id").agg([
    col("status").sort_by("ts").last().alias("latest_status"),
    col("amount").filter(col("kind") == "refund").sum().alias("refunded"),
    col("product").top_k_by("revenue", 3).alias("top_products"),
])
```

| Operation | Behavior |
| --- | --- |
| `filter(predicate)` | Keep true positions; null predicates are false. Selected null values stay in the result. |
| `sort(descending=False, nulls_last=False)` | Stable value order, with the frame sort's NaN/null rules. |
| `sort_by(by, descending=False, nulls_last=False)` | Stable ordering by a name, expression, or list of names/expressions. The flags apply to every ordering key. |
| `head(n=5)`, `tail(n=5)`, `slice(offset, length=-1)` | Select a range; negative offsets count from the end, and `-1` length means the remainder. |
| `gather(indices)` | Select literal or expression-computed integer positions, retaining repetitions. Negative positions count from the end; null indices produce null values. |
| `get(index)` | Select one scalar value; negative indices count from the end. |
| `unique(maintain_order=False)` | Distinct values, including one null; `True` retains first appearance. |
| `unique_counts()` | Int64 counts in first-appearance order, including null as a value. |
| `top_k(k=5)`, `bottom_k(k=5)` | Largest/smallest values, preferring non-null values; stable rank order. |
| `top_k_by(by, k=5)`, `bottom_k_by(by, k=5)` | Select values by ordering keys, with the same key forms as `sort_by`. |
| `max_by(by)`, `min_by(by)` | Scalar value at the largest/smallest non-null key; first input row wins ties. All-null/empty keys yield a typed null. |
| `arg_sort(descending=False, nulls_last=False)`, `reverse()` | Int64 sorted positions or reversed value order. Positions are local to each group. |

`unique().sum()` implements a distinct sum. `unique().count()` skips the
selected null, while `n_unique()` counts null as a value. `len()` after a
filter counts selected rows, including null values.

Out-of-bounds gather/get indices raise. Head/tail and top/bottom-k counts
must be nonnegative. Length-changing results cannot share `select` output
with columns of a different height. Ordering keys must align with their
values; when selecting rows first, apply the same selection to the keys.

`GroupBy.head(n)` and `GroupBy.tail(n)` keep every original column. Groups
come out in first-appearance order and selected rows keep their input order
within each group. Eager and lazy grouping support computed keys and retain
the original payload columns even when a key alias has the same name.

Top/bottom-k currently use a full stable ordering. These APIs make no
performance claim for large groups. Counts and positions use this library's
Int64 convention. Reference semantics for appearance-ordered counts and
non-null-preferred top-k are documented by
[Polars unique_counts](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.unique_counts.html)
and [Polars top_k](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.top_k.html).
