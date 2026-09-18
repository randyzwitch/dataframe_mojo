# dataframe_mojo

An experimental, native CPU dataframe library for **Mojo 1.1.0**. Import it as
`dataframe`. Storage and computation use Mojo and its standard library; there
is no Python, pandas, Polars, or Arrow runtime dependency.

This is a working first implementation, not a production engine or a portable
backend facade. The API is provisional.

## Run

With [Pixi](https://pixi.sh) installed, from this directory:

```bash
pixi run test
pixi run example
pixi run build
./build/sales
```

The manifest currently targets Linux x86-64, the platform tested locally.
The original sales example prints:

```text
east 90.0
west 180.0
```

## Expressions

Expressions describe work without executing it. `select`, `with_columns`,
`filter`, and `group_by(...).agg(...)` bind them against the input schema before
running any expression kernel. The eager evaluator operates on bounded batches.

```mojo
from dataframe import Column, DataFrame, Series, col, lit


def main() raises:
    var sales = DataFrame([
        Series("region", Column[String](["east", "west", "east", "west", "north"])),
        Series("amount", Column[Float64](
            [100, 80, -20, 120, 999],
            [True, True, True, True, False],
        )),
    ])
    var result = sales.filter(
        col("amount") > lit(Float64(0))
    ).with_columns(
        (col("amount") * lit(Float64(0.9))).alias("net")
    ).group_by("region", maintain_order=True).agg([
        col("net").sum().alias("revenue"),
        col("net").count().alias("sales"),
    ])
```

Run `pixi run example` to see both the original column-kernel example and the
expression pipeline. `build/expressions` is the compiled expression example.

Use typed literals: there is no implicit Int64/Float64 promotion. Expressions
support `+`, `-`, `*`, `>`, `.eq(...)`, `.alias(...)`, `.sum(min_count=0)`, and
`.count()`. Equality is `.eq()` rather than Python-style `==`.

`select(expr)` selects one expression; `select_exprs([expr, ...])` selects several.
`select(["name", ...])` remains the name-only projection API, with no ambiguous
empty-list overload. `with_columns` and `agg` accept an expression or a list.
Sibling expressions all see the original input, not each other's aliases.

Expressions distinguish scalar, row-valued, and aggregate results. For example:

```mojo
var totals = sales.select(col("amount").sum())  # one row
var repeated = sales.with_columns(col("amount").sum().alias("total"))
var deviations = sales.select(col("amount") - col("amount").sum())
```

Expression sums return zero for empty/all-null input, or null when the valid
count is below `min_count`. This follows Narwhals' default and differs from the
older `sum_int64`, `sum_float64`, and `group_by_sum` helpers, which retain their
original null-for-empty behavior. Integer expression sums accumulate exactly in
128 bits and check the final Int64 result, allowing parallel partial-state merging.
Floating-point expression sums permit reassociation; bitwise reproducibility is
not guaranteed. See [the expression contract](docs/expressions.md).

## Implemented

- Typed `Column[T]` with bit-packed validity and checked element access.
- Heterogeneous `Series` and runtime-schema `DataFrame`: Int64, Float64, Bool,
  and String. The runtime tag is per column, not per cell.
- Schema inspection, projection, indexed gathering, nullable Boolean filtering,
  and adding/replacing columns.
- Nullable greater-than comparisons, Float64 scalar multiplication, and
  Int64/Float64 sums.
- Hash grouping by one String column, summing one Int64 or Float64 column.
- Stable single-column sorting for all four dataframe types.
- Inner/left hash joins on one String key, including many-to-many matches.
- Expression IR, schema binding, scalar broadcasting, and grouped aggregates.
- Explicit SIMD Float64 arithmetic/comparison kernels and mergeable reduction states.
- Contract tests for validation, nulls, overflow, ordering, joins, empty shapes,
  and scalar/SIMD agreement across batch boundaries.

Useful calls:

```mojo
var selected = frame.select(["region", "amount"])
var sampled = frame.take([3, 0, 3])
var sorted = frame.sort("amount", descending=True, nulls_last=True)
var joined = frame.join(regions, on="region", how="left")
```

Operations return new dataframes. `column()` and typed extraction methods
(`int64()`, `float64()`, `bool()`, `string()`) return owned copies.
An empty dataframe can retain its row count: `DataFrame([], height=10)`.

## Deliberate limits

Execution is eager and currently single-threaded. Float64 expression arithmetic
and comparisons use SIMD; checked integer arithmetic and reductions are currently
scalar. Dataframe transformations and public column extraction copy buffers.
Expression evaluation copies bounded input slices, rather than extracting whole
columns per operation; intermediates are still materialized per batch. Strings use `List[String]`, not Arrow UTF-8
buffers. There are no performance claims yet, and there is no Arrow export.
The underscore-prefixed fields are internal and must not be mutated by callers.

There is no lazy query planner, thread scheduler, CSV/Parquet reader, GPU execution, index
alignment, implicit dtype coercion, multi-key grouping/join, or dataframe backend
adapter. Build input columns in memory for this first version.

See [semantics](docs/semantics.md) for the behavior the tests promise and
[architecture and next steps](docs/architecture.md) for the intended progression.
