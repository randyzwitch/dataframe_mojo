# dataframe_mojo

An experimental, native CPU dataframe library for **Mojo 1.1.0**. Import it as
`dataframe`. Storage and computation use Mojo and its standard library; there
is no Python, pandas, Polars, or Arrow runtime dependency.

This is a working first implementation, not a production engine or a portable
backend facade. The API is provisional.

## Run

With [Pixi](https://pixi.sh) installed, from this directory:

```bash
pixi run test          # runs every tests/test_*.mojo
pixi run example
pixi run build
pixi run bench          # CPU benchmark suite; see docs/benchmarks.md
pixi run bench-csv
./build/sales
```

Supported platforms are Linux x86-64 and macOS arm64 (both tested in CI) and
Linux aarch64 (resolved in `pixi.lock`, not yet tested in CI). Windows waits on
Mojo support. The library requires a little-endian target and checks this at
compile time.
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
    print(result)
```

```text
shape: (2, 3)
┌────────┬─────────┬───────┐
│ region ┆ revenue ┆ sales │
│ ---    ┆ ---     ┆ ---   │
│ str    ┆ f64     ┆ i64   │
╞════════╪═════════╪═══════╡
│ east   ┆ 90.0    ┆ 1     │
│ west   ┆ 180.0   ┆ 2     │
└────────┴─────────┴───────┘
```

Run `pixi run example` to see the sales, expression, and CSV pipelines. `build/expressions` and `build/read_csv` are the
compiled examples.

## CSV ingestion

`read_csv` reads local UTF-8 CSV files without Python or another dataframe
runtime. `read_csv(path)` infers a schema from a sample (identifiers such as
`007` and integers beyond Int64 stay strings); an explicit schema pins every
type:

```mojo
from dataframe import CsvField, CsvSchema, read_csv

var frame = read_csv(
    "sales.csv",
    CsvSchema([
        CsvField.string("region", False),
        CsvField.float64("amount"),
    ]),
)
```

It accepts LF or CRLF records, an optional UTF-8 BOM, quoted commas/newlines,
and doubled quotes. Empty unquoted fields are null; quoted empty strings are
values. Headers must exactly match the schema. Whitespace is preserved and
typed fields do not trim it. Boolean values are exactly `true` or `false`.
Float64 accepts the standard parser's values, including `nan`, and only explicit
`inf`, `+inf`, `-inf`, `Infinity`, `+Infinity`, and `-Infinity` may be infinite.

`write_csv(frame, path)` writes the inverse format: reading it back with
`CsvSchema.of(frame)` reproduces the frame exactly.

The reader consumes bounded file buffers and retains tokenizer state across
them. Its scalar structural scanner is the correctness reference for later SIMD
scanning and record-boundary-aware parallel decoding. See the complete
[CSV contract](docs/csv.md).

Use typed literals: there is no implicit Int64/Float64 promotion. Expressions
support `+ - * / // % **`, comparisons `< <= > >=` plus `.eq()`/`.ne()`, unary
`-`, `abs`, `sqrt`, `exp`, `log`, `floor`, `ceil`, `round`, `clip`, `.alias()`,
Kleene `& | ^ ~`, `is_null`, `is_nan`, `fill_null`, `fill_nan`, `coalesce`,
selectors (`all`, `col([...])`, `exclude`, `by_dtype`, `nth`), window
operations (`cum_sum`, `shift`, `diff`, `rank`, `rolling_*`, `forward_fill`,
`over`), `is_in`,
`is_between`, `cast`, `when(...).then(...).otherwise(...)`, a `.str()`
namespace with `concat_str`, and the reductions `sum(min_count=0)`, `count`,
`null_count`, `len`, `min`, `max`, `mean`, `first`, `last`, `n_unique`, `var`,
`std`, `median`, `quantile`, `any`, and `all`. Equality is `.eq()` rather than
Python-style `==`. See the operator table in [expressions](docs/expressions.md).

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
count is below `min_count`, following Narwhals' default. Integer sums accumulate exactly in
128 bits and check the final Int64 result, allowing parallel partial-state merging.
Floating-point expression sums permit reassociation; bitwise reproducibility is
not guaranteed. See [the expression contract](docs/expressions.md).

## Implemented

- Typed `Column[T]` with bit-packed validity and checked element access.
- Heterogeneous `Series` and runtime-schema `DataFrame`: Int64, Float64, Bool,
  and String. The runtime tag is per column, not per cell.
- Schema inspection, projection, indexed gathering, nullable Boolean filtering,
  and adding/replacing columns.
- Vertical, diagonal, and horizontal `concat`, plus `vstack`/`hstack`.
- `unique`, `n_unique`, `is_duplicated`, `drop_nulls`, and frame `fill_null`.
- `pivot` and `unpivot` reshaping.
- A `Series` API (operators, reductions, `value_counts`, `unique`, `sort`)
  backed by the same expression kernels.
- Lazy queries (`frame.lazy()`, `scan_csv`) with predicate, projection, and
  slice pushdown and `explain()`; see [lazy queries](docs/lazy.md).
- Bounded table display for `DataFrame` and `Series` (`print(frame)`,
  `to_string(max_rows=..., ...)`, `glimpse()`).
- `head`/`tail`/`slice`/`reverse`, `drop`/`rename`/`with_row_index`, row and
  cell access through the tagged `AnyValue`, null counts, and structural `equals`.
- Hash grouping by any number of columns or key expressions of any dtype.
- Stable multi-column sorting with per-column direction and null placement,
  plus `arg_sort`, `top_k`, and `bottom_k`.
- Hash joins on any number of keys of any dtype: inner, left, right, full,
  semi, anti, and cross, with `left_on`/`right_on`.
- Expression IR, schema binding, scalar broadcasting, and grouped aggregates.
- Explicit SIMD Float64 arithmetic/comparison kernels and mergeable reduction states.
- Contract tests for validation, nulls, overflow, ordering, joins, empty shapes,
  and scalar/SIMD agreement across batch boundaries.

Useful calls:

```mojo
var selected = frame.select(["region", "amount"])
var first = frame.head(3)
var cleaned = frame.drop("notes").rename({"amount": "revenue"})
var cell = frame.item(0, "region").string()
var sampled = frame.take([3, 0, 3])
var sorted = frame.sort(["region", "amount"], descending=[False, True], nulls_last=[True, True])
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

There is no thread scheduler, Parquet reader, GPU execution, index
alignment, implicit dtype coercion, or dataframe backend
adapter. Build input columns in memory for this first version.

## Using the package

`pixi run package` precompiles `dist/dataframe.mojoc`; put its directory on the
import path (`mojo run -I dist app.mojo`). Tagged releases attach the package
and a generated [API reference](docs/api.md) (`pixi run docs`). See the
[testing guide](docs/testing.md), the [changelog](CHANGELOG.md), the [stability policy](docs/stability.md), and the
[pandas/Polars migration guide](docs/migration.md).

See [semantics](docs/semantics.md) for the behavior the tests promise and
[architecture and next steps](docs/architecture.md) for the intended progression.
