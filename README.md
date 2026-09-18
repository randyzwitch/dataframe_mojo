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
The example prints:

```text
east 90.0
west 180.0
```

## Example

```mojo
from dataframe import Column, DataFrame, Series, greater_than, multiply


def main() raises:
    var sales = DataFrame([
        Series("region", Column[String](["east", "west", "east", "west", "north"])),
        Series("amount", Column[Float64](
            [100, 80, -20, 120, 999],
            [True, True, True, True, False],
        )),
    ])
    var positive = sales.filter(
        greater_than(sales.column("amount").float64(), Float64(0))
    )
    var result = positive.with_column(
        Series("net", multiply(positive.column("amount").float64(), 0.9))
    ).group_by_sum("region", "net", "revenue")

    var regions = result.column("region").string()
    var revenue = result.column("revenue").float64()
    for i in range(result.height()):
        print(regions.value(i), revenue.value(i))
```

The null payload `999` is ignored, not interpreted as a value or a sentinel.
Integer columns retain Int64 precision, including values above 2^53.

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
- Contract tests for validation, nulls, overflow, ordering, joins, and empty shapes.

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

Execution is eager, single-threaded, and scalar. Dataframe transformations and
column extraction copy buffers. Strings use `List[String]`, not Arrow UTF-8
buffers. There are no performance claims yet, and there is no Arrow export.
The underscore-prefixed fields are internal and must not be mutated by callers.

There is no lazy expression system, CSV/Parquet reader, GPU execution, index
alignment, implicit dtype coercion, multi-key grouping/join, or dataframe backend
adapter. Build input columns in memory for this first version.

See [semantics](docs/semantics.md) for the behavior the tests promise and
[architecture and next steps](docs/architecture.md) for the intended progression.
