# dataframe_mojo

Filter, join, and summarize tabular data directly in Mojo. Import `dataframe` to
work with typed, nullable columns in memory or to build a query over local
files. If you know Polars, expressions like `col("amount").sum()` will feel
familiar. The query runs in your Mojo program without a Python runtime.

## Try it

Start with the [v0.3.0 release](https://github.com/randyzwitch/dataframe_mojo/releases/tag/v0.3.0).
Its lockfile installs Mojo 1.2.0 and the matching MAX SDK. The examples below
run from a release checkout using [Pixi](https://pixi.sh/).

```bash
git clone --branch v0.3.0 --depth 1 https://github.com/randyzwitch/dataframe_mojo.git
cd dataframe_mojo
pixi install --locked
```

Suppose a sales report should exclude refunds and show gross sales by region.
Save this as `sales.mojo` in the checkout and run `pixi run mojo run -I . sales.mojo`.

```mojo
from dataframe import Column, DataFrame, Series, col


def main() raises:
    var sales = DataFrame([
        Series("region", Column[String](["east", "west", "east", "west"])),
        Series("amount", Column[Int64]([100, 80, -20, 120])),
    ])
    var totals = (
        sales.filter(col("amount") > 0)
        .group_by("region", maintain_order=True)
        .agg(col("amount").sum().alias("gross_sales"))
    )
    print(totals)
```

| region | gross_sales |
| --- | ---: |
| east | 100 |
| west | 200 |

The filter removes the `-20` refund before aggregation. The region that
appears first in the input appears first in the result because
`maintain_order=True`.

## Query a file

Use `scan_csv` when you want to describe the whole query before reading the
result. This example finds the two largest sales in the
[included CSV](examples/sales.csv). Save it as `top_sales.mojo` and run
`pixi run mojo run -I . top_sales.mojo` from the same directory.

```mojo
from dataframe import col, scan_csv


def main() raises:
    var largest = (
        scan_csv("examples/sales.csv")
        .filter(col("amount") > 0)
        .select(["region", "amount"])
        .sort("amount", descending=True)
        .head(2)
    )
    print(largest.explain())
    print(largest.collect())
```

`explain()` shows the optimized plan. `collect()` runs it and returns a
`DataFrame` with `west, 120` followed by `east, 100`. For a program that needs
the entire CSV immediately, `read_csv(path)` returns a frame in one call.

## What you can build on main

| Task | What the library offers |
| --- | --- |
| Analyze tables | Use expressions to derive columns, combine tables with joins, and aggregate by one or more keys. Calculate ranks within partitions, roll up totals, or match a timestamp to a preceding observation. [Expressions](docs/expressions.md) · [Grouped aggregations](docs/group-expressions.md) · [Window functions](docs/window-functions.md) · [As-of joins](docs/join-asof.md) |
| Query a local dataset | Scan CSV or Parquet across a glob, select the fields you need, and filter on Hive partition values. The optimizer can skip whole files and prune Parquet row groups. [Multiple files and partitions](docs/datasets.md) · [Lazy queries](docs/lazy.md) |
| Keep domain types intact | Represent money with decimals, timestamps with time zones, and nested records with list and struct columns. Nulls have their own validity bits, so a missing float stays distinct from NaN. [Types and semantics](docs/semantics.md) · [Time windows](docs/time-windows.md) |
| Exchange data with other tools | Share compatible column buffers through the Arrow C Data Interface. For files, read and write Parquet, Arrow IPC/Feather, and newline-delimited JSON. [Arrow interchange](docs/arrow.md) · [Parquet](docs/parquet.md) · [IPC](docs/ipc.md) · [JSON](docs/json.md) |
| Work with geographic data | Read features from GeoJSON or a shapefile, filter on their attributes, and write GeoParquet with coordinate reference system metadata. [Geospatial guide](docs/geospatial.md) · [Shapefiles](docs/shapefile.md) |

**Execution and memory.** The CPU executor can push filters, projections, and
limits into supported scans. Some pipelines stream batches. Sorts, join build
tables, group state, and collected results can still require substantial RAM.
On Apple silicon, the Metal provider can run eligible reductions, including
some filtered and grouped aggregations. Automatic GPU selection requires
matching calibration data. [Lazy execution](docs/lazy.md) · [Apple Metal](docs/apple-metal.md)

**File dependencies.** The examples above use plain CSV and need only the
locked environment. Parquet, IPC, and compressed CSV on main use the optional
Arrow C++ library `libdfparquet`. With a C++20 compiler available, build it
from the repository root with `pixi run -e native build-dfparquet`.
[CSV options](docs/csv.md) · [Parquet setup](docs/parquet.md)

## Use it in your own Mojo project

The v0.3.0 Pixi Git dependency omits a required provider. From the tagged
checkout, run `pixi run package` to build the dataframe and its platform
provider together in `dist/`. Then run your app from the same environment
with that directory on the import path. [Packaging issue](https://github.com/randyzwitch/dataframe_mojo/issues/1384)

```bash
pixi run mojo run -I dist /path/to/your/app.mojo
```

Compiled Mojo packages require the compiler version that built them. Release
v0.3.0 uses Mojo 1.2.0. The development branch pins Mojo
1.3.0.dev2026111005 and requires Pixi 0.84.2. To try the features on main,
clone the default branch, run `pixi install --locked`, then `pixi run example`.
The public API is experimental and may change between minor releases.
[Release API](https://github.com/randyzwitch/dataframe_mojo/blob/v0.3.0/docs/api.md) · [Development API](docs/api.md) · [Stability policy](docs/stability.md)

## Go further

- **Coming from pandas or Polars?** Start with equivalent operations, dtype rules, and ordering guarantees in the [migration guide](docs/migration.md).
- **Looking for working code?** Run `pixi run example` for the [sales report](examples/sales.mojo), [expression examples](examples/expressions.mojo), and [CSV reader](examples/read_csv.mojo).
- **Evaluating a workload?** Use the repository’s benchmark suites and result checks to measure the operations you care about. [Benchmark guide](docs/design/benchmarks.md)
- **Contributing?** The testing guide covers native dependencies, the CPU suite, and interchange checks. [Testing guide](docs/testing.md) · [Architecture](docs/architecture.md)

[MIT license](LICENSE)
