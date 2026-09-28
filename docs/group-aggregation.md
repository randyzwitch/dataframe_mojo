# Group-by aggregation over worker ranges (#306)

## What changed

Two group-by fast paths accepted only particular aggregation lists:

- one Int64 key with exactly two aggregations, a Float64 `sum` and a
  `count` (the `grouped_*` query in `bench_vs_polars`), and
- one Int64 key with at least three `sum`, `count` or `mean` aggregations,
  all over Float64 columns.

Any other list, including `sum` alone, took the general path and ran 5 to
15 times slower on the same key. Both paths are removed.

In their place, when the key has low cardinality (the same sampled estimate
that already picks whole-frame encoding), each worker reduces its own row
range into mergeable per-group state and the states are merged in order.
This is the state the streaming executor already merges across batches
(`_StreamReduction`, now in `frame.mojo`), so it serves every reduction that
streaming supports (sum, count, len, min, max, mean, first, last, std, var,
any, all, null_count, n_unique, and expressions over them) on any key types,
including strings and struct keys. Ranges are zero-copy slices, so no column
is gathered. A range holding more groups than the state budget falls back to
hash partitioning. Groups keep first-occurrence order.

High-cardinality keys use hash partitioning, as every list other than the
benchmark pair already did.

## Measurements

AMD Threadripper 3970X, 32 workers (`DATAFRAME_THREADS=32`,
`POLARS_MAX_THREADS=32`), Mojo 1.2.0.dev2026092105 (e9569894), Polars 1.44.2. Both engines read the
`bench_vs_polars` inputs (`left_1000000.csv`, `left_10000000.csv`), warm up
once and report the best of seven runs in milliseconds. Loading is untimed.
Main is `5ab269d`. One round per binary, run back to back on an otherwise
idle machine; treat differences under about 10% as noise.

`key_low` has 16 groups, `key_skew` 1,000 with a dominant few, and `key_high`
about rows/10. `sum_count` is the benchmark's pair; `sum_count_mean` adds a
Float64 mean.

| Rows | Aggregations | Key | main ms | this change ms | Polars ms | vs Polars |
|---|---|---|---:|---:|---:|---:|
| 1M | sum_count | key_low | 2.2 | 3.3 | 12.9 | 0.25x |
| 1M | sum_count | key_high | 13.7 | 24.0 | 13.3 | 1.80x |
| 1M | sum_count | key_skew | 2.9 | 3.8 | 15.5 | 0.25x |
| 1M | sum_count_mean | key_low | 16.6 | 3.8 | 12.6 | 0.30x |
| 1M | sum_count_mean | key_high | 28.6 | 28.0 | 16.0 | 1.75x |
| 1M | sum_count_mean | key_skew | 17.1 | 4.6 | 14.8 | 0.31x |
| 1M | sum | key_low | 11.7 | 3.0 | 4.2 | 0.71x |
| 1M | sum | key_high | 22.3 | 23.4 | 14.8 | 1.58x |
| 1M | sum | key_skew | 12.0 | 3.6 | 14.4 | 0.25x |
| 1M | min_max_int | key_low | 14.0 | 3.6 | 4.9 | 0.73x |
| 1M | min_max_int | key_high | 20.6 | 22.1 | 14.3 | 1.54x |
| 1M | min_max_int | key_skew | 15.1 | 4.4 | 14.8 | 0.30x |
| 10M | sum_count | key_low | 10.8 | 23.5 | 115.8 | 0.20x |
| 10M | sum_count | key_high | 113.8 | 189.8 | 166.1 | 1.14x |
| 10M | sum_count | key_skew | 10.9 | 25.1 | 120.3 | 0.21x |
| 10M | sum_count_mean | key_low | 174.5 | 26.7 | 116.6 | 0.23x |
| 10M | sum_count_mean | key_high | 219.3 | 223.1 | 172.1 | 1.30x |
| 10M | sum_count_mean | key_skew | 137.1 | 27.1 | 126.6 | 0.21x |
| 10M | sum | key_low | 102.4 | 22.1 | 11.2 | 1.97x |
| 10M | sum | key_high | 151.0 | 151.8 | 169.5 | 0.90x |
| 10M | sum | key_skew | 102.8 | 23.5 | 132.1 | 0.18x |
| 10M | min_max_int | key_low | 121.0 | 26.2 | 14.6 | 1.80x |
| 10M | min_max_int | key_high | 155.3 | 156.5 | 171.8 | 0.91x |
| 10M | min_max_int | key_skew | 121.2 | 26.2 | 112.0 | 0.23x |

On low-cardinality keys every list now runs at about the same speed, 3.3 to
6.5 times faster than main, except the benchmark pair, which loses its
special case and is 1.3 to 2.3 times slower. On high-cardinality keys the
pair is 1.7 times slower at 1M and 10M rows; the other lists are unchanged.
The pair's high-cardinality path read source rows by bucket without a gather;
bringing that to every list is a separate change to the partitioned path.

## Reproduce

```bash
mojo build -O3 -I . benchmarks/bench_agg_shapes.mojo -o build/bench_agg_shapes
DATAFRAME_THREADS=32 build/bench_agg_shapes build/bench_polars/left_10000000.csv 7
POLARS_MAX_THREADS=32 pixi run -e oracle python3 scripts/bench_agg_shapes_polars.py build/bench_polars/left_10000000.csv 7
```
