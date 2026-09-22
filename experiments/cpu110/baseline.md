# Baseline: #154 (`7a01b37`)

Measured 2026-09-22, Threadripper 3970X, Linux x86-64, pinned Mojo 1.2
nightly, Polars 1.44.2. Best of three after warmup, one process per thread
cap, no concurrent compiles/tests/benchmarks. These are baseline results,
not final claims for the experimental candidates. Caps are equal; engines
may choose fewer workers internally. The driver verifies output heights
and checksums. Polars sort uses its faster default unstable mode, while
Mojo's sort is stable.

```sh
pixi run -e oracle bench-polars --sizes 1000000,10000000 --threads 1 --reps 3 --runner build/bench_csv_numeric149
pixi run -e oracle bench-polars --sizes 1000000,10000000 --threads 32 --reps 3 --runner build/bench_csv_numeric149
```

## Thread cap 1

# polars=1.44.2 threads=1 reps=3 machine=x86_64 Linux
| workload | rows | dataframe_mojo ms | polars ms | mojo / polars |
|---|---|---|---|---|
| csv_read | 1,000,000 | 320.73 | 213.46 | 1.5x |
| arithmetic_chain | 1,000,000 | 19.49 | 2.26 | 8.6x |
| nullable_compare | 1,000,000 | 16.97 | 0.87 | 19.5x |
| filter | 1,000,000 | 51.33 | 7.23 | 7.1x |
| global_sum | 1,000,000 | 4.67 | 0.63 | 7.4x |
| grouped_low | 1,000,000 | 27.69 | 21.09 | 1.3x |
| grouped_high | 1,000,000 | 44.43 | 61.74 | 0.7x |
| grouped_skew | 1,000,000 | 28.32 | 25.18 | 1.1x |
| grouped_str | 1,000,000 | 37.02 | 35.11 | 1.1x |
| join_inner | 1,000,000 | 176.71 | 131.18 | 1.3x |
| sort_multi | 1,000,000 | 705.93 | 356.11 | 2.0x |
| csv_read | 10,000,000 | 3778.23 | 2258.07 | 1.7x |
| arithmetic_chain | 10,000,000 | 198.13 | 17.78 | 11.1x |
| nullable_compare | 10,000,000 | 171.90 | 7.52 | 22.9x |
| filter | 10,000,000 | 525.88 | 69.18 | 7.6x |
| global_sum | 10,000,000 | 47.57 | 7.54 | 6.3x |
| grouped_low | 10,000,000 | 284.44 | 207.45 | 1.4x |
| grouped_high | 10,000,000 | 1116.91 | 1383.23 | 0.8x |
| grouped_skew | 10,000,000 | 290.31 | 218.63 | 1.3x |
| grouped_str | 10,000,000 | 379.43 | 439.74 | 0.9x |
| join_inner | 10,000,000 | 5077.84 | 1912.55 | 2.7x |
| sort_multi | 10,000,000 | 17274.16 | 6559.63 | 2.6x |

## Thread cap 32

# polars=1.44.2 threads=32 reps=3 machine=x86_64 Linux
| workload | rows | dataframe_mojo ms | polars ms | mojo / polars |
|---|---|---|---|---|
| csv_read | 1,000,000 | 23.91 | 14.39 | 1.7x |
| arithmetic_chain | 1,000,000 | 4.61 | 3.73 | 1.2x |
| nullable_compare | 1,000,000 | 2.64 | 3.76 | 0.7x |
| filter | 1,000,000 | 15.29 | 6.19 | 2.5x |
| global_sum | 1,000,000 | 2.18 | 0.28 | 7.7x |
| grouped_low | 1,000,000 | 23.14 | 13.89 | 1.7x |
| grouped_high | 1,000,000 | 27.04 | 13.56 | 2.0x |
| grouped_skew | 1,000,000 | 23.84 | 14.07 | 1.7x |
| grouped_str | 1,000,000 | 32.72 | 15.77 | 2.1x |
| join_inner | 1,000,000 | 138.20 | 25.67 | 5.4x |
| sort_multi | 1,000,000 | 95.28 | 57.81 | 1.6x |
| csv_read | 10,000,000 | 173.48 | 88.47 | 2.0x |
| arithmetic_chain | 10,000,000 | 32.16 | 9.19 | 3.5x |
| nullable_compare | 10,000,000 | 8.70 | 6.77 | 1.3x |
| filter | 10,000,000 | 101.76 | 27.92 | 3.6x |
| global_sum | 10,000,000 | 12.85 | 1.52 | 8.4x |
| grouped_low | 10,000,000 | 240.79 | 108.27 | 2.2x |
| grouped_high | 10,000,000 | 206.07 | 178.01 | 1.2x |
| grouped_skew | 10,000,000 | 212.68 | 107.80 | 2.0x |
| grouped_str | 10,000,000 | 312.79 | 140.17 | 2.2x |
| join_inner | 10,000,000 | 2661.54 | 175.13 | 15.2x |
| sort_multi | 10,000,000 | 1195.22 | 609.66 | 2.0x |
