# Raw-key join full-result matrix

Milliseconds, best of three; unique right-side keys, exact full-frame comparison.
Production is PR #155. Raw prototype wall time includes setup, phases, and final output assembly.
Polars maintains left/right row order. Both inputs carry payload columns.

| Rows | Key | Shape | Cardinality | Threads | Production | Raw prototype | Polars |
|---|---|---|---:|---:|---:|---:|---:|
| 10000000 | int64 | repeat | 1000000 | 1 | 416.07 | 1518.07 | 1032.34 |
| 10000000 | int64 | repeat | 100 | 1 | 323.67 | 542.07 | 140.15 |
| 1000000 | int64 | repeat | 100000 | 1 | 36.47 | 74.43 | 25.76 |
| 1000000 | int64 | repeat | 100 | 1 | 32.33 | 68.20 | 14.73 |
| 10000000 | int64 | skew | 100 | 1 | 329.44 | 654.44 | 139.19 |
| 1000000 | int64 | skew | 100 | 1 | 32.91 | 69.94 | 14.03 |
| 10000000 | int64 | sparse | 1000000 | 1 | 1333.10 | 1459.74 | 965.52 |
| 1000000 | int64 | sparse | 100000 | 1 | 62.77 | 75.44 | 25.44 |
| 10000000 | string | repeat | 1000000 | 1 | 2160.97 | 2286.47 | 1129.66 |
| 1000000 | string | repeat | 100000 | 1 | 88.93 | 117.03 | 36.22 |
| 10000000 | string | skew | 100 | 1 | 736.34 | 1029.16 | 206.92 |
| 1000000 | string | skew | 100 | 1 | 70.79 | 102.69 | 30.31 |
| 10000000 | int64 | repeat | 1000000 | 32 | 223.93 | 436.51 | 84.91 |
| 10000000 | int64 | repeat | 100 | 32 | 196.09 | 362.88 | 34.34 |
| 1000000 | int64 | repeat | 100000 | 32 | 28.47 | 51.76 | 7.46 |
| 1000000 | int64 | repeat | 100 | 32 | 25.53 | 49.26 | 7.20 |
| 10000000 | int64 | skew | 100 | 32 | 197.97 | 467.61 | 34.47 |
| 1000000 | int64 | skew | 100 | 32 | 30.58 | 57.42 | 7.02 |
| 10000000 | int64 | sparse | 1000000 | 32 | 825.98 | 418.66 | 87.20 |
| 1000000 | int64 | sparse | 100000 | 32 | 63.36 | 53.27 | 7.25 |
| 10000000 | string | repeat | 1000000 | 32 | 1002.20 | 568.15 | 114.93 |
| 1000000 | string | repeat | 100000 | 32 | 101.85 | 62.48 | 11.21 |
| 10000000 | string | skew | 100 | 32 | 447.65 | 573.84 | 51.11 |
| 1000000 | string | skew | 100 | 32 | 55.53 | 79.66 | 11.99 |
