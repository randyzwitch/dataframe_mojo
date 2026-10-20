Paired complete-query warm medians. A ratio above 1 means the candidate was faster than the baseline. Per-round ranges show variability; they are not confidence intervals. CPU timings and profile counters are reported separately in the JSON.

| Rows | Workload | Variant | Baseline ms | Candidate ms | CPU ms | Ratio | Round ratio range |
|---:|---|---|---:|---:|---:|---:|---:|
| 4097 | group_hash | base | 12.341 | 8.651 | 0.177 | 1.427 | 1.258–1.458 |
| 4097 | group_hash | sorted | 12.268 | 8.659 | 0.186 | 1.417 | 1.412–1.687 |
| 4097 | group_hash | nulls | 12.514 | 9.014 | 0.192 | 1.388 | 1.382–1.592 |
| 4097 | group_high | base | 11.518 | 8.947 | 0.246 | 1.287 | 1.275–1.420 |
| 4097 | group_high | sorted | 12.229 | 8.888 | 0.227 | 1.376 | 1.267–1.447 |
| 4097 | group_high | nulls | 12.802 | 9.336 | 0.270 | 1.371 | 1.259–1.393 |
| 4097 | group_skew | base | 16.985 | 14.240 | 0.178 | 1.193 | 1.176–1.206 |
| 4097 | group_skew | sorted | 17.525 | 14.521 | 0.217 | 1.207 | 1.185–1.212 |
| 4097 | group_skew | nulls | 16.902 | 14.050 | 0.199 | 1.203 | 1.120–1.219 |
| 1048576 | group_hash | base | 220.737 | 67.152 | 4.770 | 3.287 | 3.254–3.462 |
| 1048576 | group_hash | sorted | 212.495 | 65.086 | 5.643 | 3.265 | 3.144–3.311 |
| 1048576 | group_hash | nulls | 302.612 | 69.199 | 6.073 | 4.373 | 4.259–4.407 |
| 1048576 | group_high | base | 438.199 | 271.463 | 66.116 | 1.614 | 1.601–1.631 |
| 1048576 | group_high | sorted | 433.424 | 262.946 | 69.998 | 1.648 | 1.648–1.661 |
| 1048576 | group_high | nulls | 492.742 | 370.360 | 66.609 | 1.330 | 1.320–1.338 |
| 1048576 | group_skew | base | 956.434 | 185.613 | 7.989 | 5.153 | 4.997–5.241 |
| 1048576 | group_skew | sorted | 959.846 | 183.706 | 9.352 | 5.225 | 5.062–5.258 |
| 1048576 | group_skew | nulls | 942.147 | 186.848 | 9.721 | 5.042 | 5.026–5.177 |

| Rows | Workload | Variant | First use baseline/candidate ms | Requested peak baseline/candidate bytes | Launches baseline/candidate |
|---:|---|---|---:|---:|---:|
| 4097 | group_hash | base | 42.369/41.733 | 2473062/2866278 | 35/24 |
| 4097 | group_hash | sorted | 42.518/41.473 | 2473062/2866278 | 35/24 |
| 4097 | group_hash | nulls | 42.919/41.837 | 2473062/2866278 | 36/25 |
| 4097 | group_high | base | 42.949/41.671 | 2473062/2866278 | 36/25 |
| 4097 | group_high | sorted | 42.602/41.585 | 2473062/2866278 | 36/25 |
| 4097 | group_high | nulls | 43.140/42.025 | 2473062/2866278 | 37/26 |
| 4097 | group_skew | base | 46.151/46.920 | 2473062/2866278 | 33/22 |
| 4097 | group_skew | sorted | 47.538/47.285 | 2473062/2866278 | 35/24 |
| 4097 | group_skew | nulls | 45.813/46.252 | 2473062/2866278 | 34/23 |
| 1048576 | group_hash | base | 238.619/102.041 | 632718208/641368960 | 42/28 |
| 1048576 | group_hash | sorted | 237.630/100.285 | 632718208/641368960 | 42/28 |
| 1048576 | group_hash | nulls | 304.372/106.223 | 632718208/641368960 | 43/29 |
| 1048576 | group_high | base | 463.251/300.303 | 632718208/641368960 | 51/33 |
| 1048576 | group_high | sorted | 466.101/295.626 | 632718208/641368960 | 51/33 |
| 1048576 | group_high | nulls | 521.155/400.861 | 632718208/641368960 | 52/34 |
| 1048576 | group_skew | base | 956.478/214.792 | 632718208/641368960 | 40/26 |
| 1048576 | group_skew | sorted | 967.287/213.332 | 632718208/641368960 | 42/28 |
| 1048576 | group_skew | nulls | 945.650/210.317 | 632718208/641368960 | 41/27 |

First use is a fresh query process with runtime initialization; it is not a cold-machine measurement.

These are measured development cases, not a general GPU-over-CPU claim. Collection timings include staging and result construction. Profile memory is requested allocation accounting, not measured device residency. Retain build receipts and raw samples with this derived report.
