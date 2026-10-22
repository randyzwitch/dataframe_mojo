Paired complete-query warm medians. A ratio above 1 means the candidate was faster than the baseline. Per-round ranges show variability; they are not confidence intervals. CPU timings and profile counters are reported separately in the JSON.

| Rows | Workload | Variant | Baseline ms | Candidate ms | CPU ms | Ratio | Round ratio range |
|---:|---|---|---:|---:|---:|---:|---:|
| 4097 | group_hash | base | 12.179 | 8.506 | 0.176 | 1.432 | 1.321–1.797 |
| 4097 | group_hash | sorted | 12.268 | 8.558 | 0.185 | 1.434 | 1.418–1.462 |
| 4097 | group_hash | nulls | 12.662 | 8.934 | 0.190 | 1.417 | 1.408–1.420 |
| 4097 | group_high | base | 12.557 | 8.971 | 0.244 | 1.400 | 1.328–1.439 |
| 4097 | group_high | sorted | 11.942 | 8.960 | 0.228 | 1.333 | 1.290–1.430 |
| 4097 | group_high | nulls | 12.891 | 9.287 | 0.273 | 1.388 | 1.380–1.458 |
| 4097 | group_skew | base | 16.724 | 14.121 | 0.178 | 1.184 | 1.171–1.222 |
| 4097 | group_skew | sorted | 17.496 | 14.462 | 0.210 | 1.210 | 1.184–1.243 |
| 4097 | group_skew | nulls | 16.403 | 13.809 | 0.194 | 1.188 | 1.193–1.436 |
| 1048576 | group_hash | base | 219.186 | 58.263 | 5.226 | 3.762 | 3.734–3.820 |
| 1048576 | group_hash | sorted | 214.340 | 56.217 | 5.033 | 3.813 | 3.668–3.861 |
| 1048576 | group_hash | nulls | 301.435 | 68.210 | 6.442 | 4.419 | 4.330–4.569 |
| 1048576 | group_high | base | 434.918 | 270.564 | 64.529 | 1.607 | 1.402–1.611 |
| 1048576 | group_high | sorted | 435.357 | 266.344 | 69.593 | 1.635 | 1.607–1.646 |
| 1048576 | group_high | nulls | 490.962 | 366.595 | 69.484 | 1.339 | 1.336–1.352 |
| 1048576 | group_skew | base | 960.671 | 185.442 | 8.215 | 5.180 | 5.120–5.248 |
| 1048576 | group_skew | sorted | 951.912 | 185.428 | 11.098 | 5.134 | 5.110–5.241 |
| 1048576 | group_skew | nulls | 938.473 | 186.064 | 9.256 | 5.044 | 5.018–5.195 |

| Rows | Workload | Variant | First use baseline/candidate ms | Requested peak baseline/candidate bytes | Launches baseline/candidate |
|---:|---|---|---:|---:|---:|
| 4097 | group_hash | base | 42.724/40.764 | 2473062/2866278 | 35/24 |
| 4097 | group_hash | sorted | 42.452/41.461 | 2473062/2866278 | 35/24 |
| 4097 | group_hash | nulls | 42.934/42.043 | 2473062/2866278 | 36/25 |
| 4097 | group_high | base | 42.862/42.175 | 2473062/2866278 | 36/25 |
| 4097 | group_high | sorted | 41.617/42.101 | 2473062/2866278 | 36/25 |
| 4097 | group_high | nulls | 43.241/42.087 | 2473062/2866278 | 37/26 |
| 4097 | group_skew | base | 46.246/46.885 | 2473062/2866278 | 33/22 |
| 4097 | group_skew | sorted | 47.718/47.263 | 2473062/2866278 | 35/24 |
| 4097 | group_skew | nulls | 45.600/46.540 | 2473062/2866278 | 34/23 |
| 1048576 | group_hash | base | 235.197/92.623 | 632718208/641368960 | 42/24 |
| 1048576 | group_hash | sorted | 238.873/91.878 | 632718208/641368960 | 42/24 |
| 1048576 | group_hash | nulls | 305.399/110.542 | 632718208/641368960 | 43/29 |
| 1048576 | group_high | base | 461.713/299.038 | 632718208/641368960 | 51/33 |
| 1048576 | group_high | sorted | 459.167/297.317 | 632718208/641368960 | 51/33 |
| 1048576 | group_high | nulls | 506.113/401.893 | 632718208/641368960 | 52/34 |
| 1048576 | group_skew | base | 958.539/209.790 | 632718208/641368960 | 40/26 |
| 1048576 | group_skew | sorted | 965.199/213.453 | 632718208/641368960 | 42/28 |
| 1048576 | group_skew | nulls | 937.523/212.657 | 632718208/641368960 | 41/27 |

First use is a fresh query process with runtime initialization; it is not a cold-machine measurement.

These are measured development cases, not a general GPU-over-CPU claim. Collection timings include staging and result construction. Profile memory is requested allocation accounting, not measured device residency. Retain build receipts and raw samples with this derived report.
