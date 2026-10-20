Paired complete-query warm medians. A ratio above 1 means the candidate was faster than the baseline. Per-round ranges show variability; they are not confidence intervals. CPU timings and profile counters are reported separately in the JSON.

| Rows | Workload | Variant | Baseline ms | Candidate ms | CPU ms | Ratio | Round ratio range |
|---:|---|---|---:|---:|---:|---:|---:|
| 4097 | group_hash | base | 8.675 | 8.506 | 0.177 | 1.020 | 0.999–1.022 |
| 4097 | group_hash | sorted | 8.477 | 8.567 | 0.185 | 0.989 | 0.908–1.013 |
| 4097 | group_hash | nulls | 8.905 | 8.965 | 0.193 | 0.993 | 0.969–1.123 |
| 4097 | group_high | base | 8.881 | 8.997 | 0.241 | 0.987 | 0.929–1.004 |
| 4097 | group_high | sorted | 8.933 | 8.715 | 0.229 | 1.025 | 1.002–1.125 |
| 4097 | group_high | nulls | 9.338 | 9.256 | 0.265 | 1.009 | 0.959–1.166 |
| 4097 | group_skew | base | 14.223 | 14.214 | 0.181 | 1.001 | 0.908–1.011 |
| 4097 | group_skew | sorted | 14.313 | 14.565 | 0.223 | 0.983 | 0.951–0.999 |
| 4097 | group_skew | nulls | 14.234 | 13.361 | 0.197 | 1.065 | 1.002–1.070 |
| 1048576 | group_hash | base | 65.254 | 56.623 | 5.080 | 1.152 | 1.083–1.179 |
| 1048576 | group_hash | sorted | 65.810 | 57.755 | 4.936 | 1.139 | 1.084–1.153 |
| 1048576 | group_hash | nulls | 67.943 | 67.717 | 6.370 | 1.003 | 0.994–1.010 |
| 1048576 | group_high | base | 268.023 | 271.414 | 81.071 | 0.988 | 0.981–1.007 |
| 1048576 | group_high | sorted | 265.952 | 264.181 | 66.245 | 1.007 | 1.002–1.011 |
| 1048576 | group_high | nulls | 367.325 | 368.484 | 70.412 | 0.997 | 0.993–1.014 |
| 1048576 | group_skew | base | 184.387 | 185.514 | 7.505 | 0.994 | 0.925–1.006 |
| 1048576 | group_skew | sorted | 186.756 | 185.803 | 7.958 | 1.005 | 0.994–1.025 |
| 1048576 | group_skew | nulls | 184.261 | 183.953 | 8.533 | 1.002 | 0.972–1.023 |

| Rows | Workload | Variant | First use baseline/candidate ms | Requested peak baseline/candidate bytes | Launches baseline/candidate |
|---:|---|---|---:|---:|---:|
| 4097 | group_hash | base | 41.280/40.610 | 2866278/2866278 | 24/24 |
| 4097 | group_hash | sorted | 41.595/41.613 | 2866278/2866278 | 24/24 |
| 4097 | group_hash | nulls | 40.671/41.792 | 2866278/2866278 | 25/25 |
| 4097 | group_high | base | 41.470/41.588 | 2866278/2866278 | 25/25 |
| 4097 | group_high | sorted | 41.666/41.706 | 2866278/2866278 | 25/25 |
| 4097 | group_high | nulls | 41.775/41.397 | 2866278/2866278 | 26/26 |
| 4097 | group_skew | base | 46.826/46.829 | 2866278/2866278 | 22/22 |
| 4097 | group_skew | sorted | 45.655/47.329 | 2866278/2866278 | 24/24 |
| 4097 | group_skew | nulls | 45.859/47.045 | 2866278/2866278 | 23/23 |
| 1048576 | group_hash | base | 101.139/91.506 | 641368960/641368960 | 28/24 |
| 1048576 | group_hash | sorted | 102.108/91.477 | 641368960/641368960 | 28/24 |
| 1048576 | group_hash | nulls | 106.405/108.579 | 641368960/641368960 | 29/29 |
| 1048576 | group_high | base | 300.418/299.677 | 641368960/641368960 | 33/33 |
| 1048576 | group_high | sorted | 297.433/299.941 | 641368960/641368960 | 33/33 |
| 1048576 | group_high | nulls | 403.679/402.660 | 641368960/641368960 | 34/34 |
| 1048576 | group_skew | base | 211.292/214.430 | 641368960/641368960 | 26/26 |
| 1048576 | group_skew | sorted | 213.427/214.779 | 641368960/641368960 | 28/28 |
| 1048576 | group_skew | nulls | 213.355/213.730 | 641368960/641368960 | 27/27 |

First use is a fresh query process with runtime initialization; it is not a cold-machine measurement.

These are measured development cases, not a general GPU-over-CPU claim. Collection timings include staging and result construction. Profile memory is requested allocation accounting, not measured device residency. Retain build receipts and raw samples with this derived report.
