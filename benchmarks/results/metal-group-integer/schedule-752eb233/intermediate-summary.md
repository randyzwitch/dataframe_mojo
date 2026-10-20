Paired complete-query warm medians. A ratio above 1 means the candidate was faster than the baseline. Per-round ranges show variability; they are not confidence intervals. CPU timings and profile counters are reported separately in the JSON.

| Rows | Workload | Variant | Baseline ms | Candidate ms | CPU ms | Ratio | Round ratio range |
|---:|---|---|---:|---:|---:|---:|---:|
| 4097 | group_hash | base | 11.528 | 8.634 | 0.175 | 1.335 | 1.285–1.360 |
| 4097 | group_hash | sorted | 11.696 | 8.506 | 0.184 | 1.375 | 1.367–1.421 |
| 4097 | group_hash | nulls | 11.961 | 9.002 | 0.190 | 1.329 | 1.317–1.588 |
| 4097 | group_high | base | 11.712 | 8.955 | 0.247 | 1.308 | 1.218–1.335 |
| 4097 | group_high | sorted | 11.976 | 8.960 | 0.226 | 1.337 | 1.313–1.353 |
| 4097 | group_high | nulls | 12.441 | 9.306 | 0.276 | 1.337 | 1.332–1.360 |
| 4097 | group_skew | base | 15.791 | 14.137 | 0.179 | 1.117 | 1.086–1.217 |
| 4097 | group_skew | sorted | 17.542 | 14.442 | 0.220 | 1.215 | 1.200–1.276 |
| 4097 | group_skew | nulls | 15.708 | 13.427 | 0.198 | 1.170 | 1.167–1.256 |
| 1048576 | group_hash | base | 62.749 | 57.039 | 4.721 | 1.100 | 1.098–1.108 |
| 1048576 | group_hash | sorted | 62.876 | 58.761 | 5.337 | 1.070 | 1.046–1.126 |
| 1048576 | group_hash | nulls | 130.404 | 67.699 | 5.949 | 1.926 | 1.901–1.941 |
| 1048576 | group_high | base | 273.572 | 269.575 | 66.509 | 1.015 | 1.003–1.019 |
| 1048576 | group_high | sorted | 271.422 | 268.459 | 60.557 | 1.011 | 1.005–1.015 |
| 1048576 | group_high | nulls | 371.298 | 369.443 | 64.865 | 1.005 | 1.003–1.010 |
| 1048576 | group_skew | base | 911.564 | 185.925 | 8.494 | 4.903 | 4.863–4.980 |
| 1048576 | group_skew | sorted | 904.498 | 187.233 | 7.845 | 4.831 | 4.785–4.892 |
| 1048576 | group_skew | nulls | 887.588 | 184.542 | 9.929 | 4.810 | 4.782–4.810 |

| Rows | Workload | Variant | First use baseline/candidate ms | Requested peak baseline/candidate bytes | Launches baseline/candidate |
|---:|---|---|---:|---:|---:|
| 4097 | group_hash | base | 42.127/41.612 | 2604134/2866278 | 24/24 |
| 4097 | group_hash | sorted | 42.396/41.336 | 2604134/2866278 | 24/24 |
| 4097 | group_hash | nulls | 42.339/41.710 | 2604134/2866278 | 25/25 |
| 4097 | group_high | base | 41.508/42.281 | 2604134/2866278 | 25/25 |
| 4097 | group_high | sorted | 42.650/42.091 | 2604134/2866278 | 25/25 |
| 4097 | group_high | nulls | 42.835/42.317 | 2604134/2866278 | 26/26 |
| 4097 | group_skew | base | 46.124/46.073 | 2604134/2866278 | 22/22 |
| 4097 | group_skew | sorted | 48.402/47.610 | 2604134/2866278 | 24/24 |
| 4097 | group_skew | nulls | 46.226/46.730 | 2604134/2866278 | 23/23 |
| 1048576 | group_hash | base | 93.238/92.965 | 641106816/641368960 | 24/24 |
| 1048576 | group_hash | sorted | 92.948/91.966 | 641106816/641368960 | 24/24 |
| 1048576 | group_hash | nulls | 169.321/106.698 | 641106816/641368960 | 25/29 |
| 1048576 | group_high | base | 301.052/301.192 | 641106816/641368960 | 33/33 |
| 1048576 | group_high | sorted | 298.322/301.157 | 641106816/641368960 | 33/33 |
| 1048576 | group_high | nulls | 399.332/400.690 | 641106816/641368960 | 34/34 |
| 1048576 | group_skew | base | 932.645/214.289 | 641106816/641368960 | 22/26 |
| 1048576 | group_skew | sorted | 928.847/216.048 | 641106816/641368960 | 24/28 |
| 1048576 | group_skew | nulls | 902.679/213.755 | 641106816/641368960 | 23/27 |

First use is a fresh query process with runtime initialization; it is not a cold-machine measurement.

These are measured development cases, not a general GPU-over-CPU claim. Collection timings include staging and result construction. Profile memory is requested allocation accounting, not measured device residency. Retain build receipts and raw samples with this derived report.
