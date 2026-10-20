Paired complete-query warm medians. A ratio above 1 means the candidate was faster than the baseline. Per-round ranges show variability; they are not confidence intervals. CPU timings and profile counters are reported separately in the JSON.

| Rows | Workload | Variant | Baseline ms | Candidate ms | CPU ms | Ratio | Round ratio range |
|---:|---|---|---:|---:|---:|---:|---:|
| 4097 | group_hash | base | 11.668 | 8.336 | 0.176 | 1.400 | 1.382–1.494 |
| 4097 | group_hash | sorted | 11.703 | 7.794 | 0.186 | 1.502 | 1.381–1.576 |
| 4097 | group_hash | nulls | 12.073 | 8.876 | 0.191 | 1.360 | 1.332–1.422 |
| 4097 | group_high | base | 11.936 | 8.942 | 0.243 | 1.335 | 1.271–1.336 |
| 4097 | group_high | sorted | 11.927 | 8.864 | 0.230 | 1.346 | 1.217–1.356 |
| 4097 | group_high | nulls | 12.188 | 9.343 | 0.268 | 1.305 | 1.280–1.328 |
| 4097 | group_skew | base | 16.967 | 14.190 | 0.180 | 1.196 | 1.145–1.228 |
| 4097 | group_skew | sorted | 16.876 | 14.624 | 0.202 | 1.154 | 1.116–1.220 |
| 4097 | group_skew | nulls | 16.513 | 14.042 | 0.196 | 1.176 | 1.163–1.284 |
| 1048576 | group_hash | base | 62.827 | 67.008 | 5.182 | 0.938 | 0.920–0.953 |
| 1048576 | group_hash | sorted | 62.318 | 66.828 | 4.598 | 0.933 | 0.922–0.961 |
| 1048576 | group_hash | nulls | 130.169 | 69.005 | 6.331 | 1.886 | 1.846–1.923 |
| 1048576 | group_high | base | 275.631 | 270.559 | 65.882 | 1.019 | 1.002–1.023 |
| 1048576 | group_high | sorted | 270.010 | 264.581 | 64.455 | 1.021 | 1.002–1.035 |
| 1048576 | group_high | nulls | 373.905 | 368.420 | 67.741 | 1.015 | 1.001–1.022 |
| 1048576 | group_skew | base | 917.547 | 187.788 | 7.811 | 4.886 | 4.841–4.966 |
| 1048576 | group_skew | sorted | 906.675 | 185.227 | 8.959 | 4.895 | 4.806–4.905 |
| 1048576 | group_skew | nulls | 884.402 | 185.979 | 9.144 | 4.755 | 4.724–4.821 |

| Rows | Workload | Variant | First use baseline/candidate ms | Requested peak baseline/candidate bytes | Launches baseline/candidate |
|---:|---|---|---:|---:|---:|
| 4097 | group_hash | base | 42.406/42.049 | 2604134/2866278 | 24/24 |
| 4097 | group_hash | sorted | 41.775/40.558 | 2604134/2866278 | 24/24 |
| 4097 | group_hash | nulls | 42.818/42.072 | 2604134/2866278 | 25/25 |
| 4097 | group_high | base | 42.903/41.811 | 2604134/2866278 | 25/25 |
| 4097 | group_high | sorted | 42.743/41.667 | 2604134/2866278 | 25/25 |
| 4097 | group_high | nulls | 43.200/42.920 | 2604134/2866278 | 26/26 |
| 4097 | group_skew | base | 46.232/46.893 | 2604134/2866278 | 22/22 |
| 4097 | group_skew | sorted | 48.470/47.354 | 2604134/2866278 | 24/24 |
| 4097 | group_skew | nulls | 45.983/46.362 | 2604134/2866278 | 23/23 |
| 1048576 | group_hash | base | 94.972/101.998 | 641106816/641368960 | 24/28 |
| 1048576 | group_hash | sorted | 94.270/105.087 | 641106816/641368960 | 24/28 |
| 1048576 | group_hash | nulls | 163.200/106.686 | 641106816/641368960 | 25/29 |
| 1048576 | group_high | base | 301.825/298.419 | 641106816/641368960 | 33/33 |
| 1048576 | group_high | sorted | 300.846/300.193 | 641106816/641368960 | 33/33 |
| 1048576 | group_high | nulls | 404.470/402.598 | 641106816/641368960 | 34/34 |
| 1048576 | group_skew | base | 932.953/213.081 | 641106816/641368960 | 22/26 |
| 1048576 | group_skew | sorted | 925.392/215.028 | 641106816/641368960 | 24/28 |
| 1048576 | group_skew | nulls | 901.397/216.246 | 641106816/641368960 | 23/27 |

First use is a fresh query process with runtime initialization; it is not a cold-machine measurement.

These are measured development cases, not a general GPU-over-CPU claim. Collection timings include staging and result construction. Profile memory is requested allocation accounting, not measured device residency. Retain build receipts and raw samples with this derived report.
