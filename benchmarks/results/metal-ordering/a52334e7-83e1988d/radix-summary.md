Paired complete-query warm medians. A ratio above 1 means the candidate was faster than the baseline. Per-round ranges show variability; they are not confidence intervals. CPU timings and profile counters are reported separately in the JSON.

| Rows | Workload | Variant | Baseline ms | Candidate ms | CPU ms | Ratio | Round ratio range |
|---:|---|---|---:|---:|---:|---:|---:|
| 4097 | sort_topk31 | base | 5.986 | 4.832 | 0.038 | 1.239 | 0.974–1.266 |
| 4097 | sort_topk31 | sorted | 5.501 | 5.267 | 0.033 | 1.044 | 1.053–1.192 |
| 4097 | sort_topk31 | nulls | 5.490 | 5.297 | 0.042 | 1.036 | 1.022–1.112 |
| 4097 | sort_topk1024 | base | 5.619 | 5.134 | 0.147 | 1.094 | 1.045–1.104 |
| 4097 | sort_topk1024 | sorted | 5.591 | 5.018 | 0.050 | 1.114 | 1.016–1.139 |
| 4097 | sort_topk1024 | nulls | 5.572 | 5.150 | 0.157 | 1.082 | 1.072–1.082 |
| 1048576 | sort_topk31 | base | 47.554 | 44.170 | 1.124 | 1.077 | 1.077–1.093 |
| 1048576 | sort_topk31 | sorted | 47.020 | 43.607 | 1.021 | 1.078 | 1.040–1.095 |
| 1048576 | sort_topk31 | nulls | 47.837 | 43.287 | 1.709 | 1.105 | 1.073–1.147 |
| 1048576 | sort_topk1024 | base | 47.096 | 47.100 | 3.154 | 1.000 | 0.999–1.015 |
| 1048576 | sort_topk1024 | sorted | 46.831 | 46.596 | 1.827 | 1.005 | 0.993–1.016 |
| 1048576 | sort_topk1024 | nulls | 47.323 | 47.356 | 3.483 | 0.999 | 0.973–1.008 |

| Rows | Workload | Variant | First use baseline/candidate ms | Requested peak baseline/candidate bytes | Launches baseline/candidate |
|---:|---|---|---:|---:|---:|
| 4097 | sort_topk31 | base | 30.585/29.611 | 550998/208238 | 39/17 |
| 4097 | sort_topk31 | sorted | 29.629/29.261 | 550998/208238 | 39/17 |
| 4097 | sort_topk31 | nulls | 28.905/29.842 | 550998/208238 | 39/17 |
| 4097 | sort_topk1024 | base | 29.561/29.704 | 567134/307786 | 39/17 |
| 4097 | sort_topk1024 | sorted | 29.693/29.575 | 567134/307786 | 39/17 |
| 4097 | sort_topk1024 | nulls | 28.906/29.629 | 567134/307786 | 39/17 |
| 1048576 | sort_topk31 | base | 79.124/74.601 | 140805896/52464820 | 57/24 |
| 1048576 | sort_topk31 | sorted | 79.218/74.677 | 140805896/52464820 | 57/24 |
| 1048576 | sort_topk31 | nulls | 80.030/75.368 | 140805896/52464820 | 57/24 |
| 1048576 | sort_topk1024 | base | 79.058/79.890 | 140822032/52564368 | 57/24 |
| 1048576 | sort_topk1024 | sorted | 78.374/78.855 | 140822032/52564368 | 57/24 |
| 1048576 | sort_topk1024 | nulls | 80.479/80.760 | 140822032/52564368 | 57/24 |

First use is a fresh query process with runtime initialization; it is not a cold-machine measurement.

These are measured development cases, not a general GPU-over-CPU claim. Collection timings include staging and result construction. Profile memory is requested allocation accounting, not measured device residency. Retain build receipts and raw samples with this derived report.
