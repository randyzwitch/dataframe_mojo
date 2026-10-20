Paired complete-query warm medians. A ratio above 1 means the candidate was faster than the baseline. Per-round ranges show variability; they are not confidence intervals. CPU timings and profile counters are reported separately in the JSON.

| Rows | Workload | Variant | Baseline ms | Candidate ms | CPU ms | Ratio | Round ratio range |
|---:|---|---|---:|---:|---:|---:|---:|
| 4097 | join_hash | base | 8.570 | 7.390 | 0.284 | 1.160 | 1.160–1.171 |
| 4097 | join_hash | sorted | 8.632 | 7.476 | 0.316 | 1.155 | 1.050–1.170 |
| 4097 | join_hash | nulls | 8.487 | 8.035 | 0.294 | 1.056 | 1.069–1.150 |
| 4097 | join_fanout | base | 10.860 | 8.473 | 0.340 | 1.282 | 1.249–1.337 |
| 4097 | join_fanout | sorted | 11.200 | 8.101 | 0.332 | 1.383 | 1.248–1.480 |
| 4097 | join_fanout | nulls | 10.462 | 8.437 | 0.363 | 1.240 | 1.228–1.243 |
| 262144 | join_hash | base | 170.756 | 71.756 | 8.906 | 2.380 | 2.375–2.438 |
| 262144 | join_hash | sorted | 168.229 | 67.616 | 8.389 | 2.488 | 2.306–2.576 |
| 262144 | join_hash | nulls | 164.591 | 69.984 | 9.401 | 2.352 | 2.336–2.462 |
| 262144 | join_fanout | base | 287.206 | 107.496 | 12.546 | 2.672 | 2.620–2.687 |
| 262144 | join_fanout | sorted | 283.917 | 109.285 | 9.419 | 2.598 | 2.592–2.632 |
| 262144 | join_fanout | nulls | 268.301 | 102.928 | 10.033 | 2.607 | 2.590–2.626 |

| Rows | Workload | Variant | First use baseline/candidate ms | Requested peak baseline/candidate bytes | Launches baseline/candidate |
|---:|---|---|---:|---:|---:|
| 4097 | join_hash | base | 32.436/31.983 | 2019909/895077 | 18/29 |
| 4097 | join_hash | sorted | 32.047/31.464 | 2019909/895077 | 18/29 |
| 4097 | join_hash | nulls | 32.120/31.679 | 1957956/885490 | 18/29 |
| 4097 | join_fanout | base | 34.876/32.574 | 3250278/1125196 | 18/29 |
| 4097 | join_fanout | sorted | 34.580/32.274 | 3250278/1125196 | 18/29 |
| 4097 | join_fanout | nulls | 35.249/32.234 | 3065316/1093522 | 18/29 |
| 262144 | join_hash | base | 195.970/94.731 | 129180172/48857190 | 23/34 |
| 262144 | join_hash | sorted | 192.375/94.854 | 129180172/48857190 | 23/34 |
| 262144 | join_hash | nulls | 188.784/94.644 | 125237941/48247660 | 23/34 |
| 262144 | join_fanout | base | 318.559/133.754 | 208019980/63602790 | 23/34 |
| 262144 | join_fanout | sorted | 320.084/134.554 | 208019980/63602790 | 23/34 |
| 262144 | join_fanout | nulls | 297.220/128.934 | 196193287/61577592 | 23/34 |

First use is a fresh query process with runtime initialization; it is not a cold-machine measurement.

These are measured development cases, not a general GPU-over-CPU claim. Collection timings include staging and result construction. Profile memory is requested allocation accounting, not measured device residency. Retain build receipts and raw samples with this derived report.
