Paired complete-query warm medians. A ratio above 1 means the candidate was faster than the baseline. Per-round ranges show variability; they are not confidence intervals. CPU timings and profile counters are reported separately in the JSON.

| Rows | Workload | Variant | Baseline ms | Candidate ms | CPU ms | Ratio | Round ratio range |
|---:|---|---|---:|---:|---:|---:|---:|
| 4097 | sort_radix | base | 6.624 | 5.901 | 0.169 | 1.123 | 1.038–1.145 |
| 4097 | sort_radix | sorted | 6.741 | 5.827 | 0.069 | 1.157 | 1.149–1.245 |
| 4097 | sort_radix | nulls | 6.682 | 6.112 | 0.181 | 1.093 | 0.999–1.176 |
| 4097 | sort_topk31 | base | 6.095 | 4.742 | 0.040 | 1.285 | 1.258–1.397 |
| 4097 | sort_topk31 | sorted | 6.388 | 4.579 | 0.034 | 1.395 | 1.273–1.436 |
| 4097 | sort_topk31 | nulls | 6.146 | 4.677 | 0.049 | 1.314 | 1.247–1.314 |
| 4097 | sort_topk1024 | base | 6.301 | 4.965 | 0.147 | 1.269 | 1.126–1.316 |
| 4097 | sort_topk1024 | sorted | 6.592 | 4.911 | 0.056 | 1.342 | 1.281–1.397 |
| 4097 | sort_topk1024 | nulls | 6.510 | 5.129 | 0.155 | 1.269 | 1.253–1.297 |
| 1048576 | sort_radix | base | 338.280 | 211.320 | 10.041 | 1.601 | 1.589–1.606 |
| 1048576 | sort_radix | sorted | 327.478 | 209.435 | 5.361 | 1.564 | 1.520–1.631 |
| 1048576 | sort_radix | nulls | 325.756 | 212.279 | 14.557 | 1.535 | 1.522–1.563 |
| 1048576 | sort_topk31 | base | 172.419 | 42.679 | 1.037 | 4.040 | 4.006–4.072 |
| 1048576 | sort_topk31 | sorted | 172.634 | 42.315 | 1.246 | 4.080 | 4.008–4.160 |
| 1048576 | sort_topk31 | nulls | 162.845 | 43.018 | 1.974 | 3.786 | 3.670–3.952 |
| 1048576 | sort_topk1024 | base | 174.535 | 47.724 | 2.575 | 3.657 | 3.608–3.854 |
| 1048576 | sort_topk1024 | sorted | 169.907 | 46.742 | 1.599 | 3.635 | 3.653–3.725 |
| 1048576 | sort_topk1024 | nulls | 172.134 | 46.847 | 3.374 | 3.674 | 3.551–3.867 |

| Rows | Workload | Variant | First use baseline/candidate ms | Requested peak baseline/candidate bytes | Launches baseline/candidate |
|---:|---|---|---:|---:|---:|
| 4097 | sort_radix | base | 30.688/31.005 | 615856/617072 | 16/39 |
| 4097 | sort_radix | sorted | 29.338/30.890 | 615856/617072 | 16/39 |
| 4097 | sort_radix | nulls | 30.120/29.505 | 615856/617072 | 16/39 |
| 4097 | sort_topk31 | base | 29.233/29.350 | 549782/208238 | 16/17 |
| 4097 | sort_topk31 | sorted | 29.726/29.480 | 549782/208238 | 16/17 |
| 4097 | sort_topk31 | nulls | 29.223/29.569 | 549782/208238 | 16/17 |
| 4097 | sort_topk1024 | base | 29.464/30.448 | 565918/307786 | 16/17 |
| 4097 | sort_topk1024 | sorted | 29.789/29.656 | 565918/307786 | 16/17 |
| 4097 | sort_topk1024 | nulls | 29.666/29.987 | 565918/307786 | 16/17 |
| 1048576 | sort_radix | base | 362.621/236.394 | 157581456/157844752 | 23/57 |
| 1048576 | sort_radix | sorted | 370.324/232.557 | 157581456/157844752 | 23/57 |
| 1048576 | sort_radix | nulls | 340.961/237.161 | 157581456/157844752 | 23/57 |
| 1048576 | sort_topk31 | base | 187.982/75.771 | 140542600/52464820 | 23/24 |
| 1048576 | sort_topk31 | sorted | 187.362/80.665 | 140542600/52464820 | 23/24 |
| 1048576 | sort_topk31 | nulls | 183.124/76.178 | 140542600/52464820 | 23/24 |
| 1048576 | sort_topk1024 | base | 189.365/79.702 | 140558736/52564368 | 23/24 |
| 1048576 | sort_topk1024 | sorted | 192.506/83.825 | 140558736/52564368 | 23/24 |
| 1048576 | sort_topk1024 | nulls | 186.618/81.685 | 140558736/52564368 | 23/24 |

First use is a fresh query process with runtime initialization; it is not a cold-machine measurement.

These are measured development cases, not a general GPU-over-CPU claim. Collection timings include staging and result construction. Profile memory is requested allocation accounting, not measured device residency. Retain build receipts and raw samples with this derived report.
