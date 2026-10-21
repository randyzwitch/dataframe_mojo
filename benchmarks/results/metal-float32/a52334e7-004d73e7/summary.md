Paired complete-query warm medians. A ratio above 1 means the candidate was faster than the baseline. Per-round ranges show variability; they are not confidence intervals. CPU timings and profile counters are reported separately in the JSON.

| Rows | Workload | Variant | Baseline ms | Candidate ms | CPU ms | Ratio | Round ratio range |
|---:|---|---|---:|---:|---:|---:|---:|
| 4097 | float32_affine_sum | base | 6.114 | 1.219 | 0.082 | 5.016 | 4.874–6.039 |
| 4097 | float32_affine_sum | sorted | 6.217 | 1.026 | 0.083 | 6.059 | 5.704–6.265 |
| 4097 | float32_affine_sum | nulls | 6.279 | 1.239 | 0.081 | 5.068 | 4.721–5.591 |
| 4097 | filtered_float32_affine_sum | base | 7.299 | 1.120 | 0.088 | 6.517 | 5.841–7.328 |
| 4097 | filtered_float32_affine_sum | sorted | 6.998 | 1.202 | 0.091 | 5.822 | 5.011–6.995 |
| 4097 | filtered_float32_affine_sum | nulls | 7.118 | 1.460 | 0.102 | 4.875 | 4.648–5.780 |
| 1048576 | float32_affine_sum | base | 26.989 | 6.424 | 2.778 | 4.201 | 3.941–4.378 |
| 1048576 | float32_affine_sum | sorted | 28.295 | 6.273 | 3.022 | 4.511 | 4.492–4.843 |
| 1048576 | float32_affine_sum | nulls | 28.108 | 7.787 | 2.208 | 3.610 | 3.463–3.668 |
| 1048576 | filtered_float32_affine_sum | base | 31.698 | 5.138 | 1.766 | 6.169 | 6.075–6.202 |
| 1048576 | filtered_float32_affine_sum | sorted | 32.420 | 5.656 | 2.417 | 5.732 | 5.539–5.806 |
| 1048576 | filtered_float32_affine_sum | nulls | 32.735 | 7.986 | 2.763 | 4.099 | 3.927–4.189 |

| Rows | Workload | Variant | First use baseline/candidate ms | Requested peak baseline/candidate bytes | Launches baseline/candidate |
|---:|---|---|---:|---:|---:|
| 4097 | float32_affine_sum | base | 35.998/25.325 | 345847/16629 | 4/2 |
| 4097 | float32_affine_sum | sorted | 36.069/25.521 | 345847/16629 | 4/2 |
| 4097 | float32_affine_sum | nulls | 36.064/25.813 | 345847/17142 | 4/2 |
| 4097 | filtered_float32_affine_sum | base | 36.903/25.541 | 792876/16629 | 8/2 |
| 4097 | filtered_float32_affine_sum | sorted | 36.990/25.423 | 792876/16629 | 8/2 |
| 4097 | filtered_float32_affine_sum | nulls | 37.048/25.722 | 792876/17142 | 8/2 |
| 1048576 | float32_affine_sum | base | 60.635/30.780 | 88146531/4206629 | 4/2 |
| 1048576 | float32_affine_sum | sorted | 58.721/30.731 | 88146531/4206629 | 4/2 |
| 1048576 | float32_affine_sum | nulls | 60.352/32.181 | 88146531/4337701 | 4/2 |
| 1048576 | filtered_float32_affine_sum | base | 62.105/29.426 | 202474403/4206629 | 8/2 |
| 1048576 | filtered_float32_affine_sum | sorted | 62.172/30.130 | 202474403/4206629 | 8/2 |
| 1048576 | filtered_float32_affine_sum | nulls | 61.807/32.624 | 202474403/4337701 | 8/2 |

First use is a fresh query process with runtime initialization; it is not a cold-machine measurement.

These are measured development cases, not a general GPU-over-CPU claim. Collection timings include staging and result construction. Profile memory is requested allocation accounting, not measured device residency. Retain build receipts and raw samples with this derived report.
