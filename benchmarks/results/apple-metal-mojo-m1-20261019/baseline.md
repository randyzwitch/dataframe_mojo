# Apple Metal development measurements

Development mechanisms only, not external-suite evidence or a general GPU speedup. Chain32 includes fusion advantages over the current CPU executor.

Times below are medians of complete warm collections, including allocation, staging, synchronization, and CPU result construction. First-use and all individual samples remain in the JSON. Every selected failed or unsupported case remains visible.

| Runtime | Rows | Workload | Variant | CPU ms | Metal ms | CPU/Metal | Launches | Peak payload MiB |
|---|---:|---|---|---:|---:|---:|---:|---:|
| mojo | 4097 | expression15 | base | 1.123 | 6.742 | 0.17 | 1 | 0.37 |
| mojo | 4097 | expression15 | nulls | 1.142 | 6.833 | 0.17 | 1 | 0.37 |
| mojo | 4097 | filter_half | base | 0.082 | 6.721 | 0.01 | 5 | 0.79 |
| mojo | 4097 | filter_half | nulls | 0.106 | 6.313 | 0.02 | 5 | 0.79 |
| mojo | 4097 | filtered_int_sum | base | 0.070 | 7.792 | 0.01 | 9 | 0.76 |
| mojo | 4097 | filtered_int_sum | nulls | 0.099 | 7.491 | 0.01 | 9 | 0.76 |
| mojo | 4097 | group_hash | base | 0.228 | 10.982 | 0.02 | 35 | 2.36 |
| mojo | 4097 | group_hash | nulls | 0.259 | 11.369 | 0.02 | 36 | 2.36 |
| mojo | 4097 | group_high | base | 0.283 | 11.922 | 0.02 | 36 | 2.36 |
| mojo | 4097 | group_high | nulls | 0.324 | 11.852 | 0.03 | 37 | 2.36 |
| mojo | 4097 | int_sum | base | 0.036 | 6.676 | 0.01 | 4 | 0.33 |
| mojo | 4097 | int_sum | nulls | 0.044 | 6.829 | 0.01 | 4 | 0.33 |
| mojo | 4097 | join_hash | base | 0.275 | 8.521 | 0.03 | 18 | 1.93 |
| mojo | 4097 | join_hash | nulls | 0.300 | 8.532 | 0.04 | 18 | 1.87 |
| mojo | 4097 | projection | base | 0.071 | 6.077 | 0.01 | 1 | 0.36 |
| mojo | 4097 | projection | nulls | 0.074 | 5.654 | 0.01 | 1 | 0.36 |
| mojo | 4097 | sort_radix | base | 0.172 | 6.739 | 0.03 | 16 | 0.59 |
| mojo | 4097 | sort_radix | nulls | 0.208 | 6.597 | 0.03 | 16 | 0.59 |
| mojo | 65536 | expression15 | base | 5.536 | 19.157 | 0.29 | 1 | 5.76 |
| mojo | 65536 | expression15 | nulls | 5.579 | 19.026 | 0.29 | 1 | 5.76 |
| mojo | 65536 | filter_half | base | 0.758 | 10.100 | 0.08 | 5 | 12.57 |
| mojo | 65536 | filter_half | nulls | 0.891 | 10.249 | 0.09 | 5 | 12.57 |
| mojo | 65536 | filtered_int_sum | base | 0.454 | 9.197 | 0.05 | 9 | 12.08 |
| mojo | 65536 | filtered_int_sum | nulls | 0.556 | 9.294 | 0.06 | 9 | 12.08 |
| mojo | 65536 | group_hash | base | 3.015 | 22.702 | 0.13 | 38 | 37.71 |
| mojo | 65536 | group_hash | nulls | 2.462 | 25.680 | 0.10 | 39 | 37.71 |
| mojo | 65536 | group_high | base | 10.602 | 33.520 | 0.32 | 43 | 37.71 |
| mojo | 65536 | group_high | nulls | 8.954 | 39.552 | 0.23 | 44 | 37.71 |
| mojo | 65536 | int_sum | base | 0.196 | 7.673 | 0.03 | 4 | 5.27 |
| mojo | 65536 | int_sum | nulls | 0.193 | 7.724 | 0.02 | 4 | 5.27 |
| mojo | 65536 | join_hash | base | 2.922 | 48.952 | 0.06 | 21 | 30.80 |
| mojo | 65536 | join_hash | nulls | 2.949 | 47.857 | 0.06 | 21 | 29.86 |
| mojo | 65536 | projection | base | 0.392 | 11.794 | 0.03 | 1 | 5.76 |
| mojo | 65536 | projection | nulls | 0.388 | 11.600 | 0.03 | 1 | 5.76 |
| mojo | 65536 | sort_radix | base | 4.734 | 26.008 | 0.18 | 19 | 9.39 |
| mojo | 65536 | sort_radix | nulls | 4.051 | 25.811 | 0.16 | 19 | 9.39 |
| mojo | 1048576 | expression15 | base | 22.234 | 165.565 | 0.13 | 1 | 92.13 |
| mojo | 1048576 | expression15 | nulls | 24.267 | 166.644 | 0.15 | 1 | 92.13 |
| mojo | 1048576 | filter_half | base | 2.096 | 64.626 | 0.03 | 5 | 201.16 |
| mojo | 1048576 | filter_half | nulls | 4.511 | 66.447 | 0.07 | 5 | 201.16 |
| mojo | 1048576 | filtered_int_sum | base | 1.830 | 27.227 | 0.07 | 9 | 193.09 |
| mojo | 1048576 | filtered_int_sum | nulls | 2.191 | 27.217 | 0.08 | 9 | 193.09 |
| mojo | 1048576 | group_hash | base | 12.389 | 213.637 | 0.06 | 42 | 603.41 |
| mojo | 1048576 | group_hash | nulls | 13.412 | 281.898 | 0.05 | 43 | 603.41 |
| mojo | 1048576 | group_high | base | 79.298 | 355.060 | 0.22 | 51 | 603.41 |
| mojo | 1048576 | group_high | nulls | 78.147 | 452.098 | 0.17 | 52 | 603.41 |
| mojo | 1048576 | int_sum | base | 2.057 | 18.134 | 0.11 | 4 | 84.06 |
| mojo | 1048576 | int_sum | nulls | 0.392 | 18.638 | 0.02 | 4 | 84.06 |
| mojo | 1048576 | join_hash | base | 34.188 | 664.609 | 0.05 | 25 | 492.78 |
| mojo | 1048576 | join_hash | nulls | 32.517 | 641.736 | 0.05 | 25 | 477.74 |
| mojo | 1048576 | projection | base | 3.393 | 101.412 | 0.03 | 1 | 92.13 |
| mojo | 1048576 | projection | nulls | 2.853 | 103.470 | 0.03 | 1 | 92.13 |
| mojo | 1048576 | sort_radix | base | 10.835 | 327.558 | 0.03 | 23 | 150.28 |
| mojo | 1048576 | sort_radix | nulls | 12.521 | 320.066 | 0.04 | 23 | 150.28 |

## Worst measured variant per case

Lowest CPU/Metal ratio among the selected variants; ratios below one favor CPU. These are per-mechanism results, not an aggregate suite score.

| Runtime | Rows | Workload | Worst variant | CPU/Metal |
|---|---:|---|---|---:|
| mojo | 4097 | expression15 | base | 0.17 |
| mojo | 4097 | filter_half | base | 0.01 |
| mojo | 4097 | filtered_int_sum | base | 0.01 |
| mojo | 4097 | group_hash | base | 0.02 |
| mojo | 4097 | group_high | base | 0.02 |
| mojo | 4097 | int_sum | base | 0.01 |
| mojo | 4097 | join_hash | base | 0.03 |
| mojo | 4097 | projection | base | 0.01 |
| mojo | 4097 | sort_radix | base | 0.03 |
| mojo | 65536 | expression15 | base | 0.29 |
| mojo | 65536 | filter_half | base | 0.08 |
| mojo | 65536 | filtered_int_sum | base | 0.05 |
| mojo | 65536 | group_hash | nulls | 0.10 |
| mojo | 65536 | group_high | nulls | 0.23 |
| mojo | 65536 | int_sum | nulls | 0.02 |
| mojo | 65536 | join_hash | base | 0.06 |
| mojo | 65536 | projection | base | 0.03 |
| mojo | 65536 | sort_radix | nulls | 0.16 |
| mojo | 1048576 | expression15 | base | 0.13 |
| mojo | 1048576 | filter_half | base | 0.03 |
| mojo | 1048576 | filtered_int_sum | base | 0.07 |
| mojo | 1048576 | group_hash | nulls | 0.05 |
| mojo | 1048576 | group_high | nulls | 0.17 |
| mojo | 1048576 | int_sum | nulls | 0.02 |
| mojo | 1048576 | join_hash | nulls | 0.05 |
| mojo | 1048576 | projection | nulls | 0.03 |
| mojo | 1048576 | sort_radix | base | 0.03 |

Process outcomes: {'ok': 108, 'unsupported': 0, 'failed': 0}.

## Provenance and limits

```json
{
  "recorded_at_utc": "2026-10-19T20:14:00.616795+00:00",
  "platform": "macOS-26.5.2-arm64-arm-64bit-Mach-O",
  "machine": "arm64",
  "cpu": "Apple M1",
  "physical_cores": "8",
  "memory_bytes": "17179869184",
  "git_commit": "ce86f3dd03539cefb432b4309cfd4f77632ed142",
  "engine_status": "",
  "benchmark_status": "",
  "mojo_version": "Mojo 1.2.0.dev2026100905 (3b71af75)",
  "xcode": "Xcode 26.6\nBuild version 17F113",
  "sdk": "26.5",
  "compile_flags": [
    "-O3",
    "-I",
    "."
  ],
  "dataframe_threads": 8,
  "implementation": "mojo",
  "binaries_sha256": {
    "mojo": "6a1bfea27c06cfa9f9655c05c2ca9204258f275e39afda5352d197d31856f6c3"
  },
  "source_sha256": "779ef4339b6370f91270ff8844001654c071f23a1edd29100bd0a0c517a4998f",
  "driver_sha256": "3528ba955f20b027f2f1c43d752022c27324170362166db9a129579ccea03058",
  "environment": {},
  "method": "Fresh process per case/build/round; first-use collections retained separately; two extra warmups per engine; alternating CPU/Metal timed order; process/build order rotated per round; inputs and exact CPU-reference validation untimed; complete result materialization timed; result release untimed.",
  "scope": "Development mechanisms only, not external-suite evidence or a general GPU speedup. Chain32 includes fusion advantages over the current CPU executor.",
  "data": "Row mechanisms use a deterministic 1024-value numeric distribution. Algorithm cases use Int32 keys: group_hash has min(N,1024) classes; group_high, join_hash and sort_radix have max(1,N/2) classes. Base keys are (37*i) modulo cardinality; sorted keys are monotone bins. Null variants invalidate every twentieth key; grouped values additionally invalidate every thirteenth row. Grouped sums use i modulo 17 minus 8; sort and self-left-join carry original Int64 row IDs. Float32 row values are exact multiples of 1/1024.",
  "limitations": [
    "No external-suite coverage claimed; Float64 and floating reductions are outside these cases. Algorithm comparisons preserve complete output order and values against CPU.",
    "First use includes context/shader initialization but is not a cold machine or cold filesystem measurement.",
    "CPU and GPU share host memory and power limits; short timings can vary with scheduler and thermal state.",
    "The Mojo profile is a separate warm collection after timed samples; complete caller-wall timing remains the performance measure.",
    "Workload exposure: chain32 was used during fusion development; other cases broaden mechanism coverage without a holdout claim."
  ]
}
```
