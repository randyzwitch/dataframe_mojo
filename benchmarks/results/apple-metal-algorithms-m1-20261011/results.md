# Apple Metal development measurements

Development mechanisms only, not external-suite evidence or a general GPU speedup. Chain32 includes fusion advantages over the current CPU executor.

Times below are medians of complete warm collections, including allocation, staging, synchronization, and CPU result construction. First-use and all individual samples remain in the JSON. Every selected failed or unsupported case remains visible.

| Runtime | Rows | Workload | Variant | CPU ms | Metal ms | CPU/Metal | Launches | Shared MiB |
|---|---:|---|---|---:|---:|---:|---:|---:|
| baseline | 65536 | group_hash | base | 1.210 | 3.260 | 0.37 | 46 | 11.19 |
| baseline | 65536 | group_hash | nulls | 1.514 | 2.583 | 0.59 | 46 | 11.19 |
| baseline | 65536 | group_hash | sorted | 1.381 | 2.130 | 0.65 | 46 | 11.19 |
| baseline | 65536 | group_high | base | 4.360 | 11.225 | 0.39 | 46 | 11.19 |
| baseline | 65536 | group_high | nulls | 4.588 | 11.364 | 0.40 | 46 | 11.19 |
| baseline | 65536 | group_high | sorted | 4.251 | 10.893 | 0.39 | 46 | 11.19 |
| baseline | 65536 | join_hash | base | 2.108 | 6.473 | 0.33 | 23 | 9.74 |
| baseline | 65536 | join_hash | nulls | 2.513 | 6.005 | 0.42 | 23 | 9.57 |
| baseline | 65536 | join_hash | sorted | 2.188 | 5.625 | 0.39 | 23 | 9.74 |
| baseline | 65536 | sort_radix | base | 2.733 | 2.435 | 1.12 | 22 | 3.54 |
| baseline | 65536 | sort_radix | nulls | 2.694 | 2.429 | 1.11 | 22 | 3.54 |
| baseline | 65536 | sort_radix | sorted | 0.618 | 2.198 | 0.28 | 22 | 3.54 |
| baseline | 1048576 | group_hash | base | 9.366 | 42.929 | 0.22 | 54 | 179.00 |
| baseline | 1048576 | group_hash | nulls | 11.517 | 41.371 | 0.28 | 54 | 179.00 |
| baseline | 1048576 | group_hash | sorted | 6.023 | 18.087 | 0.33 | 54 | 179.00 |
| baseline | 1048576 | group_high | base | 37.157 | 198.347 | 0.19 | 54 | 179.00 |
| baseline | 1048576 | group_high | nulls | 41.630 | 199.976 | 0.21 | 54 | 179.00 |
| baseline | 1048576 | group_high | sorted | 37.437 | 193.708 | 0.19 | 54 | 179.00 |
| baseline | 1048576 | join_hash | base | 28.155 | 97.774 | 0.29 | 27 | 155.81 |
| baseline | 1048576 | join_hash | nulls | 29.593 | 98.230 | 0.30 | 27 | 153.04 |
| baseline | 1048576 | join_hash | sorted | 24.186 | 73.775 | 0.33 | 27 | 155.81 |
| baseline | 1048576 | sort_radix | base | 9.836 | 28.123 | 0.35 | 26 | 56.63 |
| baseline | 1048576 | sort_radix | nulls | 12.271 | 27.693 | 0.44 | 26 | 56.63 |
| baseline | 1048576 | sort_radix | sorted | 4.806 | 24.241 | 0.20 | 26 | 56.63 |
| metal | 65536 | group_hash | base | 1.185 | 2.498 | 0.47 | 70 | 12.45 |
| metal | 65536 | group_hash | nulls | 1.524 | 2.499 | 0.61 | 70 | 12.45 |
| metal | 65536 | group_hash | sorted | 1.393 | 2.450 | 0.57 | 70 | 12.45 |
| metal | 65536 | group_high | base | 5.191 | 12.117 | 0.43 | 70 | 12.45 |
| metal | 65536 | group_high | nulls | 4.739 | 11.922 | 0.40 | 70 | 12.45 |
| metal | 65536 | group_high | sorted | 4.590 | 11.618 | 0.40 | 70 | 12.45 |
| metal | 65536 | join_hash | base | 2.131 | 4.106 | 0.52 | 25 | 12.00 |
| metal | 65536 | join_hash | nulls | 2.559 | 4.048 | 0.63 | 25 | 11.83 |
| metal | 65536 | join_hash | sorted | 2.162 | 4.018 | 0.54 | 25 | 12.00 |
| metal | 65536 | sort_radix | base | 2.621 | 2.966 | 0.88 | 36 | 4.30 |
| metal | 65536 | sort_radix | nulls | 2.787 | 2.969 | 0.94 | 36 | 4.30 |
| metal | 65536 | sort_radix | sorted | 0.630 | 2.894 | 0.22 | 36 | 4.30 |
| metal | 1048576 | group_hash | base | 6.134 | 22.791 | 0.27 | 81 | 199.25 |
| metal | 1048576 | group_hash | nulls | 7.272 | 22.503 | 0.32 | 81 | 199.25 |
| metal | 1048576 | group_hash | sorted | 5.724 | 18.105 | 0.32 | 81 | 199.25 |
| metal | 1048576 | group_high | base | 35.349 | 213.543 | 0.17 | 81 | 199.25 |
| metal | 1048576 | group_high | nulls | 39.077 | 217.047 | 0.18 | 81 | 199.25 |
| metal | 1048576 | group_high | sorted | 35.855 | 206.631 | 0.17 | 81 | 199.25 |
| metal | 1048576 | join_hash | base | 27.886 | 50.941 | 0.55 | 28 | 192.06 |
| metal | 1048576 | join_hash | nulls | 28.909 | 49.702 | 0.58 | 28 | 189.29 |
| metal | 1048576 | join_hash | sorted | 24.252 | 49.369 | 0.49 | 28 | 192.06 |
| metal | 1048576 | sort_radix | base | 9.832 | 30.005 | 0.33 | 36 | 68.88 |
| metal | 1048576 | sort_radix | nulls | 11.722 | 30.427 | 0.39 | 36 | 68.88 |
| metal | 1048576 | sort_radix | sorted | 5.816 | 27.772 | 0.21 | 36 | 68.88 |

## Worst measured variant per case

Lowest CPU/Metal ratio among the selected variants; ratios below one favor CPU. These are per-mechanism results, not an aggregate suite score.

| Runtime | Rows | Workload | Worst variant | CPU/Metal |
|---|---:|---|---|---:|
| baseline | 65536 | group_hash | base | 0.37 |
| baseline | 65536 | group_high | base | 0.39 |
| baseline | 65536 | join_hash | base | 0.33 |
| baseline | 65536 | sort_radix | sorted | 0.28 |
| baseline | 1048576 | group_hash | base | 0.22 |
| baseline | 1048576 | group_high | base | 0.19 |
| baseline | 1048576 | join_hash | base | 0.29 |
| baseline | 1048576 | sort_radix | sorted | 0.20 |
| metal | 65536 | group_hash | base | 0.47 |
| metal | 65536 | group_high | sorted | 0.40 |
| metal | 65536 | join_hash | base | 0.52 |
| metal | 65536 | sort_radix | sorted | 0.22 |
| metal | 1048576 | group_hash | base | 0.27 |
| metal | 1048576 | group_high | base | 0.17 |
| metal | 1048576 | join_hash | sorted | 0.49 |
| metal | 1048576 | sort_radix | sorted | 0.21 |

Process outcomes: {'ok': 96, 'unsupported': 0, 'failed': 0}.

## Provenance and limits

```json
{
  "recorded_at_utc": "2026-10-11T14:38:02.313283+00:00",
  "platform": "macOS-26.5.2-arm64-arm-64bit-Mach-O",
  "machine": "arm64",
  "cpu": "Apple M1",
  "physical_cores": "8",
  "memory_bytes": "17179869184",
  "git_commit": "576ee8181bbba21f5a9db8170e85d547054c2be2",
  "engine_status": "",
  "benchmark_status": "M benchmarks/bench_metal.mojo\n M scripts/bench_metal.py",
  "mojo_version": "Mojo 1.2.0.dev2026092105 (e9569894)",
  "xcode": "Xcode 26.6\nBuild version 17F113",
  "sdk": "26.5",
  "compile_flags": [
    "-O3",
    "-I",
    "."
  ],
  "dataframe_threads": 8,
  "binary_sha256": "4489bf860d4d520b18a4db2451b11e75f9b3ae2dc3087e938e96948f558bbcf0",
  "source_sha256": "16a92f31c634a5c3f50913cbf414e6f2bead6fca7845c5c094558dc03bb11f0d",
  "driver_sha256": "17dca7a4149ad5652ac268691a7c5f1bc32b0488fb1db5e9a5d33ae013c634b6",
  "library_sha256": {
    "metal": "364d775c9b7d9aa2699bb65ad4846876c61e096a6f382913825c87644459f1eb",
    "baseline": "b11f935b9941478e15266bc1e97b1b17e59e2d42f4dfc09c2b4c6f46c8b079b4"
  },
  "environment": {},
  "method": "Fresh process per case/build/round; first-use collections retained separately; two extra warmups per engine; alternating CPU/Metal timed order; process/build order rotated per round; inputs and exact CPU-reference validation untimed; complete result materialization timed; result release untimed.",
  "scope": "Development mechanisms only, not external-suite evidence or a general GPU speedup. Chain32 includes fusion advantages over the current CPU executor.",
  "data": "Row mechanisms use a deterministic 1024-value numeric distribution. Algorithm cases use Int32 keys: group_hash has min(N,1024) classes; group_high, join_hash and sort_radix have max(1,N/2) classes. Base keys are (37*i) modulo cardinality; sorted keys are monotone bins. Null variants invalidate every twentieth key; grouped values additionally invalidate every thirteenth row. Grouped sums use i modulo 17 minus 8; sort and self-left-join carry original Int64 row IDs. Float32 row values are exact multiples of 1/1024.",
  "limitations": [
    "No external-suite coverage claimed; Float64 and floating reductions are outside these cases. Algorithm comparisons preserve complete output order and values against CPU.",
    "First use includes context/shader initialization but is not a cold machine or cold filesystem measurement.",
    "CPU and GPU share host memory and power limits; short timings can vary with scheduler and thermal state.",
    "Native profile is a separate warm collection after timed samples; its GPU interval is not a caller-wall substitute.",
    "Workload exposure: chain32 was used during fusion development; other cases broaden mechanism coverage without a holdout claim."
  ]
}
```
