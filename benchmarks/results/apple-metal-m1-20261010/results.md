# Apple Metal development measurements

Development mechanisms only, not external-suite evidence or a general GPU speedup. Chain32 includes fusion advantages over the current CPU executor.

Times below are medians of complete warm collections, including allocation, staging, synchronization, and CPU result construction. First-use and all individual samples remain in the JSON. Every selected failed or unsupported case remains visible.

| Runtime | Rows | Workload | Variant | CPU ms | Metal ms | CPU/Metal | Launches | Shared MiB |
|---|---:|---|---|---:|---:|---:|---:|---:|
| baseline | 65536 | chain32 | base | 22.230 | 1.550 | 14.34 | 34 | 10.34 |
| baseline | 65536 | chain32 | nulls | 22.206 | 1.638 | 13.56 | 34 | 10.34 |
| baseline | 65536 | chain32 | sorted | 22.239 | 1.625 | 13.69 | 34 | 10.34 |
| baseline | 65536 | count | base | 0.172 | 0.762 | 0.23 | 9 | 2.80 |
| baseline | 65536 | count | nulls | 0.367 | 0.751 | 0.49 | 9 | 2.80 |
| baseline | 65536 | count | sorted | 0.160 | 0.720 | 0.22 | 9 | 2.80 |
| baseline | 65536 | expression15 | base | 7.165 | 0.707 | 10.13 | 3 | 0.65 |
| baseline | 65536 | expression15 | nulls | 7.176 | 0.661 | 10.85 | 3 | 0.65 |
| baseline | 65536 | expression15 | sorted | 7.185 | 0.709 | 10.13 | 3 | 0.65 |
| baseline | 65536 | filter_half | base | 0.251 | 0.620 | 0.40 | 7 | 2.15 |
| baseline | 65536 | filter_half | nulls | 0.472 | 0.640 | 0.74 | 7 | 2.15 |
| baseline | 65536 | filter_half | sorted | 0.254 | 0.629 | 0.40 | 7 | 2.15 |
| baseline | 65536 | filter_sparse | base | 0.164 | 0.618 | 0.26 | 7 | 2.15 |
| baseline | 65536 | filter_sparse | nulls | 0.172 | 0.615 | 0.28 | 7 | 2.15 |
| baseline | 65536 | filter_sparse | sorted | 0.169 | 0.649 | 0.26 | 7 | 2.15 |
| baseline | 65536 | int_sum | base | 0.113 | 0.509 | 0.22 | 4 | 0.65 |
| baseline | 65536 | int_sum | nulls | 0.124 | 0.544 | 0.23 | 4 | 0.65 |
| baseline | 65536 | int_sum | sorted | 0.119 | 0.579 | 0.21 | 4 | 0.65 |
| baseline | 65536 | projection | base | 0.593 | 0.532 | 1.11 | 3 | 0.65 |
| baseline | 65536 | projection | nulls | 0.589 | 0.511 | 1.15 | 3 | 0.65 |
| baseline | 65536 | projection | sorted | 0.582 | 0.533 | 1.09 | 3 | 0.65 |
| baseline | 1048576 | chain32 | base | 97.168 | 15.646 | 6.21 | 34 | 165.38 |
| baseline | 1048576 | chain32 | nulls | 92.066 | 15.778 | 5.83 | 34 | 165.38 |
| baseline | 1048576 | chain32 | sorted | 95.100 | 15.879 | 5.99 | 34 | 165.38 |
| baseline | 1048576 | count | base | 1.827 | 4.537 | 0.40 | 9 | 44.67 |
| baseline | 1048576 | count | nulls | 1.854 | 4.677 | 0.40 | 9 | 44.67 |
| baseline | 1048576 | count | sorted | 1.440 | 4.521 | 0.32 | 9 | 44.67 |
| baseline | 1048576 | expression15 | base | 27.434 | 3.196 | 8.59 | 3 | 10.38 |
| baseline | 1048576 | expression15 | nulls | 26.875 | 3.186 | 8.44 | 3 | 10.38 |
| baseline | 1048576 | expression15 | sorted | 25.811 | 3.170 | 8.14 | 3 | 10.38 |
| baseline | 1048576 | filter_half | base | 1.675 | 4.061 | 0.41 | 7 | 34.39 |
| baseline | 1048576 | filter_half | nulls | 2.016 | 4.135 | 0.49 | 7 | 34.39 |
| baseline | 1048576 | filter_half | sorted | 1.726 | 4.123 | 0.42 | 7 | 34.39 |
| baseline | 1048576 | filter_sparse | base | 1.841 | 3.478 | 0.53 | 7 | 34.39 |
| baseline | 1048576 | filter_sparse | nulls | 1.500 | 3.502 | 0.43 | 7 | 34.39 |
| baseline | 1048576 | filter_sparse | sorted | 1.594 | 3.454 | 0.46 | 7 | 34.39 |
| baseline | 1048576 | int_sum | base | 1.084 | 1.763 | 0.61 | 4 | 10.39 |
| baseline | 1048576 | int_sum | nulls | 1.020 | 1.732 | 0.59 | 4 | 10.39 |
| baseline | 1048576 | int_sum | sorted | 0.669 | 1.727 | 0.39 | 4 | 10.39 |
| baseline | 1048576 | projection | base | 3.302 | 2.171 | 1.52 | 3 | 10.38 |
| baseline | 1048576 | projection | nulls | 2.860 | 2.170 | 1.32 | 3 | 10.38 |
| baseline | 1048576 | projection | sorted | 2.998 | 2.196 | 1.36 | 3 | 10.38 |
| baseline | 8388608 | chain32 | base | 528.717 | 118.171 | 4.47 | 34 | 1323.00 |
| baseline | 8388608 | chain32 | nulls | 521.148 | 119.209 | 4.37 | 34 | 1323.00 |
| baseline | 8388608 | chain32 | sorted | 514.265 | 116.084 | 4.43 | 34 | 1323.00 |
| baseline | 8388608 | count | base | 3.061 | 28.663 | 0.11 | 9 | 357.16 |
| baseline | 8388608 | count | nulls | 6.990 | 28.991 | 0.24 | 9 | 357.16 |
| baseline | 8388608 | count | sorted | 2.010 | 28.958 | 0.07 | 9 | 357.16 |
| baseline | 8388608 | expression15 | base | 180.875 | 20.087 | 9.00 | 3 | 83.00 |
| baseline | 8388608 | expression15 | nulls | 176.786 | 19.987 | 8.85 | 3 | 83.00 |
| baseline | 8388608 | expression15 | sorted | 179.668 | 20.564 | 8.74 | 3 | 83.00 |
| baseline | 8388608 | filter_half | base | 5.006 | 29.762 | 0.17 | 7 | 275.13 |
| baseline | 8388608 | filter_half | nulls | 10.275 | 29.038 | 0.35 | 7 | 275.13 |
| baseline | 8388608 | filter_half | sorted | 4.912 | 27.464 | 0.18 | 7 | 275.13 |
| baseline | 8388608 | filter_sparse | base | 2.107 | 23.218 | 0.09 | 7 | 275.13 |
| baseline | 8388608 | filter_sparse | nulls | 2.071 | 23.224 | 0.09 | 7 | 275.13 |
| baseline | 8388608 | filter_sparse | sorted | 2.009 | 22.967 | 0.09 | 7 | 275.13 |
| baseline | 8388608 | int_sum | base | 1.788 | 9.226 | 0.19 | 4 | 83.02 |
| baseline | 8388608 | int_sum | nulls | 1.770 | 9.325 | 0.19 | 4 | 83.02 |
| baseline | 8388608 | int_sum | sorted | 1.406 | 9.401 | 0.15 | 4 | 83.02 |
| baseline | 8388608 | projection | base | 17.681 | 13.409 | 1.32 | 3 | 83.00 |
| baseline | 8388608 | projection | nulls | 18.468 | 13.086 | 1.41 | 3 | 83.00 |
| baseline | 8388608 | projection | sorted | 20.732 | 13.207 | 1.57 | 3 | 83.00 |
| metal | 65536 | chain32 | base | 22.331 | 0.983 | 22.72 | 3 | 0.65 |
| metal | 65536 | chain32 | nulls | 22.373 | 0.978 | 22.89 | 3 | 0.65 |
| metal | 65536 | chain32 | sorted | 22.328 | 0.871 | 25.65 | 3 | 0.65 |
| metal | 65536 | count | base | 0.167 | 0.807 | 0.21 | 8 | 2.80 |
| metal | 65536 | count | nulls | 0.361 | 0.734 | 0.49 | 8 | 2.80 |
| metal | 65536 | count | sorted | 0.163 | 0.729 | 0.22 | 8 | 2.80 |
| metal | 65536 | expression15 | base | 7.176 | 0.661 | 10.86 | 3 | 0.65 |
| metal | 65536 | expression15 | nulls | 7.167 | 0.659 | 10.87 | 3 | 0.65 |
| metal | 65536 | expression15 | sorted | 7.177 | 0.671 | 10.69 | 3 | 0.65 |
| metal | 65536 | filter_half | base | 0.257 | 0.647 | 0.40 | 7 | 2.15 |
| metal | 65536 | filter_half | nulls | 0.477 | 0.666 | 0.72 | 7 | 2.15 |
| metal | 65536 | filter_half | sorted | 0.256 | 0.664 | 0.39 | 7 | 2.15 |
| metal | 65536 | filter_sparse | base | 0.169 | 0.635 | 0.27 | 7 | 2.15 |
| metal | 65536 | filter_sparse | nulls | 0.170 | 0.616 | 0.28 | 7 | 2.15 |
| metal | 65536 | filter_sparse | sorted | 0.167 | 0.634 | 0.26 | 7 | 2.15 |
| metal | 65536 | int_sum | base | 0.110 | 0.508 | 0.22 | 4 | 0.65 |
| metal | 65536 | int_sum | nulls | 0.123 | 0.566 | 0.22 | 4 | 0.65 |
| metal | 65536 | int_sum | sorted | 0.116 | 0.551 | 0.21 | 4 | 0.65 |
| metal | 65536 | projection | base | 0.593 | 0.515 | 1.15 | 3 | 0.65 |
| metal | 65536 | projection | nulls | 0.602 | 0.534 | 1.13 | 3 | 0.65 |
| metal | 65536 | projection | sorted | 0.593 | 0.544 | 1.09 | 3 | 0.65 |
| metal | 1048576 | chain32 | base | 89.592 | 4.598 | 19.48 | 3 | 10.38 |
| metal | 1048576 | chain32 | nulls | 93.433 | 4.454 | 20.98 | 3 | 10.38 |
| metal | 1048576 | chain32 | sorted | 91.150 | 4.551 | 20.03 | 3 | 10.38 |
| metal | 1048576 | count | base | 1.524 | 4.546 | 0.34 | 8 | 44.67 |
| metal | 1048576 | count | nulls | 1.959 | 4.660 | 0.42 | 8 | 44.67 |
| metal | 1048576 | count | sorted | 2.429 | 4.416 | 0.55 | 8 | 44.67 |
| metal | 1048576 | expression15 | base | 28.429 | 3.083 | 9.22 | 3 | 10.38 |
| metal | 1048576 | expression15 | nulls | 25.419 | 3.149 | 8.07 | 3 | 10.38 |
| metal | 1048576 | expression15 | sorted | 28.059 | 3.199 | 8.77 | 3 | 10.38 |
| metal | 1048576 | filter_half | base | 2.417 | 3.983 | 0.61 | 7 | 34.39 |
| metal | 1048576 | filter_half | nulls | 2.134 | 4.096 | 0.52 | 7 | 34.39 |
| metal | 1048576 | filter_half | sorted | 1.813 | 3.965 | 0.46 | 7 | 34.39 |
| metal | 1048576 | filter_sparse | base | 1.675 | 3.459 | 0.48 | 7 | 34.39 |
| metal | 1048576 | filter_sparse | nulls | 1.656 | 3.547 | 0.47 | 7 | 34.39 |
| metal | 1048576 | filter_sparse | sorted | 1.869 | 3.435 | 0.54 | 7 | 34.39 |
| metal | 1048576 | int_sum | base | 0.860 | 1.749 | 0.49 | 4 | 10.39 |
| metal | 1048576 | int_sum | nulls | 1.170 | 1.752 | 0.67 | 4 | 10.39 |
| metal | 1048576 | int_sum | sorted | 1.214 | 1.738 | 0.70 | 4 | 10.39 |
| metal | 1048576 | projection | base | 2.919 | 2.213 | 1.32 | 3 | 10.38 |
| metal | 1048576 | projection | nulls | 2.914 | 2.225 | 1.31 | 3 | 10.38 |
| metal | 1048576 | projection | sorted | 3.412 | 2.206 | 1.55 | 3 | 10.38 |
| metal | 8388608 | chain32 | base | 532.382 | 25.674 | 20.74 | 3 | 83.00 |
| metal | 8388608 | chain32 | nulls | 515.293 | 26.154 | 19.70 | 3 | 83.00 |
| metal | 8388608 | chain32 | sorted | 524.591 | 26.009 | 20.17 | 3 | 83.00 |
| metal | 8388608 | count | base | 1.984 | 28.497 | 0.07 | 8 | 357.16 |
| metal | 8388608 | count | nulls | 6.707 | 28.685 | 0.23 | 8 | 357.16 |
| metal | 8388608 | count | sorted | 1.972 | 28.410 | 0.07 | 8 | 357.16 |
| metal | 8388608 | expression15 | base | 180.265 | 20.205 | 8.92 | 3 | 83.00 |
| metal | 8388608 | expression15 | nulls | 180.890 | 19.965 | 9.06 | 3 | 83.00 |
| metal | 8388608 | expression15 | sorted | 180.576 | 20.000 | 9.03 | 3 | 83.00 |
| metal | 8388608 | filter_half | base | 5.336 | 29.359 | 0.18 | 7 | 275.13 |
| metal | 8388608 | filter_half | nulls | 9.425 | 29.514 | 0.32 | 7 | 275.13 |
| metal | 8388608 | filter_half | sorted | 5.462 | 27.943 | 0.20 | 7 | 275.13 |
| metal | 8388608 | filter_sparse | base | 2.467 | 24.197 | 0.10 | 7 | 275.13 |
| metal | 8388608 | filter_sparse | nulls | 2.275 | 23.285 | 0.10 | 7 | 275.13 |
| metal | 8388608 | filter_sparse | sorted | 1.898 | 22.883 | 0.08 | 7 | 275.13 |
| metal | 8388608 | int_sum | base | 3.380 | 9.223 | 0.37 | 4 | 83.02 |
| metal | 8388608 | int_sum | nulls | 1.744 | 9.447 | 0.18 | 4 | 83.02 |
| metal | 8388608 | int_sum | sorted | 1.734 | 9.528 | 0.18 | 4 | 83.02 |
| metal | 8388608 | projection | base | 17.181 | 13.227 | 1.30 | 3 | 83.00 |
| metal | 8388608 | projection | nulls | 17.707 | 13.190 | 1.34 | 3 | 83.00 |
| metal | 8388608 | projection | sorted | 18.654 | 13.104 | 1.42 | 3 | 83.00 |

## Worst measured variant per case

Lowest CPU/Metal ratio among the selected variants; ratios below one favor CPU. These are per-mechanism results, not an aggregate suite score.

| Runtime | Rows | Workload | Worst variant | CPU/Metal |
|---|---:|---|---|---:|
| baseline | 65536 | chain32 | nulls | 13.56 |
| baseline | 65536 | count | sorted | 0.22 |
| baseline | 65536 | expression15 | base | 10.13 |
| baseline | 65536 | filter_half | sorted | 0.40 |
| baseline | 65536 | filter_sparse | sorted | 0.26 |
| baseline | 65536 | int_sum | sorted | 0.21 |
| baseline | 65536 | projection | sorted | 1.09 |
| baseline | 1048576 | chain32 | nulls | 5.83 |
| baseline | 1048576 | count | sorted | 0.32 |
| baseline | 1048576 | expression15 | sorted | 8.14 |
| baseline | 1048576 | filter_half | base | 0.41 |
| baseline | 1048576 | filter_sparse | nulls | 0.43 |
| baseline | 1048576 | int_sum | sorted | 0.39 |
| baseline | 1048576 | projection | nulls | 1.32 |
| baseline | 8388608 | chain32 | nulls | 4.37 |
| baseline | 8388608 | count | sorted | 0.07 |
| baseline | 8388608 | expression15 | sorted | 8.74 |
| baseline | 8388608 | filter_half | base | 0.17 |
| baseline | 8388608 | filter_sparse | sorted | 0.09 |
| baseline | 8388608 | int_sum | sorted | 0.15 |
| baseline | 8388608 | projection | base | 1.32 |
| metal | 65536 | chain32 | base | 22.72 |
| metal | 65536 | count | base | 0.21 |
| metal | 65536 | expression15 | sorted | 10.69 |
| metal | 65536 | filter_half | sorted | 0.39 |
| metal | 65536 | filter_sparse | sorted | 0.26 |
| metal | 65536 | int_sum | sorted | 0.21 |
| metal | 65536 | projection | sorted | 1.09 |
| metal | 1048576 | chain32 | base | 19.48 |
| metal | 1048576 | count | base | 0.34 |
| metal | 1048576 | expression15 | nulls | 8.07 |
| metal | 1048576 | filter_half | sorted | 0.46 |
| metal | 1048576 | filter_sparse | nulls | 0.47 |
| metal | 1048576 | int_sum | base | 0.49 |
| metal | 1048576 | projection | nulls | 1.31 |
| metal | 8388608 | chain32 | nulls | 19.70 |
| metal | 8388608 | count | sorted | 0.07 |
| metal | 8388608 | expression15 | base | 8.92 |
| metal | 8388608 | filter_half | base | 0.18 |
| metal | 8388608 | filter_sparse | sorted | 0.08 |
| metal | 8388608 | int_sum | sorted | 0.18 |
| metal | 8388608 | projection | base | 1.30 |

Process outcomes: {'ok': 252, 'unsupported': 0, 'failed': 0}.

## Provenance and limits

```json
{
  "recorded_at_utc": "2026-10-10T15:54:49.738782+00:00",
  "platform": "macOS-26.5.2-arm64-arm-64bit-Mach-O",
  "machine": "arm64",
  "cpu": "Apple M1",
  "physical_cores": "8",
  "memory_bytes": "17179869184",
  "git_commit": "4d443de4378d54dfd2f950c2ebde9e59d6a34a54",
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
  "binary_sha256": "7f6d57516a3658150d9c5c78088849b10b1334f82fea0b8072c1ef6b2c303e64",
  "source_sha256": "538615931e20e30b6596426b3e519a5d04bfac0dd950572f3376d8ae464333f5",
  "driver_sha256": "49710b6c9f0a73765c07c8a0240f8d1f459c330c892d70e35c0db66c47419fc3",
  "library_sha256": {
    "metal": "bbd16bc0605befbee090dff44ead764400dd9d30cbff320b0740fde357e57d30",
    "baseline": "10c6b147166363bddbd2296c3fee117e1996a43d9675ea5fc703d8a83b74e076"
  },
  "environment": {},
  "method": "Fresh process per case/build/round; first-use collections retained separately; two extra warmups per engine; alternating CPU/Metal timed order; process/build order rotated per round; inputs and exact CPU-reference validation untimed; complete result materialization timed; result release untimed.",
  "scope": "Development mechanisms only, not external-suite evidence or a general GPU speedup. Chain32 includes fusion advantages over the current CPU executor.",
  "data": "Deterministic 1024-value numeric distribution. Base cycles; sorted uses monotone Float32 or Int32 bins; nulls permutes bins with fixed LCG constants and marks exactly every twentieth row invalid. Float32 values are exact multiples of 1/1024; Int32 values are bin modulo 17 minus 8.",
  "limitations": [
    "No external-suite coverage claimed; joins, group-by, Float64 and floating reductions are outside these cases.",
    "First use includes context/shader initialization but is not a cold machine or cold filesystem measurement.",
    "CPU and GPU share host memory and power limits; short timings can vary with scheduler and thermal state.",
    "Native profile is a separate warm collection after timed samples; its GPU interval is not a caller-wall substitute.",
    "Workload exposure: chain32 was used during fusion development; other cases broaden mechanism coverage without a holdout claim."
  ]
}
```
