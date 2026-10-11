# Apple Metal development measurements

Development mechanisms only, not external-suite evidence or a general GPU speedup. Chain32 includes fusion advantages over the current CPU executor.

Times below are medians of complete warm collections, including allocation, staging, synchronization, and CPU result construction. First-use and all individual samples remain in the JSON. Every selected failed or unsupported case remains visible.

| Runtime | Rows | Workload | Variant | CPU ms | Metal ms | CPU/Metal | Launches | Shared MiB |
|---|---:|---|---|---:|---:|---:|---:|---:|
| baseline | 1048576 | count | base | 1.657 | 3.743 | 0.44 | 8 | 44.67 |
| baseline | 1048576 | count | nulls | 2.107 | 3.897 | 0.54 | 8 | 44.67 |
| baseline | 1048576 | count | sorted | 1.661 | 3.480 | 0.48 | 8 | 44.67 |
| baseline | 1048576 | filter_unused | base | 2.159 | 4.220 | 0.51 | 9 | 44.52 |
| baseline | 1048576 | filter_unused | nulls | 1.952 | 4.168 | 0.47 | 9 | 44.52 |
| baseline | 1048576 | filter_unused | sorted | 1.496 | 3.963 | 0.38 | 9 | 44.52 |
| baseline | 1048576 | filtered_int_sum | base | 1.617 | 3.127 | 0.52 | 8 | 34.41 |
| baseline | 1048576 | filtered_int_sum | nulls | 1.774 | 3.150 | 0.56 | 8 | 34.41 |
| baseline | 1048576 | filtered_int_sum | sorted | 1.715 | 3.015 | 0.57 | 8 | 34.41 |
| baseline | 8388608 | count | base | 2.053 | 22.919 | 0.09 | 8 | 357.16 |
| baseline | 8388608 | count | nulls | 6.638 | 23.765 | 0.28 | 8 | 357.16 |
| baseline | 8388608 | count | sorted | 1.968 | 23.052 | 0.09 | 8 | 357.16 |
| baseline | 8388608 | filter_unused | base | 5.375 | 30.546 | 0.18 | 9 | 356.13 |
| baseline | 8388608 | filter_unused | nulls | 9.870 | 32.156 | 0.31 | 9 | 356.13 |
| baseline | 8388608 | filter_unused | sorted | 4.920 | 30.230 | 0.16 | 9 | 356.13 |
| baseline | 8388608 | filtered_int_sum | base | 2.856 | 16.706 | 0.17 | 8 | 275.14 |
| baseline | 8388608 | filtered_int_sum | nulls | 6.080 | 17.043 | 0.36 | 8 | 275.14 |
| baseline | 8388608 | filtered_int_sum | sorted | 1.641 | 15.652 | 0.10 | 8 | 275.14 |
| metal | 1048576 | count | base | 1.422 | 2.325 | 0.61 | 5 | 20.66 |
| metal | 1048576 | count | nulls | 2.062 | 3.275 | 0.63 | 5 | 20.66 |
| metal | 1048576 | count | sorted | 1.739 | 2.348 | 0.74 | 5 | 20.66 |
| metal | 1048576 | filter_unused | base | 2.123 | 4.013 | 0.53 | 8 | 44.52 |
| metal | 1048576 | filter_unused | nulls | 2.144 | 3.902 | 0.55 | 8 | 44.52 |
| metal | 1048576 | filter_unused | sorted | 1.608 | 3.807 | 0.42 | 8 | 44.52 |
| metal | 1048576 | filtered_int_sum | base | 1.222 | 2.017 | 0.61 | 5 | 15.39 |
| metal | 1048576 | filtered_int_sum | nulls | 1.917 | 2.058 | 0.93 | 5 | 15.39 |
| metal | 1048576 | filtered_int_sum | sorted | 1.437 | 1.800 | 0.80 | 5 | 15.39 |
| metal | 8388608 | count | base | 2.337 | 10.957 | 0.21 | 5 | 165.03 |
| metal | 8388608 | count | nulls | 6.614 | 13.406 | 0.49 | 5 | 165.03 |
| metal | 8388608 | count | sorted | 2.120 | 11.455 | 0.19 | 5 | 165.03 |
| metal | 8388608 | filter_unused | base | 4.717 | 28.948 | 0.16 | 8 | 356.13 |
| metal | 8388608 | filter_unused | nulls | 9.927 | 29.375 | 0.34 | 8 | 356.13 |
| metal | 8388608 | filter_unused | sorted | 4.876 | 28.802 | 0.17 | 8 | 356.13 |
| metal | 8388608 | filtered_int_sum | base | 3.456 | 10.923 | 0.32 | 5 | 123.02 |
| metal | 8388608 | filtered_int_sum | nulls | 5.553 | 10.966 | 0.51 | 5 | 123.02 |
| metal | 8388608 | filtered_int_sum | sorted | 1.932 | 9.040 | 0.21 | 5 | 123.02 |

## Worst measured variant per case

Lowest CPU/Metal ratio among the selected variants; ratios below one favor CPU. These are per-mechanism results, not an aggregate suite score.

| Runtime | Rows | Workload | Worst variant | CPU/Metal |
|---|---:|---|---|---:|
| baseline | 1048576 | count | base | 0.44 |
| baseline | 1048576 | filter_unused | sorted | 0.38 |
| baseline | 1048576 | filtered_int_sum | base | 0.52 |
| baseline | 8388608 | count | sorted | 0.09 |
| baseline | 8388608 | filter_unused | sorted | 0.16 |
| baseline | 8388608 | filtered_int_sum | sorted | 0.10 |
| metal | 1048576 | count | base | 0.61 |
| metal | 1048576 | filter_unused | sorted | 0.42 |
| metal | 1048576 | filtered_int_sum | base | 0.61 |
| metal | 8388608 | count | sorted | 0.19 |
| metal | 8388608 | filter_unused | base | 0.16 |
| metal | 8388608 | filtered_int_sum | sorted | 0.21 |

Process outcomes: {'ok': 72, 'unsupported': 0, 'failed': 0}.

## Provenance and limits

```json
{
  "recorded_at_utc": "2026-10-11T01:51:48.853277+00:00",
  "platform": "macOS-26.5.2-arm64-arm-64bit-Mach-O",
  "machine": "arm64",
  "cpu": "Apple M1",
  "physical_cores": "8",
  "memory_bytes": "17179869184",
  "git_commit": "97f5f4513d4e6d17ef269019f1c36aca8e1956e7",
  "engine_status": "M dataframe/metal.mojo\n M native/dfmetal/dfmetal.h\n M native/dfmetal/dfmetal.mm\n M native/dfmetal/dfmetal.sym\n M native/dfmetal/dfmetal_join.inc\n M native/dfmetal/test_native.py\n?? native/dfmetal/test_resident.py",
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
  "binary_sha256": "2adda3c33588aafbf822f559185fb79d46f7148407ab7f381dcd6e998bfebe1f",
  "source_sha256": "6d96f66888fefcaaef63899665b80b936c220520c628982f9d4afba08a183e6d",
  "driver_sha256": "251e7c1c4cb94f15d17249d7e84a96e8a8678e5f256ea51fd18b75daffb360d8",
  "library_sha256": {
    "metal": "b11f935b9941478e15266bc1e97b1b17e59e2d42f4dfc09c2b4c6f46c8b079b4",
    "baseline": "1a30dfd8eeaebe7fbaf5afe6e132c157cf4bf9f0d8b03c1e6fc195a1098a84b8"
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
