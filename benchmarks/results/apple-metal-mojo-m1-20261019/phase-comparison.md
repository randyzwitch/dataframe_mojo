# Apple Metal development measurements

Development mechanisms only, not external-suite evidence or a general GPU speedup. Chain32 includes fusion advantages over the current CPU executor.

Times below are medians of complete warm collections, including allocation, staging, synchronization, and CPU result construction. First-use and all individual samples remain in the JSON. Every selected failed or unsupported case remains visible.

| Runtime | Rows | Workload | Variant | CPU ms | Metal ms | CPU/Metal | Launches | Peak payload MiB |
|---|---:|---|---|---:|---:|---:|---:|---:|
| baseline | 4097 | cumulative_sum | base | 0.080 | 16.407 | 0.00 | 18 | 0.80 |
| baseline | 4097 | cumulative_sum | nulls | 0.082 | 15.709 | 0.01 | 18 | 0.80 |
| baseline | 4097 | decimal_sum | base | 0.080 | 19.466 | 0.00 | 5 | 0.33 |
| baseline | 4097 | decimal_sum | nulls | 0.144 | 18.778 | 0.01 | 5 | 0.33 |
| baseline | 4097 | expression15 | base | 1.120 | 6.915 | 0.16 | 1 | 0.37 |
| baseline | 4097 | expression15 | nulls | 1.161 | 6.942 | 0.17 | 1 | 0.37 |
| baseline | 4097 | filter_half | base | 0.073 | 6.792 | 0.01 | 5 | 0.79 |
| baseline | 4097 | filter_half | nulls | 0.101 | 6.781 | 0.01 | 5 | 0.79 |
| baseline | 4097 | filtered_int_sum | base | 0.070 | 8.185 | 0.01 | 9 | 0.76 |
| baseline | 4097 | filtered_int_sum | nulls | 0.095 | 8.340 | 0.01 | 9 | 0.76 |
| baseline | 4097 | float64_arithmetic | base | 0.042 | 6.141 | 0.01 | 1 | 0.38 |
| baseline | 4097 | float64_arithmetic | nulls | 0.052 | 6.117 | 0.01 | 1 | 0.38 |
| baseline | 4097 | join_fanout | base | 0.298 | 11.966 | 0.02 | 18 | 3.10 |
| baseline | 4097 | join_fanout | nulls | 0.349 | 11.351 | 0.03 | 18 | 2.92 |
| baseline | 4097 | projection | base | 0.061 | 6.099 | 0.01 | 1 | 0.36 |
| baseline | 4097 | projection | nulls | 0.071 | 6.148 | 0.01 | 1 | 0.36 |
| baseline | 4097 | wide_float32 | base | 0.179 | 13.882 | 0.01 | 7 | 1.46 |
| baseline | 4097 | wide_float32 | nulls | 0.188 | 15.142 | 0.01 | 7 | 1.46 |
| baseline | 1048576 | cumulative_sum | base | 19.935 | 2272.164 | 0.01 | 25 | 205.16 |
| baseline | 1048576 | cumulative_sum | nulls | 21.476 | 2186.693 | 0.01 | 25 | 205.16 |
| baseline | 1048576 | decimal_sum | base | 9.470 | 3152.024 | 0.00 | 5 | 84.06 |
| baseline | 1048576 | decimal_sum | nulls | 12.581 | 3035.325 | 0.00 | 5 | 84.06 |
| baseline | 1048576 | expression15 | base | 18.593 | 164.915 | 0.11 | 1 | 92.13 |
| baseline | 1048576 | expression15 | nulls | 20.192 | 165.577 | 0.12 | 1 | 92.13 |
| baseline | 1048576 | filter_half | base | 2.625 | 64.644 | 0.04 | 5 | 201.16 |
| baseline | 1048576 | filter_half | nulls | 5.364 | 65.900 | 0.08 | 5 | 201.16 |
| baseline | 1048576 | filtered_int_sum | base | 2.883 | 27.423 | 0.11 | 9 | 193.09 |
| baseline | 1048576 | filtered_int_sum | nulls | 2.841 | 27.495 | 0.10 | 9 | 193.09 |
| baseline | 1048576 | float64_arithmetic | base | 4.514 | 98.749 | 0.05 | 1 | 96.13 |
| baseline | 1048576 | float64_arithmetic | nulls | 1.889 | 98.993 | 0.02 | 1 | 96.13 |
| baseline | 1048576 | join_fanout | base | 44.531 | 1113.114 | 0.04 | 25 | 793.53 |
| baseline | 1048576 | join_fanout | nulls | 42.907 | 1034.797 | 0.04 | 25 | 748.42 |
| baseline | 1048576 | projection | base | 4.236 | 101.254 | 0.04 | 1 | 92.13 |
| baseline | 1048576 | projection | nulls | 4.112 | 101.302 | 0.04 | 1 | 92.13 |
| baseline | 1048576 | wide_float32 | base | 18.323 | 669.109 | 0.03 | 7 | 373.00 |
| baseline | 1048576 | wide_float32 | nulls | 19.048 | 669.752 | 0.03 | 7 | 373.00 |
| mojo | 4097 | cumulative_sum | base | 0.077 | 16.279 | 0.00 | 18 | 0.80 |
| mojo | 4097 | cumulative_sum | nulls | 0.087 | 16.366 | 0.01 | 18 | 0.80 |
| mojo | 4097 | decimal_sum | base | 0.077 | 19.291 | 0.00 | 5 | 0.33 |
| mojo | 4097 | decimal_sum | nulls | 0.143 | 18.865 | 0.01 | 5 | 0.33 |
| mojo | 4097 | expression15 | base | 1.115 | 6.854 | 0.16 | 1 | 0.37 |
| mojo | 4097 | expression15 | nulls | 1.128 | 6.928 | 0.16 | 1 | 0.37 |
| mojo | 4097 | filter_half | base | 0.078 | 7.056 | 0.01 | 5 | 0.79 |
| mojo | 4097 | filter_half | nulls | 0.102 | 7.088 | 0.01 | 5 | 0.79 |
| mojo | 4097 | filtered_int_sum | base | 0.074 | 8.414 | 0.01 | 9 | 0.76 |
| mojo | 4097 | filtered_int_sum | nulls | 0.096 | 8.402 | 0.01 | 9 | 0.76 |
| mojo | 4097 | float64_arithmetic | base | 0.043 | 6.112 | 0.01 | 1 | 0.38 |
| mojo | 4097 | float64_arithmetic | nulls | 0.052 | 5.982 | 0.01 | 1 | 0.38 |
| mojo | 4097 | join_fanout | base | 0.309 | 11.753 | 0.03 | 18 | 3.10 |
| mojo | 4097 | join_fanout | nulls | 0.346 | 11.728 | 0.03 | 18 | 2.92 |
| mojo | 4097 | projection | base | 0.067 | 6.729 | 0.01 | 1 | 0.36 |
| mojo | 4097 | projection | nulls | 0.073 | 6.161 | 0.01 | 1 | 0.36 |
| mojo | 4097 | wide_float32 | base | 0.174 | 15.111 | 0.01 | 7 | 1.46 |
| mojo | 4097 | wide_float32 | nulls | 0.188 | 15.191 | 0.01 | 7 | 1.46 |
| mojo | 1048576 | cumulative_sum | base | 14.052 | 2274.029 | 0.01 | 25 | 205.16 |
| mojo | 1048576 | cumulative_sum | nulls | 18.209 | 2191.229 | 0.01 | 25 | 205.16 |
| mojo | 1048576 | decimal_sum | base | 5.012 | 3159.626 | 0.00 | 5 | 84.06 |
| mojo | 1048576 | decimal_sum | nulls | 12.386 | 3038.725 | 0.00 | 5 | 84.06 |
| mojo | 1048576 | expression15 | base | 21.855 | 164.370 | 0.13 | 1 | 92.13 |
| mojo | 1048576 | expression15 | nulls | 21.081 | 165.389 | 0.13 | 1 | 92.13 |
| mojo | 1048576 | filter_half | base | 7.149 | 64.559 | 0.11 | 5 | 201.16 |
| mojo | 1048576 | filter_half | nulls | 5.412 | 65.747 | 0.08 | 5 | 201.16 |
| mojo | 1048576 | filtered_int_sum | base | 2.713 | 27.433 | 0.10 | 9 | 193.09 |
| mojo | 1048576 | filtered_int_sum | nulls | 2.849 | 27.407 | 0.10 | 9 | 193.09 |
| mojo | 1048576 | float64_arithmetic | base | 4.337 | 98.769 | 0.04 | 1 | 96.13 |
| mojo | 1048576 | float64_arithmetic | nulls | 1.885 | 98.826 | 0.02 | 1 | 96.13 |
| mojo | 1048576 | join_fanout | base | 47.057 | 1111.892 | 0.04 | 25 | 793.53 |
| mojo | 1048576 | join_fanout | nulls | 42.647 | 1037.296 | 0.04 | 25 | 748.42 |
| mojo | 1048576 | projection | base | 4.244 | 101.052 | 0.04 | 1 | 92.13 |
| mojo | 1048576 | projection | nulls | 4.130 | 101.292 | 0.04 | 1 | 92.13 |
| mojo | 1048576 | wide_float32 | base | 19.277 | 669.181 | 0.03 | 7 | 373.00 |
| mojo | 1048576 | wide_float32 | nulls | 15.262 | 669.941 | 0.02 | 7 | 373.00 |

## Worst measured variant per case

Lowest CPU/Metal ratio among the selected variants; ratios below one favor CPU. These are per-mechanism results, not an aggregate suite score.

| Runtime | Rows | Workload | Worst variant | CPU/Metal |
|---|---:|---|---|---:|
| baseline | 4097 | cumulative_sum | base | 0.00 |
| baseline | 4097 | decimal_sum | base | 0.00 |
| baseline | 4097 | expression15 | base | 0.16 |
| baseline | 4097 | filter_half | base | 0.01 |
| baseline | 4097 | filtered_int_sum | base | 0.01 |
| baseline | 4097 | float64_arithmetic | base | 0.01 |
| baseline | 4097 | join_fanout | base | 0.02 |
| baseline | 4097 | projection | base | 0.01 |
| baseline | 4097 | wide_float32 | nulls | 0.01 |
| baseline | 1048576 | cumulative_sum | base | 0.01 |
| baseline | 1048576 | decimal_sum | base | 0.00 |
| baseline | 1048576 | expression15 | base | 0.11 |
| baseline | 1048576 | filter_half | base | 0.04 |
| baseline | 1048576 | filtered_int_sum | nulls | 0.10 |
| baseline | 1048576 | float64_arithmetic | nulls | 0.02 |
| baseline | 1048576 | join_fanout | base | 0.04 |
| baseline | 1048576 | projection | nulls | 0.04 |
| baseline | 1048576 | wide_float32 | base | 0.03 |
| mojo | 4097 | cumulative_sum | base | 0.00 |
| mojo | 4097 | decimal_sum | base | 0.00 |
| mojo | 4097 | expression15 | base | 0.16 |
| mojo | 4097 | filter_half | base | 0.01 |
| mojo | 4097 | filtered_int_sum | base | 0.01 |
| mojo | 4097 | float64_arithmetic | base | 0.01 |
| mojo | 4097 | join_fanout | base | 0.03 |
| mojo | 4097 | projection | base | 0.01 |
| mojo | 4097 | wide_float32 | base | 0.01 |
| mojo | 1048576 | cumulative_sum | base | 0.01 |
| mojo | 1048576 | decimal_sum | base | 0.00 |
| mojo | 1048576 | expression15 | nulls | 0.13 |
| mojo | 1048576 | filter_half | nulls | 0.08 |
| mojo | 1048576 | filtered_int_sum | base | 0.10 |
| mojo | 1048576 | float64_arithmetic | nulls | 0.02 |
| mojo | 1048576 | join_fanout | nulls | 0.04 |
| mojo | 1048576 | projection | nulls | 0.04 |
| mojo | 1048576 | wide_float32 | nulls | 0.02 |

## Measured Metal slowdowns

Cases with CPU/Metal below one, ordered by the largest Metal slowdown. The slowdown factor is Metal time divided by CPU time.

| Runtime | Rows | Workload | Variant | CPU/Metal | Metal slower by |
|---|---:|---|---|---:|---:|
| mojo | 1048576 | decimal_sum | base | 0.00 | 630.41x |
| baseline | 1048576 | decimal_sum | base | 0.00 | 332.83x |
| mojo | 4097 | decimal_sum | base | 0.00 | 250.53x |
| mojo | 1048576 | decimal_sum | nulls | 0.00 | 245.33x |
| baseline | 4097 | decimal_sum | base | 0.00 | 244.86x |
| baseline | 1048576 | decimal_sum | nulls | 0.00 | 241.26x |
| mojo | 4097 | cumulative_sum | base | 0.00 | 211.41x |
| baseline | 4097 | cumulative_sum | base | 0.00 | 206.38x |
| baseline | 4097 | cumulative_sum | nulls | 0.01 | 191.57x |
| mojo | 4097 | cumulative_sum | nulls | 0.01 | 188.11x |
| mojo | 1048576 | cumulative_sum | base | 0.01 | 161.84x |
| baseline | 4097 | float64_arithmetic | base | 0.01 | 146.23x |
| mojo | 4097 | float64_arithmetic | base | 0.01 | 142.13x |
| mojo | 4097 | decimal_sum | nulls | 0.01 | 131.92x |
| baseline | 4097 | decimal_sum | nulls | 0.01 | 130.40x |
| mojo | 1048576 | cumulative_sum | nulls | 0.01 | 120.33x |
| baseline | 4097 | float64_arithmetic | nulls | 0.01 | 117.63x |
| baseline | 4097 | filtered_int_sum | base | 0.01 | 116.94x |
| baseline | 1048576 | cumulative_sum | base | 0.01 | 113.98x |
| mojo | 4097 | float64_arithmetic | nulls | 0.01 | 113.95x |
| mojo | 4097 | filtered_int_sum | base | 0.01 | 113.70x |
| baseline | 1048576 | cumulative_sum | nulls | 0.01 | 101.82x |
| mojo | 4097 | projection | base | 0.01 | 100.43x |
| baseline | 4097 | projection | base | 0.01 | 99.99x |
| baseline | 4097 | filter_half | base | 0.01 | 92.41x |
| mojo | 4097 | filter_half | base | 0.01 | 90.46x |
| baseline | 4097 | filtered_int_sum | nulls | 0.01 | 87.79x |
| mojo | 4097 | filtered_int_sum | nulls | 0.01 | 87.52x |
| mojo | 4097 | wide_float32 | base | 0.01 | 86.60x |
| baseline | 4097 | projection | nulls | 0.01 | 86.59x |
| mojo | 4097 | projection | nulls | 0.01 | 84.40x |
| mojo | 4097 | wide_float32 | nulls | 0.01 | 81.02x |
| baseline | 4097 | wide_float32 | nulls | 0.01 | 80.76x |
| baseline | 4097 | wide_float32 | base | 0.01 | 77.34x |
| mojo | 4097 | filter_half | nulls | 0.01 | 69.15x |
| baseline | 4097 | filter_half | nulls | 0.01 | 67.14x |
| baseline | 1048576 | float64_arithmetic | nulls | 0.02 | 52.42x |
| mojo | 1048576 | float64_arithmetic | nulls | 0.02 | 52.41x |
| mojo | 1048576 | wide_float32 | nulls | 0.02 | 43.89x |
| baseline | 4097 | join_fanout | base | 0.02 | 40.15x |
| mojo | 4097 | join_fanout | base | 0.03 | 37.97x |
| baseline | 1048576 | wide_float32 | base | 0.03 | 36.52x |
| baseline | 1048576 | wide_float32 | nulls | 0.03 | 35.16x |
| mojo | 1048576 | wide_float32 | base | 0.03 | 34.71x |
| mojo | 4097 | join_fanout | nulls | 0.03 | 33.90x |
| baseline | 4097 | join_fanout | nulls | 0.03 | 32.52x |
| baseline | 1048576 | join_fanout | base | 0.04 | 25.00x |
| baseline | 1048576 | projection | nulls | 0.04 | 24.64x |
| baseline | 1048576 | filter_half | base | 0.04 | 24.63x |
| mojo | 1048576 | projection | nulls | 0.04 | 24.52x |
| mojo | 1048576 | join_fanout | nulls | 0.04 | 24.32x |
| baseline | 1048576 | join_fanout | nulls | 0.04 | 24.12x |
| baseline | 1048576 | projection | base | 0.04 | 23.90x |
| mojo | 1048576 | projection | base | 0.04 | 23.81x |
| mojo | 1048576 | join_fanout | base | 0.04 | 23.63x |
| mojo | 1048576 | float64_arithmetic | base | 0.04 | 22.77x |
| baseline | 1048576 | float64_arithmetic | base | 0.05 | 21.87x |
| baseline | 1048576 | filter_half | nulls | 0.08 | 12.29x |
| mojo | 1048576 | filter_half | nulls | 0.08 | 12.15x |
| mojo | 1048576 | filtered_int_sum | base | 0.10 | 10.11x |
| baseline | 1048576 | filtered_int_sum | nulls | 0.10 | 9.68x |
| mojo | 1048576 | filtered_int_sum | nulls | 0.10 | 9.62x |
| baseline | 1048576 | filtered_int_sum | base | 0.11 | 9.51x |
| mojo | 1048576 | filter_half | base | 0.11 | 9.03x |
| baseline | 1048576 | expression15 | base | 0.11 | 8.87x |
| baseline | 1048576 | expression15 | nulls | 0.12 | 8.20x |
| mojo | 1048576 | expression15 | nulls | 0.13 | 7.85x |
| mojo | 1048576 | expression15 | base | 0.13 | 7.52x |
| baseline | 4097 | expression15 | base | 0.16 | 6.17x |
| mojo | 4097 | expression15 | base | 0.16 | 6.15x |
| mojo | 4097 | expression15 | nulls | 0.16 | 6.14x |
| baseline | 4097 | expression15 | nulls | 0.17 | 5.98x |

Process outcomes: {'ok': 144, 'unsupported': 0, 'failed': 0}.

## Provenance and limits

```json
{
  "recorded_at_utc": "2026-10-19T20:47:50.167214+00:00",
  "platform": "macOS-26.5.2-arm64-arm-64bit-Mach-O",
  "machine": "arm64",
  "cpu": "Apple M1",
  "physical_cores": "8",
  "memory_bytes": "17179869184",
  "git_commit": "3e701487c8f6aa10a9036f4aa9c349bb7d509122",
  "source_tree": {
    "status": "",
    "dataframe_status": "",
    "dataframe_diff_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "dataframe_content_sha256": "d14ff34bf3288252c95b6c73bdd2576610f2aca4c8f46bb676a8fc9bee10135c"
  },
  "engine_status": "",
  "benchmark_status": "",
  "mojo_version": "Mojo 1.2.0.dev2026100905 (3b71af75)",
  "compiler_identity": "Mojo 1.2.0.dev2026100905 (3b71af75)",
  "compiler_executable": "/Users/randyzwitch/github/dataframe_mojo-metal-868/.pixi/envs/gpu/bin/mojo",
  "compiler_executable_sha256": "ac42575fdf41cd8ced537812c20cbdda361223434ba90c866c6c42007df1a37b",
  "pixi_lock_sha256": "098d02ba4d166a75640ed6b96aab77fdfe6cb468eded7838f23bba244496e277",
  "xcode": "Xcode 26.6\nBuild version 17F113",
  "sdk": "26.5",
  "compile_flags": [
    "-O3",
    "-I",
    "."
  ],
  "dataframe_threads": 8,
  "powerconditions": {
    "battery": {
      "status": "available",
      "returncode": 0,
      "output": "Now drawing from 'AC Power'"
    },
    "thermal": {
      "status": "available",
      "returncode": 0,
      "output": "Note: No thermal warning level has been recorded\nNote: No performance warning level has been recorded\nNote: No CPU power status has been recorded"
    }
  },
  "implementation": "mojo",
  "binaries_sha256": {
    "mojo": "b3bc4dcc5932e1f93a1cc3707cdb598d70e33e97cef5284042dc2a46affde2d1",
    "baseline": "0e82b185bd1bc0838272d031e2e058c8af5ff3e4361269b0282c731727c1afd2"
  },
  "binary_receipts": {
    "mojo": {
      "status": "verified",
      "schema_version": 1,
      "built_at_utc": "2026-10-19T20:47:50.080001+00:00",
      "binary_path": "/Users/randyzwitch/GitHub/dataframe_mojo-metal-868-profile-measurements/build/metal-868/profiled",
      "binary_sha256": "b3bc4dcc5932e1f93a1cc3707cdb598d70e33e97cef5284042dc2a46affde2d1",
      "build_command": [
        "mojo",
        "build",
        "-O3",
        "-I",
        ".",
        "benchmarks/bench_metal.mojo",
        "-o",
        "/Users/randyzwitch/GitHub/dataframe_mojo-metal-868-profile-measurements/build/metal-868/profiled"
      ],
      "git_commit": "3e701487c8f6aa10a9036f4aa9c349bb7d509122",
      "git_status": "",
      "engine_status": "",
      "engine_content_sha256": "d14ff34bf3288252c95b6c73bdd2576610f2aca4c8f46bb676a8fc9bee10135c",
      "engine_diff_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      "compiler_identity": "Mojo 1.2.0.dev2026100905 (3b71af75)",
      "build_platform": "macOS-26.5.2-arm64-arm-64bit-Mach-O",
      "build_sdk": "26.5",
      "build_xcode": "Xcode 26.6\nBuild version 17F113",
      "compiler_executable": "/Users/randyzwitch/github/dataframe_mojo-metal-868/.pixi/envs/gpu/bin/mojo",
      "compiler_executable_sha256": "ac42575fdf41cd8ced537812c20cbdda361223434ba90c866c6c42007df1a37b",
      "pixi_lock_sha256": "098d02ba4d166a75640ed6b96aab77fdfe6cb468eded7838f23bba244496e277",
      "benchmark_source_sha256": "265694f6b3edb061f230f665cff61abf069bc8be00d418894dbd5ddd62e8192c",
      "driver_source_sha256": "12ccfc8edda7eefa2c776520a3f81aca734af3e019a7aaf5ed2a6f67679ef745",
      "compile_flags": [
        "-O3",
        "-I",
        "."
      ],
      "powerconditions": {
        "battery": {
          "status": "available",
          "returncode": 0,
          "output": "Now drawing from 'AC Power'"
        },
        "thermal": {
          "status": "available",
          "returncode": 0,
          "output": "Note: No thermal warning level has been recorded\nNote: No performance warning level has been recorded\nNote: No CPU power status has been recorded"
        }
      }
    },
    "baseline": {
      "status": "verified",
      "schema_version": 1,
      "built_at_utc": "2026-10-19T20:41:20.980950+00:00",
      "binary_path": "/Users/randyzwitch/github/dataframe_mojo-metal-868/build/metal-868/current-baseline",
      "binary_sha256": "0e82b185bd1bc0838272d031e2e058c8af5ff3e4361269b0282c731727c1afd2",
      "build_command": [
        "mojo",
        "build",
        "-O3",
        "-I",
        ".",
        "benchmarks/bench_metal.mojo",
        "-o",
        "/Users/randyzwitch/github/dataframe_mojo-metal-868/build/metal-868/current-baseline"
      ],
      "git_commit": "d01d7229f0acbe2ec9fb186a377aed55dcda12a1",
      "git_status": "",
      "engine_status": "",
      "engine_content_sha256": "fb2f76ecefa9c938b2383906b3e647151f3e8c2f68967c548598a8b6672d7d13",
      "engine_diff_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      "compiler_identity": "Mojo 1.2.0.dev2026100905 (3b71af75)",
      "build_platform": "macOS-26.5.2-arm64-arm-64bit-Mach-O",
      "build_sdk": "26.5",
      "build_xcode": "Xcode 26.6\nBuild version 17F113",
      "compiler_executable": "/Users/randyzwitch/github/dataframe_mojo-metal-868/.pixi/envs/gpu/bin/mojo",
      "compiler_executable_sha256": "ac42575fdf41cd8ced537812c20cbdda361223434ba90c866c6c42007df1a37b",
      "pixi_lock_sha256": "098d02ba4d166a75640ed6b96aab77fdfe6cb468eded7838f23bba244496e277",
      "benchmark_source_sha256": "265694f6b3edb061f230f665cff61abf069bc8be00d418894dbd5ddd62e8192c",
      "driver_source_sha256": "12ccfc8edda7eefa2c776520a3f81aca734af3e019a7aaf5ed2a6f67679ef745",
      "compile_flags": [
        "-O3",
        "-I",
        "."
      ],
      "powerconditions": {
        "battery": {
          "status": "available",
          "returncode": 0,
          "output": "Now drawing from 'AC Power'"
        },
        "thermal": {
          "status": "available",
          "returncode": 0,
          "output": "Note: No thermal warning level has been recorded\nNote: No performance warning level has been recorded\nNote: No CPU power status has been recorded"
        }
      }
    }
  },
  "source_sha256": "265694f6b3edb061f230f665cff61abf069bc8be00d418894dbd5ddd62e8192c",
  "driver_sha256": "12ccfc8edda7eefa2c776520a3f81aca734af3e019a7aaf5ed2a6f67679ef745",
  "environment": {},
  "method": "Fresh process per case/build/round; first-use collections retained separately; two extra warmups per engine; alternating CPU/Metal timed order; process/build order rotated per round; inputs and exact CPU-reference validation untimed; complete result materialization timed; result release untimed.",
  "scope": "Development mechanisms only, not external-suite evidence or a general GPU speedup. Chain32 includes fusion advantages over the current CPU executor.",
  "data": "Row mechanisms use a deterministic 1024-value numeric distribution. Algorithm cases use Int32 keys: group_hash and group_skew have min(N,1024) classes; group_high, join_hash and sort_radix have max(1,N/2) classes; join_fanout uses ceil(N/4) cyclic or monotone balanced classes, so each key occurs at most four times and a self-left-join emits at most 4N rows. Base non-fanout keys are (37*i) modulo cardinality; sorted keys are monotone bins. group_skew maps three quarters of rows to key zero. Null variants invalidate every twentieth key; grouped values additionally invalidate every thirteenth row. Grouped sums use i modulo 17 minus 8; sort and self-left-join carry original Int64 row IDs. Float32 row values are exact multiples of 1/1024. Float64 arithmetic uses exact binary fractions k/1024 with k in [0,1023] and evaluates x*2+0.25. Decimal input is Decimal(15,2) cents with unscaled values in [-8,8]; the query multiplies by Decimal(15,2) 1.25 then sums. Wide Float32 projects eight columns with binary-exact offsets [0,0.125,...,0.875], at most 8N output values or 256 MiB at the default 8,388,608 rows.",
  "limitations": [
    "No external-suite coverage claimed. Floating reductions remain outside these cases. Algorithm comparisons preserve complete output order and values against CPU; this CPU check is not an independent oracle.",
    "First use includes context/shader initialization but is not a cold machine or cold filesystem measurement.",
    "CPU and GPU share host memory and power limits; short timings can vary with scheduler and thermal state.",
    "The Mojo profile is a separate warm collection after timed samples; complete caller-wall timing remains the performance measure. Reported host phase timings are retained as measured and may overlap; they are not added. GPU-only time and pipeline compilation attribution are unavailable unless the runtime supplies those fields.",
    "Workload exposure: chain32 was used during fusion development; other cases broaden mechanism coverage without a holdout claim.",
    "Power and thermal snapshots use read-only pmset queries when available. Missing commands, unsupported output, and timeouts are retained as unavailable; the snapshot is context, not a controlled power measurement."
  ]
}
```
