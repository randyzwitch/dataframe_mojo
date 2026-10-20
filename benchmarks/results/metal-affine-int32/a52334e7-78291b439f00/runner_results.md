# Apple Metal development measurements

Development mechanisms only, not external-suite evidence or a general GPU speedup. Chain32 includes fusion advantages over the current CPU executor.

Times below are medians of complete warm collections, including allocation, staging, synchronization, and CPU result construction. First-use and all individual samples remain in the JSON. Every selected failed or unsupported case remains visible.

| Runtime | Rows | Workload | Variant | CPU ms | Metal ms | CPU/Metal | Launches | Peak payload MiB |
|---|---:|---|---|---:|---:|---:|---:|---:|
| baseline | 4097 | expression15 | base | 1.136 | 6.999 | 0.16 | 1 | 0.37 |
| baseline | 4097 | expression15 | nulls | 1.137 | 6.915 | 0.16 | 1 | 0.37 |
| baseline | 4097 | expression15 | sorted | 1.116 | 7.262 | 0.15 | 1 | 0.37 |
| baseline | 4097 | filter_half | base | 0.073 | 7.142 | 0.01 | 5 | 0.79 |
| baseline | 4097 | filter_half | nulls | 0.088 | 7.068 | 0.01 | 5 | 0.79 |
| baseline | 4097 | filter_half | sorted | 0.095 | 7.109 | 0.01 | 5 | 0.79 |
| baseline | 4097 | filtered_int32_affine_sum | base | 0.128 | 8.534 | 0.01 | 10 | 0.76 |
| baseline | 4097 | filtered_int32_affine_sum | nulls | 0.131 | 8.452 | 0.02 | 10 | 0.76 |
| baseline | 4097 | filtered_int32_affine_sum | sorted | 0.081 | 8.449 | 0.01 | 10 | 0.76 |
| baseline | 4097 | int32_affine_sum | base | 0.067 | 7.330 | 0.01 | 5 | 0.33 |
| baseline | 4097 | int32_affine_sum | nulls | 0.117 | 7.242 | 0.02 | 5 | 0.33 |
| baseline | 4097 | int32_affine_sum | sorted | 0.085 | 7.202 | 0.01 | 5 | 0.33 |
| baseline | 4097 | projection | base | 0.062 | 6.217 | 0.01 | 1 | 0.36 |
| baseline | 4097 | projection | nulls | 0.062 | 6.187 | 0.01 | 1 | 0.36 |
| baseline | 4097 | projection | sorted | 0.072 | 6.204 | 0.01 | 1 | 0.36 |
| baseline | 1048576 | expression15 | base | 23.830 | 164.520 | 0.14 | 1 | 92.13 |
| baseline | 1048576 | expression15 | nulls | 23.949 | 165.025 | 0.15 | 1 | 92.13 |
| baseline | 1048576 | expression15 | sorted | 21.951 | 164.875 | 0.13 | 1 | 92.13 |
| baseline | 1048576 | filter_half | base | 9.361 | 64.388 | 0.15 | 5 | 201.16 |
| baseline | 1048576 | filter_half | nulls | 3.674 | 65.927 | 0.06 | 5 | 201.16 |
| baseline | 1048576 | filter_half | sorted | 5.009 | 64.445 | 0.08 | 5 | 201.16 |
| baseline | 1048576 | filtered_int32_affine_sum | base | 2.459 | 33.877 | 0.07 | 10 | 193.09 |
| baseline | 1048576 | filtered_int32_affine_sum | nulls | 2.710 | 33.286 | 0.08 | 10 | 193.09 |
| baseline | 1048576 | filtered_int32_affine_sum | sorted | 3.585 | 32.888 | 0.11 | 10 | 193.09 |
| baseline | 1048576 | int32_affine_sum | base | 2.879 | 26.960 | 0.11 | 5 | 84.06 |
| baseline | 1048576 | int32_affine_sum | nulls | 5.173 | 26.961 | 0.19 | 5 | 84.06 |
| baseline | 1048576 | int32_affine_sum | sorted | 3.504 | 25.973 | 0.13 | 5 | 84.06 |
| baseline | 1048576 | projection | base | 4.339 | 100.964 | 0.04 | 1 | 92.13 |
| baseline | 1048576 | projection | nulls | 4.273 | 101.160 | 0.04 | 1 | 92.13 |
| baseline | 1048576 | projection | sorted | 4.640 | 100.966 | 0.05 | 1 | 92.13 |
| mojo | 4097 | expression15 | base | 1.129 | 3.264 | 0.35 | 1 | 0.05 |
| mojo | 4097 | expression15 | nulls | 1.143 | 4.170 | 0.27 | 1 | 0.05 |
| mojo | 4097 | expression15 | sorted | 1.129 | 3.627 | 0.31 | 1 | 0.05 |
| mojo | 4097 | filter_half | base | 0.065 | 2.838 | 0.02 | 5 | 0.08 |
| mojo | 4097 | filter_half | nulls | 0.076 | 3.230 | 0.02 | 5 | 0.08 |
| mojo | 4097 | filter_half | sorted | 0.065 | 2.502 | 0.03 | 5 | 0.08 |
| mojo | 4097 | filtered_int32_affine_sum | base | 0.086 | 1.088 | 0.08 | 2 | 0.02 |
| mojo | 4097 | filtered_int32_affine_sum | nulls | 0.091 | 1.469 | 0.06 | 2 | 0.02 |
| mojo | 4097 | filtered_int32_affine_sum | sorted | 0.068 | 1.248 | 0.05 | 2 | 0.02 |
| mojo | 4097 | int32_affine_sum | base | 0.065 | 1.249 | 0.05 | 2 | 0.02 |
| mojo | 4097 | int32_affine_sum | nulls | 0.111 | 1.435 | 0.08 | 2 | 0.02 |
| mojo | 4097 | int32_affine_sum | sorted | 0.056 | 1.077 | 0.05 | 2 | 0.02 |
| mojo | 4097 | projection | base | 0.059 | 1.970 | 0.03 | 1 | 0.05 |
| mojo | 4097 | projection | nulls | 0.058 | 2.365 | 0.02 | 1 | 0.05 |
| mojo | 4097 | projection | sorted | 0.056 | 2.339 | 0.02 | 1 | 0.05 |
| mojo | 1048576 | expression15 | base | 24.454 | 37.499 | 0.65 | 1 | 12.00 |
| mojo | 1048576 | expression15 | nulls | 21.193 | 38.751 | 0.55 | 1 | 12.50 |
| mojo | 1048576 | expression15 | sorted | 24.322 | 37.460 | 0.65 | 1 | 12.00 |
| mojo | 1048576 | filter_half | base | 1.970 | 15.328 | 0.13 | 5 | 20.14 |
| mojo | 1048576 | filter_half | nulls | 1.917 | 17.154 | 0.11 | 5 | 20.52 |
| mojo | 1048576 | filter_half | sorted | 1.111 | 15.410 | 0.07 | 5 | 20.14 |
| mojo | 1048576 | filtered_int32_affine_sum | base | 1.975 | 2.665 | 0.74 | 2 | 4.06 |
| mojo | 1048576 | filtered_int32_affine_sum | nulls | 2.460 | 4.022 | 0.61 | 2 | 4.19 |
| mojo | 1048576 | filtered_int32_affine_sum | sorted | 1.953 | 2.348 | 0.83 | 2 | 4.06 |
| mojo | 1048576 | int32_affine_sum | base | 0.855 | 2.306 | 0.37 | 2 | 4.06 |
| mojo | 1048576 | int32_affine_sum | nulls | 3.350 | 3.799 | 0.88 | 2 | 4.19 |
| mojo | 1048576 | int32_affine_sum | sorted | 0.707 | 2.301 | 0.31 | 2 | 4.06 |
| mojo | 1048576 | projection | base | 2.281 | 20.306 | 0.11 | 1 | 12.00 |
| mojo | 1048576 | projection | nulls | 2.546 | 22.405 | 0.11 | 1 | 12.50 |
| mojo | 1048576 | projection | sorted | 2.379 | 20.402 | 0.12 | 1 | 12.00 |

## Worst measured variant per case

Lowest CPU/Metal ratio among the selected variants; ratios below one favor CPU. These are per-mechanism results, not an aggregate suite score.

| Runtime | Rows | Workload | Worst variant | CPU/Metal |
|---|---:|---|---|---:|
| baseline | 4097 | expression15 | sorted | 0.15 |
| baseline | 4097 | filter_half | base | 0.01 |
| baseline | 4097 | filtered_int32_affine_sum | sorted | 0.01 |
| baseline | 4097 | int32_affine_sum | base | 0.01 |
| baseline | 4097 | projection | base | 0.01 |
| baseline | 1048576 | expression15 | sorted | 0.13 |
| baseline | 1048576 | filter_half | nulls | 0.06 |
| baseline | 1048576 | filtered_int32_affine_sum | base | 0.07 |
| baseline | 1048576 | int32_affine_sum | base | 0.11 |
| baseline | 1048576 | projection | nulls | 0.04 |
| mojo | 4097 | expression15 | nulls | 0.27 |
| mojo | 4097 | filter_half | base | 0.02 |
| mojo | 4097 | filtered_int32_affine_sum | sorted | 0.05 |
| mojo | 4097 | int32_affine_sum | sorted | 0.05 |
| mojo | 4097 | projection | sorted | 0.02 |
| mojo | 1048576 | expression15 | nulls | 0.55 |
| mojo | 1048576 | filter_half | sorted | 0.07 |
| mojo | 1048576 | filtered_int32_affine_sum | nulls | 0.61 |
| mojo | 1048576 | int32_affine_sum | sorted | 0.31 |
| mojo | 1048576 | projection | base | 0.11 |

## Measured Metal slowdowns

Cases with CPU/Metal below one, ordered by the largest Metal slowdown. The slowdown factor is Metal time divided by CPU time.

| Runtime | Rows | Workload | Variant | CPU/Metal | Metal slower by |
|---|---:|---|---|---:|---:|
| baseline | 4097 | int32_affine_sum | base | 0.01 | 109.40x |
| baseline | 4097 | filtered_int32_affine_sum | sorted | 0.01 | 104.31x |
| baseline | 4097 | projection | base | 0.01 | 100.27x |
| baseline | 4097 | projection | nulls | 0.01 | 99.79x |
| baseline | 4097 | filter_half | base | 0.01 | 97.84x |
| baseline | 4097 | projection | sorted | 0.01 | 86.17x |
| baseline | 4097 | int32_affine_sum | sorted | 0.01 | 84.73x |
| baseline | 4097 | filter_half | nulls | 0.01 | 80.32x |
| baseline | 4097 | filter_half | sorted | 0.01 | 74.83x |
| baseline | 4097 | filtered_int32_affine_sum | base | 0.01 | 66.67x |
| baseline | 4097 | filtered_int32_affine_sum | nulls | 0.02 | 64.52x |
| baseline | 4097 | int32_affine_sum | nulls | 0.02 | 61.90x |
| mojo | 4097 | filter_half | base | 0.02 | 43.66x |
| mojo | 4097 | filter_half | nulls | 0.02 | 42.50x |
| mojo | 4097 | projection | sorted | 0.02 | 41.77x |
| mojo | 4097 | projection | nulls | 0.02 | 40.78x |
| mojo | 4097 | filter_half | sorted | 0.03 | 38.49x |
| mojo | 4097 | projection | base | 0.03 | 33.39x |
| baseline | 1048576 | projection | nulls | 0.04 | 23.67x |
| baseline | 1048576 | projection | base | 0.04 | 23.27x |
| baseline | 1048576 | projection | sorted | 0.05 | 21.76x |
| mojo | 4097 | int32_affine_sum | sorted | 0.05 | 19.23x |
| mojo | 4097 | int32_affine_sum | base | 0.05 | 19.22x |
| mojo | 4097 | filtered_int32_affine_sum | sorted | 0.05 | 18.35x |
| baseline | 1048576 | filter_half | nulls | 0.06 | 17.94x |
| mojo | 4097 | filtered_int32_affine_sum | nulls | 0.06 | 16.14x |
| mojo | 1048576 | filter_half | sorted | 0.07 | 13.87x |
| baseline | 1048576 | filtered_int32_affine_sum | base | 0.07 | 13.78x |
| mojo | 4097 | int32_affine_sum | nulls | 0.08 | 12.93x |
| baseline | 1048576 | filter_half | sorted | 0.08 | 12.87x |
| mojo | 4097 | filtered_int32_affine_sum | base | 0.08 | 12.65x |
| baseline | 1048576 | filtered_int32_affine_sum | nulls | 0.08 | 12.28x |
| baseline | 1048576 | int32_affine_sum | base | 0.11 | 9.36x |
| baseline | 1048576 | filtered_int32_affine_sum | sorted | 0.11 | 9.17x |
| mojo | 1048576 | filter_half | nulls | 0.11 | 8.95x |
| mojo | 1048576 | projection | base | 0.11 | 8.90x |
| mojo | 1048576 | projection | nulls | 0.11 | 8.80x |
| mojo | 1048576 | projection | sorted | 0.12 | 8.58x |
| mojo | 1048576 | filter_half | base | 0.13 | 7.78x |
| baseline | 1048576 | expression15 | sorted | 0.13 | 7.51x |
| baseline | 1048576 | int32_affine_sum | sorted | 0.13 | 7.41x |
| baseline | 1048576 | expression15 | base | 0.14 | 6.90x |
| baseline | 1048576 | expression15 | nulls | 0.15 | 6.89x |
| baseline | 1048576 | filter_half | base | 0.15 | 6.88x |
| baseline | 4097 | expression15 | sorted | 0.15 | 6.51x |
| baseline | 4097 | expression15 | base | 0.16 | 6.16x |
| baseline | 4097 | expression15 | nulls | 0.16 | 6.08x |
| baseline | 1048576 | int32_affine_sum | nulls | 0.19 | 5.21x |
| mojo | 4097 | expression15 | nulls | 0.27 | 3.65x |
| mojo | 1048576 | int32_affine_sum | sorted | 0.31 | 3.25x |
| mojo | 4097 | expression15 | sorted | 0.31 | 3.21x |
| mojo | 4097 | expression15 | base | 0.35 | 2.89x |
| mojo | 1048576 | int32_affine_sum | base | 0.37 | 2.70x |
| mojo | 1048576 | expression15 | nulls | 0.55 | 1.83x |
| mojo | 1048576 | filtered_int32_affine_sum | nulls | 0.61 | 1.63x |
| mojo | 1048576 | expression15 | sorted | 0.65 | 1.54x |
| mojo | 1048576 | expression15 | base | 0.65 | 1.53x |
| mojo | 1048576 | filtered_int32_affine_sum | base | 0.74 | 1.35x |
| mojo | 1048576 | filtered_int32_affine_sum | sorted | 0.83 | 1.20x |
| mojo | 1048576 | int32_affine_sum | nulls | 0.88 | 1.13x |

Process outcomes: {'ok': 180, 'unsupported': 0, 'failed': 0}.

## Provenance and limits

```json
{
  "recorded_at_utc": "2026-10-19T22:50:50.065384+00:00",
  "platform": "macOS-26.5.2-arm64-arm-64bit-Mach-O",
  "machine": "arm64",
  "cpu": "Apple M1",
  "physical_cores": "8",
  "memory_bytes": "17179869184",
  "git_commit": "78291b439f004ec27519d3e1273ed84abff07ca4",
  "source_tree": {
    "status": "M benchmarks/bench_metal.mojo\n M benchmarks/tools/bench_metal.py",
    "dataframe_status": "",
    "dataframe_diff_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "dataframe_content_sha256": "c22f832b624b8eae173fdfcb235d961fd5f5b01c2a9ab66ba90258debde2b542"
  },
  "engine_status": "",
  "benchmark_status": "M benchmarks/bench_metal.mojo\n M benchmarks/tools/bench_metal.py",
  "mojo_version": "Mojo 1.2.0.dev2026100905 (3b71af75)",
  "compiler_identity": "Mojo 1.2.0.dev2026100905 (3b71af75)",
  "compiler_executable": "/Users/randyzwitch/github/dataframe_mojo-metal-868/.pixi/envs/oracle/bin/mojo",
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
    "mojo": "694d53123b7ff01b857175cd989bf746556cf1a38283121dba88155aabe0afd0",
    "baseline": "3e8f057f50e9f006a23997050d95dfa292b0d50626da55f5e00e8b18c865956a"
  },
  "binary_receipts": {
    "mojo": {
      "status": "verified",
      "schema_version": 1,
      "built_at_utc": "2026-10-19T22:50:49.914921+00:00",
      "binary_path": "/Users/randyzwitch/github/dataframe_mojo-metal-868-fused-int32-bench/build/paired-a52334e7-78291b439f00/bench_metal_candidate",
      "binary_sha256": "694d53123b7ff01b857175cd989bf746556cf1a38283121dba88155aabe0afd0",
      "build_command": [
        "mojo",
        "build",
        "-O3",
        "-I",
        ".",
        "benchmarks/bench_metal.mojo",
        "-o",
        "/Users/randyzwitch/github/dataframe_mojo-metal-868-fused-int32-bench/build/paired-a52334e7-78291b439f00/bench_metal_candidate"
      ],
      "git_commit": "78291b439f004ec27519d3e1273ed84abff07ca4",
      "git_status": "M benchmarks/bench_metal.mojo\n M benchmarks/tools/bench_metal.py",
      "engine_status": "",
      "engine_content_sha256": "c22f832b624b8eae173fdfcb235d961fd5f5b01c2a9ab66ba90258debde2b542",
      "engine_diff_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      "compiler_identity": "Mojo 1.2.0.dev2026100905 (3b71af75)",
      "build_platform": "macOS-26.5.2-arm64-arm-64bit-Mach-O",
      "build_sdk": "26.5",
      "build_xcode": "Xcode 26.6\nBuild version 17F113",
      "compiler_executable": "/Users/randyzwitch/github/dataframe_mojo-metal-868/.pixi/envs/oracle/bin/mojo",
      "compiler_executable_sha256": "ac42575fdf41cd8ced537812c20cbdda361223434ba90c866c6c42007df1a37b",
      "pixi_lock_sha256": "098d02ba4d166a75640ed6b96aab77fdfe6cb468eded7838f23bba244496e277",
      "benchmark_source_sha256": "383b12bbc2906d0f345617bb23bae1ba0b38142cdfb8c2cc52e6611a238b3864",
      "driver_source_sha256": "16996036fc761f19d323b09e7c971c112c464e4680ad6d8158fe87c55e639aad",
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
      "built_at_utc": "2026-10-19T22:45:10.260976+00:00",
      "binary_path": "/Users/randyzwitch/github/dataframe_mojo-metal-868-fused-int32-bench/build/paired-a52334e7-78291b439f00/bench_metal_baseline",
      "binary_sha256": "3e8f057f50e9f006a23997050d95dfa292b0d50626da55f5e00e8b18c865956a",
      "build_command": [
        "mojo",
        "build",
        "-O3",
        "-I",
        ".",
        "benchmarks/bench_metal.mojo",
        "-o",
        "/Users/randyzwitch/github/dataframe_mojo-metal-868-fused-int32-bench/build/paired-a52334e7-78291b439f00/bench_metal_baseline"
      ],
      "git_commit": "a52334e758b7138c547883eb776d744442a1a0db",
      "git_status": "M benchmarks/bench_metal.mojo\n M benchmarks/tools/bench_metal.py",
      "engine_status": "",
      "engine_content_sha256": "579b19cbd598cf423107a282daccf7ecc154fdae820700d6290b91fb917f4c32",
      "engine_diff_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      "compiler_identity": "Mojo 1.2.0.dev2026100905 (3b71af75)",
      "build_platform": "macOS-26.5.2-arm64-arm-64bit-Mach-O",
      "build_sdk": "26.5",
      "build_xcode": "Xcode 26.6\nBuild version 17F113",
      "compiler_executable": "/Users/randyzwitch/github/dataframe_mojo-metal-868/.pixi/envs/oracle/bin/mojo",
      "compiler_executable_sha256": "ac42575fdf41cd8ced537812c20cbdda361223434ba90c866c6c42007df1a37b",
      "pixi_lock_sha256": "098d02ba4d166a75640ed6b96aab77fdfe6cb468eded7838f23bba244496e277",
      "benchmark_source_sha256": "383b12bbc2906d0f345617bb23bae1ba0b38142cdfb8c2cc52e6611a238b3864",
      "driver_source_sha256": "16996036fc761f19d323b09e7c971c112c464e4680ad6d8158fe87c55e639aad",
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
  "source_sha256": "383b12bbc2906d0f345617bb23bae1ba0b38142cdfb8c2cc52e6611a238b3864",
  "driver_sha256": "16996036fc761f19d323b09e7c971c112c464e4680ad6d8158fe87c55e639aad",
  "environment": {
    "DATAFRAME_THREADS": "8"
  },
  "method": "Fresh process per case/build/round; first-use collections retained separately; two extra warmups per engine; alternating CPU/Metal timed order; process/build order rotated per round; inputs and exact CPU-reference validation untimed; complete result materialization timed; result release untimed.",
  "scope": "Development mechanisms only, not external-suite evidence or a general GPU speedup. Chain32 includes fusion advantages over the current CPU executor.",
  "data": "Row mechanisms use a deterministic 1024-value numeric distribution. Algorithm cases use Int32 keys: group_hash and group_skew have min(N,1024) classes; group_high, join_hash and sort_radix have max(1,N/2) classes; join_fanout uses ceil(N/4) cyclic or monotone balanced classes, so each key occurs at most four times and a self-left-join emits at most 4N rows. Base non-fanout keys are (37*i) modulo cardinality; sorted keys are monotone bins. group_skew maps three quarters of rows to key zero. Null variants invalidate every twentieth key; grouped values additionally invalidate every thirteenth row. Grouped sums use i modulo 17 minus 8; sort and self-left-join carry original Int64 row IDs. Float32 row values are exact multiples of 1/1024. Float64 arithmetic uses exact binary fractions k/1024 with k in [0,1023] and evaluates x*2+0.25. Decimal input is Decimal(15,2) cents with unscaled values in [-8,8]; the query multiplies by Decimal(15,2) 1.25 then sums. Wide Float32 projects eight columns with binary-exact offsets [0,0.125,...,0.875], at most 8N output values or 256 MiB at the default 8,388,608 rows. Optional int32_affine_sum and filtered_int32_affine_sum workloads use the same Int32 base/sorted/nulls input patterns as int_sum, compute sum(cast_Int64((x+3)*2)), and optionally filter x>0 before the expression. They are explicitly selectable with --workloads and excluded from the default set.",
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
