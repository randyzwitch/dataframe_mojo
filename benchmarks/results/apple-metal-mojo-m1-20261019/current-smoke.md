# Apple Metal development measurements

Development mechanisms only, not external-suite evidence or a general GPU speedup. Chain32 includes fusion advantages over the current CPU executor.

Times below are medians of complete warm collections, including allocation, staging, synchronization, and CPU result construction. First-use and all individual samples remain in the JSON. Every selected failed or unsupported case remains visible.

| Runtime | Rows | Workload | Variant | CPU ms | Metal ms | CPU/Metal | Launches | Peak payload MiB |
|---|---:|---|---|---:|---:|---:|---:|---:|
| mojo | 4097 | cumulative_sum | base | 0.082 | 16.622 | 0.00 | 18 | 0.80 |
| mojo | 4097 | cumulative_sum | nulls | 0.087 | 18.250 | 0.00 | 18 | 0.80 |
| mojo | 4097 | cumulative_sum | sorted | 0.068 | 16.904 | 0.00 | 18 | 0.80 |
| mojo | 4097 | decimal_sum | base | 0.084 | 19.301 | 0.00 | 5 | 0.33 |
| mojo | 4097 | decimal_sum | nulls | 0.118 | 18.936 | 0.01 | 5 | 0.33 |
| mojo | 4097 | decimal_sum | sorted | 0.081 | 19.050 | 0.00 | 5 | 0.33 |
| mojo | 4097 | float64_arithmetic | base | 0.051 | 6.029 | 0.01 | 1 | 0.38 |
| mojo | 4097 | float64_arithmetic | nulls | 0.051 | 5.678 | 0.01 | 1 | 0.38 |
| mojo | 4097 | float64_arithmetic | sorted | 0.045 | 6.169 | 0.01 | 1 | 0.38 |
| mojo | 4097 | group_skew | base | 0.191 | 16.680 | 0.01 | 33 | 2.36 |
| mojo | 4097 | group_skew | nulls | 0.214 | 16.759 | 0.01 | 34 | 2.36 |
| mojo | 4097 | group_skew | sorted | 0.225 | 17.340 | 0.01 | 35 | 2.36 |
| mojo | 4097 | join_fanout | base | 0.291 | 11.994 | 0.02 | 18 | 3.10 |
| mojo | 4097 | join_fanout | nulls | 0.344 | 11.455 | 0.03 | 18 | 2.92 |
| mojo | 4097 | join_fanout | sorted | 0.314 | 11.730 | 0.03 | 18 | 3.10 |
| mojo | 4097 | projection | base | 0.074 | 6.383 | 0.01 | 1 | 0.36 |
| mojo | 4097 | projection | nulls | 0.071 | 6.154 | 0.01 | 1 | 0.36 |
| mojo | 4097 | projection | sorted | 0.065 | 6.144 | 0.01 | 1 | 0.36 |
| mojo | 4097 | rolling_sum | base | 0.344 | 7.880 | 0.04 | 18 | 0.80 |
| mojo | 4097 | rolling_sum | nulls | 0.340 | 8.222 | 0.04 | 18 | 0.80 |
| mojo | 4097 | rolling_sum | sorted | 0.354 | 8.275 | 0.04 | 18 | 0.80 |
| mojo | 4097 | wide_float32 | base | 0.182 | 15.043 | 0.01 | 7 | 1.46 |
| mojo | 4097 | wide_float32 | nulls | 0.183 | 14.172 | 0.01 | 7 | 1.46 |
| mojo | 4097 | wide_float32 | sorted | 0.177 | 15.058 | 0.01 | 7 | 1.46 |

## Worst measured variant per case

Lowest CPU/Metal ratio among the selected variants; ratios below one favor CPU. These are per-mechanism results, not an aggregate suite score.

| Runtime | Rows | Workload | Worst variant | CPU/Metal |
|---|---:|---|---|---:|
| mojo | 4097 | cumulative_sum | sorted | 0.00 |
| mojo | 4097 | decimal_sum | sorted | 0.00 |
| mojo | 4097 | float64_arithmetic | sorted | 0.01 |
| mojo | 4097 | group_skew | base | 0.01 |
| mojo | 4097 | join_fanout | base | 0.02 |
| mojo | 4097 | projection | sorted | 0.01 |
| mojo | 4097 | rolling_sum | nulls | 0.04 |
| mojo | 4097 | wide_float32 | sorted | 0.01 |

## Measured Metal slowdowns

Cases with CPU/Metal below one, ordered by the largest Metal slowdown. The slowdown factor is Metal time divided by CPU time.

| Runtime | Rows | Workload | Variant | CPU/Metal | Metal slower by |
|---|---:|---|---|---:|---:|
| mojo | 4097 | cumulative_sum | sorted | 0.00 | 248.60x |
| mojo | 4097 | decimal_sum | sorted | 0.00 | 235.19x |
| mojo | 4097 | decimal_sum | base | 0.00 | 229.77x |
| mojo | 4097 | cumulative_sum | nulls | 0.00 | 208.57x |
| mojo | 4097 | cumulative_sum | base | 0.00 | 202.71x |
| mojo | 4097 | decimal_sum | nulls | 0.01 | 159.80x |
| mojo | 4097 | float64_arithmetic | sorted | 0.01 | 135.58x |
| mojo | 4097 | float64_arithmetic | base | 0.01 | 119.38x |
| mojo | 4097 | float64_arithmetic | nulls | 0.01 | 111.34x |
| mojo | 4097 | projection | sorted | 0.01 | 94.53x |
| mojo | 4097 | group_skew | base | 0.01 | 87.56x |
| mojo | 4097 | projection | nulls | 0.01 | 86.67x |
| mojo | 4097 | projection | base | 0.01 | 85.67x |
| mojo | 4097 | wide_float32 | sorted | 0.01 | 84.83x |
| mojo | 4097 | wide_float32 | base | 0.01 | 82.42x |
| mojo | 4097 | group_skew | nulls | 0.01 | 78.31x |
| mojo | 4097 | group_skew | sorted | 0.01 | 77.24x |
| mojo | 4097 | wide_float32 | nulls | 0.01 | 77.23x |
| mojo | 4097 | join_fanout | base | 0.02 | 41.15x |
| mojo | 4097 | join_fanout | sorted | 0.03 | 37.42x |
| mojo | 4097 | join_fanout | nulls | 0.03 | 33.30x |
| mojo | 4097 | rolling_sum | nulls | 0.04 | 24.18x |
| mojo | 4097 | rolling_sum | sorted | 0.04 | 23.34x |
| mojo | 4097 | rolling_sum | base | 0.04 | 22.94x |

Process outcomes: {'ok': 24, 'unsupported': 0, 'failed': 0}.

## Provenance and limits

```json
{
  "recorded_at_utc": "2026-10-19T20:41:21.047600+00:00",
  "platform": "macOS-26.5.2-arm64-arm-64bit-Mach-O",
  "machine": "arm64",
  "cpu": "Apple M1",
  "physical_cores": "8",
  "memory_bytes": "17179869184",
  "git_commit": "d01d7229f0acbe2ec9fb186a377aed55dcda12a1",
  "source_tree": {
    "status": "",
    "dataframe_status": "",
    "dataframe_diff_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "dataframe_content_sha256": "fb2f76ecefa9c938b2383906b3e647151f3e8c2f68967c548598a8b6672d7d13"
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
    "mojo": "0e82b185bd1bc0838272d031e2e058c8af5ff3e4361269b0282c731727c1afd2"
  },
  "binary_receipts": {
    "mojo": {
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
