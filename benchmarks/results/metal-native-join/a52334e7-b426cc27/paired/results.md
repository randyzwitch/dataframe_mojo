# Apple Metal development measurements

Development mechanisms only, not external-suite evidence or a general GPU speedup. Chain32 includes fusion advantages over the current CPU executor.

Times below are medians of complete warm collections, including allocation, staging, synchronization, and CPU result construction. First-use and all individual samples remain in the JSON. Every selected failed or unsupported case remains visible.

| Runtime | Rows | Workload | Variant | CPU ms | Metal ms | CPU/Metal | Launches | Peak payload MiB |
|---|---:|---|---|---:|---:|---:|---:|---:|
| baseline | 4097 | join_fanout | base | 0.357 | 10.860 | 0.03 | 18 | 3.10 |
| baseline | 4097 | join_fanout | nulls | 0.376 | 10.462 | 0.04 | 18 | 2.92 |
| baseline | 4097 | join_fanout | sorted | 0.346 | 11.200 | 0.03 | 18 | 3.10 |
| baseline | 4097 | join_hash | base | 0.319 | 8.570 | 0.04 | 18 | 1.93 |
| baseline | 4097 | join_hash | nulls | 0.267 | 8.487 | 0.03 | 18 | 1.87 |
| baseline | 4097 | join_hash | sorted | 0.298 | 8.632 | 0.03 | 18 | 1.93 |
| baseline | 262144 | join_fanout | base | 11.397 | 287.206 | 0.04 | 23 | 198.38 |
| baseline | 262144 | join_fanout | nulls | 11.123 | 268.301 | 0.04 | 23 | 187.10 |
| baseline | 262144 | join_fanout | sorted | 11.974 | 283.917 | 0.04 | 23 | 198.38 |
| baseline | 262144 | join_hash | base | 13.225 | 170.756 | 0.08 | 23 | 123.20 |
| baseline | 262144 | join_hash | nulls | 12.228 | 164.591 | 0.07 | 23 | 119.44 |
| baseline | 262144 | join_hash | sorted | 10.724 | 168.229 | 0.06 | 23 | 123.20 |
| mojo | 4097 | join_fanout | base | 0.340 | 8.473 | 0.04 | 29 | 1.07 |
| mojo | 4097 | join_fanout | nulls | 0.363 | 8.437 | 0.04 | 29 | 1.04 |
| mojo | 4097 | join_fanout | sorted | 0.332 | 8.101 | 0.04 | 29 | 1.07 |
| mojo | 4097 | join_hash | base | 0.284 | 7.390 | 0.04 | 29 | 0.85 |
| mojo | 4097 | join_hash | nulls | 0.294 | 8.035 | 0.04 | 29 | 0.84 |
| mojo | 4097 | join_hash | sorted | 0.316 | 7.476 | 0.04 | 29 | 0.85 |
| mojo | 262144 | join_fanout | base | 12.546 | 107.496 | 0.12 | 34 | 60.66 |
| mojo | 262144 | join_fanout | nulls | 10.033 | 102.928 | 0.10 | 34 | 58.72 |
| mojo | 262144 | join_fanout | sorted | 9.419 | 109.285 | 0.09 | 34 | 60.66 |
| mojo | 262144 | join_hash | base | 8.906 | 71.756 | 0.12 | 34 | 46.59 |
| mojo | 262144 | join_hash | nulls | 9.401 | 69.984 | 0.13 | 34 | 46.01 |
| mojo | 262144 | join_hash | sorted | 8.389 | 67.616 | 0.12 | 34 | 46.59 |

## Worst measured variant per case

Lowest CPU/Metal ratio among the selected variants; ratios below one favor CPU. These are per-mechanism results, not an aggregate suite score.

| Runtime | Rows | Workload | Worst variant | CPU/Metal |
|---|---:|---|---|---:|
| baseline | 4097 | join_fanout | sorted | 0.03 |
| baseline | 4097 | join_hash | nulls | 0.03 |
| baseline | 262144 | join_fanout | base | 0.04 |
| baseline | 262144 | join_hash | sorted | 0.06 |
| mojo | 4097 | join_fanout | base | 0.04 |
| mojo | 4097 | join_hash | nulls | 0.04 |
| mojo | 262144 | join_fanout | sorted | 0.09 |
| mojo | 262144 | join_hash | sorted | 0.12 |

## Measured Metal slowdowns

Cases with CPU/Metal below one, ordered by the largest Metal slowdown. The slowdown factor is Metal time divided by CPU time.

| Runtime | Rows | Workload | Variant | CPU/Metal | Metal slower by |
|---|---:|---|---|---:|---:|
| baseline | 4097 | join_fanout | sorted | 0.03 | 32.37x |
| baseline | 4097 | join_hash | nulls | 0.03 | 31.79x |
| baseline | 4097 | join_fanout | base | 0.03 | 30.42x |
| baseline | 4097 | join_hash | sorted | 0.03 | 28.97x |
| baseline | 4097 | join_fanout | nulls | 0.04 | 27.82x |
| mojo | 4097 | join_hash | nulls | 0.04 | 27.33x |
| baseline | 4097 | join_hash | base | 0.04 | 26.87x |
| mojo | 4097 | join_hash | base | 0.04 | 26.02x |
| baseline | 262144 | join_fanout | base | 0.04 | 25.20x |
| mojo | 4097 | join_fanout | base | 0.04 | 24.92x |
| mojo | 4097 | join_fanout | sorted | 0.04 | 24.40x |
| baseline | 262144 | join_fanout | nulls | 0.04 | 24.12x |
| baseline | 262144 | join_fanout | sorted | 0.04 | 23.71x |
| mojo | 4097 | join_hash | sorted | 0.04 | 23.66x |
| mojo | 4097 | join_fanout | nulls | 0.04 | 23.24x |
| baseline | 262144 | join_hash | sorted | 0.06 | 15.69x |
| baseline | 262144 | join_hash | nulls | 0.07 | 13.46x |
| baseline | 262144 | join_hash | base | 0.08 | 12.91x |
| mojo | 262144 | join_fanout | sorted | 0.09 | 11.60x |
| mojo | 262144 | join_fanout | nulls | 0.10 | 10.26x |
| mojo | 262144 | join_fanout | base | 0.12 | 8.57x |
| mojo | 262144 | join_hash | sorted | 0.12 | 8.06x |
| mojo | 262144 | join_hash | base | 0.12 | 8.06x |
| mojo | 262144 | join_hash | nulls | 0.13 | 7.44x |

Process outcomes: {'ok': 72, 'unsupported': 0, 'failed': 0}.

## Provenance and limits

```json
{
  "recorded_at_utc": "2026-10-20T15:12:03.301799+00:00",
  "platform": "macOS-26.5.2-arm64-arm-64bit-Mach-O",
  "machine": "arm64",
  "cpu": "Apple M1",
  "physical_cores": "8",
  "memory_bytes": "17179869184",
  "git_commit": "01f05c364790da1144b3bea8152a50bc0522cccc",
  "source_tree": {
    "status": "",
    "dataframe_status": "",
    "dataframe_diff_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "dataframe_content_sha256": "dae9ea5fa4928ecbd9ce5494e5d3952b841e8adeb92f84563a242f556d5d45b4"
  },
  "engine_status": "",
  "benchmark_status": "",
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
  "warmups_per_engine": 1,
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
    "mojo": "b0076d032b880cbdcfe2f83f039bbf424b4c5c79b9522ff936fa36dc3038bb24",
    "baseline": "1f4eb0ebab4f7eccf034c0ad8f4c52a5486ef3c997d36cab01f988f1ec731ab4"
  },
  "binary_receipts": {
    "mojo": {
      "status": "verified",
      "schema_version": 1,
      "built_at_utc": "2026-10-20T15:12:01.978222+00:00",
      "binary_path": "/Users/randyzwitch/GitHub/dataframe_mojo-metal-native-join-candidate-b426/build/bench_metal",
      "binary_sha256": "b0076d032b880cbdcfe2f83f039bbf424b4c5c79b9522ff936fa36dc3038bb24",
      "build_command": [
        "mojo",
        "build",
        "-O3",
        "-I",
        ".",
        "benchmarks/bench_metal.mojo",
        "-o",
        "/Users/randyzwitch/GitHub/dataframe_mojo-metal-native-join-candidate-b426/build/bench_metal"
      ],
      "git_commit": "01f05c364790da1144b3bea8152a50bc0522cccc",
      "git_status": "",
      "engine_status": "",
      "engine_content_sha256": "dae9ea5fa4928ecbd9ce5494e5d3952b841e8adeb92f84563a242f556d5d45b4",
      "engine_diff_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      "compiler_identity": "Mojo 1.2.0.dev2026100905 (3b71af75)",
      "build_platform": "macOS-26.5.2-arm64-arm-64bit-Mach-O",
      "build_sdk": "26.5",
      "build_xcode": "Xcode 26.6\nBuild version 17F113",
      "compiler_executable": "/Users/randyzwitch/github/dataframe_mojo-metal-868/.pixi/envs/oracle/bin/mojo",
      "compiler_executable_sha256": "ac42575fdf41cd8ced537812c20cbdda361223434ba90c866c6c42007df1a37b",
      "pixi_lock_sha256": "098d02ba4d166a75640ed6b96aab77fdfe6cb468eded7838f23bba244496e277",
      "benchmark_source_sha256": "cd8db3c71a6d4acb08bc71f1a8d14bedf64dcdbb7ea9697c152b3a4a0d2ad360",
      "driver_source_sha256": "81d2f14b5d59cc03a40d25e75899ce2f30091eef3f1ceba141fd9216184dc5b0",
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
      "built_at_utc": "2026-10-20T15:05:57.800165+00:00",
      "binary_path": "/Users/randyzwitch/GitHub/dataframe_mojo-metal-native-join-baseline-a523/build/bench_metal",
      "binary_sha256": "1f4eb0ebab4f7eccf034c0ad8f4c52a5486ef3c997d36cab01f988f1ec731ab4",
      "build_command": [
        "mojo",
        "build",
        "-O3",
        "-I",
        ".",
        "benchmarks/bench_metal.mojo",
        "-o",
        "/Users/randyzwitch/GitHub/dataframe_mojo-metal-native-join-baseline-a523/build/bench_metal"
      ],
      "git_commit": "46aea1f009fe9b7f038a10c1fde97d29fb5f99cf",
      "git_status": "",
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
      "benchmark_source_sha256": "cd8db3c71a6d4acb08bc71f1a8d14bedf64dcdbb7ea9697c152b3a4a0d2ad360",
      "driver_source_sha256": "81d2f14b5d59cc03a40d25e75899ce2f30091eef3f1ceba141fd9216184dc5b0",
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
  "source_sha256": "cd8db3c71a6d4acb08bc71f1a8d14bedf64dcdbb7ea9697c152b3a4a0d2ad360",
  "driver_sha256": "81d2f14b5d59cc03a40d25e75899ce2f30091eef3f1ceba141fd9216184dc5b0",
  "environment": {},
  "method": "Fresh process per case/build/round; first-use collections retained separately; configured extra warmups per engine; alternating CPU/Metal timed order; process/build order rotated per round; inputs and exact CPU-reference validation untimed; complete result materialization timed; result release untimed.",
  "scope": "Development mechanisms only, not external-suite evidence or a general GPU speedup. Chain32 includes fusion advantages over the current CPU executor.",
  "data": "Row mechanisms use a deterministic 1024-value numeric distribution. Algorithm cases use Int32 keys: group_hash and group_skew have min(N,1024) classes; group_high, join_hash and sort_radix have max(1,N/2) classes; join_fanout uses ceil(N/4) cyclic or monotone balanced classes, so each key occurs at most four times and a self-left-join emits at most 4N rows. Base non-fanout keys are (37*i) modulo cardinality; sorted keys are monotone bins. group_skew maps three quarters of rows to key zero. Null variants invalidate every twentieth key; grouped values additionally invalidate every thirteenth row. Grouped sums use i modulo 17 minus 8; sort and self-left-join carry original Int64 row IDs. Float32 row values are exact multiples of 1/1024. Float64 arithmetic uses exact binary fractions k/1024 with k in [0,1023] and evaluates x*2+0.25. Decimal input is Decimal(15,2) cents with unscaled values in [-8,8]; the query multiplies by Decimal(15,2) 1.25 then sums. Wide Float32 projects eight columns with binary-exact offsets [0,0.125,...,0.875], at most 8N output values or 256 MiB at the default 8,388,608 rows. Optional int32_affine_sum and filtered_int32_affine_sum workloads use the same Int32 base/sorted/nulls input patterns as int_sum, compute sum(cast_Int64((x+3)*2)), and optionally filter x>0 before the expression. Optional native64 workloads use typed Int64 values i%17-8 and typed UInt64 values i%17; sorted variants use monotone bins over the same ranges, and null variants use an LCG-distributed value with every twentieth row null. int64_affine_sum sums (x+3)*2, whose per-row magnitude is at most 22; int64_wide_sum16 emits 16 differently aliased sums of the same Int64 input. The harness validates every output against its exact CPU result, which is not an independent oracle. The optional integer window workloads reuse the Int32 int_sum input patterns, retain all N output rows, and compute cumulative SUM or rolling MIN/MAX with min_count=1 and widths encoded in their names (257 or 4097). Optional sort_topk31 and sort_topk1024 reuse sort_radix keys and original Int64 row IDs, perform stable ascending nulls-last ordering, and retain min(N,K) rows. These optional workloads are explicitly selectable with --workloads and excluded from the default set.",
  "native64_sum_cases": {
    "int64_sum": {
      "input_dtype": "Int64",
      "input_values": "i % 17 - 8",
      "expression": "sum(x)",
      "maximum_absolute_result_per_input_row": 8
    },
    "uint64_sum": {
      "input_dtype": "UInt64",
      "input_values": "i % 17",
      "expression": "sum(x)",
      "maximum_result_per_input_row": 16
    },
    "int64_affine_sum": {
      "input_dtype": "Int64",
      "input_values": "i % 17 - 8",
      "expression": "sum((x + 3) * 2)",
      "maximum_absolute_result_per_input_row": 22
    },
    "int64_wide_sum16": {
      "input_dtype": "Int64",
      "input_values": "i % 17 - 8",
      "expression": "16 separately aliased sum(x) outputs",
      "maximum_absolute_result_per_input_row": 8
    }
  },
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
