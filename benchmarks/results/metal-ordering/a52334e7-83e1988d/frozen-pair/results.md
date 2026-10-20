# Apple Metal development measurements

Development mechanisms only, not external-suite evidence or a general GPU speedup. Chain32 includes fusion advantages over the current CPU executor.

Times below are medians of complete warm collections, including allocation, staging, synchronization, and CPU result construction. First-use and all individual samples remain in the JSON. Every selected failed or unsupported case remains visible.

| Runtime | Rows | Workload | Variant | CPU ms | Metal ms | CPU/Metal | Launches | Peak payload MiB |
|---|---:|---|---|---:|---:|---:|---:|---:|
| baseline | 4097 | sort_radix | base | 0.184 | 6.624 | 0.03 | 16 | 0.59 |
| baseline | 4097 | sort_radix | nulls | 0.181 | 6.682 | 0.03 | 16 | 0.59 |
| baseline | 4097 | sort_radix | sorted | 0.071 | 6.741 | 0.01 | 16 | 0.59 |
| baseline | 4097 | sort_topk1024 | base | 0.165 | 6.301 | 0.03 | 16 | 0.54 |
| baseline | 4097 | sort_topk1024 | nulls | 0.172 | 6.510 | 0.03 | 16 | 0.54 |
| baseline | 4097 | sort_topk1024 | sorted | 0.070 | 6.592 | 0.01 | 16 | 0.54 |
| baseline | 4097 | sort_topk31 | base | 0.051 | 6.095 | 0.01 | 16 | 0.52 |
| baseline | 4097 | sort_topk31 | nulls | 0.051 | 6.146 | 0.01 | 16 | 0.52 |
| baseline | 4097 | sort_topk31 | sorted | 0.038 | 6.388 | 0.01 | 16 | 0.52 |
| baseline | 1048576 | sort_radix | base | 11.069 | 338.280 | 0.03 | 23 | 150.28 |
| baseline | 1048576 | sort_radix | nulls | 12.438 | 325.756 | 0.04 | 23 | 150.28 |
| baseline | 1048576 | sort_radix | sorted | 8.541 | 327.478 | 0.03 | 23 | 150.28 |
| baseline | 1048576 | sort_topk1024 | base | 10.705 | 174.535 | 0.06 | 23 | 134.05 |
| baseline | 1048576 | sort_topk1024 | nulls | 12.054 | 172.134 | 0.07 | 23 | 134.05 |
| baseline | 1048576 | sort_topk1024 | sorted | 12.716 | 169.907 | 0.07 | 23 | 134.05 |
| baseline | 1048576 | sort_topk31 | base | 11.411 | 172.419 | 0.07 | 23 | 134.03 |
| baseline | 1048576 | sort_topk31 | nulls | 13.609 | 162.845 | 0.08 | 23 | 134.03 |
| baseline | 1048576 | sort_topk31 | sorted | 12.177 | 172.634 | 0.07 | 23 | 134.03 |
| mojo | 4097 | sort_radix | base | 0.169 | 5.901 | 0.03 | 39 | 0.59 |
| mojo | 4097 | sort_radix | nulls | 0.181 | 6.112 | 0.03 | 39 | 0.59 |
| mojo | 4097 | sort_radix | sorted | 0.069 | 5.827 | 0.01 | 39 | 0.59 |
| mojo | 4097 | sort_topk1024 | base | 0.147 | 4.965 | 0.03 | 17 | 0.29 |
| mojo | 4097 | sort_topk1024 | nulls | 0.155 | 5.129 | 0.03 | 17 | 0.29 |
| mojo | 4097 | sort_topk1024 | sorted | 0.056 | 4.911 | 0.01 | 17 | 0.29 |
| mojo | 4097 | sort_topk31 | base | 0.040 | 4.742 | 0.01 | 17 | 0.20 |
| mojo | 4097 | sort_topk31 | nulls | 0.049 | 4.677 | 0.01 | 17 | 0.20 |
| mojo | 4097 | sort_topk31 | sorted | 0.034 | 4.579 | 0.01 | 17 | 0.20 |
| mojo | 1048576 | sort_radix | base | 10.041 | 211.320 | 0.05 | 57 | 150.53 |
| mojo | 1048576 | sort_radix | nulls | 14.557 | 212.279 | 0.07 | 57 | 150.53 |
| mojo | 1048576 | sort_radix | sorted | 5.361 | 209.435 | 0.03 | 57 | 150.53 |
| mojo | 1048576 | sort_topk1024 | base | 2.575 | 47.724 | 0.05 | 24 | 50.13 |
| mojo | 1048576 | sort_topk1024 | nulls | 3.374 | 46.847 | 0.07 | 24 | 50.13 |
| mojo | 1048576 | sort_topk1024 | sorted | 1.599 | 46.742 | 0.03 | 24 | 50.13 |
| mojo | 1048576 | sort_topk31 | base | 1.037 | 42.679 | 0.02 | 24 | 50.03 |
| mojo | 1048576 | sort_topk31 | nulls | 1.974 | 43.018 | 0.05 | 24 | 50.03 |
| mojo | 1048576 | sort_topk31 | sorted | 1.246 | 42.315 | 0.03 | 24 | 50.03 |

## Worst measured variant per case

Lowest CPU/Metal ratio among the selected variants; ratios below one favor CPU. These are per-mechanism results, not an aggregate suite score.

| Runtime | Rows | Workload | Worst variant | CPU/Metal |
|---|---:|---|---|---:|
| baseline | 4097 | sort_radix | sorted | 0.01 |
| baseline | 4097 | sort_topk1024 | sorted | 0.01 |
| baseline | 4097 | sort_topk31 | sorted | 0.01 |
| baseline | 1048576 | sort_radix | sorted | 0.03 |
| baseline | 1048576 | sort_topk1024 | base | 0.06 |
| baseline | 1048576 | sort_topk31 | base | 0.07 |
| mojo | 4097 | sort_radix | sorted | 0.01 |
| mojo | 4097 | sort_topk1024 | sorted | 0.01 |
| mojo | 4097 | sort_topk31 | sorted | 0.01 |
| mojo | 1048576 | sort_radix | sorted | 0.03 |
| mojo | 1048576 | sort_topk1024 | sorted | 0.03 |
| mojo | 1048576 | sort_topk31 | base | 0.02 |

## Measured Metal slowdowns

Cases with CPU/Metal below one, ordered by the largest Metal slowdown. The slowdown factor is Metal time divided by CPU time.

| Runtime | Rows | Workload | Variant | CPU/Metal | Metal slower by |
|---|---:|---|---|---:|---:|
| baseline | 4097 | sort_topk31 | sorted | 0.01 | 168.11x |
| mojo | 4097 | sort_topk31 | sorted | 0.01 | 134.68x |
| baseline | 4097 | sort_topk31 | nulls | 0.01 | 120.51x |
| baseline | 4097 | sort_topk31 | base | 0.01 | 119.51x |
| mojo | 4097 | sort_topk31 | base | 0.01 | 118.55x |
| mojo | 4097 | sort_topk31 | nulls | 0.01 | 95.45x |
| baseline | 4097 | sort_radix | sorted | 0.01 | 94.94x |
| baseline | 4097 | sort_topk1024 | sorted | 0.01 | 94.17x |
| mojo | 4097 | sort_topk1024 | sorted | 0.01 | 87.70x |
| mojo | 4097 | sort_radix | sorted | 0.01 | 84.45x |
| mojo | 1048576 | sort_topk31 | base | 0.02 | 41.16x |
| mojo | 1048576 | sort_radix | sorted | 0.03 | 39.07x |
| baseline | 1048576 | sort_radix | sorted | 0.03 | 38.34x |
| baseline | 4097 | sort_topk1024 | base | 0.03 | 38.19x |
| baseline | 4097 | sort_topk1024 | nulls | 0.03 | 37.85x |
| baseline | 4097 | sort_radix | nulls | 0.03 | 36.92x |
| baseline | 4097 | sort_radix | base | 0.03 | 36.00x |
| mojo | 4097 | sort_radix | base | 0.03 | 34.92x |
| mojo | 1048576 | sort_topk31 | sorted | 0.03 | 33.96x |
| mojo | 4097 | sort_topk1024 | base | 0.03 | 33.78x |
| mojo | 4097 | sort_radix | nulls | 0.03 | 33.77x |
| mojo | 4097 | sort_topk1024 | nulls | 0.03 | 33.09x |
| baseline | 1048576 | sort_radix | base | 0.03 | 30.56x |
| mojo | 1048576 | sort_topk1024 | sorted | 0.03 | 29.23x |
| baseline | 1048576 | sort_radix | nulls | 0.04 | 26.19x |
| mojo | 1048576 | sort_topk31 | nulls | 0.05 | 21.79x |
| mojo | 1048576 | sort_radix | base | 0.05 | 21.05x |
| mojo | 1048576 | sort_topk1024 | base | 0.05 | 18.53x |
| baseline | 1048576 | sort_topk1024 | base | 0.06 | 16.30x |
| baseline | 1048576 | sort_topk31 | base | 0.07 | 15.11x |
| mojo | 1048576 | sort_radix | nulls | 0.07 | 14.58x |
| baseline | 1048576 | sort_topk1024 | nulls | 0.07 | 14.28x |
| baseline | 1048576 | sort_topk31 | sorted | 0.07 | 14.18x |
| mojo | 1048576 | sort_topk1024 | nulls | 0.07 | 13.88x |
| baseline | 1048576 | sort_topk1024 | sorted | 0.07 | 13.36x |
| baseline | 1048576 | sort_topk31 | nulls | 0.08 | 11.97x |

Process outcomes: {'ok': 108, 'unsupported': 0, 'failed': 0}.

## Provenance and limits

```json
{
  "recorded_at_utc": "2026-10-20T16:16:30.641249+00:00",
  "platform": "macOS-26.5.2-arm64-arm-64bit-Mach-O",
  "machine": "arm64",
  "cpu": "Apple M1",
  "physical_cores": "8",
  "memory_bytes": "17179869184",
  "git_commit": "b390cd46446b8096fde4e136adade38d7526ec68",
  "source_tree": {
    "status": "",
    "dataframe_status": "",
    "dataframe_diff_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "dataframe_content_sha256": "65eed3cd1fca25f0a3baa583bda5d4f698f9e37a61cf6ee307ee83473fd28ebc"
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
    "mojo": "265352a05c0b27dc9c2bcdb1827396e9899f1e4e098af143f9924d673b0c7533",
    "baseline": "2afb0d2b915f65c98d222f936f5f4fda33f43a0cee04397029b004177b5db175"
  },
  "binary_receipts": {
    "mojo": {
      "status": "verified",
      "schema_version": 1,
      "built_at_utc": "2026-10-20T16:16:28.605610+00:00",
      "binary_path": "/Users/randyzwitch/GitHub/dataframe_mojo-metal-ordering-candidate-83/build/bench_metal",
      "binary_sha256": "265352a05c0b27dc9c2bcdb1827396e9899f1e4e098af143f9924d673b0c7533",
      "build_command": [
        "mojo",
        "build",
        "-O3",
        "-I",
        ".",
        "benchmarks/bench_metal.mojo",
        "-o",
        "/Users/randyzwitch/GitHub/dataframe_mojo-metal-ordering-candidate-83/build/bench_metal"
      ],
      "git_commit": "b390cd46446b8096fde4e136adade38d7526ec68",
      "git_status": "",
      "engine_status": "",
      "engine_content_sha256": "65eed3cd1fca25f0a3baa583bda5d4f698f9e37a61cf6ee307ee83473fd28ebc",
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
      "built_at_utc": "2026-10-20T16:04:11.369299+00:00",
      "binary_path": "/Users/randyzwitch/GitHub/dataframe_mojo-metal-ordering-frozen-a523/build/bench_metal",
      "binary_sha256": "2afb0d2b915f65c98d222f936f5f4fda33f43a0cee04397029b004177b5db175",
      "build_command": [
        "mojo",
        "build",
        "-O3",
        "-I",
        ".",
        "benchmarks/bench_metal.mojo",
        "-o",
        "/Users/randyzwitch/GitHub/dataframe_mojo-metal-ordering-frozen-a523/build/bench_metal"
      ],
      "git_commit": "a97c026c2addd6015e6225f4b2cc364ccfc5b12b",
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
