# Apple Metal development measurements

Development mechanisms only, not external-suite evidence or a general GPU speedup. Chain32 includes fusion advantages over the current CPU executor.

Times below are medians of complete warm collections, including allocation, staging, synchronization, and CPU result construction. First-use and all individual samples remain in the JSON. Every selected failed or unsupported case remains visible.

| Runtime | Rows | Workload | Variant | CPU ms | Metal ms | CPU/Metal | Launches | Peak payload MiB |
|---|---:|---|---|---:|---:|---:|---:|---:|
| baseline | 4097 | sort_topk1024 | base | 0.161 | 5.619 | 0.03 | 39 | 0.54 |
| baseline | 4097 | sort_topk1024 | nulls | 0.153 | 5.572 | 0.03 | 39 | 0.54 |
| baseline | 4097 | sort_topk1024 | sorted | 0.052 | 5.591 | 0.01 | 39 | 0.54 |
| baseline | 4097 | sort_topk31 | base | 0.042 | 5.986 | 0.01 | 39 | 0.53 |
| baseline | 4097 | sort_topk31 | nulls | 0.044 | 5.490 | 0.01 | 39 | 0.53 |
| baseline | 4097 | sort_topk31 | sorted | 0.033 | 5.501 | 0.01 | 39 | 0.53 |
| baseline | 1048576 | sort_topk1024 | base | 3.271 | 47.096 | 0.07 | 57 | 134.30 |
| baseline | 1048576 | sort_topk1024 | nulls | 3.854 | 47.323 | 0.08 | 57 | 134.30 |
| baseline | 1048576 | sort_topk1024 | sorted | 1.693 | 46.831 | 0.04 | 57 | 134.30 |
| baseline | 1048576 | sort_topk31 | base | 1.145 | 47.554 | 0.02 | 57 | 134.28 |
| baseline | 1048576 | sort_topk31 | nulls | 1.784 | 47.837 | 0.04 | 57 | 134.28 |
| baseline | 1048576 | sort_topk31 | sorted | 1.322 | 47.020 | 0.03 | 57 | 134.28 |
| mojo | 4097 | sort_topk1024 | base | 0.147 | 5.134 | 0.03 | 17 | 0.29 |
| mojo | 4097 | sort_topk1024 | nulls | 0.157 | 5.150 | 0.03 | 17 | 0.29 |
| mojo | 4097 | sort_topk1024 | sorted | 0.050 | 5.018 | 0.01 | 17 | 0.29 |
| mojo | 4097 | sort_topk31 | base | 0.038 | 4.832 | 0.01 | 17 | 0.20 |
| mojo | 4097 | sort_topk31 | nulls | 0.042 | 5.297 | 0.01 | 17 | 0.20 |
| mojo | 4097 | sort_topk31 | sorted | 0.033 | 5.267 | 0.01 | 17 | 0.20 |
| mojo | 1048576 | sort_topk1024 | base | 3.154 | 47.100 | 0.07 | 24 | 50.13 |
| mojo | 1048576 | sort_topk1024 | nulls | 3.483 | 47.356 | 0.07 | 24 | 50.13 |
| mojo | 1048576 | sort_topk1024 | sorted | 1.827 | 46.596 | 0.04 | 24 | 50.13 |
| mojo | 1048576 | sort_topk31 | base | 1.124 | 44.170 | 0.03 | 24 | 50.03 |
| mojo | 1048576 | sort_topk31 | nulls | 1.709 | 43.287 | 0.04 | 24 | 50.03 |
| mojo | 1048576 | sort_topk31 | sorted | 1.021 | 43.607 | 0.02 | 24 | 50.03 |

## Worst measured variant per case

Lowest CPU/Metal ratio among the selected variants; ratios below one favor CPU. These are per-mechanism results, not an aggregate suite score.

| Runtime | Rows | Workload | Worst variant | CPU/Metal |
|---|---:|---|---|---:|
| baseline | 4097 | sort_topk1024 | sorted | 0.01 |
| baseline | 4097 | sort_topk31 | sorted | 0.01 |
| baseline | 1048576 | sort_topk1024 | sorted | 0.04 |
| baseline | 1048576 | sort_topk31 | base | 0.02 |
| mojo | 4097 | sort_topk1024 | sorted | 0.01 |
| mojo | 4097 | sort_topk31 | sorted | 0.01 |
| mojo | 1048576 | sort_topk1024 | sorted | 0.04 |
| mojo | 1048576 | sort_topk31 | sorted | 0.02 |

## Measured Metal slowdowns

Cases with CPU/Metal below one, ordered by the largest Metal slowdown. The slowdown factor is Metal time divided by CPU time.

| Runtime | Rows | Workload | Variant | CPU/Metal | Metal slower by |
|---|---:|---|---|---:|---:|
| baseline | 4097 | sort_topk31 | sorted | 0.01 | 166.70x |
| mojo | 4097 | sort_topk31 | sorted | 0.01 | 159.61x |
| baseline | 4097 | sort_topk31 | base | 0.01 | 142.52x |
| mojo | 4097 | sort_topk31 | base | 0.01 | 127.16x |
| mojo | 4097 | sort_topk31 | nulls | 0.01 | 126.12x |
| baseline | 4097 | sort_topk31 | nulls | 0.01 | 124.77x |
| baseline | 4097 | sort_topk1024 | sorted | 0.01 | 107.52x |
| mojo | 4097 | sort_topk1024 | sorted | 0.01 | 100.36x |
| mojo | 1048576 | sort_topk31 | sorted | 0.02 | 42.71x |
| baseline | 1048576 | sort_topk31 | base | 0.02 | 41.53x |
| mojo | 1048576 | sort_topk31 | base | 0.03 | 39.30x |
| baseline | 4097 | sort_topk1024 | nulls | 0.03 | 36.42x |
| baseline | 1048576 | sort_topk31 | sorted | 0.03 | 35.57x |
| mojo | 4097 | sort_topk1024 | base | 0.03 | 34.93x |
| baseline | 4097 | sort_topk1024 | base | 0.03 | 34.90x |
| mojo | 4097 | sort_topk1024 | nulls | 0.03 | 32.80x |
| baseline | 1048576 | sort_topk1024 | sorted | 0.04 | 27.66x |
| baseline | 1048576 | sort_topk31 | nulls | 0.04 | 26.81x |
| mojo | 1048576 | sort_topk1024 | sorted | 0.04 | 25.50x |
| mojo | 1048576 | sort_topk31 | nulls | 0.04 | 25.33x |
| mojo | 1048576 | sort_topk1024 | base | 0.07 | 14.93x |
| baseline | 1048576 | sort_topk1024 | base | 0.07 | 14.40x |
| mojo | 1048576 | sort_topk1024 | nulls | 0.07 | 13.60x |
| baseline | 1048576 | sort_topk1024 | nulls | 0.08 | 12.28x |

Process outcomes: {'ok': 72, 'unsupported': 0, 'failed': 0}.

## Provenance and limits

```json
{
  "recorded_at_utc": "2026-10-20T16:17:36.739493+00:00",
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
    "baseline": "b746d0d4385a15a546d42d60d6b8d3c25811c27cd8224b11a1dc48e1c93552a2"
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
      "built_at_utc": "2026-10-20T16:10:03.063593+00:00",
      "binary_path": "/Users/randyzwitch/GitHub/dataframe_mojo-metal-ordering-radix-a03/build/bench_metal",
      "binary_sha256": "b746d0d4385a15a546d42d60d6b8d3c25811c27cd8224b11a1dc48e1c93552a2",
      "build_command": [
        "mojo",
        "build",
        "-O3",
        "-I",
        ".",
        "benchmarks/bench_metal.mojo",
        "-o",
        "/Users/randyzwitch/GitHub/dataframe_mojo-metal-ordering-radix-a03/build/bench_metal"
      ],
      "git_commit": "a2d3e36422c63b33164ddded18aa3855911f77b3",
      "git_status": "",
      "engine_status": "",
      "engine_content_sha256": "41ab5123c99bea4a3fcf834655adb1c0ec102a012a9a86e007f7e0764a41cbb9",
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
