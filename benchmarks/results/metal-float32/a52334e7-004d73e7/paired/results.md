# Apple Metal development measurements

Development mechanisms only, not external-suite evidence or a general GPU speedup. Chain32 includes fusion advantages over the current CPU executor.

Times below are medians of complete warm collections, including allocation, staging, synchronization, and CPU result construction. First-use and all individual samples remain in the JSON. Every selected failed or unsupported case remains visible.

| Runtime | Rows | Workload | Variant | CPU ms | Metal ms | CPU/Metal | Launches | Peak payload MiB |
|---|---:|---|---|---:|---:|---:|---:|---:|
| baseline | 4097 | filtered_float32_affine_sum | base | 0.126 | 7.299 | 0.02 | 8 | 0.76 |
| baseline | 4097 | filtered_float32_affine_sum | nulls | 0.139 | 7.118 | 0.02 | 8 | 0.76 |
| baseline | 4097 | filtered_float32_affine_sum | sorted | 0.107 | 6.998 | 0.02 | 8 | 0.76 |
| baseline | 4097 | float32_affine_sum | base | 0.089 | 6.114 | 0.01 | 4 | 0.33 |
| baseline | 4097 | float32_affine_sum | nulls | 0.088 | 6.279 | 0.01 | 4 | 0.33 |
| baseline | 4097 | float32_affine_sum | sorted | 0.106 | 6.217 | 0.02 | 4 | 0.33 |
| baseline | 1048576 | filtered_float32_affine_sum | base | 3.194 | 31.698 | 0.10 | 8 | 193.09 |
| baseline | 1048576 | filtered_float32_affine_sum | nulls | 3.295 | 32.735 | 0.10 | 8 | 193.09 |
| baseline | 1048576 | filtered_float32_affine_sum | sorted | 3.303 | 32.420 | 0.10 | 8 | 193.09 |
| baseline | 1048576 | float32_affine_sum | base | 3.889 | 26.989 | 0.14 | 4 | 84.06 |
| baseline | 1048576 | float32_affine_sum | nulls | 3.781 | 28.108 | 0.13 | 4 | 84.06 |
| baseline | 1048576 | float32_affine_sum | sorted | 3.901 | 28.295 | 0.14 | 4 | 84.06 |
| mojo | 4097 | filtered_float32_affine_sum | base | 0.088 | 1.120 | 0.08 | 2 | 0.02 |
| mojo | 4097 | filtered_float32_affine_sum | nulls | 0.102 | 1.460 | 0.07 | 2 | 0.02 |
| mojo | 4097 | filtered_float32_affine_sum | sorted | 0.091 | 1.202 | 0.08 | 2 | 0.02 |
| mojo | 4097 | float32_affine_sum | base | 0.082 | 1.219 | 0.07 | 2 | 0.02 |
| mojo | 4097 | float32_affine_sum | nulls | 0.081 | 1.239 | 0.07 | 2 | 0.02 |
| mojo | 4097 | float32_affine_sum | sorted | 0.083 | 1.026 | 0.08 | 2 | 0.02 |
| mojo | 1048576 | filtered_float32_affine_sum | base | 1.766 | 5.138 | 0.34 | 2 | 4.01 |
| mojo | 1048576 | filtered_float32_affine_sum | nulls | 2.763 | 7.986 | 0.35 | 2 | 4.14 |
| mojo | 1048576 | filtered_float32_affine_sum | sorted | 2.417 | 5.656 | 0.43 | 2 | 4.01 |
| mojo | 1048576 | float32_affine_sum | base | 2.778 | 6.424 | 0.43 | 2 | 4.01 |
| mojo | 1048576 | float32_affine_sum | nulls | 2.208 | 7.787 | 0.28 | 2 | 4.14 |
| mojo | 1048576 | float32_affine_sum | sorted | 3.022 | 6.273 | 0.48 | 2 | 4.01 |

## Worst measured variant per case

Lowest CPU/Metal ratio among the selected variants; ratios below one favor CPU. These are per-mechanism results, not an aggregate suite score.

| Runtime | Rows | Workload | Worst variant | CPU/Metal |
|---|---:|---|---|---:|
| baseline | 4097 | filtered_float32_affine_sum | sorted | 0.02 |
| baseline | 4097 | float32_affine_sum | nulls | 0.01 |
| baseline | 1048576 | filtered_float32_affine_sum | nulls | 0.10 |
| baseline | 1048576 | float32_affine_sum | nulls | 0.13 |
| mojo | 4097 | filtered_float32_affine_sum | nulls | 0.07 |
| mojo | 4097 | float32_affine_sum | nulls | 0.07 |
| mojo | 1048576 | filtered_float32_affine_sum | base | 0.34 |
| mojo | 1048576 | float32_affine_sum | nulls | 0.28 |

## Measured Metal slowdowns

Cases with CPU/Metal below one, ordered by the largest Metal slowdown. The slowdown factor is Metal time divided by CPU time.

| Runtime | Rows | Workload | Variant | CPU/Metal | Metal slower by |
|---|---:|---|---|---:|---:|
| baseline | 4097 | float32_affine_sum | nulls | 0.01 | 71.35x |
| baseline | 4097 | float32_affine_sum | base | 0.01 | 68.70x |
| baseline | 4097 | filtered_float32_affine_sum | sorted | 0.02 | 65.40x |
| baseline | 4097 | float32_affine_sum | sorted | 0.02 | 58.65x |
| baseline | 4097 | filtered_float32_affine_sum | base | 0.02 | 57.93x |
| baseline | 4097 | filtered_float32_affine_sum | nulls | 0.02 | 51.21x |
| mojo | 4097 | float32_affine_sum | nulls | 0.07 | 15.30x |
| mojo | 4097 | float32_affine_sum | base | 0.07 | 14.87x |
| mojo | 4097 | filtered_float32_affine_sum | nulls | 0.07 | 14.31x |
| mojo | 4097 | filtered_float32_affine_sum | sorted | 0.08 | 13.21x |
| mojo | 4097 | filtered_float32_affine_sum | base | 0.08 | 12.73x |
| mojo | 4097 | float32_affine_sum | sorted | 0.08 | 12.36x |
| baseline | 1048576 | filtered_float32_affine_sum | nulls | 0.10 | 9.93x |
| baseline | 1048576 | filtered_float32_affine_sum | base | 0.10 | 9.92x |
| baseline | 1048576 | filtered_float32_affine_sum | sorted | 0.10 | 9.82x |
| baseline | 1048576 | float32_affine_sum | nulls | 0.13 | 7.43x |
| baseline | 1048576 | float32_affine_sum | sorted | 0.14 | 7.25x |
| baseline | 1048576 | float32_affine_sum | base | 0.14 | 6.94x |
| mojo | 1048576 | float32_affine_sum | nulls | 0.28 | 3.53x |
| mojo | 1048576 | filtered_float32_affine_sum | base | 0.34 | 2.91x |
| mojo | 1048576 | filtered_float32_affine_sum | nulls | 0.35 | 2.89x |
| mojo | 1048576 | filtered_float32_affine_sum | sorted | 0.43 | 2.34x |
| mojo | 1048576 | float32_affine_sum | base | 0.43 | 2.31x |
| mojo | 1048576 | float32_affine_sum | sorted | 0.48 | 2.08x |

Process outcomes: {'ok': 72, 'unsupported': 0, 'failed': 0}.

## Provenance and limits

```json
{
  "recorded_at_utc": "2026-10-20T14:58:07.791520+00:00",
  "platform": "macOS-26.5.2-arm64-arm-64bit-Mach-O",
  "machine": "arm64",
  "cpu": "Apple M1",
  "physical_cores": "8",
  "memory_bytes": "17179869184",
  "git_commit": "e90323de8965bb775abbdb91d336bc5fda530d89",
  "source_tree": {
    "status": "",
    "dataframe_status": "",
    "dataframe_diff_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "dataframe_content_sha256": "0e6bfbf67197885a9abeffc8f12c275a8d7f39825cf4110a08a788af77581025"
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
    "mojo": "98bd5c12553ee3577a75b7b3e9f6d55ced1dcc04ba3187b55dace7d2f281c3c8",
    "baseline": "09e48a5a7507696776bd4defca65f3965f471cf1a608aa5d9136e3cb5338fc60"
  },
  "binary_receipts": {
    "mojo": {
      "status": "verified",
      "schema_version": 1,
      "built_at_utc": "2026-10-20T14:58:06.165525+00:00",
      "binary_path": "/Users/randyzwitch/GitHub/dataframe_mojo-metal-paired-float32/build/bench_metal",
      "binary_sha256": "98bd5c12553ee3577a75b7b3e9f6d55ced1dcc04ba3187b55dace7d2f281c3c8",
      "build_command": [
        "mojo",
        "build",
        "-O3",
        "-I",
        ".",
        "benchmarks/bench_metal.mojo",
        "-o",
        "/Users/randyzwitch/GitHub/dataframe_mojo-metal-paired-float32/build/bench_metal"
      ],
      "git_commit": "e90323de8965bb775abbdb91d336bc5fda530d89",
      "git_status": "",
      "engine_status": "",
      "engine_content_sha256": "0e6bfbf67197885a9abeffc8f12c275a8d7f39825cf4110a08a788af77581025",
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
      "built_at_utc": "2026-10-20T14:52:11.677288+00:00",
      "binary_path": "/Users/randyzwitch/GitHub/dataframe_mojo-metal-float32-benchmark/build/bench_metal",
      "binary_sha256": "09e48a5a7507696776bd4defca65f3965f471cf1a608aa5d9136e3cb5338fc60",
      "build_command": [
        "mojo",
        "build",
        "-O3",
        "-I",
        ".",
        "benchmarks/bench_metal.mojo",
        "-o",
        "/Users/randyzwitch/GitHub/dataframe_mojo-metal-float32-benchmark/build/bench_metal"
      ],
      "git_commit": "ee063de049f9fcbfd95184c9b310304e1ce1fe56",
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
