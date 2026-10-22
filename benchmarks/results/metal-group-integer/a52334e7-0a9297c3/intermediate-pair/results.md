# Apple Metal development measurements

Development mechanisms only, not external-suite evidence or a general GPU speedup. Chain32 includes fusion advantages over the current CPU executor.

Times below are medians of complete warm collections, including allocation, staging, synchronization, and CPU result construction. First-use and all individual samples remain in the JSON. Every selected failed or unsupported case remains visible.

| Runtime | Rows | Workload | Variant | CPU ms | Metal ms | CPU/Metal | Launches | Peak payload MiB |
|---|---:|---|---|---:|---:|---:|---:|---:|
| baseline | 4097 | group_hash | base | 0.204 | 11.668 | 0.02 | 24 | 2.48 |
| baseline | 4097 | group_hash | nulls | 0.223 | 12.073 | 0.02 | 25 | 2.48 |
| baseline | 4097 | group_hash | sorted | 0.236 | 11.703 | 0.02 | 24 | 2.48 |
| baseline | 4097 | group_high | base | 0.292 | 11.936 | 0.02 | 25 | 2.48 |
| baseline | 4097 | group_high | nulls | 0.326 | 12.188 | 0.03 | 26 | 2.48 |
| baseline | 4097 | group_high | sorted | 0.259 | 11.927 | 0.02 | 25 | 2.48 |
| baseline | 4097 | group_skew | base | 0.194 | 16.967 | 0.01 | 22 | 2.48 |
| baseline | 4097 | group_skew | nulls | 0.216 | 16.513 | 0.01 | 23 | 2.48 |
| baseline | 4097 | group_skew | sorted | 0.235 | 16.876 | 0.01 | 24 | 2.48 |
| baseline | 1048576 | group_hash | base | 7.335 | 62.827 | 0.12 | 24 | 611.41 |
| baseline | 1048576 | group_hash | nulls | 11.837 | 130.169 | 0.09 | 25 | 611.41 |
| baseline | 1048576 | group_hash | sorted | 7.207 | 62.318 | 0.12 | 24 | 611.41 |
| baseline | 1048576 | group_high | base | 76.339 | 275.631 | 0.28 | 33 | 611.41 |
| baseline | 1048576 | group_high | nulls | 66.978 | 373.905 | 0.18 | 34 | 611.41 |
| baseline | 1048576 | group_high | sorted | 73.816 | 270.010 | 0.27 | 33 | 611.41 |
| baseline | 1048576 | group_skew | base | 11.370 | 917.547 | 0.01 | 22 | 611.41 |
| baseline | 1048576 | group_skew | nulls | 13.260 | 884.402 | 0.01 | 23 | 611.41 |
| baseline | 1048576 | group_skew | sorted | 13.726 | 906.675 | 0.02 | 24 | 611.41 |
| mojo | 4097 | group_hash | base | 0.176 | 8.336 | 0.02 | 24 | 2.73 |
| mojo | 4097 | group_hash | nulls | 0.191 | 8.876 | 0.02 | 25 | 2.73 |
| mojo | 4097 | group_hash | sorted | 0.186 | 7.794 | 0.02 | 24 | 2.73 |
| mojo | 4097 | group_high | base | 0.243 | 8.942 | 0.03 | 25 | 2.73 |
| mojo | 4097 | group_high | nulls | 0.268 | 9.343 | 0.03 | 26 | 2.73 |
| mojo | 4097 | group_high | sorted | 0.230 | 8.864 | 0.03 | 25 | 2.73 |
| mojo | 4097 | group_skew | base | 0.180 | 14.190 | 0.01 | 22 | 2.73 |
| mojo | 4097 | group_skew | nulls | 0.196 | 14.042 | 0.01 | 23 | 2.73 |
| mojo | 4097 | group_skew | sorted | 0.202 | 14.624 | 0.01 | 24 | 2.73 |
| mojo | 1048576 | group_hash | base | 5.182 | 67.008 | 0.08 | 28 | 611.66 |
| mojo | 1048576 | group_hash | nulls | 6.331 | 69.005 | 0.09 | 29 | 611.66 |
| mojo | 1048576 | group_hash | sorted | 4.598 | 66.828 | 0.07 | 28 | 611.66 |
| mojo | 1048576 | group_high | base | 65.882 | 270.559 | 0.24 | 33 | 611.66 |
| mojo | 1048576 | group_high | nulls | 67.741 | 368.420 | 0.18 | 34 | 611.66 |
| mojo | 1048576 | group_high | sorted | 64.455 | 264.581 | 0.24 | 33 | 611.66 |
| mojo | 1048576 | group_skew | base | 7.811 | 187.788 | 0.04 | 26 | 611.66 |
| mojo | 1048576 | group_skew | nulls | 9.144 | 185.979 | 0.05 | 27 | 611.66 |
| mojo | 1048576 | group_skew | sorted | 8.959 | 185.227 | 0.05 | 28 | 611.66 |

## Worst measured variant per case

Lowest CPU/Metal ratio among the selected variants; ratios below one favor CPU. These are per-mechanism results, not an aggregate suite score.

| Runtime | Rows | Workload | Worst variant | CPU/Metal |
|---|---:|---|---|---:|
| baseline | 4097 | group_hash | base | 0.02 |
| baseline | 4097 | group_high | sorted | 0.02 |
| baseline | 4097 | group_skew | base | 0.01 |
| baseline | 1048576 | group_hash | nulls | 0.09 |
| baseline | 1048576 | group_high | nulls | 0.18 |
| baseline | 1048576 | group_skew | base | 0.01 |
| mojo | 4097 | group_hash | base | 0.02 |
| mojo | 4097 | group_high | sorted | 0.03 |
| mojo | 4097 | group_skew | base | 0.01 |
| mojo | 1048576 | group_hash | sorted | 0.07 |
| mojo | 1048576 | group_high | nulls | 0.18 |
| mojo | 1048576 | group_skew | base | 0.04 |

## Measured Metal slowdowns

Cases with CPU/Metal below one, ordered by the largest Metal slowdown. The slowdown factor is Metal time divided by CPU time.

| Runtime | Rows | Workload | Variant | CPU/Metal | Metal slower by |
|---|---:|---|---|---:|---:|
| baseline | 4097 | group_skew | base | 0.01 | 87.46x |
| baseline | 1048576 | group_skew | base | 0.01 | 80.70x |
| mojo | 4097 | group_skew | base | 0.01 | 78.83x |
| baseline | 4097 | group_skew | nulls | 0.01 | 76.45x |
| mojo | 4097 | group_skew | sorted | 0.01 | 72.40x |
| baseline | 4097 | group_skew | sorted | 0.01 | 71.81x |
| mojo | 4097 | group_skew | nulls | 0.01 | 71.64x |
| baseline | 1048576 | group_skew | nulls | 0.01 | 66.70x |
| baseline | 1048576 | group_skew | sorted | 0.02 | 66.06x |
| baseline | 4097 | group_hash | base | 0.02 | 57.20x |
| baseline | 4097 | group_hash | nulls | 0.02 | 54.14x |
| baseline | 4097 | group_hash | sorted | 0.02 | 49.59x |
| mojo | 4097 | group_hash | base | 0.02 | 47.36x |
| mojo | 4097 | group_hash | nulls | 0.02 | 46.47x |
| baseline | 4097 | group_high | sorted | 0.02 | 46.05x |
| mojo | 4097 | group_hash | sorted | 0.02 | 41.90x |
| baseline | 4097 | group_high | base | 0.02 | 40.88x |
| mojo | 4097 | group_high | sorted | 0.03 | 38.54x |
| baseline | 4097 | group_high | nulls | 0.03 | 37.39x |
| mojo | 4097 | group_high | base | 0.03 | 36.80x |
| mojo | 4097 | group_high | nulls | 0.03 | 34.86x |
| mojo | 1048576 | group_skew | base | 0.04 | 24.04x |
| mojo | 1048576 | group_skew | sorted | 0.05 | 20.67x |
| mojo | 1048576 | group_skew | nulls | 0.05 | 20.34x |
| mojo | 1048576 | group_hash | sorted | 0.07 | 14.53x |
| mojo | 1048576 | group_hash | base | 0.08 | 12.93x |
| baseline | 1048576 | group_hash | nulls | 0.09 | 11.00x |
| mojo | 1048576 | group_hash | nulls | 0.09 | 10.90x |
| baseline | 1048576 | group_hash | sorted | 0.12 | 8.65x |
| baseline | 1048576 | group_hash | base | 0.12 | 8.57x |
| baseline | 1048576 | group_high | nulls | 0.18 | 5.58x |
| mojo | 1048576 | group_high | nulls | 0.18 | 5.44x |
| mojo | 1048576 | group_high | base | 0.24 | 4.11x |
| mojo | 1048576 | group_high | sorted | 0.24 | 4.10x |
| baseline | 1048576 | group_high | sorted | 0.27 | 3.66x |
| baseline | 1048576 | group_high | base | 0.28 | 3.61x |

Process outcomes: {'ok': 108, 'unsupported': 0, 'failed': 0}.

## Provenance and limits

```json
{
  "recorded_at_utc": "2026-10-20T14:43:48.508203+00:00",
  "platform": "macOS-26.5.2-arm64-arm-64bit-Mach-O",
  "machine": "arm64",
  "cpu": "Apple M1",
  "physical_cores": "8",
  "memory_bytes": "17179869184",
  "git_commit": "b3e231a652d5e64571a2ab416e93d037aa06afbb",
  "source_tree": {
    "status": "",
    "dataframe_status": "",
    "dataframe_diff_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "dataframe_content_sha256": "11b0037108c776e23b6bb9d6dd3389510982f92f28439cf4374ebbc2c1362261"
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
    "mojo": "8e6a646f1aa5579ba454b6795fa0c9bc6ed38c2f3e3e6869923048342bcbee44",
    "baseline": "ea9edee7a88d12d6e94d84fa3aafb9a6e78dad72308474d4c857c0f44a4b9185"
  },
  "binary_receipts": {
    "mojo": {
      "status": "verified",
      "schema_version": 1,
      "built_at_utc": "2026-10-20T14:35:49.465005+00:00",
      "binary_path": "/Users/randyzwitch/GitHub/dataframe_mojo-metal-paired-group-current/build/bench_metal",
      "binary_sha256": "8e6a646f1aa5579ba454b6795fa0c9bc6ed38c2f3e3e6869923048342bcbee44",
      "build_command": [
        "mojo",
        "build",
        "-O3",
        "-I",
        ".",
        "benchmarks/bench_metal.mojo",
        "-o",
        "/Users/randyzwitch/GitHub/dataframe_mojo-metal-paired-group-current/build/bench_metal"
      ],
      "git_commit": "b3e231a652d5e64571a2ab416e93d037aa06afbb",
      "git_status": "",
      "engine_status": "",
      "engine_content_sha256": "11b0037108c776e23b6bb9d6dd3389510982f92f28439cf4374ebbc2c1362261",
      "engine_diff_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      "compiler_identity": "Mojo 1.2.0.dev2026100905 (3b71af75)",
      "build_platform": "macOS-26.5.2-arm64-arm-64bit-Mach-O",
      "build_sdk": "26.5",
      "build_xcode": "Xcode 26.6\nBuild version 17F113",
      "compiler_executable": "/Users/randyzwitch/github/dataframe_mojo-metal-868/.pixi/envs/oracle/bin/mojo",
      "compiler_executable_sha256": "ac42575fdf41cd8ced537812c20cbdda361223434ba90c866c6c42007df1a37b",
      "pixi_lock_sha256": "098d02ba4d166a75640ed6b96aab77fdfe6cb468eded7838f23bba244496e277",
      "benchmark_source_sha256": "704598dab6810b1197074b35400839e3ca298ba1a04755cf35c8671fd3552e97",
      "driver_source_sha256": "09bcbbb472ef308a945b4d4d33783c6b832c34cce9359ac1e749f3ac98618fc7",
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
      "built_at_utc": "2026-10-20T14:43:46.352465+00:00",
      "binary_path": "/Users/randyzwitch/GitHub/dataframe_mojo-metal-paired-group-intermediate-v2/build/bench_metal",
      "binary_sha256": "ea9edee7a88d12d6e94d84fa3aafb9a6e78dad72308474d4c857c0f44a4b9185",
      "build_command": [
        "mojo",
        "build",
        "-O3",
        "-I",
        ".",
        "benchmarks/bench_metal.mojo",
        "-o",
        "/Users/randyzwitch/GitHub/dataframe_mojo-metal-paired-group-intermediate-v2/build/bench_metal"
      ],
      "git_commit": "f758ce9a067cd3ddf67b0c3393f5d26d684ddeb0",
      "git_status": "",
      "engine_status": "",
      "engine_content_sha256": "f10aba9f86f6e194c817bdd8d33bf4bce2ee81f4c577a696d672e9d7bb535b1f",
      "engine_diff_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      "compiler_identity": "Mojo 1.2.0.dev2026100905 (3b71af75)",
      "build_platform": "macOS-26.5.2-arm64-arm-64bit-Mach-O",
      "build_sdk": "26.5",
      "build_xcode": "Xcode 26.6\nBuild version 17F113",
      "compiler_executable": "/Users/randyzwitch/github/dataframe_mojo-metal-868/.pixi/envs/oracle/bin/mojo",
      "compiler_executable_sha256": "ac42575fdf41cd8ced537812c20cbdda361223434ba90c866c6c42007df1a37b",
      "pixi_lock_sha256": "098d02ba4d166a75640ed6b96aab77fdfe6cb468eded7838f23bba244496e277",
      "benchmark_source_sha256": "704598dab6810b1197074b35400839e3ca298ba1a04755cf35c8671fd3552e97",
      "driver_source_sha256": "09bcbbb472ef308a945b4d4d33783c6b832c34cce9359ac1e749f3ac98618fc7",
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
  "source_sha256": "704598dab6810b1197074b35400839e3ca298ba1a04755cf35c8671fd3552e97",
  "driver_sha256": "09bcbbb472ef308a945b4d4d33783c6b832c34cce9359ac1e749f3ac98618fc7",
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
