# Apple Metal development measurements

Development mechanisms only, not external-suite evidence or a general GPU speedup. Chain32 includes fusion advantages over the current CPU executor.

Times below are medians of complete warm collections, including allocation, staging, synchronization, and CPU result construction. First-use and all individual samples remain in the JSON. Every selected failed or unsupported case remains visible.

| Runtime | Rows | Workload | Variant | CPU ms | Metal ms | CPU/Metal | Launches | Peak payload MiB |
|---|---:|---|---|---:|---:|---:|---:|---:|
| baseline | 4097 | group_hash | base | 0.207 | 12.179 | 0.02 | 35 | 2.36 |
| baseline | 4097 | group_hash | nulls | 0.221 | 12.662 | 0.02 | 36 | 2.36 |
| baseline | 4097 | group_hash | sorted | 0.237 | 12.268 | 0.02 | 35 | 2.36 |
| baseline | 4097 | group_high | base | 0.297 | 12.557 | 0.02 | 36 | 2.36 |
| baseline | 4097 | group_high | nulls | 0.325 | 12.891 | 0.03 | 37 | 2.36 |
| baseline | 4097 | group_high | sorted | 0.267 | 11.942 | 0.02 | 36 | 2.36 |
| baseline | 4097 | group_skew | base | 0.194 | 16.724 | 0.01 | 33 | 2.36 |
| baseline | 4097 | group_skew | nulls | 0.215 | 16.403 | 0.01 | 34 | 2.36 |
| baseline | 4097 | group_skew | sorted | 0.244 | 17.496 | 0.01 | 35 | 2.36 |
| baseline | 1048576 | group_hash | base | 11.182 | 219.186 | 0.05 | 42 | 603.41 |
| baseline | 1048576 | group_hash | nulls | 12.850 | 301.435 | 0.04 | 43 | 603.41 |
| baseline | 1048576 | group_hash | sorted | 11.121 | 214.340 | 0.05 | 42 | 603.41 |
| baseline | 1048576 | group_high | base | 69.744 | 434.918 | 0.16 | 51 | 603.41 |
| baseline | 1048576 | group_high | nulls | 76.173 | 490.962 | 0.16 | 52 | 603.41 |
| baseline | 1048576 | group_high | sorted | 66.203 | 435.357 | 0.15 | 51 | 603.41 |
| baseline | 1048576 | group_skew | base | 12.575 | 960.671 | 0.01 | 40 | 603.41 |
| baseline | 1048576 | group_skew | nulls | 14.602 | 938.473 | 0.02 | 41 | 603.41 |
| baseline | 1048576 | group_skew | sorted | 13.662 | 951.912 | 0.01 | 42 | 603.41 |
| mojo | 4097 | group_hash | base | 0.176 | 8.506 | 0.02 | 24 | 2.73 |
| mojo | 4097 | group_hash | nulls | 0.190 | 8.934 | 0.02 | 25 | 2.73 |
| mojo | 4097 | group_hash | sorted | 0.185 | 8.558 | 0.02 | 24 | 2.73 |
| mojo | 4097 | group_high | base | 0.244 | 8.971 | 0.03 | 25 | 2.73 |
| mojo | 4097 | group_high | nulls | 0.273 | 9.287 | 0.03 | 26 | 2.73 |
| mojo | 4097 | group_high | sorted | 0.228 | 8.960 | 0.03 | 25 | 2.73 |
| mojo | 4097 | group_skew | base | 0.178 | 14.121 | 0.01 | 22 | 2.73 |
| mojo | 4097 | group_skew | nulls | 0.194 | 13.809 | 0.01 | 23 | 2.73 |
| mojo | 4097 | group_skew | sorted | 0.210 | 14.462 | 0.01 | 24 | 2.73 |
| mojo | 1048576 | group_hash | base | 5.226 | 58.263 | 0.09 | 24 | 611.66 |
| mojo | 1048576 | group_hash | nulls | 6.442 | 68.210 | 0.09 | 29 | 611.66 |
| mojo | 1048576 | group_hash | sorted | 5.033 | 56.217 | 0.09 | 24 | 611.66 |
| mojo | 1048576 | group_high | base | 64.529 | 270.564 | 0.24 | 33 | 611.66 |
| mojo | 1048576 | group_high | nulls | 69.484 | 366.595 | 0.19 | 34 | 611.66 |
| mojo | 1048576 | group_high | sorted | 69.593 | 266.344 | 0.26 | 33 | 611.66 |
| mojo | 1048576 | group_skew | base | 8.215 | 185.442 | 0.04 | 26 | 611.66 |
| mojo | 1048576 | group_skew | nulls | 9.256 | 186.064 | 0.05 | 27 | 611.66 |
| mojo | 1048576 | group_skew | sorted | 11.098 | 185.428 | 0.06 | 28 | 611.66 |

## Worst measured variant per case

Lowest CPU/Metal ratio among the selected variants; ratios below one favor CPU. These are per-mechanism results, not an aggregate suite score.

| Runtime | Rows | Workload | Worst variant | CPU/Metal |
|---|---:|---|---|---:|
| baseline | 4097 | group_hash | base | 0.02 |
| baseline | 4097 | group_high | sorted | 0.02 |
| baseline | 4097 | group_skew | base | 0.01 |
| baseline | 1048576 | group_hash | nulls | 0.04 |
| baseline | 1048576 | group_high | sorted | 0.15 |
| baseline | 1048576 | group_skew | base | 0.01 |
| mojo | 4097 | group_hash | base | 0.02 |
| mojo | 4097 | group_high | sorted | 0.03 |
| mojo | 4097 | group_skew | base | 0.01 |
| mojo | 1048576 | group_hash | sorted | 0.09 |
| mojo | 1048576 | group_high | nulls | 0.19 |
| mojo | 1048576 | group_skew | base | 0.04 |

## Measured Metal slowdowns

Cases with CPU/Metal below one, ordered by the largest Metal slowdown. The slowdown factor is Metal time divided by CPU time.

| Runtime | Rows | Workload | Variant | CPU/Metal | Metal slower by |
|---|---:|---|---|---:|---:|
| baseline | 4097 | group_skew | base | 0.01 | 86.21x |
| mojo | 4097 | group_skew | base | 0.01 | 79.33x |
| baseline | 1048576 | group_skew | base | 0.01 | 76.40x |
| baseline | 4097 | group_skew | nulls | 0.01 | 76.29x |
| baseline | 4097 | group_skew | sorted | 0.01 | 71.70x |
| mojo | 4097 | group_skew | nulls | 0.01 | 71.18x |
| baseline | 1048576 | group_skew | sorted | 0.01 | 69.68x |
| mojo | 4097 | group_skew | sorted | 0.01 | 68.87x |
| baseline | 1048576 | group_skew | nulls | 0.02 | 64.27x |
| baseline | 4097 | group_hash | base | 0.02 | 58.84x |
| baseline | 4097 | group_hash | nulls | 0.02 | 57.29x |
| baseline | 4097 | group_hash | sorted | 0.02 | 51.76x |
| mojo | 4097 | group_hash | base | 0.02 | 48.33x |
| mojo | 4097 | group_hash | nulls | 0.02 | 47.02x |
| mojo | 4097 | group_hash | sorted | 0.02 | 46.26x |
| baseline | 4097 | group_high | sorted | 0.02 | 44.73x |
| baseline | 4097 | group_high | base | 0.02 | 42.28x |
| baseline | 4097 | group_high | nulls | 0.03 | 39.66x |
| mojo | 4097 | group_high | sorted | 0.03 | 39.30x |
| mojo | 4097 | group_high | base | 0.03 | 36.77x |
| mojo | 4097 | group_high | nulls | 0.03 | 34.02x |
| baseline | 1048576 | group_hash | nulls | 0.04 | 23.46x |
| mojo | 1048576 | group_skew | base | 0.04 | 22.57x |
| mojo | 1048576 | group_skew | nulls | 0.05 | 20.10x |
| baseline | 1048576 | group_hash | base | 0.05 | 19.60x |
| baseline | 1048576 | group_hash | sorted | 0.05 | 19.27x |
| mojo | 1048576 | group_skew | sorted | 0.06 | 16.71x |
| mojo | 1048576 | group_hash | sorted | 0.09 | 11.17x |
| mojo | 1048576 | group_hash | base | 0.09 | 11.15x |
| mojo | 1048576 | group_hash | nulls | 0.09 | 10.59x |
| baseline | 1048576 | group_high | sorted | 0.15 | 6.58x |
| baseline | 1048576 | group_high | nulls | 0.16 | 6.45x |
| baseline | 1048576 | group_high | base | 0.16 | 6.24x |
| mojo | 1048576 | group_high | nulls | 0.19 | 5.28x |
| mojo | 1048576 | group_high | base | 0.24 | 4.19x |
| mojo | 1048576 | group_high | sorted | 0.26 | 3.83x |

Process outcomes: {'ok': 108, 'unsupported': 0, 'failed': 0}.

## Provenance and limits

```json
{
  "recorded_at_utc": "2026-10-20T15:36:18.886038+00:00",
  "platform": "macOS-26.5.2-arm64-arm-64bit-Mach-O",
  "machine": "arm64",
  "cpu": "Apple M1",
  "physical_cores": "8",
  "memory_bytes": "17179869184",
  "git_commit": "a61e8dd98cf4550b2f4988a6ab09139a00469388",
  "source_tree": {
    "status": "",
    "dataframe_status": "",
    "dataframe_diff_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "dataframe_content_sha256": "5bae65bb1f5d7fe8dac2df79a8e5a5b1767f9d069030d751eb12a0317af72247"
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
    "mojo": "2115571fa1356aeb42f2941920c16c049b406ec2654882be8fba7d0a082281b7",
    "baseline": "fde59b3869eeb50785cc37b8b2e92234f531d320d6cf5b5b42920bc7aa937633"
  },
  "binary_receipts": {
    "mojo": {
      "status": "verified",
      "schema_version": 1,
      "built_at_utc": "2026-10-20T15:36:16.773394+00:00",
      "binary_path": "/Users/randyzwitch/GitHub/dataframe_mojo-metal-oct20-group-schedule-benchmark-v2/build/bench_metal",
      "binary_sha256": "2115571fa1356aeb42f2941920c16c049b406ec2654882be8fba7d0a082281b7",
      "build_command": [
        "mojo",
        "build",
        "-O3",
        "-I",
        ".",
        "benchmarks/bench_metal.mojo",
        "-o",
        "/Users/randyzwitch/GitHub/dataframe_mojo-metal-oct20-group-schedule-benchmark-v2/build/bench_metal"
      ],
      "git_commit": "a61e8dd98cf4550b2f4988a6ab09139a00469388",
      "git_status": "",
      "engine_status": "",
      "engine_content_sha256": "5bae65bb1f5d7fe8dac2df79a8e5a5b1767f9d069030d751eb12a0317af72247",
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
      "built_at_utc": "2026-10-20T14:29:56.219212+00:00",
      "binary_path": "/Users/randyzwitch/GitHub/dataframe_mojo-metal-group-benchmark-v2/build/bench_metal",
      "binary_sha256": "fde59b3869eeb50785cc37b8b2e92234f531d320d6cf5b5b42920bc7aa937633",
      "build_command": [
        "mojo",
        "build",
        "-O3",
        "-I",
        ".",
        "benchmarks/bench_metal.mojo",
        "-o",
        "/Users/randyzwitch/GitHub/dataframe_mojo-metal-group-benchmark-v2/build/bench_metal"
      ],
      "git_commit": "5a07d32fd8c2e7fbcc65ae717a4949dedbe07bad",
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
