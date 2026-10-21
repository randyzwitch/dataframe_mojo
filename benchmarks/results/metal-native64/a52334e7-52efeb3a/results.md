# Apple Metal development measurements

Development mechanisms only, not external-suite evidence or a general GPU speedup. This run measures native Int64 and UInt64 SUM cases.

Times below are medians of complete warm collections, including allocation, staging, synchronization, and CPU result construction. First-use and all individual samples remain in the JSON. Every selected failed or unsupported case remains visible.

| Runtime | Rows | Workload | Variant | CPU ms | Metal ms | CPU/Metal | Launches | Peak payload MiB |
|---|---:|---|---|---:|---:|---:|---:|---:|
| baseline | 4097 | int64_affine_sum | base | 0.077 | 7.177 | 0.01 | 5 | 0.33 |
| baseline | 4097 | int64_affine_sum | nulls | 0.078 | 7.358 | 0.01 | 5 | 0.33 |
| baseline | 4097 | int64_affine_sum | sorted | 0.094 | 7.319 | 0.01 | 5 | 0.33 |
| baseline | 4097 | int64_sum | base | 0.032 | 7.222 | 0.00 | 4 | 0.33 |
| baseline | 4097 | int64_sum | nulls | 0.033 | 7.108 | 0.00 | 4 | 0.33 |
| baseline | 4097 | int64_sum | sorted | 0.045 | 7.216 | 0.01 | 4 | 0.33 |
| baseline | 4097 | int64_wide_sum16 | base | 0.176 | 20.584 | 0.01 | 19 | 2.81 |
| baseline | 4097 | int64_wide_sum16 | nulls | 0.187 | 20.810 | 0.01 | 19 | 2.81 |
| baseline | 4097 | int64_wide_sum16 | sorted | 0.163 | 20.658 | 0.01 | 19 | 2.81 |
| baseline | 4097 | uint64_sum | base | 0.044 | 7.008 | 0.01 | 4 | 0.33 |
| baseline | 4097 | uint64_sum | nulls | 0.046 | 7.231 | 0.01 | 4 | 0.33 |
| baseline | 4097 | uint64_sum | sorted | 0.031 | 7.180 | 0.00 | 4 | 0.33 |
| baseline | 1048576 | int64_affine_sum | base | 3.527 | 29.032 | 0.12 | 5 | 84.06 |
| baseline | 1048576 | int64_affine_sum | nulls | 3.746 | 26.105 | 0.14 | 5 | 84.06 |
| baseline | 1048576 | int64_affine_sum | sorted | 3.621 | 27.832 | 0.13 | 5 | 84.06 |
| baseline | 1048576 | int64_sum | base | 1.658 | 19.119 | 0.09 | 4 | 84.06 |
| baseline | 1048576 | int64_sum | nulls | 2.271 | 18.499 | 0.12 | 4 | 84.06 |
| baseline | 1048576 | int64_sum | sorted | 2.859 | 19.009 | 0.15 | 4 | 84.06 |
| baseline | 1048576 | int64_wide_sum16 | base | 6.274 | 65.777 | 0.10 | 19 | 715.00 |
| baseline | 1048576 | int64_wide_sum16 | nulls | 4.316 | 66.194 | 0.07 | 19 | 715.00 |
| baseline | 1048576 | int64_wide_sum16 | sorted | 4.328 | 67.039 | 0.06 | 19 | 715.00 |
| baseline | 1048576 | uint64_sum | base | 3.422 | 19.949 | 0.17 | 4 | 84.06 |
| baseline | 1048576 | uint64_sum | nulls | 3.087 | 20.338 | 0.15 | 4 | 84.06 |
| baseline | 1048576 | uint64_sum | sorted | 3.246 | 20.072 | 0.16 | 4 | 84.06 |
| mojo | 4097 | int64_affine_sum | base | 0.071 | 1.053 | 0.07 | 2 | 0.03 |
| mojo | 4097 | int64_affine_sum | nulls | 0.074 | 1.452 | 0.05 | 2 | 0.03 |
| mojo | 4097 | int64_affine_sum | sorted | 0.072 | 1.222 | 0.06 | 2 | 0.03 |
| mojo | 4097 | int64_sum | base | 0.035 | 1.132 | 0.03 | 2 | 0.03 |
| mojo | 4097 | int64_sum | nulls | 0.029 | 1.216 | 0.02 | 2 | 0.03 |
| mojo | 4097 | int64_sum | sorted | 0.028 | 1.021 | 0.03 | 2 | 0.03 |
| mojo | 4097 | int64_wide_sum16 | base | 0.145 | 2.332 | 0.06 | 32 | 0.03 |
| mojo | 4097 | int64_wide_sum16 | nulls | 0.158 | 2.536 | 0.06 | 32 | 0.03 |
| mojo | 4097 | int64_wide_sum16 | sorted | 0.145 | 2.317 | 0.06 | 32 | 0.03 |
| mojo | 4097 | uint64_sum | base | 0.027 | 0.897 | 0.03 | 2 | 0.03 |
| mojo | 4097 | uint64_sum | nulls | 0.026 | 1.333 | 0.02 | 2 | 0.03 |
| mojo | 4097 | uint64_sum | sorted | 0.027 | 1.170 | 0.02 | 2 | 0.03 |
| mojo | 1048576 | int64_affine_sum | base | 2.411 | 3.897 | 0.62 | 2 | 8.09 |
| mojo | 1048576 | int64_affine_sum | nulls | 2.177 | 5.084 | 0.43 | 2 | 8.22 |
| mojo | 1048576 | int64_affine_sum | sorted | 2.349 | 3.446 | 0.68 | 2 | 8.09 |
| mojo | 1048576 | int64_sum | base | 0.284 | 3.212 | 0.09 | 2 | 8.09 |
| mojo | 1048576 | int64_sum | nulls | 0.328 | 4.687 | 0.07 | 2 | 8.22 |
| mojo | 1048576 | int64_sum | sorted | 0.359 | 3.206 | 0.11 | 2 | 8.09 |
| mojo | 1048576 | int64_wide_sum16 | base | 1.407 | 11.331 | 0.12 | 32 | 8.09 |
| mojo | 1048576 | int64_wide_sum16 | nulls | 2.431 | 21.578 | 0.11 | 32 | 8.22 |
| mojo | 1048576 | int64_wide_sum16 | sorted | 1.581 | 11.166 | 0.14 | 32 | 8.09 |
| mojo | 1048576 | uint64_sum | base | 0.370 | 3.158 | 0.12 | 2 | 8.09 |
| mojo | 1048576 | uint64_sum | nulls | 0.392 | 4.590 | 0.09 | 2 | 8.22 |
| mojo | 1048576 | uint64_sum | sorted | 0.452 | 3.103 | 0.15 | 2 | 8.09 |

## Worst measured variant per case

Lowest CPU/Metal ratio among the selected variants; ratios below one favor CPU. These are per-mechanism results, not an aggregate suite score.

| Runtime | Rows | Workload | Worst variant | CPU/Metal |
|---|---:|---|---|---:|
| baseline | 4097 | int64_affine_sum | nulls | 0.01 |
| baseline | 4097 | int64_sum | base | 0.00 |
| baseline | 4097 | int64_wide_sum16 | sorted | 0.01 |
| baseline | 4097 | uint64_sum | sorted | 0.00 |
| baseline | 1048576 | int64_affine_sum | base | 0.12 |
| baseline | 1048576 | int64_sum | base | 0.09 |
| baseline | 1048576 | int64_wide_sum16 | sorted | 0.06 |
| baseline | 1048576 | uint64_sum | nulls | 0.15 |
| mojo | 4097 | int64_affine_sum | nulls | 0.05 |
| mojo | 4097 | int64_sum | nulls | 0.02 |
| mojo | 4097 | int64_wide_sum16 | base | 0.06 |
| mojo | 4097 | uint64_sum | nulls | 0.02 |
| mojo | 1048576 | int64_affine_sum | nulls | 0.43 |
| mojo | 1048576 | int64_sum | nulls | 0.07 |
| mojo | 1048576 | int64_wide_sum16 | nulls | 0.11 |
| mojo | 1048576 | uint64_sum | nulls | 0.09 |

## Measured Metal slowdowns

Cases with CPU/Metal below one, ordered by the largest Metal slowdown. The slowdown factor is Metal time divided by CPU time.

| Runtime | Rows | Workload | Variant | CPU/Metal | Metal slower by |
|---|---:|---|---|---:|---:|
| baseline | 4097 | uint64_sum | sorted | 0.00 | 231.61x |
| baseline | 4097 | int64_sum | base | 0.00 | 225.69x |
| baseline | 4097 | int64_sum | nulls | 0.00 | 215.39x |
| baseline | 4097 | int64_sum | sorted | 0.01 | 160.36x |
| baseline | 4097 | uint64_sum | base | 0.01 | 159.27x |
| baseline | 4097 | uint64_sum | nulls | 0.01 | 157.20x |
| baseline | 4097 | int64_wide_sum16 | sorted | 0.01 | 126.74x |
| baseline | 4097 | int64_wide_sum16 | base | 0.01 | 116.95x |
| baseline | 4097 | int64_wide_sum16 | nulls | 0.01 | 111.28x |
| baseline | 4097 | int64_affine_sum | nulls | 0.01 | 94.33x |
| baseline | 4097 | int64_affine_sum | base | 0.01 | 93.21x |
| baseline | 4097 | int64_affine_sum | sorted | 0.01 | 77.86x |
| mojo | 4097 | uint64_sum | nulls | 0.02 | 51.27x |
| mojo | 4097 | uint64_sum | sorted | 0.02 | 43.33x |
| mojo | 4097 | int64_sum | nulls | 0.02 | 41.93x |
| mojo | 4097 | int64_sum | sorted | 0.03 | 36.46x |
| mojo | 4097 | uint64_sum | base | 0.03 | 33.22x |
| mojo | 4097 | int64_sum | base | 0.03 | 32.34x |
| mojo | 4097 | int64_affine_sum | nulls | 0.05 | 19.62x |
| mojo | 4097 | int64_affine_sum | sorted | 0.06 | 16.97x |
| mojo | 4097 | int64_wide_sum16 | base | 0.06 | 16.08x |
| mojo | 4097 | int64_wide_sum16 | nulls | 0.06 | 16.05x |
| mojo | 4097 | int64_wide_sum16 | sorted | 0.06 | 15.98x |
| baseline | 1048576 | int64_wide_sum16 | sorted | 0.06 | 15.49x |
| baseline | 1048576 | int64_wide_sum16 | nulls | 0.07 | 15.34x |
| mojo | 4097 | int64_affine_sum | base | 0.07 | 14.83x |
| mojo | 1048576 | int64_sum | nulls | 0.07 | 14.29x |
| mojo | 1048576 | uint64_sum | nulls | 0.09 | 11.71x |
| baseline | 1048576 | int64_sum | base | 0.09 | 11.53x |
| mojo | 1048576 | int64_sum | base | 0.09 | 11.31x |
| baseline | 1048576 | int64_wide_sum16 | base | 0.10 | 10.48x |
| mojo | 1048576 | int64_sum | sorted | 0.11 | 8.93x |
| mojo | 1048576 | int64_wide_sum16 | nulls | 0.11 | 8.88x |
| mojo | 1048576 | uint64_sum | base | 0.12 | 8.54x |
| baseline | 1048576 | int64_affine_sum | base | 0.12 | 8.23x |
| baseline | 1048576 | int64_sum | nulls | 0.12 | 8.15x |
| mojo | 1048576 | int64_wide_sum16 | base | 0.12 | 8.05x |
| baseline | 1048576 | int64_affine_sum | sorted | 0.13 | 7.69x |
| mojo | 1048576 | int64_wide_sum16 | sorted | 0.14 | 7.06x |
| baseline | 1048576 | int64_affine_sum | nulls | 0.14 | 6.97x |
| mojo | 1048576 | uint64_sum | sorted | 0.15 | 6.87x |
| baseline | 1048576 | int64_sum | sorted | 0.15 | 6.65x |
| baseline | 1048576 | uint64_sum | nulls | 0.15 | 6.59x |
| baseline | 1048576 | uint64_sum | sorted | 0.16 | 6.18x |
| baseline | 1048576 | uint64_sum | base | 0.17 | 5.83x |
| mojo | 1048576 | int64_affine_sum | nulls | 0.43 | 2.34x |
| mojo | 1048576 | int64_affine_sum | base | 0.62 | 1.62x |
| mojo | 1048576 | int64_affine_sum | sorted | 0.68 | 1.47x |

Process outcomes: {'ok': 144, 'unsupported': 0, 'failed': 0}.

## Provenance and limits

```json
{
  "recorded_at_utc": "2026-10-20T01:20:04.822793+00:00",
  "platform": "macOS-26.5.2-arm64-arm-64bit-Mach-O",
  "machine": "arm64",
  "cpu": "Apple M1",
  "physical_cores": "8",
  "memory_bytes": "17179869184",
  "git_commit": "52efeb3ab784c8949366870b7cd6b007b3bf77f9",
  "source_tree": {
    "status": "M benchmarks/bench_metal.mojo\n M benchmarks/tools/bench_metal.py",
    "dataframe_status": "",
    "dataframe_diff_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "dataframe_content_sha256": "6e229c2833237e84762f4c972fab671f9d92c81b4413204e5ffc63a1792f1f45"
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
  "warmups_per_engine": 3,
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
    "mojo": "e6dbfa9d4aaf291c286fedb9f9f6ab195ca914426270f57529f8d9ab338ba957",
    "baseline": "1f75cc1a626130a5ec430a2ac7fa7ac20df2b35ed285350e2b0794345e2b2ca1"
  },
  "binary_receipts": {
    "mojo": {
      "status": "verified",
      "schema_version": 1,
      "built_at_utc": "2026-10-20T01:20:04.584761+00:00",
      "binary_path": "/Users/randyzwitch/github/dataframe_mojo-metal-868/build/metal-native64-paired-fixed-52efeb3a/worktrees/candidate/build/bench_metal",
      "binary_sha256": "e6dbfa9d4aaf291c286fedb9f9f6ab195ca914426270f57529f8d9ab338ba957",
      "build_command": [
        "mojo",
        "build",
        "-O3",
        "-I",
        ".",
        "benchmarks/bench_metal.mojo",
        "-o",
        "/Users/randyzwitch/github/dataframe_mojo-metal-868/build/metal-native64-paired-fixed-52efeb3a/worktrees/candidate/build/bench_metal"
      ],
      "git_commit": "52efeb3ab784c8949366870b7cd6b007b3bf77f9",
      "git_status": "M benchmarks/bench_metal.mojo\n M benchmarks/tools/bench_metal.py",
      "engine_status": "",
      "engine_content_sha256": "6e229c2833237e84762f4c972fab671f9d92c81b4413204e5ffc63a1792f1f45",
      "engine_diff_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      "compiler_identity": "Mojo 1.2.0.dev2026100905 (3b71af75)",
      "build_platform": "macOS-26.5.2-arm64-arm-64bit-Mach-O",
      "build_sdk": "26.5",
      "build_xcode": "Xcode 26.6\nBuild version 17F113",
      "compiler_executable": "/Users/randyzwitch/github/dataframe_mojo-metal-868/.pixi/envs/oracle/bin/mojo",
      "compiler_executable_sha256": "ac42575fdf41cd8ced537812c20cbdda361223434ba90c866c6c42007df1a37b",
      "pixi_lock_sha256": "098d02ba4d166a75640ed6b96aab77fdfe6cb468eded7838f23bba244496e277",
      "benchmark_source_sha256": "73fb33f4727100509684cd266785e71d8226d2231bc55cfb2ed8e6967e3d9ea2",
      "driver_source_sha256": "3d28a1663d9a4b682d170e142b03d81500669f55999d999840718dd6bac288a1",
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
      "built_at_utc": "2026-10-20T01:14:18.032711+00:00",
      "binary_path": "/Users/randyzwitch/github/dataframe_mojo-metal-868/build/metal-native64-paired-fixed-52efeb3a/worktrees/baseline/build/bench_metal",
      "binary_sha256": "1f75cc1a626130a5ec430a2ac7fa7ac20df2b35ed285350e2b0794345e2b2ca1",
      "build_command": [
        "mojo",
        "build",
        "-O3",
        "-I",
        ".",
        "benchmarks/bench_metal.mojo",
        "-o",
        "/Users/randyzwitch/github/dataframe_mojo-metal-868/build/metal-native64-paired-fixed-52efeb3a/worktrees/baseline/build/bench_metal"
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
      "benchmark_source_sha256": "73fb33f4727100509684cd266785e71d8226d2231bc55cfb2ed8e6967e3d9ea2",
      "driver_source_sha256": "3d28a1663d9a4b682d170e142b03d81500669f55999d999840718dd6bac288a1",
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
  "source_sha256": "73fb33f4727100509684cd266785e71d8226d2231bc55cfb2ed8e6967e3d9ea2",
  "driver_sha256": "3d28a1663d9a4b682d170e142b03d81500669f55999d999840718dd6bac288a1",
  "environment": {},
  "method": "Fresh process per case/build/round; first-use collections retained separately; configured extra warmups per engine; alternating CPU/Metal timed order; process/build order rotated per round; inputs and exact CPU-reference validation untimed; complete result materialization timed; result release untimed.",
  "scope": "Development mechanisms only, not external-suite evidence or a general GPU speedup. Chain32 includes fusion advantages over the current CPU executor.",
  "data": "Row mechanisms use a deterministic 1024-value numeric distribution. Algorithm cases use Int32 keys: group_hash and group_skew have min(N,1024) classes; group_high, join_hash and sort_radix have max(1,N/2) classes; join_fanout uses ceil(N/4) cyclic or monotone balanced classes, so each key occurs at most four times and a self-left-join emits at most 4N rows. Base non-fanout keys are (37*i) modulo cardinality; sorted keys are monotone bins. group_skew maps three quarters of rows to key zero. Null variants invalidate every twentieth key; grouped values additionally invalidate every thirteenth row. Grouped sums use i modulo 17 minus 8; sort and self-left-join carry original Int64 row IDs. Float32 row values are exact multiples of 1/1024. Float64 arithmetic uses exact binary fractions k/1024 with k in [0,1023] and evaluates x*2+0.25. Decimal input is Decimal(15,2) cents with unscaled values in [-8,8]; the query multiplies by Decimal(15,2) 1.25 then sums. Wide Float32 projects eight columns with binary-exact offsets [0,0.125,...,0.875], at most 8N output values or 256 MiB at the default 8,388,608 rows. Optional int32_affine_sum and filtered_int32_affine_sum workloads use the same Int32 base/sorted/nulls input patterns as int_sum, compute sum(cast_Int64((x+3)*2)), and optionally filter x>0 before the expression. Optional native64 workloads use typed Int64 values i%17-8 and typed UInt64 values i%17; sorted variants use monotone bins over the same ranges, and null variants use an LCG-distributed value with every twentieth row null. int64_affine_sum sums (x+3)*2, whose per-row magnitude is at most 22; int64_wide_sum16 emits 16 differently aliased sums of the same Int64 input. The harness validates every output against its exact CPU result, which is not an independent oracle. All these workloads are explicitly selectable with --workloads and excluded from the default set.",
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
