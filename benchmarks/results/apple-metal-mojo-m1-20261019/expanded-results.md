# Apple Metal development measurements

Development mechanisms only, not external-suite evidence or a general GPU speedup. Chain32 includes fusion advantages over the current CPU executor.

Times below are medians of complete warm collections, including allocation, staging, synchronization, and CPU result construction. First-use and all individual samples remain in the JSON. Every selected failed or unsupported case remains visible.

| Runtime | Rows | Workload | Variant | CPU ms | Metal ms | CPU/Metal | Launches | Peak payload MiB |
|---|---:|---|---|---:|---:|---:|---:|---:|
| mojo | 4097 | cumulative_sum | base | 0.080 | 17.017 | 0.00 | 18 | 0.80 |
| mojo | 4097 | cumulative_sum | nulls | 0.081 | 16.608 | 0.00 | 18 | 0.80 |
| mojo | 4097 | cumulative_sum | sorted | 0.076 | 17.038 | 0.00 | 18 | 0.80 |
| mojo | 4097 | group_skew | base | 0.198 | 17.005 | 0.01 | 33 | 2.36 |
| mojo | 4097 | group_skew | nulls | 0.222 | 16.808 | 0.01 | 34 | 2.36 |
| mojo | 4097 | group_skew | sorted | 0.229 | 17.700 | 0.01 | 35 | 2.36 |
| mojo | 4097 | rolling_sum | base | 0.350 | 8.267 | 0.04 | 18 | 0.80 |
| mojo | 4097 | rolling_sum | nulls | 0.345 | 8.214 | 0.04 | 18 | 0.80 |
| mojo | 4097 | rolling_sum | sorted | 0.350 | 7.771 | 0.05 | 18 | 0.80 |
| mojo | 65536 | cumulative_sum | base | 1.891 | 156.399 | 0.01 | 21 | 12.82 |
| mojo | 65536 | cumulative_sum | nulls | 2.156 | 151.573 | 0.01 | 21 | 12.82 |
| mojo | 65536 | cumulative_sum | sorted | 2.042 | 156.759 | 0.01 | 21 | 12.82 |
| mojo | 65536 | group_skew | base | 2.921 | 69.417 | 0.04 | 36 | 37.71 |
| mojo | 65536 | group_skew | nulls | 2.530 | 65.433 | 0.04 | 37 | 37.71 |
| mojo | 65536 | group_skew | sorted | 3.060 | 69.275 | 0.04 | 38 | 37.71 |
| mojo | 65536 | rolling_sum | base | 5.022 | 21.017 | 0.24 | 21 | 12.82 |
| mojo | 65536 | rolling_sum | nulls | 4.965 | 21.115 | 0.24 | 21 | 12.82 |
| mojo | 65536 | rolling_sum | sorted | 5.015 | 20.999 | 0.24 | 21 | 12.82 |
| mojo | 1048576 | cumulative_sum | base | 19.507 | 2270.560 | 0.01 | 25 | 205.16 |
| mojo | 1048576 | cumulative_sum | nulls | 20.960 | 2189.409 | 0.01 | 25 | 205.16 |
| mojo | 1048576 | cumulative_sum | sorted | 17.622 | 2263.122 | 0.01 | 25 | 205.16 |
| mojo | 1048576 | group_skew | base | 13.213 | 957.206 | 0.01 | 40 | 603.41 |
| mojo | 1048576 | group_skew | nulls | 12.793 | 931.457 | 0.01 | 41 | 603.41 |
| mojo | 1048576 | group_skew | sorted | 11.248 | 959.221 | 0.01 | 42 | 603.41 |
| mojo | 1048576 | rolling_sum | base | 80.001 | 214.622 | 0.37 | 25 | 205.16 |
| mojo | 1048576 | rolling_sum | nulls | 78.423 | 212.820 | 0.37 | 25 | 205.16 |
| mojo | 1048576 | rolling_sum | sorted | 79.769 | 213.628 | 0.37 | 25 | 205.16 |

## Worst measured variant per case

Lowest CPU/Metal ratio among the selected variants; ratios below one favor CPU. These are per-mechanism results, not an aggregate suite score.

| Runtime | Rows | Workload | Worst variant | CPU/Metal |
|---|---:|---|---|---:|
| mojo | 4097 | cumulative_sum | sorted | 0.00 |
| mojo | 4097 | group_skew | base | 0.01 |
| mojo | 4097 | rolling_sum | nulls | 0.04 |
| mojo | 65536 | cumulative_sum | base | 0.01 |
| mojo | 65536 | group_skew | nulls | 0.04 |
| mojo | 65536 | rolling_sum | nulls | 0.24 |
| mojo | 1048576 | cumulative_sum | sorted | 0.01 |
| mojo | 1048576 | group_skew | sorted | 0.01 |
| mojo | 1048576 | rolling_sum | nulls | 0.37 |

## Measured Metal slowdowns

Cases with CPU/Metal below one, ordered by the largest Metal slowdown. The slowdown factor is Metal time divided by CPU time.

| Runtime | Rows | Workload | Variant | CPU/Metal | Metal slower by |
|---|---:|---|---|---:|---:|
| mojo | 4097 | cumulative_sum | sorted | 0.00 | 222.73x |
| mojo | 4097 | cumulative_sum | base | 0.00 | 212.71x |
| mojo | 4097 | cumulative_sum | nulls | 0.00 | 205.04x |
| mojo | 1048576 | cumulative_sum | sorted | 0.01 | 128.43x |
| mojo | 1048576 | cumulative_sum | base | 0.01 | 116.40x |
| mojo | 1048576 | cumulative_sum | nulls | 0.01 | 104.45x |
| mojo | 4097 | group_skew | base | 0.01 | 86.10x |
| mojo | 1048576 | group_skew | sorted | 0.01 | 85.28x |
| mojo | 65536 | cumulative_sum | base | 0.01 | 82.71x |
| mojo | 4097 | group_skew | sorted | 0.01 | 77.29x |
| mojo | 65536 | cumulative_sum | sorted | 0.01 | 76.77x |
| mojo | 4097 | group_skew | nulls | 0.01 | 75.71x |
| mojo | 1048576 | group_skew | nulls | 0.01 | 72.81x |
| mojo | 1048576 | group_skew | base | 0.01 | 72.44x |
| mojo | 65536 | cumulative_sum | nulls | 0.01 | 70.29x |
| mojo | 65536 | group_skew | nulls | 0.04 | 25.86x |
| mojo | 4097 | rolling_sum | nulls | 0.04 | 23.81x |
| mojo | 65536 | group_skew | base | 0.04 | 23.77x |
| mojo | 4097 | rolling_sum | base | 0.04 | 23.62x |
| mojo | 65536 | group_skew | sorted | 0.04 | 22.64x |
| mojo | 4097 | rolling_sum | sorted | 0.05 | 22.20x |
| mojo | 65536 | rolling_sum | nulls | 0.24 | 4.25x |
| mojo | 65536 | rolling_sum | sorted | 0.24 | 4.19x |
| mojo | 65536 | rolling_sum | base | 0.24 | 4.18x |
| mojo | 1048576 | rolling_sum | nulls | 0.37 | 2.71x |
| mojo | 1048576 | rolling_sum | base | 0.37 | 2.68x |
| mojo | 1048576 | rolling_sum | sorted | 0.37 | 2.68x |

Process outcomes: {'ok': 54, 'unsupported': 0, 'failed': 0}.

## Provenance and limits

```json
{
  "recorded_at_utc": "2026-10-19T20:23:06.885365+00:00",
  "platform": "macOS-26.5.2-arm64-arm-64bit-Mach-O",
  "machine": "arm64",
  "cpu": "Apple M1",
  "physical_cores": "8",
  "memory_bytes": "17179869184",
  "git_commit": "aa00dc9f53dbf15180128a3e1a4c9c3c25278110",
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
  "implementation": "mojo",
  "binaries_sha256": {
    "mojo": "f1fdf113a03c4a8bb4acbf76c17eff192d388810873a48af26a09284c805d122"
  },
  "source_sha256": "cb8a3f062121a6661b560a750ad85ee005291b3a77e0e932c37e2e883e8cad51",
  "driver_sha256": "1bec3292f544f318aa3382051ddd90606056ce1c1545d41ad437772a85103606",
  "environment": {},
  "method": "Fresh process per case/build/round; first-use collections retained separately; two extra warmups per engine; alternating CPU/Metal timed order; process/build order rotated per round; inputs and exact CPU-reference validation untimed; complete result materialization timed; result release untimed.",
  "scope": "Development mechanisms only, not external-suite evidence or a general GPU speedup. Chain32 includes fusion advantages over the current CPU executor.",
  "data": "Row mechanisms use a deterministic 1024-value numeric distribution. Algorithm cases use Int32 keys: group_hash and group_skew have min(N,1024) classes; group_high, join_hash and sort_radix have max(1,N/2) classes. Base keys are (37*i) modulo cardinality; sorted keys are monotone bins. group_skew maps three quarters of rows to key zero. Null variants invalidate every twentieth key; grouped values additionally invalidate every thirteenth row. Grouped sums use i modulo 17 minus 8; sort and self-left-join carry original Int64 row IDs. Float32 row values are exact multiples of 1/1024.",
  "limitations": [
    "No external-suite coverage claimed; Float64 and floating reductions are outside these cases. Algorithm comparisons preserve complete output order and values against CPU.",
    "First use includes context/shader initialization but is not a cold machine or cold filesystem measurement.",
    "CPU and GPU share host memory and power limits; short timings can vary with scheduler and thermal state.",
    "The Mojo profile is a separate warm collection after timed samples; complete caller-wall timing remains the performance measure. Reported host phase timings are retained as measured and may overlap; they are not added. GPU-only time and pipeline compilation attribution are unavailable unless the runtime supplies those fields.",
    "Workload exposure: chain32 was used during fusion development; other cases broaden mechanism coverage without a holdout claim."
  ]
}
```
