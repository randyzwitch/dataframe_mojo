# Paired native Int64/UInt64 SUM benchmark results

Evidence-only summary of the completed paired run in `results.json`. It contains 2 row counts × 4 workloads × 3 variants = 24 case cells, with 3 rounds and 3 timed samples per round. Warm medians below combine the nine timed samples for each runtime/case cell.

The Metal comparison is baseline-binary median divided by candidate-binary median. It measures implementation change within this harness, not GPU versus CPU speedup. The candidate CPU warm median is lower than candidate Metal in 24/24 cells.

Environment: macOS-26.5.2-arm64-arm-64bit-Mach-O; Apple M1, 8 physical cores, 16 GiB RAM; Mojo 1.2.0.dev2026100905 (3b71af75); 8 DataFrame threads; flags `-O3, -I, .`.

Source pins: benchmark/driver contents match commit `3447f6026a4a9d3745d56040ae4b93bc157de477`; baseline engine `a52334e758b7138c547883eb776d744442a1a0db`; candidate engine `52efeb3ab784c8949366870b7cd6b007b3bf77f9`. Benchmark SHA-256 `73fb33f4727100509684cd266785e71d8226d2231bc55cfb2ed8e6967e3d9ea2`; driver SHA-256 `3d28a1663d9a4b682d170e142b03d81500669f55999d999840718dd6bac288a1`; pixi.lock SHA-256 `098d02ba4d166a75640ed6b96aab77fdfe6cb468eded7838f23bba244496e277`. Engine/dataframe source trees were clean; benchmark and driver were benchmark-only working-tree changes whose file contents match the cited source commit. Binary hashes and build commands are retained in the receipts and `results.json`.

Selection: rows `[4097, 1048576]`; workloads `int64_sum, uint64_sum, int64_affine_sum, int64_wide_sum16`; variants `base, sorted, nulls`; 3 rounds × 3 repetitions; 8 threads. Each case ran in a fresh process. Input generation and CPU result validation were untimed; complete result materialization was timed; engine/process order was rotated across rounds.

All 144/144 selected run keys completed and matched the CPU executor. That equality check validates execution but is not an independent oracle. First-use timings include context and shader initialization, not a cold-machine measurement. See `results.md` for per-runtime warm tables and `validation/` for qualification logs.

| Rows | Workload | Variant | Baseline CPU ms | Baseline Metal ms | Candidate CPU ms | Candidate Metal ms | Metal change | Launches baseline/candidate |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 4,097 | int64_sum | base | 0.032 | 7.222 | 0.035 | 1.132 | 6.38× | 4/2 |
| 4,097 | int64_sum | sorted | 0.045 | 7.216 | 0.028 | 1.021 | 7.07× | 4/2 |
| 4,097 | int64_sum | nulls | 0.033 | 7.108 | 0.029 | 1.216 | 5.85× | 4/2 |
| 4,097 | uint64_sum | base | 0.044 | 7.008 | 0.027 | 0.897 | 7.81× | 4/2 |
| 4,097 | uint64_sum | sorted | 0.031 | 7.180 | 0.027 | 1.170 | 6.14× | 4/2 |
| 4,097 | uint64_sum | nulls | 0.046 | 7.231 | 0.026 | 1.333 | 5.42× | 4/2 |
| 4,097 | int64_affine_sum | base | 0.077 | 7.177 | 0.071 | 1.053 | 6.82× | 5/2 |
| 4,097 | int64_affine_sum | sorted | 0.094 | 7.319 | 0.072 | 1.222 | 5.99× | 5/2 |
| 4,097 | int64_affine_sum | nulls | 0.078 | 7.358 | 0.074 | 1.452 | 5.07× | 5/2 |
| 4,097 | int64_wide_sum16 | base | 0.176 | 20.584 | 0.145 | 2.332 | 8.83× | 19/32 |
| 4,097 | int64_wide_sum16 | sorted | 0.163 | 20.658 | 0.145 | 2.317 | 8.92× | 19/32 |
| 4,097 | int64_wide_sum16 | nulls | 0.187 | 20.810 | 0.158 | 2.536 | 8.21× | 19/32 |
| 1,048,576 | int64_sum | base | 1.658 | 19.119 | 0.284 | 3.212 | 5.95× | 4/2 |
| 1,048,576 | int64_sum | sorted | 2.859 | 19.009 | 0.359 | 3.206 | 5.93× | 4/2 |
| 1,048,576 | int64_sum | nulls | 2.271 | 18.499 | 0.328 | 4.687 | 3.95× | 4/2 |
| 1,048,576 | uint64_sum | base | 3.422 | 19.949 | 0.370 | 3.158 | 6.32× | 4/2 |
| 1,048,576 | uint64_sum | sorted | 3.246 | 20.072 | 0.452 | 3.103 | 6.47× | 4/2 |
| 1,048,576 | uint64_sum | nulls | 3.087 | 20.338 | 0.392 | 4.590 | 4.43× | 4/2 |
| 1,048,576 | int64_affine_sum | base | 3.527 | 29.032 | 2.411 | 3.897 | 7.45× | 5/2 |
| 1,048,576 | int64_affine_sum | sorted | 3.621 | 27.832 | 2.349 | 3.446 | 8.08× | 5/2 |
| 1,048,576 | int64_affine_sum | nulls | 3.746 | 26.105 | 2.177 | 5.084 | 5.13× | 5/2 |
| 1,048,576 | int64_wide_sum16 | base | 6.274 | 65.777 | 1.407 | 11.331 | 5.81× | 19/32 |
| 1,048,576 | int64_wide_sum16 | sorted | 4.328 | 67.039 | 1.581 | 11.166 | 6.00× | 19/32 |
| 1,048,576 | int64_wide_sum16 | nulls | 4.316 | 66.194 | 2.431 | 21.578 | 3.07× | 19/32 |

Metal improvement range by workload across all sizes and variants:
- `int64_sum`: 3.95–7.07×
- `uint64_sum`: 4.43–7.81×
- `int64_affine_sum`: 5.07–8.08×
- `int64_wide_sum16`: 3.07–8.92×

Qualification note: the archived prior full-gate log records four failures. Selector expansion, categorical grouped row count and qcut whole-DAG synchronization parity were independently reproduced on baseline. The fourth, an ordered profile/resource-estimate failure, was followed by focused resource-bound fixes whose tests passed; the full gate was not rerun as all-green. Logs and the qualification receipt are retained here.

Limitations: development mechanism measurements only, not external-suite evidence or a general GPU speedup. CPU and Metal share host and power limits; timing variation is possible. Profile data is a separate warm collection and host phase timings may overlap. GPU-only time and pipeline compilation attribution are unavailable unless the runtime provides them.
