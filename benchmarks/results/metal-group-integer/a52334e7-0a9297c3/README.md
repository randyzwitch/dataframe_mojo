# Grouped integer replacement measurements

The tested #902 candidate improves all 18 warm cases against the frozen Metal baseline, but regresses two large uniform cases against the #865 intermediate. These results support retaining the large-skew improvement while revising admission before treating this implementation as ready.

| Comparison | 4,097 rows | 1,048,576 rows |
| --- | --- | --- |
| Frozen baseline / candidate | 1.19–1.43× | 1.33–5.22× |
| #865 intermediate / candidate | 1.15–1.50× | 0.93–4.89× |

Ratios above 1 mean the candidate was faster. Against #865, the non-null uniform cases take 67.0/66.8 ms versus 62.8/62.3 ms: approximately 6.7%/7.2% slower, consistently across all three rounds. The nullable uniform case improves 1.89×, and skewed cases improve 4.75–4.89×. High-cardinality cases differ by only 1.5–2.1%; these are not a large gain. Candidate CPU is faster than candidate Metal in every measured case.

Reported peak requested bytes increase from 2,473,062 to 2,866,278 at 4,097 rows and from 632,718,208 to 641,368,960 at 1,048,576 rows versus the frozen baseline. Against #865, the increase is 262,144 bytes at either size. These are profile allocation-accounting values, not measured physical GPU residency. First-use medians also include small skewed-case regressions; the full tables retain them.

- [Frozen-baseline paired results](baseline-summary.md)
- [Intermediate paired results](intermediate-summary.md): its “Baseline” column is the #865 intermediate, not the frozen implementation.
- `baseline-pair/` and `intermediate-pair/` contain 108 workers each, their raw logs and original driver reports.
- `baseline-smoke/` and `intermediate-smoke/` retain nine validated workers each. All 234 selected workers completed and matched CPU results.

## Identity and method

The engine revisions are frozen baseline `a52334e758b7138c547883eb776d744442a1a0db`, #865 intermediate `158285f7cf225276493efd354454a5d278050fef`, and #902 candidate `0a9297c3f15cae9984ddf5eb9f261df75ada161b`. Benchmark-only measurement commits and engine content hashes are recorded separately in `attribution.json`. No engine source was modified to run the shared harness. `harness/` retains the exact benchmark and driver bytes; each result contains source/compiler/SDK/Xcode/build-flag identities and binary receipts.

Hardware: Apple M1, 16 GiB; Mojo `1.2.0.dev2026100905 (3b71af75)`; macOS SDK 26.5; Xcode 26.6 (17F113); `-O3 -I .`; eight DataFrame threads. The Mac ran the comparisons serially. Each pair uses three rounds and three timed samples per worker, after one warmup. Engine and case order rotate between rounds. The displayed warm median combines nine samples per case and runtime; the round-ratio range is descriptive variation, not a confidence interval. First use includes query-process runtime initialization, not a cold-machine measurement.

Timed collections include allocation, host staging, device execution/synchronization and result construction. Input generation and CPU result comparison are outside timings. CPU matching is an execution check, not an independent oracle. The separately retained nine-call Mac correctness gate at the candidate revision covers independent expectations, ordering, exact wide totals, overflow, budgets, collision behavior and recovery; it is not a claim that the full suite is green.

`group_hash` uses up to 1,024 key classes, `group_high` uses roughly N/2 classes, and `group_skew` maps three quarters of rows to key zero. Base, sorted and nullable variants have deterministic keys and values; null variants invalidate every twentieth key and thirteenth value. The exact construction is in the archived harness. This is development-mechanism evidence, not full-suite or full-scale coverage.

## Routes and remaining work

All 18 large uniform/skewed candidate workers per comparison use `mojo_gpu_grouped_int_partials`; the other 36 candidate workers use the existing reducer. Profiles retain partial geometry and scratch counters. “Legacy” here describes reducer admission, not proof of hash-partition probe exhaustion. This harness does not expose hash-probe fallback frequency, so that acceptance item remains unproven.

The improvement over the frozen baseline applies to the candidate stack. The comparison with #865 exposes the uniform-group regression; neither comparison alone isolates every commit's contribution. Scheduling will be revised using actual group-size evidence, with these original samples retained for comparison. No retirement of the remaining grouping paths, closure of #873/#877, or CPU/Metal placement policy is implied. #583 remains deferred.

Run `python3 validate_results.py` from any directory to verify all workers, raw outputs, source/build identities and candidate reducer routes. `SHA256SUMS.txt` records archive file integrity. `validation/` contains the passed focused-gate receipt and logs, captured after testing; executables are represented by their recorded hashes rather than included in this archive.
