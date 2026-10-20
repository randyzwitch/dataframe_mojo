# Grouping schedule correction: three-reference measurements

Candidate engine `752eb23322ef9d18e7f2d6b950f733d5e0d8a340` fixes the two large non-null uniform-group regressions observed in the earlier `0a9297c3` candidate. It uses observed hash-group size to avoid partial-reduction setup when no group reaches 4,096 rows. A qualifying large group retains the partial path even when average group size is small. This is data-based admission, not workload-name dispatch.

The corrected ten-call hardware gate passes. All **333 primary workers** validate: nine smoke workers and three 108-worker comparisons across 18 cases each. The cases cover 4,097 and 1,048,576 rows, uniform/high-cardinality/skewed groups, base/sorted/null variants, three paired rounds, one warmup and three timed samples per worker. An additional **20-worker follow-up** retains a longer check of one small-input concern. CPU remains faster in every case.

| Reference | Warm complete-query result |
| --- | --- |
| Frozen `a52334e7` | All 18 cases improve, from 1.184× to 5.180×. See [frozen summary](frozen-summary.md). |
| Intermediate #865 `158285f7` | All 18 cases improve. The two previously regressing large non-null uniform cases now improve 1.100× and 1.070×, with every paired round improving. Large skewed cases retain 4.810–4.903× gains. High-cardinality gains are small (1.005–1.015×). See [intermediate summary](intermediate-summary.md). |
| Previous #902 `0a9297c3` | The targeted uniform cases improve 1.152× and 1.139×, with every paired round improving; launches fall from 28 to 24. Other warm medians range from 0.983× to 1.065×, including slower observations. Requested peak allocation is unchanged in every case. See [previous summary](previous-summary.md). |

These are three independently paired runs, not ratios assembled from separate historical medians. Reused reference binaries are pinned by source revision, engine-content fingerprint and binary SHA. All use the same benchmark sources, compiler, lockfile, SDK and flags. The candidate measurement revision is `a61e8dd98cf4550b2f4988a6ab09139a00469388`, directly descended from the qualified engine revision. The frozen/intermediate comparisons measure whole stacks; the previous-candidate comparison isolates the scheduling/profile changes.

## Small-input follow-up and remaining limitations

The original small sorted-skew comparison was 1.7% slower (14.565 ms versus 14.313 ms), with all three round ratios below one. A predeclared follow-up used the same binaries for ten paired rounds, one warmup and five samples per worker. Its median was 1.1% faster (14.495 ms versus 14.661 ms), with round ratios from 0.927 to 1.121. Both results remain archived; the follow-up does not erase the original slower observation or establish a reliable improvement in this case. See [follow-up summary](skew-followup-summary.md) and its separately retained plan, driver and raw logs.

First-use results are mixed: the targeted large uniform cases improve against the previous candidate, but other first-use medians include slower observations. Every value remains in the tables. First use is a fresh query process, not a cold machine. Requested peak allocations remain above the frozen/intermediate implementations; they are allocation accounting, not measured device residency. The earlier archive remains available with its original regression evidence.

This is a narrow integer-grouping correction. Primitive timing, hash fallback-frequency measurements, dense/general typed partitions, dictionary generalization, remaining serial/order-sensitive states and broad replacement coverage are still outstanding under #873/#877. These results do not justify retiring legacy paths or enabling automatic CPU/Metal placement (#583).

## Validation and retained failure

The first schedule revision `e610ca2c` built but failed its hardware gate because the new profile counter was missing from heterogeneous DAG report defaults. Its receipt records `build_exit=0`, `test_exit=1`; those logs remain under `validation/failed/`. Revision `752eb233` adds the missing default and passes the same gate, including composed queries, threshold boundaries, low-average skew, nullable groups, budget limits and retained fallback behavior. The failed result is not relabeled as a pass.

Run `python3 validate_results.py .` from this directory to replay all 353 workers, verify matrix coverage and actual candidate routes, check build attribution and shared binary identities, and validate passing and failed gate log hashes. `qualification.json` ties the execution driver to the passed receipt. Gate executable hashes are retained, but executables are not included. The candidate source fingerprint is independently pinned from the qualified local source; the source tree itself is not duplicated here.

Run `sha256sum -c SHA256SUMS.txt` for archive integrity. Each report is reproducible with `python3 summarize_results.py <pair-directory>/results.json <summary-prefix>`. Raw compiler output and machine-local paths are preserved. Per-round ranges describe observed variation, not confidence intervals.
