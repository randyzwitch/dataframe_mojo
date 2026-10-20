# Remaining Metal performance gap to CPU

These complete-query warm medians describe the current measured candidates, not a competitive-backend claim. CPU wins every case below. The factor is Metal time divided by CPU time; lower is better and parity is 1×. Improvements over an older Metal implementation do not establish competitiveness.

| Family | Rows | CPU ms range | Metal ms range | Metal / CPU range |
| --- | ---: | ---: | ---: | ---: |
| Float32 affine/filter sums | 4,097 | 0.081–0.102 | 1.026–1.460 | **12.4–15.3× slower** |
| Float32 affine/filter sums | 1,048,576 | 1.766–3.022 | 5.138–7.986 | **2.1–3.5× slower** |
| Native Int32 self-left joins | 4,097 | 0.284–0.363 | 7.390–8.473 | **23.2–27.3× slower** |
| Native Int32 self-left joins | 262,144 | 8.389–12.546 | 67.616–109.285 | **7.4–11.6× slower** |
| Grouped integer reductions | 4,097 | 0.177–0.265 | 8.506–14.565 | **34.9–78.5× slower** |
| Grouped integer reductions | 1,048,576 | 4.936–81.071 | 56.623–368.484 | **3.3–24.7× slower** |

Ranges summarize the measured variants; the fastest CPU and Metal endpoints need not describe the same case. Each slowdown factor is computed from matched per-case medians. These are narrow mechanisms, not the entire development suite, and should not replace the separately archived baseline failures and exclusions.

## Largest measured gaps in these candidates

| Workload | Rows | Variant | CPU ms | Metal ms | Metal / CPU |
| --- | ---: | --- | ---: | ---: | ---: |
| group_skew | 4,097 | base | 0.181 | 14.214 | 78.5× |
| group_skew | 4,097 | nulls | 0.197 | 13.361 | 67.8× |
| group_skew | 4,097 | sorted | 0.223 | 14.565 | 65.3× |
| group_hash | 4,097 | base | 0.177 | 8.506 | 48.1× |
| group_hash | 4,097 | nulls | 0.193 | 8.965 | 46.5× |
| group_hash | 4,097 | sorted | 0.185 | 8.567 | 46.3× |
| group_high | 4,097 | sorted | 0.229 | 8.715 | 38.1× |
| group_high | 4,097 | base | 0.241 | 8.997 | 37.3× |
| group_high | 4,097 | nulls | 0.265 | 9.256 | 34.9× |
| join_hash | 4,097 | nulls | 0.294 | 8.035 | 27.3× |

## Interpretation and next measurements

The measured small-query gap reaches 78.5×. Incremental Metal improvements leave this unresolved. Small-workload overhead and the larger cases must be assessed separately; a faster kernel alone cannot establish a competitive complete query.

Existing profile collections point to final readback/export as a major phase in the large native join cases (approximately 56–67% of profiled wall time). The phase includes more than device transfer, so its contents require source-level attribution before assigning a cause or forecasting gains. Even eliminating that entire phase would yield only an approximately 2.3–3× ceiling for those profiled cases. Those profile-derived ceilings are not achieved speedups and must not be extrapolated to the worst CPU gaps.

Priority is to identify measured structural costs—submission/waits, repeated staging, generic storage, result export and redundant work across composed queries—and validate changes against absolute CPU and Metal query times. The ordering/top-k comparison is running; the source-level export audit is recorded below. Automatic CPU/Metal placement remains deferred under #583.

## Native join export audit

At engine `b426cc27f0d87df9dcd469c1843a0ef1e578ad45`, the root native export loops over output fields and reads each typed column separately. Each column reader allocates zero-filled host values and optional compact validity bytes, enqueues copies, and synchronizes before returning. These benchmark outputs have no deferred error planes or offset validity repacking. The final construction of the DataFrame shares the resulting column owners. See the pinned [root export](https://github.com/randyzwitch/dataframe_mojo/blob/b426cc27f0d87df9dcd469c1843a0ef1e578ad45/dataframe/accelerator/gpu/runtime.mojo#L1279) and [typed readback](https://github.com/randyzwitch/dataframe_mojo/blob/b426cc27f0d87df9dcd469c1843a0ef1e578ad45/dataframe/accelerator/gpu/typed_storage.mojo#L947).

The following are medians of three separate profile collections for the 262,144-row base variant, from the [archived raw results](metal-native-join/a52334e7-b426cc27/paired/results.json). They are not the unprofiled warm timing samples used above. Phase medians need not sum to the query median.

| Workload | Output rows | Profile query ms | Region wall ms | Readback ms | Result copy ms | Query download bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| join_hash | 524,288 | 67.103 | 29.588 | 37.499 | 0.002 | 10,551,344 |
| join_fanout | 1,048,576 | 109.092 | 35.176 | 73.488 | 0.003 | 21,102,640 |

The output fields contain 20 value bytes per row plus one packed validity bitmap. The query download counters equal that payload plus 48 control/status bytes. There are three per-column export drains; the reported nine synchronizations and 34 launches count the entire query, not just export. The negligible result-copy interval is consistent with shared column ownership rather than another full values copy.

The effective payload/readback rates are only approximately 281 and 287 MB/s. These include allocation, zero initialization, transfer and waiting; they are **not measured device bandwidth**. Doubling the output roughly doubles the readback phase, but does not identify which component is responsible. Batching the three drains, reducing host initialization, or changing transfer staging are hypotheses requiring separate measurements and complete-query CPU comparisons.

Earlier isolated probes on the same M1 and pinned compiler already narrow the next experiment. For a 4 MiB Float32 payload, the [primitive transfer probe](apple-metal-mojo-m1-20261019/transfer-probe/README.md) measured warm medians of 0.048 ms host allocation, 14.004 ms download enqueue and 0.552 ms subsequent wait. These are separate component medians from a primitive, not an attribution of the join query. The [explicit HostBuffer comparison](apple-metal-host-buffer-m1-20261019/README.md) then measured 14.269 ms raw-List download versus 14.288 ms HostBuffer download plus copying into owned List storage (medians of per-sample totals). The latter's fast enqueue shifted the cost into owned-storage copying and demonstrated no meaningful total win. Neither fewer explicit waits nor switching to HostBuffer alone is supported as the principal fix by those probes. A different host-memory destination requires its own measured experiment, including ownership-copy and allocation costs.

A join followed by a reducing consumer can test whether residency avoids a large intermediate export. That is a distinct composed workload and cannot substitute for closing the full-output CPU gap. Full-output queries must still return their requested values; their current export cost has not been shown to be unavoidable.

## Reproducibility

The per-case data and source-summary hashes are in [metal-cpu-gap.json](metal-cpu-gap.json). The referenced archives retain raw samples, correctness checks, exact build receipts and the original slower/inconclusive observations. These measurements use different qualified candidate stacks; the report does not pretend they are one integrated replacement build.
