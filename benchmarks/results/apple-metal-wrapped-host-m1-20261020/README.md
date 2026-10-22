# Wrapped caller-owned host-memory readback probe

For a 4 MiB Float32 payload on the Apple M1, wrapped caller-owned storage reduces the median time through an owned result List from **14.2215 ms to 0.9680 ms** (about **14.7×**). This includes allocation, registration, copy submission, transfer wait, deferred unmap and the final owned-storage copy. It is a primitive result, **not a dataframe query speedup**. The small payload shows no improvement over the raw-List path.

All 30 samples validate exact Float32 values: two sizes, three paths and five repetitions. The wrapped path also validates initialized padding sentinels. Path order rotates by repetition. Both exact-length and padded source DeviceBuffers are initialized and synchronized before measurement; their combined requested bytes are reported for every path. The context is initialized once. Rep 0 is first use for that size/path, not a cold-device measurement; warm summaries use the four later repetitions.

| Rows | Path | First total ms | Warm median total ms | Warm range ms |
| ---: | --- | ---: | ---: | ---: |
| 4,097 | Raw List | 0.6020 | 0.3865 | 0.3790–0.3910 |
| 4,097 | HostBuffer then owned List | 10.4630 | 0.2950 | 0.2820–0.3000 |
| 4,097 | Wrapped caller storage then owned List | 0.6790 | 0.4015 | 0.3860–0.4250 |
| 1,048,576 | Raw List | 18.1480 | 14.2215 | 14.1440–14.8110 |
| 1,048,576 | HostBuffer then owned List | 16.4140 | 14.3225 | 14.1780–14.7530 |
| 1,048,576 | Wrapped caller storage then owned List | 1.2760 | 0.9680 | 0.9070–1.1250 |

The large raw-List path spends a median 13.6370 ms in download enqueue. HostBuffer reduces its enqueue to 0.0130 ms but spends 13.6735 ms copying into owned storage, reproducing the earlier unsuccessful control. Wrapped caller storage takes 0.0085 ms for copy enqueue, 0.5935 ms for transfer wait, 0.0945 ms for the owned-List copy and 0.2435 ms for release/unmap wait. These component medians need not sum to the median of per-sample totals. They do not measure device-only bandwidth or identify runtime-internal cache behavior.

The wrapped destination uses 16 KiB-aligned caller allocation, exposed through `wrap_host_memory`, with a buffer-to-buffer copy. The source and destination ranges are padded to a page multiple. For 4,097 rows, 16,388 logical bytes require 32,768 transfer bytes; the large case is already page-exact. The final owned result always contains only logical rows. The wrapper is destroyed and its deferred release synchronized before the allocation owner is freed. The final bulk copy retains ordinary owned List storage.

This path adds temporary host storage: a padded staging allocation alongside the owned result (4 MiB extra at the large size). Reported byte counts are requested payload/storage sizes, not physical residency measurements. Allocation/free policy, thresholds, arbitrary slices, validity/error planes, all numeric dtypes, budget enforcement and full-query performance require engine-level validation before adopting it. In particular, the small result does not justify routing every readback through this mechanism.

## Reproduction and evidence

`identity.json` pins source, runner, binary and build-log SHA-256 values, compiler identity and executable hash, lockfile, SDK/Xcode and machine details. The Mac build used Mojo 1.2.0.dev2026100905 (3b71af75), the same pinned compiler and lockfile as the paired query measurements, with `-O3 -I .`. Both build and run exited 0. The executable hash was checked during review; the binary itself is not archived.

`run_probe.py` contains the exclusive-output, source/compiler/lock and idle-machine guards and reproduces the two-size, five-repetition matrix on the recorded Mac environment. `samples.log` is complete raw stdout; `summary.json` retains every sample and per-phase first/warm values. `validate_results.py` replays the raw output, checks exact matrix/order, byte arithmetic, phase totals and retained build identities. Run `sha256sum -c SHA256SUMS.txt` for archive integrity. No engine change is included.
