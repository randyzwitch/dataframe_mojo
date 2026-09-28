# Typed scalar hash probes

> **Update (#304).** The typed string count probe described below was removed
> with inner-join count fusion. The typed Int64 and string membership probes
> for semi and anti joins remain.


Semi/anti joins now choose Int64 or string probe loops once per job rather
than repeating generic key dispatch for each row. Inner-join count probes
similarly specialize one-string keys. Column references and all-valid flags
are retained outside the row loop. Inline slot lookup still checks exact
packed integer bits or full string bytes after hash equality. No bounds
checks are removed. Compound and other scalar types retain generic equality.

Null probes remain unmatched (and therefore survive anti joins); null build
keys never enter the index. Duplicate build keys do not duplicate membership
rows, and count probes retain their multiplicities and overflow checks.

## Validation

Typed-probe, count, hash and semi/anti modules pass on both hosts (30 tests
each). New full-result tests cover conventional/native view combinations,
sliced offsets, nulls, empty strings, long shared prefixes, embedded NULs,
distinct Unicode byte sequences, signed extremes and compound NaN/signed-zero
fallbacks. Existing tests cover actual slot collisions and repeated groups.

## Count queries

Baseline: #302 (`b29c1b2`). Both hosts use four workers, unpinned CPUs, Mojo
1.2.0.dev2026092105 (e9569894) and matching source/input hashes. The same
duplicate-string query runs at 1M/10M right rows with base/shuffled layouts.
Three alternating process rounds each have one warmup and five checked
samples. Every timed full answer passes (240 results across hosts). Times
are medians of per-round medians; loading is outside timing.

| Host | Right rows | Layout | Generic ms | Typed ms | Speedup |
|---|---:|---|---:|---:|---:|
| Linux | 1,000,000 | base | 88.166 | 77.883 | 1.13x |
| Linux | 1,000,000 | shuffled | 90.407 | 80.270 | 1.13x |
| Linux | 10,000,000 | base | 526.571 | 425.328 | 1.24x |
| Linux | 10,000,000 | shuffled | 555.666 | 459.454 | 1.21x |
| M1 | 1,000,000 | base | 29.207 | 25.374 | 1.15x |
| M1 | 1,000,000 | shuffled | 30.442 | 26.964 | 1.13x |
| M1 | 10,000,000 | base | 265.404 | 226.451 | 1.17x |
| M1 | 10,000,000 | shuffled | 260.555 | 224.691 | 1.16x |

## Membership queries

The 1M-row semi/anti matrix uses five alternating process rounds, one warmup
and one timed query per process. These 120 timing samples check result height;
warmed heights/checksums agree across revisions. The semantic tests above
make complete ordered-result assertions. The table reports sample medians.

| Host | Case | Layout | Generic ms | Typed ms | Speedup |
|---|---|---|---:|---:|---:|
| Linux | semi_unmatched | base | 18.891 | 22.859 | 0.83x |
| Linux | semi_unmatched | shuffled | 29.799 | 28.258 | 1.05x |
| Linux | semi_unmatched | wide | 65.183 | 59.731 | 1.09x |
| Linux | anti_unmatched | base | 24.600 | 26.250 | 0.94x |
| Linux | anti_unmatched | shuffled | 29.428 | 22.569 | 1.30x |
| Linux | anti_unmatched | wide | 66.336 | 64.214 | 1.03x |
| M1 | semi_unmatched | base | 7.429 | 7.463 | 1.00x |
| M1 | semi_unmatched | shuffled | 8.698 | 8.728 | 1.00x |
| M1 | semi_unmatched | wide | 22.470 | 22.174 | 1.01x |
| M1 | anti_unmatched | base | 7.405 | 7.447 | 0.99x |
| M1 | anti_unmatched | shuffled | 8.754 | 8.713 | 1.00x |
| M1 | anti_unmatched | wide | 22.259 | 22.162 | 1.00x |


The initial Linux base semi result was 21% slower. Because both revisions
showed two timing clusters, a focused 15-round repeat was run after all
compilation stopped. It measured 25.457 ms generic versus 19.102 ms typed
for semi and 25.258 versus 24.395 ms for anti—the semi difference reversed
sign. Both revisions ranged roughly 18.7–26.8 ms. Base layouts use the
unchanged bounded-range route, so no consistent base-layout gain or
regression is claimed. The initial samples above are retained, along with
[all repeat samples](benchmarks/typed-membership-linux-base-check.json).
M1 membership timings are effectively unchanged; the repeat does not justify
claiming a general membership speedup.

The combined feature checkout, including #300, also passed all nine affected
modules (56 tests) on both Linux and M1; see the
[integration record](performance-followups-integration.md).

## Profile and limits

A separate Linux 199 Hz user-cycle/DWARF recording of 20 shuffled count
repetitions no longer places generic row equality among entries above 1%.
The specialized count worker accounts for 43.69%, string access 17.13%,
byte hashing 17.19% and column hashing 12.79%. These normalized whole-process
samples include loading and warmup, and are not query-only counters or
speedup estimates. Actual byte comparison and hash lookup remain necessary.
Raw captures remain in `build/profiles`.

[Linux count samples](benchmarks/typed-count-linux.json),
[M1 count samples](benchmarks/typed-count-mac.json),
[Linux membership samples](benchmarks/typed-membership-linux.json),
[M1 membership samples](benchmarks/typed-membership-mac.json), and
[Linux profile summary](benchmarks/typed-probes-profile-linux.json) retain
all evidence. Timings vary across cases; small changes are not robust proof
of improvement. Raw whole-process peak RSS includes loading/runtime/warmup.

Reproduce counts using `scripts/bench_upstream_revision.py --cases duplicate_strings`
and membership using `scripts/bench_join_revision.py --cases semi_unmatched,anti_unmatched`,
with #302 baseline/candidate runners, matching CSV inputs and output JSON paths.
