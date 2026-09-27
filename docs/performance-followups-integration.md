# Performance follow-up integration validation

The five opportunities identified after #297 are implemented as separate
reviewable changes. Probe efficiency is split into two PRs because ownership
and typed dispatch address distinct costs.

| Change | Review | Parent |
|---|---|---|
| Prepared streaming joins | [#298](https://github.com/randyzwitch/dataframe_mojo/pull/298) | main |
| Measured build-side selection | [#299](https://github.com/randyzwitch/dataframe_mojo/pull/299) | #298 |
| Safe integer range filters | [#300](https://github.com/randyzwitch/dataframe_mojo/pull/300) | #299 |
| Inner-join count fusion | [#301](https://github.com/randyzwitch/dataframe_mojo/pull/301) | #299 |
| Borrow string-view byte blocks | [#302](https://github.com/randyzwitch/dataframe_mojo/pull/302) | #301 |
| Typed scalar probes | [implementation and evidence](typed-join-probes.md) | #302 |

A local integration checkout merged the typed-probe engine (`1383484`)
and the range-filter engine (`916620c`) cleanly with the ort strategy. Both
Linux and M1 passed nine affected modules, 56 tests each, with identical
engine and lockfile hashes. The [machine-readable record](benchmarks/performance-followups-integration.json)
lists exact parents, digests and modules. This checks the combined semantics;
it is not an additional all-changes performance comparison.

The individual reports retain paired measurements and limitations. In
particular, unselective ranges do not benefit from filtering, membership
timings vary by host/layout, and count fusion deliberately handles only
eligible in-memory inner-join projections. These changes do not establish
performance parity with Polars or DuckDB across all workloads.

To reproduce the combined source, start from the recorded typed-probe parent
and merge the recorded range-filter parent, then run `scripts/run_tests.sh`
with the nine modules listed in the JSON record using the pinned environment.
The test runs used TEST_JOBS=4 on Linux and TEST_JOBS=2 on M1.
