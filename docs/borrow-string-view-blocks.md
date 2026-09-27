# Borrow immutable byte blocks in string-view reads

Long Arrow Utf8View reads previously copied an Arc reference to the owning
byte block into a local variable. That inserted an atomic retain/release on
each scalar read, including hashing and join equality. Borrow the existing
reference instead. The storage continues to own the immutable block for the
whole access; descriptor offsets, lengths and borrowed-slice semantics do not
change. No unchecked indexing or new unsafe operations are introduced.

This is separate from typed probe dispatch: the only engine change is the
local `var` to `ref` binding. Benchmark provenance additionally records the
string-column and string-view source digests.

## Evidence

The string-view, string-column and join-count modules pass on Linux and M1
(19 tests each). They cover inline/external blocks, sliced views, validity,
ownership, gathers and count/equality semantics. The paired query runner
checks all 240 timed answers, using the same 1M/10M duplicate-string inputs,
four workers, Mojo 1.2.0.dev2026092105 (e9569894) and unpinned CPUs. Each
process warms once; three alternating process rounds each have five samples.
Loading is outside timing. Times are medians of per-round medians.

Baseline: #301 (`f68f73a`).

| Host | Right rows | Layout | Copy ms | Borrow ms | Speedup |
|---|---:|---|---:|---:|---:|
| Linux | 1,000,000 | base | 102.122 | 87.445 | 1.17x |
| Linux | 1,000,000 | shuffled | 124.486 | 90.217 | 1.38x |
| Linux | 10,000,000 | base | 617.787 | 524.775 | 1.18x |
| Linux | 10,000,000 | shuffled | 902.223 | 554.823 | 1.63x |
| M1 | 1,000,000 | base | 35.684 | 28.963 | 1.23x |
| M1 | 1,000,000 | shuffled | 38.868 | 30.524 | 1.27x |
| M1 | 10,000,000 | base | 313.639 | 265.482 | 1.18x |
| M1 | 10,000,000 | shuffled | 330.062 | 260.622 | 1.27x |

Separate Linux perf recordings (199 Hz user cycles, DWARF, 20 repetitions)
show StringColumn._get falling from 47.05% to 12.37% of self samples. Arc
destruction is no longer among entries above 1%. Generic row equality now
accounts for 47.65%, motivating the separate typed-probe change. These are
whole-process, normalized samples including load/warmup, not query-only
counters or direct speedup estimates. Raw captures remain in `build/profiles`.

[Linux timings](benchmarks/borrow-views-linux.json),
[M1 timings](benchmarks/borrow-views-mac.json), and
[Linux profile summary](benchmarks/borrow-views-profile-linux.json) retain
the evidence. Input and candidate-source hashes match across hosts. RSS
is whole-process peak, including loading and warmup. Reproduce with
`scripts/bench_upstream_revision.py`, #301 as the baseline, this candidate,
the PR297 CSV directory and `--cases duplicate_strings`.
